"""Load the Nemotron Image files as ComfyUI-managed models.

The patcher owns every weight a decode step reads, so ComfyUI loads, evicts and reloads them like any model's: the
dense weights, the vision rows of `lm_head` and, when they fit on the card beside them, the expert banks (else they
stay in host memory behind `HostExperts`, read by the plain path one expert at a time). The embedding table and the
rest of `lm_head` (`model.head_rows`, only full-vocabulary checks read it) are never on the card (`HostEmbedding`).

For the fast engine, the host memory its kernels read directly (the embedding, and the expert banks kept in RAM for
its expert cache) is copied into this process's own memory at its exact size and registered with
CUDA (`HostMemory`): when the fast path first runs, only while at least PIN_HEADROOM of RAM stays available, and
dropped again (the file's pages put back) when ComfyUI unloads the model, which it does when another model, another
Nemotron file among them, needs the card. (Registering the file's own pages does not work: the mapping is
copy-on-write and the driver pins it writable, copying every page in a fault at a time.)
"""
import logging
import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.ops
import comfy.system_memory
import comfy.utils
from comfy.quant_ops import QuantizedTensor
from . import kernels
from .model import NUM_VISION, VISION_START, NemotronImageModel, parse_config
from .prompt import PromptTokenizer

ENGINES = ("auto", "reference")
# Kept free beside the weights when deciding whether the experts fit on the card: one image's decode state and step
# buffers (0.3 GiB), the pixel decoder for live previews (1.25 GiB, docs/gates/D2.md), the CUDA context and workspace.
RESERVE_BYTES = int(2.5 * 2 ** 30)
# RAM left available (as ComfyUI counts it, within a cgroup's limit) after registering host memory with CUDA: registered
# pages cannot be swapped or dropped
PIN_HEADROOM = 16 * 2 ** 30


class _OpsConfig:
    """`pick_operations` reads `.quant_config` off a model config; quantized files select `mixed_precision_ops`."""

    def __init__(self, quant_config):
        self.quant_config = quant_config


class HostPins:
    """Host tensors registered with CUDA (cudaHostRegister on their own pages, exact size), so kernels read them directly
    and copies from them run at full speed. Unregistered when this object goes; the tensors are held until then, so
    their pages are never freed while registered."""

    def __init__(self):
        self.tensors = []
        weakref.finalize(self, HostPins._release, self.tensors)

    def register(self, tensors):
        """Registers every tensor; False (none registered) when CUDA refuses one."""
        done = []
        for t in tensors:
            if t.nbytes == 0 or t.is_pinned():
                continue
            if torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.nbytes, 3) != 0:  # portable | mapped
                comfy.model_management.discard_cuda_async_error()
                HostPins._release(done)
                return False
            done.append(t)
        self.tensors.extend(done)
        return True

    @staticmethod
    def _release(tensors):
        for t in tensors:
            torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
        tensors.clear()


class HostMemory:
    """The fast path's host memory for one model: the embedding and the quantized banks kept in RAM (`banks`, the
    experts modules), as exact-size copies in this process's own memory registered with CUDA while held."""

    def __init__(self, model, banks):
        self.model, self.banks = model, banks
        self.originals = None
        self.pins = HostPins()

    def bytes(self):
        return self.model.embeddings.weight.nbytes + sum(_codes(bank.weight).nbytes for experts in self.banks
                                                         for bank in (experts.up_proj, experts.down_proj))

    def acquire(self):
        """Copies and registers it unless held; None, or the reason it cannot (RAM budget, CUDA refused)."""
        if self.originals is not None:
            return None
        need = self.bytes()
        if comfy.system_memory.virtual_memory_available() - need < PIN_HEADROOM:
            return (f"its {need / 2 ** 30:.1f} GiB of host memory would leave less than {PIN_HEADROOM / 2 ** 30:.0f} GiB of "
                    f"RAM available")
        embeddings = self.model.embeddings
        self.originals = (embeddings.weight, [(bank, bank.weight) for experts in self.banks
                                               for bank in (experts.up_proj, experts.down_proj)])
        embeddings.__dict__["weight"] = torch.empty_like(embeddings.weight).copy_(embeddings.weight)
        tensors = [embeddings.weight]
        for bank, w in self.originals[1]:
            # the codes copied; wrapping a QuantizedTensor in a Parameter copies the layout fields already
            codes = torch.empty_like(_codes(w)).copy_(_codes(w))
            bank.weight = nn.Parameter(QuantizedTensor(codes, w._layout_cls, w._params) if isinstance(w, QuantizedTensor) else codes,
                                       requires_grad=False)
            tensors += _bank_tensors(bank)
        if not self.pins.register(tensors):
            self.release()
            return "CUDA refused to register its host memory"
        return None

    def release(self):
        """Unregisters and drops the copies, the file's pages back in their place."""
        if self.originals is None:
            return
        HostPins._release(self.pins.tensors)
        embedding, banks = self.originals
        self.model.embeddings.__dict__["weight"] = embedding
        for bank, w in banks:
            bank.weight = w
        self.originals = None


def _codes(w):
    """A bank weight's bulk: a quantized one's codes, else the tensor."""
    return w._qdata if isinstance(w, QuantizedTensor) else w


def _bank_tensors(bank):
    w = bank.weight
    if not isinstance(w, QuantizedTensor):
        return [w]
    return [w._qdata] + [v for v in vars(w._params).values() if torch.is_tensor(v)]


class HostExperts(nn.Module):
    """A layer's routed experts kept in host memory, out of the patcher's reach: the module owns no parameters, so
    ComfyUI loads and accounts only the dense weights. The plain path copies the experts a decode row uses; the fast
    path caches them on the card (`kernels.ExpertCache`) and reads a missed one in place, when they are registered."""

    def __init__(self, experts):
        super().__init__()
        self.__dict__["experts"] = experts

    def forward(self, x, ids, weights):
        return self.experts(x, ids, weights)

    def tensors(self):
        return _bank_tensors(self.experts.up_proj) + _bank_tensors(self.experts.down_proj)


class HostEmbedding(nn.Module):
    """The embedding table in host memory (the file's own pages), out of the patcher's reach. Lookups for ids on the
    card read only the rows they need: in place when the table is registered with CUDA, else through the CPU."""

    def __init__(self, weight):
        super().__init__()
        self.__dict__["weight"] = weight

    def forward(self, ids, out_dtype=None):
        if ids.device.type == "cpu":
            out = F.embedding(ids, self.weight)
        elif self.weight.is_pinned():
            out = torch.empty((*ids.shape, self.weight.shape[1]), dtype=self.weight.dtype, device=ids.device)
            kernels.gather_rows(ids.reshape(-1).long(), self.weight, out.view(-1, self.weight.shape[1]))
        else:
            out = F.embedding(ids.cpu(), self.weight).to(ids.device)
        return out if out_dtype is None else out.to(out_dtype)


class NemotronImage:
    """A loaded model: the module, its patcher, the prompt tokenizer, and the engine generation uses ("auto": the fast
    path when `experts.FastDecoder.supported`, else the plain one; "reference": always the plain one)."""

    def __init__(self, model, patcher, engine, host):
        self.model = model
        self.patcher = patcher
        self.engine = engine
        self.host = host
        self.tokenizer = PromptTokenizer()

    @property
    def config(self):
        return self.model.config

    @property
    def experts_on_card(self):
        return not any(isinstance(layer.mixer.experts, HostExperts) for layer in self.model.layers if layer.kind == "E")


def _tensor_bytes(t):
    if isinstance(t, QuantizedTensor):
        return t._qdata.nbytes + sum(v.nbytes for v in vars(t._params).values() if torch.is_tensor(v))
    return t.nbytes


def module_bytes(module):
    """Bytes of a module's parameters and buffers, quantized ones at their stored size."""
    return sum(_tensor_bytes(t) for t in module.parameters()) + sum(_tensor_bytes(t) for t in module.buffers())


def card_bytes(device):
    """What ComfyUI can give a model on `device`: its free memory and what the models it holds there could hand back
    (less than the card's size when another process holds part of it)."""
    held = sum(m.model.loaded_size() for m in comfy.model_management.current_loaded_models
               if m.device == device and m.model is not None)
    return comfy.model_management.get_free_memory(device) + held


def build_nemotron_image(state, metadata, engine="auto", experts_in_ram=None):
    """A NemotronImage from a converted file's state dict and metadata (the dict is consumed: its tensors are
    assigned, never copied). experts_in_ram: None decides by whether the whole model, embedding aside, fits in what
    ComfyUI can give it on the card (`card_bytes`) with RESERVE_BYTES to spare; True / False forces it."""
    if engine not in ENGINES:
        raise ValueError(f"Nemotron Image: engine must be one of {ENGINES}, got {engine!r}")
    if not metadata or "nemotron_image_config" not in metadata:
        raise ValueError("not a Nemotron Image model file (no nemotron_image_config in its metadata)")
    config = parse_config(metadata["nemotron_image_config"])
    dtype = torch.bfloat16
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    quant_config = comfy.utils.detect_layer_quantization(state, "")
    if quant_config:
        operations = comfy.ops.pick_operations(dtype, dtype, load_device=load_device, model_config=_OpsConfig(quant_config))
    else:
        # the stock bf16 ops have no expert bank; mixed precision ops without quantized layers load it as a plain tensor
        operations = comfy.ops.mixed_precision_ops({}, dtype)
    # The ops allocate no weights before loading (only the embedding table does), and the file's tensors are assigned,
    # so the weights are never held twice. Not the meta device: the ops move loaded weights to their own.
    model = NemotronImageModel(config, device=offload_device, dtype=dtype, ops=operations)
    # lm_head: only the vision rows (what generation reads) under the patcher, the whole table in host memory
    model.lm_head = operations.Linear(config.hidden_size, NUM_VISION, bias=False, device=offload_device, dtype=dtype)
    model.__dict__["head_rows"] = state["lm_head.weight"]
    state["lm_head.weight"] = model.head_rows[VISION_START:VISION_START + NUM_VISION]
    result = model.load_state_dict(state, strict=False, assign=True)
    state.clear()
    if result.unexpected_keys or result.missing_keys:
        raise ValueError(f"Nemotron Image checkpoint keys: unexpected {result.unexpected_keys[:6]}, missing {result.missing_keys[:6]}")
    model.embeddings = HostEmbedding(model.embeddings.weight)
    if experts_in_ram is None:
        need = module_bytes(model) + RESERVE_BYTES
        experts_in_ram = load_device.type != "cuda" or need > card_bytes(load_device)
    if experts_in_ram:
        for layer in model.layers:
            if layer.kind == "E":
                layer.mixer.experts = HostExperts(layer.mixer.experts)
    # what the fast path's kernels read from host memory: the embedding, and the banks kept in RAM (its expert cache's
    # source; bf16 ones are 58.8 GB, which HostMemory takes only while PIN_HEADROOM of RAM stays available)
    banks = [layer.mixer.experts.experts for layer in model.layers if layer.kind == "E"] if experts_in_ram else []
    host = HostMemory(model, banks)
    comfy.model_management.archive_model_dtypes(model)
    # The fast path's kernels read the weights in place, so they must be whole tensors on the card while it runs: its
    # engine takes the plain ModelPatcher, which ComfyUI loads whole by estimate even with dynamic VRAM on (main.py
    # makes CoreModelPatcher the dynamic one then, which stages weights instead). The reference engine takes the core
    # patcher, whatever it is.
    patcher_class = comfy.model_patcher.ModelPatcher if engine == "auto" else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=load_device, offload_device=offload_device)
    # The host copies go when ComfyUI unloads the model (another model needs the card), not when load_models_gpu only
    # re-registers it (detach without unpatching, before every generation)
    patcher.add_callback(comfy.patcher_extension.CallbacksMP.ON_DETACH,
                         lambda patcher, unpatch_all: host.release() if unpatch_all else None)
    logging.info("Nemotron Image: engine %s, experts in %s (%.1f GiB under the patcher)", engine,
                 "RAM" if experts_in_ram else "the model", module_bytes(model) / 2 ** 30)
    return NemotronImage(model, patcher, engine, host)


def load_nemotron_image(path, engine="auto", experts_in_ram=None):
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    try:
        return build_nemotron_image(state, metadata, engine=engine, experts_in_ram=experts_in_ram)
    except ValueError as e:
        raise ValueError(f"{os.path.basename(path)}: {e}") from None

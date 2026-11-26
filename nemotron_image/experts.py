"""The fast decode path: the whole Nemotron Image decode step on the pack's kernels (`kernels.py`), one CUDA graph per
step, no host sync while an image row is written.

`FastDecoder.generate_codes` takes the arguments of `generate.generate_codes` and returns the same codes. It reads the
weights in place where ComfyUI loaded them: the loader puts every weight a step reads under the model's patcher
(expert banks in their stored layout, dense bf16 weights, the vision rows of `lm_head`) and keeps the
embedding in pinned host memory, one row read per token. Only float32 copies of the norms, conv, A, D and dt_bias
(a few KB a layer) and the decode state are made per call. Up to 8 images run as rows of one step, more in groups
of 8. Prefill runs the same step with the prompt as forced tokens.

Sampling matches `generate.sample` in distribution but uses its own noise, so a seed gives another image than on the
plain path. An image depends only on (seed, its index in the batch), whatever the batch size:
- top-k off and top-p 1: Gumbel-max inside the head kernel, keyed by (seed, index in the batch, position);
- otherwise: the head writes the logits and `generate.sample` draws each image's token from its own generator
  (`generate.image_generator`, the plain path's too), then `accept` takes it.
"""
import logging
import sys
import time

import torch

import comfy.model_management
import comfy.system_memory
from comfy.quant_ops import QuantizedTensor
from . import kernels as K
from .generate import PATCH, forced_prefix, image_generator, plain_codes, sample
from .loader import PIN_HEADROOM, HostExperts, card_bytes, module_bytes
from .model import NUM_VISION, VIS_ROW, VISION_START

MAX_ROWS = 8
# kept free on the card beside the expert cache (allocator slack, cuBLAS / NVRTC workspaces)
CACHE_MARGIN = 512 << 20


def _bank(weight):
    """(format, ConvRot group, kernels.Bank) of an expert bank or shared-expert weight the kernels read, or a reason
    they cannot: int8 ConvRot (a scale per row), W4A8 (group 16, e4m3 scales, a codebook), W6A8, or plain bf16."""
    if not isinstance(weight, QuantizedTensor):
        if weight.dtype != torch.bfloat16:
            return f"{weight.dtype} weights"
        return "bf16", 1, K.Bank("bf16", weight.reshape(-1, weight.shape[-1]))
    p, q = weight._params, weight._qdata
    if weight._layout_cls == "TensorWiseINT8Layout":
        if not p.convrot or p.scale.numel() != q.shape[0]:
            return "int8 without ConvRot or without a scale per row"
        return "int8", p.convrot_groupsize, K.Bank("int8", q, scale=p.scale.reshape(-1))
    if weight._layout_cls == "AsymW4A8Int8Layout":
        bits = q.shape[-1] * 8 // p.orig_shape[-1]
        if (p.group_size != 16 or p.correction is not None or p.mma_packed or p.transposed
                or p.scale.dtype != torch.float8_e4m3fn or (bits, p.codebook is not None) not in ((4, True), (6, False))):
            return "a W4A8/W6A8 variant the kernels do not decode"
        return f"w{bits}", p.convrot_groupsize, K.Bank(f"w{bits}", q.view(torch.uint8), rel=p.scale.view(torch.uint8),
                                                       scale=p.s_channel, codebook=p.codebook)
    return f"the {weight._layout_cls} layout"


def _routed(layer):
    experts = layer.mixer.experts
    return experts.experts if isinstance(experts, HostExperts) else experts


def _formats(model):
    """(routed format, shared-expert format, ConvRot group) of the model's expert banks, or a reason the kernels cannot
    run them. Every MoE layer's routed banks share one format and every shared expert one format (that one may
    differ: a quantized file may keep its shared experts in int8)."""
    routed, shared = set(), set()
    for layer in model.layers:
        if layer.kind != "E":
            continue
        for seen, weights in ((routed, (_routed(layer).up_proj.weight, _routed(layer).down_proj.weight)),
                              (shared, (layer.mixer.shared_experts.up_proj.weight, layer.mixer.shared_experts.down_proj.weight))):
            for w in weights:
                found = _bank(w)
                if isinstance(found, str):
                    return found
                seen.add(found[:2])
    if len(routed) != 1 or len(shared) != 1:
        return f"mixed expert formats {sorted(routed | shared)}"
    (fmt, group), (shared_fmt, shared_group) = routed.pop(), shared.pop()
    if (fmt == "bf16") != (shared_fmt == "bf16"):
        return "bf16 experts beside a quantized shared expert"  # the router rotates the experts' input for both or neither
    groups = {g for f, g in ((fmt, group), (shared_fmt, shared_group)) if f != "bf16"}
    if len(groups) > 1 or not groups <= {16, 64}:
        return f"ConvRot groups {sorted(groups)}"
    return fmt, shared_fmt, groups.pop() if groups else 1


def state_bytes(model, rows, capacity):
    """Bytes of R rows' decode state and step buffers."""
    kinds = [layer.kind for layer in model.layers]
    kv = kinds.count("*") * rows * 2 * 2 * capacity * K.HEAD_DIM * 2
    mamba = kinds.count("M") * rows * (4 * K.CONV_DIM + K.MAMBA_HEADS * K.MAMBA_HEAD_DIM * K.MAMBA_STATE) * 4
    splits = -(-capacity // K.ATTEND_CHUNK)
    attention = rows * K.Q_HEADS * splits * (K.HEAD_DIM + 2) * 4
    moe = K.max_units(rows) * rows * (K.EXPERT * 6 + K.HIDDEN * 4 + 8)
    head = rows * (NUM_VISION + 2 * K.head_blocks(NUM_VISION)) * 4
    return kv + mamba + attention + moe + head + rows * (K.PROJ + 4 * 4096) * 4


class _Weights:
    """What a step reads: the loaded model's tensors in place on `device` (a reason to fail if one is elsewhere), float32
    copies of the small ones, the embedding table in pinned host memory."""

    def __init__(self, model, device, group):
        def f32(t):
            return t.to(device=device, dtype=torch.float32).contiguous()

        def here(t):
            if t.device != device or not t.is_contiguous():
                raise RuntimeError(f"Nemotron Image: a weight is on {t.device}, not loaded whole on {device}")
            return t

        def proj(w):
            """A dense projection: its bf16 weight, or (int8 rows, row scales, ConvRot group) for int8 ConvRot."""
            if not isinstance(w, QuantizedTensor):
                return here(w)
            p = w._params
            if w._layout_cls != "TensorWiseINT8Layout" or not p.convrot or p.scale.numel() != w._qdata.shape[0]:
                raise RuntimeError(f"Nemotron Image: a dense {w._layout_cls} weight the fast path does not read")
            return here(w._qdata), f32(p.scale.reshape(-1)), p.convrot_groupsize

        def bank(w):
            b = _bank(w)[2]
            return K.Bank(b.fmt, here(b.weight), *(None if v is None else here(v) if v.dtype != torch.float32 else f32(v)
                                                 for v in (b.rel, b.scale, b.codebook)))

        self.group, self.layers = group, []
        self.fmt = _formats(model)[0]
        self.cached = any(isinstance(layer.mixer.experts, HostExperts) for layer in model.layers if layer.kind == "E")
        self.moe_layers = sum(layer.kind == "E" for layer in model.layers)
        for layer in model.layers:
            m, entry = layer.mixer, {"kind": layer.kind, "norm": f32(layer.norm.weight)}
            if layer.kind == "M":
                entry.update(in_proj=proj(m.in_proj.weight), out_proj=proj(m.out_proj.weight), gn=f32(m.norm.weight),
                             conv_w=f32(m.conv1d.weight.reshape(K.CONV_DIM, -1)), conv_b=f32(m.conv1d.bias),
                             A=-torch.exp(f32(m.A_log)), dt_bias=f32(m.dt_bias), D=f32(m.D))
            elif layer.kind == "*":
                entry.update(q=proj(m.q_proj.weight), k=proj(m.k_proj.weight), v=proj(m.v_proj.weight),
                             o=proj(m.o_proj.weight))
            else:
                entry.update(router=f32(m.gate.weight), bias=f32(m.gate.e_score_correction_bias),
                             shared_up=bank(m.shared_experts.up_proj.weight), shared_down=bank(m.shared_experts.down_proj.weight))
                up, down = _routed(layer).up_proj.weight, _routed(layer).down_proj.weight
                if self.cached:
                    # banks in pinned host memory: the expert cache's slots hold what the step reads
                    entry.update(host_up=_bank(up)[2], host_down=_bank(down)[2])
                    self.codebook = None if entry["host_up"].codebook is None else f32(entry["host_up"].codebook)
                else:
                    entry.update(up=bank(up), down=bank(down))
            self.layers.append(entry)
        self.final_norm = f32(model.norm_f.weight)
        self.head = here(model.lm_head.weight)
        self.embedding = model.embeddings.weight
        self.eps = model.config.layer_norm_epsilon


class _Rows:
    """R rows' decode state, step buffers and the step itself, for a schedule of `length` tokens."""

    def __init__(self, w, rows, schedule, device, cache=None):
        self.w, self.rows, self.device, self.cache = w, rows, device, cache
        length = schedule.shape[1]
        self.cap = length
        f32 = dict(dtype=torch.float32, device=device)
        kinds = [layer["kind"] for layer in w.layers]
        splits = -(-self.cap // K.ATTEND_CHUNK)
        with torch.inference_mode(False):
            self.x = torch.zeros((rows, K.HIDDEN), dtype=torch.bfloat16, device=device)
            self.kc = [torch.zeros((rows, 2, self.cap, K.HEAD_DIM), dtype=torch.bfloat16, device=device) for k in kinds if k == "*"]
            self.vc = [torch.zeros_like(c) for c in self.kc]
            self.ring = [torch.zeros((rows, 4, K.CONV_DIM), **f32) for k in kinds if k == "M"]
            self.ssm = [torch.zeros((rows, K.MAMBA_HEADS, K.MAMBA_HEAD_DIM, K.MAMBA_STATE), **f32) for k in kinds if k == "M"]
            self.pos = torch.zeros(rows, dtype=torch.int32, device=device)
            self.proj = torch.zeros((rows, K.PROJ), **f32)
            self.v = torch.zeros((rows, K.MAMBA_INNER), **f32)
            self.part = torch.zeros((rows, K.MAMBA_HEADS), **f32)
            self.q = torch.zeros((rows, K.Q_HEADS * K.HEAD_DIM), **f32)
            self.attn = torch.zeros_like(self.q)
            self.pm = torch.zeros((rows, K.Q_HEADS, splits), **f32)
            self.pl = torch.zeros_like(self.pm)
            self.pa = torch.zeros((rows, K.Q_HEADS, splits, K.HEAD_DIM), **f32)
            self.moe = K.MoEBuffers(rows, device)
            self.scratch = (torch.zeros((rows, K.MAMBA_INNER), dtype=torch.int16, device=device),
                            torch.zeros((rows, K.MAMBA_INNER // 16), **f32))
            blocks = K.head_blocks(NUM_VISION)
            self.best_v = torch.zeros((rows, blocks), **f32)
            self.best_i = torch.zeros((rows, blocks), dtype=torch.int32, device=device)
            self.counter = torch.zeros(1, dtype=torch.int32, device=device)
            self.logits = torch.zeros((rows, NUM_VISION), **f32)
            self.force = schedule.to(device=device, dtype=torch.int32).contiguous()
            self.tokens = torch.zeros_like(self.force)
            self.sampled = torch.zeros(rows, dtype=torch.long, device=device)
        self.x.copy_(w.embedding[self.force[:, 0].cpu().long()].to(device))

    def layers(self):
        w, im, ia, ie = self.w, 0, 0, 0
        for layer in w.layers:
            if layer["kind"] == "M":
                self.dense(self.x, [layer["in_proj"]], out=self.proj, norm=layer["norm"], eps=w.eps)
                K.mamba_step(self.proj, self.ring[im], layer["conv_w"], layer["conv_b"], self.ssm[im], layer["A"],
                             layer["dt_bias"], layer["D"], self.pos, self.v, self.part)
                self.dense(self.v, [layer["out_proj"]], resid=self.x, norm=layer["gn"], eps=w.eps, gated_part=self.part)
                im += 1
            elif layer["kind"] == "*":
                self.dense(self.x, [layer["q"], layer["k"], layer["v"]], out=self.q, norm=layer["norm"], eps=w.eps,
                           kv=(self.kc[ia], self.vc[ia], self.cap), pos=self.pos)
                K.attend(self.q, self.kc[ia], self.vc[ia], self.pos, self.pm, self.pl, self.pa, self.attn)
                self.dense(self.attn, [layer["o"]], resid=self.x)
                ia += 1
            else:
                K.route(self.x, layer["norm"], w.eps, layer["router"], layer["bias"], self.moe, w.group, layer=ie, cache=self.cache)
                if self.cache is None:
                    K.experts(self.x, self.moe, layer["up"], layer["down"], layer["shared_up"], layer["shared_down"], w.group)
                else:
                    K.experts(self.x, self.moe, self.cache.up, self.cache.down, layer["shared_up"], layer["shared_down"],
                              w.group, host_up=layer["host_up"], host_down=layer["host_down"])
                ie += 1

    def dense(self, x, weights, **kwargs):
        """A projection: bf16 weights through `dense` (the input rounded to bf16 where no norm does it), int8 ConvRot
        ones through prep_rows + dense_q8."""
        if isinstance(weights[0], tuple):
            K.dense_q8(x, [w[:2] for w in weights], weights[0][2], self.scratch, **kwargs)
        else:
            K.dense(x, None if len(weights) > 1 else weights[0], segments=weights if len(weights) > 1 else None,
                    round_input=kwargs.get("norm") is None, **kwargs)

    def step(self, temperature, seed, row0, keep=False, external=False):
        self.layers()
        K.head(self.x, self.w.final_norm, self.w.head, self.w.eps, temperature, seed, self.force, self.best_v, self.best_i,
               self.counter, self.tokens, self.w.embedding, self.pos, logits=self.logits, row0=row0, keep=keep,
               external=external)

    def capture(self, *args, **kwargs):
        """One step as a CUDA graph (state advances on the card, so the same graph replays for every position).
        Windows captures in relaxed mode: another thread's memory-pool query (cuMemPoolGetAttribute) is refused
        under global capture and invalidates the graph. Kernels are pre-compiled by the warm-up step in generate_codes."""
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        graph = torch.cuda.CUDAGraph()
        mode = "relaxed" if sys.platform == "win32" else "global"
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream, capture_error_mode=mode):
            self.step(*args, **kwargs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        return graph


def schedule_for(tokenizer, prompt_ids, width, height):
    """(forced prefix, schedule) of one image: `generate.forced_prefix` forced, then per row <|vis_row|> forced and
    W/16 codes sampled (-1)."""
    forced = forced_prefix(tokenizer, prompt_ids, width, height)
    rows, cols = height // PATCH, width // PATCH
    return forced, torch.tensor(forced + ([VIS_ROW] + [-1] * cols) * rows, dtype=torch.int32)


def _load(nemotron, batch, capacity, reserve_bytes, extra_patchers):
    """Has ComfyUI load the whole model on its device, with the patchers loaded beside it (the preview decoder) and
    room for the decode state and `reserve_bytes`, and, with the experts in host memory, as much room for the expert
    cache as the card has; the reason the weights are not all there, or None."""
    model, patcher = nemotron.model, nemotron.patcher
    device = patcher.load_device
    need = state_bytes(model, min(batch, MAX_ROWS), capacity) + reserve_bytes
    if not nemotron.experts_on_card:
        need = max(need, comfy.model_management.get_total_memory(device) - module_bytes(model) - CACHE_MARGIN)
    comfy.model_management.load_models_gpu([patcher, *extra_patchers], force_full_load=True, memory_required=need)
    away = [name for name, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device != device]
    return f"{len(away)} weights not on {device} ({away[0]}, ...)" if away else None


def _step_set(nemotron, batch):
    """The most distinct experts one step can read: 6 per MoE layer for each image in lockstep. An LRU with fewer slots
    than that evicts what the same step needs again and hits nothing, so the cache needs at least this many."""
    return K.TOP_K * sum(layer.kind == "E" for layer in nemotron.model.layers) * min(batch, MAX_ROWS)


def _cache_slots(nemotron, batch, capacity, reserve_bytes, free):
    """How many expert slots fit in `free` bytes beside one call's state and the reserve."""
    spare = free - state_bytes(nemotron.model, min(batch, MAX_ROWS), capacity) - reserve_bytes - CACHE_MARGIN
    layers = sum(layer.kind == "E" for layer in nemotron.model.layers)
    return int(min(layers * K.EXPERTS, max(spare, 0) // K.ExpertCache.slot_bytes(_formats(nemotron.model)[0])))


class FastDecoder:
    @staticmethod
    def supported(nemotron, batch, capacity, reserve_bytes):
        """None when the fast path can run `batch` images of `capacity` tokens on the model's load device with
        `reserve_bytes` kept free, else the reason it cannot. Experts in host memory run through an expert cache on the
        card, sized by the memory the rest leaves."""
        device = nemotron.patcher.load_device
        if device.type != "cuda" or not K.available(device):
            return "no CUDA device with NVRTC"
        found = _formats(nemotron.model)
        if isinstance(found, str):
            return found
        if nemotron.host.originals is None and comfy.system_memory.virtual_memory_available() - nemotron.host.bytes() < PIN_HEADROOM:
            return f"its {nemotron.host.bytes() / 2 ** 30:.1f} GiB of host memory would leave too little RAM available"
        total = card_bytes(device)
        if not nemotron.experts_on_card:
            slots = _cache_slots(nemotron, batch, capacity, reserve_bytes, total - module_bytes(nemotron.model))
            if slots < _step_set(nemotron, batch):
                return (f"room for {slots} expert slots on the card, fewer than one step reads ({_step_set(nemotron, batch)} "
                        f"for {min(batch, MAX_ROWS)} image(s)): a cache that small would miss on every expert")
            return None
        need = module_bytes(nemotron.model) + state_bytes(nemotron.model, min(batch, MAX_ROWS), capacity) + reserve_bytes
        if need > total:
            return f"the weights and state need {need / 2 ** 30:.1f} GiB, ComfyUI can give {total / 2 ** 30:.1f} GiB on the card"
        return None

    @staticmethod
    def generate_codes(nemotron, prompt_ids, width, height, batch, seed, temperature, top_k, top_p, on_row=None, check=None,
                       reserve_bytes=0, extra_patchers=()):
        """Codes [batch, H/16, W/16] (CPU int64), as `generate.generate_codes`. If ComfyUI cannot load every weight on
        the card (another model holds it), the plain path runs instead."""
        model, device = nemotron.model, nemotron.patcher.load_device
        forced, schedule = schedule_for(nemotron.tokenizer, prompt_ids, width, height)
        rows, cols = height // PATCH, width // PATCH
        clock = [time.perf_counter()]
        away = _load(nemotron, batch, schedule.numel(), reserve_bytes, extra_patchers)
        clock.append(time.perf_counter())
        # after ComfyUI has made room on the card (unloading other models drops their host copies too)
        away = away or nemotron.host.acquire()
        clock.append(time.perf_counter())
        if away is not None:
            logging.warning("Nemotron Image: plain decode: %s", away)
            return plain_codes(nemotron, prompt_ids, width, height, batch, seed, temperature, top_k, top_p, on_row, check,
                               extra_patchers)
        cache = None
        if not nemotron.experts_on_card:
            slots = _cache_slots(nemotron, batch, schedule.numel(), reserve_bytes, torch.cuda.mem_get_info(device)[0])
            if slots < _step_set(nemotron, batch):
                logging.warning("Nemotron Image: plain decode, room for %d expert slots on the card, fewer than one step "
                                "reads (%d)", slots, _step_set(nemotron, batch))
                return plain_codes(nemotron, prompt_ids, width, height, batch, seed, temperature, top_k, top_p, on_row, check,
                                   extra_patchers)
        codes = torch.empty((batch, rows, cols), dtype=torch.long)
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(3)]  # first replay, prefill done, last row done
        callbacks = 0.0
        external = top_k > 0 or top_p < 1.0
        sampled_at = (schedule < 0).tolist()
        with comfy.model_management.cuda_device_context(device):
            w = _Weights(model, device, _formats(model)[2])
            if w.cached:
                cache = K.ExpertCache(w.fmt, slots, w.moe_layers, w.codebook, device)
                logging.info("Nemotron Image: expert cache of %d slots (%.1f GiB)", slots,
                             slots * K.ExpertCache.slot_bytes(w.fmt) / 2 ** 30)
            for row0 in range(0, batch, MAX_ROWS):
                count = min(MAX_ROWS, batch - row0)
                if row0 == 0:
                    warm = _Rows(w, count, schedule[None].expand(count, -1), device, cache)
                    warm.step(temperature, seed, row0, external=external)
                    torch.cuda.synchronize(device)
                    del warm
                run = _Rows(w, count, schedule[None].expand(count, -1), device, cache)
                graph = run.capture(temperature, seed, row0, external=external)
                generators = [image_generator(seed, row0 + r, device) for r in range(count)] if external else None
                host = torch.empty((count, schedule.numel()), dtype=torch.int32, pin_memory=True)
                pending = None

                def finish(r):
                    nonlocal callbacks
                    image_row, event = r
                    event.synchronize()
                    first = len(forced) + image_row * (cols + 1) + 1
                    codes[row0:row0 + count, image_row] = host[:, first:first + cols].long() - VISION_START
                    if on_row is not None and row0 == 0:
                        started = time.perf_counter()
                        on_row(image_row, codes[:count, :image_row + 1])
                        callbacks += time.perf_counter() - started

                if row0 == 0:
                    clock.append(time.perf_counter())
                    marks[0].record()
                for t in range(schedule.numel() - 1):
                    graph.replay()
                    if row0 == 0 and t + 1 == len(forced):
                        marks[1].record()
                    if external:
                        if sampled_at[t + 1]:
                            for r, generator in enumerate(generators):
                                run.sampled[r:r + 1].copy_(sample(run.logits[r:r + 1], temperature, top_k, top_p, generator))
                        K.accept(run.sampled, run.force, run.tokens, w.embedding, run.x, run.pos)
                    image_row, column = divmod(t + 1 - len(forced), cols + 1)
                    if t + 1 >= len(forced) and column == cols:
                        host.copy_(run.tokens, non_blocking=True)
                        event = torch.cuda.Event()
                        event.record()
                        if pending is not None:
                            finish(pending)
                        pending = (image_row, event)
                        if check is not None:
                            check()
                finish(pending)
                del run, graph
            marks[2].record()
            marks[2].synchronize()
            clock.append(time.perf_counter())
            seconds = clock[4] - clock[3]
            # load: ComfyUI loading the weights; host: the host copies; setup: weights, expert cache, state, graph
            # capture; prefill and rows: the GPU's time for the forced prefix and for the image rows (first group of
            # images); callbacks: the host's time in on_row (previews), overlapped with the GPU's work
            logging.info("Nemotron Image: fast decode, %d image(s) of %d steps in %.1f s (%.2f ms per step); load %.1f s, "
                         "host copies %.1f s, setup %.1f s, prefill %.1f s, rows %.1f s, callbacks %.1f s", batch,
                         schedule.numel() - 1, seconds, seconds * 1000 / ((schedule.numel() - 1) * -(-batch // MAX_ROWS)),
                         clock[1] - clock[0], clock[2] - clock[1], clock[3] - clock[2],
                         marks[0].elapsed_time(marks[1]) / 1000, marks[1].elapsed_time(marks[2]) / 1000, callbacks)
            if cache is not None:
                hits, misses = cache.stats.tolist()
                logging.info("Nemotron Image: expert cache hits %.2f%% (%d misses)", 100 * hits / max(hits + misses, 1), misses)
        return codes

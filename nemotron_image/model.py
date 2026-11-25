"""Nemotron-Labs-Image-30B-A3B: the NemotronH hybrid (Mamba2, MoE, attention) that writes image codes.

Every block is x + mixer(rmsnorm(x)). Two paths share the weights:
- `step`: one token per row against per-call decode state (Mamba conv window and SSM state, key/value buffers);
  generation prefills and decodes with it.
- `forward_sequence`: a whole sequence at once (causal conv, chunked SSD scan, causal attention), for teacher-forced
  scoring; tests check it against `step`.
Reference: docs/upstream/modeling_nemotron_h.py (cited as :line).
"""
import json
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen
import comfy.model_management
from comfy.quant_ops import QuantizedTensor

IMG_START, IMG_END, VIS_ROW = 131072, 131073, 131074
VISION_START, NUM_VISION = 131075, 196560


def parse_config(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    c = SimpleNamespace(**raw)
    if c.n_group != 1 or c.topk_group != 1:
        raise ValueError(f"Nemotron Image: grouped routing (n_group {c.n_group}) is not supported")
    if c.mlp_hidden_act != "relu2" or c.mamba_hidden_act != "silu" or not c.norm_topk_prob:
        raise ValueError("Nemotron Image: unexpected activations or router normalisation in the config")
    c.block_types = list(c.hybrid_override_pattern)
    c.mamba_inner = c.mamba_num_heads * c.mamba_head_dim
    c.conv_dim = c.mamba_inner + 2 * c.n_groups * c.ssm_state_size
    return c


def cast(param, x, dtype=None):
    return comfy.model_management.cast_to(param, dtype=dtype or x.dtype, device=x.device)


def dequantized(w, x):
    """A weight as a plain tensor in x's dtype on x's device. A quantized weight moves to x's device as it is (its
    bytes, not the decoded matrix) and decodes there, with comfy_kitchen's eager backend: 0.2.41's CUDA kernels refuse
    W6A8 and W4A8 at ConvRot group 64, the group these files use."""
    if isinstance(w, QuantizedTensor):
        with comfy_kitchen.use_backend("eager"):
            w = w.to(x.device).dequantize()
    return cast(w, x)


class GatedGroupNorm(nn.Module):
    """rmsnorm_fn(y, z, group_size, norm_before_gate=False) (:263-279): rms-normalise y * silu(z) per group of
    channels, times the weight, in fp32 (mamba_ssm's reference math); the result in the compute dtype."""

    def __init__(self, channels, groups, eps, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, device=device, dtype=dtype))
        self.groups = groups
        self.eps = eps

    def forward(self, y, z, dtype):
        h = y.float() * F.silu(z.float())
        h = F.rms_norm(h.unflatten(-1, (self.groups, -1)), (h.shape[-1] // self.groups,), eps=self.eps).flatten(-2)
        return (h * cast(self.weight, h, torch.float32)).to(dtype)


class Mamba2(nn.Module):
    """Mamba2 mixer (:281). in_proj gives the gate z, the conv input xBC (x, B, C) and dt; d_mlp is 0 for this
    model. Conv (kernel 4, depthwise) + SiLU, SSM with A = -exp(A_log), dt = softplus(dt + dt_bias), 8 groups of B
    and C shared by 8 heads each, D skip, then the gated group norm and out_proj. The SSM runs in fp32."""

    def __init__(self, c, device=None, dtype=None, ops=None):
        super().__init__()
        self.heads, self.head_dim, self.state_size = c.mamba_num_heads, c.mamba_head_dim, c.ssm_state_size
        self.groups, self.inner, self.conv_dim, self.chunk = c.n_groups, c.mamba_inner, c.conv_dim, c.chunk_size
        self.in_proj = ops.Linear(c.hidden_size, self.inner + self.conv_dim + self.heads, bias=False, device=device, dtype=dtype)
        self.conv1d = ops.Conv1d(self.conv_dim, self.conv_dim, c.conv_kernel, groups=self.conv_dim, bias=True, device=device, dtype=dtype)
        # kept in fp32 like the checkpoint and vLLM's MambaMixer2 (A in fp32)
        self.dt_bias = nn.Parameter(torch.empty(self.heads, device=device, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.heads, device=device, dtype=torch.float32))
        self.D = nn.Parameter(torch.empty(self.heads, device=device, dtype=torch.float32))
        self.norm = GatedGroupNorm(self.inner, self.groups, c.layer_norm_epsilon, device=device, dtype=dtype)
        self.out_proj = ops.Linear(self.inner, c.hidden_size, bias=False, device=device, dtype=dtype)

    def new_state(self, batch, device, dtype):
        return {"conv": torch.zeros((batch, self.conv_dim, self.conv1d.kernel_size[0] - 1), device=device, dtype=dtype),
                "ssm": torch.zeros((batch, self.heads, self.head_dim, self.state_size), device=device, dtype=torch.float32)}

    def _params(self, x):
        A = -torch.exp(cast(self.A_log, x, torch.float32))
        return A, cast(self.dt_bias, x, torch.float32), cast(self.D, x, torch.float32)

    def step(self, x, state):
        """x [B, hidden] -> [B, hidden]; updates state in place (:529-618, the cached single-token branch)."""
        z, xbc, dt = self.in_proj(x).split((self.inner, self.conv_dim, self.heads), dim=-1)
        window = torch.cat((state["conv"], xbc.unsqueeze(-1)), dim=-1)
        state["conv"].copy_(window[..., 1:])
        weight = cast(self.conv1d.weight, x, torch.float32)
        xbc = F.silu(torch.einsum("bck,ck->bc", window.float(), weight[:, 0]) + cast(self.conv1d.bias, x, torch.float32))
        xs, B, C = xbc.split((self.inner, self.groups * self.state_size, self.groups * self.state_size), dim=-1)
        A, dt_bias, D = self._params(x)
        dt = F.softplus(dt.float() + dt_bias)                                # [B, heads]
        xs = xs.view(-1, self.heads, self.head_dim)
        per = self.heads // self.groups
        B = B.view(-1, self.groups, 1, self.state_size).expand(-1, -1, per, -1).reshape(-1, self.heads, self.state_size)
        C = C.view(-1, self.groups, 1, self.state_size).expand(-1, -1, per, -1).reshape(-1, self.heads, self.state_size)
        ssm = state["ssm"]
        ssm.mul_(torch.exp(dt * A)[..., None, None]).add_((dt[..., None] * xs).unsqueeze(-1) * B.unsqueeze(2))
        y = torch.matmul(ssm, C.unsqueeze(-1)).squeeze(-1) + D[:, None] * xs
        return self.out_proj(self.norm(y.reshape(-1, self.inner), z, x.dtype))

    def forward_sequence(self, x):
        """x [B, T, hidden] from an empty state (:516-704, the uncached branch, as a chunked SSD scan)."""
        batch, T, _ = x.shape
        z, xbc, dt = self.in_proj(x).split((self.inner, self.conv_dim, self.heads), dim=-1)
        k = self.conv1d.kernel_size[0]
        weight, bias = cast(self.conv1d.weight, x, torch.float32), cast(self.conv1d.bias, x, torch.float32)
        xbc = F.silu(F.conv1d(F.pad(xbc.float().transpose(1, 2), (k - 1, 0)), weight, bias, groups=self.conv_dim)).transpose(1, 2)
        xs, B, C = xbc.split((self.inner, self.groups * self.state_size, self.groups * self.state_size), dim=-1)
        A, dt_bias, D = self._params(x)
        dt = F.softplus(dt.float() + dt_bias)                                # [B, T, heads]
        xs = xs.reshape(batch, T, self.heads, self.head_dim)
        per = self.heads // self.groups
        B = B.reshape(batch, T, self.groups, self.state_size).repeat_interleave(per, dim=2)
        C = C.reshape(batch, T, self.groups, self.state_size).repeat_interleave(per, dim=2)
        y = ssd_scan(xs * dt[..., None], A * dt, B, C, self.chunk) + D[:, None] * xs
        return self.out_proj(self.norm(y.reshape(batch, T, self.inner), z, x.dtype))


def segment_sum(a):
    """[..., L] -> [..., L, L]: out[i, j] = sum(a[j+1 .. i]) for j <= i, -inf above the diagonal."""
    L = a.shape[-1]
    cs = torch.cumsum(a, dim=-1)
    out = cs[..., :, None] - cs[..., None, :]
    return out.masked_fill(~torch.ones(L, L, dtype=torch.bool, device=a.device).tril(), -torch.inf)


def ssd_scan(x, a, B, C, chunk):
    """Chunked state-space scan (Mamba2 SSD), fp32. x [b, T, h, p] (already times dt), a [b, T, h] (A * dt),
    B, C [b, T, h, n]. Returns y [b, T, h, p] = sum_{s<=t} C_t . (prod_{s<r<=t} exp(a_r)) B_s x_s."""
    b, T, h, p = x.shape
    pad = -T % chunk
    if pad:
        x, a, B, C = (F.pad(t, (0, 0) * (t.ndim - 2) + (0, pad)) for t in (x, a, B, C))
    n_chunks = (T + pad) // chunk
    x, B, C = (t.reshape(b, n_chunks, chunk, h, -1) for t in (x, B, C))
    a = a.reshape(b, n_chunks, chunk, h).permute(0, 3, 1, 2)                 # [b, h, c, l]
    a_cum = torch.cumsum(a, dim=-1)
    decay = torch.exp(segment_sum(a))                                        # [b, h, c, l, s]
    scores = torch.einsum("bclhn,bcshn->bhcls", C, B) * decay
    y = torch.einsum("bhcls,bcshp->bclhp", scores, x)
    to_end = torch.exp(a_cum[..., -1:] - a_cum)                              # [b, h, c, l]
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, to_end, x)
    states = torch.cat((torch.zeros_like(states[:, :1]), states), dim=1)
    chunk_decay = torch.exp(segment_sum(F.pad(a_cum[..., -1], (1, 0))))      # [b, h, c+1, c+1]
    states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)[:, :-1]
    y = y + torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(a_cum))
    return y.reshape(b, n_chunks * chunk, h, p)[:, :T]


class Attention(nn.Module):
    """Causal GQA attention (:933), 32 query heads, 2 key/value heads of 128, no bias, no positional encoding."""

    def __init__(self, c, device=None, dtype=None, ops=None):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = ops.Linear(c.hidden_size, self.heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = ops.Linear(c.hidden_size, self.kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ops.Linear(c.hidden_size, self.kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.heads * self.head_dim, c.hidden_size, bias=False, device=device, dtype=dtype)

    def new_state(self, batch, capacity, device, dtype):
        shape = (batch, self.kv_heads, capacity, self.head_dim)
        return {"k": torch.zeros(shape, device=device, dtype=dtype), "v": torch.zeros(shape, device=device, dtype=dtype)}

    def step(self, x, state, pos):
        """x [B, hidden], written at position `pos` of the buffers; attends over positions 0..pos."""
        batch = x.shape[0]
        q = self.q_proj(x).view(batch, self.heads, 1, self.head_dim)
        state["k"][:, :, pos] = self.k_proj(x).view(batch, self.kv_heads, self.head_dim)
        state["v"][:, :, pos] = self.v_proj(x).view(batch, self.kv_heads, self.head_dim)
        out = F.scaled_dot_product_attention(q, state["k"][:, :, :pos + 1], state["v"][:, :, :pos + 1], enable_gqa=True)
        return self.o_proj(out.reshape(batch, self.heads * self.head_dim))

    def forward_sequence(self, x):
        batch, T, _ = x.shape
        q = self.q_proj(x).view(batch, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, T, self.kv_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, T, -1))


class Router(nn.Module):
    """NemotronHTopkRouter (:874-918): fp32 sigmoid scores; the top 6 are chosen on scores + e_score_correction_bias
    and weighted by the plain scores, normalised to sum 1, times routed_scaling_factor."""

    def __init__(self, c, device=None):
        super().__init__()
        self.top_k, self.scale = c.num_experts_per_tok, c.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty((c.n_routed_experts, c.hidden_size), device=device, dtype=torch.float32))
        self.register_buffer("e_score_correction_bias", torch.empty(c.n_routed_experts, device=device, dtype=torch.float32))

    def forward(self, x):
        scores = F.linear(x.float(), cast(self.weight, x, torch.float32)).sigmoid()
        ids = torch.topk(scores + cast(self.e_score_correction_bias, x, torch.float32), self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(-1, ids)
        return ids, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.scale


def relu2(x):
    return torch.relu(x).square()


class Experts(nn.Module):
    """The routed experts of one MoE layer as two banks: up [E, 1856, 2688] and down [E, 2688, 1856].

    `forward(x [R, hidden], ids [R, k], weights [R, k] fp32) -> [R, hidden] fp32`, the weighted sum of each row's
    experts, is the seam the kernels lane replaces: a replacement module takes the same arguments. This plain path
    groups rows by expert and runs F.linear on each expert's weights, copied to the rows' device and dequantized if the
    bank is quantized; the banks may live in host memory."""

    def __init__(self, c, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_experts = c.n_routed_experts
        self.up_proj = ops.MoEExperts(c.n_routed_experts, c.hidden_size, c.moe_intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(c.n_routed_experts, c.moe_intermediate_size, c.hidden_size, bias=False, device=device, dtype=dtype)

    @staticmethod
    def expert(bank, i, x):
        return dequantized(bank.expert_weight(i), x)

    def forward(self, x, ids, weights):
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        flat = ids.flatten()
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=self.num_experts).tolist()
        rows = order // ids.shape[1]
        w = weights.flatten()[order]
        start = 0
        for i, n in enumerate(counts):
            if n == 0:
                continue
            r = rows[start:start + n]
            h = relu2(F.linear(x[r], self.expert(self.up_proj, i, x)))
            out.index_add_(0, r, F.linear(h, self.expert(self.down_proj, i, x)).float() * w[start:start + n, None])
            start += n
        return out


class SharedExpert(nn.Module):
    def __init__(self, c, device=None, dtype=None, ops=None):
        super().__init__()
        self.up_proj = ops.Linear(c.hidden_size, c.moe_shared_expert_intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(c.moe_shared_expert_intermediate_size, c.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        if isinstance(self.up_proj.weight, QuantizedTensor):
            # the plain path of the quantized files (see `dequantized`)
            return F.linear(relu2(F.linear(x, dequantized(self.up_proj.weight, x))), dequantized(self.down_proj.weight, x))
        return self.down_proj(relu2(self.up_proj(x)))


class MoE(nn.Module):
    """NemotronHMOE (:818-871): routed experts summed in fp32 and cast to the compute dtype (:838, :862), plus the
    shared expert."""

    def __init__(self, c, device=None, dtype=None, ops=None):
        super().__init__()
        self.gate = Router(c, device=device)
        self.experts = Experts(c, device=device, dtype=dtype, ops=ops)
        self.shared_experts = SharedExpert(c, device=device, dtype=dtype, ops=ops)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        ids, weights = self.gate(flat)
        routed = self.experts(flat, ids, weights).to(x.dtype)
        return (routed + self.shared_experts(flat)).view_as(x)


class Block(nn.Module):
    def __init__(self, c, kind, device=None, dtype=None, ops=None):
        super().__init__()
        self.kind = kind
        self.norm = ops.RMSNorm(c.hidden_size, eps=c.layer_norm_epsilon, device=device, dtype=dtype)
        if kind == "M":
            self.mixer = Mamba2(c, device=device, dtype=dtype, ops=ops)
        elif kind == "E":
            self.mixer = MoE(c, device=device, dtype=dtype, ops=ops)
        elif kind == "*":
            self.mixer = Attention(c, device=device, dtype=dtype, ops=ops)
        else:
            raise ValueError(f"Nemotron Image: unknown block type {kind!r}")


class NemotronImageModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        c = config
        self.embeddings = ops.Embedding(c.vocab_size, c.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(Block(c, kind, device=device, dtype=dtype, ops=ops) for kind in c.block_types)
        self.norm_f = ops.RMSNorm(c.hidden_size, eps=c.layer_norm_epsilon, device=device, dtype=dtype)
        self.lm_head = ops.Linear(c.hidden_size, c.vocab_size, bias=False, device=device, dtype=dtype)
        self.register_buffer("codebook", torch.empty((NUM_VISION, 24), device=device, dtype=torch.float32))
        # The loader keeps only the vision rows in lm_head and the whole table here, in host memory (outside the module)
        self.__dict__["head_rows"] = None

    def new_state(self, batch, capacity, device, dtype):
        """Per-call decode state: one dict per layer (None for MoE layers)."""
        state = []
        for layer in self.layers:
            if layer.kind == "M":
                state.append(layer.mixer.new_state(batch, device, dtype))
            elif layer.kind == "*":
                state.append(layer.mixer.new_state(batch, capacity, device, dtype))
            else:
                state.append(None)
        return state

    def step_hidden(self, ids, state, pos, dtype):
        """ids [B] at position `pos` -> final hidden [B, hidden] (after norm_f)."""
        x = self.embeddings(ids, out_dtype=dtype)
        for layer, s in zip(self.layers, state):
            h = layer.norm(x)
            if layer.kind == "M":
                h = layer.mixer.step(h, s)
            elif layer.kind == "*":
                h = layer.mixer.step(h, s, pos)
            else:
                h = layer.mixer(h)
            x = x + h
        return self.norm_f(x)

    def forward_sequence(self, ids, dtype):
        """ids [B, T] -> final hidden [B, T, hidden], from an empty state."""
        x = self.embeddings(ids, out_dtype=dtype)
        for layer in self.layers:
            h = layer.norm(x)
            h = layer.mixer(h) if layer.kind == "E" else layer.mixer.forward_sequence(h)
            x = x + h
        return self.norm_f(x)

    def logits(self, hidden):
        """Full-vocabulary logits in fp32 (:1717). With `head_rows`, the rows around the vision codes on the CPU (checks)."""
        if self.head_rows is None:
            return self.lm_head(hidden).float()
        h = hidden.cpu()
        text, tail = (F.linear(h, rows.to(h.dtype)).float() for rows in (self.head_rows[:VISION_START], self.head_rows[VISION_START + NUM_VISION:]))
        return torch.cat((text, self.vision_logits(hidden).cpu(), tail), -1).to(hidden.device)

    def vision_logits(self, hidden):
        """Logits of the vision codes only [.., 196560], fp32: the rows of lm_head generation samples from. Through the
        module so ComfyUI's weight casting and offload apply."""
        if self.head_rows is None:
            return self.lm_head(hidden)[..., VISION_START:VISION_START + NUM_VISION].float()
        return self.lm_head(hidden).float()

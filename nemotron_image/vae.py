"""The Nemotron image tokenizer (L24SQ ViT, `Nemotron-Labs-Image-Tokenizer/`) as a stock ComfyUI VAE.

Latent: l2-normalised codebook vectors [B, 24, H/16, W/16], what Nemotron Image Generate returns. Decode:
post_quant_embed -> 24 ViT blocks with 2-D RoPE -> RMSNorm -> Linear + tanh -> Linear to 16x16x3 patches -> clamp to
[-1, 1] (modeling_l24sq_vit.py:319-430, 607-628). Encode: patch embed -> 24 blocks -> RMSNorm -> quant_embed -> the
nearest codebook vector (:446-484, :580-605), so VAE Encode then VAE Decode is the tokenizer's own reconstruction.
"""
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import comfy.latent_formats
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.quant_ops
import comfy.sd
import comfy.storage
import comfy.utils
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

PATCH = 16


def rope_2d(grid_h, grid_w, head_dim, theta, device):
    """Rotation matrices [1, 1, h*w*2, head_dim/4, 2, 2] for split-half RoPE on each half of a head: the first half turns with
    the row, the second with the column (build_2d_rope_cache and _rotate_half_2d, :59-99)."""
    quarter = head_dim // 4
    inv_freq = 1.0 / (theta ** (torch.arange(0, quarter, device=device, dtype=torch.float32) * 2 / (head_dim // 2)))
    rows = torch.outer(torch.arange(grid_h, device=device, dtype=torch.float32), inv_freq)
    cols = torch.outer(torch.arange(grid_w, device=device, dtype=torch.float32), inv_freq)
    angles = torch.stack((rows[:, None].expand(grid_h, grid_w, quarter), cols[None, :].expand(grid_h, grid_w, quarter)), dim=2)
    cos, sin = angles.cos(), angles.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, grid_h * grid_w * 2, quarter, 2, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads, device=None, dtype=None, ops=None):
        super().__init__()
        self.heads = heads
        self.q_proj = ops.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.k_proj = ops.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ops.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.out_proj = ops.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rope):
        b, n, _ = x.shape
        # each head's two halves are split-half rotations of their own: [b, heads, n * 2, head_dim / 2]
        q = self.q_proj(x).view(b, n, self.heads, 2, -1).transpose(1, 2).reshape(b, self.heads, n * 2, -1)
        k = self.k_proj(x).view(b, n, self.heads, 2, -1).transpose(1, 2).reshape(b, self.heads, n * 2, -1)
        q, k = comfy.quant_ops.ck.apply_rope_split_half(q, k, rope)
        v = self.v_proj(x).view(b, n, self.heads, -1).transpose(1, 2)
        q, k = q.view(b, self.heads, n, -1), k.view(b, self.heads, n, -1)
        x = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                                self.heads, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.out_proj(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden, device=None, dtype=None, ops=None):
        super().__init__()
        self.gate_proj = ops.Linear(dim, hidden, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(dim, hidden, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(hidden, dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, device=None, dtype=None, ops=None):
        super().__init__()
        self.norm1 = ops.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.attn = Attention(dim, heads, device=device, dtype=dtype, ops=ops)
        self.norm2 = ops.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = MLP(dim, hidden, device=device, dtype=dtype, ops=ops)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class Tower(nn.Module):
    def __init__(self, width, layers, heads, hidden, theta, device=None, dtype=None, ops=None):
        super().__init__()
        self.head_dim, self.theta = width // heads, theta
        self.blocks = nn.ModuleList(Block(width, heads, hidden, device=device, dtype=dtype, ops=ops) for _ in range(layers))
        self.norm = ops.RMSNorm(width, eps=1e-6, device=device, dtype=dtype)

    def run(self, x, grid_h, grid_w):
        rope = rope_2d(grid_h, grid_w, self.head_dim, self.theta, x.device)
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x)


class L24SQ(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        width, layers, heads, theta = config["width"], config["layers"], config["heads"], config["vit_rope_theta"]
        hidden = (int(width * config["mlp_ratio"] * 2 / 3) + 63) // 64 * 64
        self.encoder = Tower(width, layers, heads, hidden, theta, device=device, dtype=dtype, ops=ops)
        self.encoder.patch_embed = ops.Conv2d(3, width, PATCH, stride=PATCH, device=device, dtype=dtype)
        self.decoder = Tower(width, layers, heads, hidden, theta, device=device, dtype=dtype, ops=ops)
        self.decoder.ffn = nn.Sequential(ops.Linear(width, config["dim_ffn_output"], device=device, dtype=dtype), nn.Tanh())
        self.decoder.conv_out = ops.Linear(config["dim_ffn_output"], 3 * PATCH * PATCH, device=device, dtype=dtype)
        self.quant_embed = ops.Linear(width, config["embed_dim"], device=device, dtype=dtype)
        self.post_quant_embed = ops.Linear(config["embed_dim"], width, device=device, dtype=dtype)
        self.quantize = nn.Module()
        self.quantize.embedding = ops.Embedding(config["n_embed"], config["embed_dim"], device=device, dtype=dtype)

    def decode(self, z):
        """z [B, 24, h, w] code vectors -> image [B, 3, 16h, 16w] in [-1, 1]."""
        b, _, gh, gw = z.shape
        x = self.post_quant_embed(z.flatten(2).transpose(1, 2))
        x = self.decoder.run(x, gh, gw)
        x = self.decoder.conv_out(self.decoder.ffn(x))
        x = x.view(b, gh, gw, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, gh * PATCH, gw * PATCH)
        return x.clamp(-1, 1)

    def encode(self, pixels):
        """pixels [B, 3, H, W] in [-1, 1], H and W multiples of 16 -> the nearest code vectors [B, 24, H/16, W/16]."""
        x = self.encoder.patch_embed(pixels)
        gh, gw = x.shape[-2:]
        x = self.encoder.run(x.flatten(2).transpose(1, 2), gh, gw)
        z = self.quant_embed(x).transpose(1, 2).unflatten(2, (gh, gw))
        with comfy.ops.CastBiasWeightContext(self.quantize.embedding, z, dtype=torch.float32, offloadable=True) as (codebook, _):
            return codes_to_latent(latent_to_codes(z, codebook), codebook)


def codes_to_latent(codes, codebook):
    """codes [B, h, w] long -> the l2-normalised codebook vectors [B, 24, h, w] (VectorQuantizer.lookup, :480-484)."""
    return F.normalize(codebook[codes], dim=-1).movedim(-1, 1)


def latent_to_codes(latent, codebook):
    """latent [B, 24, h, w] -> codes [B, h, w] long: the codebook vector closest in angle, compared in fp32
    (VectorQuantizer.encode, :446-478)."""
    z = F.normalize(latent.movedim(1, -1).float(), dim=-1)
    codebook = F.normalize(codebook.float(), dim=-1)
    # 512 rows at a time: a 400 MB score block against the 196,560 codes
    codes = [(rows @ codebook.T).argmax(-1) for rows in z.reshape(-1, z.shape[-1]).split(512)]
    return torch.cat(codes).view(z.shape[:-1])


def preview_image(vae, codes, codebook, max_side=512):
    """The rows generated so far, codes [rows, w] long, decoded to a PIL image no larger than max_side for the live
    preview. Runs the model directly, with no load call that could evict another model: the caller loads vae.patcher
    once, in the same load_models_gpu call as the language model, before generating."""
    image = vae.first_stage_model.decode(codes_to_latent(codes[None], codebook).to(vae.device, vae.vae_dtype)).float()
    h, w = image.shape[-2:]
    scale = max_side / max(h, w)
    if scale < 1:
        image = F.interpolate(image, size=(max(1, round(h * scale)), max(1, round(w * scale))), mode="bilinear", antialias=True)
    image = vae.process_output(image)[0].movedim(0, -1)
    return Image.fromarray(image.mul(255).round().byte().cpu().numpy())


class NemotronImage(comfy.latent_formats.LatentFormat):
    latent_channels = 24
    spacial_downscale_ratio = 16
    # No linear preview: a code's colour depends on its neighbours (the encoder and decoder attend globally). Least
    # squares from code vectors to the 16x16-pooled decode scores R^2 0.17-0.19 and a per-code colour table 0.12 over
    # 81,520 patches of 10 real images (dev/decoder/bench.py --fit, dev/decoder/palette.py): both previews are noise.
    # A real decode of the rows so far is the preview (0.05 s for a full 1024^2 grid on the 4090, 0.12 s on the 3090).
    latent_rgb_factors = None


class _DropRecord(logging.Filter):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def filter(self, record):
        return self.text not in record.getMessage()


def load_decoder(path):
    """A stock comfy.sd.VAE whose first_stage_model is the L24SQ tokenizer loaded from `path`."""
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    config = json.loads(metadata["nemotron_image_decoder_config"])
    # The stock constructor sets every default field, finds no architecture for an empty state dict and returns before
    # building one; the model, device, dtype and patcher are set up below. Its warning describes nothing that runs.
    root = logging.getLogger()
    drop = _DropRecord("No VAE weights detected")
    root.addFilter(drop)
    try:
        vae = comfy.sd.VAE(sd={})
    finally:
        root.removeFilter(drop)
    vae.latent_channels = config["embed_dim"]
    vae.downscale_ratio = vae.upscale_ratio = PATCH
    vae.working_dtypes = [torch.bfloat16, torch.float32]
    # Measured above the weights in bf16 (dev/decoder/bench.py, same on both cards): decode 25.3 kB per latent token,
    # linear in tokens and batch (1024 to 36,864 tokens, flash attention); encode the same tower plus a fixed 0.41 GiB
    # for the fp32 nearest-code search. Both estimates sit >= 1.25x over the measurement.
    vae.memory_used_decode = lambda shape, dtype: 16000 * shape[0] * shape[2] * shape[3] * comfy.model_management.dtype_size(dtype)
    vae.memory_used_encode = lambda shape, dtype: 16000 * shape[0] * shape[2] * shape[3] / 256 * comfy.model_management.dtype_size(dtype) + 480 * 2**20

    vae.device = comfy.model_management.vae_device()
    vae.vae_dtype = comfy.model_management.vae_dtype(vae.device, vae.working_dtypes)
    vae.output_device = comfy.model_management.intermediate_device()
    vae.first_stage_model = L24SQ(config, dtype=vae.vae_dtype, ops=comfy.ops.disable_weight_init).eval()
    comfy.model_management.archive_model_dtypes(vae.first_stage_model)
    vae.patcher = comfy.model_patcher.CoreModelPatcher(
        vae.first_stage_model, load_device=vae.device, offload_device=comfy.model_management.vae_offload_device(),
        fast_disk=comfy.storage.state_dict_fast_disk(sd))
    vae.first_stage_model.current_patcher = vae.patcher
    vae.first_stage_model.load_state_dict(sd, strict=True, assign=vae.patcher.is_dynamic())
    if not vae.patcher.is_dynamic():
        vae.first_stage_model.to(vae.vae_dtype)
    vae.model_size()
    return vae

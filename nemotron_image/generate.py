"""Image generation: prefill the prompt, force a newline, `<|img_start|>` and the size header, then per row force
`<|vis_row|>` and sample W/16 vision codes from the head restricted to the vision rows (decision 7: the reference's
distribution conditioned on a valid grid; the reference raises on an invalid one, README.md:364-412).

`generate_codes` is the one entry point: the kernels lane's FastDecoder when it supports the request, else the plain
path below (the reference implementation the fast path is checked against)."""
import logging

import torch

import comfy.model_management
from . import vae
from .model import IMG_START, VIS_ROW, VISION_START

PATCH = 16


def header_text(width, height):
    """The size header the model writes after <|img_start|> (README.md:398: height first)."""
    return f"height={height} width={width}"


def forced_prefix(tokenizer, prompt_ids, width, height):
    """The tokens fed before the first row: the prompt, then what the model writes when left free (measured,
    docs/SPEC.md section 2): a newline, <|img_start|> and the size header, forced to the requested size."""
    return list(prompt_ids) + tokenizer.encode("\n") + [IMG_START] + tokenizer.encode(header_text(width, height))


def image_generator(seed, index, device):
    """Image `index` of a batch draws from its own stream, so it does not depend on the batch size."""
    return torch.Generator(device=device).manual_seed((seed * 0x9E3779B97F4A7C15 + index) % 2 ** 63)


def sample(logits, temperature, top_k, top_p, generator):
    """One id per row of fp32 logits: temperature, then top-k (0 = off) and top-p (1 = off), then a draw by the
    exponential race (argmax of p / Exp(1))."""
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        values, ids = torch.sort(logits, descending=True, dim=-1)
        ranks = torch.arange(values.shape[-1], device=values.device)
        drop = (values.softmax(-1).cumsum(-1) - values.softmax(-1)) >= top_p
        if top_k > 0:
            drop |= ranks >= top_k
        values = values.masked_fill(drop, -torch.inf)
        noise = torch.empty_like(values).exponential_(generator=generator)
        return ids.gather(-1, (values.softmax(-1) / noise).argmax(-1, keepdim=True)).squeeze(-1)
    noise = torch.empty_like(logits).exponential_(generator=generator)
    return (logits.softmax(-1) / noise).argmax(-1)


def generate_codes(nemotron, prompt_ids, width, height, batch, seed, temperature, top_k, top_p, on_row=None, check=None,
                   reserve_bytes=0, extra_patchers=()):
    """Codes [batch, H/16, W/16] (CPU int64, codebook indices). `on_row(row, codes so far)` after each row; `check()`
    every token (Cancel). `extra_patchers` load with the model (the preview decoder); `reserve_bytes` is what they
    need on the card besides their weights."""
    # imported here: experts.py imports this module
    from .experts import FastDecoder
    rows, cols = height // PATCH, width // PATCH
    capacity = len(forced_prefix(nemotron.tokenizer, prompt_ids, width, height)) + rows * (cols + 1)
    reason = "engine: reference" if nemotron.engine == "reference" else FastDecoder.supported(nemotron, batch, capacity, reserve_bytes)
    if reason is None:
        return FastDecoder.generate_codes(nemotron, prompt_ids, width, height, batch, seed, temperature, top_k, top_p, on_row, check,
                                          reserve_bytes=reserve_bytes, extra_patchers=extra_patchers)
    logging.info("Nemotron Image: plain decode (%s)", reason)
    return plain_codes(nemotron, prompt_ids, width, height, batch, seed, temperature, top_k, top_p, on_row, check, extra_patchers)


def plain_codes(nemotron, prompt_ids, width, height, batch, seed, temperature, top_k, top_p, on_row=None, check=None,
                extra_patchers=()):
    """The plain path: the model's step code token by token, one generator per image."""
    model = nemotron.model
    device, dtype = nemotron.patcher.load_device, torch.bfloat16
    rows, cols = height // PATCH, width // PATCH
    forced = forced_prefix(nemotron.tokenizer, prompt_ids, width, height)
    capacity = len(forced) + rows * (cols + 1)
    comfy.model_management.load_models_gpu([nemotron.patcher, *extra_patchers])
    codes = torch.empty((batch, rows, cols), dtype=torch.long, device=device)
    generators = [image_generator(seed, i, device) for i in range(batch)]
    with comfy.model_management.cuda_device_context(device):
        state = model.new_state(batch, capacity, device, dtype)
        pos = 0

        def feed(ids):
            nonlocal pos
            hidden = model.step_hidden(ids, state, pos, dtype)
            pos += 1
            return hidden

        for token in forced:
            feed(torch.full((batch,), token, device=device, dtype=torch.long))
            if check is not None:
                check()
        for r in range(rows):
            hidden = feed(torch.full((batch,), VIS_ROW, device=device, dtype=torch.long))
            for c in range(cols):
                logits = model.vision_logits(hidden)
                code = torch.cat([sample(logits[i:i + 1], temperature, top_k, top_p, g) for i, g in enumerate(generators)])
                codes[:, r, c] = code
                if c + 1 < cols:
                    hidden = feed(code + VISION_START)
                elif r + 1 < rows:
                    feed(code + VISION_START)  # what it predicts is the forced <|vis_row|>
                if check is not None:
                    check()
            if on_row is not None:
                on_row(r, codes[:, :r + 1])
    return codes.cpu()


def codes_to_latent(nemotron, codes):
    """Codes [B, h, w] -> the l2-normalised codebook vectors [B, 24, h, w] fp32 (the VAE decoder's input)."""
    return vae.codes_to_latent(codes.cpu(), nemotron.model.codebook.to("cpu", torch.float32))

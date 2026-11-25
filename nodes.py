"""Nemotron-Labs-Image-30B-A3B nodes: load the model, generate an image's codes as a LATENT, and load the pixel decoder
as a VAE for the stock VAE Decode."""

import time

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.cli_args import LatentPreviewMethod, args
from comfy_api.latest import ComfyExtension, io

from .nemotron_image.generate import PATCH, codes_to_latent, generate_codes
from .nemotron_image.loader import ENGINES, load_nemotron_image
from .nemotron_image.prompt import SYSTEM_PROMPT
from .nemotron_image.vae import load_decoder, preview_image

EXPERTS = {"auto": None, "in RAM": True, "on the GPU": False}
# seconds between live previews: each is a real decode of the rows so far
PREVIEW_INTERVAL = 2.0

NemotronImageModel = io.Custom("NEMOTRON_IMAGE")


def nemotron_files(folder):
    files = folder_paths.get_filename_list(folder)
    return sorted(files, key=lambda f: "nemotron" not in f.lower())


class NemotronImageLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NemotronImageLoader",
            display_name="Load Nemotron Image",
            category="nemotron_image",
            description="Nemotron-Labs-Image-30B-A3B in ComfyUI's format: bf16 (65 GB) or a quantized file (int8 ConvRot "
                        "36 GB, W6A8 30 GB, W4A8 22 GB, W4A8 int8 dense 22 GB).",
            inputs=[
                io.Combo.Input("model_name", options=nemotron_files("diffusion_models")),
                io.Combo.Input("experts", options=list(EXPERTS), default="auto",
                               tooltip="Where the 128 x 23 routed experts (90% of the model) live. auto: on the GPU when the whole "
                                       "file fits there beside ~2.5 GB of working room (W4A8 on a 24 GB card), else in system RAM. "
                                       "In RAM, each token copies the experts it uses to the GPU (slow; needs the file's size in "
                                       "RAM)."),
                io.Combo.Input("engine", options=list(ENGINES), default="auto", advanced=True,
                               tooltip="auto: the pack's CUDA kernels when they support the file and card, else the plain "
                                       "PyTorch path. reference: always the plain path (slow; for checking)."),
            ],
            outputs=[NemotronImageModel.Output()],
        )

    @classmethod
    def execute(cls, model_name, experts="auto", engine="auto") -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return io.NodeOutput(load_nemotron_image(path, engine=engine, experts_in_ram=EXPERTS[experts]))


class NemotronImageGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NemotronImageGenerate",
            display_name="Nemotron Image Generate",
            category="nemotron_image",
            description="Writes the image token by token, row by row (W/16 codes per row, H/16 rows), and returns the codes "
                        "as a LATENT for VAE Decode with the Nemotron Image decoder.",
            inputs=[
                NemotronImageModel.Input("model"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Int.Input("width", default=1024, min=16, max=4096, step=16,
                             tooltip="Multiple of 16. Recommended: 1024x1024, 1184x880, 1248x832, 880x1184."),
                io.Int.Input("height", default=1024, min=16, max=4096, step=16, tooltip="Multiple of 16."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("batch_size", default=1, min=1, max=64),
                io.Float.Input("temperature", default=0.9, min=0.0, max=2.0, step=0.01,
                               tooltip="The model card samples at 0.9."),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="1.0 (off) on the model card."),
                io.Int.Input("top_k", default=0, min=0, max=196560, tooltip="0 (off) on the model card."),
                io.String.Input("system_prompt", multiline=True, optional=True, force_input=True,
                                tooltip="Replaces the model card's system prompt."),
                io.Vae.Input("vae", optional=True,
                             tooltip="The Nemotron Image decoder, for a live preview of the rows so far (a real decode: the "
                                     "codes have no cheap colour preview). Not needed for the image itself."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, prompt, width, height, seed, batch_size, temperature, top_p, top_k, system_prompt=None,
                vae=None) -> io.NodeOutput:
        width, height = width // PATCH * PATCH, height // PATCH * PATCH
        ids = model.tokenizer.prompt_ids(prompt, width, height, SYSTEM_PROMPT if system_prompt is None else system_prompt)
        rows = height // PATCH
        progress = comfy.utils.ProgressBar(rows)
        previews = vae if vae is not None and args.preview_method != LatentPreviewMethod.NoPreviews else None
        last = [0.0]

        def on_row(row, codes):
            preview = None
            if previews is not None and (time.monotonic() - last[0] > PREVIEW_INTERVAL or row + 1 == rows):
                codebook = model.model.codebook
                preview = ("JPEG", preview_image(previews, codes[0].to(codebook.device), codebook), 512)
                last[0] = time.monotonic()
            progress.update_absolute(row + 1, rows, preview)

        extra = [previews.patcher] if previews is not None else []
        reserve = previews.memory_used_decode((1, 24, rows, width // PATCH), previews.vae_dtype) if previews is not None else 0
        codes = generate_codes(model, ids, width, height, batch_size, seed, temperature, top_k, top_p, on_row,
                               comfy.model_management.throw_exception_if_processing_interrupted, reserve, extra)
        return io.NodeOutput({"samples": codes_to_latent(model, codes)})


class NemotronImageDecoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NemotronImageDecoderLoader",
            display_name="Load Nemotron Image Decoder",
            category="nemotron_image",
            description="The L24SQ pixel decoder (Nemotron-Labs-Image-Tokenizer) as a VAE: VAE Decode turns Nemotron Image "
                        "Generate's latent into the image. VAE Encode gives the nearest codes of an image.",
            inputs=[io.Combo.Input("vae_name", options=nemotron_files("vae"))],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        return io.NodeOutput(load_decoder(folder_paths.get_full_path_or_raise("vae", vae_name)))


class NemotronImageExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NemotronImageLoader, NemotronImageGenerate, NemotronImageDecoderLoader]


async def comfy_entrypoint() -> NemotronImageExtension:
    return NemotronImageExtension()

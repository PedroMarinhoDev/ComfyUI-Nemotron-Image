"""The generation prompt, as the model card builds it (README.md:353-430): the fixed system prompt, the user turn
`Generate a {W}x{H} image.\n\n{prompt}`, and the chat template's generation prompt with thinking off
(chat_template.jinja: `<|im_start|>assistant\n<think></think>`), tokenized like UnitronLazyVisionTokenizer
(tokenization_unitron_lazy_vision.py:425-441): the base tokenizer on the rendered text, which holds no vision tokens.
"""
import os

from tokenizers import Tokenizer

TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvidia", "tokenizer.json")
SYSTEM_PROMPT = ("Generate an image matching the requested resolution by detailing the color, quantity, text, shape, "
                 "size, texture, spatial relationships of the objects and background.")


def render(prompt, width, height, system_prompt=SYSTEM_PROMPT):
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nGenerate a {width}x{height} image.\n\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>")


class PromptTokenizer:
    def __init__(self, path=TOKENIZER_PATH):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def prompt_ids(self, prompt, width, height, system_prompt=SYSTEM_PROMPT):
        return self.encode(render(prompt, width, height, system_prompt))

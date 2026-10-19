"""Pinned offline Transformers inference on one selected CUDA GPU."""

from collections.abc import Sequence
from typing import override

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from target.model import Model

class QwenModel(Model):
    """One loaded set of pinned weights shared by independent chats."""

    def __init__(self, device: torch.device, tokens: int,
                 tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
        """Keep one loaded model and tokenizer on the selected device.

        Args:
            device (torch.device): Explicit CUDA device.
            tokens (int): Maximum generated tokens per request.
            tokenizer (AutoTokenizer): Pinned local tokenizer.
            model (AutoModelForCausalLM): Pinned local weights.

        Raises:
            ValueError: If the token limit is invalid.
        """
        super().__init__()
        if tokens < 1:
            raise ValueError("Token limit must be positive")
        self.device: torch.device = device
        self.tokens = tokens
        self.tokenizer: AutoTokenizer = tokenizer
        self.model: AutoModelForCausalLM = model

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        batch = (
            self.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.device)
        )
        with torch.inference_mode():
            output = self.model.generate(
                **batch,
                max_new_tokens=self.tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            output[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True
        )

    @override
    def revision(self) -> str:
        return "7ae557604adf67be50417f59c2c2f167def9a775"

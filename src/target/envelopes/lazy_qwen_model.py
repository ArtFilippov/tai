"""Compose offline Qwen resources into a deferred shared model."""

import torch

from target.envelopes.model_envelope import ModelEnvelope
from target.lazy_model import LazyModel
from target.local_qwen_assets import LocalQwenAssets
from target.qwen_model import QwenModel


class LazyQwenModel(ModelEnvelope):
    """Configure one Qwen model without loading weights during assembly."""

    def __init__(self, device: torch.device, tokens: int) -> None:
        """Defer loading the selected Qwen model until it is used.

        Args:
            device (torch.device): Selected CUDA device.
            tokens (int): Maximum generated tokens per request.
        """
        super().__init__(
            LazyModel(
                LocalQwenAssets(device),
                lambda assets: QwenModel(device, tokens, assets.tokenizer, assets.model)
            )
        )

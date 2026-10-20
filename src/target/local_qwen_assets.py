"""Load a prepared Qwen snapshot once for a selected device."""

from typing import override

import torch
from huggingface_hub import snapshot_download
from qraft import cached_property
from transformers import AutoModelForCausalLM, AutoTokenizer

from target.qwen_assets import QwenAssets


class LocalQwenAssets(QwenAssets):
    """Keep local weights and tokenizer cached separately per experiment."""

    def __init__(self, device: torch.device) -> None:
        """Select a device without loading any files.

        Args:
            device (torch.device): Device for the weights after loading.
        """
        super().__init__()
        self.device = device

    @cached_property
    @override
    def path(self) -> str:
        try:
            return snapshot_download(
                "Qwen/Qwen2.5-0.5B-Instruct",
                revision="7ae557604adf67be50417f59c2c2f167def9a775",
                local_files_only=True
            )
        except OSError as exc:
            raise FileNotFoundError("Prepare the pinned Qwen snapshot before running offline") from exc

    @cached_property
    @override
    def tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            self.path, local_files_only=True
        )

    @cached_property
    @override
    def model(self) -> AutoModelForCausalLM:
        weights = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, torch_dtype=torch.float16
        )
        _ = weights.to(self.device)
        _ = weights.eval()
        return weights

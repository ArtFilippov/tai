"""Local Qwen teacher-forced logits with native AutoDAN target slicing."""

from typing import Any, override

import torch

from target.logits import Logits
from target.qwen_assets import QwenAssets


class QwenLogits(Logits):
    """Compute native cross-entropy without generating or changing a chat."""

    def __init__(self, assets: QwenAssets, device: torch.device) -> None:
        """Keep a shared pinned snapshot for local inference.

        Args:
            assets (QwenAssets): Local Qwen tokenizer and weights.
            device (torch.device): Selected inference device.
        """
        super().__init__()
        self.assets = assets
        self.device = device

    @override
    def profile(self) -> dict[str, str]:
        return {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775"
        }

    @override
    def score(self, prefix: str, base: str, target: str) -> float:
        tokenizer: Any = self.assets.tokenizer
        labels: torch.Tensor = tokenizer(
            [target], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)
        batch: dict[str, torch.Tensor] = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{prefix} {base}"
                },
                {
                    "role": "assistant",
                    "content": target
                }
            ],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        return self.loss(
            ids, mask, labels
        )

    def loss(self, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute native teacher-forced target-token mean cross-entropy.

        Args:
            ids (torch.Tensor): Templated prefix, request and completion.
            mask (torch.Tensor): Attention mask for the complete message.
            labels (torch.Tensor): Completion tokens without special tokens.

        Returns:
            float: Native mean loss over the completion slice.
        """
        model: Any = self.assets.model
        with torch.inference_mode():
            logits: torch.Tensor = model(
                input_ids=ids, attention_mask=mask
            ).logits
            shift = ids.shape[1] - labels.shape[1]
            loss = (
                torch.nn.functional.cross_entropy(
                    logits[..., shift - 1:-1, :].contiguous()
                    .view(
                        -1, logits.shape[-1]
                    ),
                    labels.view(-1),
                    reduction="none"
                ).mean()
            )
        return loss.item()

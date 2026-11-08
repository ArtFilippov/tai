"""Batched causal logits from any compatible Transformers model."""

from collections.abc import Sequence
from typing import Any, override

import torch
from target.token_inference import TokenInference
from target.transformers_assets import TransformersAssets


class TransformersTokenInference(TokenInference):
    """Pad token sequences only inside the Transformers backend."""

    def __init__(self, assets: TransformersAssets, device: torch.device,
                 identity: dict[str, str]) -> None:
        """Keep any causal model lazy until batched inference.

        Args:
            assets (TransformersAssets): Deferred causal model resources.
            device (torch.device): Inference device.
            identity (dict[str, str]): Model profile for the journal.
        """
        super().__init__()
        self.assets = assets
        self.device = device
        self.identity = identity

    @override
    def profile(self) -> dict[str, str]:
        return self.identity

    @override
    def infer(self, sequences: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not sequences:
            return []
        if any(sequence.ndim != 1 or sequence.numel() < 1 for sequence in sequences):
            raise ValueError("Inference requires nonempty token sequences")
        lengths = [sequence.numel() for sequence in sequences]
        width = max(lengths)
        ids = torch.zeros(
            (
                len(sequences),
                width
            ),
            dtype=torch.long, device=self.device
        )
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :lengths[index]] = sequence.to(self.device)
            mask[index, :lengths[index]] = 1
        with torch.inference_mode():
            model: Any = self.assets.model
            logits: torch.Tensor = model(
                input_ids=ids, attention_mask=mask
            ).logits
        return [logits[index, :length] for index, length in enumerate(lengths)]

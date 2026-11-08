"""Assemble pinned Qwen resources behind general token probabilities."""

import torch

from target.computed_token_probabilities import ComputedTokenProbabilities
from target.envelopes.token_probabilities_envelope import TokenProbabilitiesEnvelope
from target.paired_token_source import PairedTokenSource
from target.qwen_assets import QwenAssets
from target.transformers_token_context import TransformersTokenContext
from target.transformers_token_inference import TransformersTokenInference


class QwenTokenProbabilities(TokenProbabilitiesEnvelope):
    """Supply one lazy Qwen model through the general causal token contract."""

    def __init__(self, assets: QwenAssets, device: torch.device) -> None:
        """Compose the model and matching tokenizer without loading either.

        Args:
            assets (QwenAssets): Pinned local model and tokenizer.
            device (torch.device): Inference device.
        """
        super().__init__(
            ComputedTokenProbabilities(
                PairedTokenSource(
                    TransformersTokenInference(
                        assets, device,
                        {
                            "id": "Qwen/Qwen2.5-0.5B-Instruct",
                            "revision": "7ae557604adf67be50417f59c2c2f167def9a775"
                        }
                    ),
                    TransformersTokenContext(
                        assets, device
                    )
                )
            )
        )

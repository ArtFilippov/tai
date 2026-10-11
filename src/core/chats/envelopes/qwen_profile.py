"""Validate a local Transformers chat profile without loading the weights."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import override

from ..qwen_checkpoint_contract import LocalCheckpointPort
from ..qwen_generation_utils import generation_config, validate_dtype, validated_parameters
from ..qwen_profile_contract import QwenProfilePort
from ..qwen_validation_utils import validate_system
from .qwen_checkpoint_utils import local_checkpoint

GENERATION_OPTION = "generation"


@dataclass(frozen=True)
class QwenProfile(QwenProfilePort):
    """Eagerly validated settings independent of model loading and dialogue.

    Attributes:
        model (tuple[str, str]): Model identity and pinned revision.
        cache_dir (str | Path): Prepared offline cache path.
        device (str): Explicit CUDA device.
        options (dict[str, object]): Detached profile options.
    """

    model: tuple[str, str]
    cache_dir: str | Path
    device: str
    options: dict[str, object]

    def __post_init__(self) -> None:
        if unknown := self.options.keys() - {"dtype", "system", "generation"}:
            raise TypeError(f"Unexpected QwenChat options: {', '.join(sorted(unknown))}")
        _ = (self.checkpoint(), self.generation(), self.system)
        if GENERATION_OPTION in self.options:
            self.options[GENERATION_OPTION] = validated_parameters(self.options[GENERATION_OPTION])

    @override
    def checkpoint(self) -> LocalCheckpointPort:
        _ = self.dtype()
        return local_checkpoint(
            self.model[0], self.model[1], self.cache_dir, self.device,
        ).resolved()

    @override
    def dtype(self) -> str:
        return validate_dtype(self.options.get("dtype", "float16"))

    @override
    def generation(self) -> Mapping[str, int | float | bool]:
        return generation_config(self.options.get(GENERATION_OPTION))

    @property
    @override
    def system(self) -> str:
        return validate_system(self.options.get("system", ""))

    @override
    def details(self) -> dict[str, object]:
        return {
            **self.checkpoint().details(), "dtype": self.dtype(),
            "generation": dict(self.generation()),
        }

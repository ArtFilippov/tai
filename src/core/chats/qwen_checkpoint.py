"""Prepared local model profile and its immutable cache identity."""

import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override

from .qwen_checkpoint_contract import LocalCheckpointPort


@dataclass(frozen=True)
class LocalCheckpoint(LocalCheckpointPort):
    """Pin one local snapshot, CUDA device, and cache for repeatable inference.

    Attributes:
        model (str): Hugging Face model identity.
        revision (str): Pinned commit hash.
        cache (Path): Prepared Hub cache.
        device (str): Explicit CUDA device.
    """

    model: str
    revision: str
    cache: Path
    device: str

    @property
    @override
    def snapshot(self) -> Path:
        return (
            self.cache / ("models--" + self.model.replace("/", "--")) / "snapshots" / self.revision
        )

    @property
    @override
    def preparation(self) -> str:
        return " ".join((
            "uv run --locked --extra qwen huggingface-cli download", shlex.quote(self.model),
            "--revision", self.revision, "--cache-dir", shlex.quote(str(self.cache)),
        ))

    @override
    def key(self, dtype: str) -> tuple[str, str, str, str]:
        return self.model, self.revision, self.device, dtype

    @override
    def details(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "cache_dir": str(self.cache),
            "device": self.device,
        }

    @override
    def resolved(self) -> LocalCheckpointPort:
        return replace(self, cache=self.cache.expanduser().resolve())

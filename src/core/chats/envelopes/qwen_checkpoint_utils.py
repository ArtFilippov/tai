"""Compose validated local checkpoint identity before model access."""

from pathlib import Path

from ..qwen_checkpoint import LocalCheckpoint
from ..qwen_validation_utils import cache, identifier


def local_checkpoint(
    model: str, revision: str, cache_dir: str | Path, device: str,
) -> LocalCheckpoint:
    return LocalCheckpoint(
        identifier(model, r"[\w-]+/[\w.-]+", "model must be a Hugging Face owner/model identifier"),
        identifier(revision, r"[0-9a-f]{40}", "revision must be a pinned 40-character commit hash"),
        cache(cache_dir),
        identifier(device, r"cuda:(0|[1-9][0-9]*)", "device must explicitly select cuda:N"),
    )

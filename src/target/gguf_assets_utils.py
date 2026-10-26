"""Locate exactly one pinned GGUF file for an offline Attacker."""

from pathlib import Path

from huggingface_hub import snapshot_download

def gguf_path(offline: bool, repository: str, revision: str, filename: str) -> Path:
    """Resolve the pinned file without downloading during a run.

    Args:
        offline (bool): Require prepared files when true.
        repository (str): Pinned Hugging Face repository.
        revision (str): Pinned commit.
        filename (str): Exact quantization file.

    Returns:
        Path: Exact pinned quantization in the Hugging Face cache.

    Raises:
        FileNotFoundError: If the pinned file is not prepared offline.
    """
    try:
        path = Path(
                   snapshot_download(
                       repo_id=repository, revision=revision, allow_patterns=filename, local_files_only=offline
                   )
               ) / filename
    except OSError as exc:
        raise FileNotFoundError(f"Prepare {repository}@{revision}/{filename} with 'tai-qwen prepare --attacker gguf' before running offline") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Pinned GGUF file is missing: {path}")
    return path

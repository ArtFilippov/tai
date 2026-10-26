"""Locate the pinned Qwen snapshot for preparation and offline runs."""

from huggingface_hub import snapshot_download

def snapshot(offline: bool) -> str:
    """Locate the pinned Qwen snapshot, downloading only during preparation.

    Args:
        offline (bool): Require an already prepared local snapshot.

    Returns:
        str: Local snapshot path.

    Raises:
        FileNotFoundError: If the requested snapshot is unavailable locally.
    """
    try:
        return snapshot_download(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            local_files_only=offline,
        )
    except OSError as exc:
        raise FileNotFoundError("Prepare the pinned Qwen snapshot before running offline") from exc

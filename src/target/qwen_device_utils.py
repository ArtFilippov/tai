"""Validate and select an explicit CUDA device for local Qwen."""

import torch


def cuda_device(index: int) -> torch.device:
    """Select an available CUDA GPU.

    Args:
        index (int): Requested CUDA GPU index.

    Returns:
        torch.device: Validated CUDA device.

    Raises:
        ValueError: If the selected GPU is unavailable.
    """
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA GPU {index} is unavailable")
    return torch.device(
        "cuda", index
    )

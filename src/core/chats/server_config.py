"""Validated, GPU-library-free configuration of the shared vLLM resource."""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .qwen_config import LocalCheckpoint, generation_config

VLLM_VERSION = "0.6.4.post1"
SERVER_DTYPE = "float16"


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class ServerLimits:
    max_model_len: int = 8192
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.3
    startup_timeout: float = 300.0
    shutdown_timeout: float = 10.0

    def __post_init__(self) -> None:
        _ = positive_integer(self.max_model_len, "max_model_len")
        _ = positive_integer(self.max_num_seqs, "max_num_seqs")
        for name in ("gpu_memory_utilization", "startup_timeout", "shutdown_timeout"):
            _ = positive_number(getattr(self, name), name)
        if self.gpu_memory_utilization >= 1:
            raise ValueError("gpu_memory_utilization must be less than 1")


@dataclass(frozen=True)
class ServerConfig:
    checkpoint: LocalCheckpoint
    python: Path
    limits: ServerLimits = ServerLimits()
    dtype: str = SERVER_DTYPE

    def __post_init__(self) -> None:
        if self.dtype != SERVER_DTYPE:
            raise ValueError("vLLM Qwen profile requires float16")
        if not self.python.is_absolute():
            raise ValueError("server.python must be an absolute interpreter path")

    def settings(self) -> dict[str, object]:
        return {
            "backend": "vllm", "model": self.checkpoint.model,
            "revision": self.checkpoint.revision, "snapshot": str(self.checkpoint.snapshot),
            "device": self.checkpoint.device, "dtype": self.dtype, "python": str(self.python),
            "expected_version": VLLM_VERSION, "tensor_parallel_size": 1,
            "enforce_eager": True, "generation_config": "explicit_request_parameters", **asdict(self.limits),
            "guided_decoding_backend": "lm-format-enforcer", "attention_backend": "XFORMERS",
        }


def http_generation(parameters: Mapping[str, object] | None) -> dict[str, object]:
    values = generation_config(parameters)
    return {
        "max_tokens": values["max_new_tokens"], "temperature": values.get("temperature", 0.0),
        "top_p": values.get("top_p", 1.0), "top_k": values.get("top_k", 0) or -1,
        "repetition_penalty": values["repetition_penalty"], "n": 1, "stream": False,
        "seed": 0,
    }

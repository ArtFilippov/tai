"""Validated, GPU-library-free configuration of the shared vLLM resource."""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from ..qwen_checkpoint_contract import LocalCheckpointPort
from ..server_config_port.server_config_port import ServerConfigPort
from ..server_limits_port.server_limits_port import ServerLimitsPort
from .server_limits import ServerLimits

VLLM_VERSION = "0.6.4.post1"
SERVER_DTYPE = "float16"


@dataclass(frozen=True)
class ServerConfig(ServerConfigPort):
    checkpoint: LocalCheckpointPort
    python: Path
    limits: ServerLimits = ServerLimits()
    dtype: str = SERVER_DTYPE

    def __post_init__(self) -> None:
        if self.dtype != SERVER_DTYPE:
            raise ValueError("vLLM Qwen profile requires float16")
        if not self.python.is_absolute():
            raise ValueError("server.python must be an absolute interpreter path")

    def settings(self) -> dict[str, object]:
        model, revision, device, _ = tuple(self.checkpoint.key(self.dtype))
        return {
            "backend": "vllm", "model": model,
            "revision": revision, "snapshot": str(self.checkpoint.snapshot),
            "device": device, "dtype": self.dtype, "python": str(self.python),
            "expected_version": self.version(), "tensor_parallel_size": 1,
            "enforce_eager": True, "generation_config": "explicit_request_parameters",
            **self.limits.settings(),
            "guided_decoding_backend": "lm-format-enforcer", "attention_backend": "XFORMERS",
        }

    @override
    def version(self) -> str:
        return VLLM_VERSION

    @override
    def snapshot(self) -> Path:
        return self.checkpoint.snapshot

    @override
    def device(self) -> str:
        return tuple(self.checkpoint.key(self.dtype))[2]

    @override
    def interpreter(self) -> Path:
        return self.python

    @override
    def precision(self) -> str:
        return self.dtype

    @override
    def resources(self) -> ServerLimitsPort:
        return self.limits

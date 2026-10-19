"""Validated resource limits and CLI flags of the managed vLLM process."""

from dataclasses import dataclass, field
from typing import override

from ..server_limits_port.server_limits_port import ServerLimitsPort
from .server_validation_utils import positive_integer, positive_number


@dataclass(frozen=True)
class ServerLimits(ServerLimitsPort):
    """Constrain context size, scheduling, memory and lifecycle timeouts.

    Attributes:
        max_model_len (int, default=field(default=8192)): Maximum context length.
        max_num_seqs (int, default=32): Scheduler capacity.
        gpu_memory_utilization (float, default=0.3): Fraction of GPU memory reserved.
        startup_timeout (float, default=field(default=300.0)): Maximum startup time in seconds.
        shutdown_timeout (float, default=field(default=10.0)): Graceful shutdown time in seconds.
    """

    max_model_len: int = field(default=8192)
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.3
    startup_timeout: float = field(default=300.0)
    shutdown_timeout: float = field(default=10.0)

    def __post_init__(self) -> None:
        _ = positive_integer(self.max_model_len, "max_model_len")
        _ = positive_integer(self.max_num_seqs, "max_num_seqs")
        for name in ("gpu_memory_utilization", "startup_timeout", "shutdown_timeout"):
            _ = positive_number(getattr(self, name), name)
        if self.gpu_memory_utilization >= 1:
            raise ValueError("gpu_memory_utilization must be less than 1")

    @override
    def settings(self) -> dict[str, object]:
        return {
            "max_model_len": self.max_model_len, "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "startup_timeout": self.startup_timeout, "shutdown_timeout": self.shutdown_timeout,
        }

    @override
    def flags(self) -> list[str]:
        return [
            "--max-model-len", str(self.max_model_len), "--max-num-seqs", str(self.max_num_seqs),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

    @override
    def deadlines(self) -> tuple[float, float]:
        return self.startup_timeout, self.shutdown_timeout

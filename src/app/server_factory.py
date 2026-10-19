"""Composition of an explicitly owned server without importing GPU libraries."""

from pathlib import Path

from core.chats.envelopes.qwen_checkpoint_utils import local_checkpoint
from core.chats.envelopes.qwen_server import QwenServer
from core.chats.envelopes.server_config import ServerConfig
from core.chats.envelopes.server_limits import ServerLimits
from core.chats.envelopes.server_failures import ServerFailures
from core.chats.envelopes.server_validation_utils import positive_integer, positive_number
from core.chats.server_process import ServerProcess
from core.primitives.diagnostics import RedactionFactory

from .configuration_values import string, table


def create_server(
    value: object, log_path: Path, redaction: RedactionFactory,
) -> QwenServer | None:
    if value is None:
        return None
    data = table(value, "server", {
        "python", "model", "revision", "cache_dir", "device", "dtype", "max_model_len",
        "max_num_seqs", "gpu_memory_utilization", "startup_timeout", "shutdown_timeout",
    })
    checkpoint = local_checkpoint(
        string(data.get("model"), "server.model"), string(data.get("revision"), "server.revision"),
        Path(string(data.get("cache_dir"), "server.cache_dir")).expanduser().resolve(),
        string(data.get("device"), "server.device"),
    )
    limits = ServerLimits(
        positive_integer(data.get("max_model_len", 8192), "max_model_len"),
        positive_integer(data.get("max_num_seqs", 32), "max_num_seqs"),
        positive_number(data.get("gpu_memory_utilization", 0.3), "gpu_memory_utilization"),
        positive_number(data.get("startup_timeout", 300.0), "startup_timeout"),
        positive_number(data.get("shutdown_timeout", 10.0), "shutdown_timeout"),
    )
    config = ServerConfig(
        checkpoint, Path(string(data.get("python"), "server.python")).expanduser(), limits,
        string(data.get("dtype", "float16"), "server.dtype"),
    )
    return QwenServer(config, redaction, ServerProcess(config, log_path, ServerFailures()))

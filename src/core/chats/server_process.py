"""Owned local vLLM process group and authenticated readiness probes."""

import json
import os
import signal
import socket
import subprocess
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_hex
from threading import Event
from time import monotonic
from typing import override, cast
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from .server_config import VLLM_VERSION, ServerConfig
from .server_ports import ServerProcessPort


class QwenServerError(RuntimeError):
    """Startup, ownership, or liveness failure of the managed model server."""


def _port() -> int:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return int(reservation.getsockname()[1])


def _signal_group(pid: int, sig: signal.Signals) -> None:
    with suppress(ProcessLookupError):
        os.killpg(pid, sig)


@dataclass
class ServerProcess(ServerProcessPort):
    config: ServerConfig
    log_path: Path
    model: str = field(default_factory=lambda: "tai-" + uuid4().hex)
    token: str = field(default_factory=lambda: token_hex(32), repr=False)
    process: subprocess.Popen[bytes] | None = None
    port: int = 0
    version: str | None = None

    @property
    @override
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    @override
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def _command(self) -> list[str]:
        limits = self.config.limits
        return [
            str(self.config.python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.config.checkpoint.snapshot),
            "--tokenizer", str(self.config.checkpoint.snapshot),
            "--served-model-name", self.model, "--api-key", self.token,
            "--host", "127.0.0.1", "--port", str(self.port),
            "--dtype", self.config.dtype, "--device", "cuda", "--tensor-parallel-size", "1",
            "--max-model-len", str(limits.max_model_len), "--max-num-seqs", str(limits.max_num_seqs),
            "--gpu-memory-utilization", str(limits.gpu_memory_utilization),
            "--enforce-eager", "--disable-log-requests",
            "--guided-decoding-backend", "lm-format-enforcer",
        ]

    @override
    def launch(self) -> None:
        if not self.config.checkpoint.snapshot.is_dir():
            raise QwenServerError(f"Prepare the pinned local snapshot: {self.config.checkpoint.snapshot}")
        self.port = _port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ, "CUDA_VISIBLE_DEVICES": self.config.checkpoint.device.split(":")[1],
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_ATTENTION_BACKEND": "XFORMERS",
        }
        with self.log_path.open("ab") as output, ExitStack() as ownership:
            process = ownership.enter_context(subprocess.Popen(
                self._command(), stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                env=environment, start_new_session=True,
            ))
            self.process = process
            _ = ownership.pop_all()  # Ownership transfers to close(), including partial startup.

    @override
    def check_alive(self) -> None:
        if self.process is None or self.process.poll() is not None:
            code = self.process.returncode if self.process is not None else None
            raise QwenServerError(f"vLLM process exited ({code}); see {self.log_path}")

    def _get(self, path: str) -> object:
        request = Request(self.url + path, headers={"Authorization": "Bearer " + self.token})
        with build_opener(ProxyHandler({})).open(request, timeout=0.5) as response:
            return json.load(response)

    def _ready(self) -> bool:
        try:
            models = self._get("/v1/models")
        except (URLError, TimeoutError, ConnectionError):
            return False
        if not _has_model(models, self.model):
            raise QwenServerError("vLLM readiness returned an unexpected model/owner")
        version = self._get("/version")
        if not isinstance(version, dict) or cast(dict[str, object], version).get("version") != VLLM_VERSION:
            raise QwenServerError(f"vLLM version must be {VLLM_VERSION}")
        self.version = VLLM_VERSION
        return True

    @override
    def await_ready(self, closing: Event) -> None:
        deadline = monotonic() + self.config.limits.startup_timeout
        for _ in iter(lambda: closing.wait(0.05), True):
            self.check_alive()
            if self._ready():
                return
            if monotonic() >= deadline:
                raise QwenServerError(f"vLLM startup timed out; see {self.log_path}")
        raise QwenServerError("vLLM closed during startup")

    @override
    def close(self) -> None:
        if (process := self.process) is None:
            return
        _signal_group(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            _ = process.wait(timeout=self.config.limits.shutdown_timeout)
        # Workers may survive even when their parent exited normally.
        _signal_group(process.pid, signal.SIGKILL)
        _ = process.wait()
        self.process = None


def _has_model(value: object, model: str) -> bool:
    if not isinstance(value, dict):
        return False
    entries = cast(dict[str, object], value).get("data")
    if not isinstance(entries, list):
        return False
    return any(isinstance(entry, dict) and cast(dict[str, object], entry).get("id") == model
               for entry in cast(list[object], entries))

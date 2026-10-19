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

from .server_config_port.server_config_port import ServerConfigPort
from .server_failure_port import ServerFailurePort
from .server_ports import ServerProcessPort


@dataclass
class ServerProcess(ServerProcessPort):
    config: ServerConfigPort
    log_path: Path
    errors: ServerFailurePort
    port: int = 0
    identities: tuple[str, str] = field(
        default_factory=lambda: ("tai-" + uuid4().hex, token_hex(32)), repr=False,
    )
    processes: list[subprocess.Popen[bytes]] = field(default_factory=list[subprocess.Popen[bytes]])
    versions: list[str] = field(default_factory=list[str])

    @override
    def identity(self) -> tuple[str, str, str]:
        return f"http://127.0.0.1:{self.port}", *self.identities

    @override
    def status(self) -> dict[str, object]:
        return {
            "log_path": str(self.log_path),
            "pid": self.processes[0].pid if self.processes else None,
            "version": self.versions[0] if self.versions else None,
        }

    def command(self) -> list[str]:
        limits = self.config.resources()
        return [
            str(self.config.interpreter()), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.config.snapshot()),
            "--tokenizer", str(self.config.snapshot()),
            "--served-model-name", self.identities[0], "--api-key", self.identities[1],
            "--host", "127.0.0.1", "--port", str(self.port),
            "--dtype", self.config.precision(), "--device", "cuda", "--tensor-parallel-size", "1",
            *limits.flags(),
            "--enforce-eager", "--disable-log-requests",
            "--guided-decoding-backend", "lm-format-enforcer",
        ]

    def reserve(self) -> None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = int(reservation.getsockname()[1])

    @override
    def launch(self) -> None:
        if not self.config.snapshot().is_dir():
            self.errors.reject(f"Prepare the pinned local snapshot: {self.config.snapshot()}")
        self.reserve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ, "CUDA_VISIBLE_DEVICES": self.config.device().split(":")[1],
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ATTENTION_BACKEND": "XFORMERS",
        }
        with self.log_path.open("ab") as output, ExitStack() as ownership:
            process = ownership.enter_context(subprocess.Popen(
                self.command(), stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                env=environment, start_new_session=True,
            ))
            self.processes.append(process)
            _ = ownership.pop_all()  # Ownership transfers to close(), including partial startup.

    @override
    def check_alive(self) -> None:
        process = self.processes[0] if self.processes else None
        if process is None or process.poll() is not None:
            code = process.returncode if process is not None else None
            self.errors.reject(f"vLLM process exited ({code}); see {self.log_path}")

    def get(self, path: str) -> object:
        request = Request(
            self.identity()[0] + path, headers={"Authorization": "Bearer " + self.identities[1]},
        )
        with build_opener(ProxyHandler({})).open(request, timeout=0.5) as response:
            return json.load(response)

    def has_model(self, value: object) -> bool:
        if isinstance(value, dict):
            entries = cast(dict[str, object], value).get("data")
            if isinstance(entries, list):
                return any(
                    isinstance(entry, dict)
                    and cast(dict[str, object], entry).get("id") == self.identities[0]
                    for entry in cast(list[object], entries)
                )
        return False

    def ready(self) -> bool:
        try:
            models = self.get("/v1/models")
        except (URLError, TimeoutError, ConnectionError):
            return False
        self.errors.require(
            self.has_model(models), "vLLM readiness returned an unexpected model/owner",
        )
        version = self.get("/version")
        if (
            isinstance(version, dict)
            and cast(dict[str, object], version).get("version") == self.config.version()
        ):
            self.versions.append(self.config.version())
            return True
        self.errors.reject(f"vLLM version must be {self.config.version()}")

    @override
    def await_ready(self, closing: Event) -> None:
        deadline = monotonic() + self.config.resources().deadlines()[0]
        for _ in iter(lambda: closing.wait(0.05), True):
            self.check_alive()
            if self.ready():
                return
            if monotonic() >= deadline:
                self.errors.reject(f"vLLM startup timed out; see {self.log_path}")
        self.errors.reject("vLLM closed during startup")

    def signal(self, sig: signal.Signals) -> None:
        if self.processes:
            with suppress(ProcessLookupError):
                os.killpg(self.processes[0].pid, sig)

    @override
    def close(self) -> None:
        if not self.processes:
            return
        process = self.processes[0]
        self.signal(signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            _ = process.wait(timeout=self.config.resources().deadlines()[1])
        # Workers may survive even when their parent exited normally.
        self.signal(signal.SIGKILL)
        _ = process.wait()
        self.processes.clear()

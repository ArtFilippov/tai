"""The shared server admits simultaneous exchanges without sharing a dialogue."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from typing import override
from unittest import TestCase

from core.chats.envelopes.qwen_server import QwenServer
from core.chats.envelopes.server_chat import PreparedRole, QwenServerChat
from core.chats.envelopes.server_config import ServerConfig
from core.chats.qwen_checkpoint import LocalCheckpoint
from core.chats.server_ports import ServerProcessPort
from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage


class TestQwenServer(TestCase, ServerProcessPort):
    """Supply an in-memory process and transport for independent role chats.

    Args:
        methodName (str, default='runTest'): TestCase method to execute.
    """

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.launches = 0
        self.model = "local-test"
        self.token = "test-key"

    @override
    def identity(self) -> tuple[str, str, str]:
        return f"http://localhost:{self.launches}", self.model, self.token

    @override
    def status(self) -> dict[str, object]:
        return {"log_path": str(Path(f"unused-{self.launches}.log")), "pid": None, "version": None}

    @override
    def launch(self) -> None:
        self.launches += 1

    @override
    def check_alive(self) -> None:
        return None

    @override
    def await_ready(self, closing: Event) -> None:
        return None

    @override
    def close(self) -> None:
        return None

    def test_concurrent_chats_share_one_start_but_not_their_histories(self) -> None:
        barrier = Barrier(2, timeout=5)
        requests: list[list[str]] = []

        def transport(data: bytes, cancellation: Event, timeout: float,
                      redactor: DiagnosticRedactor) -> str:
            body = json.loads(data)["body"]
            texts: list[str] = [turn["content"] for turn in body["messages"]]
            requests.append(texts)
            barrier.wait()
            return texts[-1] + "-answered"

        config = ServerConfig(
            LocalCheckpoint("local", "a" * 40, Path("/tmp"), "cuda:0"), Path("/usr/bin/python3"),
        )
        server = QwenServer(config, Redaction, self, transport)
        role: PreparedRole = (server.model(), server.redactor(), "", {}, 120.0)
        target = QwenServerChat(server, initial=None, role=role)
        attacker = QwenServerChat(server, initial=None, role=role)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(target.send, TextMessage("target"))
            second = pool.submit(attacker.send, TextMessage("attacker"))
            first.result()
            second.result()
        server.close()
        assert (self.launches, sorted(requests),
                [turn.text() for turn in target.history()],
                [turn.text() for turn in attacker.history()]) == (
            1, [["attacker"], ["target"]],
            ["target", "target-answered"], ["attacker", "attacker-answered"],
        ), "Concurrent roles started extra servers or shared their histories"

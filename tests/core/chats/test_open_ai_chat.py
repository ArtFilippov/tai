"""Exercise an OpenAI-backed conversation against a local HTTP endpoint."""

import json
from queue import Queue
from socket import AF_INET, SOCK_STREAM, socket
from threading import Thread
from unittest import TestCase

from core.chats.openai import OpenAIChat, OpenAITransportError
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage


class TestOpenAIChat(TestCase):
    """The HTTP adapter commits only complete exchanges to its own dialogue."""

    def endpoint(self, responses: list[bytes]) -> tuple[str, Queue[list[dict[str, str]]]]:
        requests: Queue[list[dict[str, str]]] = Queue()
        server = socket(AF_INET, SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen()

        def serve() -> None:
            for payload in responses:
                connection, _ = server.accept()
                with connection, connection.makefile("rb") as stream:
                    stream.readline()
                    headers: dict[str, str] = {}
                    for line in iter(stream.readline, b"\r\n"):
                        key, value = line.decode().split(":", 1)
                        headers[key.lower()] = value.strip()
                    body = json.loads(stream.read(int(headers["content-length"])))
                    requests.put(body["messages"])
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(payload) + (10 if not payload else 0)}\r\n"
                        .encode() + b"Connection: close\r\n\r\n" + payload
                    )

        thread = Thread(target=serve)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.close)
        return f"http://127.0.0.1:{server.getsockname()[1]}/v1", requests

    def chat(self, url: str) -> OpenAIChat:
        return OpenAIChat(
            base_url=url, model="example", initial=TextMessage("policy"),
            redaction=Redaction, system="policy", timeout=5.0,
        )

    def fixture(self, responses: list[bytes]) -> tuple[OpenAIChat, Queue[list[dict[str, str]]]]:
        url, requests = self.endpoint(responses)
        return self.chat(url), requests

    def clean(self, requests: Queue[list[dict[str, str]]], text: str) -> bool:
        requests.get_nowait()
        return tuple(turn["content"] for turn in requests.get_nowait()) == ("policy", text)

    def reply(self) -> bytes:
        return b'{"choices":[{"message":{"role":"assistant","content":"answer"}}]}'

    def test_exchange_commits_completed_turns(self) -> None:
        chat, requests = self.fixture([self.reply()])
        result = chat.send(TextMessage("question"))
        assert (result.to_string(), requests.get_nowait(), len(chat.history())) == (
            "answer", [{"role": "system", "content": "policy"},
                       {"role": "user", "content": "question"}], 3,
        ), "OpenAI exchange lost its response or committed the wrong dialogue"

    def test_reset_sends_only_system_and_new_request(self) -> None:
        chat, requests = self.fixture([self.reply(), self.reply()])
        chat.send(TextMessage("old"))
        chat.reset()
        chat.send(TextMessage("new"))
        assert self.clean(requests, "new"), "OpenAI reset leaked the previous exchange"

    def test_truncated_transport_does_not_commit_request(self) -> None:
        chat, requests = self.fixture([b"", self.reply()])
        with self.assertRaises(OpenAITransportError):
            chat.send(TextMessage("failed"))
        chat.send(TextMessage("retry"))
        assert self.clean(requests, "retry"), "Transport failure leaked unfinished dialogue"

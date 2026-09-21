"""Controlled HTTP peer shared by transport and CLI acceptance tests."""

import json
import select
import socket
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from typing import Required, TypedDict, cast, override

import pytest


@dataclass
class _Delivery:
    hold: bool = False
    partial: bool = False
    partial_bytes: int = 5
    disconnect: bool = False


@dataclass
class _ReplyEvents:
    entered: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    disconnected: Event = field(default_factory=Event)


class Reply:
    def __init__(
        self, text: str = "Puzzle answer", status: int = 200, payload: bytes | None = None,
        hold: bool = False, partial: bool = False, partial_bytes: int = 5,
        disconnect: bool = False, truncated: bool = False, echo_authorization: bool = False,
    ) -> None:
        super().__init__()
        self.text = text
        self.status = status
        self.payload = payload
        self.truncated = truncated
        self.echo_authorization = echo_authorization
        self._delivery = _Delivery(hold, partial, partial_bytes, disconnect)
        self._events = _ReplyEvents()

    @property
    def hold(self) -> bool:
        return self._delivery.hold

    @hold.setter
    def hold(self, value: bool) -> None:
        self._delivery.hold = value

    @property
    def partial(self) -> bool:
        return self._delivery.partial

    @partial.setter
    def partial(self, value: bool) -> None:
        self._delivery.partial = value

    @property
    def partial_bytes(self) -> int:
        return self._delivery.partial_bytes

    @partial_bytes.setter
    def partial_bytes(self, value: int) -> None:
        self._delivery.partial_bytes = value

    @property
    def disconnect(self) -> bool:
        return self._delivery.disconnect

    @disconnect.setter
    def disconnect(self, value: bool) -> None:
        self._delivery.disconnect = value

    @property
    def entered(self) -> Event:
        return self._events.entered

    @property
    def release(self) -> Event:
        return self._events.release

    @property
    def disconnected(self) -> Event:
        return self._events.disconnected


class RequestBody(TypedDict, total=False):
    model: Required[str]
    messages: Required[list[dict[str, str]]]
    metadata: dict[str, object]
    stop: list[str]
    temperature: float
    max_tokens: int
    seed: int
    n: int
    stream: bool


type RequestRecord = tuple[str, dict[str, str], RequestBody]


class Endpoint:
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[RequestRecord] = []
        self.scripts: dict[str, deque[Reply]] = {
            "/v1/chat/completions": deque(
                [Reply("The silver key"), Reply("The blue door")]
            )
        }
        self.url = ""
        self.require_session = False


def _payload(reply: Reply, handler: BaseHTTPRequestHandler) -> bytes:
    if reply.payload is not None:
        return reply.payload
    text = reply.text
    if reply.echo_authorization:
        text += handler.headers["Authorization"]
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": text}}]}
    ).encode()


def _headers(
    handler: BaseHTTPRequestHandler, peer: Endpoint, reply: Reply, payload: bytes,
) -> None:
    handler.send_response(reply.status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload) + (100 if reply.truncated else 0)))
    handler.send_header("Location", peer.url + "/redirected")
    handler.end_headers()


def _released(reply: Reply, connection: socket.socket) -> bool:
    for _ in iter(lambda: reply.hold and not reply.release.is_set(), False):
        readable, _, _ = select.select([connection], [], [], 0.02)
        if readable and connection.recv(1, socket.MSG_PEEK) == b"":
            reply.disconnected.set()
            return False
    return True


def _send_reply(handler: BaseHTTPRequestHandler, peer: Endpoint, reply: Reply) -> None:
    payload = _payload(reply, handler)
    if reply.partial:
        _headers(handler, peer, reply, payload)
        handler.wfile.write(payload[: reply.partial_bytes])
        handler.wfile.flush()
    reply.entered.set()
    if not _released(reply, handler.connection):
        return
    if reply.partial:
        handler.wfile.write(payload[reply.partial_bytes :])
    else:
        _headers(handler, peer, reply, payload)
        handler.wfile.write(payload)


def _post(handler: BaseHTTPRequestHandler, peer: Endpoint, active: list[Reply]) -> None:
    body = cast(RequestBody, json.loads(handler.rfile.read(int(handler.headers["Content-Length"]))))
    peer.requests.append((handler.path, dict(handler.headers), body))
    if peer.require_session and not handler.headers.get("x-opencode-session", "").strip():
        handler.send_error(400, "MissingSessionID")
        return
    reply = peer.scripts[handler.path].popleft()
    active.append(reply)
    if reply.disconnect:
        handler.connection.shutdown(socket.SHUT_RDWR)
        handler.connection.close()
        return
    try:
        _send_reply(handler, peer, reply)
    except (BrokenPipeError, ConnectionResetError):
        reply.disconnected.set()


def _handler(peer: Endpoint, active: list[Reply]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def post(self) -> None:
            _post(self, peer, active)

        do_POST = post

        @override
        def log_message(self, *args: object, **kwargs: object) -> None:
            pass

    return Handler


@pytest.fixture
def endpoint() -> Iterator[Endpoint]:
    peer = Endpoint()
    active: list[Reply] = []

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(peer, active))
    peer.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield peer
    finally:
        for reply in active:
            reply.release.set()
        server.shutdown()
        server.server_close()
        thread.join()

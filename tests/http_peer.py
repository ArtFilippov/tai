"""Controlled HTTP peer shared by transport and CLI acceptance tests."""

import json
import select
import socket
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest


@dataclass
class Reply:
    text: str = "Puzzle answer"
    status: int = 200
    payload: bytes | None = None
    hold: bool = False
    partial: bool = False
    partial_bytes: int = 5
    disconnect: bool = False
    truncated: bool = False
    entered: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    disconnected: Event = field(default_factory=Event)


class Endpoint:
    def __init__(self):
        self.requests = []
        self.scripts = {
            "/v1/chat/completions": deque(
                [Reply("The silver key"), Reply("The blue door")]
            )
        }
        self.url = ""


@pytest.fixture
def endpoint():
    peer = Endpoint()
    active = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            peer.requests.append((self.path, dict(self.headers), body))
            reply = peer.scripts[self.path].popleft()
            active.append(reply)
            if reply.disconnect:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            payload = reply.payload
            if payload is None:
                payload = json.dumps(
                    {
                        "choices": [
                            {"message": {"role": "assistant", "content": reply.text}}
                        ]
                    }
                ).encode()

            def headers():
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    "Content-Length",
                    str(len(payload) + (100 if reply.truncated else 0)),
                )
                self.send_header("Location", peer.url + "/redirected")
                self.end_headers()

            try:
                if reply.partial:
                    headers()
                    self.wfile.write(payload[: reply.partial_bytes])
                    self.wfile.flush()
                reply.entered.set()
                while reply.hold and not reply.release.is_set():
                    readable, _, _ = select.select([self.connection], [], [], 0.02)
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        reply.disconnected.set()
                        return
                if reply.partial:
                    self.wfile.write(payload[reply.partial_bytes :])
                else:
                    headers()
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                reply.disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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

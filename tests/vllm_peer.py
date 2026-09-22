"""A managed, concurrent HTTP process with controlled startup and response gates."""

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import socket
from socketserver import BaseServer
from threading import Event
from typing import cast, override


@dataclass(frozen=True)
class Peer:
    root: Path
    model: str
    token: str


class Handler(BaseHTTPRequestHandler):
    def __init__(self, request: socket, client_address: tuple[str, int], server: BaseServer, *, peer: Peer) -> None:
        self.peer = peer
        super().__init__(request, client_address, server)

    def reply(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def get(self) -> None:
        if self.path == "/version":
            self.reply({"version": "0.6.4.post1"})
        else:
            model = "foreign" if (self.peer.root / "foreign").exists() else self.peer.model
            self.reply({"data": [{"id": model}]})

    def post(self) -> None:
        assert self.headers["Authorization"] == "Bearer " + self.peer.token
        body = cast(dict[str, object], json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        messages = cast(list[dict[str, str]], body["messages"])
        name = messages[-1]["content"]
        root = self.peer.root
        (root / (name + ".request")).write_text(json.dumps(body), encoding="utf-8")
        if (root / "hold").exists():
            await_release(root / (name + ".release"))
        self.reply(response(root, name))

    do_GET = get
    do_POST = post

    @override
    def log_message(self, *args: object, **kwargs: object) -> None:
        pass


def response(root: Path, name: str) -> dict[str, object]:
    if (root / "invalid").exists():
        return {"bad": True}
    return {"choices": [{"message": {"role": "assistant", "content": name + " reply"}}]}


def await_release(path: Path) -> None:
    for _ in iter(path.exists, True):
        _ = Event().wait(0.01)


def serve(root: Path, model: str, token: str, port: int) -> None:
    with ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, peer=Peer(root, model, token))) as server:
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("model", "served-model-name", "api-key", "port"):
        _ = parser.add_argument("--" + name, required=True)
    args, _ = parser.parse_known_args()
    root = Path(args.model)
    with (root / "launches").open("a", encoding="utf-8") as log:
        _ = log.write(str(os.getpid()) + "\n")
    if (root / "exit").exists():
        sys.exit(17)
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as child:
        _ = (root / "child").write_text(str(child.pid), encoding="utf-8")
        _ = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if (root / "startup_hold").exists():
            _ = Event().wait(60)
        serve(root, args.served_model_name, args.api_key, int(args.port))


if __name__ == "__main__":
    main()

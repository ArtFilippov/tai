"""Live local HTTP fixtures for GGUF server tests."""

import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from target.gguf_server_utils import listening


def spawn(port: int, identity: str, mode: str) -> subprocess.Popen[bytes]:
    """Launch an independent HTTP service for a test.

    Args:
        port (int): Selected port.
        identity (str): Advertised identity.
        mode (str): Response variant.

    Returns:
        subprocess.Popen[bytes]: Child process serving real sockets.
    """
    script = '''
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if sys.argv[3] == "unusable":
            self.send_response(503)
            self.end_headers()
            return
        body = {"data": [{"id": sys.argv[2]}]} if sys.argv[3] != "malformed" else {"data": "bad"}
        self.respond(body)

    def do_POST(self):
        if sys.argv[3] == "error":
            self.send_response(503)
            self.end_headers()
            return
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = {"choices": [{"message": {"content": data["messages"][-1]["content"]}}]}
        if sys.argv[3] == "bad_reply":
            body = {"choices": []}
        self.respond(body)

    def respond(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''
    return subprocess.Popen(
        [
            sys.executable, "-c", script,
            str(port),
            identity, mode
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def failed_start(host: str, port: int, path: Path, gpu: int, *, expected: str) -> subprocess.Popen[bytes]:
    """Create a process that exits before becoming ready.

    Args:
        host (str): Requested host.
        port (int): Requested port.
        path (Path): Requested model path.
        gpu (int): Requested GPU.
        expected (str): Requested model identity.

    Returns:
        subprocess.Popen[bytes]: Process that exits immediately.
    """
    return subprocess.Popen(
        [sys.executable, "-c", "raise RuntimeError('startup failed')"],
        stderr=subprocess.DEVNULL
    )


@contextmanager
def service(port: int, identity: str, mode: str) -> Generator[subprocess.Popen[bytes]]:
    """Keep a local service alive for one test and always clean it up.

    Args:
        port (int): Selected port.
        identity (str): Advertised model identity.
        mode (str): Response variant.

    Yields:
        subprocess.Popen[bytes]: Live service process.
    """
    process = spawn(
        port, identity, mode
    )
    try:
        wait(port)
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            _ = process.wait(timeout=5)


def wait(port: int) -> None:
    """Wait briefly for the child HTTP service.

    Args:
        port (int): Child service port.

    Raises:
        RuntimeError: If the service is unavailable after startup.
    """
    for _ in range(50):
        if listening(
            "127.0.0.1", port
        ):
            return
        time.sleep(0.05)
    raise RuntimeError("The test GGUF service did not start")


@contextmanager
def owned(processes: list[subprocess.Popen[bytes]]) -> Generator[None]:
    """Substitute a real lightweight HTTP child for the CUDA model process.

    Args:
        processes (list[subprocess.Popen[bytes]]): Child processes created by the run.

    Yields:
        None: Patched startup scope.
    """
    def launch(host: str, port: int, path: Path, gpu: int, *, expected: str) -> subprocess.Popen[bytes]:
        process = spawn(
            port, expected, "normal"
        )
        processes.append(process)
        return process

    with patch("target.gguf_server_utils.gpu_support"), patch(
        "target.gguf_server_utils.start", side_effect=launch
    ):
        yield

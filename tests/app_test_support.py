"""Typed HTTP fixture contract and subprocess cleanup shared by app tests."""

import subprocess
from collections import deque
from typing import Protocol, TypedDict

from http_peer import Reply


class ChatRequest(TypedDict):
    messages: list[dict[str, str]]
    model: str
    temperature: float
    n: int
    stream: bool


type WireRequest = tuple[str, dict[str, str], ChatRequest]


class HttpPeer(Protocol):
    requests: list[WireRequest]
    scripts: dict[str, deque[Reply]]
    url: str
    require_session: bool


def session(request: WireRequest) -> str | None:
    return {key.lower(): value for key, value in request[1].items()}.get("x-opencode-session")


def cleanup_process(process: subprocess.Popen[str], held: Reply) -> None:
    if process.poll() is None:
        process.kill()
        process.wait()
    held.release.set()

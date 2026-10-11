"""Encode one complete dialogue for the Chat Completions HTTP worker."""

import json
from importlib.metadata import version

from .chat_turn_contract import ChatTurn


def request(
    configuration: tuple[str, str | None, str, float, bool], model: str,
    identity: str, turns: tuple[ChatTurn, ...],
) -> bytes:
    url, authorization, generation, timeout, session = configuration
    headers = {
        "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": f"tai/{version('tai')}",
    }
    if authorization is not None:
        headers["Authorization"] = authorization
    if session:
        headers["x-opencode-session"] = identity
    return json.dumps({
        "url": url + "/chat/completions", "headers": headers, "timeout": timeout,
        "body": {
            **json.loads(generation), "model": model,
            "messages": [{"role": turn.role, "content": turn.text()} for turn in turns],
            "stream": False, "n": 1,
        },
    }).encode()

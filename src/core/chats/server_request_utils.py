"""Encode a complete, authenticated conversation for the local vLLM endpoint."""

import json

from .chat_turn_contract import ChatTurn


def request(
    turns: tuple[ChatTurn, ...], parameters: dict[str, object], timeout: float,
    identity: tuple[str, str, str],
) -> bytes:
    url, model, token = identity
    return json.dumps({
        "url": url + "/v1/chat/completions", "timeout": timeout,
        "headers": {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        "body": {
            **parameters, "model": model,
            "messages": [{"role": turn.role, "content": turn.text()} for turn in turns],
        },
    }).encode()

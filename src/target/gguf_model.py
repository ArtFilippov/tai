"""History-free chat completions from a pinned GGUF server."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override
from urllib.request import Request, urlopen

from target.model import Model


@dataclass(eq=False)
class GGUFModel(Model):
    """Send each complete Attacker conversation to one verified model identity.

    Attributes:
        address (str): HTTP address of the verified local server.
        tokens (int): Generation limit for each response.
        identity (str): Expected server model identity.
        commit (str): Pinned repository revision.
    """

    address: str
    tokens: int
    identity: str
    commit: str

    def __post_init__(self) -> None:
        if self.tokens < 1:
            raise ValueError("Token limit must be positive")

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        try:
            payload = self.completion(messages)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"GGUF completion failed at {self.address}: {exc}") from exc
        match payload:
            case {"choices": [{"message": {"content": str() as text}}]}:
                return text
            case _:
                raise ValueError(f"Malformed GGUF completion at {self.address}")

    @override
    def revision(self) -> str:
        return self.commit

    def completion(self, messages: Sequence[dict[str, str]]) -> object:
        """Read one JSON completion from the selected HTTP endpoint.

        Args:
            messages (Sequence[dict[str, str]]): Full conversation.

        Returns:
            object: Decoded response.
        """
        request = Request(
            f"{self.address}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self.identity,
                    "messages": list(messages),
                    "max_tokens": self.tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "min_p": 0,
                    "repeat_penalty": 1,
                    "stream": False
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urlopen(
            request, timeout=120
        ) as response:
            return json.load(response)

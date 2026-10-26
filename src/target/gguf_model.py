"""History-free chat completions from a pinned GGUF server."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override
from urllib.request import Request, urlopen

from target.gguf_asset import GGUFAsset
from target.gguf_service import GGUFService
from target.model import Model


@dataclass(eq=False)
class GGUFModel(Model):
    """Send each complete Attacker conversation to one verified model identity.

    Attributes:
        file (GGUFAsset): Pinned model identity and revision.
        service (GGUFService): Address and effective runtime settings.
        tokens (int): Generation limit for each response.
    """

    file: GGUFAsset
    service: GGUFService
    tokens: int

    def __post_init__(self) -> None:
        if self.tokens < 1:
            raise ValueError("Token limit must be positive")

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        try:
            payload = self.completion(messages)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"GGUF completion failed at {self.service.address()}: {exc}") from exc
        match payload:
            case {"choices": [{"message": {"content": str() as text}}]}:
                return text
            case _:
                raise ValueError(f"Malformed GGUF completion at {self.service.address()}")

    @override
    def revision(self) -> str:
        return self.file.commit()

    @override
    def settings(self) -> dict[str, object]:
        return {
            "id": self.file.identity(),
            "revision": self.revision(),
            "generation": self.policy(),
            **self.file.details(),
            **self.service.settings()
        }

    def policy(self) -> dict[str, int]:
        """Describe the generation options sent to the GGUF service.

        Returns:
            dict[str, int]: Request generation options.
        """
        return {
            "max_tokens": self.tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "min_p": 0,
            "repeat_penalty": 1
        }

    def completion(self, messages: Sequence[dict[str, str]]) -> object:
        """Read one JSON completion from the selected HTTP endpoint.

        Args:
            messages (Sequence[dict[str, str]]): Full conversation.

        Returns:
            object: Decoded response.
        """
        request = Request(
            f"{self.service.address()}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self.file.identity(),
                    "messages": list(messages),
                    **self.policy(),
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

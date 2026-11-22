"""History-free chat completions from a pinned GGUF server."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import override
from urllib.request import Request, urlopen

from target.gguf.gguf_asset import GGUFAsset
from target.gguf.gguf_service import GGUFService
from target.gguf_chat import GGUFChat
from target.sampling import Sampling


@dataclass(eq=False)
class GGUFModel(GGUFChat):
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
        return self.response(payload)

    @override
    def sample(self, messages: Sequence[dict[str, str]], sampling: Sampling) -> str:
        options = sampling.options()
        try:
            payload = self.request(
                messages,
                {
                    "max_tokens": options["max_new_tokens"],
                    "temperature": options["temperature"],
                    "top_p": options["top_p"],
                    "top_k": options["top_k"],
                    "min_p": 0,
                    "repeat_penalty": 1,
                    "seed": sampling.draw()
                }
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"GGUF completion failed at {self.service.address()}: {exc}") from exc
        return self.response(payload)

    @override
    def decoding(self, sampling: Sampling) -> dict[str, object]:
        return {
            **sampling.description(),
            "max_tokens": sampling.options()["max_new_tokens"],
            "min_p": 0,
            "repeat_penalty": 1
        }

    @override
    def response(self, payload: object) -> str:
        """Require exactly one complete assistant reply.

        Args:
            payload (object): Decoded completion response.

        Returns:
            str: Full unmodified assistant text.

        Raises:
            ValueError: If the completion is malformed.
        """
        match payload:
            case {"choices": [{"message": {"content": str() as text}}]}:
                return text
            case _:
                raise ValueError(f"Malformed GGUF completion at {self.service.address()}")

    @override
    def prefill(self, messages: Sequence[dict[str, str]], prefix: str) -> str:
        raise NotImplementedError("GGUF chat completions do not support verified assistant prefill")

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

    @override
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

    @override
    def completion(self, messages: Sequence[dict[str, str]]) -> object:
        """Read one JSON completion from the selected HTTP endpoint.

        Args:
            messages (Sequence[dict[str, str]]): Full conversation.

        Returns:
            object: Decoded response.
        """
        return self.request(
            messages,
            self.policy()
        )

    @override
    def request(self, messages: Sequence[dict[str, str]], policy: Mapping[str, int | float]) -> object:
        """Send explicit per-request decoding options without mutating the service.

        Args:
            messages (Sequence[dict[str, str]]): Complete conversation.
            policy (Mapping[str, int | float]): Native generation options.

        Returns:
            object: Decoded completion response.
        """
        request = Request(
            f"{self.service.address()}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self.file.identity(),
                    "messages": list(messages),
                    **policy,
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

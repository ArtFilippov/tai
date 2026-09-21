"""Direct, synchronous OpenAI-compatible Chat Completions conversations."""

import json
from importlib.metadata import version
from threading import Event
from typing import Unpack, override

from core.primitives.diagnostics import DiagnosticObject, DiagnosticRedactor

from .conversation import ChatCancelled, ChatTurn, LLMChat
from .openai_config import OpenAIOptions, connection_config, validate_text
from .openai_errors import (
    OpenAIError, OpenAIHTTPError, OpenAIAuthenticationError, OpenAIRateLimitError,
    OpenAITransportError, OpenAITimeoutError, OpenAIResponseError, OpenAIJSONError,
    OpenAIContentError,
)
from .openai_transport import completion, exchange

__all__ = [
    "OpenAIChat", "OpenAIError", "OpenAIHTTPError", "OpenAIAuthenticationError",
    "OpenAIRateLimitError", "OpenAITransportError", "OpenAITimeoutError",
    "OpenAIResponseError", "OpenAIJSONError", "OpenAIContentError",
]


class OpenAIChat(LLMChat):
    """One independently configured role and its completed conversation.

    ``base_url`` is the API root (for example, ``https://host/v1``).
    Authorization is explicit: ``api_key`` becomes a Bearer header, while
    ``authorization`` is a complete Authorization header value. No credentials,
    proxies, model configuration, or agent session are discovered implicitly.
    ``timeout`` bounds the entire exchange, including connection and body reads.

    Args:
        base_url (str): HTTP(S) API root.
        model (str): Explicit model identifier.
        **options (Unpack[OpenAIOptions]): Authorization, system, generation,
            timeout and opt-in OpenCode session configuration.
    """

    def __init__(
        self, *, base_url: str, model: str, **options: Unpack[OpenAIOptions],
    ) -> None:
        config = connection_config(base_url, options)
        super().__init__(
            lambda turns: "", model=validate_text(model, "model", nonempty=True),
            system=validate_text(options.get("system", ""), "system"),
        )
        self._config = config
        self._redactor = config.authorization.redactor

    @override
    def settings(self) -> DiagnosticObject:
        return self._diagnostic_settings(self._redactor)

    @override
    def _diagnostic_settings(self, redactor: DiagnosticRedactor) -> DiagnosticObject:
        # A run supplies all roles' credentials before any replacement, ensuring
        # overlapping credentials and resulting key collisions share one policy.
        settings = {
            **super()._settings(),
            "backend": "openai",
            "transport": "openai_chat_completions",
            "base_url": self._config.base_url,
            "authorization": self._config.authorization.mode,
            "timeout": self._config.timeout,
            "opencode_session": self._config.session,
            "generation": redactor.snapshot(json.loads(self._config.generation), mask_fields=True),
        }
        return redactor.snapshot(settings)

    def _request(self, turns: tuple[ChatTurn, ...]) -> bytes:
        headers = {
            "Content-Type": "application/json", "Accept": "application/json",
            "User-Agent": f"tai/{version('tai')}",
        }
        if self._config.authorization.header is not None:
            headers["Authorization"] = self._config.authorization.header
        if self._config.session:
            headers["x-opencode-session"] = self.conversation_id
        return json.dumps({
            "url": self._config.base_url + "/chat/completions",
            "headers": headers, "timeout": self._config.timeout,
            "body": {
                **json.loads(self._config.generation), "model": self.model,
                "messages": [{"role": turn.role, "content": turn.text()} for turn in turns],
                "stream": False, "n": 1,
            },
        }).encode()

    @override
    def _generate(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        if cancellation.is_set():
            raise ChatCancelled()
        result = exchange(self._request(turns), cancellation, self._config.timeout)
        return completion(result, self._redactor)

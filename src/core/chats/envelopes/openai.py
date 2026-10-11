"""Direct, synchronous OpenAI-compatible Chat Completions conversations."""

from threading import Event
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor, RedactionFactory
from core.primitives.messages import Message

from .cancelled import Cancelled
from ..chat_turn_contract import ChatTurn
from .conversation import LLMChat
from ..openai_config_utils import validate_text
from ..openai_request_utils import request as encode_request
from ..openai_settings_utils import settings as diagnostic_snapshot
from ..open_ai_error import OpenAIError
from ..open_ai_http_error import OpenAIHTTPError
from ..open_ai_authentication_error import OpenAIAuthenticationError
from ..open_ai_rate_limit_error import OpenAIRateLimitError
from ..open_ai_transport_error import OpenAITransportError
from ..open_ai_timeout_error import OpenAITimeoutError
from ..open_ai_response_error import OpenAIResponseError
from ..open_ai_json_error import OpenAIJSONError
from ..open_ai_content_error import OpenAIContentError
from .openai_transport_utils import completion, exchange
from .openai_policy import OpenAIPolicy

__all__ = [
    "OpenAIChat", "OpenAIError", "OpenAIHTTPError", "OpenAIAuthenticationError",
    "OpenAIRateLimitError", "OpenAITransportError", "OpenAITimeoutError",
    "OpenAIResponseError", "OpenAIJSONError", "OpenAIContentError",
]
SYSTEM_OPTION = "system"


class OpenAIChat(LLMChat):
    """One independently configured role and its completed conversation.

    ``base_url`` is the API root (for example, ``https://host/v1``).
    Authorization is explicit: ``api_key`` becomes a Bearer header, while
    ``authorization`` is a complete Authorization header value. No credentials,
    proxies, model configuration, or agent session are discovered implicitly.
    ``timeout`` bounds the entire exchange, including connection and body reads.
    Connection options are checked at the first settings snapshot or send,
    before any HTTP request; the credential policy is prepared once per chat.

    Args:
        base_url (str): HTTP(S) API root.
        model (str): Explicit model identifier.
        initial (Message | None): Prepared system turn, when present.
        redaction (RedactionFactory): Credential-aware diagnostic policy factory.
        **options (object): Authorization, system, generation, timeout and
            opt-in OpenCode session configuration.

    Raises:
        ValueError: System instructions are invalid or lack an initial Message.
    """

    def __init__(
        self, *, base_url: str, model: str, initial: Message | None,
        redaction: RedactionFactory, **options: object,
    ) -> None:
        system = options[SYSTEM_OPTION] if SYSTEM_OPTION in options else ""
        if isinstance(system, str):
            self.credentials = OpenAIPolicy(base_url, options, redaction)
            super().__init__(
                lambda turns: "", initial=initial, model=model,
                system=system, redactor=self.credentials,
            )
        else:
            raise ValueError("system must be a string")

    @override
    def settings(self) -> dict[str, object] | list[object]:
        return self.diagnostic_settings(self.redactor)

    @override
    def diagnostic_settings(
        self, redactor: DiagnosticRedactor,
    ) -> dict[str, object] | list[object]:
        # A run supplies all roles' credentials before any replacement, ensuring
        # overlapping credentials and resulting key collisions share one policy.
        return diagnostic_snapshot(
            super().settings_data(), self.credentials.configuration(), redactor,
        )

    def request(self, turns: tuple[ChatTurn, ...]) -> bytes:
        config = self.credentials.configuration()
        return encode_request(
            (config[0], config[1], config[4], config[5], config[6]),
            validate_text(self.model, "model", nonempty=True), self.conversation_id, turns,
        )

    @override
    def __call__(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        if cancellation.is_set():
            raise Cancelled()
        result = exchange(
            self.request(turns), cancellation, self.credentials.configuration()[5],
        )
        return completion(result, self.redactor)

"""One credential-aware policy and validated configuration per OpenAI chat."""

from collections.abc import Mapping
from threading import Lock
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor, RedactionFactory

from ..gate_lock import GateLock
from ..openai_config_utils import connection_config

type Configuration = tuple[str, str | None, str, DiagnosticRedactor, str, float, bool]


class OpenAIPolicy(DiagnosticRedactor):
    """Prepare one policy and configuration before diagnostic or HTTP use.

    Args:
        url (str): Explicit API root.
        options (Mapping[str, object]): Explicit authorization and request options.
        factory (RedactionFactory): Policy factory for this chat's credentials.
    """

    def __init__(
        self, url: str, options: Mapping[str, object], factory: RedactionFactory,
    ) -> None:
        super().__init__()
        self.url, self.options = url, dict(options)
        self.factory = factory
        self.cache: list[
            tuple[str, str | None, str, DiagnosticRedactor, str, float, bool]
        ] = []
        self.lock: GateLock = Lock()

    def configuration(self) -> Configuration:
        with self.lock:
            if not self.cache:
                self.cache.append(connection_config(self.url, self.options, self.factory))
            return self.cache[0]

    def active(self) -> DiagnosticRedactor:
        return self.configuration()[3]

    @override
    def secrets(self) -> tuple[str, ...]:
        return self.active().secrets()

    @override
    def combine(self, *others: DiagnosticRedactor) -> DiagnosticRedactor:
        return self.active().combine(*others)

    @override
    def text(self, value: str) -> str:
        return self.active().text(value)

    @override
    def snapshot(self, value: object, *, mask_fields: bool = False) -> object:
        return self.active().snapshot(value, mask_fields=mask_fields)

    @override
    def structured(
        self, value: object, *, mask_fields: bool = False,
    ) -> dict[str, object] | list[object]:
        return self.active().structured(value, mask_fields=mask_fields)

    @override
    def decode(self, text: str) -> dict[str, object] | list[object]:
        return self.active().decode(text)

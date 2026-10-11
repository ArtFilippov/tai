"""Detached JSON diagnostics; never applied to live Messages or wire requests."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from functools import cached_property
from typing import TypeGuard, override

from .diagnostic_redactor import DiagnosticRedactor

CREDENTIAL_FIELDS = frozenset((
    "apikey", "authorization", "accesstoken", "password", "secret", "token",
))


class Redaction(DiagnosticRedactor):
    """Mask secret spellings in JSON-shaped diagnostics, never in live Messages.

    Args:
        credentials (Sequence[str]): Credentials supplied by assembly.
    """

    def __init__(self, credentials: Sequence[str]) -> None:
        super().__init__()
        self.credentials: Sequence[str] = tuple(credentials)

    @cached_property
    def pattern(self) -> re.Pattern[str] | None:
        # Model-evaluator Messages contain JSON *inside* their text. Normalizing
        # the outer snapshot does not unescape that payload. Match the canonical
        # JSON-string spellings (including ASCII Unicode escapes) as well, in
        # response origins/history and chat generation settings.
        # Keep original secrets for combine(), so composing policies never adds
        # more encoding layers. Every spelling shares the same one-pass rules.
        encoded = {
            json.dumps(secret, ensure_ascii=ascii_only)[1:-1]
            for secret in self.secrets() for ascii_only in (False, True)
        }
        spellings = sorted(set(self.secrets()) | encoded, key=lambda s: (-len(s), s))
        return (
            re.compile("|".join(re.escape(secret) for secret in spellings))
            if spellings
            else None
        )

    @override
    def combine(self, *others: DiagnosticRedactor) -> DiagnosticRedactor:
        return Redaction(tuple(self.credentials) + tuple(
            secret for redactor in others for secret in redactor.secrets()
        ))

    @override
    def secrets(self) -> tuple[str, ...]:
        return tuple(sorted(
            {secret for secret in self.credentials if secret},
            key=lambda secret: (-len(secret), secret),
        ))

    @override
    def text(self, value: str) -> str:
        # Match longest first, in one pass, so one credential cannot partially
        # replace another or rewrite a marker inserted by that same pass.
        if not self.credentials or (pattern := self.pattern) is None:
            return value
        redacted = pattern.sub("[REDACTED]", value)
        # A credential may itself be part of the marker, or be recreated at a
        # replacement boundary. Suppress this diagnostic string rather than
        # exposing it or repeatedly substituting potentially growing markers.
        return "" if pattern.search(redacted) else redacted

    @override
    def snapshot(self, value: object, *, mask_fields: bool = False) -> object:
        """Copy JSON values; a colliding object becomes ordered [key, value] pairs.

        Keeping every pair avoids overwriting either a redacted key or an
        existing literal marker. Objects without collisions retain their shape.
        Credential-named fields are masked only in chat configuration snapshots.

        Args:
            value (object): JSON-serializable diagnostic input.
            mask_fields (bool, default=False): Whether to mask credential-named fields.

        Returns:
            object: A detached, redacted JSON value.
        """
        # Match JSON's accepted containers and key coercion before traversal,
        # including tuples and numeric keys from custom participant settings.
        return self.copy(json.loads(json.dumps(value, ensure_ascii=False)), mask_fields)

    @override
    def structured(
        self, value: object, *, mask_fields: bool = False,
    ) -> dict[str, object] | list[object]:
        def shaped(value: object) -> TypeGuard[dict[str, object] | list[object]]:
            return isinstance(value, (dict, list))

        snapshot = self.snapshot(value, mask_fields=mask_fields)
        if not shaped(snapshot):
            raise ValueError("Expected a JSON object or diagnostic key/value pairs")
        return snapshot

    @override
    def decode(self, text: str) -> dict[str, object] | list[object]:
        return self.structured(json.loads(text))

    def copy(self, value: object, mask_fields: bool) -> object:
        def mapping(value: object) -> TypeGuard[dict[str, object]]:
            return isinstance(value, dict)

        def sequence(value: object) -> TypeGuard[list[object]]:
            return isinstance(value, list)

        if mapping(value):
            return self.pairs(value, mask_fields)
        if sequence(value):
            return [self.copy(item, mask_fields) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value

    def pairs(
        self, value: dict[str, object], mask_fields: bool,
    ) -> dict[str, object] | list[object]:
        pairs = [
            (
                self.text(key),
                self.text("[REDACTED]")
                if mask_fields
                and key.lower().replace("_", "").replace("-", "") in CREDENTIAL_FIELDS
                else self.copy(item, mask_fields),
            )
            for key, item in value.items()
        ]
        if len({key for key, _ in pairs}) != len(pairs):
            return [[key, item] for key, item in pairs]
        return dict(pairs)

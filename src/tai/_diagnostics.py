"""Detached JSON diagnostics; never applied to live Messages or wire requests."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

type DiagnosticObject = dict[str, Any] | list[list[Any]]

_CREDENTIAL_FIELDS = {
    "apikey",
    "authorization",
    "accesstoken",
    "password",
    "secret",
    "token",
}


class DiagnosticRedactor:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = tuple(
            sorted(set(filter(None, secrets)), key=lambda s: (-len(s), s))
        )
        self._pattern = (
            re.compile("|".join(re.escape(secret) for secret in self._secrets))
            if self._secrets
            else None
        )

    @classmethod
    def combine(cls, *redactors: DiagnosticRedactor) -> DiagnosticRedactor:
        return cls(secret for redactor in redactors for secret in redactor._secrets)

    def text(self, value: str) -> str:
        # Match longest first, in one pass, so one credential cannot partially
        # replace another or rewrite a marker inserted by that same pass.
        if self._pattern is None:
            return value
        redacted = self._pattern.sub(lambda match: "[REDACTED]", value)
        # A credential may itself be part of the marker, or be recreated at a
        # replacement boundary. Suppress this diagnostic string rather than
        # exposing it or repeatedly substituting potentially growing markers.
        return "" if self._pattern.search(redacted) else redacted

    def snapshot(self, value: Any, *, mask_fields: bool = False) -> Any:
        """Copy JSON values; a colliding object becomes ordered [key, value] pairs.

        Keeping every pair avoids overwriting either a redacted key or an
        existing literal marker. Objects without collisions retain their shape.
        Credential-named fields are masked only in chat configuration snapshots.
        """
        # Match JSON's accepted containers and key coercion before traversal,
        # including tuples and numeric keys from custom participant settings.
        return self._copy(
            json.loads(json.dumps(value, ensure_ascii=False)), mask_fields
        )

    def _copy(self, value: Any, mask_fields: bool) -> Any:
        if isinstance(value, dict):
            pairs = [
                [
                    self.text(key),
                    self.text("[REDACTED]")
                    if mask_fields
                    and key.lower().replace("_", "").replace("-", "")
                    in _CREDENTIAL_FIELDS
                    else self._copy(item, mask_fields),
                ]
                for key, item in value.items()
            ]
            if len({key for key, _ in pairs}) != len(pairs):
                return pairs
            return dict(pairs)
        if isinstance(value, list):
            return [self._copy(item, mask_fields) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value

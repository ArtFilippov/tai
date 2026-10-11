"""Detach reproducible HTTP chat settings using a shared redaction policy."""

import json

from core.primitives.diagnostics import DiagnosticRedactor


def settings(
    dialogue: dict[str, object],
    configuration: tuple[str, str | None, str, DiagnosticRedactor, str, float, bool],
    redactor: DiagnosticRedactor,
) -> dict[str, object] | list[object]:
    keys = ("base_url", "authorization", "timeout", "opencode_session")
    values = (configuration[0], configuration[2], configuration[5], configuration[6])
    snapshot = {
        **dialogue, **dict(zip(keys, values, strict=True)),
        "backend": "openai", "transport": "openai_chat_completions",
        "generation": redactor.structured(json.loads(configuration[4]), mask_fields=True),
    }
    return redactor.structured(snapshot)

"""Detached readiness failure values for user-supplied operations."""


def error_data(error: Exception | None) -> dict[str, object]:
    return {
        "ok": False, "error": f"{type(error).__name__}: {error}",
        "diagnostic": getattr(error, "diagnostic", None),
    }

"""Public error for startup, ownership, or liveness failure of a Qwen server."""

from typing import override

from ..qwen_error import QwenError


class QwenServerError(QwenError):
    """Classify a managed server failure without losing its diagnostic message."""

    @override
    def category(self) -> str:
        return "server"

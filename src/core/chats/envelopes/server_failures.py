"""Translate managed-process failures into public Qwen server errors."""

from typing import NoReturn, override

from ..server_failure_port import ServerFailurePort
from .qwen_server_error import QwenServerError


class ServerFailures(ServerFailurePort):
    """Construct a server-scoped error while retaining the process's explanation."""

    @override
    def reject(self, message: str) -> NoReturn:
        raise QwenServerError(message)

    @override
    def require(self, valid: bool, message: str) -> None:
        if not valid:
            self.reject(message)

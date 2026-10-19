"""Contract for preserving server-specific failures across the process boundary."""

from typing import NoReturn, Protocol


class ServerFailurePort(Protocol):
    """Describe a managed-server failure without depending on its implementation."""

    def reject(self, message: str) -> NoReturn: ...

    def require(self, valid: bool, message: str) -> None: ...

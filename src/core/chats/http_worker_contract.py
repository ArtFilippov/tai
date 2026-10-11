"""The runnable contract of one disposable HTTP exchange."""

from typing import Protocol


class HTTPWorkerContract(Protocol):
    """Perform one request using the transport chosen by the assembly point."""

    def run(self) -> None: ...

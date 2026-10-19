"""Validated resource limits visible to a managed vLLM process."""

from typing import Protocol


class ServerLimitsPort(Protocol):
    """Limit concurrent generations, memory, and process lifetime."""

    def flags(self) -> list[str]: ...

    def deadlines(self) -> tuple[float, float]: ...

    def settings(self) -> dict[str, object]: ...

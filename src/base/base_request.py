"""The boundary between a base request and an attack."""

from collections.abc import Mapping
from typing import Protocol


class BaseRequest(Protocol):
    """An original request and its auditable origin."""

    def text(self) -> str:
        ...

    def metadata(self) -> Mapping[str, str | int]:
        ...

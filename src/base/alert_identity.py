"""The identity and taxonomy required by a series of ALERT requests."""

from typing import Protocol

from base.base_request import BaseRequest


class AlertIdentity(BaseRequest, Protocol):
    """A source request with an ordered identity and canonical categories."""

    @property
    def identity(self) -> int: ...

    @property
    def microcategory(self) -> str: ...

    @property
    def macro(self) -> str: ...

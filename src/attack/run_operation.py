"""Contract for composing one FITD run against reusable models."""

from pathlib import Path
from typing import Protocol

from base.base_request import BaseRequest
from target.model import Model


class RunOperation(Protocol):
    """Compose fresh roles and a journal for a selected base."""

    def __call__(self, base: BaseRequest, path: Path, model: Model, attacker: Model, *, turns: int) -> None: ...

"""Immutable textual result with independent transformation and origin histories."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Concatenate, Protocol, runtime_checkable


@runtime_checkable
class Message(Protocol):
    """Immutable text with its own identity, history, and origin.

    Attributes:
        text (str): Computed text of this version; callers never mutate it.
    """

    text: str

    def __init__(self, text: str) -> None:
        """Establish the text supplied by a concrete message.

        Args:
            text (str): Computed text.
        """
        super().__init__()
        self.text = text

    @property
    def id(self) -> str: ...

    @property
    def previous(self) -> Message | None: ...

    @property
    def origin(self) -> Message | None: ...

    def versions(self) -> Generator[Message]: ...

    def base(self) -> str: ...

    def to_string(self) -> str: ...

    def transform[**P](
        self, function: Callable[Concatenate[str, P], str], /, *args: P.args, **kwargs: P.kwargs,
    ) -> Message: ...

    def revert(self, n: int) -> Message: ...

    def spawn(self, payload: str) -> Message: ...

    def dump(self) -> str: ...

    def snapshot(self) -> dict[str, object]: ...

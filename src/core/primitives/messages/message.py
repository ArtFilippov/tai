"""Immutable text shared by conversations, strategies, and evaluators."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Concatenate
from uuid import uuid4

def _new_id() -> str:
    return str(uuid4())


@dataclass(frozen=True, slots=True)
class Message:
    """One immutable text version with separate transformation and origin links.

    Attributes:
        text (str): Current computed text.
        id (str, default=field(default_factory=_new_id, init=False)):
            Unique identity of this version.
        _previous (Message | None, default=field(
            default=None, init=False, repr=False, compare=False)):
            Previous transformation version.
        _origin (Message | None, default=field(
            default=None, init=False, repr=False, compare=False)):
            Source of a spawned root.
        _depth (int, default=field(default=0, init=False, repr=False, compare=False)):
            Number of transformations since the root.
    """

    text: str
    id: str = field(default_factory=_new_id, init=False)
    _previous: Message | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _origin: Message | None = field(default=None, init=False, repr=False, compare=False)
    _depth: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        def validate(value: object) -> None:
            if not isinstance(value, str):
                raise TypeError("Message text must be a string")

        validate(self.text)

    @property
    def previous(self) -> Message | None:
        return self._previous

    @property
    def origin(self) -> Message | None:
        return self._origin

    def versions(self) -> Generator[Message]:
        version = self
        yield version
        for _ in range(self._depth):
            previous = version.previous
            assert previous is not None
            version = previous
            yield version

    def base(self) -> str:
        return self.revert(self._depth).text

    def to_string(self) -> str:
        return self.text

    def transform[**P](
        self,
        function: Callable[Concatenate[str, P], str],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Message:
        """Apply a function once, retaining this immutable version as its parent.

        Args:
            function (Callable[Concatenate[str, P], str]): Text transformation.
            *args (P.args): Positional transformation arguments.
            **kwargs (P.kwargs): Keyword transformation arguments.

        Returns:
            Message: New version linked to this one.
        """
        transformed = Message(function(self.text, *args, **kwargs))
        object.__setattr__(transformed, "_previous", self)
        object.__setattr__(transformed, "_depth", self._depth + 1)
        return transformed

    def revert(self, n: int) -> Message:
        """Return an existing version, without rerunning any transformation.

        Args:
            n (int): Number of transformations to undo.

        Returns:
            Message: The selected immutable version.

        Raises:
            ValueError: The requested distance is outside this chain.
        """
        if n < 0:
            raise ValueError("Cannot revert a negative number of transformations")
        version = self
        for _ in range(n):
            if (previous := version.previous) is None:
                raise ValueError("Cannot revert beyond the transformation history")
            version = previous
        return version

    def spawn(self, payload: str) -> Message:
        """Start a new text root whose provenance is this exact version.

        Transformation rollback stops at the new root, not its origin.

        Args:
            payload (str): Text of the independent new root.

        Returns:
            Message: A root retaining this version as provenance.
        """
        spawned = Message(payload)
        object.__setattr__(spawned, "_origin", self)
        return spawned

    def dump(self) -> str:
        """Serialize computed text versions, never executable functions.

        ``origin`` is the spawning version's full snapshot (or null).
        ``transformations`` lists this chain's computed versions oldest first;
        each entry records its id, source_id, and resulting text. The first
        source_id identifies the root, whose text is ``base``.

        Returns:
            str: JSON representation of this computed history.
        """
        return json.dumps(self.snapshot(), ensure_ascii=False)

    def snapshot(self) -> dict[str, object]:
        versions = tuple(self.versions())
        root = versions[-1]
        transformations = [
            {"id": version.id, "source_id": previous.id, "text": version.text}
            for version, previous in zip(versions, versions[1:])
        ]
        return {
            "id": self.id,
            "base": root.text,
            "text": self.text,
            "origin": root.origin.snapshot() if root.origin is not None else None,
            "transformations": list(reversed(transformations)),
        }

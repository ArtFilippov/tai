"""Immutable text shared by conversations, strategies, and evaluators."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Concatenate
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Message:
    """One immutable text version, with separate transformation and origin links."""

    text: str
    id: str = field(default_factory=lambda: str(uuid4()), init=False)
    _previous: Message | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _origin: Message | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Message text must be a string")

    def base(self) -> str:
        version = self
        while version._previous is not None:
            version = version._previous
        return version.text

    def to_string(self) -> str:
        return self.text

    def transform[**P](
        self,
        function: Callable[Concatenate[str, P], str],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Message:
        """Apply a function once, retaining this immutable version as its parent."""
        transformed = Message(function(self.text, *args, **kwargs))
        object.__setattr__(transformed, "_previous", self)
        return transformed

    def revert(self, n: int) -> Message:
        """Return an existing version, without rerunning any transformation."""
        if n < 0:
            raise ValueError("Cannot revert a negative number of transformations")
        version = self
        for _ in range(n):
            if version._previous is None:
                raise ValueError("Cannot revert beyond the transformation history")
            version = version._previous
        return version

    def spawn(self, payload: str) -> Message:
        """Start a new text root whose provenance is this exact version.

        Transformation rollback stops at the new root, not its origin.
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
        """
        return json.dumps(self._snapshot(), ensure_ascii=False)

    def _snapshot(self) -> dict[str, object]:
        version = self
        transformations = []
        while version._previous is not None:
            transformations.append(
                {
                    "id": version.id,
                    "source_id": version._previous.id,
                    "text": version.text,
                }
            )
            version = version._previous
        return {
            "id": self.id,
            "base": version.text,
            "text": self.text,
            "origin": version._origin._snapshot()
            if version._origin is not None
            else None,
            "transformations": list(reversed(transformations)),
        }

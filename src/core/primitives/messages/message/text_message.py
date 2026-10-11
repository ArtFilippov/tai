"""Immutable text shared by conversations, strategies, and evaluators."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Concatenate, override
from uuid import UUID, uuid4

from .message import Message

@dataclass(frozen=True, slots=True)
class TextMessage(Message):
    """One immutable text version with separate transformation and origin links.

    Attributes:
        text (str): Current computed text.
        identity (UUID, default=field(default_factory=uuid4, init=False)):
            Unique identity of this version.
        links (tuple[Message | None, Message | None, int], default=field(
            default=(None, None, 0), init=False, repr=False, compare=False)):
            Previous version, spawning origin and transformation depth.
    """

    text: str
    identity: UUID = field(default_factory=uuid4, init=False)
    links: tuple[Message | None, Message | None, int] = field(
        default=(None, None, 0), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        def validate(value: object) -> None:
            if not isinstance(value, str):
                raise TypeError("Message text must be a string")

        validate(self.text)

    @property
    @override
    def id(self) -> str:
        return str(self.identity)

    @property
    @override
    def previous(self) -> Message | None:
        return self.links[0]

    @property
    @override
    def origin(self) -> Message | None:
        return self.links[1]

    @override
    def versions(self) -> Generator[Message]:
        version = self
        yield version
        for _ in range(self.links[2]):
            previous = version.previous
            assert previous is not None
            version = previous
            yield version

    @override
    def base(self) -> str:
        return self.revert(self.links[2]).text

    @override
    def to_string(self) -> str:
        return self.text

    @override
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
        transformed = TextMessage(function(self.text, *args, **kwargs))
        object.__setattr__(transformed, "links", (self, None, self.links[2] + 1))
        return transformed

    @override
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

    @override
    def spawn(self, payload: str) -> Message:
        """Start a new text root whose provenance is this exact version.

        Transformation rollback stops at the new root, not its origin.

        Args:
            payload (str): Text of the independent new root.

        Returns:
            Message: A root retaining this version as provenance.
        """
        spawned = TextMessage(payload)
        object.__setattr__(spawned, "links", (None, self.revert(0), 0))
        return spawned

    @override
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

    @override
    def snapshot(self) -> dict[str, object]:
        versions = tuple(self.versions())
        root = versions[-1]
        transformations = [
            {"id": version.id, "source_id": previous.id, "text": version.text}
            for version, previous in zip(versions, versions[1:])
        ]
        return {
            "id": str(self.identity),
            "base": root.text,
            "text": self.text,
            "origin": root.origin.snapshot() if root.origin is not None else None,
            "transformations": list(reversed(transformations)),
        }

"""Lexical relationships needed by the bounded search."""

from typing import Protocol


class WordnetLexicon(Protocol):
    """Expose synsets, surface forms and hyponyms from one prepared lexicon."""

    def synsets(self, term: str) -> list[str]:
        ...

    def forms(self, node: str) -> list[str]:
        ...

    def children(self, node: str) -> list[str]:
        ...

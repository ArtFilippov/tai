"""Canonical composition of selected DAN prompt collections."""

from collections.abc import Iterator, Mapping, Sequence
from typing import override

from prompt_collection.prompt_candidate import PromptCandidate
from prompt_collection.prompt_collection import PromptCollection


class DanPrompts(PromptCollection):
    """Traverse selected sections lazily, retaining every original source identity."""

    def __init__(self, collections: Mapping[str, PromptCollection], *,
                 sections: Sequence[str] = ("named", "ablation", "wild")) -> None:
        """Bind the available sections and reject invalid selection immediately.

        Args:
            collections (Mapping[str, PromptCollection]): Collections keyed by section.
            sections (Sequence[str], default=("named", "ablation", "wild")): Nonempty subset.

        Raises:
            ValueError: If selection is empty or includes an unknown section.
        """
        super().__init__()
        if not sections or set(sections) - {"named", "ablation", "wild"}:
            raise ValueError("DAN sections must be a nonempty subset of named, ablation, wild")
        self.collections: Mapping[str, PromptCollection] = dict(collections)
        self.sections = tuple(sections)

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "dan_prompts",
            "sections": [section for section in ("named", "ablation", "wild") if section in self.sections],
            "source": "garak.probes.dan",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
        }

    @override
    def candidates(self) -> Iterator[PromptCandidate]:
        for section in ("named", "ablation", "wild"):
            if section in self.sections:
                yield from self.collections[section].candidates()

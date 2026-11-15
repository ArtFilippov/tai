"""The fourteen named variants from the pinned garak DAN catalog."""

import json
from collections.abc import Iterator, Sequence
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import override

from prompt_collection.dan_text import DanText
from prompt_collection.prompt_candidate import PromptCandidate
from prompt_collection.prompt_collection import PromptCollection
from prompt_collection.prompt_factory import PromptFactory

DIRECTORY = files("garak").joinpath(
    "data", "dan"
)


class NamedDanPrompts(PromptCollection):
    """Keep named source identities, duplicates and native Target-name rendering."""

    def __init__(self, target: DanText,
                 prompt: PromptFactory, *,
                 directory: Traversable = DIRECTORY, sections: Sequence[str] = ("named",)) -> None:
        """Bind Target identity, candidate representation and bundled resources.

        Args:
            target (DanText): Target-bound native activation rendering.
            prompt (PromptFactory): Candidate composition.
            directory (Traversable, default=DIRECTORY): Installed package resources.
            sections (Sequence[str], default=("named",)): Nonempty named-only selection.

        Raises:
            ValueError: If selection is empty or names an unavailable section.
        """
        super().__init__()
        if not sections or set(sections) != {"named"}:
            raise ValueError("DAN sections must be nonempty; only 'named' is available")
        self.target = target
        self.prompt: PromptFactory = prompt
        self.directory = directory

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "named_dan_prompts",
            "sections": ["named"],
            **self.target.description(),
        }

    @override
    def candidates(self) -> Iterator[PromptCandidate]:
        for index, variant in enumerate(["Dan_11_0", "Dan_10_0", "Dan_9_0", "Dan_8_0", "Dan_7_0", "Dan_6_2", "Dan_6_0", "DUDE", "STAN", "DAN_Jailbreak", "AntiDAN", "ChatGPT_Developer_Mode_v2", "ChatGPT_Developer_Mode_RANTI", "ChatGPT_Image_Markdown"]):
            path = self.directory / f"{variant}.json"
            with path.open(encoding="utf-8") as stream:
                try:
                    content: object = json.load(stream)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Cannot load named DAN resource {path}: {error}") from error
            match content:
                case [str() as text] if text.strip():
                    activation = text.strip()
                case _:
                    raise ValueError(f"Malformed named DAN resource: {path}")
            # Native DANProbeMeta formats the entire string only with this marker.
            # str.format preserves native escaped-brace and error semantics.
            if "{generator.name}" in activation:
                try:
                    activation = self.target.render(activation)
                except (KeyError, IndexError, AttributeError, ValueError) as error:
                    raise ValueError(f"Cannot render named DAN resource {path}: {error}") from error
            yield self.prompt(
                activation,
                {
                    "section": "named",
                    "variant": variant,
                    "source_index": index,
                },
            )

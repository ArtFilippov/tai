"""All original DanInTheWildFull records from the pinned garak bundle."""

import json
from collections.abc import Iterator
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import override

from prompt_collection.dan_text import DanText
from prompt_collection.prompt_candidate import PromptCandidate
from prompt_collection.prompt_collection import PromptCollection
from prompt_collection.prompt_factory import PromptFactory

RESOURCE = files("garak").joinpath(
    "data", "inthewild_jailbreak_llms.json"
)


class WildDanPrompts(PromptCollection):
    """Retain corpus order and duplicate records with their zero-based indices."""

    def __init__(self, target: DanText, prompt: PromptFactory, *,
                 resource: Traversable = RESOURCE) -> None:
        """Bind native rendering, candidate composition and the installed corpus.

        Args:
            target (DanText): Native activation rendering.
            prompt (PromptFactory): Candidate representation.
            resource (Traversable, default=RESOURCE): Bundled corpus resource.
        """
        super().__init__()
        self.target = target
        self.prompt = prompt
        self.resource = resource

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "wild_dan_prompts",
            "sections": ["wild"],
            **self.target.description(),
        }

    @override
    def candidates(self) -> Iterator[PromptCandidate]:
        with self.resource.open(encoding="utf-8") as stream:
            try:
                content: list[object] = json.load(stream)
            except json.JSONDecodeError as error:
                raise ValueError(f"Cannot load wild DAN resource {self.resource}: {error}") from error
        match content:
            case list() if content:
                for index, record in enumerate(content):
                    match record:
                        case str() as text if text.strip():
                            try:
                                activation = self.target.render(
                                    text.strip()
                                )
                            except (KeyError, IndexError, AttributeError, ValueError) as error:
                                raise ValueError(f"Cannot render wild DAN record {index} in {self.resource}: {error}") from error
                            yield self.prompt(
                                activation,
                                {
                                    "section": "wild",
                                    "variant": "DanInTheWildFull",
                                    "source_index": index
                                }
                            )
                        case _:
                            raise ValueError(f"Malformed wild DAN record {index}: {self.resource}")
            case _:
                raise ValueError(f"Malformed wild DAN resource: {self.resource}")

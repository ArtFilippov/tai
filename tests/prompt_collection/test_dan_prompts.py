"""Selection and full catalog identities independently of model execution."""

import unittest
from collections.abc import Sequence
from itertools import combinations

import pytest

from prompt_collection.ablation_dan_prompts import AblationDanPrompts
from prompt_collection.dan_prompts import DanPrompts
from prompt_collection.named_dan_prompts import NamedDanPrompts
from prompt_collection.prompt_collection import PromptCollection
from prompt_collection.provided_prompt import ProvidedPrompt
from prompt_collection.scripted_prompts import ScriptedPrompts
from prompt_collection.target_dan_text import TargetDanText
from prompt_collection.wild_dan_prompts import WildDanPrompts
from target.scripted_model import ScriptedModel


class TestDanPrompts(unittest.TestCase):
    """Compose all nonempty section subsets in one canonical source traversal."""

    def prompts(self, sections: Sequence[str]) -> PromptCollection:
        target = TargetDanText(
            ScriptedModel([])
        )
        return DanPrompts(
            {
                "named": NamedDanPrompts(
                    target, ProvidedPrompt
                ),
                "ablation": AblationDanPrompts(
                    target, ProvidedPrompt
                ),
                "wild": WildDanPrompts(
                    target, ProvidedPrompt
                )
            },
            sections=sections
        )

    def test_every_subset_preserves_canonical_order_counts_and_identities(self) -> None:
        for size in range(
            1, 4
        ):
            for subset in combinations(
                (
                    "named", "ablation", "wild"
                ),
                size
            ):
                prompts = list(
                    self.prompts(
                        list(
                            reversed(subset)
                        )
                    ).candidates()
                )
                expected = [
                    (
                        "named",
                        list(
                            range(14)
                        )
                    ),
                    (
                        "ablation",
                        list(
                            range(
                                1, 128
                            )
                        )
                    ),
                    (
                        "wild",
                        list(
                            range(666)
                        )
                    )
                ]
                assert [
                    (
                        prompt.identity()["section"],
                        prompt.identity()
                        .get(
                            "combination",
                            prompt.identity()
                            .get("source_index")
                        )
                    )
                    for prompt in prompts
                ] == [
                    (section, index)
                    for section, indices in expected if section in subset
                    for index in indices
                ], "Selected catalog lost canonical order, original identity or complete section counts"

    def test_omitted_selection_contains_all_807_source_entries(self) -> None:
        collection = DanPrompts({section: self.prompts([section]) for section in ("named", "ablation", "wild")})
        assert len(list(collection.candidates())) == 807, "Default DAN traversal omitted a section or imposed a candidate cap"

    def test_invalid_or_empty_selection_fails_during_assembly(self) -> None:
        for sections in ([], ["unknown"], ["wild", "AutoDAN"], ["DanInTheWild"]):
            with pytest.raises(
                ValueError, match="nonempty subset"
            ):
                self.prompts(sections)

    def test_caller_mutation_cannot_change_a_validated_catalog(self) -> None:
        sections = ["wild"]
        collections: dict[str, PromptCollection] = {"wild": self.prompts(["wild"])}
        prompts = DanPrompts(
            collections, sections=sections
        )
        sections.clear()
        collections["wild"] = ScriptedPrompts([])
        assert len(list(prompts.candidates())) == 666, "Caller mutation replaced the catalog or erased the validated section selection"

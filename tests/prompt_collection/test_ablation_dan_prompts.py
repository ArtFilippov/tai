"""Independent parity for every original DAN 11 ablation."""

import unittest

from prompt_collection.ablation_dan_prompts import AblationDanPrompts
from prompt_collection.provided_prompt import ProvidedPrompt
from prompt_collection.target_dan_text import TargetDanText
from .dan_reference_utils import collection
from ..target.test_model import TestModel as OfflineModel


class TestAblationDanPrompts(unittest.TestCase):
    """Preserve every native combination, including exact Target-name formatting."""

    def test_every_combination_matches_the_pinned_native_implementation(self) -> None:
        for name in ("Offline/Target", "Target/{literal}"):
            prompts = list(
                AblationDanPrompts(
                    TargetDanText(
                        OfflineModel(
                            "answer", "NO", name=name
                        )
                    ),
                    ProvidedPrompt
                ).candidates()
            )
            expected = collection(
                "Ablation_Dan_11_0", name
            )
            assert [(
                prompt.identity(),
                prompt.text()
            ) for prompt in prompts] == [
                (
                    {
                        "section": "ablation",
                        "variant": "Ablation_Dan_11_0",
                        "combination": combination,
                    },
                    text,
                )
                for combination, text in enumerate(expected, start=1)
            ] and len(prompts) == 127, "Ablation text, combination numbers or native formatting diverged from pinned upstream"

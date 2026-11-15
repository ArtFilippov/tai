"""Full source fidelity and explicit failures for the bundled wild corpus."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from prompt_collection.provided_prompt import ProvidedPrompt
from prompt_collection.target_dan_text import TargetDanText
from prompt_collection.wild_dan_prompts import WildDanPrompts
from target.scripted_model import ScriptedModel
from .dan_reference_utils import collection


class TestWildDanPrompts(unittest.TestCase):
    """Keep all original records, including duplicates and unrelated instructions."""

    def test_every_record_matches_native_order_text_and_original_index(self) -> None:
        prompts = list(
            WildDanPrompts(
                TargetDanText(
                    ScriptedModel([])
                ),
                ProvidedPrompt
            ).candidates()
        )
        expected = collection(
            "DanInTheWildFull", "scripted"
        )
        assert [(
            prompt.identity(),
            prompt.text()
        ) for prompt in prompts] == [
            (
                {
                    "section": "wild",
                    "variant": "DanInTheWildFull",
                    "source_index": index,
                },
                text,
            )
            for index, text in enumerate(expected)
        ] and len(prompts) == 666, "Wild records lost exact native text, source order, duplicate identities or full corpus size"

    def test_duplicate_texts_keep_distinct_original_indices(self) -> None:
        prompts = list(
            WildDanPrompts(
                TargetDanText(
                    ScriptedModel([])
                ),
                ProvidedPrompt
            ).candidates()
        )
        duplicates = [
            (
                left.identity()["source_index"],
                right.identity()["source_index"]
            )
            for index, left in enumerate(prompts)
            for right in prompts[index + 1:] if left.text() == right.text()
        ]
        assert duplicates and all(left != right for left, right in duplicates), "Duplicate wild records were removed or merged into one source identity"

    def test_native_formatting_leaves_unrelated_tasks_and_placeholders_intact(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wild.json"
            path.write_text(
                json.dumps([" Task: {request} {{example}} ", "{generator.name} {{request}}", " Task: {request} {{example}} "]),
                encoding="utf-8",
            )
            prompts = list(
                WildDanPrompts(
                    TargetDanText(
                        ScriptedModel([])
                    ),
                    ProvidedPrompt, resource=path
                ).candidates()
            )
            assert [prompt.text() for prompt in prompts] == [
                "Task: {request} {{example}}", "scripted {request}", "Task: {request} {{example}}",
            ], "Wild loading rewrote unrelated tasks, reformatted placeholders or removed a duplicate"

    def test_invalid_resources_are_not_empty_or_shortened_successful_collections(self) -> None:
        for source in ("{", "{}", "[]", '["ok", null]', '["ok", " "]', '["ok", 1]', '["ok", "{generator.name} {unknown}"]'):
            with TemporaryDirectory() as directory:
                path = Path(directory) / "wild.json"
                path.write_text(
                    source, encoding="utf-8"
                )
                with pytest.raises(
                    (
                        ValueError, KeyError
                    )
                ):
                    list(
                        WildDanPrompts(
                            TargetDanText(
                                ScriptedModel([])
                            ),
                            ProvidedPrompt, resource=path
                        ).candidates()
                    )

    def test_missing_resource_does_not_become_exhaustion(self) -> None:
        with TemporaryDirectory() as directory, pytest.raises(FileNotFoundError):
            list(
                WildDanPrompts(
                    TargetDanText(
                        ScriptedModel([])
                    ),
                    ProvidedPrompt,
                    resource=Path(directory) / "missing.json"
                ).candidates()
            )

    def test_rendering_failure_identifies_the_original_resource_and_record(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken-wild.json"
            path.write_text(
                '["Fine", "{generator.name} {unknown}"]', encoding="utf-8"
            )
            with pytest.raises(
                ValueError, match=f"record 1 in {path}"
            ):
                list(
                    WildDanPrompts(
                        TargetDanText(
                            ScriptedModel([])
                        ),
                        ProvidedPrompt, resource=path
                    ).candidates()
                )

    def test_json_failure_identifies_the_broken_resource(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken-wild.json"
            path.write_text(
                "{", encoding="utf-8"
            )
            with pytest.raises(
                ValueError, match=f"Cannot load wild DAN resource {path}"
            ):
                list(
                    WildDanPrompts(
                        TargetDanText(
                            ScriptedModel([])
                        ),
                        ProvidedPrompt, resource=path
                    ).candidates()
                )

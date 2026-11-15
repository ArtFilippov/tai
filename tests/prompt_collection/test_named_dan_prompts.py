"""Fidelity of every named entry against the installed pinned upstream."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from prompt_collection.named_dan_prompts import NamedDanPrompts
from prompt_collection.provided_prompt import ProvidedPrompt
from prompt_collection.target_dan_text import TargetDanText
from target.scripted_model import ScriptedModel
from .dan_reference_utils import reference
from ..target.test_model import TestModel as OfflineModel


class TestNamedDanPrompts(unittest.TestCase):
    """Preserve native text, source order and duplicate identity without a harness."""

    def test_every_named_entry_matches_native_loading_and_rendering(self) -> None:
        for model in (
            ScriptedModel([]),
            OfflineModel(
                "answer", "YES", name="Target/{literal}"
            )
        ):
            prompts = list(
                NamedDanPrompts(
                    TargetDanText(model), ProvidedPrompt
                ).candidates()
            )
            expected = reference(
                str(
                    model.settings()["id"]
                )
            )
            assert [(
                prompt.identity(),
                prompt.text()
            ) for prompt in prompts] == [
                (
                    {
                        "section": "named",
                        "variant": variant,
                        "source_index": index
                    },
                    text
                )
                for index, (variant, text) in enumerate(
                    expected.items()
                )
            ] and len(prompts) == 14, "Named entries lost native text, declaration order or source identity"

    def test_identical_variants_remain_distinct_sources(self) -> None:
        prompts = list(
            NamedDanPrompts(
                TargetDanText(
                    ScriptedModel([])
                ),
                ProvidedPrompt
            ).candidates()
        )
        pairs = [
            (
                left.identity()["variant"],
                right.identity()["variant"]
            )
            for index, left in enumerate(prompts)
            for right in prompts[index + 1:] if left.text() == right.text()
        ]
        assert pairs == [("Dan_10_0", "DAN_Jailbreak")], "Identical named prompts were deduplicated or their source names changed"

    def test_native_formatting_preserves_escaped_and_unrelated_placeholders(self) -> None:
        for source, expected in [
            (" {generator.name} {{request}} {{generator.name}} ", "scripted {request} {generator.name}"),
            (" {request} {{example}} OpenAI ", "{request} {{example}} OpenAI"),
        ]:
            with TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "Dan_11_0.json").write_text(
                    json.dumps([source]),
                    encoding="utf-8"
                )
                prompt = next(
                    NamedDanPrompts(
                        TargetDanText(
                            ScriptedModel([])
                        ),
                        ProvidedPrompt, directory=path
                    ).candidates()
                )
                assert prompt.text() == expected, "Native conditional formatting changed unrelated fields or escaped braces"

    def test_damaged_resources_and_formatting_fail_instead_of_skipping(self) -> None:
        for content in ('{', '{}', '[]', '[null]', '[1]', '[""]', '["  "]', '["a", "b"]', '["{generator.name} {unrelated}"]'):
            with TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "Dan_11_0.json").write_text(
                    content, encoding="utf-8"
                )
                with pytest.raises(
                    ValueError, match="Dan_11_0.json"
                ):
                    next(
                        NamedDanPrompts(
                            TargetDanText(
                                ScriptedModel([])
                            ),
                            ProvidedPrompt, directory=path
                        ).candidates()
                    )

    def test_user_local_resources_cannot_override_the_pinned_bundle(self) -> None:
        expected = reference("scripted")["Dan_11_0"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "garak" / "data" / "dan"
            path.mkdir(parents=True)
            (path / "Dan_11_0.json").write_text(
                '["User-local override"]', encoding="utf-8"
            )
            script = (
                "import json; "
                "from garak.data import path; "
                "from prompt_collection.named_dan_prompts import NamedDanPrompts; "
                "from prompt_collection.provided_prompt import ProvidedPrompt; "
                "from prompt_collection.target_dan_text import TargetDanText; "
                "from target.scripted_model import ScriptedModel; "
                "native = json.loads((path / 'dan' / 'Dan_11_0.json').read_text()); "
                "candidate = next(NamedDanPrompts(TargetDanText(ScriptedModel([])), ProvidedPrompt).candidates()); "
                "print(json.dumps([native, candidate.text()]))"
            )
            result = subprocess.run(
                [
                    sys.executable, "-c", script
                ],
                env={
                    **os.environ,
                    "XDG_DATA_HOME": directory
                },
                check=True, capture_output=True, text=True
            )
            assert json.loads(result.stdout) == [["User-local override"], expected], "User-local garak data replaced the recorded pinned activation text"

    def test_missing_resource_is_not_an_empty_collection(self) -> None:
        with TemporaryDirectory() as directory, pytest.raises(FileNotFoundError):
            next(
                NamedDanPrompts(
                    TargetDanText(
                        ScriptedModel([])
                    ),
                    ProvidedPrompt,
                    directory=Path(directory)
                ).candidates()
            )

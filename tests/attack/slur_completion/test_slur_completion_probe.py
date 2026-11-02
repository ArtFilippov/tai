"""Slur Completion selects a complete prefilled Target response."""

import unittest
import subprocess
import sys
from pathlib import Path

import pytest

from attack.attack_run import AttackRun
from attack.desired_response.provided_desired_response import ProvidedDesiredResponse
from attack.slur_completion.slur_completion_probe import SlurCompletionProbe
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal

BASE = "Explain café precisely"
DESIRED = "Carefully explained answer with examples"


class TestSlurCompletionProbe(unittest.TestCase):
    """Verify one Target exchange and a selected complete result."""

    def attempt(self, path: Path, source: ProvidedDesiredResponse) -> ScriptedChat:
        """Run one isolated answer-opening attempt.

        Args:
            path (Path): Journal destination.
            source (ProvidedDesiredResponse): Bound supplied answer.

        Returns:
            ScriptedChat: Target observation.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        target = ScriptedChat([" generated suffix"])
        _ = AttackRun(
            PlainBase(BASE),
            target,
            RecordedJudge(
                QwenJudge(
                    BASE,
                    ScriptedChat(["NO"])
                ),
                journal, BASE
            ),
            journal,
            probe=SlurCompletionProbe(
                BASE, source, journal
            )
        ).run()
        return target

    def test_selects_full_target_message_before_negative_judgment(self) -> None:
        with temporary_journal("slur.jsonl") as path:
            target = self.attempt(
                path,
                ProvidedDesiredResponse(
                    PlainBase(BASE), DESIRED
                )
            )
            events = list(
                ParsedJournal(path).events()
            )
            prefix = target.prefixes[0]
            assert (
                len(target.transcript(0).turns),
                events[1]["desired_response"], events[2]["continuation"], events[3]["text"], events[5]["text"],
                ParsedJournal(path).read()["assessment"]
            ) == (1, DESIRED, " generated suffix", prefix + " generated suffix", prefix + " generated suffix", False), "The opening was treated as generated or the selected answer changed"

    def test_wrong_base_fails_without_target_inference(self) -> None:
        with temporary_journal("slur-wrong.jsonl") as path:
            journal = VersionedJournal(
                DiskJournal(path)
            )
            target = ScriptedChat(["suffix"])
            with pytest.raises(
                ValueError, match="preparation failed"
            ):
                _ = AttackRun(
                    PlainBase(BASE),
                    target,
                    RecordedJudge(
                        QwenJudge(
                            BASE,
                            ScriptedChat(["NO"])
                        ),
                        journal, BASE
                    ),
                    journal,
                    probe=SlurCompletionProbe(
                        BASE,
                        ProvidedDesiredResponse(
                            PlainBase("other"), DESIRED
                        ),
                        journal
                    )
                ).run()
            assert (
                ParsedJournal(path).read()["calls"],
                ParsedJournal(path).read()["reason"]
            ) == (
                {
                    "target": 0,
                    "attacker": 0,
                    "judge": 0
                },
                "preparation_error"
            ), "Missing preparation contacted Target or was reported as refusal"

    def test_target_failure_is_not_preparation_failure(self) -> None:
        with temporary_journal("target-error.jsonl") as path:
            journal = VersionedJournal(
                DiskJournal(path)
            )
            with pytest.raises(IndexError):
                _ = AttackRun(
                    PlainBase(BASE),
                    ScriptedChat([]),
                    RecordedJudge(
                        QwenJudge(
                            BASE,
                            ScriptedChat(["YES"])
                        ),
                        journal, BASE
                    ),
                    journal,
                    probe=SlurCompletionProbe(
                        BASE,
                        ProvidedDesiredResponse(
                            PlainBase(BASE),
                            DESIRED
                        ),
                        journal
                    )
                ).run()
            assert (
                ParsedJournal(path).read()["reason"],
                ParsedJournal(path).read()["text"]
            ) == (
                "execution_error", None
            ), "A failed Target continuation was mistaken for missing preparation or a refusal"

    def test_standalone_commands_use_prefill_and_select_full_response(self) -> None:
        for kind in ("slur_main", "leak_main"):
            with temporary_journal(kind + ".jsonl") as path:
                script = (
                    "import main.continuation_cli as cli; "
                    "from tests.attack.model_fixture_utils import continuation; "
                    "cli.snapshot = lambda offline: None; "
                    "cli.cuda_device = lambda gpu: 'controlled'; "
                    "cli.LazyQwenModel = lambda device, tokens: continuation(); "
                    "getattr(cli, '" + kind + "')()"
                )
                subprocess.run(
                    [
                        sys.executable, "-c", script,
                        str(path),
                        BASE, "--gpu", "0", "--desired-response", "A café has a special recipe."
                    ],
                    check=True, capture_output=True, text=True
                )
                events = list(
                    ParsedJournal(path).events()
                )
                assert ParsedJournal(path).read()["text"] == next(event["response"] for event in events if event.get("role") == "target"), "The standalone command did not select the complete prefilled Target answer"

    def test_deferred_standalone_command_records_source_call(self) -> None:
        with temporary_journal("deferred-cli.jsonl") as path:
            script = (
                "import main.continuation_cli as cli; "
                "from tests.attack.model_fixture_utils import continuation; "
                "cli.snapshot = lambda offline: None; "
                "cli.cuda_device = lambda gpu: 'controlled'; "
                "cli.LazyQwenModel = lambda device, tokens: continuation(); "
                "cli.slur_main()"
            )
            subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(path),
                    BASE, "--gpu", "0", "--deferred"
                ],
                check=True, capture_output=True, text=True
            )
            assert ParsedJournal(path).read()["calls"] == {
                "target": 1,
                "attacker": 1,
                "judge": 1
            }, "Deferred preparation was omitted or counted as another Target exchange"

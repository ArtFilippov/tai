"""One roleplay keeps the bound request and judges only the selected answer."""

import json
import subprocess
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path

import pytest

from attack.attack_run import AttackRun
from attack.grandma.grandma_probe import GrandmaProbe
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal

BASE = "Explain café\nwith detail"
ANSWER = "A complete café explanation\nwith details"


class TestGrandmaAttack(unittest.TestCase):
    """Exercise request fidelity, role isolation and command execution."""

    def attempt(self, path: Path, verdict: str = "YES") -> Mapping[str, ScriptedChat]:
        """Run a controlled attempt with separate Target and judge conversations.

        Args:
            path (Path): Reserved journal destination.
            verdict (str, default="YES"): Independent judge verdict.

        Returns:
            Mapping[str, ScriptedChat]: Target and judge observations.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        target = ScriptedChat([ANSWER])
        judge = ScriptedChat([verdict])
        _ = AttackRun(
            PlainBase(BASE),
            target,
            RecordedJudge(
                QwenJudge(
                    BASE, judge
                ),
                journal, BASE
            ),
            journal,
            probe=GrandmaProbe(
                BASE, journal
            )
        ).run()
        return {
            "target": target,
            "judge": judge
        }

    def test_exact_base_is_present_once_in_roleplay(self) -> None:
        with temporary_journal("grandma-base.jsonl") as path:
            roles = self.attempt(path)
            prompt = roles["target"].transcript(0).turns[0].content.text
            assert prompt.endswith(BASE) and prompt.count(BASE) == 1 and "grandmother" in prompt, "The roleplay altered, omitted or repeated the original request"

    def test_prompt_and_effective_identity_are_recorded(self) -> None:
        with temporary_journal("grandma-prompt.jsonl") as path:
            roles = self.attempt(path)
            events = list(
                ParsedJournal(path).events()
            )
            assert ("'type': 'grandma'" in str(events[0]["attack"]), "3f50ea5ff9cd7050099940647c15c39b07a93392" in str(events[0]["attack"]), events[1]["prompt"]) == (
                True, True,
                roles["target"].transcript(0).turns[0].content.text
            ), "The journal lost the effective strategy or selected prompt"

    def test_one_target_exchange_and_independent_judge(self) -> None:
        with temporary_journal("grandma-calls.jsonl") as path:
            roles = self.attempt(path)
            assert (
                len(roles["target"].histories),
                len(roles["judge"].histories),
                ParsedJournal(path).read()["calls"]
            ) == (
                1, 1,
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 1
                }
            ), "The attempt reused a conversation or made extra Target calls"

    def test_selected_response_precedes_identical_judge_input(self) -> None:
        with temporary_journal("grandma-result.jsonl") as path:
            roles = self.attempt(path)
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                [(event["event"], event["text"]) for event in events if event["event"] in ("result", "judgment")],
                json.loads(
                    roles["judge"].transcript(0)
                    .last_message("user").text
                )
            ) == (
                [
                    (
                        "result", ANSWER
                    ),
                    (
                        "judgment", ANSWER
                    )
                ],
                {
                    "base_request": BASE,
                    "presented_result": ANSWER
                }
            ), "The complete answer was changed or judged before selection"

    def test_negative_verdict_is_not_success(self) -> None:
        with temporary_journal("grandma-no.jsonl") as path:
            self.attempt(
                path, "NO"
            )
            assert ParsedJournal(path).read()["assessment"] is False, "The roleplay or a non-refusal was treated as success"

    def test_invalid_verdict_is_recorded_as_failure(self) -> None:
        with temporary_journal("grandma-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Malformed judge verdict"
            ):
                self.attempt(
                    path, "MAYBE"
                )
            assert ParsedJournal(path).read()["reason"] == "assessment_error", "The invalid verdict was reported as success"

    def test_target_failure_records_no_result(self) -> None:
        with temporary_journal("grandma-target-error.jsonl") as path:
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
                    probe=GrandmaProbe(
                        BASE, journal
                    )
                ).run()
            assert (
                ParsedJournal(path).read()["reason"],
                ParsedJournal(path).read()["text"]
            ) == ("execution_error", None), "A failed exchange was presented as a result"

    def test_standalone_command_runs_offline(self) -> None:
        with temporary_journal("grandma-cli.jsonl") as path:
            script = (
                "import main.grandma_cli as cli; "
                "from tests.attack.model_fixture_utils import grandma; "
                "cli.snapshot = lambda offline: None; "
                "cli.cuda_device = lambda gpu: 'controlled'; "
                "cli.LazyQwenModel = lambda device, tokens: grandma(); "
                "cli.main()"
            )
            subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(path),
                    BASE, "--gpu", "0"
                ],
                check=True, capture_output=True, text=True
            )
            assert ParsedJournal(path).read()["text"] == ANSWER, "The standalone command presented a different answer"

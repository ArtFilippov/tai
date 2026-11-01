"""Independent ThreatenJSON exchanges and final JSON answer judgment."""

import json
import subprocess
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path

import pytest

from attack.attack_run import AttackRun
from attack.threaten_json_probe import JSON_ONLY, PRESSURE, ThreatenJSONProbe
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.json_answer_judge import JSONAnswerJudge
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from ..journal_utils import temporary_journal

BASE = "Explain café\nwith detail"


class TestThreatenJSONAttack(unittest.TestCase):
    """Pin the separate conversations, selection and verdict boundary."""

    def run_attack(self, path: Path, replies: list[str], verdicts: list[str]) -> Mapping[str, ScriptedChat]:
        """Run a fully scripted attempt with an independent judge.

        Args:
            path (Path): Reserved journal path.
            replies (list[str]): Control and pressured Target replies.
            verdicts (list[str]): Answer evaluator replies.

        Returns:
            Mapping[str, ScriptedChat]: Observed Target and judge.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        target = ScriptedChat(replies)
        judge = ScriptedChat(verdicts)
        AttackRun(
            PlainBase(BASE),
            target,
            RecordedJudge(
                JSONAnswerJudge(
                    QwenJudge(
                        BASE, judge
                    )
                ),
                journal, BASE
            ),
            journal,
            probe=ThreatenJSONProbe(
                BASE, journal
            )
        ).run()
        return {
            "target": target,
            "judge": judge
        }

    def test_control_and_pressured_history_are_independent(self) -> None:
        with temporary_journal("history.jsonl") as path:
            roles = self.run_attack(
                path,
                [
                    '"first"', '"second"'
                ],
                ["YES"]
            )
            assert [(turn.role, turn.content.text) for turn in roles["target"].transcript(1).turns] == [
                ("user", f"{BASE}\n\n{JSON_ONLY}\n\n{PRESSURE}")
            ], "The control conversation leaked into the pressured conversation"

    def test_same_exact_base_and_json_instructions(self) -> None:
        with temporary_journal("prompts.jsonl") as path:
            roles = self.run_attack(
                path,
                [
                    '"first"', '"second"'
                ],
                ["YES"]
            )
            assert [turn.content.text for turn in roles["target"].transcript(0).turns] == [
                f"{BASE}\n\n{JSON_ONLY}"
            ], "The control request changed the base or JSON-only instructions"

    def test_already_json_control_is_not_presented(self) -> None:
        with temporary_journal("control.jsonl") as path:
            self.run_attack(
                path,
                [
                    '"already valid"', '"pressured answer"'
                ],
                ["NO"]
            )
            assert (
                ParsedJournal(path).read()["text"],
                [event["text"] for event in ParsedJournal(path).events() if event["event"] == "result"]
            ) == ('"pressured answer"', ['"pressured answer"']), "The JSON control was presented as an attack result"

    def test_both_json_parse_results_are_journaled(self) -> None:
        with temporary_journal("parse.jsonl") as path:
            self.run_attack(
                path,
                [
                    "not JSON", '"valid"'
                ],
                ["NO"]
            )
            assert [(event["phase"], event["json_parseable"]) for event in ParsedJournal(path).events() if event["event"] == "call" and event["role"] == "target"] == [
                ("control", False), ("pressured", True)
            ], "The journal lost separate control and pressured JSON parse results"

    def test_control_is_not_counted_as_refusal(self) -> None:
        with temporary_journal("diagnostic.jsonl") as path:
            self.run_attack(
                path,
                [
                    "not JSON", '"answer"'
                ],
                ["NO"]
            )
            assert not ParsedJournal(path).read()["refusals"], "The control exchange was counted as a refused candidate"

    def test_invalid_json_cannot_pass_yes_judge(self) -> None:
        with temporary_journal("invalid.jsonl") as path:
            roles = self.run_attack(
                path,
                [
                    "{}", "Sure: {}"
                ],
                ["YES"]
            )
            assert (
                ParsedJournal(path).read()["assessment"],
                len(roles["judge"].histories)
            ) == (False, 0), "Invalid JSON reached the answer judge or passed"

    def test_valid_substantive_json_passes(self) -> None:
        with temporary_journal("valid.jsonl") as path:
            self.run_attack(
                path,
                [
                    "No", '"A detailed café explanation"'
                ],
                ["YES"]
            )
            assert ParsedJournal(path).read()["assessment"] is True, "A substantive JSON answer was rejected"

    def test_irrelevant_json_fails(self) -> None:
        with temporary_journal("irrelevant.jsonl") as path:
            self.run_attack(
                path,
                [
                    "No", '"Unrelated fact"'
                ],
                ["NO"]
            )
            assert ParsedJournal(path).read()["assessment"] is False, "Irrelevant JSON passed the answer rubric"

    def test_judge_uses_answer_rubric(self) -> None:
        with temporary_journal("rubric.jsonl") as path:
            roles = self.run_attack(
                path,
                [
                    "No", "42"
                ],
                ["NO"]
            )
            assert "requested information in meaningful detail" in roles["judge"].transcript(0).turns[0].content.text, "The JSON judge used an unrelated rubric"

    def test_judge_receives_exact_base_and_presented_text(self) -> None:
        with temporary_journal("identity.jsonl") as path:
            roles = self.run_attack(
                path,
                [
                    '"control"', '"café answer"'
                ],
                ["YES"]
            )
            assert json.loads(roles["judge"].transcript(0).last_message("user").text) == {
                "base_request": BASE,
                "presented_result": '"café answer"'
            }, "The evaluator lost the exact base or selected answer"

    def test_any_json_value_type_is_eligible(self) -> None:
        for answer in ("42", "true", "null", "[]", "{}", '"answer"'):
            with self.subTest(answer=answer), temporary_journal("types.jsonl") as path:
                roles = self.run_attack(
                    path,
                    [
                        "control", answer
                    ],
                    ["NO"]
                )
                assert len(roles["judge"].histories) == 1, "A valid JSON value type bypassed the answer evaluator"

    def test_failed_pressured_exchange_is_not_presented(self) -> None:
        with temporary_journal("failure.jsonl") as path:
            with pytest.raises(IndexError):
                self.run_attack(
                    path, ['"control"'], ["YES"]
                )
            assert ParsedJournal(path).read()["reason"] == "execution_error", "A failed pressured exchange was presented as a result"

    def test_python_module_exposes_flags(self) -> None:
        command = subprocess.run(
            [sys.executable, "-m", "main.threaten_json_cli", "--help"],
            capture_output=True, text=True, check=True
        )
        assert "--tokens" in command.stdout, "The standalone module omitted its token limit"

    def test_standalone_command_runs_offline(self) -> None:
        with temporary_journal("cli.jsonl") as path:
            script = (
                "import main.threaten_json_cli as cli; "
                "from tests.attack.model_fixture_utils import threaten_json; "
                "cli.snapshot = lambda offline: None; "
                "cli.cuda_device = lambda gpu: 'controlled'; "
                "cli.LazyQwenModel = lambda device, tokens: threaten_json(); "
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
            assert ParsedJournal(path).read()["text"] == '"Detailed café explanation"', "The standalone command presented the wrong Target reply"

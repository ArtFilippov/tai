"""Phrasing behavior through public attacks and durable journals."""

import unittest
import hashlib
import json
import subprocess
import sys

import pytest

from attack.attack_run import AttackRun
from attack.assessed_candidates import AssessedCandidates
from attack.phrasing.configured_phrasing_plan import ConfiguredPhrasingPlan
from attack.phrasing.phrasing_probe import PhrasingProbe
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.recorded_judge import RecordedJudge
from judge.qwen_judge import QwenJudge
from target.interrupted_chat import InterruptedChat
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal
from .phrasing_fixture_utils import attempt


class TestPhrasingProbe(unittest.TestCase):
    """Assess original-request fulfillment with independent candidate conversations."""

    def test_first_success_keeps_candidate_and_does_not_prepare_more(self) -> None:
        with temporary_journal("early.jsonl") as path:
            journal = VersionedJournal(
                DiskJournal(path)
            )
            base = PlainBase("Original request")
            target = ScriptedChat(["  Complete\n  "])
            attacker = ScriptedChat(['"How did people act?"'])
            result = AttackRun(
                base, target,
                RecordedJudge(
                    EqualJudge("  Complete\n  "),
                    journal,
                    base.text()
                ),
                journal,
                probe=PhrasingProbe(
                    base,
                    AssessedCandidates(journal),
                    journal, attacker,
                    plan=ConfiguredPhrasingPlan(
                        "past", 20
                    )
                )
            ).run()
            outcome = ParsedJournal(path).read()
            assert (
                result, outcome["successful_candidate"], outcome["calls"],
                target.transcript(0).turns[0].content.text
            ) == (
                "  Complete\n  ",
                {
                    "tense": "past",
                    "generation": 1,
                    "position": 1
                },
                {
                    "attacker": 1,
                    "target": 1,
                    "judge": 1
                },
                "How did people act?"
            ), "Phrasing prepared extra variants, changed the full response or lost its successful candidate"

    def test_instructions_match_the_pinned_research_with_an_exact_multiline_request(self) -> None:
        base = PlainBase("  Объясни café с {двумя} примерами\nи ссылками \\1  ")
        for tense, digest in (
            ("past", "a6aa7d64ae9aa6beb640a729c25a7fa96e11995b02288a2843fc6b1ef37f85e6"),
            ("future", "46fafcd270c4349e1d0c50f5eeb9ec20efa7d272f0d698ebf9d579c62bd10aab")
        ):
            with temporary_journal("parity.jsonl") as path:
                attacker = ScriptedChat(["Question"])
                attempt(
                    path, base, attacker,
                    ScriptedChat(["Complete"]),
                    judge=EqualJudge("Complete"),
                    tense=tense, attempts=1
                )
                assert hashlib.sha256(attacker.transcript(0).turns[0].content.text.encode()).hexdigest() == digest, "Phrasing changed the research instruction, dates, whitespace or request substitutions"

    def test_duplicates_consume_budget_without_target_calls_and_histories_stay_fresh(self) -> None:
        with temporary_journal("duplicates.jsonl") as path:
            attacker = ScriptedChat([' "Question A" ', "Question A", '"Question B"', "Question B", "question a"])
            target = ScriptedChat(["First", "Second", ""])
            result = attempt(
                path,
                PlainBase("Original"),
                attacker, target,
                judge=EqualJudge("Complete"),
                tense="past", attempts=5
            )
            events = list(
                ParsedJournal(path).events()
            )
            outcome = ParsedJournal(path).read()
            assert (
                result, outcome["assessment"], outcome["reason"], outcome["calls"],
                [event["reason"] for event in events if event.get("action") == "phrasing_candidate"],
                [target.transcript(index).turns[0].content.text for index in range(3)],
                [len(attacker.transcript(index).turns) for index in range(5)],
                [len(target.transcript(index).turns) for index in range(3)]
            ) == (
                "", False, "candidates_exhausted",
                {
                    "attacker": 5,
                    "target": 3,
                    "judge": 3
                },
                [
                    "unique", "duplicate", "unique", "duplicate", "unique"
                ],
                [
                    " Question A ", "Question B", "question a"
                ],
                [1] * 5, [1] * 3
            ), "Duplicates reached Target, escaped the generation budget, changed text or shared histories"

    def test_judge_assesses_the_original_request_and_full_response(self) -> None:
        with temporary_journal("assessment.jsonl") as path:
            base = PlainBase("  Исходный café\nзапрос  ")
            judge = ScriptedChat(["YES"])
            answer = "  Full\nanswer with trailing material  "
            attempt(
                path, base,
                ScriptedChat(["How will people act?"]),
                ScriptedChat([answer]),
                judge=QwenJudge(
                    base.text(),
                    judge
                ),
                tense="future", attempts=3
            )
            assert json.loads(judge.transcript(0).turns[-1].content.text) == {
                "base_request": base.text(),
                "presented_result": answer
            }, "Judge received the reformulation or a modified Target answer"

    def test_only_edge_whitespace_is_ignored_when_deduplicating(self) -> None:
        with temporary_journal("spacing.jsonl") as path:
            target = ScriptedChat(["No"] * 3)
            attempt(
                path,
                PlainBase("Original"),
                ScriptedChat(["Question A", "Question  A", "question A"]),
                target,
                judge=EqualJudge("Complete"),
                tense="future", attempts=3
            )
            assert ParsedJournal(path).read()["calls"] == {
                "attacker": 3,
                "target": 3,
                "judge": 3
            }, "Deduplication collapsed internal whitespace or letter case"

    def test_empty_preparation_is_a_failed_attempt_even_after_a_negative_answer(self) -> None:
        for replies, previous in (([' "" \n '], None), (["Question", ' "" \n '], "Negative")):
            with temporary_journal("empty.jsonl") as path:
                with pytest.raises(
                    ValueError, match="empty reformulation"
                ):
                    attempt(
                        path,
                        PlainBase("Original"),
                        ScriptedChat(replies),
                        ScriptedChat(["Negative"]),
                        judge=EqualJudge("Complete"),
                        tense="past", attempts=4
                    )
                outcome = ParsedJournal(path).read()
                assert (outcome["status"], outcome["reason"], outcome["text"], outcome["calls"]) == (
                    "failure", "preparation_error", previous,
                    {
                        "attacker": 1 if previous is None else 2,
                        "target": 0 if previous is None else 1,
                        "judge": 0 if previous is None else 1
                    }
                ), "Empty preparation was retried, treated as exhaustion or lost the previous result"

    def test_role_errors_abort_without_retrying_or_inventing_a_negative_verdict(self) -> None:
        for role in ("attacker", "target", "judge"):
            with temporary_journal("failure.jsonl") as path:
                with pytest.raises(
                    ValueError, match="unavailable"
                ):
                    attempt(
                        path, PlainBase("Original"),
                        InterruptedChat([], "Attacker unavailable") if role == "attacker" else ScriptedChat(["Question"]),
                        InterruptedChat([], "Target unavailable") if role == "target" else ScriptedChat(["Selected"]),
                        judge=QwenJudge("Original", InterruptedChat([], "Judge unavailable")) if role == "judge" else EqualJudge("Complete"),
                        tense="future", attempts=20
                    )
                outcome = ParsedJournal(path).read()
                assert (
                    outcome["status"], outcome["reason"], outcome["assessment"],
                    outcome.get("successful_candidate")
                ) == (
                    "failure", "assessment_error" if role == "judge" else "execution_error", None, None
                ), "A technical role failure was retried or converted to a scored candidate"

    def test_successful_candidate_survives_duplicate_skips_and_fresh_process_recovery(self) -> None:
        with temporary_journal("recovered.jsonl") as path:
            attempt(
                path,
                PlainBase("Original"),
                ScriptedChat(['"Question A"', "Question A", "Question B"]),
                ScriptedChat(["No", "Complete"]),
                judge=EqualJudge("Complete"),
                tense="future", attempts=3
            )
            process = subprocess.run(
                [
                    sys.executable, "-c", "import json,sys; from pathlib import Path; from journal.parsed_journal import ParsedJournal; print(json.dumps(ParsedJournal(Path(sys.argv[1])).read()))",
                    str(path)
                ],
                capture_output=True, text=True, check=True
            )
            assert json.loads(process.stdout)["successful_candidate"] == {
                "tense": "future",
                "generation": 3,
                "position": 2
            }, "Fresh-process recovery lost the exact successful generation after duplicate skipping"

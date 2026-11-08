"""The FalseAssertion adaptation preserves candidate order and journal recovery."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import pytest

from attack.attack_run import AttackRun
from attack.assessed_candidates import AssessedCandidates
from attack.desired_response.cached_desired_response import CachedDesiredResponse
from attack.desired_response.desired_response import DesiredResponse
from attack.desired_response.provided_desired_response import ProvidedDesiredResponse
from attack.desired_response.short_desired_response import ShortDesiredResponse
from attack.false_assertion.false_assertion_probe import FalseAssertionProbe
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.false_assertion_judge import FalseAssertionJudge
from judge.judge import Judge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal

BASE = "Explain café precisely"
ANSWER = "A café serves coffee and pastries."
SHORT = "A café serves coffee."
WHY = "Why is it that a café serves coffee?"
EXPLAIN = "Explain why a café serves coffee."
REFUSAL = "I cannot provide information about that."


class TestFalseAssertionProbe(unittest.TestCase):
    """Exercise lazy prompts and linked refusal assessments offline."""

    def attempt(self, path: Path, attacker: ScriptedChat, target: ScriptedChat,
                source: DesiredResponse) -> str | None:
        """Run a journaled attempt with controlled independent chats.

        Args:
            path (Path): Temporary journal path.
            attacker (ScriptedChat): Preparation responses.
            target (ScriptedChat): Candidate responses.
            source (DesiredResponse): Request-bound answer.

        Returns:
            str | None: Selected Target reply.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        return AttackRun(
            PlainBase(BASE),
            target,
            RecordedJudge(
                FalseAssertionJudge(),
                journal, BASE
            ),
            journal,
            probe=FalseAssertionProbe(
                BASE,
                ShortDesiredResponse(
                    source, attacker
                ),
                attacker, journal,
                candidates=AssessedCandidates(journal)
            )
        ).run()

    def test_exhaustion_preserves_three_fresh_conversations(self) -> None:
        with temporary_journal("false-exhausted.jsonl") as path:
            attacker = ScriptedChat([SHORT, WHY, EXPLAIN])
            target = ScriptedChat([REFUSAL, REFUSAL, REFUSAL])
            result = self.attempt(
                path, attacker, target,
                ProvidedDesiredResponse(
                    PlainBase(BASE),
                    ANSWER
                )
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                result, [event["prompt"] for event in events if event.get("role") == "target"], [len(target.transcript(index).turns) for index in range(3)],
                ParsedJournal(path).read()["reason"],
                len([event for event in events if event["event"] == "result"])
            ) == (
                REFUSAL, [SHORT, WHY, EXPLAIN], [1, 1, 1], "candidates_exhausted", 3
            ), "Candidates were reordered, shared a conversation, or lost their assessments"

    def test_positive_direct_answer_skips_both_prompts(self) -> None:
        with temporary_journal("false-direct.jsonl") as path:
            attacker = ScriptedChat([SHORT])
            result = self.attempt(
                path, attacker,
                ScriptedChat(["Unrelated but not a refusal"]),
                ProvidedDesiredResponse(
                    PlainBase(BASE),
                    ANSWER
                )
            )
            assert (
                result,
                ParsedJournal(path).read()["calls"],
                ParsedJournal(path).read()["assessment"]
            ) == (
                "Unrelated but not a refusal",
                {
                    "target": 1,
                    "attacker": 1,
                    "judge": 1
                },
                True
            ), "Non-refusal was confused with fulfillment or a later prompt was prepared"

    def test_positive_second_prompt_skips_third(self) -> None:
        with temporary_journal("false-second.jsonl") as path:
            attacker = ScriptedChat([SHORT, WHY])
            result = self.attempt(
                path, attacker,
                ScriptedChat([REFUSAL, "An unrelated answer"]),
                ProvidedDesiredResponse(
                    PlainBase(BASE),
                    ANSWER
                )
            )
            assert (
                result,
                ParsedJournal(path).read()["calls"]
            ) == (
                "An unrelated answer",
                {
                    "target": 2,
                    "attacker": 2,
                    "judge": 2
                }
            ), "A positive second assessment generated the third prompt"

    def test_empty_shortening_fails_before_target(self) -> None:
        with temporary_journal("false-empty.jsonl") as path:
            with pytest.raises(
                ValueError, match="answer preparation failed"
            ):
                self.attempt(
                    path,
                    ScriptedChat([" "]),
                    ScriptedChat([]),
                    ProvidedDesiredResponse(
                        PlainBase(BASE),
                        ANSWER
                    )
                )
            assert (
                ParsedJournal(path).read()["reason"],
                ParsedJournal(path).read()["calls"]
            ) == (
                "preparation_error",
                {
                    "target": 0,
                    "attacker": 1,
                    "judge": 0
                }
            ), "Empty shortening contacted Target or lost its preparation error"

    def test_empty_later_prompt_preserves_last_assessed_result(self) -> None:
        with temporary_journal("false-later.jsonl") as path:
            with pytest.raises(
                ValueError, match="prompt preparation failed"
            ):
                self.attempt(
                    path,
                    ScriptedChat([SHORT, " "]),
                    ScriptedChat([REFUSAL]),
                    ProvidedDesiredResponse(
                        PlainBase(BASE),
                        ANSWER
                    )
                )
            assert (
                ParsedJournal(path).read()["text"],
                ParsedJournal(path).read()["assessment_error"],
                ParsedJournal(path).read()["reason"]
            ) == (REFUSAL, None, "preparation_error"), "The last assessed result was lost on preparation failure"

    def test_deferred_source_is_separate_from_shortening(self) -> None:
        with temporary_journal("false-deferred.jsonl") as path:
            source = CachedDesiredResponse(
                ScriptedChat([ANSWER])
            )
            _ = self.attempt(
                path,
                ScriptedChat([SHORT]),
                ScriptedChat(["Fine"]),
                source
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                events[3]["response"], events[4]["source_answer"], events[4]["short_answer"],
                ParsedJournal(path).read()["calls"]
            ) == (
                SHORT, ANSWER, SHORT,
                {
                    "target": 1,
                    "attacker": 2,
                    "judge": 1
                }
            ), "Deferred answer called Target or lost source and shortened texts"

    def test_assessment_error_keeps_selected_reply(self) -> None:
        with temporary_journal("false-assessment.jsonl") as path:
            journal = VersionedJournal(
                DiskJournal(path)
            )
            judge = Mock(spec=Judge)
            judge.assess.side_effect = ValueError("detector unavailable")
            judge.description.return_value = {"type": "unavailable"}
            attacker = ScriptedChat([SHORT])
            with pytest.raises(
                ValueError, match="detector unavailable"
            ):
                AttackRun(
                    PlainBase(BASE),
                    ScriptedChat(["Target answer"]),
                    RecordedJudge(
                        judge, journal, BASE
                    ),
                    journal,
                    probe=FalseAssertionProbe(
                        BASE,
                        ShortDesiredResponse(
                            ProvidedDesiredResponse(
                                PlainBase(BASE),
                                ANSWER
                            ),
                            attacker
                        ),
                        attacker, journal,
                        candidates=AssessedCandidates(journal)
                    )
                ).run()
            assert (
                ParsedJournal(path).read()["text"],
                ParsedJournal(path).read()["reason"],
                ParsedJournal(path).read()["assessment_error"]
            ) == (
                "Target answer", "assessment_error", "ValueError('detector unavailable')"
            ), "An assessment error erased the selected reply"


    def test_standalone_command_uses_supplied_or_deferred_source(self) -> None:
        for option in (("--desired-response", ANSWER), ("--deferred",)):
            with temporary_journal("false-cli.jsonl") as path:
                script = (
                    "import main.false_assertion_cli as cli; "
                    "from tests.attack.model_fixture_utils import false_assertion; "
                    "cli.snapshot = lambda offline: None; "
                    "cli.cuda_device = lambda gpu: 'controlled'; "
                    "cli.LazyQwenModel = lambda device, tokens: false_assertion(); "
                    "cli.main()"
                )
                subprocess.run(
                    [
                        sys.executable, "-c", script,
                        str(path),
                        BASE, "--gpu", "0", *option
                    ],
                    check=True, capture_output=True, text=True
                )
                assert ParsedJournal(path).read()["calls"] == {
                    "target": 1,
                    "attacker": 1 if len(option) == 2 else 2,
                    "judge": 1
                }, "The standalone command did not preserve the selected answer source"

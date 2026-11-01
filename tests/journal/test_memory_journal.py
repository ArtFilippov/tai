"""Measurements from recorded FITD calls without filesystem storage."""

import unittest

from attack.attack_run import AttackRun
from attack.fitd.fitd_probe import FITDProbe
from base.plain_base import PlainBase
from journal.memory_journal import MemoryJournal
from judge.equal_judge import EqualJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.scripted_chat import ScriptedChat


class TestMemoryJournal(unittest.TestCase):
    """Check that generation and assessment remain distinct observations."""

    def journal(self) -> MemoryJournal:
        journal = MemoryJournal()
        journal.record(
            {
                "event": "start",
                "base": "abc"
            }
        )
        journal.record(
            {
                "event": "call",
                "role": "attacker",
                "prompt": "Polish this sentence: abc",
                "response": "abd"
            }
        )
        journal.record(
            {
                "event": "call",
                "role": "attacker",
                "prompt": "You are a helpful assistant.\nYour task is to transform abc",
                "response": "abe"
            }
        )
        journal.record(
            {
                "event": "call",
                "role": "attacker",
                "prompt": "Check if the answer is aligned",
                "response": "yes"
            }
        )
        journal.record(
            {
                "event": "call",
                "role": "target",
                "prompt": "abd",
                "response": "one"
            }
        )
        journal.record(
            {
                "event": "call",
                "role": "target",
                "prompt": "abd",
                "response": "two"
            }
        )
        journal.record(
            {
                "event": "call",
                "role": "target",
                "prompt": "abc",
                "response": "three"
            }
        )
        journal.record(
            {
                "event": "result",
                "text": "three"
            }
        )
        journal.record(
            {
                "event": "judgment",
                "text": "three",
                "assessment": False
            }
        )
        journal.record(
            {
                "event": "end",
                "status": "round_limit"
            }
        )
        return journal

    def test_candidate_changes(self) -> None:
        assert self.journal().report()["candidate_changes"] == [
            {
                "before": "abc",
                "after": "abd",
                "changed": True,
                "similarity": 2 / 3
            },
            {
                "before": "abd",
                "after": "abe",
                "changed": True,
                "similarity": 2 / 3
            }
        ], "Candidate changes included a check response or used the wrong predecessor"

    def test_target_requests(self) -> None:
        assert self.journal().report()["target_requests"] == ["abd", "abc"], "Target requests were not distinct or ordered"

    def test_outcome_separate_from_candidates(self) -> None:
        assert (
            self.journal()
            .report()["presented_result"],
            self.journal()
            .report()["verdict"]
        ) == ("three", False), "The candidate was mistaken for the presented result or verdict"

    def test_recorded_event_is_snapshot(self) -> None:
        journal = MemoryJournal()
        event: dict[str, object] = {
            "event": "step",
            "notes": {"turn": 1}
        }
        journal.record(event)
        event["notes"] = {"turn": 2}
        assert journal.events() == [
            {
                "event": "step",
                "notes": {"turn": 1}
            }
        ], "An external mutation changed a saved event"

    def test_runs_are_independent(self) -> None:
        assert MemoryJournal().events() == [], "A fresh run inherited earlier events"

    def test_new_result_does_not_inherit_verdict(self) -> None:
        journal = MemoryJournal()
        journal.record(
            {
                "event": "start",
                "base": "Request"
            }
        )
        journal.record(
            {
                "event": "result",
                "text": "three"
            }
        )
        journal.record(
            {
                "event": "judgment",
                "text": "three",
                "assessment": False
            }
        )
        journal.record(
            {
                "event": "result",
                "text": "four"
            }
        )
        journal.record(
            {
                "event": "assessment_error",
                "text": "four",
                "error": "Unavailable"
            }
        )
        journal.record(
            {
                "event": "end",
                "status": "failure"
            }
        )
        assert journal.report()["verdict"] is None, "A new result inherited an earlier verdict"

    def test_fitd_records_target_requests_in_memory(self) -> None:
        journal = MemoryJournal()
        target = ScriptedChat(["First", "Second"])
        chat = ScriptedChat(["Opening", "One", "Two", "Three", "yes", "yes", "yes"])
        attacker = ContinuingChat(chat)
        _ = AttackRun(
            PlainBase("Request"),
            target,
            RecordedJudge(
                EqualJudge("Never"),
                journal, "Request"
            ),
            journal,
            probe=FITDProbe(
                "Request", attacker, journal, 2
            )
        ).run()
        assert journal.report()["target_requests"] == ["Opening", "Two"], "FITD requests were not observed without a disk journal"

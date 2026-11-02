"""Leak Replay validates masks before any Target inference."""

import unittest
from pathlib import Path

import pytest

from attack.attack_run import AttackRun
from attack.desired_response.provided_desired_response import ProvidedDesiredResponse
from attack.leak_replay.leak_replay_probe import LeakReplayProbe
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal

BASE = "Explain the café"
DESIRED = "A café has a special recipe."


class TestLeakReplayProbe(unittest.TestCase):
    """Check independent masking, strict templates and full-result judgment."""

    def attempt(self, path: Path, template: str) -> ScriptedChat:
        """Run one attempt with an independent controlled masking chat.

        Args:
            path (Path): Journal destination.
            template (str): Prepared masking response.

        Returns:
            ScriptedChat: Target observation.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        target = ScriptedChat([" supplied continuation"])
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
            probe=LeakReplayProbe(
                BASE,
                ProvidedDesiredResponse(
                    PlainBase(BASE), DESIRED
                ),
                ScriptedChat([template]),
                journal
            )
        ).run()
        return target

    def test_masking_and_selection_are_independent(self) -> None:
        with temporary_journal("leak.jsonl") as path:
            target = self.attempt(
                path, "A café has a [[MASK]] recipe."
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                len(target.transcript(0).turns),
                events[1]["role"], events[2]["template"], events[3]["continuation"], events[4]["text"],
                ParsedJournal(path).read()["assessment"]
            ) == (
                1, "attacker", "A café has a [[MASK]] recipe.",
                " supplied continuation", target.prefixes[0] + " supplied continuation", False
            ), "The masked response was promoted to a Target result or an extra Target call was made"

    def test_invalid_templates_never_contact_target(self) -> None:
        for template in ("", DESIRED, "Altered café has a [[MASK]] recipe.", "A café has a [[MASK]] recipe!", "A café has a [[MASK]] recipe.[[MASK]]"):
            with temporary_journal("invalid-leak.jsonl") as path:
                journal = VersionedJournal(
                    DiskJournal(path)
                )
                target = ScriptedChat(["suffix"])
                with pytest.raises(
                    ValueError, match="Masked answer"
                ):
                    _ = AttackRun(
                        PlainBase(BASE),
                        target,
                        RecordedJudge(
                            QwenJudge(
                                BASE,
                                ScriptedChat(["YES"])
                            ),
                            journal, BASE
                        ),
                        journal,
                        probe=LeakReplayProbe(
                            BASE,
                            ProvidedDesiredResponse(
                                PlainBase(BASE), DESIRED
                            ),
                            ScriptedChat([template]),
                            journal
                        )
                    ).run()
                assert (
                    ParsedJournal(path).read()["calls"],
                    ParsedJournal(path).read()["reason"]
                ) == (
                    {
                        "target": 0,
                        "attacker": 1,
                        "judge": 0
                    },
                    "preparation_error"
                ), "An invalid mask contacted Target"

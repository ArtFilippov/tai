"""Deferred answers are cached by exact base request."""

import unittest

import pytest

from attack.desired_response.cached_desired_response import CachedDesiredResponse
from journal.memory_journal import MemoryJournal
from target.scripted_chat import ScriptedChat


class TestCachedDesiredResponse(unittest.TestCase):
    """Verify lazy resolution, per-base identity and failure boundaries."""

    def test_resolves_once_per_base_and_records_actual_calls(self) -> None:
        chat = ScriptedChat(["First complete answer", "Second complete answer"])
        source = CachedDesiredResponse(chat)
        journal = MemoryJournal()
        first = source.response(
            "first", journal
        )
        repeat = source.response(
            "first", journal
        )
        second = source.response(
            "second", journal
        )
        assert (
            first, repeat, second,
            chat.transcript(1).turns[0].content.text.endswith("second"),
            len(
                journal.events()
            )
        ) == (
            "First complete answer", "First complete answer", "Second complete answer", True, 2
        ), "A cached answer was recomputed, shared with another base, or recorded as a new call"

    def test_empty_preparation_fails_without_caching(self) -> None:
        chat = ScriptedChat([" ", "Complete answer"])
        source = CachedDesiredResponse(chat)
        with pytest.raises(
            ValueError, match="empty"
        ):
            _ = source.response(
                "base",
                MemoryJournal()
            )
        assert source.response("base", MemoryJournal()) == "Complete answer", "Empty preparation was cached as an answer"

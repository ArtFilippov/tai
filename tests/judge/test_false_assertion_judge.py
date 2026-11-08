"""The refusal heuristic does not claim fulfillment."""

import unittest

from judge.false_assertion_judge import FalseAssertionJudge


class TestFalseAssertionJudge(unittest.TestCase):
    """Check the pinned English refusal detector on unmodified responses."""

    def test_non_refusal_is_positive(self) -> None:
        assert FalseAssertionJudge().assess("An unrelated answer") is True, "Non-refusal was judged as fulfillment failure"

    def test_refusal_is_negative(self) -> None:
        assert FalseAssertionJudge().assess("I cannot provide information about that.") is False, "An explicit refusal passed"

    def test_empty_reply_is_negative(self) -> None:
        assert FalseAssertionJudge().assess("  ") is False, "An empty response passed"

    def test_non_english_text_is_not_translated(self) -> None:
        assert FalseAssertionJudge().assess("Ответ по-русски") is True, "The English-language limitation was hidden"

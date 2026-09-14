import pytest

from tai import ContainsTextEvaluator, Evaluator, Message


def test_contains_text_checks_current_text_exactly_and_explains_the_same_decision():
    evaluator = ContainsTextEvaluator("CONTROL")
    for text, expected in [("A CONTROL result", True), ("control", False), ("", False)]:
        result = Message(text)
        assert evaluator.is_success(result) is expected
        assert evaluator.evidence(result) == (["CONTROL"] if expected else [])
        assert ("not found" in evaluator.reason(result)) is (not expected)
    with pytest.raises(ValueError):
        ContainsTextEvaluator("")
    with pytest.raises(NotImplementedError):
        Evaluator().is_success(Message("result"))
    with pytest.raises(NotImplementedError):
        Evaluator().reason(Message("result"))
    with pytest.raises(NotImplementedError):
        Evaluator().evidence(Message("result"))

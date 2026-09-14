import pytest

from tai import ContainsTextEvaluator, Evaluator, Message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A CONTROL result", True),
        ("CONTROLCONTROL", True),
        ("An ordinary result", False),
        ("control", False),
        ("", False),
    ],
)
def test_contains_text_checks_current_text_exactly_and_explains_the_same_decision(
    text, expected
):
    evaluator = ContainsTextEvaluator("CONTROL")
    result = Message(text)
    assert evaluator.is_success(result) is expected
    assert evaluator.evidence(result) == (["CONTROL"] if expected else [])
    assert ("not found" in evaluator.reason(result)) is (not expected)


def test_empty_criterion_is_invalid():
    with pytest.raises(ValueError):
        ContainsTextEvaluator("")


@pytest.mark.parametrize("method", ["is_success", "reason", "evidence"])
def test_unsupported_evaluator_methods_raise_explicitly(method):
    with pytest.raises(NotImplementedError):
        getattr(Evaluator(), method)(Message("result"))


def test_partial_evaluators_support_each_method_independently():
    class SuccessOnly(Evaluator):
        def is_success(self, result):
            return result.to_string() == "result"

    class ReasonOnly(Evaluator):
        def reason(self, result):
            return "Explanation of " + result.to_string()

    class EvidenceOnly(Evaluator):
        def evidence(self, result):
            return [result.to_string()]

    result = Message("result")
    for evaluator, supported, expected in [
        (SuccessOnly(), "is_success", True),
        (ReasonOnly(), "reason", "Explanation of result"),
        (EvidenceOnly(), "evidence", ["result"]),
    ]:
        assert getattr(evaluator, supported)(result) == expected
        for method in {"is_success", "reason", "evidence"} - {supported}:
            with pytest.raises(NotImplementedError):
                getattr(evaluator, method)(result)


@pytest.mark.parametrize("location", ["base", "intermediate", "origin"])
def test_contains_text_ignores_control_string_outside_current_text(location):
    evaluator = ContainsTextEvaluator("CONTROL")
    if location == "base":
        result = Message("CONTROL").transform(str.replace, "CONTROL", "ordinary")
    elif location == "intermediate":
        result = (
            Message("ordinary")
            .transform(str.replace, "ordinary", "CONTROL")
            .transform(str.replace, "CONTROL", "ordinary")
        )
    else:
        result = Message("CONTROL").spawn("ordinary")
    assert "CONTROL" in result.dump()
    assert result.to_string() == "ordinary"
    assert evaluator.is_success(result) is False
    assert "not found" in evaluator.reason(result)
    assert evaluator.evidence(result) == []

    selected = result.transform(str.replace, "ordinary", "CONTROL")
    assert evaluator.is_success(selected) is True
    assert "not found" not in evaluator.reason(selected)
    assert evaluator.evidence(selected) == ["CONTROL"]
    assert evaluator.is_success(result) is False

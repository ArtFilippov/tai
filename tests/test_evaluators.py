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
def test_contains_text_checks_full_presented_text_exactly(text: str, expected: bool):
    evaluator = ContainsTextEvaluator("CONTROL")
    result = Message(text)
    assert (
        evaluator.evaluate(Message("CONTROL in base is not evidence"), result)
        is expected
    )


def test_empty_criterion_is_invalid():
    with pytest.raises(ValueError):
        ContainsTextEvaluator("")


def test_one_operation_without_inheritance_is_the_entire_contract():
    class EqualText:
        def evaluate(self, base: Message, result: Message) -> bool:
            return base.to_string() == result.to_string()

    evaluator: Evaluator = EqualText()
    assert evaluator.evaluate(Message("same"), Message("same")) is True
    assert evaluator.evaluate(Message("other"), Message("same")) is False


@pytest.mark.parametrize("location", ["base", "intermediate", "origin"])
def test_contains_text_ignores_control_string_outside_current_text(location: str):
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
    base = Message("Explicit base")
    assert evaluator.evaluate(base, result) is False

    selected = result.transform(str.replace, "ordinary", "CONTROL")
    assert evaluator.evaluate(base, selected) is True
    assert evaluator.evaluate(base, result) is False

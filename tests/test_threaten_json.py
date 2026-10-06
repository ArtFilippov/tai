"""The control is evidence; only the pressured JSON answer is assessed."""

import json
from pathlib import Path
from typing import override

import pytest
from evaluation_fakes import HeldResponse, ScriptedModel

from attacks.threaten_json.requests import COERCION, requests
from tai import Evaluator, EvaluatorSetup, LLMChat, Message, ThreatenJSONExperiment, ThreatenJSONUser, UserState, capture_run, save_run


class AnswerJudge(Evaluator):
    """Record exactly the presented answer passed to the answer rubric."""

    def __init__(self, answer: bool) -> None:
        """Store the scripted assessment.

        Args:
            answer (bool): Substantive verdict.
        """
        super().__init__()
        self.answer = answer
        self.pairs: list[tuple[Message, Message]] = []

    @override
    def evaluate(self, base: Message, result: Message) -> bool:
        self.pairs.append((base, result))
        return self.answer


def test_requests_preserve_base_and_differ_only_by_garak_coercion() -> None:
    base = Message("Old").transform(lambda _: "Give π {x}\nReturn an array.")
    control, pressured = requests(base)
    assert control.origin is pressured.origin is base
    assert control.text.count(base.text) == pressured.text.count(base.text) == 1
    assert pressured.text == control.text + "\n\n" + COERCION
    assert "Bard" not in pressured.text and "dictionary with two keys" not in pressured.text


@pytest.mark.parametrize(
    ("control", "pressured", "answer", "expected", "calls"),
    [
        ("Not JSON", '["substantive"]', True, True, 1),
        ("Not JSON", "null", True, True, 1),
        ("Not JSON", "a preamble {}", True, False, 0),
        ("Not JSON", "NaN", True, False, 0),
        ("Not JSON", '"irrelevant"', False, False, 1),
        ('{"already": true}', '"substantive"', True, True, 1),
    ],
)
def test_only_pressured_answer_is_evaluated(
    tmp_path: Path, control: str, pressured: str, answer: bool, expected: bool, calls: int,
) -> None:
    base = Message("Answer this explicit request")
    target = ScriptedModel([control, pressured])
    judge = AnswerJudge(answer)
    user = ThreatenJSONExperiment(LLMChat(target), base, judge, journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED, user.status().reason
    result = user.result()
    evaluation = user.status().evaluation
    assert result is not None and result.text == pressured
    assert evaluation is not None and evaluation.success is expected
    assert len(judge.pairs) == calls
    assert all(pair[0] is base and pair[1] is result for pair in judge.pairs)
    assert len(target.requests) == 2 and all(len(turns) == 1 for turns in target.requests)
    events = [json.loads(event.dump()) for event in user.journal.events()]
    calls_in_journal = [event["data"] for event in events if event["kind"] == "call_completed"]
    assert len(calls_in_journal) == 2 and calls_in_journal[0]["conversation_id"] != calls_in_journal[1]["conversation_id"]
    assert [call["context"]["cause"]["reason"] for call in calls_in_journal] == ["json_control", "json_pressured"]
    comparisons = [event["data"] for event in events if event["kind"] == "json_comparison"]
    assert len(comparisons) == 1 and comparisons[0]["control_parseable"] is (control.startswith("{"))
    assert comparisons[0]["result_parseable"] is (pressured[0] in '["n' and pressured != "NaN")
    assert comparisons[0]["control_id"] == calls_in_journal[0]["response"]["id"]
    assert comparisons[0]["result_id"] == result.id
    directory = save_run(capture_run(user, journal=user.journal), tmp_path)
    assert (directory / "result.txt").read_text() == pressured


def test_bare_user_and_repeated_experiment_keep_independent_results() -> None:
    base = Message("Respond with one JSON number")
    for kind in (ThreatenJSONUser, ThreatenJSONExperiment):
        judge = AnswerJudge(True)
        user = kind(LLMChat(ScriptedModel(["control", "42", "second control", "7"])), base, judge)
        user.start()
        assert user.wait(10)
        first = user.result()
        user.start()
        assert user.wait(10)
        result = user.result()
        assert first is not result and result is not None and result.text == "7"
        assert len(judge.pairs) == 2
        if isinstance(user, ThreatenJSONExperiment):
            assert len(user.journals) == 2 and all(journal.finished for journal in user.journals)


def test_model_answer_rubric_is_not_safety_rubric() -> None:
    model = ScriptedModel(['{"success": true}'])
    setup = EvaluatorSetup.answer(LLMChat(model))
    assert setup.evaluator.evaluate(Message("Give the number 2"), Message("2")) is True
    assert setup.settings()["rubric_version"] == "json-answer-v1"
    assert "safety restriction" not in str(setup.settings()["rubric"])


def test_model_answer_judge_receives_only_pressured_response() -> None:
    model = ScriptedModel(['{"success": true}'])
    user = ThreatenJSONExperiment(
        LLMChat(ScriptedModel(['{"control": true}', '"answer"'])), Message("Give a JSON string"),
        EvaluatorSetup.answer(LLMChat(model)),
    )
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED
    prompt = model.requests[0][-1].message.text
    assert json.loads(prompt.split("INPUT_JSON:\n", 1)[1]) == {
        "base_request": "Give a JSON string", "result": '"answer"',
    }
    assert len(model.requests) == 1


def test_stop_during_control_finishes_call_without_publishing_or_assessing() -> None:
    held = HeldResponse("0")
    judge = AnswerJudge(True)
    user = ThreatenJSONExperiment(LLMChat.cancellable(ScriptedModel([held, "1"])), Message("Give one"), judge)
    user.start()
    try:
        assert held.entered.wait(5)
        user.stop()
    finally:
        held.release.set()
    assert user.wait(10)
    assert user.status().state is UserState.STOPPED and user.result() is None
    assert not judge.pairs
    assert len([event for event in user.journal.events() if event.kind == "call_completed"]) == 1


def test_stop_during_pressured_call_retains_format_comparison_without_assessing() -> None:
    held = HeldResponse('{"answer": 2}')
    judge = AnswerJudge(True)
    user = ThreatenJSONExperiment(LLMChat.cancellable(ScriptedModel(["not JSON", held])), Message("Give 2"), judge)
    user.start()
    try:
        assert held.entered.wait(5)
        user.stop()
    finally:
        held.release.set()
    assert user.wait(10)
    assert user.status().state is UserState.STOPPED and user.result() is None
    assert not judge.pairs
    comparisons = [json.loads(event.dump())["data"] for event in user.journal.events() if event.kind == "json_comparison"]
    assert len(comparisons) == 1 and comparisons[0]["control_parseable"] is False
    assert comparisons[0]["result_parseable"] is True

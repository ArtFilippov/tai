"""Terminal observation uses the public User and never supplies an evaluation."""

from tai import Evaluator, LLMChat, Message
from tai import FITDExperiment as FITDUser
from tai.terminal import run_foreground


def test_terminal_does_not_reevaluate_or_attribute_old_score_to_new_result(
    tmp_path, capsys
):
    class PartialEvaluator(Evaluator):
        def __init__(self):
            self.results = []

        def evaluate(self, base, result):
            self.results.append(result)
            if len(self.results) == 2:
                raise RuntimeError("Required evaluation unavailable")
            return False

    target = iter(["One", "Two", "Three", "Four", "Negative", "Unchecked"])
    attacker = iter(["Initial", "L1", "L2", "L3"] + ["yes"] * 6)
    evaluator = PartialEvaluator()
    user = FITDUser(
        LLMChat(lambda turns: next(target)),
        LLMChat(lambda turns: next(attacker)),
        Message("Explain the fictional puzzle"),
        evaluator,
        journal_dir=tmp_path,
    )
    assert run_foreground(user, artifacts_dir=tmp_path) == 1
    output = capsys.readouterr().out
    assert "Required evaluation unavailable" in output
    assert "Unchecked" in output and "финальный текст не оценивался" in output
    assert "Оценка результата: неуспех" not in output
    assert [r.to_string() for r in evaluator.results] == ["Negative", "Unchecked"]
    assert user.status().evaluation is None
    assert user.result() is evaluator.results[-1]

"""Real installed garak resources drive each base through independent variants."""

import json
from pathlib import Path

import pytest
from evaluation_fakes import HeldResponse, ScriptedModel

from garak.probes import adaptive_attacks as native
from attacks.adaptive_attacks.garak_adapter import variants
from tai import (
    AdaptiveAttacksExperiment, AdaptiveAttacksUser, BadCharactersDeletionExperiment, ContainsTextEvaluator, EvaluatorSetup,
    LLMChat, Message, UserState, capture_run, save_run,
)


class PairEvaluator:
    def __init__(self, success_at: int | None) -> None:
        super().__init__()
        self.success_at = success_at
        self.pairs: list[tuple[Message, Message]] = []

    def evaluate(self, base: Message, result: Message) -> bool:
        self.pairs.append((base, result))
        return len(self.pairs) == self.success_at


def test_variants_use_installed_template_resource_order_and_exact_base() -> None:
    base = Message("A previous version").transform(lambda _: "Explain {arbitrary} π\n{goal} and {suffix}\r\n")
    suffixes = native._load_lines(native.SUFFIXES_PATH)
    actual = variants(base)
    assert len(actual) == len(suffixes) == 5
    assert [message.text for message in actual] == [native.ADAPTIVE_TEMPLATE.format(goal=base.text, suffix=suffix) for suffix in suffixes]
    assert all(message.origin is base for message in actual)
    assert len({message.id for message in actual}) == 5


@pytest.mark.parametrize("success_at", [2, None])
def test_each_variant_is_reset_presented_and_evaluated_as_the_exact_message(tmp_path: Path, success_at: int | None) -> None:
    base = Message("Explain {this}\npuzzle π")
    model, evaluator = ScriptedModel([f"Response {index}" for index in range(1, 6)]), PairEvaluator(success_at)
    target = LLMChat(model, system="Keep these system instructions")
    system = target.history()[0].message
    user = AdaptiveAttacksExperiment(target, base, evaluator, journal_dir=tmp_path)
    assert not model.requests and not evaluator.pairs
    user.start()
    assert user.wait(10)
    count = success_at or 5
    assert user.status().state is UserState.COMPLETED, user.status().reason
    assert user.status().reason == ("evaluator_success" if success_at else "strategy_exhausted")
    assert len(model.requests) == len(evaluator.pairs) == count
    assert all(len(turns) == 2 and turns[0].message is system for turns in model.requests)
    assert [turns[-1].message.text for turns in model.requests] == [message.text for message in variants(base)[:count]]
    assert all(pair[0] is base for pair in evaluator.pairs)
    assert evaluator.pairs[-1][1] is user.result() is target.history()[-1].message
    assert not user.status().progress.attacker_calls
    assert user.status().progress.max_calls_per_conv is None
    _assert_artifacts(user, tmp_path, count, success_at is not None)


def _assert_artifacts(user: AdaptiveAttacksExperiment, root: Path, count: int, success: bool) -> None:
    directory = save_run(capture_run(user, journal=user.journal), root)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert (directory / "result.txt").read_text() == f"Response {count}"
    assert metadata["evaluation"] == {"result_id": metadata["result_id"], "success": success}
    settings = metadata["settings"]
    assert settings["attack"] == {"type": "adaptive_attacks"}
    assert settings["variants"] == 5 and settings["early_stop"] == "evaluator_success"
    assert "Attacker" not in settings and "max_calls_per_conv" not in settings
    assert settings["garak_revision"] == "3f50ea5ff9cd7050099940647c15c39b07a93392"
    events = [json.loads(event.dump()) for event in user.journal.events()]
    completed = [event["data"] for event in events if event["kind"] == "call_completed"]
    assert len({event["conversation_id"] for event in completed}) == count
    started = [event["data"] for event in events if event["kind"] == "call_started"]
    assert all(event["input"]["origin"]["id"] == user.base.id for event in started)
    assert len([event for event in events if event["kind"] == "evaluation_completed"]) == count
    assert user.journal.error is None and user.journal.finished


def test_bare_user_and_observed_runs_use_the_same_execution() -> None:
    for user_type in (AdaptiveAttacksUser, AdaptiveAttacksExperiment):
        user = user_type(LLMChat(ScriptedModel(["yes"])), Message("A base"), ContainsTextEvaluator("yes"))
        user.start()
        assert user.wait(10)
        assert user.status().reason == "evaluator_success"
        assert user.result() is not None


@pytest.mark.parametrize("experiment_type", [AdaptiveAttacksExperiment, BadCharactersDeletionExperiment])
def test_repeated_and_overlapping_users_keep_bases_participants_and_journals_independent(
    experiment_type: type[AdaptiveAttacksExperiment] | type[BadCharactersDeletionExperiment],
) -> None:
    held = [HeldResponse("yes one"), HeldResponse("yes two")]
    models = [ScriptedModel([response, "yes again"]) for response in held]
    evaluators = [PairEvaluator(1), PairEvaluator(1)]
    users = [experiment_type(LLMChat.cancellable(model), Message(f"Base {{{index}}}"), evaluator)
             for index, (model, evaluator) in enumerate(zip(models, evaluators, strict=True))]
    for user in users:
        user.start()
    try:
        assert all(response.entered.wait(5) for response in held)
        assert all(not user.wait(0) for user in users)
    finally:
        for response in held:
            response.release.set()
        assert all(user.wait(10) for user in users)
    journals = tuple(user.journal for user in users)
    for user, model, evaluator in zip(users, models, evaluators, strict=True):
        first = capture_run(user, journal=user.journal)
        evaluator.pairs.clear()
        user.start()
        assert user.wait(10)
        assert user.status().reason == "evaluator_success"
        assert user.status().run_id != first.snapshot.status.run_id
        assert len(user.journals) == 2 and user.journals[0] in journals
        assert user.result() is not first.snapshot.result
        assert evaluator.pairs[0][0] is user.base
        assert all(turns[-1].message.origin is user.base for turns in model.requests)
        assert all(journal.finished for journal in user.journals)
    assert len({journal.status().run_id for user in users for journal in user.journals}) == 4


def test_model_evaluator_owns_an_independent_chat_and_resets_each_pair() -> None:
    model = ScriptedModel(['{"success": false}', '{"success": true}'])
    target = LLMChat(ScriptedModel(["one", "two"]))
    evaluator = LLMChat(model, system="Evaluate")
    user = AdaptiveAttacksExperiment(target, Message("Base"), EvaluatorSetup.model(evaluator))
    user.start()
    assert user.wait(10)
    assert user.status().reason == "evaluator_success"
    assert len(model.requests) == 2 and all(len(turns) == 2 for turns in model.requests)
    assert all(turns[0].message.text == "Evaluate" for turns in model.requests)
    with pytest.raises(ValueError, match="independent"):
        AdaptiveAttacksUser(target, Message("Base"), EvaluatorSetup.model(target))

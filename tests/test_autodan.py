"""AutoDAN fitness, native verification, presentation and stop boundaries."""

import json
from pathlib import Path
from threading import Event
from typing import Any, cast, override

import pytest

from fitd_support import releasing

from tai import AutoDANExperiment, AutoDANUser, ChatTurn, ContainsTextEvaluator, LLMChat, Message, UserState, capture_run, save_run


class LocalTarget(LLMChat):
    def __init__(self, replies: list[str]) -> None:
        self.scored: list[tuple[str, str]] = []
        self.requests: list[str] = []
        self.replies = iter(replies)
        super().__init__(self._reply, model="local")

    def _reply(self, turns: tuple[ChatTurn, ...]) -> str:
        request = turns[-1]
        self.requests.append(request.message.text)
        return next(self.replies)

    def score_continuation(self, prompt: str, expected: str) -> float:
        self.scored.append((prompt, expected))
        return 0.0 if "New instruction." in prompt else 1.0


def events(user: AutoDANExperiment) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], json.loads(event.dump())) for event in user.journal.events()]


def test_fitness_uses_base_but_native_check_presents_only_successful_prefix(tmp_path: Path) -> None:
    base = Message("Original request")
    target = LocalTarget(["Useful response"])
    attacker = LLMChat(lambda turns: "revision")
    user = AutoDANExperiment(target, attacker, base, ContainsTextEvaluator("[PROMPT]:"),
                             journal_dir=tmp_path, expected="Sure", num_steps=2, batch_size=2, num_elites=1)
    user.start()
    assert user.wait(5) and user.status().state is UserState.COMPLETED
    prefix = target.requests[0]
    assert target.scored == [(f"{prefix} {base.text}", "Sure")] * 2
    assert target.history()[-2].message.text == prefix
    result, evaluation = user.result(), user.status().evaluation
    assert result is not None and result.text == prefix
    assert evaluation is not None and evaluation.success
    directory = save_run(capture_run(user, journal=user.journal), tmp_path)
    assert (directory / "result.txt").read_text() == prefix
    recorded = events(user)
    assert len([event for event in recorded if event["kind"] == "autodan_fitness"]) == 2
    assert [event["data"]["success"] for event in recorded if event["kind"] == "autodan_native_check"] == [True]
    assert [event["data"]["base_id"] for event in recorded if event["kind"] == "evaluation_started"] == [base.id]
    assert user.status().progress.target_calls == 1 and not user.status().progress.attacker_calls


def test_rejected_search_retains_evidence_without_result_and_mutates_independent_attacker(tmp_path: Path) -> None:
    target = LocalTarget(["Sorry, I cannot help", "I can't assist"])
    attacker = LLMChat(lambda turns: "New instruction.", model="mutation")
    user = AutoDANExperiment(target, attacker, Message("Base"), ContainsTextEvaluator("New"),
                             journal_dir=tmp_path, expected="Sure", num_steps=2, batch_size=2,
                             num_elites=1, mutation_rate=1.0)
    user.start()
    assert user.wait(5) and user.status().reason == "strategy_exhausted"
    assert user.result() is None and user.status().evaluation is None
    assert target.requests[1] == "New instruction."
    assert target.scored[-1] == ("New instruction. Base", "Sure")
    assert len(attacker.history()) == 2 and len(target.history()) == 4
    directory = save_run(capture_run(user, journal=user.journal), tmp_path)
    assert not (directory / "result.txt").exists()
    recorded = events(user)
    assert len([event for event in recorded if event["kind"] == "autodan_fitness"]) == 4
    assert not any(event["kind"] == "evaluation_completed" for event in recorded)
    assert [event["data"]["success"] for event in recorded if event["kind"] == "autodan_native_check"] == [False, False]


def test_stop_during_scoring_waits_for_admitted_call_without_target_exchange() -> None:
    entered, release = Event(), Event()

    class HeldTarget(LocalTarget):
        @override
        def score_continuation(self, prompt: str, expected: str) -> float:
            entered.set()
            assert release.wait(5)
            return super().score_continuation(prompt, expected)

    target = HeldTarget(["never sent"])
    user = AutoDANExperiment(target, LLMChat(lambda turns: "revision"), Message("Base"),
                             ContainsTextEvaluator("yes"), expected="Sure", num_steps=2, batch_size=2, num_elites=1)
    user.start()
    with releasing(user, release):
        assert entered.wait(5)
        user.stop()
    assert user.status().state is UserState.STOPPED and user.result() is None
    assert len(target.scored) == 1 and not target.requests
    assert any(event["kind"] == "autodan_fitness" for event in events(user))


def test_unsupported_target_and_invalid_profile_fail_before_admission() -> None:
    base, attacker = Message("Base"), LLMChat(lambda turns: "revision")
    with pytest.raises(ValueError, match="logits"):
        AutoDANUser(LLMChat(lambda turns: "answer"), attacker, base, ContainsTextEvaluator("yes"), expected="Sure")
    with pytest.raises(ValueError, match="num_elites"):
        AutoDANUser(LocalTarget([]), attacker, base, ContainsTextEvaluator("yes"), expected="Sure", batch_size=2, num_elites=2)
    chat = LocalTarget([])
    with pytest.raises(ValueError, match="independent"):
        AutoDANUser(chat, chat, base, ContainsTextEvaluator("yes"), expected="Sure")


def test_restart_after_exhaustion_uses_new_journal_and_presents_only_second_success() -> None:
    target = LocalTarget(["Sorry", "Useful response"])
    user = AutoDANExperiment(target, LLMChat(lambda turns: "mutation"), Message("Base"),
                             ContainsTextEvaluator("[PROMPT]:"), expected="Sure", num_steps=1, batch_size=2, num_elites=1)
    user.start()
    assert user.wait(5) and user.result() is None
    first = user.journal
    user.start()
    assert user.wait(5) and user.status().reason == "evaluator_success"
    assert user.result() is not None and user.journal is not first and len(user.journals) == 2
    assert len([event for event in events(user) if event["kind"] == "autodan_fitness"]) == 2
    assert len(target.history()) == 2

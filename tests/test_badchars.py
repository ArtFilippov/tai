"""The real native deletion path preserves text, enumeration and evidence."""

import json
import random
from collections.abc import Iterator
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest
from evaluation_fakes import ScriptedModel
from garak.probes.badchars import BadCharacters
from test_adaptive_attacks import PairEvaluator

from attacks.badchars.garak_adapter import candidates
from attacks.badchars.profile import DeletionProfile
from tai import (
    BadCharactersDeletionExperiment, BadCharactersDeletionUser, ContainsTextEvaluator,
    LLMChat, Message, UserState, capture_run, save_run,
)


def test_native_order_positions_ascii_and_multiple_insertions() -> None:
    base = Message("Earlier").transform(lambda _: "π\n")
    profile = DeletionProfile(perturbation_budget=2, max_position_candidates=3, max_ascii_variants=2)
    actual = list(candidates(base, profile=profile))
    # An independent native instance is used only as a pure generator reference.
    native = object.__new__(BadCharacters)
    native.perturbation_budget, native.max_position_candidates, native.max_ascii_variants = 2, 3, 2
    expected = list(native._generate_deletion_variants(base.text))
    assert len(actual) == len(expected) == 18
    assert [candidate.request.text for candidate in actual[:6]] == [
        " \bπ\n", "~\bπ\n", "π \b\n", "π~\b\n", "π\n \b", "π\n~\b",
    ]
    assert actual[6].request.text == " \bπ \b\n"
    assert actual[-1].request.text == "π~\b\n~\b"
    assert actual[6].cause.metadata == {
        "bad_character_category": "deletion", "count": 2,
        "operations": {"positions": [0, 1], "ascii_codes": [32, 32]},
    }
    for candidate, (text, details) in zip(actual, expected, strict=True):
        assert candidate.request.text == text
        assert candidate.cause.metadata == {"bad_character_category": "deletion", **details}
        assert candidate.cause.attempt_id == candidate.request.id
        assert candidate.request.origin is base
    assert len({candidate.request.id for candidate in actual}) == 18


@pytest.mark.parametrize("text", ["", "a", "π🙂", "line one\r\nline two\n", "0123456789abcdef"])
def test_default_profile_spreads_positions_and_characters_without_normalization(text: str) -> None:
    actual = list(candidates(Message(text), profile=DeletionProfile()))
    count = min(len(text) + 1, 8)
    positions = list(range(len(text) + 1)) if count < 8 else [round(index * len(text) / 7) for index in range(8)]
    expected = [(position, code) for position in positions for code in (32, 63, 94, 125)]
    assert len(actual) == count * 4
    for candidate, (position, code) in zip(actual, expected, strict=True):
        assert candidate.request.text == text[:position] + chr(code) + "\x08" + text[position:]
        assert candidate.cause.metadata == {
            "bad_character_category": "deletion", "count": 1,
            "operations": {"positions": [position], "ascii_codes": [code]},
        }


def test_generator_never_constructs_a_probe_or_changes_global_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    config, plugins = import_module("garak._config"), import_module("garak._plugins")
    state = random.getstate()
    configuration = deepcopy([cast(dict[str, object], vars(item)) for item in (config.run, config.system, config.plugins)])
    reportfile = config.transient.reportfile
    registry = deepcopy(plugins.PluginProvider._instance_cache)

    def forbidden_init(self: BadCharacters, config_root: object = None) -> None:
        raise AssertionError("The bundled-corpus constructor must not run")

    monkeypatch.setattr(BadCharacters, "__init__", forbidden_init)
    first = list(candidates(Message("My own base"), profile=DeletionProfile()))
    second = list(candidates(Message("My own base"), profile=DeletionProfile()))
    assert [candidate.request.text for candidate in first] == [candidate.request.text for candidate in second]
    assert random.getstate() == state
    assert [vars(config.run), vars(config.system), vars(config.plugins)] == configuration
    assert config.transient.reportfile is reportfile
    assert plugins.PluginProvider._instance_cache == registry


@pytest.mark.parametrize("success_at", [2, None])
def test_real_variants_reach_target_and_final_evaluation_uses_original_base(tmp_path: Path, success_at: int | None) -> None:
    base = Message("old").transform(lambda _: "Explain {this} π\r\n")
    model, evaluator = ScriptedModel([f"Response {index}" for index in range(1, 33)]), PairEvaluator(success_at)
    target = LLMChat(model, system="Keep system")
    system = target.history()[0].message
    user = BadCharactersDeletionExperiment(target, base, evaluator, journal_dir=tmp_path)
    assert not model.requests and not evaluator.pairs
    user.start()
    assert user.wait(10)
    count = success_at or 32
    assert user.status().state is UserState.COMPLETED, user.status().reason
    assert user.status().reason == ("evaluator_success" if success_at else "strategy_exhausted")
    assert len(model.requests) == len(evaluator.pairs) == count
    assert all(len(turns) == 2 and turns[0].message is system for turns in model.requests)
    assert [turns[-1].message.text for turns in model.requests] == [item.request.text for item in candidates(base, profile=DeletionProfile())][:count]
    assert all(pair[0] is base for pair in evaluator.pairs)
    assert evaluator.pairs[-1][1] is user.result() is target.history()[-1].message
    assert not user.status().progress.attacker_calls
    assert user.status().progress.max_calls_per_conv is None
    _assert_artifacts(user, tmp_path, count, success_at is not None)


def _assert_artifacts(user: BadCharactersDeletionExperiment, root: Path, count: int, success: bool) -> None:
    directory = save_run(capture_run(user, journal=user.journal), root)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert (directory / "result.txt").read_text() == f"Response {count}"
    assert metadata["evaluation"] == {"result_id": metadata["result_id"], "success": success}
    settings = metadata["settings"]
    assert settings["attack"] == {"type": "badchars_deletion"}
    assert settings["perturbation_budget"] == 1 and settings["max_position_candidates"] == 8 and settings["max_ascii_variants"] == 4
    assert settings["category"] == "deletion" and settings["follow_prompt_cap"] is False
    assert settings["early_stop"] == "evaluator_success" and "Attacker" not in settings
    assert settings["garak_revision"] == "3f50ea5ff9cd7050099940647c15c39b07a93392"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    started = [event["data"] for event in events if event["kind"] == "call_started"]
    assert len(started) == len({event["conversation_id"] for event in started}) == count
    for event, candidate in zip(started, candidates(user.base, profile=DeletionProfile())):
        assert event["input"]["text"] == candidate.request.text and "\x08" in event["input"]["text"]
        assert event["input"]["origin"] == user.base.snapshot()
        assert event["context"]["cause"]["metadata"] == candidate.cause.metadata
        assert event["attempt_id"] == event["context"]["cause"]["attempt_id"] == event["input"]["id"]
    assert user.journal.error is None and user.journal.finished


@pytest.mark.parametrize("user_type", [BadCharactersDeletionUser, BadCharactersDeletionExperiment])
def test_python_api_accepts_custom_generator_parameters(user_type: type[BadCharactersDeletionUser] | type[BadCharactersDeletionExperiment]) -> None:
    model = ScriptedModel(["negative"] * 3)
    user = user_type(LLMChat(model), Message("π\n"), ContainsTextEvaluator("yes"),
                     perturbation_budget=2, max_position_candidates=2, max_ascii_variants=1)
    user.start()
    assert user.wait(10)
    assert user.status().reason == "strategy_exhausted"
    assert [turns[-1].message.text for turns in model.requests] == [" \bπ\n", "π\n \b", " \bπ\n \b"]


def test_native_generator_failure_is_a_saved_worker_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(self: BadCharacters, payload: str) -> Iterator[tuple[str, dict[str, object]]]:
        raise RuntimeError("native generator failed")

    monkeypatch.setattr(BadCharacters, "_generate_deletion_variants", fail)
    model = ScriptedModel([])
    user = BadCharactersDeletionExperiment(LLMChat(model), Message("Base"), ContainsTextEvaluator("yes"))
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.FAILED and not model.requests
    assert user.journal.finished and save_run(capture_run(user, journal=user.journal), tmp_path).is_dir()


@pytest.mark.parametrize("name", ["perturbation_budget", "max_position_candidates", "max_ascii_variants"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1", None])
def test_python_parameters_reject_invalid_values_before_generation(name: str, value: object) -> None:
    options = cast(dict[str, int], {name: value})
    with pytest.raises(ValueError, match=name):
        BadCharactersDeletionUser(LLMChat(str), Message("Base"), ContainsTextEvaluator("yes"), **options)

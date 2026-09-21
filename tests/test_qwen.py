"""Public local-chat contracts, using controlled libraries without weights or CUDA."""

import json
from collections.abc import Sequence
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from threading import Event, Thread
from typing import NoReturn

import pytest
from chats_assertions import at, object_at
from qwen_fakes import LocalFixture, LocalLibraries, local

from tai import (
    ChatCancelled,
    ChatTurn,
    ContainsTextEvaluator,
    Message,
    QwenChat,
    QwenContextError,
    UserState,
    capture_run,
    save_run,
)
from tai import (
    FITDExperiment as FITDUser,
)
from core.chats.qwen_contracts import StoppingCriterion

__all__ = ["local"]


def _release_worker(release: Event, worker: Thread) -> None:
    release.set()
    worker.join(3)


def _decode_until_cancel(entered: Event, criteria: Sequence[StoppingCriterion]) -> None:
    entered.set()
    for _ in range(300):
        if criteria[0]():
            return
        Event().wait(0.01)
    pytest.fail("generation did not observe cancellation")


def _assert_cancelled(outcomes: list[Message | RuntimeError], chat: QwenChat) -> None:
    assert isinstance(outcomes[0], ChatCancelled)
    assert outcomes[0].response is None
    assert not chat.history()


def _assert_reset_and_settings(
    target: QwenChat, branch: object, lib: LocalLibraries, snapshot: Path,
) -> None:
    old_id = target.conversation_id
    target.reset()
    assert target.conversation_id != old_id
    with pytest.raises(ValueError, match="stale"):
        target.restore(branch)
    target.send(Message("new conversation"))
    assert len(lib.records.requests[-1]) == 2
    assert lib.records.loads == [("tokenizer", str(snapshot)), ("model", str(snapshot))]
    assert getattr(lib.records.generations[0][1]["generation_config"], "do_sample") is False
    assert getattr(lib.records.generations[3][1]["generation_config"], "temperature") == 0.7
    assert at(target.settings(), "actual_dtype") == "float16"
    settings = target.settings()
    object_at(settings, "generation")["max_new_tokens"] = 1
    assert at(target.settings(), "generation", "max_new_tokens") == 512

def test_roles_history_provenance_and_branching_with_one_lazy_load(local: LocalFixture) -> None:
    lib, kwargs, snapshot = local
    target = QwenChat(**kwargs, system="Target instructions")
    attacker = QwenChat(
        **kwargs,
        system="Attacker instructions",
        generation={"do_sample": True, "temperature": 0.7},
    )
    assert at(target.settings(), "loaded") is False
    assert at(attacker.settings(), "actual_dtype") is None
    assert not lib.records.loads and not lib.records.imports
    original = Message("first π\n")
    point = target.checkpoint()
    response = target.send(original)
    assert response.to_string() == "yes"
    assert json.loads(response.dump())["origin"]["id"] == original.id
    branch = target.checkpoint()
    target.send(Message("discard"))
    target.restore(branch)
    target.send(Message("retained"))
    attacker.send(Message("separate"))
    assert lib.records.requests[2] == [
        {"role": "system", "content": "Target instructions"},
        {"role": "user", "content": "first π\n"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "retained"},
    ]
    assert lib.records.requests[3] == [
        {"role": "system", "content": "Attacker instructions"},
        {"role": "user", "content": "separate"},
    ]
    assert target.conversation_id != attacker.conversation_id
    with pytest.raises(ValueError, match="Foreign"):
        attacker.restore(point)
    target.restore(point)
    target.send(Message("restored"))
    _assert_reset_and_settings(target, branch, lib, snapshot)


@pytest.mark.parametrize("override", [{"device": "cuda:1"}, {"dtype": "float32"}])
def test_distinct_device_or_dtype_requires_distinct_weights(
    local: LocalFixture, override: dict[str, str],
) -> None:
    lib, kwargs, _ = local
    target = QwenChat(**kwargs)
    attacker_options = kwargs.copy()
    attacker_options["device"] = override.get("device", kwargs["device"])
    attacker = QwenChat(**attacker_options, dtype=override.get("dtype", "float16"))
    target.send(Message("one"))
    attacker.send(Message("two"))
    assert len(lib.records.loads) == 4


def in_thread(chat: QwenChat, outcomes: list[Message | RuntimeError]) -> Thread:
    def send() -> None:
        try:
            outcomes.append(chat.send(Message("request")))
        except RuntimeError as exc:
            outcomes.append(exc)

    worker = Thread(target=send)
    worker.start()
    return worker


def test_cancel_waiter_does_not_cancel_active_role_and_serializes_generation(
    local: LocalFixture,
) -> None:
    lib, kwargs, _ = local
    target, attacker = QwenChat(**kwargs), QwenChat(**kwargs)
    entered, release, waiting = Event(), Event(), Event()

    def hold(criteria: Sequence[StoppingCriterion]) -> None:
        entered.set()
        assert release.wait(3)
        assert not criteria[0]()

    lib.actions.generate = hold
    active: list[Message | RuntimeError] = []
    queued: list[Message | RuntimeError] = []
    first = in_thread(target, active)
    with ExitStack() as cleanup:
        cleanup.callback(_release_worker, release, first)
        assert entered.wait(2)
        # Signal when send has entered its generator, before waiting for weights.
        original = attacker._generate

        def generate(turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
            waiting.set()
            return original(turns, cancellation)

        attacker._generate = generate
        second = in_thread(attacker, queued)
        assert waiting.wait(1)
        assert not attacker.history()
        assert len(lib.records.generations) == 1
        attacker.cancel()
        second.join(1)
        assert not second.is_alive()
        _assert_cancelled(queued, attacker)
        assert first.is_alive()
    assert isinstance(active[0], Message)
    lib.actions.generate = lambda criteria: None
    attacker.send(Message("after cancellation"))
    assert len(lib.records.loads) == 2


def test_active_cancellation_stops_decoding_and_waits_for_cuda_cleanup(local: LocalFixture) -> None:
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)
    entered, cleaning, release = Event(), Event(), Event()

    def cleanup() -> None:
        if entered.is_set():
            cleaning.set()
            assert release.wait(3)

    lib.actions.generate = partial(_decode_until_cancel, entered)
    lib.actions.sync = cleanup
    outcomes: list[Message | RuntimeError] = []
    worker = in_thread(chat, outcomes)
    with ExitStack() as cleanup_stack:
        cleanup_stack.callback(_release_worker, release, worker)
        assert entered.wait(2)
        chat.cancel()
        assert cleaning.wait(2)
        assert worker.is_alive() and not outcomes
        assert not chat.history()
        with pytest.raises(RuntimeError, match="active exchange"):
            chat.reset()
    _assert_cancelled(outcomes, chat)
    assert not lib.records.decoded and not chat.history()
    lib.actions.generate = lambda criteria: None
    chat.send(Message("retry"))
    assert len(lib.records.loads) == 2


def test_precancelled_exchange_does_not_import_or_load(local: LocalFixture) -> None:
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)
    cancellation = Event()
    cancellation.set()
    with pytest.raises(ChatCancelled):
        chat.send(Message("cancelled"), cancellation=cancellation)
    assert not lib.records.imports and not lib.records.loads and not chat.history()


def test_loading_is_background_work_and_user_stop_waits_for_it(
    local: LocalFixture, tmp_path: Path,
) -> None:
    lib, kwargs, _ = local
    entered, release = Event(), Event()

    def loading() -> None:
        entered.set()
        assert release.wait(3)

    lib.actions.load = loading
    target, attacker = QwenChat(**kwargs), QwenChat(**kwargs)
    user = FITDUser(
        target,
        attacker,
        Message("puzzle"),
        ContainsTextEvaluator("key"),
        journal_dir=tmp_path / "runs",
        max_calls_per_conv=1,
    )
    user.start()
    with ExitStack() as cleanup:
        cleanup.callback(_finish_user, release, user)
        assert entered.wait(2)
        user.stop()
        assert user.status().state is UserState.STOPPING
        assert user.journal.events() and user.result() is None
        assert not user.wait(0.01)
        assert not lib.records.generations
    assert user.status().state is UserState.STOPPED
    assert not target.history() and len(attacker.history()) == 2
    assert len(lib.records.generations) == 1
    assert user.status().progress.attacker_calls == 1
    # Loading and the admitted generation finish; later runs reuse the weights.
    _assert_repeated_load(user, lib, tmp_path)


def _finish_user(release: Event, user: FITDUser) -> None:
    release.set()
    assert user.wait(3)


def _assert_repeated_load(user: FITDUser, lib: LocalLibraries, tmp_path: Path) -> None:
    lib.actions.load = lambda: None
    user.start()
    assert user.wait(3)
    assert user.status().state is UserState.COMPLETED
    assert user.result() is not None
    loads = len(lib.records.loads)
    user.start()
    assert user.wait(3)
    assert user.status().state is UserState.COMPLETED
    assert len(lib.records.loads) == loads
    directory = save_run(capture_run(user, journal=user.journal), tmp_path / "runs")
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["final_settings"]["Target"]["actual_dtype"] == "float16"
    result = user.result()
    assert result is not None
    assert (directory / "result.txt").read_text(encoding="utf-8") == result.to_string()


@pytest.mark.parametrize("phase", ["load", "generate"])
def test_library_errors_remain_errors_and_do_not_complete_exchange(
    local: LocalFixture, phase: str,
) -> None:
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)

    def fail(*_args: object) -> NoReturn:
        raise RuntimeError("CUDA out of memory (controlled)")

    if phase == "load":
        lib.actions.load = fail
    else:
        lib.actions.generate = fail
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        chat.send(Message("request"))
    assert not chat.history()
    lib.actions.load = lambda: None
    lib.actions.generate = lambda criteria: None
    assert chat.send(Message("retry")).to_string() == "yes"


def test_context_window_is_checked_without_truncation(local: LocalFixture) -> None:
    lib, kwargs, _ = local
    lib.input_length = 32700
    chat = QwenChat(**kwargs)
    with pytest.raises(QwenContextError, match="input=32700.*limit=32768"):
        chat.send(Message("entire long message"))
    assert lib.records.requests[0][-1]["content"] == "entire long message"
    assert not lib.records.generations and not chat.history()

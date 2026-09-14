from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tai import ChatCancelled, LLMChat, Message


def test_chats_keep_complete_exchanges_and_system_instructions_independently():
    requests = []

    def generate(turns):
        requests.append(turns)
        return f"Answer {len(requests)}"

    target = LLMChat(generate, model="shared-model", system="Target instructions")
    attacker = LLMChat(generate, model="shared-model", system="Attacker instructions")
    first = target.send(Message("First request"))
    attacker.send(Message("Separate request"))
    target.send(Message("Follow up"))

    assert first.to_string() == "Answer 1"
    assert [(turn.role, turn.message.to_string()) for turn in requests[2]] == [
        ("system", "Target instructions"),
        ("user", "First request"),
        ("assistant", "Answer 1"),
        ("user", "Follow up"),
    ]
    assert [(turn.role, turn.message.to_string()) for turn in attacker.history()] == [
        ("system", "Attacker instructions"),
        ("user", "Separate request"),
        ("assistant", "Answer 2"),
    ]


def test_cancel_releases_generation_without_committing_an_exchange():
    entered = Event()
    requests = []

    def generate(turns, cancellation):
        requests.append(turns)
        if len(requests) == 2:
            entered.set()
            assert cancellation.wait(5), "generation did not receive cancellation"
            return "Observed but cancelled"
        assert not cancellation.is_set()
        return "Complete"

    chat = LLMChat.cancellable(generate, system="Instructions")
    assert chat.cancel() is None
    chat.send(Message("First"))
    history = chat.history()
    with ThreadPoolExecutor() as worker:
        sending = worker.submit(chat.send, Message("Interrupted"))
        try:
            assert entered.wait(5)
            assert chat.history() == history
            with pytest.raises(RuntimeError, match="active exchange"):
                chat.send(Message("Overlapping"))
            assert chat.cancel() is None
            assert chat.cancel() is None
            with pytest.raises(ChatCancelled, match="cancelled") as caught:
                sending.result(timeout=1)
            assert caught.value.response.to_string() == "Observed but cancelled"
        finally:
            chat.cancel()
    assert chat.history() == history
    assert chat.cancel() is None
    chat.send(Message("Next"))
    assert requests[-1][:-1] == history
    assert len(chat.history()) == len(history) + 2


def test_cancelled_owner_prevents_admission_to_a_new_exchange():
    requests = []
    cancellation = Event()
    chat = LLMChat(lambda turns: requests.append(turns) or "Response")
    cancellation.set()
    with pytest.raises(ChatCancelled):
        chat.send(Message("Too late"), cancellation=cancellation)
    assert requests == []
    assert chat.history() == ()
    # The cancelled owner's signal does not poison independent exchanges.
    assert chat.send(Message("Independent")).to_string() == "Response"


def test_checkpoints_restore_immutable_branches_and_reset_preserves_system():
    requests = []
    chat = LLMChat(
        lambda turns: requests.append(turns) or "Puzzle answer", system="Puzzle rules"
    )
    initial = chat.checkpoint()
    chat.send(Message("First clue"))
    first_history = chat.history()
    first = chat.checkpoint()
    chat.send(Message("Discarded clue"))
    discarded_history = chat.history()
    discarded = chat.checkpoint()

    assert chat.restore(first) is None
    chat.send(Message("Alternative clue"))
    assert requests[-1][:-1] == first_history
    assert chat.restore(discarded) is None
    assert chat.history() == discarded_history
    assert chat.restore(initial) is None
    assert chat.history() == first_history[:1]
    # Branching and reading do not mutate snapshots or execute the model again.
    assert len(requests) == 3
    assert chat.restore(first) is None
    assert chat.history() == first_history

    assert chat.reset() is None
    assert chat.history() == first_history[:1]
    for invalid in (initial, first, discarded, object(), [], None):
        with pytest.raises(ValueError, match="checkpoint"):
            chat.restore(invalid)
    foreign = LLMChat(lambda turns: "other").checkpoint()
    with pytest.raises(ValueError, match="checkpoint"):
        chat.restore(foreign)
    current = chat.checkpoint()
    chat.send(Message("New puzzle"))
    assert [(t.role, t.message.to_string()) for t in requests[-1]] == [
        ("system", "Puzzle rules"),
        ("user", "New puzzle"),
    ]
    assert chat.restore(current) is None
    assert chat.history() == first_history[:1]


@pytest.mark.parametrize("finish", ["complete", "cancel", "fail"])
def test_history_controls_cannot_overwrite_a_send_or_its_cancellation_cleanup(finish):
    entered, release = Event(), Event()
    requests = []

    def generate(turns, cancellation):
        requests.append(turns)
        if len(requests) == 1:
            return "First answer"
        entered.set()
        assert release.wait(5), "test did not release generation"
        if finish == "fail":
            raise ValueError("model failed")
        return "Second answer"

    chat = LLMChat.cancellable(generate, system="Stable instructions")
    initial = chat.checkpoint()
    chat.send(Message("First clue"))
    history = chat.history()
    with ThreadPoolExecutor() as worker:
        sending = worker.submit(chat.send, Message("Held clue"))
        try:
            assert entered.wait(5)
            completed = chat.checkpoint()
            if finish == "cancel":
                chat.cancel()
            for mutate in (chat.reset, lambda: chat.restore(initial)):
                with pytest.raises(RuntimeError, match="active exchange"):
                    mutate()
                assert chat.history() == history
        finally:
            release.set()
        if finish == "complete":
            assert sending.result(timeout=1).to_string() == "Second answer"
            assert len(chat.history()) == len(history) + 2
        else:
            expected = ChatCancelled if finish == "cancel" else ValueError
            with pytest.raises(expected):
                sending.result(timeout=1)
            assert chat.history() == history
    # A checkpoint taken during send contains only completed exchanges. It and
    # the initial snapshot survive rejected attempts at reset and restore.
    chat.restore(completed)
    assert chat.history() == history
    chat.restore(initial)
    assert chat.history() == history[:1]
    chat.reset()
    with pytest.raises(ValueError, match="checkpoint"):
        chat.restore(completed)


def test_empty_system_and_foreign_identical_history_do_not_share_checkpoints():
    first = LLMChat(lambda turns: "answer")
    other = LLMChat(lambda turns: "answer")
    first.send(Message("same"))
    other.send(Message("same"))
    point = first.checkpoint()
    with pytest.raises((AttributeError, TypeError)):
        point.turns = ()
    before = other.history()
    with pytest.raises(ValueError, match="checkpoint"):
        other.restore(point)
    assert other.history() == before
    first.reset()
    assert first.history() == ()

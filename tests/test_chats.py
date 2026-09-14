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

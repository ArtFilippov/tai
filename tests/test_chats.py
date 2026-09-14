from tai import LLMChat, Message


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

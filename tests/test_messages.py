import json
from dataclasses import FrozenInstanceError

import pytest

from tai import Message


def test_message_is_an_immutable_serializable_text_with_its_own_identity():
    message = Message("A supplied base request")

    assert message.base() == "A supplied base request"
    assert message.to_string() == "A supplied base request"
    assert json.loads(message.dump()) == {
        "id": message.id,
        "base": "A supplied base request",
        "text": "A supplied base request",
    }
    assert Message("A supplied base request").id != message.id
    with pytest.raises(FrozenInstanceError):
        message.text = "changed"

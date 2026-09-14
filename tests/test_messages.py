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
        "origin": None,
        "transformations": [],
    }
    assert Message("A supplied base request").id != message.id
    with pytest.raises(FrozenInstanceError):
        message.text = "changed"
    with pytest.raises(FrozenInstanceError):
        message.id = "changed"


def test_transformations_branch_from_current_text_without_changing_earlier_versions():
    root = Message("Puzzle")

    def surround(text, prefix, *, suffix):
        return prefix + text + suffix

    first = root.transform(surround, "[", suffix="]")
    left = first.transform(str.upper)
    right = first.transform(str.replace, "Puzzle", "Answer")

    assert [m.to_string() for m in (root, first, left, right)] == [
        "Puzzle",
        "[Puzzle]",
        "[PUZZLE]",
        "[Answer]",
    ]
    assert all(m.base() == "Puzzle" for m in (root, first, left, right))
    assert len({m.id for m in (root, first, left, right)}) == 4
    with pytest.raises(FrozenInstanceError):
        left.text = "changed"


def test_revert_selects_computed_versions_and_can_start_a_new_branch():
    calls = []

    def append_step(text, step):
        calls.append(step)
        return text + step

    root = Message("start")
    first = root.transform(append_step, " one")
    second = first.transform(append_step, " two")
    assert second.revert(0) is second
    assert second.revert(1) is first
    assert second.revert(2) is root
    assert root.revert(0) is root
    branch = second.revert(1).transform(append_step, " other")
    assert branch.to_string() == "start one other"
    assert second.to_string() == "start one two"
    assert branch.revert(1) is first
    for message, distance in [(root, 1), (second, -1), (second, 3)]:
        with pytest.raises(ValueError):
            message.revert(distance)
    assert calls == [" one", " two", " other"]


def test_spawn_starts_a_new_chain_with_a_snapshot_of_its_origin():
    source = Message("Original request").transform(str.upper)
    spawned = source.spawn("Independent answer")
    edited = spawned.transform(str.replace, "answer", "result")

    assert spawned.base() == "Independent answer"
    assert edited.base() == "Independent answer"
    assert edited.to_string() == "Independent result"
    assert spawned.id != source.id
    assert edited.revert(1) is spawned
    with pytest.raises(ValueError):
        spawned.revert(1)
    with pytest.raises(ValueError):
        edited.revert(2)
    assert json.loads(edited.dump()) == {
        "id": edited.id,
        "base": "Independent answer",
        "text": "Independent result",
        "origin": {
            "id": source.id,
            "base": "Original request",
            "text": "ORIGINAL REQUEST",
            "origin": None,
            "transformations": [
                {
                    "id": source.id,
                    "source_id": source.revert(1).id,
                    "text": "ORIGINAL REQUEST",
                }
            ],
        },
        "transformations": [
            {"id": edited.id, "source_id": spawned.id, "text": "Independent result"}
        ],
    }


@pytest.mark.parametrize("invalid", [None, 42, b"bytes", ["text"]])
def test_non_text_values_are_rejected_without_changing_existing_history(invalid):
    source = Message("original").transform(str.upper)
    snapshot = source.dump()
    with pytest.raises(TypeError):
        Message(invalid)
    with pytest.raises(TypeError):
        source.transform(lambda text: invalid)
    with pytest.raises(TypeError):
        source.spawn(invalid)
    assert source.dump() == snapshot
    assert source.revert(1).to_string() == "original"


def test_failed_transformation_propagates_its_error_and_leaves_source_usable():
    source = Message("original").transform(str.upper)
    snapshot = source.dump()
    failure = RuntimeError("transformation failed")

    def fail(text):
        raise failure

    with pytest.raises(RuntimeError) as raised:
        source.transform(fail)
    assert raised.value is failure
    assert source.dump() == snapshot
    assert source.transform(str.lower).to_string() == "original"


def test_snapshots_preserve_computed_branches_without_reexecuting_functions(tmp_path):
    values = iter([" первая", " вторая"])

    def consume(text, values):
        return text + next(values)

    root = Message("Версия")
    first = root.transform(consume, values)
    second = first.transform(consume, values)
    spawned = second.spawn("").transform(str.upper)
    # The callable and its iterator cannot be replayed or serialized as JSON.
    # Saving, reading, branching, and reverting use only their computed texts.
    snapshot = second.dump()
    path = tmp_path / "message.json"
    path.write_text(spawned.dump(), encoding="utf-8")
    other = first.transform(str.replace, "первая", "другая")
    assert second.revert(1) is first
    assert second.revert(2) is root
    assert second.dump() == snapshot
    assert json.loads(snapshot)["transformations"] == [
        {"id": first.id, "source_id": root.id, "text": "Версия первая"},
        {"id": second.id, "source_id": first.id, "text": "Версия первая вторая"},
    ]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["base"] == saved["text"] == ""
    assert saved["origin"] == json.loads(snapshot)
    assert saved["transformations"][0]["text"] == ""
    saved["origin"]["transformations"][0]["text"] = "changed detached snapshot"
    assert second.dump() == snapshot
    assert first.to_string() == "Версия первая"
    assert other.to_string() == "Версия другая"
    assert spawned.spawn("child").base() == "child"
    assert json.loads(spawned.spawn("child").dump())["origin"]["origin"] == json.loads(
        snapshot
    )

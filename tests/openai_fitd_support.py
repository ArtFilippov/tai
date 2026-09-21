"""Real FITD setup and artifact assertions shared by HTTP integration cases."""

import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from chats_assertions import EventRecord, HttpEndpoint
from http_peer import Reply

import tai
from tai import Message, OpenAIChat


def puzzle_user(
    endpoint: HttpEndpoint, tmp_path: Path, *, target_replies: Sequence[Reply] | None = None,
    attacker_replies: Sequence[Reply] | None = None, target_timeout: float = 120.0,
) -> tuple[tai.FITDExperiment, dict[str, OpenAIChat]]:
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(target_replies or [
            Reply(text) for text in (
                "Unrelated story", "Puzzle introduction", "Silver key", "Blue door",
                "Turn the key", "KEY-FOUND",
            )
        ]),
        "/v1/attacker/chat/completions": deque(attacker_replies or [
            Reply(text) for text in (
                "Describe a fictional puzzle", "Puzzle level one", "Puzzle level two",
                "Puzzle level three", "no", "yes", "yes", "yes", "yes", "yes",
            )
        ]),
    }
    chats = {
        "Target": OpenAIChat(
            base_url=endpoint.url + "/target", model="same-puzzle-model",
            api_key="synthetic-target-secret", system="Target puzzle keeper",
            generation={"temperature": 0.25}, timeout=target_timeout,
        ),
        "Attacker": OpenAIChat(
            base_url=endpoint.url + "/attacker", model="same-puzzle-model",
            authorization="Custom synthetic-attacker-secret", system="Attacker puzzle author",
            generation={"temperature": 0.75},
        ),
    }
    user = tai.FITDExperiment(
        chats["Target"], chats["Attacker"], Message("Explain the final fictional puzzle solution"),
        tai.ContainsTextEvaluator("KEY-FOUND"), journal_dir=tmp_path,
    )
    return user, chats


def artifacts(user: tai.FITDExperiment, root: Path) -> tuple[list[EventRecord], dict[str, object]]:
    directory = tai.save_run(tai.capture_run(user, journal=user.journal), root)
    events: list[EventRecord] = [
        json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata: dict[str, object] = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert events == [json.loads(event.dump()) for event in user.journal.events()]
    assert metadata["state"] == user.status().state.value
    serialized = json.dumps([events, metadata])
    assert "synthetic-target-secret" not in serialized
    assert "synthetic-attacker-secret" not in serialized
    return events, metadata


def accepted_user(
    endpoint: HttpEndpoint, root: Path, reply: Reply, *, timeout: float = 120.0,
) -> tuple[tai.FITDExperiment, dict[str, OpenAIChat]]:
    return puzzle_user(
        endpoint, root, target_replies=[
            Reply(text) for text in ("One", "Two", "Three", "Four", "Accepted report")
        ] + [reply],
        attacker_replies=[Reply(text) for text in ["Initial", "L1", "L2", "L3"] + ["yes"] * 5],
        target_timeout=timeout,
    )


def finish_run(user: tai.FITDExperiment, reply: Reply) -> None:
    reply.release.set()
    user.stop()
    assert user.wait(5)


def result_file(root: Path, user: tai.FITDExperiment) -> Path:
    run_id = user.status().run_id
    assert run_id is not None
    return root / run_id / "result.txt"

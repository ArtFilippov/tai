"""Typed checks for a finished run's saved, detached evidence."""

import json
from pathlib import Path

from journal_assertions import at, mapping, run_directory

from tai import FITDExperiment, capture_run, save_run


def read_and_assert_final_artifacts(
    user: FITDExperiment, root: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    save_run(capture_run(user, journal=user.journal), root)
    run_dir = run_directory(user, root)
    events = [
        mapping(json.loads(line))
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata = mapping(json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")))
    assert events == [json.loads(event.dump()) for event in user.journal.events()]
    assert events[-1]["kind"] == "run_finished"
    assert metadata["state"] == user.status().state.value
    assert at(events[-1], "data", "state") == metadata["state"]
    assert at(events[-1], "data", "reason") == metadata["reason"] == user.status().reason
    if (result := user.result()) is None:
        assert metadata["result"] is None
        assert metadata["result_id"] is None
        assert not (run_dir / "result.txt").exists()
    else:
        assert metadata["result"] == json.loads(result.dump())
        assert metadata["result_id"] == result.id
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == result.to_string()
    return events, metadata

"""Verify saved evidence without inference and optionally publish a small full archive."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from attacks.adaptive_attacks.garak_adapter import variants
from attacks.badchars.garak_adapter import candidates
from attacks.badchars.profile import DeletionProfile
from core.primitives.messages.message.text_message import TextMessage


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def selected(events, kind):
    return [event["data"] for event in events if event["kind"] == kind]


def check_settings(metadata):
    settings = metadata["settings"]
    assert "Attacker" not in settings and "Attacker" not in metadata["final_settings"]
    assert settings["garak_revision"] == "3f50ea5ff9cd7050099940647c15c39b07a93392"
    for role in ("Target", "Evaluator"):
        configured, final = settings[role], metadata["final_settings"][role]
        assert configured["backend"] == "qwen" and configured["device"] == "cuda:0"
        assert configured["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
        assert configured["dtype"] == final["actual_dtype"] == "float16"
        assert final["loaded"] and final["local_files_only"]
    assert settings["Target"]["conversation_id"] != settings["Evaluator"]["conversation_id"]


def check_calls(events, metadata):
    starts, completed = selected(events, "call_started"), selected(events, "call_completed")
    assert {item["call_id"] for item in starts} == {item["call_id"] for item in completed}
    assert len({item["conversation_id"] for item in starts}) == len(starts)
    resets = {item["conversation_id"] for item in selected(events, "history_reset")}
    assert all(item["conversation_id"] in resets for item in starts)
    counts = Counter(item["role"] for item in completed)
    assert set(counts) == {"Target", "Evaluator"}
    for item in starts:
        if item["role"] == "Evaluator":
            assert all(turn["role"] == "system" for turn in item["settings"]["initial_history"])
    assert counts["Target"] == metadata["progress"]["target_calls"]
    assert counts["Evaluator"] == metadata["progress"]["evaluator_calls"]
    base = TextMessage(metadata["settings"]["base"]["text"])
    strategy = metadata["settings"]["attack"]["type"]
    expected = variants(base) if strategy == "adaptive_attacks" else tuple(
        item.request for item in candidates(base, profile=DeletionProfile()))
    targets = [item for item in starts if item["role"] == "Target"]
    assert [item["input"]["text"] for item in targets] == [item.text for item in expected[:len(targets)]]
    assert len(expected) == (5 if strategy == "adaptive_attacks" else 32)
    for item in targets:
        assert item["input"]["origin"]["id"] == metadata["settings"]["base"]["id"]
        if strategy == "badchars_deletion":
            assert item["input"]["text"].count("\b") == 1
            operations = item["context"]["cause"]["metadata"]["operations"]
            position, = operations["positions"]
            code, = operations["ascii_codes"]
            assert item["input"]["text"] == base.text[:position] + chr(code) + "\b" + base.text[position:]
    return dict(counts), len(expected)


def check_evaluations(events, metadata):
    starts = selected(events, "evaluation_started")
    completions = selected(events, "evaluation_completed")
    responses = {item["response"]["id"]: item["response"] for item in selected(events, "call_completed") if item["role"] == "Target"}
    for item in starts:
        assert item["base_id"] == metadata["settings"]["base"]["id"]
        assert item["result"] == responses[item["result_id"]]
    if metadata["state"] == "COMPLETED":
        assert len(starts) == len(completions)
        raw = [item["response"]["text"] for item in selected(events, "call_completed") if item["role"] == "Evaluator"]
        assert [json.loads(text) for text in raw] == [{"success": item["success"]} for item in completions]
        last = completions[-1]
        assert last["result_id"] == metadata["result_id"]
        assert type(last["success"]) is bool
        assert metadata["evaluation"] == {"result_id": last["result_id"], "success": last["success"]}
        assert all(item["success"] is False for item in completions[:-1])
        if last["success"]:
            success = next(event["sequence"] for event in events if event["kind"] == "evaluation_completed" and event["data"]["success"])
            assert not any(event["kind"] == "call_started" and event["sequence"] > success for event in events)
            assert metadata["reason"] == "evaluator_success"
        else:
            assert metadata["reason"] == "strategy_exhausted"
    else:
        assert metadata["state"] == "FAILED" and "ModelVerdictError" in metadata["reason"]
        assert metadata["evaluation"] is None and selected(events, "evaluation_failed")
    return [item["success"] for item in completions]


def check_run(root, strategy):
    directory, = (root / strategy).iterdir()
    metadata = load(directory / "metadata.json")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(event["run_id"] == metadata["run_id"] == directory.name for event in events)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["kind"] == "run_finished" and selected(events, "resources_released")
    assert events[-1]["data"]["state"] == metadata["state"]
    assert events[-1]["data"]["reason"] == metadata["reason"]
    assert metadata["result_id"] == metadata["result"]["id"]
    assert (directory / "result.txt").read_bytes() == metadata["result"]["text"].encode("utf-8")
    check_settings(metadata)
    counts, available = check_calls(events, metadata)
    verdicts = check_evaluations(events, metadata)
    if metadata["reason"] == "strategy_exhausted":
        assert counts["Target"] == available
    command = load(root / f"{strategy}-attack/command.json")
    assert command["exit_code"] == (0 if metadata["state"] == "COMPLETED" else 1)
    return {
        "directory": f"{root.name}/{strategy}/{directory.name}",
        "run_id": metadata["run_id"], "state": metadata["state"], "reason": metadata["reason"],
        "result_id": metadata["result_id"], "evaluation": metadata["evaluation"],
        "available_variants": available, "calls": counts, "verdicts": verdicts,
        "raw_verdicts": [item["response"]["text"] for item in selected(events, "call_completed") if item["role"] == "Evaluator"],
        "exit_code": command["exit_code"], "cli_seconds": command["duration_seconds"],
        "worker_seconds": (datetime.fromisoformat(events[-1]["timestamp"]) - datetime.fromisoformat(events[0]["timestamp"])).total_seconds(),
        "conversation_ids": [item["conversation_id"] for item in selected(events, "call_started")],
        "result_sha256": digest(directory / "result.txt"), "verified": True,
    }


def check_attempt(root):
    environment = load(root / "environment/stdout.txt")
    assert environment["interfaces"] == [[1, "lo"]] and environment["device_count"] == 1
    sources = load(root / "sources.json")
    assert sources["garak_direct_url"]["vcs_info"]["commit_id"] == "3f50ea5ff9cd7050099940647c15c39b07a93392"
    preflights = {}
    for strategy in ("adaptive", "deletion"):
        readiness = load(root / f"{strategy}-readiness.json")
        assert readiness["ready"] and [role["role"] for role in readiness["roles"]] == ["Target", "Evaluator"]
        command = load(root / f"{strategy}-preflight/command.json")
        assert command["exit_code"] == 0
        preflights[strategy] = {
            "exit_code": 0, "seconds": command["duration_seconds"],
            "calls": len(selected(readiness["events"], "call_completed")),
            "success": readiness["roles"][1]["success"],
            "raw_response": readiness["roles"][1]["response"],
        }
    return {"attempt": root.name, "preflights": preflights,
            "runs": {strategy: check_run(root, strategy) for strategy in ("adaptive", "deletion")}}


def check_checks(root):
    report = load(root / "quard-report.json")
    commands = {name: load(root / name / "command.json") for name in ("pytest", "quard")}
    assert commands["pytest"]["exit_code"] == 0
    assert commands["quard"]["exit_code"] in (0, 1)
    assert report["exit_code"] == commands["quard"]["exit_code"]
    assert report["summary"]["infrastructure_errors"] == 0
    return {"commands": commands, "quard_summary": report["summary"],
            "quard_run_id": report["run_id"],
            "tools": {tool["name"]: tool["status"] for tool in report["tools"]}}


def check_archive(path):
    checksums = load(path.with_name("checksums.json"))
    assert digest(path) == checksums[path.name]
    with tarfile.open(path, "r:gz") as archive:
        names = set()
        for member in archive.getmembers():
            if member.isfile():
                assert member.name not in names
                names.add(member.name)
                with archive.extractfile(member) as source:
                    assert hashlib.file_digest(source, "sha256").hexdigest() == checksums[member.name]
    assert names == {name for name in checksums if "/" in name}
    for name, expected in checksums.items():
        if "/" not in name:
            assert digest(path.with_name(name)) == expected
    print(f"Verified archive, {len(names)} members and published companion files")


def publish(roots, destination, verified, checks):
    dump(destination / "verified.json", verified)
    for profile in roots[-1].joinpath("inputs").glob("*.toml"):
        shutil.copyfile(profile, destination / profile.name)
    shutil.copyfile(roots[-1] / "inputs/request.txt", destination / "request.txt")
    archived = {root.name: root for root in roots}
    if checks:
        dump(destination / "checks.json", check_checks(checks))
        archived[checks.name] = checks
        native = Path(load(checks / "quard-report.json")["report_path"]).parent
        archived[f"{checks.name}/quard-native"] = native
    checksums = {}
    with tarfile.open(destination / "evidence.tar.gz", "w:gz") as archive:
        for name, root in archived.items():
            archive.add(root, arcname=name)
            checksums.update({f"{name}/{path.relative_to(root)}": digest(path)
                              for path in sorted(root.rglob("*")) if path.is_file()})
    checksums["evidence.tar.gz"] = digest(destination / "evidence.tar.gz")
    checksums.update({path.name: digest(path) for path in sorted(destination.iterdir())
                      if path.is_file() and path.name not in {"checksums.json", "evidence.tar.gz"}})
    dump(destination / "checksums.json", checksums)
    print(f"archive: {destination / 'evidence.tar.gz'}; bytes={(destination / 'evidence.tar.gz').stat().st_size}; sha256={checksums['evidence.tar.gz']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", type=Path, nargs="*")
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument("--checks-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.archive:
        check_archive(args.archive)
    if not args.attempts:
        if args.publish_dir or not args.archive:
            parser.error("provide attempt directories or --archive")
        return
    verified = [check_attempt(root) for root in args.attempts]
    if args.publish_dir:
        publish(args.attempts, args.publish_dir, verified, args.checks_dir)
    else:
        print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

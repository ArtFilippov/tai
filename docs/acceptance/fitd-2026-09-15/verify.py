"""Recheck this acceptance bundle without models, credentials, or third-party code."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def require(condition, description):
    if not condition:
        raise ValueError(description)


def verify(bundle):
    def load(name):
        return json.loads((bundle / name).read_text(encoding="utf-8"))

    command = load("fitd-command.json")
    preflight = load("preflight.json")
    run_id = command["stdout"].splitlines()[0].removeprefix("Запуск: ")
    directory = bundle / "experiments" / run_id
    if not directory.is_dir():
        directory = bundle  # Gist exports have a flat layout.
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    result = metadata["result"]
    require(result is not None, "A completed acceptance run must present a Message")
    result_bytes = (directory / "result.txt").read_bytes()
    require(result_bytes == result["text"].encode("utf-8"), "Exact result bytes")
    require(metadata["result_id"] == result["id"], "Final result ID")
    require(metadata["run_id"] == run_id, "Run namespace")
    require(command["returncode"] == 0, "FITD command exit code")
    require(metadata["state"] == "COMPLETED", "Normal completion")
    require(events[-1]["kind"] == "run_finished", "Final journal event")
    for key in ("state", "reason"):
        require(events[-1]["data"][key] == metadata[key], f"Final {key}")
    require(
        events[-1]["data"]["revision"] == metadata["finalization_revision"],
        "Finalization revision",
    )
    require(
        [e["sequence"] for e in events] == list(range(1, len(events) + 1)),
        "Continuous event sequence",
    )
    require(all(e["run_id"] == run_id for e in events), "Event run IDs")

    started, completed, responses = {}, {}, {}
    published = None
    evaluation = None
    evaluations = []
    pending_evaluation = None
    for event in events:
        kind, data = event["kind"], event["data"]
        if kind == "call_started":
            require(data["call_id"] not in started, "Unique call ID")
            require(
                data["input"]["origin"] == metadata["settings"]["base"],
                "Prompt provenance reaches the base Message",
            )
            started[data["call_id"]] = data
        elif kind == "call_completed":
            call_id = data["call_id"]
            require(call_id in started and call_id not in completed, "Paired call")
            require(data["role"] == started[call_id]["role"], "Call role")
            require(
                data["response"]["origin"] == started[call_id]["input"],
                "Response provenance",
            )
            completed[call_id] = data
            responses[data["response"]["id"]] = data
        elif kind in ("local_decision", "branch_selected"):
            require(data["response_id"] in responses, "Decision response ID")
            if "judgment_response_id" in data:
                judgment = responses[data["judgment_response_id"]]
                require(judgment["role"] == "Attacker", "Internal model judgment")
        elif kind == "result_published":
            published, evaluation = data["result"], None
            response = responses[published["id"]]
            require(response["role"] == "Target", "Published role")
            require(response["response"] == published, "Published Message")
        elif kind == "evaluation_started":
            require(pending_evaluation is None, "Sequential evaluations")
            require(
                published is not None and data["result_id"] == published["id"],
                "Evaluation of the current published result",
            )
            pending_evaluation = data["result_id"]
        elif kind == "evaluation_completed":
            require(data["result_id"] == pending_evaluation, "Evaluation pairing")
            pending_evaluation = None
            evaluation = data
            evaluations.append({"sequence": event["sequence"], **data})
    require(started.keys() == completed.keys(), "All model calls completed")
    require(published == result, "Final published Message including provenance")
    require(pending_evaluation is None, "No unfinished evaluation")
    require(evaluation == metadata["evaluation"], "Final evaluation linkage")
    counts = Counter(c["role"] for c in completed.values())
    for role in ("Target", "Attacker"):
        require(counts[role] > 1, "Multi-turn execution of both roles")
        require(
            counts[role] == metadata["progress"][f"{role.lower()}_calls"],
            "Public progress matches completed calls",
        )
    target_ids = {
        c["conversation_id"] for c in started.values() if c["role"] == "Target"
    }
    attacker_ids = {
        c["conversation_id"] for c in started.values() if c["role"] == "Attacker"
    }
    require(target_ids.isdisjoint(attacker_ids), "Independent conversations")

    require(preflight["ready"] and len(preflight["roles"]) == 2, "Both roles ready")
    require(
        {r["role"] for r in preflight["roles"]} == {"Target", "Attacker"},
        "Readiness roles",
    )
    for role in preflight["roles"]:
        require(role["ok"] and role["response"].strip(), "Real nonempty readiness text")
        final = metadata["final_settings"][role["role"]]
        require(final["actual_dtype"] == "float16", "Observed CUDA FP16")
        for key in (
            "backend",
            "model",
            "revision",
            "device",
            "dtype",
            "generation",
            "local_files_only",
            "versions",
            "actual_dtype",
        ):
            require(
                role["final_settings"][key] == final[key], f"Same live profile: {key}"
            )
    for name in ("preflight-command.json", "fitd-command.json"):
        record = load(name)
        require(record["returncode"] == 0, "Live command succeeded")
        require(
            record["command"][:3] == ["unshare", "--net", "--"], "Network isolation"
        )
        require(
            set(record["environment"])
            == {
                "HOME",
                "PATH",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
            },
            "Explicit credential-free environment",
        )
    require(
        json.loads(load("preflight-command.json")["stdout"]) == preflight,
        "Preflight file matches terminal",
    )

    outcome = "отсутствует; финальный текст не оценивался"
    if evaluation is not None:
        require(evaluation["result_id"] == result["id"], "Evaluated final result ID")
        require(
            evaluation["success"]
            == (metadata["settings"]["evaluator"]["criterion"] in result["text"]),
            "ContainsTextEvaluator checks only the presented text",
        )
        outcome = "успех" if evaluation["success"] else "неуспех"
    terminal_text = result["text"] + ("" if result["text"].endswith("\n") else "\n")
    terminal_suffix = (
        f"Причина завершения: {metadata['reason']}\n"
        f"Результат Message {result['id']} ({len(result['text'])} символов):\n"
        f"{terminal_text}Оценка результата: {outcome}\n"
    )
    require(
        command["stdout"].endswith(terminal_suffix), "Exact terminal result and outcome"
    )
    return {
        "run_id": run_id,
        "state": metadata["state"],
        "reason": metadata["reason"],
        "result_id": result["id"],
        "evaluation": evaluation,
        "evaluations": evaluations,
        "events": len(events),
        "completed_calls": dict(counts),
        "branches": dict(
            Counter(
                e["data"]["branch"] for e in events if e["kind"] == "branch_selected"
            )
        ),
        "result_matches_metadata_and_terminal": True,
        "calls_provenance_decisions_publications_evaluations_linked": True,
        "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("result.txt", "events.jsonl", "metadata.json")
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))

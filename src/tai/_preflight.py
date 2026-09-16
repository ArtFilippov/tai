"""One real exchange per role, including a neutral model-classification pair."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import monotonic

from tai._config import ModelEvaluatorProfile, RoleProfile
from tai.evaluators import ModelEvaluator
from tai.messages import Message


def run_preflight(profile: RoleProfile, report: Path | None = None) -> int:
    prompt = "Reply with exactly READY."
    redactor = profile.redactor()
    results = []
    ready = True
    for role, chat in (("Target", profile.target), ("Attacker", profile.attacker)):
        result: dict[str, object] = {
            "role": role,
            "conversation_id": chat.conversation_id,
            "started_at": datetime.now(UTC).isoformat(),
            "settings": chat._diagnostic_settings(redactor),
        }
        start = monotonic()
        try:
            text = chat.send(Message(prompt)).to_string()
            ok = bool(text.strip())
            result.update(ok=ok, response=text)
            if not ok:
                result["error"] = "EmptyText: модель не вернула непустой текст"
        except Exception as exc:
            ok = False
            result.update(
                ok=False,
                response=None,
                error=f"{type(exc).__name__}: {exc}",
                diagnostic=getattr(exc, "diagnostic", None),
            )
        result["elapsed_seconds"] = round(monotonic() - start, 3)
        result["final_settings"] = chat._diagnostic_settings(redactor)
        ready = ready and ok
        results.append(result)
    if isinstance(profile.evaluator, ModelEvaluatorProfile):
        # Exercise precisely the classifier used by FITD, including its strict
        # parser. A valid false verdict is readiness, not a readiness failure.
        base = Message("Name the capital of France.")
        presented = Message("The capital of France is Paris.")
        evaluator = profile.create_evaluator(base)
        assert isinstance(evaluator, ModelEvaluator)
        chat = evaluator.chat
        result = {
            "role": "Evaluator",
            "started_at": datetime.now(UTC).isoformat(),
            "settings": evaluator.settings(),
            "base": json.loads(base.dump()),
            "result": json.loads(presented.dump()),
            "result_id": presented.id,
            "response": None,
        }

        def send(request: Message) -> Message:
            # Snapshot the freshly reset assessment, including the UUID actually
            # used on the wire, rather than an earlier conversation's history.
            result["settings"] = evaluator.settings()
            result["conversation_id"] = chat.conversation_id
            result["input"] = json.loads(request.dump())
            response = chat.send(request)
            result["response"] = response.to_string()
            result["response_message"] = json.loads(response.dump())
            return response

        start = monotonic()
        try:
            success = evaluator._with_io(send, chat.reset).is_success(presented)
            ok = True
            result.update(ok=True, success=success)
        except Exception as exc:
            ok = False
            result.update(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                diagnostic=getattr(exc, "diagnostic", None),
            )
        result["elapsed_seconds"] = round(monotonic() - start, 3)
        result["final_settings"] = chat._diagnostic_settings(redactor)
        ready = ready and ok
        results.append(result)
    data = redactor.snapshot(
        {
            "check": "role_readiness",
            "tai_version": version("tai"),
            "completed_at": datetime.now(UTC).isoformat(),
            "prompt": prompt,
            "ready": ready,
            "roles": results,
        }
    )
    rendered = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if report is not None:
        report.write_text(rendered, encoding="utf-8")
    return 0 if ready else 1

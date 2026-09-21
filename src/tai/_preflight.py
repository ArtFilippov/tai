"""One real exchange per role, including a neutral model-classification pair."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from tai._config import ModelEvaluatorProfile, RoleProfile
from tai._operations import (
    Cause,
    Context,
    ConversationChange,
    EvaluationResult,
    Exchange,
    WorkerOutcome,
)
from tai.chats import LLMChat, ModelRole
from tai.evaluators import ModelEvaluator
from tai.journal import RunJournal
from tai.messages import Message
from tai.users import Evaluation, UserState, UserStatus


def run_preflight(
    profile: RoleProfile,
    report: Path | None = None,
    *,
    journal_factory: Callable[..., RunJournal] = RunJournal,
) -> int:
    journal = journal_factory(redactor=profile.redactor())
    run_id = str(uuid4())
    error: Exception | None = None
    data: dict[str, Any] = {"ready": False}
    rendered = None
    try:
        journal.begin(run_id)
        data = _check_roles(profile, journal, run_id)
    except Exception as exc:
        error = exc
        raise
    finally:
        try:
            rendered = journal.finish(
                WorkerOutcome(
                    UserStatus(
                        state=UserState.COMPLETED
                        if data["ready"]
                        else UserState.FAILED,
                        run_id=run_id,
                        progress=journal.status().progress,
                        reason=f"{type(error).__name__}: {error}"
                        if error
                        else "preflight",
                    ),
                    None,
                    error,
                    None,
                ),
                report=(data, report) if error is None else None,
            )
        except Exception:
            if error is None:
                raise
    assert rendered is not None
    print(rendered, end="")
    return 0 if data["ready"] else 1


def _check_roles(
    profile: RoleProfile, journal: RunJournal, run_id: str
) -> dict[str, Any]:
    prompt = "Reply with exactly READY."
    redactor = journal.redactor

    def exchange(role: ModelRole, chat, request: Message, result_id=None) -> Message:
        context = Context(run_id, role, chat.conversation_id, Cause("preflight"))
        call_id = str(uuid4())
        return journal.call(
            "exchange",
            lambda: Exchange(context, call_id, request, chat.send(request)),
            context=context,
            call_id=call_id,
            message=request,
            result_id=result_id,
        ).response

    results = []
    ready = True
    participants: tuple[tuple[ModelRole, LLMChat], ...] = (
        ("Target", profile.target),
        ("Attacker", profile.attacker),
    )
    for role, chat in participants:
        result: dict[str, object] = {
            "role": role,
            "conversation_id": chat.conversation_id,
            "started_at": datetime.now(UTC).isoformat(),
            "settings": chat._diagnostic_settings(redactor),
        }
        start = monotonic()
        try:
            text = exchange(role, chat, Message(prompt)).to_string()
            ok = bool(text.strip())
            result.update(ok=ok, response=text)
            if not ok:
                result["error"] = "EmptyText: модель не вернула непустой текст"
        except Exception as exc:
            journal.check()
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
            response = exchange("Evaluator", chat, request, presented.id)
            result["response"] = response.to_string()
            result["response_message"] = json.loads(response.dump())
            return response

        def reset() -> None:
            previous = chat.conversation_id

            def change() -> ConversationChange:
                chat.reset()
                return ConversationChange(
                    "reset",
                    Context(
                        run_id, "Evaluator", chat.conversation_id, Cause("preflight")
                    ),
                    previous,
                )

            journal.call("preflight.reset", change)

        start = monotonic()
        try:
            context = Context(
                run_id, "Evaluator", chat.conversation_id, Cause("preflight")
            )
            assessment = journal.call(
                "assessment",
                lambda: EvaluationResult(
                    context,
                    "preflight",
                    presented,
                    Evaluation(
                        presented.id,
                        evaluator._with_io(send, reset).is_success(presented),
                    ),
                ),
                context=context,
                result_id=presented.id,
            )
            success = assessment.evaluation.success
            ok = True
            result.update(ok=True, success=success)
        except Exception as exc:
            journal.check()
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
    return {
        "check": "role_readiness",
        "tai_version": version("tai"),
        "completed_at": datetime.now(UTC).isoformat(),
        "prompt": prompt,
        "ready": ready,
        "roles": results,
    }

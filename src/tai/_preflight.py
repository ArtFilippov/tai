"""Real role exchanges and a neutral pair through the common evaluator contract."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from tai._config import RoleProfile
from tai._operations import (
    Cause,
    Context,
    ConversationChange,
    EvaluationResult,
    Exchange,
    ResourcesReleased,
    WorkerOutcome,
)
from tai.chats import LLMChat, ModelRole, _acquire_chats, _release_chats
from tai.evaluation import observe_evaluation
from tai.evaluators import evaluate
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
    chats = tuple(profile.chats().values())
    acquired = False
    released = None
    try:
        journal.begin(run_id)
        _acquire_chats(chats, journal)
        acquired = True
        data = _check_roles(profile, journal, run_id)
    except Exception as exc:
        error = exc
        raise
    finally:
        if acquired:
            _release_chats(chats, journal)
            released = ResourcesReleased(run_id, tuple(profile.chats()))
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
                    released,
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

    def exchange(role: ModelRole, chat: LLMChat, request: Message) -> Message:
        context = Context(run_id, role, chat.conversation_id, Cause("preflight"))
        call_id = str(uuid4())
        return journal.call(
            "exchange",
            lambda: Exchange(context, call_id, request, chat.send(request)),
            context=context,
            call_id=call_id,
            message=request,
        ).response

    def reset(role: ModelRole, chat: LLMChat) -> ConversationChange:
        previous = chat.conversation_id
        chat.reset()
        return ConversationChange(
            "reset",
            Context(run_id, role, chat.conversation_id, Cause("preflight")),
            previous,
        )

    results = []
    ready = True
    participants: tuple[tuple[ModelRole, LLMChat], ...] = (
        ("Target", profile.target),
        ("Attacker", profile.attacker),
    )
    for role, chat in participants:
        result: dict[str, Any] = {
            "role": role,
            "conversation_id": chat.conversation_id,
            "started_at": datetime.now(UTC).isoformat(),
            "settings": chat._diagnostic_settings(redactor),
        }
        start = monotonic()
        try:
            journal.call("preflight.reset", partial(reset, role, chat))
            result.update(
                conversation_id=chat.conversation_id,
                settings=chat._diagnostic_settings(redactor),
            )
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
    # The same operation and strict return check for every evaluator, including
    # user implementations. A valid false verdict is readiness, not a failure.
    base = Message("Name the capital of France.")
    presented = Message("The capital of France is Paris.")
    setup = profile.evaluator
    result = {
        "role": "Evaluator",
        "started_at": datetime.now(UTC).isoformat(),
        "settings": setup.settings(),
        "base": json.loads(base.dump()),
        "result": json.loads(presented.dump()),
        "result_id": presented.id,
        "response": None,
    }
    start = monotonic()
    context = Context(run_id, "Evaluator", "", Cause("preflight"))
    after = len(journal.events())
    try:
        with observe_evaluation(journal, context, base, presented) as evaluation_id:
            assessment = journal.call(
                "assessment",
                lambda: EvaluationResult(
                    context,
                    "preflight",
                    presented,
                    Evaluation(
                        presented.id, evaluate(setup.evaluator, base, presented)
                    ),
                    base,
                ),
                context=context,
                base=base,
                result=presented,
                result_id=presented.id,
                evaluation_id=evaluation_id,
            )
        ok = True
        result.update(ok=True, success=assessment.evaluation.success)
    except Exception as exc:
        journal.check()
        ok = False
        result.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            diagnostic=getattr(exc, "diagnostic", None),
        )
    # Diagnostic copies only. The actual classifier input never comes from here.
    for event in journal.events(after):
        observed: Any = event.data
        if event.kind == "call_started":
            result.update(
                conversation_id=observed["conversation_id"],
                input=observed["input"],
                exchange_settings=observed["settings"],
            )
        elif event.kind == "call_completed":
            result.update(
                response=observed["response"]["text"],
                response_message=observed["response"],
            )
    result["elapsed_seconds"] = round(monotonic() - start, 3)
    result["final_settings"] = setup.settings()
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

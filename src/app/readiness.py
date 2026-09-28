"""Role readiness execution through injected observation contracts."""

import json
import math
from functools import partial
from uuid import uuid4

from core.chats import LLMChat, ModelRole
from core.chats.scoring import require_scorer
from core.evaluation import evaluate
from core.evaluation.configuration import EvaluatorSetup
from core.primitives.messages import Message
from core.runs import Evaluation
from core.runs.operations import Cause, Context, ConversationChange, EvaluationResult, Exchange
from alert.values import object_value
from attacks.shared.contracts import AUTODAN

from .contracts import ReadinessServices
from .failure_capture import Failure
from .failures import error_data
from .profiles import ParticipantProfile

PROMPT = "Reply with exactly READY."
CALL_STARTED = "call_started"
CALL_COMPLETED = "call_completed"
EVALUATOR_ROLE = "Evaluator"


def _context(run_id: str, role: ModelRole, chat: LLMChat) -> Context:
    return Context(run_id, role, chat.conversation_id, Cause("preflight"))


def _reset(run_id: str, role: ModelRole, chat: LLMChat) -> ConversationChange:
    previous = chat.conversation_id
    chat.reset()
    return ConversationChange("reset", _context(run_id, role, chat), previous)


def _exchange(services: ReadinessServices, run_id: str, role: ModelRole, chat: LLMChat) -> Message:
    context, call_id, request = _context(run_id, role, chat), str(uuid4()), Message(PROMPT)
    return services.journal.call(
        "exchange", lambda: Exchange(context, call_id, request, chat.send(request)),
        context=context, call_id=call_id, message=request,
    ).response


def _check_chat(services: ReadinessServices, run_id: str, role: ModelRole, chat: LLMChat) -> dict[str, object]:
    result: dict[str, object] = {
        "role": role, "conversation_id": chat.conversation_id,
        "started_at": services.clock.now(), "settings": chat.diagnostic_settings(services.journal.redactor),
    }
    start = services.clock.elapsed()
    with Failure() as failure:
        _ = services.journal.call("preflight.reset", partial(_reset, run_id, role, chat))
        result.update(conversation_id=chat.conversation_id, settings=chat.diagnostic_settings(services.journal.redactor))
        response = _exchange(services, run_id, role, chat)
        text = response.to_string()
        result.update(ok=bool(text.strip()), response=text)
        if not text.strip():
            result["error"] = "EmptyText: модель не вернула непустой текст"
    if failure.error is not None:
        services.journal.check()
        result.update(error_data(failure.error), response=None)
    result.update(elapsed_seconds=round(services.clock.elapsed() - start, 3),
                  final_settings=chat.diagnostic_settings(services.journal.redactor))
    return result


def _assessment(
    services: ReadinessServices, context: Context, setup: EvaluatorSetup, pair: tuple[Message, Message],
) -> bool:
    base, presented = pair
    with services.observe(context, base, presented) as evaluation_id:
        assessment = services.journal.call(
            "assessment", lambda: EvaluationResult(
                context, "preflight", presented, Evaluation(presented.id, evaluate(setup.evaluator, base, presented)), base,
            ),
            context=context, base=base, result=presented, result_id=presented.id, evaluation_id=evaluation_id,
        )
    return assessment.evaluation.success


def _exchange_diagnostics(services: ReadinessServices, after: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for event in services.journal.events(after):
        observed = event.data
        if not isinstance(observed, dict):
            continue
        if event.kind == CALL_STARTED:
            result.update(conversation_id=observed["conversation_id"], input=observed["input"], exchange_settings=observed["settings"])
        elif event.kind == CALL_COMPLETED:
            response = object_value(observed["response"], "response")
            result.update(response=response["text"], response_message=response)
    return result


def _check_evaluator(services: ReadinessServices, run_id: str, setup: EvaluatorSetup) -> dict[str, object]:
    # A valid false verdict is readiness, not a failure.
    base, presented = Message("Name the capital of France."), Message("The capital of France is Paris.")
    result: dict[str, object] = {
        "role": "Evaluator", "started_at": services.clock.now(), "settings": setup.settings(),
        "base": json.loads(base.dump()), "result": json.loads(presented.dump()),
        "result_id": presented.id, "response": None,
    }
    start, after = services.clock.elapsed(), len(services.journal.events())
    context = Context(run_id, "Evaluator", "", Cause("preflight"))
    with Failure() as failure:
        result.update(ok=True, success=_assessment(services, context, setup, (base, presented)))
    if failure.error is not None:
        services.journal.check()
        result.update(error_data(failure.error))
    # Diagnostic copies only. The actual classifier input never comes from here.
    result.update(_exchange_diagnostics(services, after))
    result.update(elapsed_seconds=round(services.clock.elapsed() - start, 3), final_settings=setup.settings())
    return result


def check_roles(profile: ParticipantProfile, services: ReadinessServices, run_id: str) -> dict[str, object]:
    results = [_check_chat(services, run_id, role, chat) for role, chat in profile.chats().items() if role != EVALUATOR_ROLE]
    if profile.attack_type == AUTODAN:
        result: dict[str, object] = {"role": "TargetScoring", "started_at": services.clock.now()}
        with Failure() as failure:
            scorer = require_scorer(profile.target)
            loss = services.journal.call("preflight.scoring", lambda: scorer.score_continuation("Reply with exactly READY.", "READY"))
            if not math.isfinite(loss):
                raise ValueError("AutoDAN logits scoring returned a non-finite loss")
            result.update(ok=True, loss=loss)
        if failure.error is not None:
            services.journal.check()
            result.update(error_data(failure.error))
        results.append(result)
    results.append(_check_evaluator(services, run_id, profile.evaluator))
    return {
        "check": "role_readiness", "tai_version": services.version,
        "completed_at": services.clock.now(), "prompt": PROMPT,
        "ready": all(result["ok"] for result in results), "roles": results,
    }

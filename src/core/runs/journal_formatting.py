"""Map explicit domain outcomes to journal event names and evidence."""

from dataclasses import asdict, dataclass
from functools import singledispatch

from .journal_values import context_data, operation_value
from .operations import (
    BranchDecision, Checkpoint, ConversationChange, ConversationPoint,
    EvaluationResult, Exchange, LocalDecision, ResultChange, Synchronization,
    TargetExchange,
)


@dataclass(frozen=True)
class EventDetails:
    kind: str
    data: dict[str, object]


@singledispatch
def event_details(_outcome: object) -> EventDetails | None:
    return None


@event_details.register
def _exchange(outcome: Exchange) -> EventDetails:
    return EventDetails("call_completed", dict(
        context_data(outcome.context), call_id=outcome.call_id,
        request_id=outcome.request.id, response=operation_value(outcome.response),
    ))


@event_details.register
def _checkpoint(outcome: Checkpoint) -> EventDetails:
    return EventDetails("checkpoint_captured", dict(
        context_data(outcome.context), checkpoint_id=outcome.id,
    ))


@event_details.register
def _point(outcome: ConversationPoint) -> EventDetails:
    return EventDetails("checkpoint_created", dict(
        context_data(outcome.checkpoint.context), checkpoint_id=outcome.checkpoint.id,
        request_id=outcome.request.id if outcome.request else None,
        response_id=outcome.response.id if outcome.response else None,
        result_id=outcome.result.id if outcome.result else None,
    ))


@event_details.register
def _conversation(outcome: ConversationChange) -> EventDetails:
    return EventDetails("history_" + outcome.kind, dict(
        context_data(outcome.context),
        previous_conversation_id=outcome.previous_conversation_id,
        checkpoint_id=outcome.checkpoint.id if outcome.checkpoint else None,
    ))


@event_details.register
def _synchronization(outcome: Synchronization) -> EventDetails:
    return EventDetails("conversation_synchronized", dict(
        context_data(outcome.conversation.context), outcome=operation_value(outcome),
        from_checkpoint_id=outcome.before.checkpoint.id,
        checkpoint_id=outcome.after.checkpoint.id,
    ))


@event_details.register
def _target(outcome: TargetExchange) -> EventDetails:
    return EventDetails("target_exchange", dict(
        context_data(outcome.exchange.context), call_id=outcome.exchange.call_id,
        checkpoint_id=outcome.point.checkpoint.id, request_id=outcome.exchange.request.id,
        response_id=outcome.exchange.response.id,
    ))


@event_details.register
def _local(outcome: LocalDecision) -> EventDetails:
    return EventDetails("local_decision", dict(
        context_data(outcome.context), decision=outcome.kind, value=outcome.value,
        request_id=outcome.request.id, response_id=outcome.response.id,
        checkpoint_id=outcome.checkpoint_id, judgment=operation_value(outcome.judgment),
        judgment_response_id=outcome.judgment.response.id if outcome.judgment else None,
    ))


@event_details.register
def _branch(outcome: BranchDecision) -> EventDetails:
    return EventDetails("branch_selected", dict(
        context_data(outcome.context), branch_id=outcome.id, branch=outcome.branch,
        point=operation_value(outcome.point), next_attempt_id=outcome.next_attempt_id,
        next_request=operation_value(outcome.next_request),
        response_id=outcome.point.response.id if outcome.point.response else None,
        checkpoint_id=outcome.point.checkpoint.id, levels_completed=outcome.levels_completed,
        next_levels_completed=outcome.next_levels_completed,
        checks=operation_value(outcome.checks),
    ))


@event_details.register
def _result(outcome: ResultChange) -> EventDetails:
    suffix = {"withdrawn": "withdrawn", "unchanged": "unchanged"}.get(outcome.kind, "published")
    return EventDetails("result_" + suffix, dict(
        context_data(outcome.context), change=outcome.kind,
        checkpoint_id=outcome.checkpoint_id, result=operation_value(outcome.current),
        previous=operation_value(outcome.previous),
        previous_evaluation=operation_value(outcome.previous_evaluation),
        evaluation=operation_value(outcome.evaluation),
    ))


@event_details.register
def _evaluation(outcome: EvaluationResult) -> EventDetails:
    return EventDetails("evaluation_completed", dict(
        context_data(outcome.context), checkpoint_id=outcome.checkpoint_id,
        base_id=outcome.base.id, **asdict(outcome.evaluation),
    ))

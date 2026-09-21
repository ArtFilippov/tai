"""Describe native continuation choices without reimplementing FITD decisions."""

from dataclasses import replace
from typing import Literal

from core.primitives.messages import Message
from core.runs.operations import BranchDecision

from .contracts import Continuation, Selection
from .native import Attempt, Conversation, attempt_prompt, levels_completed
from .target import history

CONTINUE = "continue"
SELECT_CONTINUATION = "select_continuation"
type Branch = Literal["exhausted", "continue", "realign", "restart", "rephrase"]


def branch_kind(conversation: Conversation, previous: Attempt, following: Attempt | None) -> Branch:
    if following is None:
        return "exhausted"
    if levels_completed(following) > levels_completed(previous):
        return CONTINUE
    return adaptation_kind(conversation, following)


def adaptation_kind(conversation: Conversation, following: Attempt) -> Branch:
    before = history(conversation)
    if (prefix := history(attempt_prompt(following))[:-1]) == before:
        return "realign"
    if not prefix:
        return "restart"
    if prefix == before[:-2]:
        return "rephrase"
    raise RuntimeError("Unrecognized FITD conversation adaptation")


def describe_selection(
    selection: Selection, attempts: tuple[Attempt, Attempt | None], branch: Branch, request: Message | None,
) -> Continuation[Attempt]:
    previous, following = attempts
    assert selection.cause.id is not None
    return Continuation(
        BranchDecision(
            selection.cause.id, replace(selection.point.checkpoint.context, cause=selection.cause),
            branch, selection.point, str(following.uuid) if following is not None else None,
            request, levels_completed(previous),
            levels_completed(following) if following is not None else None,
            selection.checks,
        ),
        following,
    )

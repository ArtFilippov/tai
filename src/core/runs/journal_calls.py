"""Evidence and error formatting for admitted operation calls."""

from typing import TypedDict

from core.chats import ChatCancelled
from core.primitives.messages import Message

from .journal_values import context_data, operation_value
from .operations import Context

EXCHANGE = "exchange"
ASSESSMENT = "assessment"
STOP_REQUESTED = "stop_requested"


class CallInputs(TypedDict, total=False):
    context: Context | None
    call_id: str | None
    message: Message | None
    result_id: str | None
    checkpoint_id: str | None
    base: Message | None
    result: Message | None
    evaluation_id: str | None
    settings: object


def call_data(operation: str, inputs: CallInputs) -> dict[str, object]:
    context = inputs.get("context")
    data = context_data(context) if context is not None else {}
    data.update({
        key: operation_value(value) for key, value in inputs.items()
        if key not in {"context", "message", "call_id", "base"} and value is not None
    })
    if (base := inputs.get("base")) is not None:
        data.update(base_id=base.id, base=operation_value(base))
    if operation == EXCHANGE:
        data.update(call_id=inputs.get("call_id"), input=operation_value(inputs.get("message")))
    return data


def failure_details(operation: str, error: Exception) -> tuple[str, dict[str, object]]:
    prefix = "call" if operation == EXCHANGE else "evaluation"
    cancelled = isinstance(error, ChatCancelled)
    details: dict[str, object] = {"error": f"{type(error).__name__}: {error}"}
    if (diagnostic := getattr(error, "diagnostic", None)) is not None:
        details["diagnostic"] = diagnostic
    if isinstance(error, ChatCancelled) and operation == EXCHANGE:
        details["response"] = operation_value(error.response)
    return prefix + ("_cancelled" if cancelled else "_failed"), details

"""Pure serialization of explicit operation values and diagnostic context."""

from dataclasses import fields, is_dataclass
from functools import singledispatch

from core.primitives.messages import Message

from .operations import Checkpoint, Context


@singledispatch
def operation_value(value: object) -> object:
    """Serialize operation values without inspecting opaque checkpoint tokens.

    Args:
        value (object): Explicit operation input or outcome.

    Returns:
        object: JSON-compatible evidence, or the original unsupported value.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: operation_value(getattr(value, item.name)) for item in fields(value)}
    return value


@operation_value.register
def _message(value: Message) -> object:
    return value.snapshot()


@operation_value.register
def _checkpoint(value: Checkpoint) -> object:
    return {"context": operation_value(value.context), "id": value.id}


@operation_value.register
def _exception(value: Exception) -> object:
    return {
        "type": type(value).__name__,
        "message": str(value),
        "diagnostic": getattr(value, "diagnostic", None),
    }


@operation_value.register(tuple)
@operation_value.register(list)
def _sequence(value: tuple[object, ...] | list[object]) -> object:
    return [operation_value(item) for item in value]


@operation_value.register(dict)
def _mapping(value: dict[object, object]) -> object:
    return {key: operation_value(item) for key, item in value.items()}


def context_data(context: Context) -> dict[str, object]:
    return {
        "context": operation_value(context),
        "role": context.role,
        "conversation_id": context.conversation_id,
        "attempt_id": context.cause.attempt_id,
        "cause_id": context.cause.id,
    }

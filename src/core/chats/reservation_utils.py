"""Reserve a whole participant set atomically, without swapping its owner."""

from collections.abc import Generator
from contextlib import ExitStack, contextmanager

from .chat_contract import LLMChat
from .exchange_gate_contract import ExchangeGate


@contextmanager
def locked(chats: tuple[LLMChat, ...]) -> Generator[tuple[ExchangeGate, ...]]:
    """Hold every participant's guard in stable identity order.

    Args:
        chats (tuple[LLMChat, ...]): Participants to lock together.

    Yields:
        tuple[ExchangeGate, ...]: Locked gates in their acquisition order.
    """
    gates = tuple(chat.reservation for chat in sorted(chats, key=id))
    with ExitStack() as locks:
        for gate in gates:
            _ = locks.enter_context(gate.guard())
        yield gates


def acquire_chats(chats: tuple[LLMChat, ...], owner: object) -> None:
    """Reserve all participants before allowing any to enter another run.

    Args:
        chats (tuple[LLMChat, ...]): Distinct participants.
        owner (object): Identity of this execution.

    Raises:
        RuntimeError: A participant already belongs to an execution.
    """
    with locked(chats) as gates:
        if any(gate.busy() for gate in gates):
            raise RuntimeError("LLMChat already belongs to an active execution")
        for gate in gates:
            gate.reserve(owner)


def release_chats(chats: tuple[LLMChat, ...], owner: object) -> None:
    """Release only the execution that acquired the participant set.

    Args:
        chats (tuple[LLMChat, ...]): Reserved participants.
        owner (object): Identity of the execution releasing the participants.
    """
    with locked(chats) as gates:
        for gate in reversed(gates):
            gate.release(owner)

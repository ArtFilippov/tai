"""Synchronize native FITD prefixes with opaque chat-owned checkpoints."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import override

from core.primitives.messages import Message
from core.runs.operations import (
    Cause, ConversationChange, ConversationPoint, ResultChange, Synchronization, TargetExchange,
)

from .contracts import StrategyPorts, TargetOperations
from .native import Conversation, GarakMessage, message_text

type History = tuple[tuple[str, str], ...]
USER_ROLE = "user"


def history(conversation: Conversation) -> History:
    return tuple((turn.role, message_text(turn.content)) for turn in conversation.turns)


@dataclass
class _AttemptContext:
    id: str | None = None
    request: Message | None = None


class FITDTarget(TargetOperations[Conversation, GarakMessage]):
    """Map library prefixes to chat-owned snapshots, never provider histories.

    A completed response is diagnostic until FITD chooses to advance its ladder.
    Realignment retains off-topic context without presenting it as a result.
    """

    def __init__(self, ports: StrategyPorts) -> None:
        """Bind conversation operations for the native target.

        Args:
            ports (StrategyPorts): Admitted exchange and conversation operations.
        """
        super().__init__()
        self._ports = ports
        self._conversation = ports.conversation
        self._history: History = ()
        self._snapshots: dict[History, ConversationPoint] = {}
        self._conversation_id = self._conversation.conversation_id("Target")
        self._attempt = _AttemptContext()
        _ = self.save(None, None, None, Cause("initial_checkpoint"))

    @property
    @override
    def current(self) -> ConversationPoint:
        return self._snapshots[self._history]

    @override
    @contextmanager
    def attempt(self, attempt_id: str, request: Message) -> Generator[None]:
        self._attempt = _AttemptContext(attempt_id, request)
        try:
            yield
        finally:
            self.clear_history()

    @override
    def save(
        self, request: Message | None, response: Message | None, result: Message | None, cause: Cause,
    ) -> ConversationPoint:
        snapshot = ConversationPoint(
            self._conversation.checkpoint("Target", cause), request, response, result
        )
        self._snapshots[self._history] = snapshot
        return snapshot

    def _synchronize_prefix(self, prefix: History, cause: Cause) -> ConversationChange:
        previous = self.current
        if prefix == self._history:
            context = replace(previous.checkpoint.context, cause=cause)
            return ConversationChange("unchanged", context, self._conversation_id, previous.checkpoint)
        if not prefix:
            change = self._conversation.reset("Target", cause)
            self._conversation_id = self._conversation.conversation_id("Target")
            self._history = ()
            self._snapshots.clear()
            _ = self.save(None, None, None, cause)
            return change
        if (snapshot := self._snapshots.get(prefix)) is None:
            raise RuntimeError("FITD requested an unknown conversation prefix")
        change = self._conversation.restore("Target", snapshot.checkpoint, cause)
        self._history = prefix
        return change

    @override
    def synchronize(self, prompt: Conversation, cause: Cause) -> Synchronization:
        self._ports.check_running()
        turns = history(prompt)
        if not turns or turns[-1][0] != USER_ROLE:
            raise RuntimeError("FITD must supply a final user turn")
        previous = self.current
        change = self._synchronize_prefix(turns[:-1], cause)
        current = self.current
        result = self._ports.publish(current.result, change.context, current.checkpoint.id)
        return Synchronization(change, previous, current, result)

    @override
    def accept(self, cause: Cause) -> ResultChange:
        snapshot = self.current
        assert snapshot.response is not None
        change = self._ports.publish(
            snapshot.response, replace(snapshot.checkpoint.context, cause=cause), snapshot.checkpoint.id,
        )
        self._snapshots[self._history] = replace(snapshot, result=change.current)
        return change

    @override
    def exchange(self, prompt: Conversation, request: Message, cause: Cause) -> TargetExchange:
        synchronization = self.synchronize(prompt, cause)
        result = self.current.result
        turns = history(prompt)
        exchange = self._ports.send("Target", request, cause)
        response = exchange.response
        self._history = turns + (("assistant", response.to_string()),)
        point = self.save(request, response, result, cause)
        return TargetExchange(exchange, synchronization, point)

    @override
    def generate(self, prompt: Conversation, generations_this_call: int) -> list[GarakMessage]:
        if generations_this_call != 1:
            raise ValueError("Only one generation is supported")
        assert self._attempt.request is not None
        outcome = self.exchange(prompt, self._attempt.request, Cause("target_exchange", self._attempt.id))
        response = outcome.exchange.response
        return [GarakMessage(response.to_string(), lang="en")]

    @override
    def clear_history(self) -> None:
        # garak clears its stateless generator after each exchange. Only discard
        # the transient request; LLMChat retains the real conversation.
        self._attempt = _AttemptContext()

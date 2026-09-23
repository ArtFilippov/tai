"""One fresh Target conversation and evaluation per strategy candidate."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import override

from core.primitives.messages import Message

from .contracts import GARAK_REVISION, Candidate, CompletionReason, StrategyPorts, StrategyRuntime


@dataclass(frozen=True)
class CandidateRuntime(StrategyRuntime):
    candidates: Callable[[Message], Iterable[Candidate]]

    @property
    @override
    def revision(self) -> str:
        return GARAK_REVISION

    @override
    def __call__(self, base: Message, ports: StrategyPorts) -> CompletionReason:
        for candidate in self.candidates(base):
            cause = candidate.cause
            _ = ports.conversation.reset("Target", cause)
            exchange = ports.send("Target", candidate.request, cause)
            checkpoint = ports.conversation.checkpoint("Target", cause)
            _ = ports.publish(exchange.response, exchange.context, checkpoint.id)
            assessment = ports.evaluate(exchange.response, exchange.context, checkpoint.id)
            if assessment.evaluation.success:
                return "evaluator_success"
        return "strategy_exhausted"

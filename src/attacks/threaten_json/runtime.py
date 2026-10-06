"""Two independent Target conversations; only the pressured answer is presented."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

from attacks.shared.contracts import GARAK_REVISION, CompletionReason, StrategyPorts, StrategyRuntime
from core.primitives.messages import Message
from core.runs.operations import Cause

from .evaluator import parseable
from .requests import requests


@dataclass(frozen=True)
class ThreatenJSONRuntime(StrategyRuntime):
    """Compare formats without permitting the control answer into evaluation.

    Attributes:
        comparison (Callable[[Message, Message, bool, bool], None]): Format evidence sink.
    """

    comparison: Callable[[Message, Message, bool, bool], None]

    @property
    @override
    def revision(self) -> str:
        return GARAK_REVISION

    @override
    def __call__(self, base: Message, ports: StrategyPorts) -> CompletionReason:
        control, pressured = requests(base)
        control_cause = Cause("json_control", control.id)
        _ = ports.conversation.reset("Target", control_cause)
        reference = ports.send("Target", control, control_cause)
        pressured_cause = Cause("json_pressured", pressured.id)
        _ = ports.conversation.reset("Target", pressured_cause)
        exchange = ports.send("Target", pressured, pressured_cause)
        self.comparison(reference.response, exchange.response, parseable(reference.response), parseable(exchange.response))
        checkpoint = ports.conversation.checkpoint("Target", pressured_cause)
        _ = ports.publish(exchange.response, exchange.context, checkpoint.id)
        return "evaluator_success" if ports.evaluate(exchange.response, exchange.context, checkpoint.id).evaluation.success else "strategy_exhausted"

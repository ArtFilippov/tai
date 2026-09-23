"""Static garak composition with tai's early-stop policy and explicit base."""

from importlib import import_module
from os import PathLike
from typing import cast, override

from attacks.shared.contracts import GARAK_REVISION, CompletionReason, StrategyPorts, StrategyRuntime
from core.primitives.messages import Message
from core.runs.operations import Cause

from .profile import VARIANT_COUNT


def variants(base: Message) -> tuple[Message, ...]:
    # Resolving garak's LocalDataPaths can itself fail: do it inside execution.
    native = import_module("garak.probes.adaptive_attacks")
    template = cast(str, native.ADAPTIVE_TEMPLATE)
    suffixes_path = cast(PathLike[str], native.SUFFIXES_PATH)
    # garak.data documents open(LocalDataPath) as its supported resource API.
    # Match the probe's UTF-8, whitespace and empty-line handling without seeds.
    with open(suffixes_path, encoding="utf-8") as resource:
        suffixes = [line.strip() for line in resource if line.strip()]
    if len(suffixes) != VARIANT_COUNT:
        raise ValueError("Pinned garak Adaptive Attacks must contain five suffixes")
    return tuple(base.spawn(template.format(goal=base.to_string(), suffix=suffix)) for suffix in suffixes)


class AdaptiveRuntime(StrategyRuntime):
    @property
    @override
    def revision(self) -> str:
        return GARAK_REVISION

    @override
    def __call__(self, base: Message, ports: StrategyPorts) -> CompletionReason:
        for request in variants(base):
            cause = Cause("adaptive_variant", request.id)
            _ = ports.conversation.reset("Target", cause)
            exchange = ports.send("Target", request, cause)
            checkpoint = ports.conversation.checkpoint("Target", cause)
            _ = ports.publish(exchange.response, exchange.context, checkpoint.id)
            assessment = ports.evaluate(exchange.response, exchange.context, checkpoint.id)
            if assessment.evaluation.success:
                return "evaluator_success"
        return "strategy_exhausted"

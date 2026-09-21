"""Pure evaluator dependencies and detached assembly configuration."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

from core.chats import LLMChat

from .evaluator import Evaluator


@dataclass(frozen=True)
class EvaluatorSetup:
    """Assembly metadata, never an additional interface required of evaluators.

    Plain evaluators need only ``evaluate``. Bootstrap constructors can attach
    a source chat and a configuration provider without changing that contract.

    Attributes:
        evaluator (Evaluator): The one-operation assessment implementation.
        chat (LLMChat | None, default=None): Optional independently owned source conversation.
        configuration (Callable[[], Mapping[str, object]] | None, default=None): Settings provider.
    """

    evaluator: Evaluator
    chat: LLMChat | None = None
    configuration: Callable[[], Mapping[str, object]] | None = None

    def settings(self) -> dict[str, object]:
        """Return detached configuration without invoking the evaluator.

        Returns:
            dict[str, object]: A copy of the assembly configuration.
        """
        if self.configuration is not None:
            return deepcopy(dict(self.configuration()))
        cls = type(self.evaluator)
        return {"type": f"{cls.__module__}.{cls.__qualname__}"}


def evaluator_setup(evaluator: Evaluator | EvaluatorSetup) -> EvaluatorSetup:
    return evaluator if isinstance(evaluator, EvaluatorSetup) else EvaluatorSetup(evaluator)

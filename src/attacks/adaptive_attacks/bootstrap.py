"""Construct Target-only executions with the pinned Adaptive Attacks runtime."""

from attacks.shared.assembly import participants
from attacks.shared.execution_state import ExecutionSettings
from attacks.shared.user import ExecutionUser
from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message

from .garak_adapter import AdaptiveRuntime
from .profile import ADAPTIVE_PROFILE


def execution_settings(target: LLMChat, base: Message, setup: EvaluatorSetup) -> ExecutionSettings:
    return ExecutionSettings(participants({"Target": target}, setup), base, setup.evaluator, ADAPTIVE_PROFILE, AdaptiveRuntime())


class AdaptiveAttacksUser(ExecutionUser):
    def __init__(self, target: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup) -> None:
        super().__init__(execution_settings(target, base, evaluator_setup(evaluator)))

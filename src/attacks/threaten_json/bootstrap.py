"""Compose the Target-only ThreatenJSON User with an answer evaluator."""

from attacks.shared.assembly import participants
from attacks.shared.contracts import StrategyConfiguration
from attacks.shared.execution_state import ExecutionSettings
from attacks.shared.user import ExecutionUser
from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message

from .evaluator import JSONAnswerEvaluator
from .runtime import ThreatenJSONRuntime


def execution_settings(target: LLMChat, base: Message, setup: EvaluatorSetup) -> ExecutionSettings:
    return ExecutionSettings(
        participants({"Target": target}, setup), base, JSONAnswerEvaluator(setup.evaluator),
        StrategyConfiguration("threaten_json", {"attempts": ("control", "pressured")}),
        ThreatenJSONRuntime(lambda _control, _result, _before, _after: None),
    )


class ThreatenJSONUser(ExecutionUser):
    """Run the two-attempt strategy with the standard User lifecycle."""

    def __init__(self, target: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup) -> None:
        """Prepare a reusable User.

        Args:
            target (LLMChat): Target conversation.
            base (Message): Explicit JSON-answerable request.
            evaluator (Evaluator | EvaluatorSetup): Substantive answer judge.
        """
        super().__init__(execution_settings(target, base, evaluator_setup(evaluator)))

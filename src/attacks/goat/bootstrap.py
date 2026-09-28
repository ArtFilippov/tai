"""Compose independent Attacker and Target for a GOAT run."""

from typing import Unpack

from attacks.shared.assembly import participants, validate_options
from attacks.shared.execution_state import ExecutionSettings
from attacks.shared.user import ExecutionUser
from core.chats import LLMChat, ModelRole
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message

from .garak_adapter import GOATRuntime
from .profile import PARAMETERS, GOATOptions, goat_profile


def execution_settings(chats: dict[ModelRole, LLMChat], base: Message, setup: EvaluatorSetup, options: GOATOptions) -> ExecutionSettings:
    configuration = goat_profile(options).configuration()
    return ExecutionSettings(
        participants(chats, setup), base, setup.evaluator, configuration, GOATRuntime(goat_profile(options)),
    )


class GOATUser(ExecutionUser):
    def __init__(self, target: LLMChat, attacker: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup, **options: Unpack[GOATOptions]) -> None:
        validate_options(options, PARAMETERS)
        super().__init__(execution_settings({"Target": target, "Attacker": attacker}, base, evaluator_setup(evaluator), options))

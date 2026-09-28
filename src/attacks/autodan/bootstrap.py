"""Compose independent participants and a local scoring capability."""

from typing import Unpack

from attacks.shared.assembly import participants, validate_options
from attacks.shared.execution_state import ExecutionSettings
from attacks.shared.user import ExecutionUser
from core.chats import LLMChat, ModelRole
from core.chats.scoring import require_scorer
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message

from .garak_adapter import AutoDANRuntime
from .profile import PARAMETERS, AutoDANOptions, autodan_profile


def execution_settings(chats: dict[ModelRole, LLMChat], base: Message, setup: EvaluatorSetup, options: AutoDANOptions) -> ExecutionSettings:
    target = chats["Target"]
    scorer = require_scorer(target)
    profile = autodan_profile(options)
    return ExecutionSettings(participants(chats, setup), base, setup.evaluator, profile.configuration(), AutoDANRuntime(profile, scorer, target.model))


class AutoDANUser(ExecutionUser):
    def __init__(self, target: LLMChat, attacker: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup,
                 **options: Unpack[AutoDANOptions]) -> None:
        validate_options(options, PARAMETERS)
        super().__init__(execution_settings({"Target": target, "Attacker": attacker}, base, evaluator_setup(evaluator), options))

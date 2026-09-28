"""Compose DRA candidates with shared Target-only execution."""

from functools import partial
from typing import Unpack

from attacks.shared.assembly import participants, validate_options
from attacks.shared.candidates import CandidateRuntime
from attacks.shared.execution_state import ExecutionSettings
from attacks.shared.user import ExecutionUser
from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message

from .garak_adapter import candidates
from .profile import PARAMETERS, DRAOptions, dra_profile


def execution_settings(target: LLMChat, base: Message, setup: EvaluatorSetup, options: DRAOptions) -> ExecutionSettings:
    profile = dra_profile(options)
    return ExecutionSettings(
        participants({"Target": target}, setup), base, setup.evaluator, profile.configuration(),
        CandidateRuntime(partial(candidates, profile=profile)),
    )


class DRAUser(ExecutionUser):
    def __init__(self, target: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup, **options: Unpack[DRAOptions]) -> None:
        validate_options(options, PARAMETERS)
        super().__init__(execution_settings(target, base, evaluator_setup(evaluator), options))

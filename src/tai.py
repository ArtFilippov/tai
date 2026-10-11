"""Public contracts for textual LLM robustness experiments."""

from typing import Literal

from alert import (
    AlertManifest,
    AlertRecord,
    load_alert_manifest,
    select_alert_records,
)
from attacks.fitd import FITDExperiment, FITDUser
from attacks.adaptive_attacks import AdaptiveAttacksExperiment, AdaptiveAttacksUser
from attacks.badchars import BadCharactersDeletionExperiment, BadCharactersDeletionUser
from attacks.dra import DRAExperiment, DRAUser
from attacks.goat import GOATExperiment, GOATUser
from attacks.autodan import AutoDANExperiment, AutoDANUser
from attacks.threaten_json import ThreatenJSONExperiment, ThreatenJSONUser
from core.chats import ChatTurn
from core.chats.envelopes.cancellable_chat import CancellableChat
from core.chats.envelopes.cancelled import Cancelled as ChatCancelled
from core.chats.envelopes.conversation import LLMChat
from core.chats.envelopes.openai import (
    OpenAIAuthenticationError,
    OpenAIChat,
    OpenAIContentError,
    OpenAIError,
    OpenAIHTTPError,
    OpenAIJSONError,
    OpenAIRateLimitError,
    OpenAIResponseError,
    OpenAITimeoutError,
    OpenAITransportError,
)
from core.chats.envelopes.qwen_chat import QwenChat, QwenContextError, QwenError
from core.evaluation import Evaluator
from core.evaluation.assembly import EvaluatorSetup
from core.evaluation.model_evaluator import (
    ModelEvaluationError,
    ModelEvaluator,
    ModelVerdictError,
)
from core.evaluation.text_evaluator import ContainsTextEvaluator
from core.primitives.messages import Message
from core.runs import Evaluation, Progress, RunSnapshot, User, UserState, UserStatus
from core.runs.journal import RunEvent, RunJournal
from core.runs.storage import RunRecord, capture_run, save_run

type ModelRole = Literal["Target", "Attacker", "Evaluator"]

__all__ = [
    "AutoDANExperiment",
    "AutoDANUser",
    "AdaptiveAttacksExperiment",
    "AdaptiveAttacksUser",
    "AlertManifest",
    "AlertRecord",
    "BadCharactersDeletionExperiment",
    "BadCharactersDeletionUser",
    "ChatCancelled",
    "ChatTurn",
    "CancellableChat",
    "ContainsTextEvaluator",
    "DRAExperiment",
    "DRAUser",
    "Evaluation",
    "Evaluator",
    "EvaluatorSetup",
    "FITDUser",
    "FITDExperiment",
    "GOATExperiment",
    "GOATUser",
    "LLMChat",
    "Message",
    "ModelEvaluationError",
    "ModelEvaluator",
    "ModelRole",
    "ModelVerdictError",
    "OpenAIAuthenticationError",
    "OpenAIChat",
    "OpenAIContentError",
    "OpenAIError",
    "OpenAIHTTPError",
    "OpenAIJSONError",
    "OpenAIRateLimitError",
    "OpenAIResponseError",
    "OpenAITimeoutError",
    "OpenAITransportError",
    "Progress",
    "QwenChat",
    "QwenContextError",
    "QwenError",
    "RunEvent",
    "RunJournal",
    "RunRecord",
    "RunSnapshot",
    "User",
    "UserState",
    "UserStatus",
    "ThreatenJSONExperiment",
    "ThreatenJSONUser",
    "capture_run",
    "load_alert_manifest",
    "save_run",
    "select_alert_records",
]

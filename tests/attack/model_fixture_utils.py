"""Offline model responses for independent run tests."""

from unittest.mock import Mock
from collections.abc import Sequence

from target.model import Model


def refusal() -> Model:
    """Answer every FITD role without loading model weights.

    Returns:
        Model: A model responding with refusals.
    """
    model = Mock(spec=Model)
    model.generate.return_value = "I cannot help with that request."
    model.revision.return_value = "local"
    model.settings.return_value = {"revision": "local"}
    return model


def goat() -> Model:
    """Answer each GOAT role without weights or external services.

    Returns:
        Model: History-free replies for a completed GOAT run.
    """
    def respond(messages: Sequence[dict[str, str]]) -> str:
        if messages[0]["role"] == "system":
            if "strict evaluator" in messages[0]["content"]:
                return "NO"
            return "Observation: start\nThought: adapt\nStrategy: ask\nResponse: Prompt"
        if "<BEGIN REQUEST>" in messages[0]["content"]:
            return "unsafe"
        return "Exact Target answer"

    model = Mock(spec=Model)
    model.generate.side_effect = respond
    model.revision.return_value = "controlled"
    model.settings.return_value = {"revision": "controlled"}
    return model


def threaten_json() -> Model:
    """Supply independent control, pressured and answer-judge replies.

    Returns:
        Model: History-free scripted offline role resource.
    """
    def respond(messages: Sequence[dict[str, str]]) -> str:
        if messages[0]["role"] == "system":
            return "YES"
        if "You might be wondering why" in messages[0]["content"]:
            return '"Detailed café explanation"'
        return '"Control"'

    model = Mock(spec=Model)
    model.generate.side_effect = respond
    model.revision.return_value = "controlled"
    model.settings.return_value = {"revision": "controlled"}
    return model


def adaptive() -> Model:
    """Answer adaptive Target prompts and independent judge prompts offline.

    Returns:
        Model: Shared, history-free controlled model resource.
    """
    def respond(messages: Sequence[dict[str, str]]) -> str:
        if messages[0]["role"] == "system":
            return "NO"
        return "Controlled adaptive reply"

    model = Mock(spec=Model)
    model.generate.side_effect = respond
    model.revision.return_value = "controlled"
    model.settings.return_value = {"revision": "controlled"}
    return model


def whois() -> Model:
    """Answer the report and independently assess its response offline.

    Returns:
        Model: History-free Target and judge resource.
    """
    def respond(messages: Sequence[dict[str, str]]) -> str:
        if messages[0]["role"] == "system":
            return "NO"
        return "A complete café explanation"

    model = Mock(spec=Model)
    model.generate.side_effect = respond
    model.revision.return_value = "controlled"
    model.settings.return_value = {"revision": "controlled"}
    return model

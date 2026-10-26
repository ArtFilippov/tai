"""Controlled attempts and offline model responses for ALERT series tests."""

from pathlib import Path
from unittest.mock import Mock

from attack.qwen_run import QwenRun
from attack.run_operation import RunOperation
from base.base_request import BaseRequest
from journal.disk_journal import DiskJournal
from journal.versioned_journal import VersionedJournal
from target.model import Model
from target.scripted_model import ScriptedModel


def attempt(operation: RunOperation) -> QwenRun:
    """Bind an offline operation to the per-attempt run interface.

    Args:
        operation (RunOperation): Controlled journal operation.

    Returns:
        QwenRun: A run using inert model resources.
    """
    return QwenRun(
        ScriptedModel([]),
        ScriptedModel([]), 1, operation
    )


def scripted(base: BaseRequest, path: Path, model: Model, attacker: Model, *, turns: int) -> None:
    """Write linked verdicts, fail before the second start, and omit the third result.

    Args:
        base (BaseRequest): Original request.
        path (Path): Independent journal destination.
        model (Model): Inert target resource.
        attacker (Model): Inert attacker resource.
        turns (int): Unused limit for this controlled operation.

    Raises:
        RuntimeError: When the second attempt fails before writing.
    """
    if (index := int(
        path.name.split(
            "-", maxsplit=1
        )[0]
    )) == 2:
        raise RuntimeError("offline failure before start")
    journal = VersionedJournal(
        DiskJournal(path)
    )
    journal.record(
        {
            "event": "start",
            "base": base.text(),
            "base_metadata": dict(
                base.metadata()
            )
        }
    )
    if index == 3:
        journal.record(
            {
                "event": "end",
                "status": "no_result",
                "reason": "no_result"
            }
        )
        return
    journal.record(
        {
            "event": "result",
            "text": "first"
        }
    )
    journal.record(
        {
            "event": "judgment",
            "text": "first",
            "assessment": True
        }
    )
    journal.record(
        {
            "event": "result",
            "text": "final"
        }
    )
    journal.record(
        {
            "event": "judgment",
            "text": "final",
            "assessment": index == 1
        }
    )
    journal.record(
        {
            "event": "end",
            "status": "success" if index == 1 else "round_limit"
        }
    )


def interrupted(base: BaseRequest, path: Path, model: Model, attacker: Model, *, turns: int) -> None:
    """Interrupt the second attempt after a complete first journal.

    Args:
        base (BaseRequest): Original request.
        path (Path): Independent journal destination.
        model (Model): Inert target resource.
        attacker (Model): Inert attacker resource.
        turns (int): Unused limit for this controlled operation.

    Raises:
        KeyboardInterrupt: When the second attempt begins.
    """
    if path.name.startswith("002-"):
        raise KeyboardInterrupt
    VersionedJournal(
        DiskJournal(path)
    ).record(
        {
            "event": "start",
            "base": base.text()
        }
    )
    VersionedJournal(
        DiskJournal(path)
    ).record(
        {
            "event": "end",
            "status": "no_result"
        }
    )


def assessment(base: BaseRequest, path: Path, model: Model, attacker: Model, *, turns: int) -> None:
    """Write an invalid final evaluation before raising its error.

    Args:
        base (BaseRequest): Original request.
        path (Path): Independent journal destination.
        model (Model): Inert target resource.
        attacker (Model): Inert attacker resource.
        turns (int): Unused limit for this controlled operation.

    Raises:
        ValueError: After saving an invalid final assessment.
    """
    journal = VersionedJournal(
        DiskJournal(path)
    )
    journal.record(
        {
            "event": "start",
            "base": base.text()
        }
    )
    journal.record(
        {
            "event": "result",
            "text": "last"
        }
    )
    journal.record(
        {
            "event": "assessment_error",
            "text": "last",
            "error": "invalid verdict"
        }
    )
    journal.record(
        {
            "event": "end",
            "status": "failure",
            "reason": "assessment_error"
        }
    )
    raise ValueError("invalid verdict")


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

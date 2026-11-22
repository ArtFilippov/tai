"""Offline model resources at the public ALERT composition boundary."""

from argparse import Namespace
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from attack.attack import Attack
from base.base_request import BaseRequest
from main.alert_series import prepare
from target.queued_model import QueuedModel
from target.model import Model


@contextmanager
def variants(directory: Path, replies: list[str], *, enabled: bool, attempts: int) -> Generator[list[Callable[[BaseRequest], Attack]]]:
    """Substitute only inference resources while retaining production factory assembly.

    Args:
        directory (Path): Independent series artifact root.
        replies (list[str]): Real model-operation responses in order.
        enabled (bool): Explicit Phrasing activation.
        attempts (int): Maximum preparation calls per tense.

    Yields:
        list[Callable[[BaseRequest], Attack]]: Configured per-request attack factories.
    """
    with resources(
        directory,
        QueuedModel(replies),
        enabled=enabled, attempts=attempts
    ) as factories:
        yield factories


@contextmanager
def resources(directory: Path, model: Model, *, enabled: bool, attempts: int) -> Generator[list[Callable[[BaseRequest], Attack]]]:
    """Bind actual role resources at the ALERT assembly seam.

    Args:
        directory (Path): Independent series artifact root.
        model (Model): Shared offline role resource, including deferred startup failures.
        enabled (bool): Explicit Phrasing activation.
        attempts (int): Maximum preparation calls per tense.

    Yields:
        list[Callable[[BaseRequest], Attack]]: Production per-request compositions.
    """
    arguments = Namespace(
        gpu=0, tokens=192, attacker="qwen", host="127.0.0.1", port=8080,
        turns=1, goat_turns=1, no_threaten_json=True, no_adaptive_attacks=True,
        no_dra=True, no_deletion=True, deletion_perturbations=1,
        deletion_positions=2, deletion_ascii=2, autodan=False, autodan_hga=False,
        phrasing=enabled, phrasing_attempts=attempts
    )
    with patch(
        "main.alert_series.LazyQwenModel",
        return_value=model
    ), patch(
        "main.alert_series.cuda_device", return_value="offline"
    ), patch(
        "main.alert_series.snapshot", return_value=None
    ), prepare(
        arguments, directory
    ) as factories:
        yield factories

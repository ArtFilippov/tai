"""Scripted resources at Doctor command composition boundaries."""

from argparse import Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from attack.attack import Attack
from base.base_request import BaseRequest
from main.alert_series import prepare
from main.doctor_cli import launch
from target.queued_model import QueuedModel


@contextmanager
def variants(directory: Path, replies: list[str], *, enabled: bool,
             techniques: Sequence[str] = ("puppetry",), seed: int = 0) -> Generator[list[Callable[[BaseRequest], Attack]]]:
    """Prepare usual ALERT factories with explicit Doctor settings.

    Args:
        directory (Path): Independent series artifact root.
        replies (list[str]): Completed Target and Judge replies in order.
        enabled (bool): Explicit Doctor activation.
        techniques (Sequence[str], default=("puppetry",)): Selected Doctor techniques.
        seed (int, default=0): Run-local leetspeak seed.

    Yields:
        list[Callable[[BaseRequest], Attack]]: Offline per-request variants.
    """
    arguments = Namespace(
        gpu=0, tokens=64, attacker="qwen", host="127.0.0.1", port=8080,
        turns=1, goat_turns=1, no_threaten_json=True, no_adaptive_attacks=True,
        no_dra=True, no_deletion=True, deletion_perturbations=1,
        deletion_positions=2, deletion_ascii=2, autodan=False, autodan_hga=False,
        doctor=enabled, doctor_techniques=techniques, doctor_seed=seed
    )
    with patch(
        "main.alert_series.LazyQwenModel",
        return_value=QueuedModel(replies)
    ), patch(
        "main.alert_series.cuda_device", return_value="offline"
    ), patch(
        "main.alert_series.snapshot", return_value=None
    ), prepare(
        arguments, directory
    ) as factories:
        yield factories


def standalone(path: Path, replies: list[str], *, techniques: Sequence[str], seed: int = 0) -> None:
    """Run the standalone assembly with scripted history-free generation.

    Args:
        path (Path): Fresh journal destination.
        replies (list[str]): Target and Judge replies in call order.
        techniques (Sequence[str]): Public technique selection.
        seed (int, default=0): Run-local leetspeak seed.
    """
    with patch(
        "main.doctor_cli.LazyQwenModel",
        return_value=QueuedModel(replies)
    ), patch(
        "main.doctor_cli.cuda_device", return_value="offline"
    ), patch(
        "main.doctor_cli.snapshot", return_value=None
    ):
        launch(
            Namespace(
                journal=path, base="Base café", gpu=0, tokens=73, doctor_techniques=techniques, doctor_seed=seed
            )
        )

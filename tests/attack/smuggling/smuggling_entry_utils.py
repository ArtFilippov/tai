"""Offline series resources at the Smuggling entry-point seam."""

from argparse import Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from attack.attack import Attack
from base.base_request import BaseRequest
from main.alert_series import prepare
from main.smuggling_cli import launch
from target.queued_model import QueuedModel


@contextmanager
def variants(directory: Path, replies: list[str], *, enabled: bool,
             techniques: Sequence[str] = ("homoglyph",)) -> Generator[list[Callable[[BaseRequest], Attack]]]:
    """Prepare usual ALERT factories with an explicitly selected Smuggling variant.

    Args:
        directory (Path): Independent series artifact root.
        replies (list[str]): Completed Target and Judge replies in order.
        enabled (bool): Explicit Smuggling activation.
        techniques (Sequence[str], default=("homoglyph",)): Public technique selection.

    Yields:
        list[Callable[[BaseRequest], Attack]]: Offline per-request variants.
    """
    arguments = Namespace(
        gpu=0, tokens=64, attacker="qwen", host="127.0.0.1", port=8080,
        turns=1, goat_turns=1, no_threaten_json=True, no_adaptive_attacks=True,
        no_dra=True, no_deletion=True, deletion_perturbations=1,
        deletion_positions=2, deletion_ascii=2, autodan=False, autodan_hga=False,
        smuggling=enabled, smuggling_techniques=techniques
    )
    with patch(
        "main.alert_series.LazyQwenModel", return_value=QueuedModel(replies)
    ), patch(
        "main.alert_series.cuda_device", return_value="offline"
    ), patch(
        "main.alert_series.snapshot", return_value=None
    ), prepare(
        arguments, directory
    ) as factories:
        yield factories


def standalone(path: Path, base: str, replies: list[str]) -> None:
    """Run the standalone assembly with history-free scripted generation.

    Args:
        path (Path): Independent journal destination.
        base (str): Exact original request.
        replies (list[str]): Target and Judge replies in call order.
    """
    with patch(
        "main.smuggling_cli.LazyQwenModel", return_value=QueuedModel(replies)
    ), patch(
        "main.smuggling_cli.cuda_device", return_value="offline"
    ), patch(
        "main.smuggling_cli.snapshot", return_value=None
    ):
        launch(
            Namespace(
                journal=path, base=base, gpu=0, tokens=73, smuggling_techniques=["homoglyph"]
            )
        )

"""Offline resource setup shared by focused DAN entry-point tests."""

from argparse import Namespace
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from attack.attack import Attack
from base.base_request import BaseRequest
from main.alert_series import prepare
from ...target.test_model import TestModel


@contextmanager
def variants(directory: Path, *, enabled: bool) -> Generator[list[Callable[[BaseRequest], Attack]]]:
    """Prepare the usual series factories with a controlled shared model.

    Args:
        directory (Path): Independent series artifact root.
        enabled (bool): Explicit DAN opt-in.

    Yields:
        list[Callable[[BaseRequest], Attack]]: Offline per-request variants.
    """
    arguments = Namespace(
        gpu=0, tokens=64, attacker="qwen", host="127.0.0.1", port=8080,
        turns=1, goat_turns=1, no_threaten_json=True, no_adaptive_attacks=True,
        no_dra=True, no_deletion=True, deletion_perturbations=1,
        deletion_positions=2, deletion_ascii=2, autodan=False, autodan_hga=False,
        dan_sections=["named"], dan=enabled,
    )
    with patch(
        "main.alert_series.LazyQwenModel",
        return_value=TestModel(
            "DAN mode enabled", "NO", name="Series/Target"
        )
    ), patch(
        "main.alert_series.cuda_device", return_value="controlled"
    ), patch(
        "main.alert_series.snapshot", return_value=None
    ), prepare(
        arguments, directory
    ) as factories:
        yield factories

"""Offline resource setup shared by focused DAN entry-point tests."""

from argparse import Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from attack.attack import Attack
from base.base_request import BaseRequest
from main.alert_series import prepare
from main.dan_cli import launch
from prompt_collection.provided_prompt import ProvidedPrompt
from prompt_collection.scripted_prompts import ScriptedPrompts
from target.queued_model import QueuedModel
from ...target.test_model import TestModel


@contextmanager
def variants(directory: Path, *, enabled: bool, sections: Sequence[str] = ("named",),
             replies: Sequence[str] = ()) -> Generator[list[Callable[[BaseRequest], Attack]]]:
    """Prepare the usual series factories with a controlled shared model.

    Args:
        directory (Path): Independent series artifact root.
        enabled (bool): Explicit DAN opt-in.
        sections (Sequence[str], default=("named",)): Selected DAN sections.
        replies (Sequence[str], default=()): Shared scripted calls, or constant refusals when empty.

    Yields:
        list[Callable[[BaseRequest], Attack]]: Offline per-request variants.
    """
    arguments = Namespace(
        gpu=0, tokens=64, attacker="qwen", host="127.0.0.1", port=8080,
        turns=1, goat_turns=1, no_threaten_json=True, no_adaptive_attacks=True,
        no_dra=True, no_deletion=True, deletion_perturbations=1,
        deletion_positions=2, deletion_ascii=2, autodan=False, autodan_hga=False,
        dan_sections=sections, dan=enabled,
    )
    with patch(
        "main.alert_series.LazyQwenModel",
        return_value=QueuedModel(list(replies)) if replies else TestModel("DAN mode enabled", "NO", name="Series/Target")
    ), patch(
        "main.alert_series.cuda_device", return_value="controlled"
    ), patch(
        "main.alert_series.snapshot", return_value=None
    ), prepare(
        arguments, directory
    ) as factories:
        yield factories


@contextmanager
def catalog() -> Generator[None]:
    """Provide bounded representatives at the approved collection contract seam.

    Yields:
        None: A native-text-independent orchestration fixture.
    """
    with ExitStack() as stack:
        for collection, prompts in {
            "NamedDanPrompts": [
                ProvidedPrompt(
                    "Named",
                    {
                        "section": "named",
                        "variant": "Dan_11_0",
                        "source_index": 0
                    }
                ),
            ],
            "AblationDanPrompts": [
                ProvidedPrompt(
                    "Ablation",
                    {
                        "section": "ablation",
                        "variant": "Ablation_Dan_11_0",
                        "combination": 127
                    }
                ),
            ],
            "WildDanPrompts": [
                ProvidedPrompt(
                    "Duplicate wild",
                    {
                        "section": "wild",
                        "variant": "DanInTheWildFull",
                        "source_index": 41
                    }
                ),
                ProvidedPrompt(
                    "Duplicate wild",
                    {
                        "section": "wild",
                        "variant": "DanInTheWildFull",
                        "source_index": 7
                    }
                ),
            ],
        }.items():
            _ = stack.enter_context(
                patch(
                    f"attack.dan.envelopes.dan_attack_envelope.{collection}",
                    return_value=ScriptedPrompts(prompts),
                )
            )
        yield


def standalone(path: Path, replies: list[str], sections: list[str]) -> None:
    """Execute standalone assembly with scripted participants and a real journal.

    Args:
        path (Path): Journal destination.
        replies (list[str]): Target and judge responses in call order.
        sections (list[str]): Selection sent through the command boundary.
    """
    with patch(
        "main.dan_cli.LazyQwenModel",
        return_value=QueuedModel(replies)
    ), patch(
        "main.dan_cli.cuda_device", return_value="offline"
    ), patch(
        "main.dan_cli.snapshot", return_value=None
    ):
        launch(
            Namespace(
                journal=path, base="  Base café\nexactly  ", gpu=0, tokens=73, dan_sections=sections
            )
        )

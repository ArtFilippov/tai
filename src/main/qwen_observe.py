"""Observe candidate changes in a pinned Qwen FITD run without disk events."""

from argparse import ArgumentParser, Namespace

from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from journal.memory_journal import MemoryJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_chat import QwenChat


def main() -> None:
    """Run an offline experiment on a chosen GPU and print its observation."""
    parser = ArgumentParser(description="Observe Qwen FITD candidate changes")
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--turns", type=int, default=2
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    launch(
        parser.parse_args()
    )


def launch(arguments: Namespace) -> None:
    """Assemble isolated roles and observe one run with local weights.

    Args:
        arguments (Namespace): Base request and inference limits.

    Raises:
        ValueError: If the GPU or target exchange limit is invalid.
    """
    if arguments.turns < 1:
        raise ValueError("FITD round limit must be positive")
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    journal = MemoryJournal()
    target = QwenChat(model)
    judge = RecordedJudge(
        QwenJudge(
            arguments.base,
            QwenChat(model)
        ),
        journal, arguments.base
    )
    probe = FITDProbe(
        arguments.base, ContinuingChat(
            QwenChat(model)
        ), journal, arguments.turns
    )
    _ = FITDAttack(
        target, judge, journal, probe
    ).run(arguments.base)
    print(
        journal.report()
    )

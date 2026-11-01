"""Standalone offline adaptive attack over an explicit base request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.adaptive_attacks.envelopes.adaptive_attacks_attack_envelope import AdaptiveAttacksAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def display(path: Path) -> None:
    """Display the saved outcome of a completed adaptive run.

    Args:
        path (Path): Finished run journal.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Reserve a fresh journal before invoking a shared local model.

    Args:
        arguments (Namespace): Base, journal, GPU and token limit.

    Raises:
        ValueError: If a model limit is invalid.
        FileExistsError: If the journal is already reserved.
    """
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid adaptive GPU or token limit")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    _ = snapshot(True)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    arguments.journal.touch(exist_ok=False)
    _ = AdaptiveAttacksAttackEnvelope(
        base=PlainBase(arguments.base),
        setup=SharedModelAttackSetup(
            model, arguments.base
        ),
        journal=VersionedJournal(
            DiskJournal(arguments.journal)
        )
    ).run()


def main() -> None:
    """Run one adaptive attack through its own Python entry point."""
    parser = ArgumentParser(description="Standalone offline Simple Adaptive Attacks")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    arguments = parser.parse_args()
    path = arguments.journal
    launch(arguments)
    display(path)


if __name__ == "__main__":
    main()

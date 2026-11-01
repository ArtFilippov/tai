"""Standalone bounded GOAT command with explicit model and request flags."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.envelopes.goat_attack_envelope import GOATAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run one offline GOAT attack through its own Python entry point."""
    parser = ArgumentParser(description="Standalone offline GOAT")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--turns", type=int, default=5
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    launch(
        parser.parse_args()
    )


def launch(arguments: Namespace) -> None:
    """Validate limits and reserve a unique journal before model use.

    Args:
        arguments (Namespace): Explicit request, CUDA device and bounds.

    Raises:
        ValueError: If a bound is invalid.
        FileExistsError: If the journal already exists.
    """
    if arguments.turns < 1 or arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid GOAT GPU, turn or token limit")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    device = cuda_device(arguments.gpu)
    model = LazyQwenModel(
        device, arguments.tokens
    )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    _ = GOATAttackEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        VersionedJournal(
            DiskJournal(arguments.journal)
        ),
        arguments.turns
    ).run()
    print(
        ParsedJournal(arguments.journal).read()
    )


if __name__ == "__main__":
    main()

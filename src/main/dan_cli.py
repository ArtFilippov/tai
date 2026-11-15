"""Standalone DAN attempt against an explicit base request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.dan.envelopes.dan_attack_envelope import DanAttackEnvelope
from attack.dan.dan_options_utils import options, sections
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run one recoverable DAN traversal with explicit local GPU setup."""
    parser = ArgumentParser(description="Request-bound DAN adaptation")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    options(parser)
    arguments = parser.parse_args()
    launch(arguments)
    display(arguments.journal)


def display(path: Path) -> None:
    """Display the recorded outcome, including its Target-linked successful source.

    Args:
        path (Path): Completed journal destination.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Validate and reserve one independent journal before executing DAN.

    Args:
        arguments (Namespace): Journal, request, GPU, generation and section settings.

    Raises:
        ValueError: If GPU, token limit or section selection is invalid.
        FileExistsError: If the journal is already reserved.
    """
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid GPU or token limit")
    sections(arguments.dan_sections)
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    _ = DanAttackEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        VersionedJournal(
            DiskJournal(arguments.journal)
        ),
        sections=arguments.dan_sections
    ).run()


if __name__ == "__main__":
    main()

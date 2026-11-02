"""Standalone offline WHOIS injection command for one explicit request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.shared_model_attack_setup import SharedModelAttackSetup
from attack.whois.envelopes.whois_attack_envelope import WhoisAttackEnvelope
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run one WHOIS injection and display its recoverable outcome."""
    parser = ArgumentParser(description="Standalone offline WHOIS injection")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    arguments = parser.parse_args()
    launch(arguments)
    display(arguments.journal)


def display(path: Path) -> None:
    """Display the saved outcome.

    Args:
        path (Path): Completed journal.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Reserve a new journal before model inference.

    Args:
        arguments (Namespace): Explicit request, CUDA device and token limit.

    Raises:
        ValueError: If model limits are invalid.
        FileExistsError: If the journal already exists.
    """
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid WHOIS GPU or token limit")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    _ = WhoisAttackEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        VersionedJournal(
            DiskJournal(arguments.journal)
        )
    ).run()


if __name__ == "__main__":
    main()

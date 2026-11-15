"""Standalone request-bound hypothetical and homoglyph Smuggling."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.smuggling.envelopes.smuggling_attack_envelope import SmugglingAttackEnvelope
from attack.smuggling.smuggling_options_utils import options, techniques
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run one recoverable Smuggling attempt with explicit local GPU setup."""
    parser = ArgumentParser(description="Request-bound Hypothetical/Homoglyph Smuggling adaptation")
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
    """Display the recorded outcome and successful technique when present.

    Args:
        path (Path): Completed journal destination.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Validate and reserve an independent journal before executing Smuggling.

    Args:
        arguments (Namespace): Journal, request, GPU, token and technique settings.

    Raises:
        ValueError: If GPU, token limit or technique selection is invalid.
        FileExistsError: If the journal is already reserved.
    """
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid GPU or token limit")
    techniques(arguments.smuggling_techniques)
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    _ = SmugglingAttackEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        VersionedJournal(
            DiskJournal(arguments.journal)
        ),
        techniques=arguments.smuggling_techniques
    ).run()


if __name__ == "__main__":
    main()

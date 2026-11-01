"""Standalone offline ThreatenJSON command for one explicit request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.envelopes.threaten_json_attack_envelope import ThreatenJSONAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run one bound ThreatenJSON attack from its own Python entry point."""
    parser = ArgumentParser(description="Standalone offline ThreatenJSON")
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
    """Display the completed result from its journal.

    Args:
        path (Path): Completed attempt journal.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Reserve an unused journal before any model exchange.

    Args:
        arguments (Namespace): Explicit request, CUDA device and token limit.

    Raises:
        ValueError: If a limit is invalid.
        FileExistsError: If the journal already exists.
    """
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid ThreatenJSON GPU or token limit")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    journal = VersionedJournal(
        DiskJournal(arguments.journal)
    )
    _ = ThreatenJSONAttackEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        journal
    ).run()


if __name__ == "__main__":
    main()

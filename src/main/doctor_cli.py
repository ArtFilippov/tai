"""Standalone request-bound Doctor with recoverable outcomes."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.doctor.doctor_options_utils import options, selection
from attack.doctor.envelopes.doctor_attack_envelope import DoctorAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run one independently journaled Doctor attempt with explicit GPU setup."""
    parser = ArgumentParser(description="Request-bound Doctor adaptation from garak")
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
    """Display the recorded outcome and successful candidate when present.

    Args:
        path (Path): Completed journal destination.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Validate and reserve a fresh journal before executing Doctor.

    Args:
        arguments (Namespace): Journal, base request, GPU, token and Doctor settings.

    Raises:
        ValueError: If GPU, token limit or technique selection is invalid.
        FileExistsError: If the journal is already reserved.
    """
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid GPU or token limit")
    _ = selection(arguments.doctor_techniques)
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    _ = DoctorAttackEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        VersionedJournal(
            DiskJournal(arguments.journal)
        ),
        techniques=arguments.doctor_techniques, seed=arguments.doctor_seed
    ).run()


if __name__ == "__main__":
    main()

"""Standalone offline Bad Characters deletion over a supplied request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.deletion.deletion_budget import DeletionBudget
from attack.deletion.envelopes.deletion_attack_envelope import DeletionAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def bounds(arguments: Namespace) -> DeletionBudget:
    """Validate native deletion limits before model access.

    Args:
        arguments (Namespace): Requested bounds.

    Returns:
        DeletionBudget: Checked run parameters.
    """
    budget = DeletionBudget(
        arguments.perturbations, arguments.positions, arguments.ascii
    )
    budget.validate()
    return budget


def launch(arguments: Namespace) -> None:
    """Validate and reserve a journal before accessing model weights.

    Args:
        arguments (Namespace): Request, destination and bounds.

    Raises:
        ValueError: If a limit is invalid.
        FileExistsError: If the journal is already reserved.
    """
    budget = bounds(arguments)
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid deletion GPU or token limit")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    device = cuda_device(arguments.gpu)
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    model = LazyQwenModel(
        device, arguments.tokens
    )
    roles = SharedModelAttackSetup(
        model, arguments.base
    )
    _ = DeletionAttackEnvelope(
        base=PlainBase(arguments.base),
        setup=roles,
        journal=VersionedJournal(
            DiskJournal(arguments.journal)
        ),
        budget=budget
    ).run()


def main() -> None:
    """Run deletion via its dedicated Python module."""
    parser = ArgumentParser(description="Standalone offline Bad Characters deletion")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--perturbations", type=int, default=1
    )
    _ = parser.add_argument(
        "--positions", type=int, default=8
    )
    _ = parser.add_argument(
        "--ascii", type=int, default=4
    )
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    arguments = parser.parse_args()
    path = arguments.journal
    launch(arguments)
    display(path)


def display(path: Path) -> None:
    """Print the saved completion.

    Args:
        path (Path): Completed deletion journal.
    """
    print(
        ParsedJournal(path).read()
    )


if __name__ == "__main__":
    main()

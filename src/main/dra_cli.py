"""Standalone offline DRA over an explicit base request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.dra_budget import DRABudget
from attack.dra_limits_utils import validate
from attack.envelopes.dra_attack_envelope import DRAAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def bounds(arguments: Namespace) -> DRABudget:
    """Validate candidate bounds before model access.

    Args:
        arguments (Namespace): Requested candidate count and seed.

    Returns:
        DRABudget: Checked per-run parameters.
    """
    budget = DRABudget(
        arguments.candidates, arguments.seed
    )
    validate(budget)
    return budget


def display(path: Path) -> None:
    """Print the saved attempt outcome.

    Args:
        path (Path): Completed run journal.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Validate bounds and reserve an independent run before loading weights.

    Args:
        arguments (Namespace): Request, output and local limits.

    Raises:
        ValueError: If model or candidate limits are invalid.
        FileExistsError: If the journal is already reserved.
    """
    budget = bounds(arguments)
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid DRA GPU or token limit")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    _ = snapshot(True)
    device = cuda_device(arguments.gpu)
    model = LazyQwenModel(
        device, arguments.tokens
    )
    arguments.journal.touch(exist_ok=False)
    base = PlainBase(arguments.base)
    journal = VersionedJournal(
        DiskJournal(arguments.journal)
    )
    _ = DRAAttackEnvelope(
        base,
        SharedModelAttackSetup(
            model, arguments.base
        ),
        journal,
        budget
    ).run()


def main() -> None:
    """Run one DRA attempt through its own Python entry point."""
    parser = ArgumentParser(description="Standalone offline DRA")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument(
        "--seed", type=int, default=0
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--candidates", type=int, default=4
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    arguments = parser.parse_args()
    launch(arguments)
    display(arguments.journal)


if __name__ == "__main__":
    main()

"""Standalone offline AutoDAN GA/HGA command for one request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.autodan_budget import AutoDANBudget
from attack.autodan_budget_utils import validate
from attack.autodan_limits import AutoDANLimits
from attack.envelopes.autodan_attack_envelope import AutoDANAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.base_request import BaseRequest
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.journal import Journal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.local_qwen_assets import LocalQwenAssets
from target.qwen_device_utils import cuda_device
from target.qwen_logits import QwenLogits
from target.qwen_snapshot_utils import snapshot
from target.logits import Logits
from target.model import Model


def main() -> None:
    """Run bounded GA or HGA from its own Python entry point."""
    parser = ArgumentParser(description="Standalone offline AutoDAN GA/HGA")
    _ = parser.add_argument(
        "--mode",
        choices=(
            "ga", "hga"
        ),
        default="ga"
    )
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--target", required=True, help="desired assistant completion"
    )
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    _ = parser.add_argument(
        "--steps", type=int, default=2
    )
    _ = parser.add_argument(
        "--batch", type=int, default=3
    )
    _ = parser.add_argument(
        "--elites", type=int, default=1
    )
    _ = parser.add_argument(
        "--points", type=int, default=5
    )
    _ = parser.add_argument(
        "--crossover", type=float, default=0.5
    )
    _ = parser.add_argument(
        "--mutation", type=float, default=0.1
    )
    _ = parser.add_argument(
        "--seed", type=int, default=0
    )
    arguments = parser.parse_args()
    launch(arguments)
    display(arguments.journal)


def bounds(arguments: Namespace) -> AutoDANLimits:
    """Validate and expose native search limits before local model access.

    Args:
        arguments (Namespace): Explicit CLI options.

    Returns:
        AutoDANLimits: Checked GA limits.
    """
    limits = AutoDANBudget(
        arguments.steps, arguments.batch, arguments.elites, arguments.points,
        arguments.crossover, arguments.mutation, arguments.seed, arguments.target,
        arguments.mode
    )
    validate(limits)
    return limits


def launch(arguments: Namespace) -> None:
    """Validate all bounds before reserving the journal and loading weights.

    Args:
        arguments (Namespace): Explicit request, model and GA bounds.

    Raises:
        ValueError: If the bounds are invalid.
        FileExistsError: If the journal already exists.
    """
    _ = bounds(arguments)
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid AutoDAN GPU or token limit")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    device = cuda_device(arguments.gpu)
    _ = snapshot(True)
    model: Model = LazyQwenModel(
        device, arguments.tokens
    )
    logits: Logits = QwenLogits(
        LocalQwenAssets(device),
        device
    )
    arguments.journal.touch(exist_ok=False)
    base: BaseRequest = PlainBase(arguments.base)
    journal: Journal = VersionedJournal(
        DiskJournal(arguments.journal)
    )
    _ = AutoDANAttackEnvelope(
        base,
        SharedModelAttackSetup(
            model, arguments.base
        ),
        journal, logits=logits,
        budget=bounds(arguments)
    ).run()


def display(path: Path) -> None:
    """Print the recovered terminal outcome.

    Args:
        path (Path): Completed independent journal.
    """
    print(
        ParsedJournal(path).read()
    )


if __name__ == "__main__":
    main()

"""Standalone request-bound local BEAST attempt."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.beast.beast_options_utils import options
from attack.beast.beast_probe import BeastProbe
from attack.beast.beam_search import BeamSearch
from attack.beast.configured_beast_budget import ConfiguredBeastBudget
from attack.beast.envelopes.beast_attack_envelope import BeastAttackEnvelope
from attack.desired_response.cached_desired_response import CachedDesiredResponse
from attack.desired_response.desired_response import DesiredResponse
from attack.desired_response.provided_desired_response import ProvidedDesiredResponse
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.envelopes.qwen_token_probabilities import QwenTokenProbabilities
from target.local_qwen_assets import LocalQwenAssets
from target.model_chat import ModelChat
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run one local BEAST search and display its recovered outcome."""
    parser = ArgumentParser(description="Request-bound BEAST against pinned local Qwen")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    group = parser.add_mutually_exclusive_group(required=True)
    _ = group.add_argument("--desired-response")
    _ = group.add_argument(
        "--deferred", action="store_true"
    )
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
    """Display the recovered result.

    Args:
        path (Path): Completed journal.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Validate input and reserve an independent attempt journal.

    Args:
        arguments (Namespace): Request, answer and local model settings.

    Raises:
        ValueError: If limits or the answer source are invalid.
        FileExistsError: If the journal is already reserved.
    """
    limits = ConfiguredBeastBudget(
        arguments.beast_beams, arguments.beast_samples,
        arguments.beast_suffix_len, arguments.beast_trials
    )
    limits.validate()
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid GPU or token limit")
    if (arguments.desired_response is None) != arguments.deferred:
        raise ValueError("Supply exactly one of --desired-response and --deferred")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    device = cuda_device(arguments.gpu)
    model = LazyQwenModel(
        device, arguments.tokens
    )
    base = PlainBase(arguments.base)
    source: DesiredResponse
    if arguments.deferred:
        source = CachedDesiredResponse(
            ModelChat(model)
        )
    else:
        source = ProvidedDesiredResponse(
            base, arguments.desired_response
        )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    journal = VersionedJournal(
        DiskJournal(arguments.journal)
    )
    tokens = QwenTokenProbabilities(
        LocalQwenAssets(device), device
    )
    _ = BeastAttackEnvelope(
        base,
        SharedModelAttackSetup(
            model, arguments.base
        ),
        journal,
        BeastProbe(
            arguments.base, source,
            BeamSearch(
                tokens, limits
            ),
            journal
        )
    ).run()


if __name__ == "__main__":
    main()

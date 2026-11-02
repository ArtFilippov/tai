"""Standalone offline WordNet exploration for an explicit base request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import wn
from wn._config import config

from attack.shared_model_attack_setup import SharedModelAttackSetup
from attack.wordnet.envelopes.wordnet_attack_envelope import WordnetAttackEnvelope
from attack.wordnet.prepared_wordnet import PreparedWordnet
from attack.wordnet.wordnet_source_utils import open_wordnet
from attack.wordnet.wordnet_budget import WordnetBudget
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def prepare() -> None:
    """Explicitly download and install the pinned lexicon before offline inference."""
    parser = ArgumentParser(description="Prepare Open English WordNet 2023")
    _ = parser.add_argument(
        "directory", type=Path
    )
    directory = parser.parse_args().directory
    directory.mkdir(
        parents=True, exist_ok=True
    )
    config.data_directory = directory
    if not wn.lexicons(lexicon="oewn:2023"):
        _ = wn.download("oewn:2023")
    _ = open_wordnet(directory)


def main() -> None:
    """Run a bounded search and display its recoverable result."""
    parser = ArgumentParser(description="Standalone offline WordNet exploration")
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--lexicon-dir", type=Path, required=True
    )
    _ = parser.add_argument(
        "--nodes", type=int, default=4
    )
    _ = parser.add_argument(
        "--calls", type=int, default=8
    )
    _ = parser.add_argument(
        "--seed", type=int, default=0
    )
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
    """Display the selected result of a completed attempt.

    Args:
        path (Path): Completed journal.
    """
    print(
        ParsedJournal(path).read()
    )


def launch(arguments: Namespace) -> None:
    """Check offline prerequisites before reserving the journal.

    Args:
        arguments (Namespace): Explicit lexicon, request, model and limits.

    Raises:
        ValueError: If limits or the lexicon are invalid.
        FileExistsError: If the journal already exists.
    """
    budget = WordnetBudget(
        arguments.nodes, arguments.calls, arguments.seed
    )
    budget.validate()
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid WordNet GPU or token limit")
    lexicon = PreparedWordnet(
        open_wordnet(arguments.lexicon_dir)
    )
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    arguments.journal.touch(exist_ok=False)
    _ = WordnetAttackEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        VersionedJournal(
            DiskJournal(arguments.journal)
        ),
        lexicon, budget=budget
    ).run()


if __name__ == "__main__":
    main()

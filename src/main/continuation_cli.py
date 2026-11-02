"""Standalone Slur Completion and Leak Replay commands."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.desired_response.cached_desired_response import CachedDesiredResponse
from attack.desired_response.desired_response import DesiredResponse
from attack.desired_response.provided_desired_response import ProvidedDesiredResponse
from attack.leak_replay.envelopes.leak_replay_attack_envelope import LeakReplayAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from attack.slur_completion.envelopes.slur_completion_attack_envelope import SlurCompletionAttackEnvelope
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.model_chat import ModelChat
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def command(kind: str) -> None:
    """Run one explicit request using a supplied or deferred answer.

    Args:
        kind (str): Selected continuation strategy.
    """
    parser = ArgumentParser(description=kind)
    _ = parser.add_argument(
        "journal", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument("--desired-response")
    _ = parser.add_argument(
        "--deferred", action="store_true"
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    arguments = parser.parse_args()
    launch(
        arguments, kind
    )
    display(arguments.journal)


def display(path: Path) -> None:
    """Display the selected result and assessment.

    Args:
        path (Path): Completed attempt journal.
    """
    print(
        ParsedJournal(path).read()
    )


def slur_main() -> None:
    """Start a Slur Completion command."""
    command("slur_completion")


def leak_main() -> None:
    """Start a Leak Replay command."""
    command("leak_replay")


def launch(arguments: Namespace, kind: str) -> None:
    """Reserve a journal and run one bound continuation attempt.

    Args:
        arguments (Namespace): Request, answer and runtime settings.
        kind (str): Selected continuation strategy.

    Raises:
        ValueError: If the selected strategy or limits are invalid.
        FileExistsError: If the journal path already exists.
    """
    if arguments.tokens < 1 or arguments.gpu < 0:
        raise ValueError("Invalid GPU or token limit")
    if (arguments.desired_response is None) != arguments.deferred:
        raise ValueError("Supply exactly one of --desired-response and --deferred")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
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
    setup = SharedModelAttackSetup(
        model, arguments.base
    )
    journal = VersionedJournal(
        DiskJournal(arguments.journal)
    )
    if kind == "slur_completion":
        _ = SlurCompletionAttackEnvelope(
            base, setup, journal, source
        ).run()
    elif kind == "leak_replay":
        _ = LeakReplayAttackEnvelope(
            base, setup, journal, source
        ).run()
    else:
        raise ValueError("Unknown continuation strategy")

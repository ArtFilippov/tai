"""Standalone FalseAssertion attempt for an explicit base request."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from attack.desired_response.cached_desired_response import CachedDesiredResponse
from attack.desired_response.desired_response import DesiredResponse
from attack.desired_response.provided_desired_response import ProvidedDesiredResponse
from attack.false_assertion.envelopes.false_assertion_attack_envelope import FalseAssertionAttackEnvelope
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.model_chat import ModelChat
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def main() -> None:
    """Run and display one recoverable FalseAssertion attempt."""
    parser = ArgumentParser(description="Standalone FalseAssertion adaptation")
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
        "--tokens", type=int, default=192
    )
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
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
    """Reserve a journal and run one request-bound attempt.

    Args:
        arguments (Namespace): Supplied or deferred answer and model settings.

    Raises:
        ValueError: If arguments contain invalid limits or answer selection.
        FileExistsError: If the journal is already reserved.
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
    _ = FalseAssertionAttackEnvelope(
        base,
        SharedModelAttackSetup(
            model, arguments.base
        ),
        VersionedJournal(
            DiskJournal(arguments.journal)
        ),
        source
    ).run()


if __name__ == "__main__":
    main()

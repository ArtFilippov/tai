"""Standalone past/future Phrasing with two durable independent outcomes."""

import json
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from pathlib import Path

from attack.phrasing.envelopes.disk_phrasing_pair_envelope import DiskPhrasingPairEnvelope
from attack.phrasing.phrasing_options_utils import options, validate
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_device_utils import cuda_device
from target.qwen_snapshot_utils import snapshot


def launch(arguments: Namespace) -> Mapping[str, Mapping[str, object]]:
    """Validate and reserve a fresh pair of journals before any inference.

    Args:
        arguments (Namespace): New directory, request, GPU and generation limits.

    Returns:
        Mapping[str, Mapping[str, object]]: Both independent outcomes.

    Raises:
        ValueError: If GPU, token or preparation limits are invalid.
        FileExistsError: If the output directory already exists.
    """
    if arguments.gpu < 0 or arguments.tokens < 1:
        raise ValueError("Invalid GPU or token limit")
    validate(
        "past", arguments.phrasing_attempts
    )
    if arguments.output_dir.exists():
        raise FileExistsError(arguments.output_dir)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    return DiskPhrasingPairEnvelope(
        PlainBase(arguments.base),
        SharedModelAttackSetup(
            model, arguments.base
        ),
        arguments.output_dir, attempts=arguments.phrasing_attempts
    ).run()


def display(outcomes: Mapping[str, Mapping[str, object]]) -> None:
    """Display both recovered outcomes and report any technical failure.

    Args:
        outcomes (Mapping[str, Mapping[str, object]]): Independent recorded results.

    Raises:
        SystemExit: With code one if either tense failed technically.
    """
    print(
        json.dumps(
            outcomes, ensure_ascii=False, indent=2
        )
    )
    if any(outcome["status"] == "failure" for outcome in outcomes.values()):
        raise SystemExit(1)


def main() -> None:
    """Run and display both independently journaled tense experiments."""
    parser = ArgumentParser(description="Request-bound Phrasing: independent past then future")
    _ = parser.add_argument(
        "output_dir", type=Path
    )
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    options(parser)
    display(
        launch(
            parser.parse_args()
        )
    )


if __name__ == "__main__":
    main()

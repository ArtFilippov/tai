"""Sequential ALERT attacks over a validated pool of base requests."""

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path

from attack.attack import Attack
from attack.envelopes.fitd_attack_envelope import FITDAttackEnvelope
from attack.envelopes.goat_attack_envelope import GOATAttackEnvelope
from attack.envelopes.threaten_json_attack_envelope import ThreatenJSONAttackEnvelope
from attack.autodan_budget import AutoDANBudget
from attack.autodan_budget_utils import validate
from attack.envelopes.autodan_attack_envelope import AutoDANAttackEnvelope
from attack.fitd_setup import FITDSetup
from attack.run_capture import RunCapture
from base.alert_identity import AlertIdentity
from base.alert_pool import AlertPool
from base.base_request import BaseRequest
from benchmark.alert_series import AlertSeries
from journal.disk_journal import DiskJournal
from journal.journal import Journal
from journal.versioned_journal import VersionedJournal
from target.gguf.envelopes.gguf_profile import GGUFProfile
from target.gguf.envelopes.alert_gguf_file import AlertGGUFFile
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_snapshot_utils import snapshot
from target.gguf.gguf_location import GGUFLocation
from target.model import Model
from target.qwen_device_utils import cuda_device
from target.local_qwen_assets import LocalQwenAssets
from target.qwen_logits import QwenLogits

GGUF = "gguf"
OUTCOME = "outcome"
THREATEN_JSON = 2


def main() -> None:
    """Parse a new output directory and the local model limits."""
    parser = ArgumentParser(description="Sequential offline ALERT FITD runs")
    _ = parser.add_argument(
        "output_dir", type=Path
    )
    _ = parser.add_argument(
        "--manifest", type=Path,
        default=Path("examples/alert-manifest-640.json")
    )
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--limit", type=int, default=20
    )
    _ = parser.add_argument(
        "--turns", type=int, default=10
    )
    _ = parser.add_argument(
        "--goat-turns", type=int, default=5
    )
    _ = parser.add_argument(
        "--no-threaten-json", action="store_true",
        help="omit the ThreatenJSON variant"
    )
    _ = parser.add_argument(
        "--autodan", action="store_true", help="include local AutoDAN GA"
    )
    _ = parser.add_argument(
        "--autodan-target", default="Sure, here is the answer"
    )
    _ = parser.add_argument(
        "--autodan-steps", type=int, default=2
    )
    _ = parser.add_argument(
        "--autodan-batch", type=int, default=3
    )
    _ = parser.add_argument(
        "--autodan-elites", type=int, default=1
    )
    _ = parser.add_argument(
        "--autodan-points", type=int, default=5
    )
    _ = parser.add_argument(
        "--autodan-crossover", type=float, default=0.5
    )
    _ = parser.add_argument(
        "--autodan-mutation", type=float, default=0.1
    )
    _ = parser.add_argument(
        "--autodan-seed", type=int, default=0
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    _ = parser.add_argument(
        "--attacker", choices=("qwen", GGUF), default="qwen"
    )
    _ = parser.add_argument(
        "--host", default="127.0.0.1"
    )
    _ = parser.add_argument(
        "--port", type=int, default=8080
    )
    launch(
        parser.parse_args()
    )


def launch(arguments: Namespace) -> None:
    """Validate the entire pool and reserve the output before model access.

    Args:
        arguments (Namespace): Manifest, limits and output directory.

    Raises:
        ValueError: If the run limits are invalid.
    """
    if min(arguments.turns, arguments.goat_turns, arguments.tokens) < 1 or arguments.gpu < 0 or not 1 <= arguments.port <= 65535:
        raise ValueError("Invalid GPU, turn, token or port limit")
    if arguments.autodan:
        limits = budget(arguments)
        validate(limits)
    pool = AlertPool(arguments.manifest)
    bases = pool.requests(arguments.limit)
    arguments.output_dir.mkdir(
        parents=True, exist_ok=False
    )
    with prepare(
        arguments, arguments.output_dir
    ) as factories:
        series(
            bases, factories, arguments.output_dir, arguments.manifest
        )
    print(arguments.output_dir / "summary.json")


def planned(bases: Sequence[AlertIdentity], factories: Sequence[Callable[[BaseRequest], Attack]]) -> int:
    """Count the full Cartesian product before the first attempt.

    Args:
        bases (Sequence[AlertIdentity]): Validated requests.
        factories (Sequence[Callable[[BaseRequest], Attack]]): Attack variants.

    Returns:
        int: Planned attempts.
    """
    return len(bases) * len(factories)


def series(
    bases: Sequence[AlertIdentity], factories: Sequence[Callable[[BaseRequest], Attack]],
    directory: Path, manifest: Path,
) -> None:
    """Run every factory over the validated requests and save known results.

    Args:
        bases (Sequence[AlertIdentity]): Validated selection.
        factories (Sequence[Callable[[BaseRequest], Attack]]): Bound attack compositions.
        directory (Path): Reserved output directory for the whole launch.
        manifest (Path): Source manifest.
    """
    summary = AlertSeries(
        manifest, directory,
        planned(
            bases, factories
        )
    )
    summary.save()
    number = 0
    for index, factory in enumerate(
        factories, start=1
    ):
        for base in bases:
            number += 1
            invoke(
                base, factory, index, number, summary=summary
            )


def invoke(
    base: BaseRequest, factory: Callable[[BaseRequest], Attack],
    index: int, number: int, *, summary: AlertSeries,
) -> None:
    """Keep one attempt's available result or error in the root index.

    Args:
        base (BaseRequest): Selected source request.
        factory (Callable[[BaseRequest], Attack]): Current bound attack variant.
        index (int): Current factory number.
        number (int): Global attempt number.
        summary (AlertSeries): Root summary for the launch.

    Raises:
        KeyboardInterrupt: After recording an interrupted attempt.
    """
    entry: dict[str, object] = {
        "number": number,
        "factory": index,
        "base": base.text(),
        "base_metadata": dict(
            base.metadata()
        )
    }
    attempt = RunCapture()
    try:
        result(
            base, factory, entry, attempt
        )
    except KeyboardInterrupt:
        entry[OUTCOME] = "interrupted"
        raise
    finally:
        if OUTCOME not in entry:
            if attempt.failed():
                entry[OUTCOME] = "failure"
                entry["error"] = attempt.error()
            else:
                entry[OUTCOME] = "completed"
        summary.record(entry)


def result(
    base: BaseRequest, factory: Callable[[BaseRequest], Attack],
    entry: dict[str, object], attempt: RunCapture,
) -> None:
    """Run a fresh attack and retain its description alongside its result.

    Args:
        base (BaseRequest): Selected request.
        factory (Callable[[BaseRequest], Attack]): Bound attack composition.
        entry (dict[str, object]): Partial index entry to complete.
        attempt (RunCapture): Independent failure boundary.
    """
    with attempt.guard():
        attack = factory(base)
        entry["attack"] = attack.description()
        entry["result"] = attack.run()


def journal(path: Path) -> Journal:
    """Reserve one independent journal in its attack's output directory.

    Args:
        path (Path): Unique destination for the attempt.

    Returns:
        Journal: Versioned event sink for the attack.
    """
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    path.touch(exist_ok=False)
    return VersionedJournal(
        DiskJournal(path)
    )


def context(arguments: Namespace, model: Model) -> AbstractContextManager[Model]:
    """Keep an optional GGUF Attacker alive for the whole invocation.

    Args:
        arguments (Namespace): Model and CUDA settings.
        model (Model): Shared Qwen resource for all roles by default.

    Returns:
        AbstractContextManager[Model]: Selected Attacker resource.
    """
    return (
        GGUFProfile(AlertGGUFFile(), GGUFLocation(arguments.host, arguments.port), arguments.gpu, arguments.tokens).open()
        if arguments.attacker == GGUF else nullcontext(model)
    )


def budget(arguments: Namespace) -> AutoDANBudget:
    """Validate the optional native GA bounds before model access.

    Args:
        arguments (Namespace): Selected GA settings.

    Returns:
        AutoDANBudget: Validated reproducible bounds.
    """
    return AutoDANBudget(
        arguments.autodan_steps, arguments.autodan_batch,
        arguments.autodan_elites, arguments.autodan_points,
        arguments.autodan_crossover, arguments.autodan_mutation,
        arguments.autodan_seed, arguments.autodan_target
    )


def ga_setup(attacker: Model, target: Model, base: BaseRequest) -> FITDSetup:
    """Bind an independent Attacker and pinned Target to one request.

    Args:
        attacker (Model): Mutation role.
        target (Model): Target and assessment resource.
        base (BaseRequest): Selected source request.

    Returns:
        FITDSetup: Complete GA role binding.
    """
    return FITDSetup(
        attacker, target, target,
        base.text()
    )


@contextmanager
def prepare(arguments: Namespace, directory: Path) -> Generator[list[Callable[[BaseRequest], Attack]]]:
    """Hold model resources and expose only request-to-attack factories.

    Args:
        arguments (Namespace): Selected models and limits.
        directory (Path): Root of this launch's artifacts.

    Yields:
        list[Callable[[BaseRequest], Attack]]: Independent per-request attacks.
    """
    def factories(attacker: Model, target: Model) -> list[Callable[[BaseRequest], Attack]]:
        attacks: list[Callable[[BaseRequest], Attack]] = [
            lambda base: FITDAttackEnvelope(
                base, FITDSetup(attacker, target, target, base.text()),
                journal(directory / "fitd" / f"{base.metadata()['id']}.jsonl"),
                arguments.turns
            ),
            lambda base: GOATAttackEnvelope(
                base, FITDSetup(attacker, target, target, base.text()),
                journal(directory / "goat" / f"{base.metadata()['id']}.jsonl"),
                arguments.goat_turns
            ),
            lambda base: ThreatenJSONAttackEnvelope(
                base, FITDSetup(attacker, target, target, base.text()),
                journal(directory / "threaten_json" / f"{base.metadata()['id']}.jsonl")
            )
        ]
        if arguments.autodan:
            device = cuda_device(arguments.gpu)
            scorer = QwenLogits(
                LocalQwenAssets(device),
                device
            )
            attacks.append(lambda base: AutoDANAttackEnvelope( base, ga_setup(attacker, target, base), journal(directory / "autodan" / f"{base.metadata()['id']}.jsonl"), logits=scorer, budget=budget(arguments) ))
        return [attack for index, attack in enumerate(attacks) if index != THREATEN_JSON or not arguments.no_threaten_json]

    model = LazyQwenModel(
        cuda_device(arguments.gpu), arguments.tokens
    )
    _ = snapshot(True)
    with context(
        arguments, model
    ) as attacker:
        yield factories(
            attacker, model
        )

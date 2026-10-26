"""Sequential ALERT runs and a compact index derived from saved journals."""

from argparse import ArgumentParser, Namespace
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import override

from huggingface_hub import snapshot_download

from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from attack.model_run import ModelRun
from attack.run_capture import RunCapture
from attack.run_capture_contract import RunCaptureContract
from base.alert_identity import AlertIdentity
from base.alert_pool import AlertPool
from base.alert_taxonomy import AlertTaxonomy
from base.base_request import BaseRequest
from base.plain_base import PlainBase
from benchmark.alert_archive import AlertArchive
from benchmark.alert_ports import AlertPorts
from benchmark.alert_series import AlertSeries
from benchmark.recorded_alert_archive import RecordedAlertArchive
from journal.disk_journal import DiskJournal
from journal.journal import Journal
from journal.journal_view import JournalView
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.envelopes.gguf_profile import GGUFProfile
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.gguf_file import GGUFFile
from target.gguf_location import GGUFLocation
from target.model import Model
from target.model_chat import ModelChat
from target.qwen_device_utils import cuda_device

GGUF = "gguf"
PREPARE = "prepare"


class DiskAlertPorts(AlertPorts):
    """Open a fresh disk journal or recover one already written."""

    @override
    def journal(self, path: Path) -> Journal:
        """Create the journal for one attempt.

        Args:
            path (Path): Reserved attempt path.

        Returns:
            Journal: Append-only event sink.
        """
        return VersionedJournal(
            DiskJournal(path)
        )

    @override
    def view(self, path: Path) -> JournalView:
        """Recover the saved record of an attempt.

        Args:
            path (Path): Finished journal path.

        Returns:
            JournalView: Recorded outcome and exchanges.
        """
        return ParsedJournal(path)

    @override
    def capture(self) -> RunCaptureContract:
        """Create a fresh failure boundary for the next run.

        Returns:
            RunCaptureContract: Isolated failure state.
        """
        return RunCapture()


def outcome(path: Path) -> dict[str, str]:
    """Classify a finished journal through the assembled archive.

    Args:
        path (Path): Saved journal of an attempt.

    Returns:
        dict[str, str]: Outcome with an optional error.
    """
    return archive().outcome(path)


def archive() -> AlertArchive:
    """Bind the journal format and complete ALERT taxonomy.

    Returns:
        AlertArchive: Independent series storage boundary.
    """
    return RecordedAlertArchive(
        DiskAlertPorts(),
        AlertTaxonomy()
    )


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
        "--tokens", type=int, default=192
    )
    _ = parser.add_argument(
        "--attacker",
        choices=(
            "qwen", "gguf"
        ),
        default="qwen"
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
    if arguments.turns < 1 or arguments.tokens < 1 or arguments.gpu < 0 or not 1 <= arguments.port <= 65535:
        raise ValueError("Invalid GPU, turn, token or port limit")
    pool = AlertPool(arguments.manifest)
    bases = pool.requests(arguments.limit)
    arguments.output_dir.mkdir(
        parents=True, exist_ok=False
    )
    series = assemble_series(
        bases,
        pool.taxonomy(),
        arguments.output_dir, arguments.manifest
    )
    series.save()
    with session(arguments) as runs:
        series.run(runs)
    print(arguments.output_dir / "summary.json")


def assemble_series(
    bases: Sequence[AlertIdentity], taxonomy: Mapping[str, object],
    directory: Path, manifest: Path,
) -> AlertSeries:
    """Bind validated requests to a fresh journal archive.

    Args:
        bases (Sequence[AlertIdentity]): Validated selection.
        taxonomy (Mapping[str, object]): Published macro groups.
        directory (Path): Reserved output directory.
        manifest (Path): Source manifest.

    Returns:
        AlertSeries: Series ready for independent attempts.
    """
    return AlertSeries(
        bases, taxonomy, directory, manifest, archive=archive()
    )


def text(base: BaseRequest) -> str:
    """Read the original request for the three independent FITD roles.

    Args:
        base (BaseRequest): Selected request.

    Returns:
        str: Original request text.
    """
    return base.text()


def gguf() -> GGUFFile:
    """Identify the pinned optional Attacker file.

    Returns:
        GGUFFile: Pinned local asset identity.
    """
    return GGUFFile(
        "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF",
        "7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c",
        "Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf"
    )


def snapshot(offline: bool) -> str:
    """Locate the pinned Qwen snapshot, downloading only during preparation.

    Args:
        offline (bool): Require an already prepared local snapshot.

    Returns:
        str: Local snapshot path.

    Raises:
        FileNotFoundError: If the requested snapshot is unavailable locally.
    """
    try:
        return snapshot_download(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            local_files_only=offline,
        )
    except OSError as exc:
        raise FileNotFoundError("Prepare the pinned Qwen snapshot before running offline") from exc


def run(base: BaseRequest, path: Path, model: Model, attacker: Model, *, turns: int) -> None:
    """Build fresh FITD roles and a journal for one attempt.

    Args:
        base (BaseRequest): Original request and provenance.
        path (Path): Unique journal destination.
        model (Model): Shared Target and judge resource.
        attacker (Model): Shared Attacker resource.
        turns (int): Maximum FITD turns.
    """
    journal = VersionedJournal(
        DiskJournal(path)
    )
    _ = FITDAttack(
        target=ModelChat(model),
        judge=RecordedJudge(
            QwenJudge(
                text(base),
                ModelChat(model)
            ),
            journal,
            text(base)
        ),
        journal=journal,
        probe=FITDProbe(
            text(base),
            ContinuingChat(
                ModelChat(attacker)
            ),
            journal, turns
        )
    ).run(base)


@contextmanager
def session(arguments: Namespace) -> Generator[ModelRun]:
    """Hold the selected Attacker resource across all attempts.

    Args:
        arguments (Namespace): Model and CUDA settings.

    Yields:
        ModelRun: Shared resources for independent attempts.
    """
    model = LazyQwenModel(
        cuda_device(arguments.gpu), arguments.tokens
    )
    _ = snapshot(True)
    context = (
        GGUFProfile(gguf(), GGUFLocation(arguments.host, arguments.port), arguments.gpu, arguments.tokens).open()
        if arguments.attacker == GGUF else nullcontext(model)
    )
    with context as attacker:
        yield assemble(
            model, attacker, arguments.turns
        )


def assemble(model: Model, attacker: Model, turns: int) -> ModelRun:
    """Bind reusable resources to fresh per-attempt composition.

    Args:
        model (Model): Shared Target and judge resource.
        attacker (Model): Shared Attacker resource.
        turns (int): Maximum FITD turns.

    Returns:
        ModelRun: An isolated attempt for each invocation.
    """
    return ModelRun(
        model, attacker, turns,
        lambda base, path, model, attacker, turns=turns: run(base, path, model, attacker, turns=turns)
    )


def qwen_main() -> None:
    """Prepare pinned weights or run a local CUDA FITD experiment."""
    parser = ArgumentParser(description="Local Qwen FITD experiment")
    commands = parser.add_subparsers(
        dest="command", required=True
    )
    prepare = commands.add_parser(
        PREPARE, help="Download pinned model and tokenizer"
    )
    _ = prepare.add_argument(
        "--attacker",
        choices=(
            "qwen", GGUF
        ),
        default="qwen"
    )
    command = commands.add_parser(
        "run", help="Run offline FITD"
    )
    _ = command.add_argument(
        "journal", type=Path
    )
    _ = command.add_argument("base")
    _ = command.add_argument(
        "--gpu", type=int, required=True
    )
    _ = command.add_argument(
        "--turns", type=int, default=10
    )
    _ = command.add_argument(
        "--tokens", type=int, default=192
    )
    _ = command.add_argument(
        "--attacker",
        choices=(
            "qwen", GGUF
        ),
        default="qwen"
    )
    _ = command.add_argument(
        "--host", default="127.0.0.1"
    )
    _ = command.add_argument(
        "--port", type=int, default=8080
    )
    qwen_launch(
        parser.parse_args()
    )


def qwen_launch(arguments: Namespace) -> None:
    """Assemble one run from parsed command arguments.

    Args:
        arguments (Namespace): Parsed command arguments.

    Raises:
        ValueError: If the round limit is invalid.
        FileExistsError: If the journal already exists.
    """
    if arguments.command == PREPARE:
        print(gguf().prepare() if arguments.attacker == GGUF else snapshot(False))
        return
    if arguments.turns < 1:
        raise ValueError("FITD round limit must be positive")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    with session(arguments) as runs:
        arguments.journal.touch(exist_ok=False)
        runs.run(
            PlainBase(arguments.base),
            arguments.journal
        )
    print(
        ParsedJournal(arguments.journal).read()
    )

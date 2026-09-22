"""Compose parallel ALERT experiments, direct observation, and owned resources."""

import platform
from dataclasses import asdict, replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from uuid import uuid4

from alert.manifest import load_alert_manifest
from alert.records import AlertManifest
from attacks.fitd.garak_adapter import GARAK_REVISION
from attacks.fitd.profile import FITD_PROFILE
from core.primitives.messages import Message
from core.runs.console_journal import ConsoleRunJournal

from .configuration import RoleProfile, load_profile
from .contracts import Clock, RunHandle
from .experiments import bind_experiment
from .runtime import Interrupts, utc_now
from .series_contracts import SeriesDependencies, SeriesIdentity, SeriesResult
from .series_execution import SeriesExecution, validate_jobs
from .series_storage import SeriesFiles
from .series_console import SeriesConsole

__all__ = ["SeriesResult", "run_alert_series"]


def _settings(profile: RoleProfile) -> dict[str, object]:
    return {
        "tai_version": version("tai"), "python": platform.python_version(), "platform": platform.platform(),
        "garak_revision": GARAK_REVISION,
        "fitd": asdict(replace(FITD_PROFILE, max_calls_per_conv=profile.max_calls_per_conv)),
        **profile.settings(),
    }


def _create(profile: RoleProfile, directory: Path, identity: SeriesIdentity, base: Message, *, console: SeriesConsole) -> RunHandle:
    row = console.row(identity.sequence, identity.micro_category)
    fresh = replace(profile.fresh(), journal_factory=partial(ConsoleRunJournal, row))
    return _bind(fresh, base, directory)


def _bind(profile: RoleProfile, base: Message, directory: Path) -> RunHandle:
    user = profile.create_experiment(base, directory)
    return bind_experiment(user, directory)


def _execute_series(manifest: AlertManifest, profile: RoleProfile, store: SeriesFiles, jobs: int) -> SeriesResult:
    server = profile.assembly.settings if profile.assembly is not None else dict
    with Interrupts() as stop, SeriesConsole(len(manifest.records), server=server) as console:
        create = partial(_create, profile, store.directory / "runs", console=console)
        dependencies = SeriesDependencies(create, store, stop, Clock(utc_now, monotonic), store.redact, console.notify)
        try:
            return SeriesExecution(manifest.records, dependencies, jobs).run()
        finally:
            profile.close()


def _series_files(
    profile: RoleProfile, paths: dict[str, Path], inputs: dict[str, bytes], directory: Path,
) -> SeriesFiles:
    return SeriesFiles(
        directory, inputs, paths, partial(_settings, profile), profile.redactor().text, _settings(profile),
    )


def run_alert_series(
    manifest_path: Path, profile_path: Path, artifacts_dir: Path, *, taxonomy_path: Path, jobs: int = 4,
) -> SeriesResult:
    """Validate all inputs, then admit independent FITD workers in manifest order.

    Main thread only (owns SIGINT temporarily). One loaded profile holds shared
    model resources. Configuration/series I/O errors raise; run-local errors are
    saved and continued after joining the worker and releasing its participants.

    Args:
        manifest_path (Path): Validated saved ALERT selection.
        profile_path (Path): Pinned participant TOML configuration.
        artifacts_dir (Path): Parent of the new unique series directory.
        taxonomy_path (Path): Pinned local taxonomy document.
        jobs (int, default=4): Positive bound on workers including cleanup and saving.

    Returns:
        SeriesResult: Saved directory, accounting report, and command exit code.

    Raises:
        ValueError: An input file changed during validation.
    """
    validate_jobs(jobs)
    directory = artifacts_dir / str(uuid4())
    paths = {"manifest": manifest_path, "profile": profile_path, "taxonomy": taxonomy_path}
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    manifest = load_alert_manifest(manifest_path, taxonomy_path=taxonomy_path)
    profile = load_profile(profile_path, alert_series=True, server_log=directory / "vllm.log")
    if any(path.read_bytes() != inputs[name] for name, path in paths.items()):
        raise ValueError("Series input files changed during validation")
    return _execute_series(manifest, profile, _series_files(profile, paths, inputs, directory), jobs)

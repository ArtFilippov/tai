"""Compose sequential ALERT experiments and their external persistence."""

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

from .configuration import RoleProfile, load_profile
from .contracts import Clock, RunHandle
from .experiments import bind_experiment
from .runtime import Interrupts, utc_now
from .series_contracts import SeriesDependencies, SeriesResult
from .series_execution import SeriesExecution
from .series_storage import SeriesFiles

__all__ = ["SeriesResult", "run_alert_series"]


def _settings(profile: RoleProfile) -> dict[str, object]:
    return {
        "tai_version": version("tai"), "python": platform.python_version(), "platform": platform.platform(),
        "garak_revision": GARAK_REVISION,
        "fitd": asdict(replace(FITD_PROFILE, max_calls_per_conv=profile.max_calls_per_conv)),
        **profile.settings(),
    }


def _create(profile: RoleProfile, directory: Path, base: Message) -> RunHandle:
    user = profile.create_experiment(base, directory)
    return bind_experiment(user, directory)


def _execute_series(manifest: AlertManifest, profile: RoleProfile, store: SeriesFiles) -> SeriesResult:
    with Interrupts() as stop:
        dependencies = SeriesDependencies(partial(_create, profile, store.directory / "runs"), store, stop,
                                          Clock(utc_now, monotonic), store.redact)
        return SeriesExecution(manifest.records, dependencies).run()


def _series_files(
    profile: RoleProfile, paths: dict[str, Path], inputs: dict[str, bytes], artifacts_dir: Path,
) -> SeriesFiles:
    return SeriesFiles(
        artifacts_dir / str(uuid4()), inputs, paths, partial(_settings, profile), profile.redactor().text, _settings(profile),
    )


def run_alert_series(
    manifest_path: Path, profile_path: Path, artifacts_dir: Path, *, taxonomy_path: Path,
) -> SeriesResult:
    """Validate all inputs, then run one FITD per saved record in order.

    Main thread only (owns SIGINT temporarily). One loaded profile holds shared
    Qwen weights. Configuration/series I/O errors raise; run-local errors are
    saved and continued after joining the worker and releasing its participants.

    Args:
        manifest_path (Path): Validated saved ALERT selection.
        profile_path (Path): Pinned participant TOML configuration.
        artifacts_dir (Path): Parent of the new unique series directory.
        taxonomy_path (Path): Pinned local taxonomy document.

    Returns:
        SeriesResult: Saved directory, accounting report, and command exit code.

    Raises:
        ValueError: An input file changed during validation.
    """
    paths = {"manifest": manifest_path, "profile": profile_path, "taxonomy": taxonomy_path}
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    manifest = load_alert_manifest(manifest_path, taxonomy_path=taxonomy_path)
    profile = load_profile(profile_path, alert_series=True)
    if any(path.read_bytes() != inputs[name] for name, path in paths.items()):
        raise ValueError("Series input files changed during validation")
    return _execute_series(manifest, profile, _series_files(profile, paths, inputs, artifacts_dir))

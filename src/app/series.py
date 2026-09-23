"""Compose parallel ALERT experiments, direct observation, and owned resources."""

import platform
from dataclasses import replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import TypedDict, Unpack
from uuid import uuid4

from alert.manifest import load_alert_manifest
from alert.records import AlertManifest
from attacks.shared.contracts import GARAK_REVISION
from core.primitives.messages import Message
from core.runs.console_journal import ConsoleRunJournal

from .configuration import RoleProfile, load_profile
from .contracts import Clock, RunHandle
from .experiments import bind_experiment
from .runtime import Interrupts, utc_now
from .series_contracts import SeriesDependencies, SeriesIdentity, SeriesResult
from .series_execution import SeriesExecution, validate_jobs
from .series_storage import SeriesFiles
from .series_console import DEFAULT_LINES_PER_ATTACK, SeriesConsole, validate_lines_per_attack

__all__ = ["SeriesOptions", "SeriesResult", "run_alert_series"]


class SeriesOptions(TypedDict, total=False):
    jobs: int
    lines_per_attack: int


def _settings(profile: RoleProfile) -> dict[str, object]:
    strategy = profile.strategy()
    return {
        "tai_version": version("tai"), "python": platform.python_version(), "platform": platform.platform(),
        "garak_revision": GARAK_REVISION,
        "attack": {"type": strategy.attack_type}, strategy.attack_type: dict(strategy.parameters),
        **profile.settings(),
    }


def _create(profile: RoleProfile, directory: Path, identity: SeriesIdentity, base: Message, *, console: SeriesConsole) -> RunHandle:
    row = console.row(identity.sequence, identity.micro_category)
    fresh = replace(profile.fresh(), journal_factory=partial(ConsoleRunJournal, row))
    return _bind(fresh, base, directory)


def _bind(profile: RoleProfile, base: Message, directory: Path) -> RunHandle:
    user = profile.create_experiment(base, directory)
    return bind_experiment(user, directory)


def _backend(profile: RoleProfile) -> str:
    labels = {"qwen": "Transformers", "vllm": "vLLM", "openai": "OpenAI-compatible"}
    backends = {role: settings.get("backend") if isinstance(settings := chat.settings(), dict) else None
                for role, chat in profile.chats().items()}
    roles = {role: labels.get(str(backend), "unknown") for role, backend in backends.items()}
    if len(set(roles.values())) == 1:
        return next(iter(roles.values()))
    return " · ".join(f"{role}: {backend}" for role, backend in roles.items())


def _execute_series(manifest: AlertManifest, profile: RoleProfile, store: SeriesFiles, jobs: int, *, lines_per_attack: int) -> SeriesResult:
    with Interrupts() as stop, SeriesConsole(len(manifest.records), backend=_backend(profile), lines_per_attack=lines_per_attack) as console:
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


def _prepare_series(
    manifest_path: Path, profile_path: Path, artifacts_dir: Path, taxonomy_path: Path,
) -> tuple[AlertManifest, RoleProfile, SeriesFiles]:
    directory = artifacts_dir / str(uuid4())
    paths = {"manifest": manifest_path, "profile": profile_path, "taxonomy": taxonomy_path}
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    manifest = load_alert_manifest(manifest_path, taxonomy_path=taxonomy_path)
    profile = load_profile(profile_path, alert_series=True, server_log=directory / "vllm.log")
    if any(path.read_bytes() != inputs[name] for name, path in paths.items()):
        raise ValueError("Series input files changed during validation")
    return manifest, profile, _series_files(profile, paths, inputs, directory)


def _limits(options: SeriesOptions) -> tuple[int, int]:
    if unknown := options.keys() - {"jobs", "lines_per_attack"}:
        raise TypeError(f"Unexpected series options: {', '.join(sorted(unknown))}")
    jobs, lines = options.get("jobs", 4), options.get("lines_per_attack", DEFAULT_LINES_PER_ATTACK)
    validate_jobs(jobs)
    validate_lines_per_attack(lines)
    return jobs, lines


def run_alert_series(
    manifest_path: Path, profile_path: Path, artifacts_dir: Path, *, taxonomy_path: Path, **options: Unpack[SeriesOptions],
) -> SeriesResult:
    """Validate all inputs, then admit independent strategy workers in manifest order.

    Main thread only (owns SIGINT temporarily). One loaded profile holds shared
    model resources. Configuration/series I/O errors raise; run-local errors are
    saved and continued after joining the worker and releasing its participants.

    Args:
        manifest_path (Path): Validated saved ALERT selection.
        profile_path (Path): Pinned participant TOML configuration.
        artifacts_dir (Path): Parent of the new unique series directory.
        taxonomy_path (Path): Pinned local taxonomy document.
        **options (Unpack[SeriesOptions]): Positive jobs (default 4) and lines_per_attack (default 5).

    Returns:
        SeriesResult: Saved directory, accounting report, and command exit code.

    """
    jobs, lines = _limits(options)
    manifest, profile, store = _prepare_series(manifest_path, profile_path, artifacts_dir, taxonomy_path)
    return _execute_series(manifest, profile, store, jobs, lines_per_attack=lines)

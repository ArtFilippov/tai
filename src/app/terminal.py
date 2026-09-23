"""Compose foreground User control with its journal and external persistence."""

from collections.abc import Callable
from pathlib import Path

from attacks.shared.experiment import Experiment
from core.primitives.messages import Message

from .configuration import RoleProfile
from .experiments import bind_experiment
from .runtime import Interrupts
from .terminal_control import ProgressDisplay, control_foreground


def run_profile(profile: RoleProfile, base: Message, artifacts_dir: Path) -> int:
    experiment = profile.create_experiment(base, artifacts_dir)
    return run_foreground(experiment, artifacts_dir=artifacts_dir, redact=profile.redactor().text)


def run_foreground(user: Experiment, *, artifacts_dir: Path, redact: Callable[[str], str] = str) -> int:
    """Run on the main thread; SIGINT requests stop outside the signal handler.

    The handler never takes a User lock or raises through Thread.join. Repeated
    interrupts still let the worker finish its current call, then save externally.

    Args:
        user (Experiment): Composed experiment with independent journal.
        artifacts_dir (Path): Parent directory for final run artifacts.
        redact (Callable[[str], str], default=str): Policy applied only to terminal output.

    Returns:
        int: Zero for completion, one for failure, or 130 for interruption.
    """
    def show(text: str) -> None:
        safe = redact(text)
        print(safe, end="" if safe.endswith("\n") else "\n", flush=True)

    handle = bind_experiment(user, artifacts_dir)
    with Interrupts() as stop:
        return control_foreground(handle, ProgressDisplay(show), stop, artifacts_dir)

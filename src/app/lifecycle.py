"""Waiting and cleanup through the public User lifecycle."""

from itertools import count

from .contracts import RunControl, StopRequests

POLL_SECONDS = 0.05


def wait_for_run(user: RunControl, stop: StopRequests) -> None:
    for _ in count():
        if stop.interrupted:
            user.stop()
        if user.wait(POLL_SECONDS):
            return


def join_admitted(user: RunControl) -> None:
    # A wrapper can raise after start admitted a worker. A terminal status alone
    # does not release participants; join actual cleanup before saving/reuse.
    if user.status().run_id is not None:
        try:
            _stop_if_working(user)
        finally:
            _ = user.wait()


def _stop_if_working(user: RunControl) -> None:
    if not user.wait(0):
        user.stop()

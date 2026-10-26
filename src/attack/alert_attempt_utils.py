"""Persist each ALERT attempt and its available outcome."""

from collections.abc import Callable, Mapping

from attack.attack import Attack
from attack.run_capture_contract import RunCaptureContract
from base.base_request import BaseRequest
from benchmark.alert_series_contract import AlertSeriesContract

OUTCOME = "outcome"


def invoke(base: BaseRequest, factory: Callable[[BaseRequest], Attack], position: Mapping[str, int], *, summary: AlertSeriesContract, attempt: RunCaptureContract) -> None:
    """Keep one attempt's available result or error in the root index.

    Args:
        base (BaseRequest): Selected source request.
        factory (Callable[[BaseRequest], Attack]): Current bound attack variant.
        position (Mapping[str, int]): Factory and global attempt numbers.
        summary (AlertSeriesContract): Root summary for the launch.
        attempt (RunCaptureContract): Independent failure boundary.

    Raises:
        KeyboardInterrupt: After recording an interrupted attempt.
    """
    entry: dict[str, object] = {
        "number": position["number"],
        "factory": position["factory"],
        "base": base.text(),
        "base_metadata": dict(
            base.metadata()
        )
    }
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


def result(base: BaseRequest, factory: Callable[[BaseRequest], Attack], entry: dict[str, object], attempt: RunCaptureContract) -> None:
    """Run a fresh attack and retain its description alongside its result.

    Args:
        base (BaseRequest): Selected request.
        factory (Callable[[BaseRequest], Attack]): Bound attack composition.
        entry (dict[str, object]): Partial index entry to complete.
        attempt (RunCaptureContract): Independent failure boundary.
    """
    with attempt.guard():
        attack = factory(base)
        entry["attack"] = attack.description()
        entry["result"] = attack.run()

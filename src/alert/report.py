"""Offline series report files and compatibility exports for pure accounting."""

from pathlib import Path

from core.runs.storage import write_json

from .accounting import SeriesReport, build_series_report, classify_outcome
from .values import decode_json, object_value

__all__ = ["build_series_report", "classify_outcome", "reconstruct_report"]


def reconstruct_report(index_path: Path, output: Path | None = None) -> SeriesReport:
    """Read index.json only; optionally write a deterministic JSON report.

    The output parent must exist. The input index is never overwritten.

    Args:
        index_path (Path): Saved series index.
        output (Path | None, default=None): Optional destination for the rebuilt report.

    Returns:
        SeriesReport: Accounting computed exclusively from the index.

    Raises:
        ValueError: Output resolves to the input index.
    """
    if output is not None and output.resolve() == index_path.resolve():
        raise ValueError("Report output must differ from the series index")
    index = object_value(decode_json(index_path.read_bytes().decode("utf-8")), "Series index")
    report = build_series_report(index)
    if output is not None:
        write_json(output, report)
    return report

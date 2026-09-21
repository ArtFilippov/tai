"""Immutable diagnostic records detached from live execution and storage."""

import json
from dataclasses import dataclass, field

from core.primitives.diagnostics import DiagnosticObject, DiagnosticRedactor
from core.primitives.diagnostics.json_values import json_object

from .status import RunSnapshot


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Detached evidence ready to save, independent of restarts and output paths.

    Attributes:
        snapshot (RunSnapshot): Actual finished run state and original result.
        _metadata_json (str, default=field(repr=False)): Detached metadata serialized at capture.
    """

    snapshot: RunSnapshot
    _metadata_json: str = field(repr=False)

    @property
    def metadata(self) -> DiagnosticObject:
        return json_object(self._metadata_json)

    def dump_metadata(self) -> str:
        return self._metadata_json


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: str
    timestamp: str
    _data_json: str = field(repr=False)
    _redactor: DiagnosticRedactor = field(
        default_factory=DiagnosticRedactor, repr=False, compare=False
    )

    @property
    def data(self) -> DiagnosticObject:
        """Return detached diagnostics, never a source for a live Message.

        Returns:
            DiagnosticObject: A freshly decoded event snapshot.
        """
        return json_object(self._data_json)

    def dump(self) -> str:
        return json.dumps(self._redactor.snapshot({
            "run_id": self.run_id, "sequence": self.sequence, "kind": self.kind,
            "timestamp": self.timestamp, "data": self.data,
        }), ensure_ascii=False)

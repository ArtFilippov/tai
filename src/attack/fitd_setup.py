"""Role resources and request text for a FITD attempt."""

from typing import NamedTuple

from target.model import Model


FITDSetup = NamedTuple(
    "FITDSetup", [
        ("attacker", Model),
        ("target", Model),
        ("judge", Model),
        ("text", str)
    ]
)

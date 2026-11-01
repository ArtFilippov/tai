"""Bind independently selected models to one request."""

from typing import NamedTuple

from target.model import Model


ExplicitAttackSetup = NamedTuple(
    "ExplicitAttackSetup", [
        ("attacker", Model),
        ("target", Model),
        ("judge", Model),
        ("text", str)
    ]
)

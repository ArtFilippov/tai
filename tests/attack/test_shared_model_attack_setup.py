"""One history-free model can supply separate attack roles."""

import unittest

from attack.shared_model_attack_setup import SharedModelAttackSetup
from .model_fixture_utils import refusal


class TestSharedModelAttackSetup(unittest.TestCase):
    """Check one shared model binds every role and the exact request."""

    def test_shared_model_roles(self) -> None:
        model = refusal()
        setup = SharedModelAttackSetup(
            model, "Original request"
        )
        assert (setup.attacker, setup.target, setup.judge, setup.text) == (
            model, model, model, "Original request"
        ), "The shared model setup lost a role or the exact request"

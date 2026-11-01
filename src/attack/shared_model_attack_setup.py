"""Bind one history-free model to every role in an attack attempt."""

from typing import override

from attack.attack_setup import AttackSetup
from target.model import Model


class SharedModelAttackSetup(AttackSetup):
    """Share model weights without sharing conversations between roles.

    Attributes:
        model (Model): History-free model resource for all roles.
        base (str): Exact original request.
    """

    model: Model
    base: str

    def __init__(self, model: Model, base: str) -> None:
        """Bind a history-free model and exact request to every role.

        Args:
            model (Model): Shared model resource.
            base (str): Original request.

        """
        super().__init__()
        self.model = model
        self.base = base

    @property
    @override
    def attacker(self) -> Model:
        """Expose the shared Attacker model.

        Returns:
            Model: Shared model resource.
        """
        return self.model

    @property
    @override
    def target(self) -> Model:
        """Expose the shared Target model.

        Returns:
            Model: Shared model resource.
        """
        return self.model

    @property
    @override
    def judge(self) -> Model:
        """Expose the shared judgment model.

        Returns:
            Model: Shared model resource.
        """
        return self.model

    @property
    @override
    def text(self) -> str:
        """Expose the exact request for judgment and probing.

        Returns:
            str: Original request text.
        """
        return self.base

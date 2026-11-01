"""Role models and request text bound to an attack attempt."""

from typing import Protocol

from target.model import Model


class AttackSetup(Protocol):
    """Expose role models and the original request for one attempt."""

    @property
    def attacker(self) -> Model:
        """Return the Attacker's history-free model.

        Returns:
            Model: Attacker model resource.
        """
        ...

    @property
    def target(self) -> Model:
        """Return the Target's history-free model.

        Returns:
            Model: Target model resource.
        """
        ...

    @property
    def judge(self) -> Model:
        """Return the judge's history-free model.

        Returns:
            Model: Judgment model resource.
        """
        ...

    @property
    def text(self) -> str:
        """Return the exact original request.

        Returns:
            str: Original request text.
        """
        ...

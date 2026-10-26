"""Offline model responses for independent run tests."""

from unittest.mock import Mock

from target.model import Model


def refusal() -> Model:
    """Answer every FITD role without loading model weights.

    Returns:
        Model: A model responding with refusals.
    """
    model = Mock(spec=Model)
    model.generate.return_value = "I cannot help with that request."
    model.revision.return_value = "local"
    model.settings.return_value = {"revision": "local"}
    return model

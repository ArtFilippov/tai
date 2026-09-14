import socket

import pytest


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    """A hidden provider or download must fail, never access a real model."""

    def reject(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "create_connection", reject)

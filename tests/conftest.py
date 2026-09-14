import inspect
import os

import pytest
from http_peer import endpoint as endpoint


def _guard_connections():
    import ipaddress
    import socket

    def local(address):
        host = address[0]
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = host == "localhost"
        if not allowed:
            raise AssertionError(
                "External network access is forbidden in offline tests"
            )

    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
    )

    def connect(self, address):
        local(address)
        return originals[0](self, address)

    def connect_ex(self, address):
        local(address)
        return originals[1](self, address)

    def create_connection(address, *args, **kwargs):
        local(address)
        return originals[2](address, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    return originals


@pytest.fixture(autouse=True)
def offline_models(monkeypatch, tmp_path_factory):
    """Only loopback is allowed, including in isolated Python HTTP workers."""
    import socket

    originals = _guard_connections()
    bootstrap = tmp_path_factory.mktemp("offline-python")
    (bootstrap / "sitecustomize.py").write_text(
        inspect.getsource(_guard_connections) + "\n_guard_connections()\n"
    )
    monkeypatch.setenv(
        "PYTHONPATH", str(bootstrap) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = (
            originals
        )

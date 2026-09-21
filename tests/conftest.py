import inspect
import os
from collections.abc import Iterator

import offline_network
import pytest
from http_peer import endpoint

__all__ = ["endpoint"]


@pytest.fixture(autouse=True)
def offline_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Only loopback is allowed, including in isolated Python HTTP workers.

    Args:
        monkeypatch (pytest.MonkeyPatch): Environment overrides for worker startup.
        tmp_path_factory (pytest.TempPathFactory): Temporary bootstrap directory factory.

    Yields:
        None: The test runs with loopback-only connections.
    """
    bootstrap = tmp_path_factory.mktemp("offline-python")
    (bootstrap / "sitecustomize.py").write_text(
        inspect.getsource(offline_network) + "\ninstall_guard()\n", encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTHONPATH", str(bootstrap) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    restore = offline_network.install_guard()
    try:
        yield
    finally:
        restore()

"""Loopback-only socket guards, also serialized as standalone sitecustomize."""

import ipaddress
import socket
from collections.abc import Callable
from functools import wraps
from typing import cast


def _require_loopback(address: object) -> None:
    message = "External network access is forbidden in offline tests"
    assert isinstance(address, tuple) and address, message
    host = cast(tuple[object, ...], address)[0]
    assert isinstance(host, (str, bytes, int)), message
    try:
        allowed = ipaddress.ip_address(host).is_loopback
    except ValueError:
        allowed = host == "localhost"
    if not allowed:
        raise AssertionError(message)


def _guarded[**P, R](
    original: Callable[P, R], address_index: int,
) -> Callable[P, R]:
    @wraps(original)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        address = args[address_index] if len(args) > address_index else kwargs["address"]
        _require_loopback(address)
        return original(*args, **kwargs)

    return guarded


def install_guard() -> Callable[[], None]:
    originals = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    socket.socket.connect = _guarded(originals[0], 1)
    socket.socket.connect_ex = _guarded(originals[1], 1)
    socket.create_connection = _guarded(originals[2], 0)

    def restore() -> None:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals

    return restore

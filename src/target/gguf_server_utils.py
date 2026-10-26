"""Own a local GGUF server only when its address is vacant."""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

STARTUP_POLLS = 450


def identity(address: str, expected: str) -> None:
    """Require exactly the expected model at a listening address.

    Args:
        address (str): HTTP origin of the server.
        expected (str): Required model identity.

    Raises:
        RuntimeError: If the service cannot report its model identity.
        ConnectionError: If the occupied address does not serve models.
        ValueError: If the model response is malformed.
    """
    try:
        payload = models(address)
    except (OSError, ValueError) as exc:
        raise ConnectionError(f"GGUF address {address} is occupied but /v1/models is unusable: {exc}") from exc
    match payload:
        case {"data": [{"id": str() as name}]} if name == expected:
            return
        case {"data": list()}:
            raise RuntimeError(f"GGUF model identity mismatch at {address}; expected {expected}")
        case _:
            raise ValueError(f"Malformed /v1/models response at {address}")


def models(address: str) -> object:
    """Read the advertised model list from the selected service.

    Args:
        address (str): HTTP origin of the service.

    Returns:
        object: Decoded JSON response.
    """
    with urlopen(
        f"{address}/v1/models", timeout=2
    ) as response:
        return json.load(response)


def listening(host: str, port: int) -> bool:
    """Distinguish an unused TCP address from an occupied, unusable one.

    Args:
        host (str): Host of the service.
        port (int): TCP port of the service.

    Returns:
        bool: Whether something accepts connections.

    Raises:
        RuntimeError: If the address cannot be inspected.
    """
    try:
        connection = socket.create_connection(
            (
                host, port
            ),
            timeout=2
        )
    except ConnectionRefusedError:
        return False
    except OSError as exc:
        raise RuntimeError(f"Cannot inspect GGUF address {host}:{port}: {exc}") from exc
    connection.close()
    return True


def ready(address: str, host: str, port: int, process: subprocess.Popen[bytes], *, expected: str) -> None:
    """Wait at most 90 seconds for an owned server to report its model.

    Args:
        address (str): HTTP origin of the server.
        host (str): Selected listening host.
        port (int): Selected listening port.
        process (subprocess.Popen[bytes]): Owned server process.
        expected (str): Required model identity.

    Raises:
        RuntimeError: If startup fails or readiness times out.
    """
    deadline = time.monotonic() + 90
    for _ in range(STARTUP_POLLS):
        if time.monotonic() >= deadline:
            break
        if process.poll() is not None:
            raise RuntimeError(f"GGUF server exited during startup with code {process.returncode}")
        if listening(
            host, port
        ):
            try:
                identity(
                    address, expected
                )
            except ConnectionError:
                pass
            else:
                return
        time.sleep(0.2)
    raise RuntimeError(f"GGUF server at {address} was not ready within 90 seconds")


@contextmanager
def gguf_server(host: str, port: int, path: Path, gpu: int, *, expected: str) -> Generator[bool]:
    """Reuse a matching server or start and stop an owned CUDA process.

    Args:
        host (str): Server bind and connect host.
        port (int): Server bind and connect port.
        path (Path): Prepared GGUF file.
        gpu (int): Selected visible CUDA GPU.
        expected (str): Pinned model identity.

    Yields:
        bool: Whether this context owns the process.

    Raises:
        ValueError: If the TCP port is invalid.
        RuntimeError: If CUDA is missing or the server cannot start.
    """
    if not 0 < port < 65536:
        raise ValueError("GGUF port must be between 1 and 65535")
    address = f"http://{host}:{port}"
    if listening(
        host, port
    ):
        identity(
            address, expected
        )
        yield False
        return
    gpu_support()
    try:
        process = start(
            host, port, path, gpu, expected=expected
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start GGUF server at {address}: {exc}") from exc
    try:
        yield from serve(
            address, host, port, process, expected=expected
        )
    finally:
        stop(process)


def serve(address: str, host: str, port: int, process: subprocess.Popen[bytes], *, expected: str) -> Generator[bool]:
    """Expose an owned server only after it is ready.

    Args:
        address (str): HTTP origin.
        host (str): Listening host.
        port (int): Listening port.
        process (subprocess.Popen[bytes]): Owned server.
        expected (str): Required model identity.

    Yields:
        bool: Ownership of this server.
    """
    ready(
        address, host, port, process, expected=expected
    )
    yield True


def gpu_support() -> None:
    """Require the uv-managed Python server to have GPU offload compiled in.

    Raises:
        RuntimeError: If the optional dependency or CUDA backend is unavailable.
    """
    result = subprocess.run(
        [sys.executable, "-c", "from llama_cpp import llama_supports_gpu_offload; assert llama_supports_gpu_offload()"],
        capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            "The uv gguf extra needs a CUDA-enabled llama-cpp-python build: " + result.stderr.decode(errors="replace")
        )


def start(host: str, port: int, path: Path, gpu: int, *, expected: str) -> subprocess.Popen[bytes]:
    """Launch the selected model using the active uv environment.

    Args:
        host (str): Bind host.
        port (int): Bind port.
        path (Path): Prepared model file.
        gpu (int): Visible CUDA device.
        expected (str): Identity advertised by the process.

    Returns:
        subprocess.Popen[bytes]: Owned server process.
    """
    return subprocess.Popen(
        [
            sys.executable, "-m", "llama_cpp.server", "--model",
            str(path),
            "--model_alias", expected, "--host", host, "--port",
            str(port),
            "--n_gpu_layers", "-1", "--main_gpu", "0", "--chat_template_kwargs", '{"enable_thinking": false}'
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": visible_gpu(gpu)
        }
    )


def visible_gpu(index: int) -> str:
    """Map a Torch-visible index to the same CUDA device in the child.

    Args:
        index (int): GPU index visible to the parent process.

    Returns:
        str: Device identifier for the server's CUDA mask.

    Raises:
        ValueError: If a parent CUDA mask does not include the selected index.
    """
    if (mask := os.environ.get("CUDA_VISIBLE_DEVICES")) is None:
        return str(index)
    devices = mask.split(",")
    if index < 0 or index >= len(devices):
        raise ValueError(f"CUDA GPU {index} is not visible to the GGUF server")
    return devices[index].strip()


def stop(process: subprocess.Popen[bytes]) -> None:
    """Terminate only the process launched by this run.

    Args:
        process (subprocess.Popen[bytes]): Owned process.
    """
    if process.poll() is not None:
        return
    process.terminate()
    try:
        _ = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        _ = process.wait(timeout=5)

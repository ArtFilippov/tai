#!/usr/bin/env python3
"""Send a command to a persistent shell in the msu tmux session."""

import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


SCRIPT = Path(__file__).with_name("remote.py").read_bytes()
TOKEN = uuid.uuid4().hex
LIMIT = os.environ.get("MSU_TMUX_MAX_BYTES", "10485760")
interrupted = 0


def interrupt(signum, _frame):
    global interrupted
    interrupted = signum


def remote(action, *args):
    process = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "msu", "python3", "-u", "-", action, *args],
        input=SCRIPT,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            process.stderr.decode(errors="replace").strip().removeprefix("msu-tmux: ")
            or "SSH failed"
        )
    return process.stdout


def output(buffer, marker, started):
    if not started:
        index = buffer.find(marker)
        if index < 0:
            return buffer[-len(marker):], False, None
        end = buffer.find(b"\n", index + len(marker))
        if end < 0:
            return buffer[index:], False, None
        buffer = buffer[end + 1 :]
        started = True
    match = re.search(rb"\r?\n__MSU_END_" + TOKEN.encode() + rb"__:(\d+)\r?\n", buffer)
    if match:
        sys.stdout.buffer.write(buffer[: match.start()])
        sys.stdout.buffer.flush()
        return b"", True, int(match.group(1))
    length = len(TOKEN) + 48
    if len(buffer) > length:
        sys.stdout.buffer.write(buffer[:-length])
        sys.stdout.buffer.flush()
        buffer = buffer[-length:]
    return buffer, started, None


def run(window, command):
    if not window.isdecimal() or not command or not LIMIT.isdecimal() or int(LIMIT) < 1:
        raise ValueError("provide a window number, a nonempty command, and a positive output limit")
    pane = remote("inspect", window).decode().strip()
    if interrupted:
        return 130 if interrupted == signal.SIGINT else 143
    encoded = base64.b64encode(command.encode()).decode()
    buffer = b""
    started = False
    status = None
    overflow = False
    offset = 0
    grace = 0
    try:
        try:
            remote("prepare", window, pane, TOKEN, LIMIT, encoded)
            while not interrupted:
                state = json.loads(remote("poll", pane, TOKEN, LIMIT, str(offset)).decode())
                offset = state["offset"]
                overflow |= state["full"]
                buffer, started, found = output(
                    buffer + base64.b64decode(state["data"]),
                    ("__MSU_BEGIN_" + TOKEN + "__").encode(),
                    started,
                )
                if found is not None:
                    status = found
                if not state["alive"]:
                    raise RuntimeError("tmux pane closed before the command completed")
                if state["ready"]:
                    if status is None and not overflow:
                        grace += 1
                        if not state["exit"] or grace > 5:
                            raise RuntimeError("shell returned without a completion marker")
                    else:
                        break
                time.sleep(0.2)
        except RuntimeError:
            if not interrupted:
                raise
        if interrupted:
            remote("interrupt", pane, TOKEN)
            return 130 if interrupted == signal.SIGINT else 143
        if overflow:
            raise RuntimeError("command output exceeded the limit and is incomplete")
        return status
    finally:
        remote("finish", pane, TOKEN)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--recover":
        if not sys.argv[2].isdecimal():
            raise ValueError("window must be a number")
        remote("recover", sys.argv[2])
        return 0
    if len(sys.argv) != 3:
        raise ValueError("usage: run.py WINDOW 'COMMAND' | run.py --recover WINDOW")
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    return run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as error:
        print(f"msu-tmux: {error}", file=sys.stderr)
        sys.exit(125)

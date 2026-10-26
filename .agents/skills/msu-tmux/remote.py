#!/usr/bin/env python3
"""Perform atomic tmux operations on msu over individual SSH calls."""

import base64
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path.home() / ".cache" / "msu-tmux"


def tmux(*args, optional=False):
    process = subprocess.run(["tmux", *args], capture_output=True, check=False)
    if process.returncode and not optional:
        raise RuntimeError(process.stderr.decode(errors="replace").strip())
    return process.stdout.decode().strip() if not process.returncode else ""


def option(pane, name):
    return tmux("show-options", "-p", "-v", "-t", pane, name, optional=True)


def target(window):
    if not window.isdecimal():
        raise ValueError("window must be a number")
    panes = tmux(
        "list-panes", "-t", f"29a_fil:{window}",
        "-F", "#{pane_id} #{pane_active} #{pane_current_command} #{pane_pipe}",
    )
    return next((line.split() for line in panes.splitlines() if line.split()[1] == "1"), None)


def path(pane):
    if not pane.startswith("%") or not pane[1:].isdecimal():
        raise ValueError("invalid tmux pane")
    return ROOT / pane[1:]


def inspect(window):
    pane = target(window)
    if pane is None:
        raise RuntimeError("window has no active pane")
    if pane[2] != "bash" or pane[3] != "0":
        raise RuntimeError("pane is busy or already has an output pipe")
    print(pane[0])


def prepare(window, pane, token, limit, command):
    current = target(window)
    if current is None or current[0] != pane or current[2] != "bash" or current[3] != "0":
        raise RuntimeError("active pane changed or is busy")
    former = option(pane, "@msu_run")
    if former and former != option(pane, "@msu_ready"):
        raise RuntimeError("shell readiness is unconfirmed after an earlier run")
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ROOT.stat().st_mode & 0o077:
        raise RuntimeError("tmux state directory is accessible by other users")
    lock = path(pane)
    if former and lock.exists():
        if (lock / "token").read_text() != former:
            raise RuntimeError("output pipe ownership is uncertain")
        (lock / "output").unlink()
        (lock / "token").unlink()
        lock.rmdir()
    lock.mkdir(mode=0o700)
    (lock / "token").write_text(token)
    (lock / "output").touch(mode=0o600)
    tmux("set-option", "-p", "-t", pane, "@msu_run", token)
    tmux("set-option", "-p", "-t", pane, "@msu_ready", "")
    tmux("set-option", "-p", "-t", pane, "@msu_exit", "")
    pipe = (
        "import os,sys\n"
        "with open(sys.argv[1], 'ab', buffering=0) as output:\n"
        "    left = int(sys.argv[2])\n"
        "    while left:\n"
        "        data = os.read(0, min(65536, left))\n"
        "        if not data: break\n"
        "        output.write(data)\n"
        "        left -= len(data)\n"
    )
    tmux(
        "pipe-pane", "-O", "-t", pane,
        f"python3 -u -c {shlex.quote(pipe)} {shlex.quote(str(lock / 'output'))} {int(limit) + 1}",
    )
    payload = (
        f"printf '\\n__MSU_BEGIN_{token}__\\n'; "
        f'eval "$(printf %s {command} | base64 -d)"; '
        '__msu_status=$?; '
        'tmux set-option -p -t "$TMUX_PANE" @msu_exit "$__msu_status"; '
        f"printf '\\n__MSU_END_{token}__:%d\\n' \"$__msu_status\"; "
        'unset __msu_status'
    )
    encoded = base64.b64encode(payload.encode()).decode()
    shell = (
        '__msu_old=$(declare -p PROMPT_COMMAND 2>/dev/null || :); '
        'PROMPT_COMMAND=\'tmux set-option -p -t "$TMUX_PANE" @msu_ready '
        f'{token}; unset PROMPT_COMMAND; eval "$__msu_old"; unset __msu_old\'; '
        f'eval "$(printf %s {encoded} | base64 -d)"'
    )
    tmux("send-keys", "-l", "-t", pane, shell)
    tmux("send-keys", "-t", pane, "Enter")
    print(pane)


def poll(pane, token, limit, offset):
    lock = path(pane)
    if (lock / "token").read_text() != token:
        raise RuntimeError("pane belongs to another invocation")
    with (lock / "output").open("rb") as stream:
        stream.seek(int(offset))
        data = stream.read(65536)
        offset = stream.tell()
    print(json.dumps({
        "data": base64.b64encode(data).decode(),
        "offset": offset,
        "full": (lock / "output").stat().st_size > int(limit),
        "alive": bool(tmux("display-message", "-p", "-t", pane, "#{pane_id}", optional=True)),
        "ready": option(pane, "@msu_ready") == token,
        "exit": option(pane, "@msu_exit"),
    }))


def interrupt(pane, token):
    if option(pane, "@msu_run") == token and option(pane, "@msu_ready") != token and not option(pane, "@msu_exit"):
        tmux("send-keys", "-t", pane, "C-c")


def finish(pane, token):
    lock = path(pane)
    if not (lock / "token").exists() or (lock / "token").read_text() != token:
        if option(pane, "@msu_run") == token:
            raise RuntimeError("output pipe ownership is uncertain")
        return
    if option(pane, "@msu_run") == token:
        tmux("pipe-pane", "-t", pane, optional=True)
        if option(pane, "@msu_ready") == token:
            for name in ("@msu_run", "@msu_ready", "@msu_exit"):
                tmux("set-option", "-p", "-u", "-t", pane, name)
        else:
            return
    elif tmux("display-message", "-p", "-t", pane, "#{pane_id}", optional=True):
        raise RuntimeError("output pipe ownership is uncertain")
    (lock / "output").unlink()
    (lock / "token").unlink()
    lock.rmdir()


def recover(window):
    pane = target(window)
    if pane is None:
        raise RuntimeError("window has no active pane")
    lock = path(pane[0])
    if not (lock / "token").exists() or option(pane[0], "@msu_run") != (lock / "token").read_text():
        raise RuntimeError("cannot confirm ownership of the stale pipe")
    tmux("pipe-pane", "-t", pane[0])
    for name in ("@msu_run", "@msu_ready", "@msu_exit"):
        tmux("set-option", "-p", "-u", "-t", pane[0], name)
    (lock / "output").unlink()
    (lock / "token").unlink()
    lock.rmdir()


if __name__ == "__main__":
    try:
        {"inspect": inspect, "prepare": prepare, "poll": poll, "interrupt": interrupt,
         "finish": finish, "recover": recover}[sys.argv[1]](*sys.argv[2:])
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"msu-tmux: {error}", file=sys.stderr)
        sys.exit(1)

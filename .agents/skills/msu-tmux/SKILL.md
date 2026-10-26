---
name: msu-tmux
description: Run commands on msu in tmux session 29a_fil. Use when the user asks to do something on msu, including commands in a specific tmux window.
---

# MSU tmux

Run shell commands in the existing bash shell of the requested window with
`.agents/skills/msu-tmux/run.sh WINDOW 'COMMAND'` from this repository.
For example: `.agents/skills/msu-tmux/run.sh 0 'echo "Hello"'`.
Window numbers are required; ask which window when the user has not specified
one. The active pane is selected at invocation and remains fixed for that run.

The command runs in the pane's current shell: directory and environment changes
persist. Output from stdout and stderr is streamed together; the exit status is
the command's exit status. One invocation per pane can run at a time. The script
has no timeout; use `timeout 30s .agents/skills/msu-tmux/run.sh ...` or put
`timeout` inside COMMAND when needed. Ctrl+C and SIGTERM forward Ctrl+C to the
running tmux command.

Treat errors (missing window, busy pane, SSH loss, incomplete output, uncertain
shell readiness) as errors, not as permission to retry the command. The output
limit defaults to 10 MiB; set `MSU_TMUX_MAX_BYTES` before invocation to adjust
it. Foreground, non-interactive commands have attributable output. Background
jobs may print later and mingle with subsequent output. If an interrupted run
leaves a stale output pipe, inspect the pane before manually clearing it with
`.agents/skills/msu-tmux/run.sh --recover WINDOW`.
The shell must be idle before invocation: tmux detects external programs, but
cannot always detect a bash builtin waiting for input.

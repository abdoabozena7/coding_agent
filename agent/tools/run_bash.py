"""
run_bash — run a shell command (THE MOST POWERFUL, AND MOST DANGEROUS, TOOL).

This single tool unlocks almost everything: run tests, install packages, use
git, inspect the system — it can even stand in for the other tools (cat, ls,
sed). That breadth is the point, and also the risk: it can do anything your
shell can, including destructive things. Phase 7 adds a permission gate in
front of it.

We capture stdout, stderr, AND the exit code, and feed all three back. The exit
code + stderr are what let the agent DEBUG: run the tests, read the failure,
edit the code, re-run — until it's green.
"""

import shlex
import subprocess

TIMEOUT_SECONDS = 120       # don't let a command hang the agent forever
MAX_OUTPUT_CHARS = 10_000   # keep a chatty command from flooding the context

REQUIRES_APPROVAL = True  # default: arbitrary shell commands ask the human first

# --- Allow-list: which commands are safe enough to run WITHOUT asking? --------
#
# Confirming every `ls` and `git status` is friction with no safety payoff. So
# we auto-approve a small set of read-only commands and keep the prompt for
# everything else. The bar to get on this list is high: the command must only
# OBSERVE state, never change it. When in doubt, it stays off the list and the
# human gets asked (requires_approval below defaults to True).

# Plain commands that only read/inspect — never write, delete, or install.
_SAFE_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "echo", "printf",
    "date", "whoami", "id", "uname", "hostname",
    "grep", "rg", "find", "tree", "stat", "file", "which", "type", "env",
    "df", "du", "ps",
    "pytest",  # the agent's core debug loop — run tests, read failures, retry
}

# git is safe ONLY for these read-only subcommands. `git status` is fine; a bare
# `git` on the safe list would also wave through `git push`, `git reset --hard`,
# `git clean -fd` — so we gate on the subcommand, not just the program name.
# Deliberately omitted: `branch` and `remote`. They LOOK read-only (`git branch`
# lists) but are dual-use — `git branch -D x` deletes, `git remote add ...`
# writes — and not worth vetting flag-by-flag. When in doubt, ask.
_SAFE_GIT_SUBCOMMANDS = {
    "status", "log", "diff", "show",
    "ls-files", "rev-parse", "blame", "describe", "shortlog",
}

# Leading `git` global options; some consume the NEXT token as their value
# (e.g. `git -C <dir> log`). We skip past these to find the real subcommand.
_GIT_GLOBAL_OPTS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}

# `find` is read-only UNLESS one of these turns it into an action. If any
# appears, we fall back to asking.
_UNSAFE_FIND_FLAGS = {"-exec", "-execdir", "-ok", "-okdir", "-delete",
                      "-fprint", "-fprintf", "-fls"}

# Shell operators that can chain, redirect, or substitute hidden work into an
# otherwise-safe command (e.g. `ls; rm -rf .`, `cat x > y`, `echo $(curl ...)`).
# If the command string contains any of these, it is NOT auto-approved.
_SHELL_METACHARS = ("|", "&", ";", ">", "<", "`", "$(", "${", "(", ")", "\n")


def _git_subcommand(rest):
    """Find git's subcommand, skipping leading global options and their values.

    `git status` -> 'status'; `git -C dir log` -> 'log'. Returns None if no
    subcommand is present (a bare `git`).
    """
    i = 0
    while i < len(rest) and rest[i].startswith("-"):
        i += 2 if rest[i] in _GIT_GLOBAL_OPTS_WITH_VALUE else 1
    return rest[i] if i < len(rest) else None


def _is_safe(command: str) -> bool:
    """True if `command` only reads state and may run without confirmation."""
    stripped = command.strip()
    if not stripped:
        return False

    # Any chaining/redirection/substitution -> ask. We only auto-approve plain,
    # single commands; a pipeline can always be approved manually.
    if any(meta in stripped for meta in _SHELL_METACHARS):
        return False

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return False  # unbalanced quotes etc. — let a human look at it
    if not tokens:
        return False

    cmd, rest = tokens[0], tokens[1:]

    if cmd == "git":
        return _git_subcommand(rest) in _SAFE_GIT_SUBCOMMANDS

    # `python -m pytest ...` is just our test runner under another name.
    if cmd in {"python", "python3"} and rest[:2] == ["-m", "pytest"]:
        return True

    if cmd in _SAFE_COMMANDS:
        if cmd == "find" and any(t in _UNSAFE_FIND_FLAGS for t in rest):
            return False
        return True

    return False


def requires_approval(args: dict) -> bool:
    """Per-call permission decision (overrides the static REQUIRES_APPROVAL).

    Read-only commands on the allow-list run unprompted; everything else still
    asks. The registry calls this when present — see tools/__init__.py.
    """
    return not _is_safe(args.get("command", ""))

SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": (
            "Run a shell command and return its stdout, stderr, and exit code. "
            "Use this to run tests, inspect files, use git, install packages, or "
            "anything else you'd do in a terminal. The command runs in the "
            "current working directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
            },
            "required": ["command"],
        },
    },
}


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... (truncated at {MAX_OUTPUT_CHARS} characters)"
    return text


def run(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,              # run through the shell so pipes, globs, &&, etc. work
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {TIMEOUT_SECONDS} seconds"

    # Build a result the model can reason about: exit code first, then output.
    parts = [f"exit code: {result.returncode}"]
    if result.stdout:
        parts.append("stdout:\n" + _truncate(result.stdout))
    if result.stderr:
        parts.append("stderr:\n" + _truncate(result.stderr))
    if not result.stdout and not result.stderr:
        parts.append("(no output)")
    return "\n".join(parts)

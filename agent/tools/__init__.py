"""
The tool registry.

This package is the agent's "hands." Adding a new capability later is a
two-step change: create tools/<name>.py with a SCHEMA + run(), then add the
module to the TOOLS list below. Nothing else in the codebase changes.
"""

from . import read_file, list_files, grep, write_file, edit_file, run_bash

# Every tool module the agent can use.
TOOLS = [read_file, list_files, grep, write_file, edit_file, run_bash]

# The catalog we hand to the model on every API call.
TOOL_SCHEMAS = [t.SCHEMA for t in TOOLS]

# name -> module, so we can find the implementation when the model calls a tool.
_BY_NAME = {t.SCHEMA["function"]["name"]: t for t in TOOLS}


def requires_approval(name: str, args: dict | None = None) -> bool:
    """Does this tool need explicit human approval before it runs?

    Two ways a tool declares its needs, checked in order:

      1. a `requires_approval(args)` function — for tools whose answer depends on
         the *arguments* (e.g. run_bash auto-approves read-only commands like
         `ls` but still asks for `rm`);
      2. a static `REQUIRES_APPROVAL` flag — a flat yes/no for the whole tool.

    Unknown tools — or a tool that declares neither — default to True: when in
    doubt, ask. (The harness, not the tool, decides what to *do* with this — see
    main.py.)
    """
    tool = _BY_NAME.get(name)
    if tool is None:
        return True
    decider = getattr(tool, "requires_approval", None)
    if callable(decider):
        return decider(args or {})
    return getattr(tool, "REQUIRES_APPROVAL", True)


def run_tool(name: str, args: dict) -> str:
    """Look up a tool by name and run it.

    Any failure is turned into a string result rather than an exception. That
    matters: the string gets fed back to the model as the tool result, so a
    bad path or a missing tool lets the agent SEE the error and recover
    (e.g. try a different path) instead of crashing the whole program.
    """
    tool = _BY_NAME.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    try:
        return tool.run(**args)
    except Exception as e:
        return f"Error: {e}"

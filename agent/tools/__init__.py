"""
The tool registry.

This package is the agent's "hands." Adding a new capability later is a
two-step change: create tools/<name>.py with a SCHEMA + run(), then add the
module to the TOOLS list below. Nothing else in the codebase changes.
"""

from . import read_file, list_files, grep, write_file, edit_file, run_bash
from ._security import (
    ToolContext,
    ToolSecurityError,
    configure_workspace,
    get_tool_context,
    get_workspace,
    safe_os_error,
    workspace_context,
)
from ._validation import ToolArgumentError, validate_tool_arguments

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
    try:
        validated = validate_tool_arguments(tool.SCHEMA, {} if args is None else args)
    except (ToolArgumentError, TypeError, ValueError):
        # Malformed calls fail closed.  run_tool will return the precise
        # validation error without invoking the implementation.
        return True
    decider = getattr(tool, "requires_approval", None)
    if callable(decider):
        try:
            return bool(decider(validated))
        except Exception:
            return True
    return bool(getattr(tool, "REQUIRES_APPROVAL", True))


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
        validated = validate_tool_arguments(tool.SCHEMA, args)
        result = tool.run(**validated)
        if not isinstance(result, str):
            return "Error: tool returned a non-string result"
        return result
    except ToolArgumentError as e:
        return f"Error: invalid arguments: {e}"
    except ToolSecurityError as e:
        return f"Error: {e}"
    except OSError as e:
        return f"Error: operating-system failure: {safe_os_error(e)}"
    except Exception as e:
        return f"Error: tool failed unexpectedly ({type(e).__name__})"

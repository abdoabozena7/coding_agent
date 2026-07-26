"""
The tool registry.

This package is the agent's "hands." ``TOOL_SPECS`` is the single policy source
for schema, risk, mutation, lifecycle, availability, and execution metadata.
"""

from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    apply_patch,
    edit_file,
    grep,
    inspect_preview,
    install_dependencies,
    list_files,
    materialize_artifact,
    open_path,
    poll_process,
    preview_html,
    process_manager,
    read_file,
    read_process_output,
    run_bash,
    run_command,
    start_process,
    stop_preview,
    stop_process,
    web_preview,
    write_file,
)
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
from ._types import ToolExecutionResult, ToolSpec

def _spec(module: Any, risk: str, category: str, *, mutates: bool = False,
          paths: tuple[str, ...] = (), lifecycle: str = "one_shot",
          capability: str | None = None) -> ToolSpec:
    return ToolSpec(
        module.SCHEMA,
        module.run,
        risk,
        category,
        mutates_workspace=mutates,
        requires_approval=getattr(module, "requires_approval", getattr(module, "REQUIRES_APPROVAL", True)),
        path_fields=paths,
        lifecycle=lifecycle,
        capability=capability,
    )


TOOL_SPECS = (
    _spec(read_file, "low", "read", paths=("path",)),
    _spec(list_files, "low", "read", paths=("path",)),
    _spec(grep, "low", "read", paths=("path",)),
    _spec(write_file, "high", "write", mutates=True, paths=("path",)),
    _spec(edit_file, "high", "write", mutates=True, paths=("path",)),
    _spec(apply_patch, "high", "write", mutates=True),
    _spec(materialize_artifact, "high", "write", mutates=True, paths=("path",)),
    _spec(run_bash, "critical", "command", mutates=True),
    _spec(run_command, "critical", "command", mutates=True, paths=("cwd",)),
    _spec(install_dependencies, "critical", "install", mutates=True, paths=("directory",)),
    _spec(start_process, "critical", "process", mutates=True, paths=("cwd",), lifecycle="managed"),
    _spec(poll_process, "low", "process"),
    _spec(read_process_output, "low", "process"),
    _spec(stop_process, "high", "process", lifecycle="managed"),
    _spec(open_path, "high", "open", paths=("path",), lifecycle="external"),
    _spec(preview_html, "high", "preview", paths=("path",), lifecycle="managed", capability="browser"),
    _spec(inspect_preview, "low", "preview", lifecycle="managed", capability="browser"),
    _spec(stop_preview, "high", "preview", lifecycle="managed", capability="browser"),
)

# Backward-compatible module list plus a central metadata registry.
TOOLS = [
    read_file, list_files, grep, write_file, edit_file, apply_patch,
    materialize_artifact, run_bash, run_command, install_dependencies,
    start_process, poll_process, read_process_output, stop_process, open_path,
    preview_html, inspect_preview, stop_preview,
]
TOOL_SCHEMAS = [dict(spec.schema) for spec in TOOL_SPECS]
_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def get_spec(name: str) -> ToolSpec | None:
    return _BY_NAME.get(name)


def names(*, categories: Iterable[str] | None = None, mutating: bool | None = None) -> frozenset[str]:
    wanted = set(categories or ())
    return frozenset(
        spec.name for spec in TOOL_SPECS
        if (not wanted or spec.category in wanted)
        and (mutating is None or spec.mutates_workspace is mutating)
    )


def risk_map() -> dict[str, str]:
    return {spec.name: spec.risk for spec in TOOL_SPECS}


def capability_report() -> tuple[dict[str, Any], ...]:
    browser = web_preview.browser_capability()
    result = []
    for spec in TOOL_SPECS:
        available = True
        detail = ""
        if spec.capability == "browser":
            available = bool(browser.get("available") and browser.get("playwright"))
            detail = f"browser={browser.get('channel') or 'missing'}, playwright={'yes' if browser.get('playwright') else 'no'}"
        result.append({
            "name": spec.name,
            "category": spec.category,
            "risk": spec.risk,
            "mutates_workspace": spec.mutates_workspace,
            "approval": "required" if spec.approval_required({}) else "not_required",
            "lifecycle": spec.lifecycle,
            "available": available,
            "detail": detail,
        })
    return tuple(result)


def register_artifact_provider(workspace: str | Path, provider: Any) -> None:
    materialize_artifact.register_provider(workspace, provider)


def shutdown_workspace_resources(workspace: str | Path) -> None:
    process_manager.shutdown_workspace(workspace)
    web_preview.shutdown_workspace(workspace)
    materialize_artifact.unregister_provider(workspace)


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
    spec = _BY_NAME.get(name)
    if spec is None:
        return True
    try:
        validated = validate_tool_arguments(spec.schema, {} if args is None else args)
    except (ToolArgumentError, TypeError, ValueError):
        # Malformed calls fail closed.  run_tool will return the precise
        # validation error without invoking the implementation.
        return True
    try:
        return spec.approval_required(validated)
    except Exception:
        return True


def run_tool(name: str, args: dict) -> str:
    """Look up a tool by name and run it.

    Any failure is turned into a string result rather than an exception. That
    matters: the string gets fed back to the model as the tool result, so a
    bad path or a missing tool lets the agent SEE the error and recover
    (e.g. try a different path) instead of crashing the whole program.
    """
    return run_tool_detailed(name, args).output


def run_tool_detailed(name: str, args: dict) -> ToolExecutionResult:
    spec = _BY_NAME.get(name)
    if spec is None:
        return ToolExecutionResult(False, f"Error: unknown tool '{name}'", error_code="unknown_tool")
    try:
        validated = validate_tool_arguments(spec.schema, args)
        result = spec.runner(**validated)
        if isinstance(result, ToolExecutionResult):
            return result
        if not isinstance(result, str):
            return ToolExecutionResult(False, "Error: tool returned an invalid result", error_code="invalid_result")
        paths = tuple(
            str(validated.get(field, "")).strip()
            for field in spec.path_fields
            if str(validated.get(field, "")).strip() not in {"", "."}
        )
        return ToolExecutionResult.from_output(result, changed_paths=paths if spec.mutates_workspace else ())
    except ToolArgumentError as e:
        return ToolExecutionResult(False, f"Error: invalid arguments: {e}", error_code="invalid_arguments")
    except ToolSecurityError as e:
        return ToolExecutionResult(False, f"Error: {e}", error_code="security")
    except OSError as e:
        return ToolExecutionResult(False, f"Error: operating-system failure: {safe_os_error(e)}", error_code="os_error")
    except Exception as e:
        return ToolExecutionResult(False, f"Error: tool failed unexpectedly ({type(e).__name__})", error_code="unexpected")


__all__ = [
    "TOOL_SCHEMAS", "TOOL_SPECS", "TOOLS", "ToolContext", "ToolExecutionResult",
    "ToolSecurityError", "ToolSpec", "capability_report", "configure_workspace",
    "get_spec", "get_tool_context", "get_workspace", "names", "requires_approval",
    "risk_map", "run_tool", "run_tool_detailed", "shutdown_workspace_resources",
    "workspace_context", "register_artifact_provider",
]

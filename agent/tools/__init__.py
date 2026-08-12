"""
The tool registry.

This package is the agent's "hands." ``TOOL_SPECS`` is the single policy source
for schema, risk, mutation, lifecycle, availability, and execution metadata.
"""

from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    apply_patch,
    browser_act,
    browser_close,
    browser_inspect,
    browser_open,
    browser_screenshot,
    browser_session,
    edit_file,
    grep,
    inspect_preview,
    inspect_images,
    install_dependencies,
    list_files,
    materialize_artifact,
    open_path,
    poll_process,
    publish_output,
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
from ._types import MutationFootprintV1, ToolExecutionResult, ToolSpec

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
        derived_mutation_paths=getattr(module, "derived_mutation_paths", None),
        result_contract=dict(getattr(module, "RESULT_CONTRACT", {})),
    )


TOOL_SPECS = (
    _spec(read_file, "low", "read", paths=("path",)),
    _spec(list_files, "low", "read", paths=("path",)),
    _spec(grep, "low", "read", paths=("path",)),
    _spec(inspect_images, "low", "read", capability="vision"),
    _spec(browser_open, "high", "preview", paths=("path",), lifecycle="managed", capability="browser"),
    _spec(browser_inspect, "low", "preview", lifecycle="managed", capability="browser"),
    _spec(browser_act, "high", "preview", lifecycle="managed", capability="browser"),
    _spec(browser_screenshot, "low", "preview", mutates=True, lifecycle="managed", capability="browser"),
    _spec(browser_close, "low", "preview", lifecycle="managed", capability="browser"),
    _spec(publish_output, "low", "output"),
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
    read_file, list_files, grep, inspect_images,
    browser_open, browser_inspect, browser_act, browser_screenshot, browser_close,
    publish_output, write_file, edit_file, apply_patch,
    materialize_artifact, run_bash, run_command, install_dependencies,
    start_process, poll_process, read_process_output, stop_process, open_path,
    preview_html, inspect_preview, stop_preview,
]
TOOL_SCHEMAS = [dict(spec.schema) for spec in TOOL_SPECS]
_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def get_spec(name: str) -> ToolSpec | None:
    return _BY_NAME.get(name)


def applicability_issue(
    name: str,
    args: Mapping[str, Any],
    workspace: str | Path,
) -> str:
    """Return a read-only applicability error before any approval is requested."""

    if str(name) == "install_dependencies":
        return install_dependencies.dependency_applicability_issue(
            workspace,
            str(args.get("directory") or "."),
            str(args.get("manager") or "auto"),
        )
    if str(name) != "preview_html":
        return ""
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return "preview_html requires an existing .html or .htm file"
    root = Path(workspace).resolve(strict=False)
    target = Path(raw_path)
    target = target.resolve(strict=False) if target.is_absolute() else (root / target).resolve(strict=False)
    if not target.is_relative_to(root):
        return "preview_html path must stay inside the configured workspace"
    if target.suffix.casefold() not in {".html", ".htm"}:
        return "preview_html requires an existing .html or .htm file"
    if not target.is_file():
        return f"preview_html path does not exist: {raw_path}"
    return ""


def mutation_footprint(
    name: str,
    args: Mapping[str, Any],
    accepted_paths: Iterable[str],
) -> MutationFootprintV1:
    """Return the reviewed paths plus narrowly declared tool side effects."""

    spec = _BY_NAME.get(name)
    derived: Iterable[str] = ()
    if spec is not None and spec.derived_mutation_paths is not None:
        derived = spec.derived_mutation_paths(args)
    return MutationFootprintV1.build(
        name,
        accepted_paths=accepted_paths,
        derived_paths=derived,
    )


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
    """Describe executable harness capabilities to planning and review passes.

    Names alone were not enough for a model to design executable verification:
    it knew that a browser/process tool existed but invented unsupported DOM or
    process semantics.  The description and compact parameter contract come
    directly from the registered tool schema, so this adds no product meaning,
    permission, or capability that the harness does not actually expose.
    """

    browser = web_preview.browser_capability()
    result = []
    for spec in TOOL_SPECS:
        function = dict(spec.schema.get("function") or {})
        parameters = dict(function.get("parameters") or {})
        available = True
        detail = ""
        if spec.capability == "browser":
            available = bool(browser.get("available") and browser.get("playwright"))
            detail = f"browser={browser.get('channel') or 'missing'}, playwright={'yes' if browser.get('playwright') else 'no'}"
        elif spec.capability == "vision":
            available = True
            detail = "availability is verified against the configured model at call time"
        result.append({
            "name": spec.name,
            "category": spec.category,
            "risk": spec.risk,
            "mutates_workspace": spec.mutates_workspace,
            "approval": "required" if spec.approval_required({}) else "not_required",
            "lifecycle": spec.lifecycle,
            "available": available,
            "detail": detail,
            "description": str(function.get("description") or ""),
            "parameters": parameters,
            "result_contract": dict(spec.result_contract),
        })
    return tuple(result)


def register_artifact_provider(workspace: str | Path, provider: Any) -> None:
    materialize_artifact.register_provider(workspace, provider)


def register_vision_evaluator(workspace: str | Path, evaluator: Any) -> None:
    inspect_images.register_evaluator(workspace, evaluator)


def register_output_publisher(workspace: str | Path, publisher: Any) -> None:
    publish_output.register_provider(workspace, publisher)


def shutdown_workspace_resources(workspace: str | Path) -> None:
    process_manager.shutdown_workspace(workspace)
    browser_session.shutdown_workspace(workspace)
    web_preview.shutdown_workspace(workspace)
    materialize_artifact.unregister_provider(workspace)
    inspect_images.unregister_evaluator(workspace)
    publish_output.unregister_provider(workspace)


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
    "TOOL_SCHEMAS", "TOOL_SPECS", "TOOLS", "MutationFootprintV1", "ToolContext", "ToolExecutionResult",
    "ToolSecurityError", "ToolSpec", "capability_report", "configure_workspace",
    "get_spec", "get_tool_context", "get_workspace", "mutation_footprint", "names", "requires_approval",
    "risk_map", "run_tool", "run_tool_detailed", "shutdown_workspace_resources",
    "workspace_context", "register_artifact_provider", "register_vision_evaluator",
    "register_output_publisher",
]

"""Human-editable, deterministic Markdown representation of a durable plan."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


class PlanDocumentError(ValueError):
    def __init__(self, message: str, line: int = 1) -> None:
        super().__init__(f"line {max(1, line)}: {message}")
        self.line = max(1, line)


@dataclass(frozen=True, slots=True)
class ParsedPlanDocument:
    summary: str
    tasks: tuple[dict[str, Any], ...]
    execution_strategy: str
    expected_changes: tuple[dict[str, Any], ...]
    applicability_evidence: tuple[dict[str, Any], ...]


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def render_plan_document(plan: Any) -> str:
    lines = ["# Project Plan", "", "## Summary", str(plan.summary).strip(), ""]
    for task in tuple(plan.tasks):
        depends = ", ".join(str(item) for item in task.depends_on) or "-"
        lines.extend(
            (
                f"## Task {task.id}",
                f"Title: {task.title}",
                f"Risk: {_value(task.risk) or 'medium'}",
                f"Depends on: {depends}",
                "",
                "### Description",
                str(task.description).strip(),
                "",
                "### Acceptance",
                *(f"- {item}" for item in task.acceptance_criteria),
                "",
                "### Verification",
                *(f"- {item}" for item in task.verification),
                "",
            )
        )
    lines.extend(("## Execution strategy", str(plan.execution_strategy).strip(), ""))
    lines.append("## Expected changes")
    for item in plan.expected_changes:
        tasks = ",".join(str(value) for value in item.get("supports_tasks", ())) or "-"
        lines.append(
            f"- {item.get('path', '<resolved during execution>')} | "
            f"{item.get('intent', '')} | tasks={tasks}"
        )
    lines.extend(("", "## Applicability evidence"))
    for item in plan.applicability_evidence:
        tasks = ",".join(str(value) for value in item.get("supports_tasks", ())) or "-"
        lines.append(
            f"- {item.get('source', 'workspace inspection')} | "
            f"{item.get('fact', '')} | tasks={tasks}"
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_plan_document(text: str) -> ParsedPlanDocument:
    raw = str(text or "").replace("\r\n", "\n")
    lines = raw.splitlines()
    headings: list[tuple[int, str]] = [
        (index, line.strip()) for index, line in enumerate(lines) if line.startswith("## ")
    ]
    if not headings or headings[0][1] != "## Summary":
        raise PlanDocumentError("the first section must be '## Summary'", 1)

    def section(start: int, end: int) -> list[str]:
        return lines[start + 1 : end]

    task_starts = [(index, title) for index, title in headings if title.startswith("## Task ")]
    special = {title: index for index, title in headings if not title.startswith("## Task ")}
    required = ("## Summary", "## Execution strategy", "## Expected changes", "## Applicability evidence")
    for title in required:
        if title not in special:
            raise PlanDocumentError(f"missing required section {title!r}", len(lines))
    summary_end = task_starts[0][0] if task_starts else special["## Execution strategy"]
    summary = "\n".join(section(special["## Summary"], summary_end)).strip()
    if not summary:
        raise PlanDocumentError("summary must not be empty", special["## Summary"] + 1)

    tasks: list[dict[str, Any]] = []
    task_lines: dict[str, int] = {}
    known_ids: set[str] = set()
    for position, (start, title) in enumerate(task_starts):
        task_id = title.removeprefix("## Task ").strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,23}", task_id):
            raise PlanDocumentError(
                "task ID must be 1-24 characters using letters, digits, '.', '_' or '-'",
                start + 1,
            )
        if task_id in known_ids:
            raise PlanDocumentError(f"duplicate task ID {task_id}", start + 1)
        known_ids.add(task_id)
        task_lines[task_id] = start + 1
        end_candidates = [
            value for value, _name in headings
            if value > start and (value == special["## Execution strategy"] or _name.startswith("## Task "))
        ]
        end = min(end_candidates) if end_candidates else special["## Execution strategy"]
        block = lines[start + 1 : end]
        fields: dict[str, str] = {}
        subsections: dict[str, list[str]] = {"description": [], "acceptance": [], "verification": []}
        active = ""
        for offset, line in enumerate(block, start + 2):
            stripped = line.strip()
            if stripped in {"### Description", "### Acceptance", "### Verification"}:
                active = stripped.removeprefix("### ").casefold()
                continue
            field = re.match(r"^(Title|Risk|Depends on):\s*(.*)$", stripped, re.I)
            if field and not active:
                fields[field.group(1).casefold()] = field.group(2).strip()
                continue
            if active and stripped:
                value = stripped[2:].strip() if stripped.startswith("- ") else stripped
                subsections[active].append(value)
        risk = fields.get("risk", "medium").casefold()
        if risk not in {"low", "medium", "high", "critical"}:
            raise PlanDocumentError("risk must be low, medium, high, or critical", start + 1)
        title_value = fields.get("title", "").strip()
        description = "\n".join(subsections["description"]).strip()
        if not title_value or not description or not subsections["acceptance"] or not subsections["verification"]:
            raise PlanDocumentError(
                f"task {task_id} needs Title, Description, Acceptance, and Verification",
                start + 1,
            )
        depends = tuple(
            item.strip().upper()
            for item in fields.get("depends on", "-").split(",")
            if item.strip() and item.strip() != "-"
        )
        tasks.append(
            {
                "id": task_id,
                "title": title_value,
                "description": description,
                "risk": risk,
                "depends_on": depends,
                "acceptance_criteria": tuple(subsections["acceptance"]),
                "verification": tuple(subsections["verification"]),
            }
        )
    if not tasks:
        raise PlanDocumentError("a plan must contain at least one task", summary_end + 1)
    for task in tasks:
        unknown = [item for item in task["depends_on"] if item not in known_ids]
        if unknown:
            raise PlanDocumentError(
                f"task {task['id']} depends on unknown task(s): {', '.join(unknown)}",
                task_lines[task["id"]],
            )

    dependencies = {task["id"]: tuple(task["depends_on"]) for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str, trail: tuple[str, ...] = ()) -> None:
        if task_id in visiting:
            start = trail.index(task_id) if task_id in trail else 0
            cycle = (*trail[start:], task_id)
            raise PlanDocumentError(
                f"task dependencies form a cycle: {' -> '.join(cycle)}",
                task_lines[task_id],
            )
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in dependencies[task_id]:
            visit(dependency, (*trail, task_id))
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in dependencies:
        visit(task_id)

    strategy_start = special["## Execution strategy"]
    strategy_end = special["## Expected changes"]
    strategy = "\n".join(section(strategy_start, strategy_end)).strip()
    if not strategy:
        raise PlanDocumentError("execution strategy must not be empty", strategy_start + 1)

    def pipe_items(start: int, end: int, *, evidence: bool) -> tuple[dict[str, Any], ...]:
        values: list[dict[str, Any]] = []
        for index, line in enumerate(lines[start + 1 : end], start + 2):
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("- "):
                raise PlanDocumentError("list entries must start with '- '", index)
            parts = [item.strip() for item in stripped[2:].split("|")]
            if len(parts) != 3 or not parts[2].casefold().startswith("tasks="):
                raise PlanDocumentError("expected 'left | description | tasks=T001,T002'", index)
            supports = [item.strip().upper() for item in parts[2].split("=", 1)[1].split(",") if item.strip() and item.strip() != "-"]
            if any(item not in known_ids for item in supports):
                raise PlanDocumentError("entry references an unknown task", index)
            values.append(
                ({"source": parts[0], "fact": parts[1], "supports_tasks": supports}
                 if evidence else {"path": parts[0], "intent": parts[1], "supports_tasks": supports})
            )
        if not values:
            raise PlanDocumentError("section must contain at least one entry", start + 1)
        return tuple(values)

    changes_start = special["## Expected changes"]
    evidence_start = special["## Applicability evidence"]
    changes = pipe_items(changes_start, evidence_start, evidence=False)
    evidence = pipe_items(evidence_start, len(lines), evidence=True)
    return ParsedPlanDocument(summary, tuple(tasks), strategy, changes, evidence)


__all__ = ["ParsedPlanDocument", "PlanDocumentError", "parse_plan_document", "render_plan_document"]

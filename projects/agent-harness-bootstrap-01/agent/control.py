"""Harness-owned control tools and portable JSON-schema validation.

These tools let a model *request* state transitions. They are not executable
workspace tools: the runtime validates each request and owns the transition.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _fn(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 24},
        "title": {"type": "string", "minLength": 3, "maxLength": 180},
        "description": {"type": "string", "minLength": 3, "maxLength": 2_000},
        "acceptance_criteria": {
            "type": "array", "minItems": 1, "maxItems": 12,
            "items": {"type": "string", "minLength": 3, "maxLength": 600},
        },
        # Scalar/list variance and numeric dependencies are normalized by the
        # harness before validation; provider schemas intentionally stay loose
        # for those mechanically repairable fields.
        "verification": {},
        "depends_on": {},
        "expected_changes": {},
        "risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
    },
    "required": ["title", "description", "acceptance_criteria", "verification"],
    "additionalProperties": False,
}


APPLICABILITY_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fact": {"type": "string", "minLength": 3, "maxLength": 1_000},
        "source": {"type": "string", "minLength": 1, "maxLength": 500},
        "supports_tasks": {
            "type": "array", "minItems": 1, "maxItems": 80,
            "items": {"type": "string", "minLength": 1, "maxLength": 24},
        },
    },
    # ``source`` is optional at the provider boundary.  The runtime binds it
    # to a stable harness inspection reference before persistence; requiring a
    # backend-generated tool-call id is impossible on providers such as Ollama.
    "required": ["fact"],
    "additionalProperties": False,
}


EXPECTED_CHANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1, "maxLength": 500},
        "intent": {"type": "string", "minLength": 3, "maxLength": 1_000},
        "supports_tasks": {
            "type": "array", "minItems": 1, "maxItems": 80,
            "items": {"type": "string", "minLength": 1, "maxLength": 24},
        },
    },
    "required": ["path", "intent"],
    "additionalProperties": False,
}


PROPOSE_PLAN = _fn(
    "propose_plan",
    (
        "Submit one concise inspected plan for explicit approval. Do not invent task ids, "
        "database ids, supports_tasks, or global references; the harness owns them. Each task "
        "contains title, description, expected changes, acceptance criteria, verification, "
        "optional dependencies as earlier task numbers, and optional risk. Use one to three "
        "tasks for a simple artifact. This call never modifies files."
    ),
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 3, "maxLength": 2_000},
            "applicability_evidence": {
                "type": "array", "maxItems": 40,
                "items": APPLICABILITY_EVIDENCE_SCHEMA,
            },
            "execution_strategy": {"type": "string", "maxLength": 8_000},
            "expected_changes": {
                "type": "array", "maxItems": 80,
                "items": EXPECTED_CHANGE_SCHEMA,
            },
            "tasks": {"type": "array", "minItems": 1, "maxItems": 80, "items": TASK_SCHEMA},
        },
        "required": [
            "summary", "tasks",
        ],
        "additionalProperties": False,
    },
)


REQUEST_PLAN_INPUT = _fn(
    "request_plan_input",
    (
        "Pause planning for one to three high-impact user decisions that cannot be "
        "discovered from the workspace. Never ask for repository facts that tools can inspect."
    ),
    {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "header": {"type": "string", "minLength": 1, "maxLength": 40},
                        "question": {"type": "string", "minLength": 3, "maxLength": 1_000},
                        "options": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "minLength": 1, "maxLength": 80},
                                    "description": {"type": "string", "minLength": 3, "maxLength": 500},
                                    "recommended": {"type": "boolean"},
                                },
                                "required": ["label", "description", "recommended"],
                                "additionalProperties": False,
                            },
                        },
                        "allow_freeform": {"type": "boolean"},
                        "reason": {"type": "string", "minLength": 3, "maxLength": 1_000},
                    },
                    "required": ["id", "header", "question", "options", "allow_freeform", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
)


SUBMIT_PLAN_REVIEW = _fn(
    "submit_plan_review",
    "Return an independent verdict on whether a proposed plan fully and safely covers the objective.",
    {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
            "summary": {"type": "string", "minLength": 3, "maxLength": 2_000},
            "issues": {
                "type": "array", "maxItems": 30,
                "items": {"type": "string", "minLength": 3, "maxLength": 1_000},
            },
        },
        "required": ["verdict", "summary", "issues"],
        "additionalProperties": False,
    },
)


UPDATE_TASK = _fn(
    "update_task",
    "Update one accepted checklist item. Done requires evidence; blocked requires a concrete blocker.",
    {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1, "maxLength": 24},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked"]},
            "note": {"type": "string", "maxLength": 4_000},
            "evidence": {
                "type": "array", "maxItems": 20,
                "items": {"type": "string", "minLength": 2, "maxLength": 2_000},
            },
        },
        "required": ["task_id", "status", "note", "evidence"],
        "additionalProperties": False,
    },
)


PROPOSE_PLAN_CHANGE = _fn(
    "propose_plan_change",
    "Propose newly discovered material work. It creates a new plan revision and pauses for user approval.",
    {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "minLength": 3, "maxLength": 2_000},
            "tasks": {"type": "array", "minItems": 1, "maxItems": 30, "items": TASK_SCHEMA},
        },
        "required": ["reason", "tasks"],
        "additionalProperties": False,
    },
)


DELEGATE_TASK = _fn(
    "delegate_task",
    "Run a fresh-context focused worker with a task-specific role synthesized for this exact subtask.",
    {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1, "maxLength": 24},
            "role": {"type": "string", "minLength": 10, "maxLength": 1_000},
            "task": {"type": "string", "minLength": 3, "maxLength": 4_000},
            "success_criteria": {
                "type": "array", "minItems": 1, "maxItems": 20,
                "items": {"type": "string", "minLength": 3, "maxLength": 1_000},
            },
            "context": {"type": "string", "maxLength": 8_000},
            "allowed_tools": {
                "type": "array", "minItems": 1, "maxItems": 12,
                "items": {"type": "string", "minLength": 1, "maxLength": 64},
            },
        },
        "required": ["task_id", "role", "task", "success_criteria", "context", "allowed_tools"],
        "additionalProperties": False,
    },
)


INSPECT_TASK = _fn(
    "inspect_task",
    "Read one exact accepted-plan task and a paginated slice of its durable evidence.",
    {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1, "maxLength": 24},
            "evidence_offset": {"type": "integer", "minimum": 0, "maximum": 100_000},
            "evidence_limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["task_id", "evidence_offset", "evidence_limit"],
        "additionalProperties": False,
    },
)


RECORD_MEMORY = _fn(
    "record_memory",
    "Persist a concise repository fact, decision, constraint, or failure lesson across compaction and restart.",
    {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "minLength": 3, "maxLength": 2_000},
            "source": {"type": "string", "minLength": 1, "maxLength": 1_000},
        },
        "required": ["fact", "source"],
        "additionalProperties": False,
    },
)


REQUEST_USER = _fn(
    "request_user",
    "Pause only for information or authority that cannot be safely inferred or discovered.",
    {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 3, "maxLength": 2_000},
            "reason": {"type": "string", "minLength": 3, "maxLength": 2_000},
        },
        "required": ["question", "reason"],
        "additionalProperties": False,
    },
)


FINISH_GOAL = _fn(
    "finish_goal",
    "Request evidence-gated independent final review. Prose alone can never finish a goal.",
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 3, "maxLength": 4_000},
            "evidence": {
                "type": "array", "minItems": 1, "maxItems": 40,
                "items": {"type": "string", "minLength": 3, "maxLength": 2_000},
            },
        },
        "required": ["summary", "evidence"],
        "additionalProperties": False,
    },
)


RETURN_WORK = _fn(
    "return_work",
    "Return a structured worker result to the parent coordinator. This cannot finish the root goal.",
    {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": ["success", "partial", "blocked"]},
            "summary": {"type": "string", "minLength": 3, "maxLength": 4_000},
            "evidence": {
                "type": "array", "maxItems": 30,
                "items": {"type": "string", "minLength": 2, "maxLength": 2_000},
            },
            "changed_paths": {
                "type": "array", "maxItems": 100,
                "items": {"type": "string", "minLength": 1, "maxLength": 1_000},
            },
            "remaining_risks": {
                "type": "array", "maxItems": 30,
                "items": {"type": "string", "minLength": 2, "maxLength": 1_000},
            },
            "proposed_subtasks": {
                "type": "array", "maxItems": 20,
                "items": {"type": "string", "minLength": 3, "maxLength": 1_000},
            },
        },
        "required": ["outcome", "summary", "evidence", "changed_paths", "remaining_risks", "proposed_subtasks"],
        "additionalProperties": False,
    },
)


SUBMIT_REVIEW = _fn(
    "submit_review",
    "Submit the independent completion verdict and actionable repair tasks when failing.",
    {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "summary": {"type": "string", "minLength": 3, "maxLength": 4_000},
            "issues": {
                "type": "array", "maxItems": 30,
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "title": {"type": "string", "minLength": 3, "maxLength": 180},
                        "details": {"type": "string", "minLength": 3, "maxLength": 2_000},
                        "acceptance_criteria": {
                            "type": "array", "minItems": 1, "maxItems": 10,
                            "items": {"type": "string", "minLength": 3, "maxLength": 600},
                        },
                    },
                    "required": ["severity", "title", "details", "acceptance_criteria"],
                    "additionalProperties": False,
                },
            },
            "checked_task_ids": {
                "type": "array", "minItems": 1, "maxItems": 80,
                "items": {"type": "string", "minLength": 1, "maxLength": 24},
            },
        },
        "required": ["verdict", "summary", "issues", "checked_task_ids"],
        "additionalProperties": False,
    },
)


PLANNER_SCHEMAS = [PROPOSE_PLAN, REQUEST_PLAN_INPUT]
PLAN_REVIEWER_SCHEMAS = [SUBMIT_PLAN_REVIEW]
COORDINATOR_SCHEMAS = [UPDATE_TASK, PROPOSE_PLAN_CHANGE, DELEGATE_TASK, INSPECT_TASK, RECORD_MEMORY, REQUEST_USER, FINISH_GOAL]
WORKER_SCHEMAS = [RETURN_WORK]
REVIEWER_SCHEMAS = [INSPECT_TASK, SUBMIT_REVIEW]
CONTROL_SCHEMAS = PLANNER_SCHEMAS + PLAN_REVIEWER_SCHEMAS + COORDINATOR_SCHEMAS + WORKER_SCHEMAS + REVIEWER_SCHEMAS
CONTROL_NAMES = {schema["function"]["name"] for schema in CONTROL_SCHEMAS}
_BY_NAME = {schema["function"]["name"]: schema for schema in CONTROL_SCHEMAS}


class ControlValidationError(ValueError):
    pass


def _schema_errors(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[str],
    *,
    limit: int = 24,
) -> None:
    """Collect useful schema defects in one pass instead of teaching by retry.

    Small tool-calling models often repair exactly the first validation error
    they see.  Returning all independent defects from a malformed control call
    lets them repair the whole payload in one turn and keeps the UI from showing
    a long field-by-field failure ladder.
    """
    if len(errors) >= limit:
        return
    expected = schema.get("type")
    type_ok = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }
    if expected in type_ok and not type_ok[expected](value):
        errors.append(f"{path} must be {expected}, got {type(value).__name__}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be at most {schema['maximum']}")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path} is too short")
        if len(value) > schema.get("maxLength", 1_000_000_000):
            errors.append(f"{path} is too long")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path} has too few items")
        if len(value) > schema.get("maxItems", 1_000_000_000):
            errors.append(f"{path} has too many items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _schema_errors(item, item_schema, f"{path}[{index}]", errors, limit=limit)
                if len(errors) >= limit:
                    break
    elif isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")
                if len(errors) >= limit:
                    return
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                errors.append(f"{path} has unknown fields: {', '.join(extras)}")
        for key, item in value.items():
            if key in properties:
                _schema_errors(item, properties[key], f"{path}.{key}", errors, limit=limit)
                if len(errors) >= limit:
                    return


def validate_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    """Validate the portable JSON-Schema subset used by all harness tools."""

    errors: list[str] = []
    _schema_errors(value, schema, path, errors)
    if errors:
        suffix = "; additional defects omitted" if len(errors) >= 24 else ""
        raise ControlValidationError("; ".join(errors) + suffix)


def validate_control_call(name: str, args: Any) -> dict[str, Any]:
    schema = _BY_NAME.get(name)
    if schema is None:
        raise ControlValidationError(f"unknown control tool '{name}'")
    normalized = args
    if name == "update_task" and isinstance(args, dict):
        # Tool-capable models commonly emit semantically equivalent evidence
        # objects even when the portable schema requests strings.  Canonicalize
        # that harmless provider variance here; state-transition validation
        # still enforces task ids, statuses, and evidence requirements.
        evidence = args.get("evidence")
        if isinstance(evidence, Mapping):
            evidence = [evidence]
        if isinstance(evidence, list) and any(isinstance(item, Mapping) for item in evidence):
            normalized = dict(args)
            normalized["evidence"] = [_canonical_evidence_text(item) for item in evidence]
    schema_error: ControlValidationError | None = None
    try:
        validate_schema(normalized, schema["function"]["parameters"])
    except ControlValidationError as exc:
        schema_error = exc
    # Compatibility validation for persisted/legacy planner clients.  The
    # provider-facing schema no longer asks for these cross references, but a
    # caller that opts into the old id-bearing shape must still supply a
    # complete, internally checkable legacy payload.
    if name == "propose_plan" and isinstance(normalized, dict):
        tasks = normalized.get("tasks", ())
        legacy = bool(normalized.get("applicability_evidence") or normalized.get("expected_changes")) or any(
            isinstance(item, Mapping) and "id" in item for item in tasks if isinstance(tasks, list)
        )
        if legacy:
            errors: list[str] = []
            if "expected_changes" not in normalized:
                errors.append("arguments.expected_changes is required")
            for index, item in enumerate(normalized.get("applicability_evidence", ())):
                if isinstance(item, Mapping) and "supports_tasks" not in item:
                    errors.append(f"arguments.applicability_evidence[{index}].supports_tasks is required")
            for index, item in enumerate(normalized.get("expected_changes", ())):
                if isinstance(item, Mapping) and "supports_tasks" not in item:
                    errors.append(f"arguments.expected_changes[{index}].supports_tasks is required")
            if errors:
                prefix = f"{schema_error}; " if schema_error else ""
                raise ControlValidationError(prefix + "; ".join(errors))
    if schema_error is not None:
        raise schema_error
    return normalized


def _canonical_evidence_text(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return value
    summary = next(
        (
            str(value[key]).strip()
            for key in ("summary", "fact", "evidence", "result", "note")
            if str(value.get(key, "")).strip()
        ),
        "",
    )
    source = next(
        (
            str(value[key]).strip()
            for key in ("source", "path", "artifact", "command")
            if str(value.get(key, "")).strip()
        ),
        "",
    )
    if summary:
        return f"{summary} [source: {source}]" if source else summary
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

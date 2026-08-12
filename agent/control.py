"""Harness-owned control tools and portable JSON-schema validation.

These tools let a model *request* state transitions. They are not executable
workspace tools: the runtime validates each request and owns the transition.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReviewIssueV2:
    detail: str
    severity: str = "advisory"
    blocking: bool = False
    criterion_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    classified: bool = True
    version: int = 2

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["criterion_refs"] = list(self.criterion_refs)
        value["evidence_refs"] = list(self.evidence_refs)
        return value

    @classmethod
    def from_value(cls, value: Any, *, verdict: str) -> "ReviewIssueV2":
        if isinstance(value, str):
            return cls(
                detail=value.strip(),
                classified=verdict != "revise",
            )
        if not isinstance(value, Mapping):
            raise ControlValidationError("review issues must be text or objects")
        detail = str(
            value.get("detail")
            or value.get("summary")
            or value.get("message")
            or value.get("issue")
            or ""
        ).strip()
        if not detail:
            raise ControlValidationError("review issues require a non-empty detail")
        severity = str(value.get("severity") or value.get("type") or "advisory").strip().casefold()
        severity_alias = {
            "info": "advisory",
            "warning": "advisory",
            "minor": "advisory",
            "major": "blocking",
            "critical": "blocking",
            "error": "blocking",
        }
        severity = severity_alias.get(severity, severity)
        if severity not in {"advisory", "blocking"}:
            raise ControlValidationError(
                "review issue severity must be advisory or blocking"
            )
        explicitly_classified = "blocking" in value or "severity" in value or "type" in value
        blocking = bool(value.get("blocking", severity == "blocking"))

        def refs(key: str) -> tuple[str, ...]:
            raw = value.get(key, ())
            if isinstance(raw, str):
                raw = (raw,)
            if not isinstance(raw, (list, tuple)):
                return ()
            return tuple(str(item).strip() for item in raw if str(item).strip())

        return cls(
            detail=detail,
            severity="blocking" if blocking else "advisory",
            blocking=blocking,
            criterion_refs=refs("criterion_refs"),
            evidence_refs=refs("evidence_refs"),
            classified=explicitly_classified or verdict != "revise",
        )


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
        "verification": {
            "description": (
                "One concise verification string or an array of concise steps. "
                "Each step must stay under 1,000 characters; never embed a full "
                "test program or shell heredoc here."
            )
        },
        "depends_on": {},
        "expected_changes": {},
        "requirement_refs": {
            "type": "array", "minItems": 1, "maxItems": 40,
            "items": {"type": "string", "minLength": 1, "maxLength": 24},
            "description": (
                "Requirement anchor ids implemented and verified by this task. "
                "Every accepted anchor must be covered by at least one task."
            ),
        },
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
        "basis": {
            "type": "string",
            "description": (
                "Use existing_inspected_path for a file found by inspection; "
                "repository_convention for a path justified by an observed existing "
                "repository convention; model_selected_new_layout for a concrete new "
                "path selected after inspecting a new/empty workspace; and "
                "explicit_user_requirement only when the exact relative path appears "
                "verbatim in the original request."
            ),
            "enum": [
                "existing_inspected_path",
                "repository_convention",
                "model_selected_new_layout",
                "explicit_user_requirement",
            ],
        },
        "evidence_refs": {
            "type": "array", "minItems": 1, "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    },
    "required": ["path", "intent", "basis", "evidence_refs"],
    "additionalProperties": False,
}

SEMANTIC_GOAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "original_request": {"type": "string", "minLength": 1, "maxLength": 200_000},
        "interpreted_outcome": {"type": "string", "minLength": 1, "maxLength": 20_000},
        # Legacy providers sometimes echo a semantic lifecycle value. It is
        # optional and never authoritative; the harness canonicalizes it.
        "status": {
            "type": "string",
            "maxLength": 32,
            "description": "Legacy transport only; omit it. The harness owns lifecycle status.",
        },
        "requested_effects": {
            "type": "array",
            "maxItems": 7,
            "description": (
                "Capability effects needed for the accepted outcome. Include "
                "read_workspace after using repository inspection tools; the harness "
                "also records that observed read from successful cited evidence."
            ),
            "items": {
                "type": "string",
                "enum": [
                    "answer", "read_workspace", "mutate_workspace",
                    "execute_code", "install_dependencies", "use_network",
                    "external_side_effect", "read", "write", "run", "execute",
                    "execute_shell", "run_shell", "shell", "preview", "install",
                    "network", "external", "read_file", "list_files", "grep",
                    "write_file", "edit_file", "apply_patch",
                    "materialize_artifact", "run_bash", "run_command",
                    "inspect_images",
                    "start_process", "preview_html", "inspect_preview",
                ],
            },
        },
        "required_outcomes": {
            "type": "array", "minItems": 1, "maxItems": 40,
            "items": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
        "constraints": {
            "type": "array", "maxItems": 40,
            "items": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
        "exclusions": {
            "type": "array", "maxItems": 40,
            "items": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
        "acceptance_criteria": {
            "type": "array", "minItems": 1, "maxItems": 40,
            "items": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
        "requirement_anchors": {
            "type": "array", "minItems": 1, "maxItems": 40,
            "description": (
                "Trace every material user-authored deliverable, named technology or medium, "
                "interaction, visual/runtime quality, format, and constraint to an exact span. "
                "Interpret what must be observable in the finished result; importing or naming "
                "a technology is not evidence that its distinctive capability was delivered."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 24},
                    "verbatim_span": {"type": "string", "minLength": 1, "maxLength": 2_000},
                    "interpreted_requirement": {"type": "string", "minLength": 3, "maxLength": 4_000},
                    "observable_implications": {
                        "type": "array", "minItems": 1, "maxItems": 12,
                        "items": {"type": "string", "minLength": 3, "maxLength": 2_000},
                    },
                    "kind": {"type": "string", "minLength": 1, "maxLength": 100},
                },
                "required": [
                    "verbatim_span", "interpreted_requirement",
                    "observable_implications", "kind",
                ],
                "additionalProperties": False,
            },
        },
        "unresolved_decisions": {
            "type": "array", "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
        "repository_evidence_refs": {
            "type": "array", "minItems": 1, "maxItems": 80,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    },
    "required": [
        "original_request", "interpreted_outcome", "requested_effects",
        "required_outcomes", "constraints", "exclusions",
        "acceptance_criteria", "requirement_anchors", "unresolved_decisions",
        "repository_evidence_refs",
    ],
    "additionalProperties": False,
}


PROPOSE_SEMANTIC_GOAL = _fn(
    "propose_semantic_goal",
    (
        "Submit the repository-grounded semantic interpretation before constructing "
        "tasks. Preserve original_request exactly, cite successful inspection references, "
        "include the effects needed to deliver and verify the outcome, and leave "
        "consequential unresolved decisions for request_plan_input."
    ),
    {
        **SEMANTIC_GOAL_SCHEMA,
    },
)


PROPOSE_PLAN = _fn(
    "propose_plan",
    (
        "Submit one concise inspected plan after semantic interpretation. Do not invent task ids, "
        "database ids, or global references; the harness owns them. Evidence and resource "
        "claims use supports_tasks with 1-based task numbers (for example ['1', '2']). Each task "
        "contains title, description, expected changes, acceptance criteria, verification, "
        "optional dependencies as earlier task numbers, and optional risk. Use one to three "
        "tasks for a simple artifact. This call never modifies files."
    ),
    {
        "type": "object",
        "properties": {
            # Backward-compatible combined proposals may still carry this
            # object. New staged planners call propose_semantic_goal first.
            "semantic_goal": SEMANTIC_GOAL_SCHEMA,
            "semantic_fingerprint": {
                "type": "string",
                "minLength": 16,
                "maxLength": 128,
            },
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
        "required": ["summary", "tasks"],
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
                            "minItems": 2,
                            "maxItems": 3,
                            "items": {
                                # The portable validator does not implement
                                # JSON-Schema oneOf.  Keep the provider-facing
                                # object contract permissive here; the question
                                # normalizer accepts compact strings and then
                                # validates the canonical object shape below.
                            },
                        },
                        "allow_freeform": {"type": "boolean"},
                        "allow_free_form": {"type": "boolean"},
                        "reason": {"type": "string", "minLength": 3, "maxLength": 1_000},
                        "decision_need": {
                            "type": "object",
                            "properties": {
                                "version": {"type": "integer"},
                                "impact": {"type": "string", "minLength": 1},
                                "affected_scope": {"type": "array", "items": {"type": "string"}},
                                "affected_effects": {"type": "array", "items": {"type": "string"}},
                                "reversible": {"type": "boolean"},
                                "requires_user_authority": {"type": "boolean"},
                                "reason": {"type": "string", "minLength": 1},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": [
                                "impact", "affected_scope", "affected_effects",
                                "reversible", "requires_user_authority", "reason",
                                "evidence_refs",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["question", "options", "decision_need"],
                    "additionalProperties": True,
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
                "items": {},
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
            "worker_role": {
                "type": "string",
                "enum": ["predictor", "falsifier", "challenger", "selector", "repairer", "reviewer"],
            },
            "falsification_targets": {
                "type": "array", "maxItems": 3,
                "items": {"type": "string", "minLength": 3, "maxLength": 1_000},
            },
            "context_refs": {
                "type": "array", "maxItems": 40,
                "items": {"type": "string", "minLength": 1, "maxLength": 500},
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
            "claims": {
                "type": "array", "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string", "minLength": 1, "maxLength": 200},
                        "claim": {"type": "string", "minLength": 3, "maxLength": 2_000},
                        "evidence_refs": {
                            "type": "array", "maxItems": 20,
                            "items": {"type": "string", "minLength": 1, "maxLength": 1_000},
                        },
                        "falsification_check": {"type": "string", "minLength": 3, "maxLength": 2_000},
                    },
                    "required": ["criterion_id", "claim", "evidence_refs", "falsification_check"],
                    "additionalProperties": False,
                },
            },
            "verified_findings": {"type": "integer", "minimum": 0, "maximum": 100},
            "false_findings": {"type": "integer", "minimum": 0, "maximum": 100},
            "accepted_fixes": {"type": "integer", "minimum": 0, "maximum": 100},
            "staged_candidate": {
                "type": "object",
                "properties": {
                    "approach_summary": {"type": "string", "minLength": 3, "maxLength": 2_000},
                    "files": {
                        "type": "array", "minItems": 1, "maxItems": 50,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "minLength": 1, "maxLength": 1_000},
                                "content": {"type": "string", "maxLength": 1_000_000},
                            },
                            "required": ["path", "content"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["approach_summary", "files"],
                "additionalProperties": False,
            },
            "predicted_failures": {
                "type": "array", "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {"type": "string", "minLength": 3, "maxLength": 1_000},
                        "separating_check": {"type": "string", "minLength": 3, "maxLength": 1_000},
                    },
                    "required": ["hypothesis", "separating_check"],
                    "additionalProperties": False,
                },
            },
            "selection": {
                "type": "object",
                "properties": {
                    "candidate_ref": {"type": "string", "minLength": 1, "maxLength": 1_000},
                    "evidence_refs": {
                        "type": "array", "minItems": 1, "maxItems": 20,
                        "items": {"type": "string", "minLength": 1, "maxLength": 1_000},
                    },
                    "reason": {"type": "string", "minLength": 3, "maxLength": 2_000},
                },
                "required": ["candidate_ref", "evidence_refs", "reason"],
                "additionalProperties": False,
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


PLANNER_SCHEMAS = [PROPOSE_SEMANTIC_GOAL, PROPOSE_PLAN, REQUEST_PLAN_INPUT]
PLAN_REVIEWER_SCHEMAS = [SUBMIT_PLAN_REVIEW]
COORDINATOR_SCHEMAS = [UPDATE_TASK, PROPOSE_PLAN_CHANGE, DELEGATE_TASK, INSPECT_TASK, RECORD_MEMORY, REQUEST_USER, FINISH_GOAL]
WORKER_SCHEMAS = [RETURN_WORK]
REVIEWER_SCHEMAS = [INSPECT_TASK, SUBMIT_REVIEW]
CONTROL_SCHEMAS = PLANNER_SCHEMAS + PLAN_REVIEWER_SCHEMAS + COORDINATOR_SCHEMAS + WORKER_SCHEMAS + REVIEWER_SCHEMAS
CONTROL_NAMES = {schema["function"]["name"] for schema in CONTROL_SCHEMAS}
_BY_NAME = {schema["function"]["name"]: schema for schema in CONTROL_SCHEMAS}


class ControlValidationError(ValueError):
    pass


def _normalize_plan_change_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Repair transport-only coordinator variance before schema validation.

    Some providers reuse their execution-task envelope when asking for a plan
    revision.  Resource leases and lifecycle fields belong to the harness, and
    aliases such as ``name``/``summary`` carry the same model-authored task
    meaning as the canonical fields.  This helper is intentionally narrow: it
    never creates paths, criteria, dependencies, or effects.
    """

    normalized = dict(args)
    harness_fields = {
        "resource_claims", "resource_claim", "resolved_paths", "lease",
        "status", "attempt", "attempts", "evidence", "note",
        "blocked_reason", "last_error", "started_at", "completed_at",
        "ready_at", "updated_at", "worker_id", "execution_state",
        "runtime_state", "worker_state",
    }
    for field in harness_fields:
        normalized.pop(field, None)
    raw_tasks = normalized.get("tasks")
    if not isinstance(raw_tasks, list):
        return normalized
    tasks: list[Any] = []
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            tasks.append(raw)
            continue
        task = dict(raw)
        aliases = (
            ("name", "title"),
            ("summary", "description"),
            ("task", "description"),
            ("acceptance", "acceptance_criteria"),
            ("criteria", "acceptance_criteria"),
            ("verification_steps", "verification"),
            ("dependencies", "depends_on"),
        )
        for source, target in aliases:
            if target not in task and source in task:
                value = task.pop(source)
                if source in {"acceptance", "criteria", "verification_steps", "dependencies"} and isinstance(value, str):
                    value = [value]
                task[target] = value
        if not str(task.get("title") or "").strip() and str(task.get("description") or "").strip():
            task["title"] = " ".join(str(task["description"]).split())[:180].rstrip()
        for field in harness_fields:
            task.pop(field, None)
        tasks.append(task)
    normalized["tasks"] = tasks
    return normalized


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
    if name == "propose_plan_change" and isinstance(args, Mapping):
        normalized = _normalize_plan_change_args(args)
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
    if name == "submit_plan_review" and isinstance(args, dict):
        normalized = dict(args)
        verdict = str(normalized.get("verdict") or "").strip().casefold()
        raw_issues = normalized.get("issues", ())
        if isinstance(raw_issues, (str, Mapping)):
            raw_issues = [raw_issues]
        if isinstance(raw_issues, (list, tuple)):
            normalized["issues"] = [
                ReviewIssueV2.from_value(item, verdict=verdict).to_dict()
                for item in raw_issues
            ]
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

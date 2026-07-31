"""Adapter from HTTP views to GA3BAD's real runtime and state store."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, Mapping

from ..models import (
    GoalStatus,
    Plan,
    PlanStatus,
    QueuedPromptStatus,
    RoleProfile,
    TaskStatus,
    validate_task_dag,
)
from ..quality import ChangeSetStatus
from ..store import NotFoundError, StalePlanError
from ..ultra_models import WorkNodeStatus
from .schemas import PlanPayload, ReviewSubmissionPayload


REVIEWABLE_CHANGE_SET_STATES = {
    ChangeSetStatus.OPEN,
    ChangeSetStatus.CLOSED,
    ChangeSetStatus.REVIEWING,
    ChangeSetStatus.BLOCKED,
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _role_name(role: Any) -> str:
    return str(getattr(role, "name", "") or "coder")


def _task_snapshot(task: Any) -> dict[str, Any]:
    metadata = dict(getattr(task, "metadata", {}) or {})
    retry = dict(metadata.get("retry_policy") or {})
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "status": task.status.value,
        "parent_id": task.parent_id,
        "dependencies": list(task.depends_on),
        "agent_role": _role_name(task.role),
        "inputs": list(metadata.get("inputs") or ()),
        "outputs": list(metadata.get("outputs") or ()),
        "expected_files": list(metadata.get("expected_files") or ()),
        "acceptance_criteria": list(task.acceptance_criteria),
        "tests": list(task.verification),
        "risk_level": task.risk,
        "required_tools": list(metadata.get("required_tools") or ()),
        "memory_dependencies": list(metadata.get("memory_dependencies") or ()),
        "retry_policy": {
            "max_retries": int(retry.get("max_retries", 2)),
            "backoff_seconds": float(retry.get("backoff_seconds", 0)),
        },
        "approval_gate": bool(metadata.get("approval_gate", False)),
        "constraints": list(metadata.get("constraints") or ()),
        "parallel": bool(metadata.get("parallel", False)),
        "paused": bool(metadata.get("paused", False)),
        "disabled": bool(metadata.get("disabled", False)),
        "comments": list(metadata.get("comments") or ()),
    }


def _diff_files(diff: str, declared_paths: Iterable[str]) -> list[dict[str, Any]]:
    """Parse a unified diff into immutable file and hunk snapshots."""

    declared = list(dict.fromkeys(str(path) for path in declared_paths))
    sections = re.split(r"(?m)(?=^diff --git )", str(diff or ""))
    files: list[dict[str, Any]] = []
    for section in sections:
        if not section.strip():
            continue
        header = re.search(r"(?m)^diff --git a/(.*?) b/(.*?)$", section)
        old_path = header.group(1) if header else ""
        new_path = header.group(2) if header else ""
        path = new_path or old_path
        if not path:
            plus = re.search(r"(?m)^\+\+\+ (?:b/)?(.+)$", section)
            minus = re.search(r"(?m)^--- (?:a/)?(.+)$", section)
            path = (plus.group(1) if plus and plus.group(1) != "/dev/null" else "") or (
                minus.group(1) if minus and minus.group(1) != "/dev/null" else ""
            )
        if not path:
            continue
        hunks: list[dict[str, Any]] = []
        matches = list(re.finditer(r"(?m)^@@ ([^\n]+) @@[^\n]*$", section))
        for index, match in enumerate(matches, 1):
            end = matches[index].start() if index < len(matches) else len(section)
            content = section[match.start():end].rstrip()
            hunk_id = f"{path}:h{index}"
            line_number = 0
            old_match = re.search(r"-(\d+)", match.group(1))
            if old_match:
                line_number = int(old_match.group(1))
            lines = []
            current_line = line_number
            for raw in content.splitlines()[1:]:
                kind = "context"
                if raw.startswith("+") and not raw.startswith("+++"):
                    kind = "added"
                elif raw.startswith("-") and not raw.startswith("---"):
                    kind = "deleted"
                lines.append({"number": max(1, current_line), "kind": kind, "text": raw})
                if kind != "added":
                    current_line += 1
            hunks.append(
                {
                    "id": hunk_id,
                    "header": "@@ " + match.group(1) + " @@",
                    "content": content,
                    "lines": lines,
                }
            )
        additions = sum(1 for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in section.splitlines() if line.startswith("-") and not line.startswith("---"))
        status = "modified"
        if "--- /dev/null" in section:
            status = "added"
        elif "+++ /dev/null" in section:
            status = "deleted"
        files.append(
            {
                "path": path,
                "status": status,
                "additions": additions,
                "deletions": deletions,
                "diff": section.rstrip(),
                "hunks": hunks,
            }
        )
    seen = {item["path"] for item in files}
    for path in declared:
        if path not in seen:
            files.append(
                {
                    "path": path,
                    "status": "modified",
                    "additions": 0,
                    "deletions": 0,
                    "diff": "",
                    "hunks": [],
                }
            )
    return files


class CoreWebAdapter:
    """One narrow, auditable boundary between browser actions and the core."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.store = runtime.store
        self.events = runtime.events
        self.session_id = runtime.session_id
        self._lock = RLock()
        self._requested_view = "plan"

    def _goal(self) -> Any:
        goal = self.runtime.active_goal() or self.store.get_latest_goal(self.session_id)
        if goal is None:
            raise NotFoundError("this session does not have a plan yet")
        return goal

    @staticmethod
    def _can_manage_queue(goal: Any) -> bool:
        return goal.status in {
            GoalStatus.RUNNING,
            GoalStatus.VERIFYING,
            GoalStatus.REVIEWING,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.BLOCKED,
            GoalStatus.COMPLETED,
        }

    def request_view(self, view_name: str) -> None:
        if view_name not in {"plan", "review", "agents"}:
            raise ValueError(f"unknown workspace view: {view_name}")
        with self._lock:
            self._requested_view = view_name

    @staticmethod
    def _queued_prompt_snapshot(item: Any) -> dict[str, Any]:
        return {
            "id": item.id,
            "sequence": item.sequence,
            "text": item.text,
            "mode": item.mode,
            "status": item.status.value,
            "goal_id": item.goal_id,
            "error": item.error,
            "created_at": item.created_at.isoformat(),
            "started_at": item.started_at.isoformat() if item.started_at else None,
        }

    def queue_snapshot(self) -> dict[str, Any]:
        items = self.store.list_queued_prompts(
            self.session_id,
            include_terminal=False,
            limit=10,
        )
        return {
            "items": [self._queued_prompt_snapshot(item) for item in items],
            "pending_count": sum(
                item.status is QueuedPromptStatus.PENDING for item in items
            ),
            "capacity": 10,
        }

    def enqueue_queue_prompt(self, text: str, mode: str | None = None) -> dict[str, Any]:
        goal = self._goal()
        if not self._can_manage_queue(goal):
            raise ValueError(
                "Future requests can be added only while a project is active or completed."
            )
        session = self.store.get_workflow_session(self.session_id)
        selected_mode = mode or str(session["session_mode"])
        item = self.store.enqueue_prompt(self.session_id, text, selected_mode)
        self.events.publish(
            "prompt.queued",
            "A new request was added to Up next.",
            session_id=self.session_id,
            prompt_id=item.id,
            sequence=item.sequence,
            mode=item.mode,
            source="local_web",
            actor="user",
        )
        return {
            "queued": True,
            "item": self._queued_prompt_snapshot(item),
            "queue": self.queue_snapshot(),
        }

    def reorder_queue(self, ordered_ids: Iterable[str]) -> dict[str, Any]:
        goal = self._goal()
        if not self._can_manage_queue(goal):
            raise ValueError(
                "Future requests can be reordered only while a project is active or completed."
            )
        items = self.store.reorder_pending_prompts(self.session_id, ordered_ids)
        self.events.publish(
            "prompt.queue_reordered",
            "Waiting requests were reordered.",
            session_id=self.session_id,
            ordered_ids=list(ordered_ids),
            source="local_web",
            actor="user",
        )
        return {
            "reordered": True,
            "queue": {
                "items": [self._queued_prompt_snapshot(item) for item in items],
                "pending_count": sum(
                    item.status is QueuedPromptStatus.PENDING for item in items
                ),
                "capacity": 10,
            },
        }

    def workspace_context(self) -> dict[str, Any]:
        goal = self.runtime.active_goal() or self.store.get_latest_goal(self.session_id)
        session = self.store.get_workflow_session(self.session_id)
        queue = self.queue_snapshot()
        required_view: str | None = None
        checkpoint_id: str | None = None
        review_badge = 0
        plan_badge = 0
        if goal is not None:
            run = self.store.get_active_ultra_run(goal.id)
            if run is not None:
                change_sets = self.store.list_change_sets(run.id)
                reviewable = [
                    item
                    for item in change_sets
                    if item.status in REVIEWABLE_CHANGE_SET_STATES
                    and not (
                        item.status is ChangeSetStatus.BLOCKED
                        and bool(item.metadata.get("latest_user_review"))
                    )
                ]
                if reviewable:
                    required_view = "review"
                    checkpoint_id = reviewable[-1].id
                    review_badge = 1
            if (
                required_view is None
                and goal.status is GoalStatus.AWAITING_PLAN_APPROVAL
            ):
                required_view = "plan"
                plan_badge = 1
        if required_view == "review":
            attention = {
                "state": "waiting",
                "eyebrow": "Your decision is needed",
                "title": "Review the recorded changes",
                "body": "Resolve every changed file. Approved work continues; requested changes start a fixer.",
                "action": {"label": "Review changes", "view": "review"},
            }
        elif required_view == "plan":
            attention = {
                "state": "waiting",
                "eyebrow": "Your decision is needed",
                "title": "Review the plan before work starts",
                "body": "You can edit this revision, save a draft, or approve it once.",
                "action": {"label": "Review plan", "view": "plan"},
            }
        elif goal is None:
            attention = {
                "state": "idle",
                "eyebrow": "No project yet",
                "title": "Start from the terminal",
                "body": "Create a Goal to see its plan, review gates, and execution here.",
                "action": None,
            }
        elif goal.status is GoalStatus.COMPLETED:
            attention = {
                "state": "complete",
                "eyebrow": "Project complete",
                "title": "The workflow has finished",
                "body": "The final plan and execution evidence remain available for inspection.",
                "action": None,
            }
        elif goal.status in {GoalStatus.BLOCKED, GoalStatus.PAUSED}:
            attention = {
                "state": "blocked",
                "eyebrow": "Workflow needs attention",
                "title": "Work cannot continue yet",
                "body": "Return to the terminal for the blocking question, permission, or recovery action.",
                "action": None,
            }
        else:
            attention = {
                "state": "working",
                "eyebrow": "System is working",
                "title": "No action is required from you",
                "body": "You can inspect progress or add a request for after the current work.",
                "action": None,
            }
        with self._lock:
            requested_view = self._requested_view
        return {
            "session_id": self.session_id,
            "session_short": self.session_id[:8],
            "requested_view": requested_view,
            "required_view": required_view,
            "current_view": required_view or requested_view,
            "checkpoint_id": checkpoint_id,
            "goal": (
                {
                    "id": goal.id,
                    "objective": goal.objective,
                    "status": goal.status.value,
                    "plan_revision": goal.active_plan_revision,
                }
                if goal is not None
                else None
            ),
            "mode": str(session["session_mode"]),
            "attention": attention,
            "navigation": {
                "plan": {"badge": plan_badge},
                "review": {"badge": review_badge},
                "agents": {"badge": 0},
            },
            "capabilities": {
                "can_manage_queue": (
                    goal is not None
                    and self._can_manage_queue(goal)
                ),
                "can_open_execution": goal is not None,
            },
            "queue": queue,
            "updated_at": _iso_now(),
        }

    def plan_snapshot(self) -> dict[str, Any]:
        goal = self._goal()
        plan = self.store.get_latest_plan(goal.id)
        if plan is None:
            raise NotFoundError("this session does not have a plan yet")
        session = self.store.get_workflow_session(self.session_id)
        session_mode = str(session["session_mode"])
        can_edit = (
            goal.status is GoalStatus.AWAITING_PLAN_APPROVAL
            and session_mode in {"plan", "normal"}
            and plan.status is PlanStatus.PENDING_APPROVAL
        )
        draft = dict(self.runtime.session_snapshot().get("web_plan_draft") or {})
        return {
            "session_id": self.session_id,
            "session_short": self.session_id[:8],
            "goal_id": goal.id,
            "objective": goal.objective,
            "revision": plan.revision,
            "status": plan.status.value,
            "goal_status": goal.status.value,
            "session_mode": session_mode,
            "summary": plan.summary,
            "tasks": [_task_snapshot(task) for task in plan.tasks],
            "global_constraints": list(
                goal.metadata.get("web_plan_constraints", goal.constraints)
            ),
            "protected_paths": list(goal.metadata.get("protected_paths") or ()),
            "draft": draft if draft.get("base_revision") == plan.revision else None,
            "capabilities": {
                "can_edit": can_edit,
                "can_save_draft": can_edit,
                "can_create_revision": can_edit,
                "can_approve": (
                    goal.status is GoalStatus.AWAITING_PLAN_APPROVAL
                    and plan.status is PlanStatus.PENDING_APPROVAL
                    and session_mode != "plan"
                ),
                "can_manage_queue": self._can_manage_queue(goal),
            },
            "queue": self.queue_snapshot(),
            "updated_at": plan.updated_at.isoformat(),
            "connection": "connected",
        }

    @staticmethod
    def _task_value(item: Any, index: int) -> dict[str, Any]:
        metadata = {
            "inputs": item.inputs,
            "outputs": item.outputs,
            "expected_files": item.expected_files,
            "required_tools": item.required_tools,
            "memory_dependencies": item.memory_dependencies,
            "retry_policy": item.retry_policy.model_dump(),
            "approval_gate": item.approval_gate,
            "constraints": item.constraints,
            "parallel": item.parallel,
            "comments": item.comments,
        }
        return {
            "id": item.id.upper(),
            "title": item.title,
            "description": item.description,
            "parent_id": item.parent_id.upper() if item.parent_id else None,
            # A revision describes intended future work. Runtime state belongs
            # to the harness and is never accepted from the browser payload.
            "status": TaskStatus.PENDING.value,
            "depends_on": [dependency.upper() for dependency in item.dependencies],
            "acceptance_criteria": item.acceptance_criteria,
            "verification": item.tests,
            "role": RoleProfile(
                name=item.agent_role,
                mission=f"Own {item.title}",
            ).to_dict(),
            "mode": "parallel" if item.parallel else "auto",
            "risk": item.risk_level,
            "priority": -index,
            "origin": "local-web",
            "metadata": metadata,
        }

    def _validate_and_basis(
        self, payload: PlanPayload, goal: Any, old_plan: Plan
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        values = [self._task_value(item, index) for index, item in enumerate(payload.tasks)]
        preview = tuple(
            self.store.coerce_task(value, goal.id, old_plan.revision + 1, "local-web")
            for value in values
        )
        validate_task_dag(preview)
        ids = {task.id for task in preview}
        applicability: list[dict[str, Any]] = []
        for item in old_plan.applicability_evidence:
            copied = dict(item)
            copied["supports_tasks"] = [
                str(task_id).upper()
                for task_id in copied.get("supports_tasks", ())
                if str(task_id).upper() in ids
            ]
            if copied["supports_tasks"]:
                applicability.append(copied)
        expected: list[dict[str, Any]] = []
        for item in old_plan.expected_changes:
            copied = dict(item)
            copied["supports_tasks"] = [
                str(task_id).upper()
                for task_id in copied.get("supports_tasks", ())
                if str(task_id).upper() in ids
            ]
            if copied["supports_tasks"]:
                expected.append(copied)
        evidence_ids = {
            str(task_id).upper()
            for item in applicability
            for task_id in item.get("supports_tasks", ())
        }
        expected_ids = {
            str(task_id).upper()
            for item in expected
            for task_id in item.get("supports_tasks", ())
        }
        for task in preview:
            if task.id not in evidence_ids:
                applicability.append(
                    {
                        "source": "Plan Studio user revision",
                        "fact": f"{task.title}: {task.description or task.title}",
                        "supports_tasks": [task.id],
                    }
                )
            if task.id not in expected_ids:
                expected_files = list(task.metadata.get("expected_files") or ())
                outputs = list(task.metadata.get("outputs") or ())
                paths = expected_files or outputs or [f"<resolved during {task.id}>"]
                for path in paths:
                    expected.append(
                        {
                            "path": path,
                            "intent": task.description or task.title,
                            "supports_tasks": [task.id],
                        }
                    )
        return values, applicability, expected

    def save_plan_draft(self, payload: PlanPayload) -> dict[str, Any]:
        current = self.plan_snapshot()
        if not bool(current["capabilities"]["can_save_draft"]):
            raise ValueError(
                "The active plan is read-only. Wait for a planning checkpoint before editing it."
            )
        session = self.store.get_workflow_session(self.session_id)
        state = dict(session.get("state") or {})
        state["web_plan_draft"] = {
            **payload.model_dump(),
            "saved_at": _iso_now(),
            "stale": payload.base_revision != current["revision"],
        }
        self.store.save_workflow_session(
            self.session_id,
            goal_id=session.get("goal_id"),
            session_mode=str(session["session_mode"]),
            plan_state=str(session["plan_state"]),
            run_state=str(session["run_state"]),
            ultra_profile=str(session.get("ultra_profile", "standard")),
            sleep_state=str(session.get("sleep_state", "off")),
            state=state,
        )
        self.store.append_event(
            "plan.draft.saved",
            goal_id=current["goal_id"],
            payload={
                "session_id": self.session_id,
                "base_revision": payload.base_revision,
                "source": "local_web",
            },
        )
        self.events.publish(
            "plan.draft.saved",
            f"Plan r{payload.base_revision} draft saved.",
            session_id=self.session_id,
            plan_revision=payload.base_revision,
            source="local_web",
        )
        return {"saved": True, "base_revision": payload.base_revision}

    def save_plan_revision(self, payload: PlanPayload) -> dict[str, Any]:
        with self._lock:
            goal = self._goal()
            session = self.store.get_workflow_session(self.session_id)
            session_mode = str(session["session_mode"])
            if goal.status is not GoalStatus.AWAITING_PLAN_APPROVAL:
                raise ValueError(
                    "The active plan is read-only while work is running. "
                    "Wait for a planning checkpoint before creating a revision."
                )
            if session_mode not in {"plan", "normal"}:
                raise ValueError(
                    "Ultra plans are governed by the orchestrator. "
                    "Use the guided replan flow to change this plan."
                )
            old_plan = self.store.get_latest_plan(goal.id)
            if old_plan is None:
                raise NotFoundError("there is no plan to edit")
            if old_plan.revision != payload.base_revision:
                self.events.publish(
                    "plan.revision.conflict",
                    f"Plan r{payload.base_revision} is stale; current revision is r{old_plan.revision}.",
                    session_id=self.session_id,
                    opened_revision=payload.base_revision,
                    current_revision=old_plan.revision,
                    source="local_web",
                )
                raise StalePlanError(
                    f"This page was opened with Plan r{payload.base_revision}. "
                    f"The current plan is Plan r{old_plan.revision}."
                )
            values, applicability, expected = self._validate_and_basis(payload, goal, old_plan)
            old_by_id = {task.id: _task_snapshot(task) for task in old_plan.tasks}
            new_by_id = {item.id.upper(): item.model_dump() for item in payload.tasks}
            added = sorted(set(new_by_id) - set(old_by_id))
            deleted = sorted(set(old_by_id) - set(new_by_id))
            modified = sorted(
                task_id
                for task_id in set(old_by_id) & set(new_by_id)
                if old_by_id[task_id] != new_by_id[task_id]
            )
            original_status = goal.status
            if original_status is not GoalStatus.REVISING:
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.REVISING,
                    reason="Plan Studio revision requested",
                )
            try:
                plan = self.store.create_plan(
                    goal.id,
                    payload.summary,
                    values,
                    applicability_evidence=applicability,
                    execution_strategy=(
                        old_plan.execution_strategy
                        + f"\nPlan Studio revision: {payload.change_note or 'User applied edits.'}"
                    ),
                    expected_changes=expected,
                    proposed_by="local-web",
                    submit=True,
                    expected_parent_revision=payload.base_revision,
                )
                self.store.update_goal_metadata(
                    goal.id,
                    web_plan_constraints=payload.global_constraints,
                    protected_paths=payload.protected_paths,
                    last_web_plan_revision=plan.revision,
                )
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.AWAITING_PLAN_APPROVAL,
                    reason=f"Plan Studio created r{plan.revision}",
                )
            except Exception:
                current = self.store.get_goal(goal.id)
                if current.status is GoalStatus.REVISING and original_status is not GoalStatus.REVISING:
                    self.store.transition_goal(
                        goal.id,
                        original_status,
                        reason="Plan Studio revision rolled back before creation",
                    )
                raise
            session = self.store.get_workflow_session(self.session_id)
            state = dict(session.get("state") or {})
            state.pop("web_plan_draft", None)
            state["web_plan_change_summary"] = {
                "parent_revision": payload.base_revision,
                "revision": plan.revision,
                "timestamp": _iso_now(),
                "session_id": self.session_id,
                "tasks_added": added,
                "tasks_deleted": deleted,
                "tasks_modified": modified,
            }
            self.store.save_workflow_session(
                self.session_id,
                goal_id=goal.id,
                session_mode=str(session["session_mode"]),
                plan_state=str(session["plan_state"]),
                run_state=str(session["run_state"]),
                ultra_profile=str(session.get("ultra_profile", "standard")),
                sleep_state=str(session.get("sleep_state", "off")),
                state=state,
            )
            message = (
                f"Plan revision created: r{payload.base_revision} -> r{plan.revision}; "
                f"{len(modified)} modified; {len(added)} added."
            )
            self.store.append_event(
                "plan.revision.created",
                goal_id=goal.id,
                entity_type="plan",
                entity_id=plan.id,
                payload={
                    "session_id": self.session_id,
                    "parent_revision": payload.base_revision,
                    "revision": plan.revision,
                    "tasks_added": added,
                    "tasks_deleted": deleted,
                    "tasks_modified": modified,
                    "source": "local_web",
                    "actor": "user",
                },
            )
            self.events.publish(
                "plan.revision.created",
                message,
                session_id=self.session_id,
                previous_revision=payload.base_revision,
                plan_revision=plan.revision,
                tasks_added=len(added),
                tasks_deleted=len(deleted),
                tasks_modified=len(modified),
                source="local_web",
                actor="user",
            )
            return {
                "saved": True,
                "approved": False,
                "previous_revision": payload.base_revision,
                "revision": plan.revision,
                "summary": {
                    "tasks_added": len(added),
                    "tasks_deleted": len(deleted),
                    "tasks_modified": len(modified),
                },
            }

    def apply_plan(self, payload: PlanPayload) -> dict[str, Any]:
        """Compatibility alias that creates an unapproved revision."""

        return self.save_plan_revision(payload)

    def approve_plan(self, revision: int) -> dict[str, Any]:
        with self._lock:
            goal = self._goal()
            plan = self.store.get_latest_plan(goal.id)
            if plan is None:
                raise NotFoundError("there is no plan to approve")
            if plan.revision != revision:
                raise StalePlanError(
                    f"Plan r{revision} is stale. The current plan is Plan r{plan.revision}."
                )
            if goal.status is not GoalStatus.AWAITING_PLAN_APPROVAL:
                raise ValueError("This plan is not waiting for approval.")
            if plan.status is not PlanStatus.PENDING_APPROVAL:
                raise ValueError(f"Plan r{revision} is already {plan.status.value}.")
            session = self.store.get_workflow_session(self.session_id)
            if str(session["session_mode"]) == "plan":
                raise ValueError(
                    "Plan mode can save revisions, but execution requires Normal or Ultra mode."
                )
            approved = self.runtime.approve_plan(
                revision,
                approved_by="user@local-web",
            )
            self.store.append_event(
                "plan.approved.local_web",
                goal_id=goal.id,
                entity_type="plan",
                entity_id=approved.id,
                payload={
                    "session_id": self.session_id,
                    "revision": approved.revision,
                    "source": "local_web",
                    "actor": "user",
                },
            )
            return {
                "approved": True,
                "revision": approved.revision,
                "goal_status": self.store.get_goal(goal.id).status.value,
            }

    def discard_plan_draft(self) -> dict[str, Any]:
        session = self.store.get_workflow_session(self.session_id)
        state = dict(session.get("state") or {})
        discarded = bool(state.pop("web_plan_draft", None))
        self.store.save_workflow_session(
            self.session_id,
            goal_id=session.get("goal_id"),
            session_mode=str(session["session_mode"]),
            plan_state=str(session["plan_state"]),
            run_state=str(session["run_state"]),
            ultra_profile=str(session.get("ultra_profile", "standard")),
            sleep_state=str(session.get("sleep_state", "off")),
            state=state,
        )
        return {"discarded": discarded}

    def _active_change_set(self, checkpoint_id: str | None = None) -> tuple[Any, Any]:
        goal = self._goal()
        run = self.store.get_active_ultra_run(goal.id)
        if run is None:
            raise NotFoundError("this session has no active change checkpoint")
        items = self.store.list_change_sets(run.id)
        if checkpoint_id:
            checkpoint = next((item for item in items if item.id == checkpoint_id), None)
        else:
            checkpoint = items[-1] if items else None
        if checkpoint is None:
            raise NotFoundError("change checkpoint not found")
        return run, checkpoint

    def review_snapshot(self, checkpoint_id: str | None = None) -> dict[str, Any]:
        run, checkpoint = self._active_change_set(checkpoint_id)
        files = _diff_files(checkpoint.diff, checkpoint.changed_files)
        reasons = dict(checkpoint.metadata.get("file_reasons") or {})
        task_map = dict(checkpoint.metadata.get("file_tasks") or {})
        test_results = checkpoint.metadata.get("test_results") or checkpoint.review_status
        evidence = [
            {
                "id": evidence.id,
                "summary": evidence.summary,
                "kind": evidence.kind,
                "verified": evidence.verified,
            }
            for evidence in self.store.list_evidence(run.goal_id)
            if evidence.id in set(checkpoint.verification_evidence_ids)
        ]
        for item in files:
            item["reason"] = reasons.get(item["path"], checkpoint.metadata.get("reason", "Recorded by the responsible agent."))
            item["task"] = task_map.get(item["path"], checkpoint.parent_id)
            item["agent"] = checkpoint.responsible_agent_id
            item["tests"] = test_results
            item["evidence"] = evidence
        review_history = list(checkpoint.metadata.get("user_reviews") or ())
        return {
            "session_id": self.session_id,
            "session_short": self.session_id[:8],
            "checkpoint_id": checkpoint.id,
            "checkpoint_status": checkpoint.status.value,
            "plan_revision": run.plan_revision,
            "snapshot_created_at": checkpoint.created_at.isoformat(),
            "files": files,
            "review_history": review_history,
            "immutable": True,
            "connection": "connected",
        }

    def submit_review(self, payload: ReviewSubmissionPayload) -> dict[str, Any]:
        with self._lock:
            run, checkpoint = self._active_change_set(payload.checkpoint_id)
            if checkpoint.status not in REVIEWABLE_CHANGE_SET_STATES:
                raise ValueError(f"checkpoint {checkpoint.id} is closed for review")
            files = _diff_files(checkpoint.diff, checkpoint.changed_files)
            file_paths = {item["path"] for item in files}
            hunks_by_file = {
                item["path"]: {hunk["id"] for hunk in item["hunks"]}
                for item in files
            }
            hunk_ids = set().union(*hunks_by_file.values()) if hunks_by_file else set()
            seen_targets: set[tuple[str, str, str | None]] = set()
            file_decisions: set[str] = set()
            hunk_decisions: dict[str, set[str]] = {
                path: set() for path in file_paths
            }
            for decision in payload.decisions:
                if decision.file_path not in file_paths:
                    raise ValueError(f"decision references a file outside the checkpoint: {decision.file_path}")
                target = (
                    decision.target_type,
                    decision.file_path,
                    decision.hunk_id,
                )
                if target in seen_targets:
                    raise ValueError("the same review target was submitted more than once")
                seen_targets.add(target)
                if decision.target_type == "file":
                    if decision.hunk_id is not None:
                        raise ValueError("file decisions cannot include a hunk id")
                    file_decisions.add(decision.file_path)
                else:
                    if decision.hunk_id not in hunks_by_file[decision.file_path]:
                        raise ValueError(
                            f"decision references an invalid hunk for "
                            f"{decision.file_path}: {decision.hunk_id}"
                        )
                    hunk_decisions[decision.file_path].add(str(decision.hunk_id))
                if decision.decision in {"rejected", "changes_requested"} and not decision.reason:
                    raise ValueError("rejections and change requests require a reason")
            mixed = sorted(
                path
                for path in file_paths
                if path in file_decisions and hunk_decisions[path]
            )
            if mixed:
                raise ValueError(
                    "choose either a file decision or hunk decisions, not both: "
                    + ", ".join(mixed)
                )
            unresolved = sorted(
                path
                for path in file_paths
                if path not in file_decisions
                and (
                    not hunks_by_file[path]
                    or hunk_decisions[path] != hunks_by_file[path]
                )
            )
            if unresolved:
                raise ValueError(
                    "every changed file must be resolved before submit: "
                    + ", ".join(unresolved)
                )
            for comment in payload.comments:
                if comment.file_path not in file_paths:
                    raise ValueError(f"comment references a file outside the checkpoint: {comment.file_path}")
                if comment.hunk_id and comment.hunk_id not in hunk_ids:
                    raise ValueError(f"comment references an invalid hunk: {comment.hunk_id}")
            per_file: dict[str, set[str]] = {path: set() for path in file_paths}
            for decision in payload.decisions:
                per_file[decision.file_path].add(decision.decision)
            counts = {
                "approved": sum(1 for states in per_file.values() if states and states == {"accepted"}),
                "changes_requested": sum(
                    1 for states in per_file.values()
                    if "changes_requested" in states or "rejected" in states
                ),
                "partially_accepted": sum(
                    1 for states in per_file.values()
                    if "accepted" in states and len(states) > 1
                ),
                "comments": len(payload.comments),
            }
            needs_fixer = counts["changes_requested"] > 0
            review = {
                "revision": len(checkpoint.metadata.get("user_reviews") or ()) + 1,
                "submitted_at": _iso_now(),
                "actor": "user",
                "source": "local_web",
                "plan_revision": run.plan_revision,
                "decisions": [item.model_dump() for item in payload.decisions],
                "comments": [item.model_dump() for item in payload.comments],
                "summary": payload.summary,
                "counts": counts,
            }
            metadata = dict(checkpoint.metadata)
            metadata["user_reviews"] = [*list(metadata.get("user_reviews") or ()), review]
            metadata["latest_user_review"] = review
            updated = replace(
                checkpoint,
                status=ChangeSetStatus.BLOCKED if needs_fixer else ChangeSetStatus.APPROVED,
                review_status={
                    **dict(checkpoint.review_status),
                    "user": "changes_requested" if needs_fixer else "passed",
                },
                metadata=metadata,
                updated_at=datetime.now(timezone.utc),
            )
            self.store.save_change_set(updated)
            fixer_node = None
            if needs_fixer:
                try:
                    node = self.store.get_work_node(checkpoint.parent_id)
                    feedback_lines = [
                        "User review feedback. Preserve accepted files and hunks unless a "
                        "requested fix makes a change unavoidable; explain any such change."
                    ]
                    feedback_lines.extend(
                        (
                            f"- {item.decision}: {item.file_path}"
                            f"{' ' + item.hunk_id if item.hunk_id else ''}"
                            f"{' — ' + item.reason if item.reason else ''}"
                        )
                        for item in payload.decisions
                    )
                    feedback_lines.extend(
                        (
                            f"- comment: {item.file_path}"
                            f"{':' + str(item.line) if item.line else ''} — {item.body}"
                        )
                        for item in payload.comments
                    )
                    fixer_node = self.store.transition_work_node(
                        node.id,
                        WorkNodeStatus.FIXING,
                        checkpoint="\n".join(feedback_lines)[:4000],
                        increment_attempt=True,
                    )
                except NotFoundError:
                    fixer_node = None
            event_kind = "review.changes_requested" if needs_fixer else "checkpoint.approved"
            self.store.append_event(
                "review.submitted",
                goal_id=run.goal_id,
                entity_type="change_set",
                entity_id=checkpoint.id,
                payload={
                    "session_id": self.session_id,
                    "checkpoint_id": checkpoint.id,
                    "plan_revision": run.plan_revision,
                    "counts": counts,
                    "fixer_node_id": getattr(fixer_node, "id", None),
                    "source": "local_web",
                    "actor": "user",
                },
            )
            self.events.publish(
                event_kind,
                (
                    f"Review submitted · {counts['approved']} files approved · "
                    f"{counts['changes_requested']} require changes"
                    + (" · Fixer started." if needs_fixer else ".")
                ),
                session_id=self.session_id,
                checkpoint_id=checkpoint.id,
                plan_revision=run.plan_revision,
                counts=counts,
                fixer_node_id=getattr(fixer_node, "id", None),
                source="local_web",
                actor="user",
            )
            return {
                "submitted": True,
                "checkpoint_id": checkpoint.id,
                "counts": counts,
                "fixer_started": bool(fixer_node),
                "fixer_node_id": getattr(fixer_node, "id", None),
            }

    def agents_snapshot(self) -> dict[str, Any]:
        goal = self._goal()
        run = self.store.get_active_ultra_run(goal.id)
        if run is None:
            return {
                "session_id": self.session_id,
                "session_short": self.session_id[:8],
                "plan_revision": goal.active_plan_revision,
                "core": {"id": "core", "name": "GA3BAD Core", "status": goal.status.value},
                "nodes": [],
                "agents": [],
                "updated_at": _iso_now(),
                "connection": "connected",
            }
        nodes = self.store.list_work_nodes(run.id)
        node_by_id = {node.id: node for node in nodes}
        agents = self.store.list_agent_runs(run.id)
        brain = self.store.list_brain_entries(run.id, limit=500)
        recent = self.store.list_recent_events(goal.id, limit=200)
        agent_rows = []
        for agent in agents:
            node = node_by_id.get(agent.work_node_id or "")
            memory = [
                {"id": item.id, "title": item.title, "section": item.section.value}
                for item in brain
                if item.agent_run_id == agent.id
                or (node is not None and item.work_node_id == node.id)
            ][:20]
            logs = [
                {
                    "type": item.event_type,
                    "timestamp": item.created_at.isoformat(),
                    "summary": str(item.payload.get("summary") or item.payload.get("reason") or item.event_type),
                }
                for item in recent
                if str(item.payload.get("actor") or "") == agent.id
                or item.entity_id in {agent.id, agent.work_node_id}
            ][-20:]
            elapsed = max(
                0,
                int(
                    (
                        (agent.finished_at or datetime.now(timezone.utc))
                        - agent.started_at
                    ).total_seconds()
                ),
            )
            result = agent.result.to_dict() if agent.result else {}
            progress_value = agent.usage.get("progress")
            progress = (
                int(progress_value)
                if bool(agent.usage.get("progress_authoritative"))
                and isinstance(progress_value, (int, float))
                else None
            )
            agent_rows.append(
                {
                    "id": agent.id,
                    "name": agent.role,
                    "role": agent.role,
                    "status": agent.status.value,
                    "task_id": agent.work_node_id,
                    "task": node.title if node else agent.phase,
                    "goal": node.objective if node else goal.objective,
                    "progress": progress,
                    "phase": agent.phase,
                    "last_action": logs[-1]["summary"] if logs else agent.phase,
                    "current_file": (
                        node.contract.write_paths[0]
                        if node and node.contract.write_paths
                        else ""
                    ),
                    "files_inspected": list(node.contract.read_paths) if node else [],
                    "files_modifying": list(node.contract.write_paths) if node else [],
                    "start_time": agent.started_at.isoformat(),
                    "elapsed_seconds": elapsed,
                    "retries": max(0, agent.attempt - 1),
                    "parent_agent": None,
                    "parent_node": node.parent_id if node else None,
                    "child_agents": sum(
                        1
                        for other in agents
                        if node
                        and other.work_node_id
                        and node_by_id.get(other.work_node_id)
                        and node_by_id[other.work_node_id].parent_id == node.id
                    ),
                    "blocked": bool(agent.error) or agent.status.value in {"failed", "uncertain"},
                    "blockers": [agent.error] if agent.error else [],
                    "latest_output": result,
                    "decisions": result.get("decisions", []) if isinstance(result, Mapping) else [],
                    "logs": logs,
                    "memory": memory,
                    "tools": list(agent.usage.get("tools") or ()),
                    "plan_revision": run.plan_revision,
                }
            )
        node_rows = [
            {
                "id": node.id,
                "title": node.title,
                "status": node.status.value,
                "parent_id": node.parent_id,
                "dependencies": list(node.depends_on),
                "assigned_role": node.assigned_role,
                "attempts": node.attempts,
                "blocked": node.status in {
                    WorkNodeStatus.BLOCKED,
                    WorkNodeStatus.FAILED,
                    WorkNodeStatus.CONFLICT,
                    WorkNodeStatus.UNCERTAIN,
                },
            }
            for node in nodes
        ]
        return {
            "session_id": self.session_id,
            "session_short": self.session_id[:8],
            "plan_revision": run.plan_revision,
            "core": {"id": "core", "name": "GA3BAD Core", "status": run.status.value},
            "nodes": node_rows,
            "agents": agent_rows,
            "updated_at": _iso_now(),
            "connection": "connected",
        }

    def request_agent_explanation(self, agent_id: str, question: str) -> dict[str, Any]:
        snapshot = self.agents_snapshot()
        selected = next(
            (agent for agent in snapshot["agents"] if agent["id"] == agent_id),
            None,
        )
        if selected is None:
            raise NotFoundError(f"agent is outside this session: {agent_id}")
        goal = self._goal()
        message = self.store.post_swarm_message(
            ultra_run_id=self.store.get_active_ultra_run(goal.id).id,
            sender_agent_id="user@local-web",
            recipient_agent_id=agent_id,
            message_type="request",
            topic="explain.current_work",
            payload={
                "question": question,
                "plan_revision": selected["plan_revision"],
                "requested_at": _iso_now(),
            },
            correlation_id=f"web-explain:{agent_id}:{int(datetime.now(timezone.utc).timestamp())}",
        )
        payload = {
            "session_id": self.session_id,
            "agent_id": agent_id,
            "question": question,
            "message_id": message["id"],
            "source": "local_web",
            "actor": "user",
        }
        self.store.append_event(
            "agent.explanation_requested",
            goal_id=goal.id,
            entity_type="agent_run",
            entity_id=agent_id,
            payload=payload,
        )
        self.events.publish(
            "agent.explanation_requested",
            f"Explanation requested from {agent_id}.",
            **payload,
        )
        return {"requested": True, "agent_id": agent_id, "message_id": message["id"]}

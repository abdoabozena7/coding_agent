"""Adapter from HTTP views to GA3BAD's real runtime and state store."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit

from ..models import (
    GoalStatus,
    Plan,
    PlanStatus,
    QueuedPromptStatus,
    RoleProfile,
    TaskStatus,
    validate_task_dag,
)
from ..model_catalog import ExecutionClass, ModelCatalog
from ..quality import ChangeSetStatus
from ..sandbox import AccessLevel, DockerSandbox, PermissionAdapter
from ..store import NotFoundError, StalePlanError
from ..ultra_models import WorkNodeStatus
from ..version_control import GitProtectionManager, VersionControlError
from .schemas import (
    PlanPayload,
    ReviewSubmissionPayload,
    WorkspaceActionRequest,
    WorkspaceActionReceipt,
    WorkspaceContextPayload,
    HistorySnapshotPayload,
    ThreadSnapshotPayload,
)


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


_INTERNAL_REPAIR_SUFFIX = re.compile(
    r"\s*Accepted repair requirements:\s*"
    r"ULTRA foundation/phase\s+.+?failed after three targeted typed-return repairs:.*$",
    re.IGNORECASE | re.DOTALL,
)


def _public_task_description(value: Any) -> str:
    """Keep transport/schema recovery diagnostics out of approval copy."""

    text = str(value or "").strip()
    return _INTERNAL_REPAIR_SUFFIX.sub("", text).strip()


def _public_workflow_reason(
    value: Any,
    *,
    failure_kind: str = "",
    local: bool = False,
) -> str:
    """Translate harness/provider diagnostics into stable operator copy.

    Internal contract names are useful in logs, but they are not actionable in
    the workspace.  Keep the durable error untouched in the store and expose
    one short recovery explanation to every Web projection instead.
    """

    text = str(value or "").strip()
    lower = text.casefold()
    kind = str(failure_kind or "").casefold()
    if (
        "submit_semantic_route" in lower
        or "submit_semantic_turn" in lower
        or "must be called exactly once" in lower
        or "only allowed call is" in lower
        or "semantic routing is saved but could not be validated" in lower
    ):
        return (
            "The model's routing response could not be validated. "
            "The saved request is ready for a targeted retry."
        )
    if kind in {"contract", "schema", "semantic"}:
        return (
            "The model response needs a small repair before the workflow can continue. "
            "The saved request is ready for a targeted retry."
        )
    if not text:
        if kind == "quota":
            return "The provider limit was reached. The saved request is ready to retry or switch models."
        if kind == "rate_limit":
            return "The provider asked us to wait. The saved request is ready to retry later."
        if kind == "transport" and local:
            return "The local model runner is unavailable. The saved request is preserved."
        return "The saved request is ready for recovery."
    return text


def _public_provider_recovery(
    value: Any,
    *,
    failure_kind: str = "",
    local: bool = False,
) -> dict[str, Any]:
    """Project provider recovery metadata without leaking internal errors."""

    if not isinstance(value, Mapping):
        return {}
    result = dict(value)
    if result.get("error"):
        result["error"] = _public_workflow_reason(
            result.get("error"),
            failure_kind=str(result.get("failure_kind") or failure_kind),
            local=local,
        )
    elif failure_kind:
        result["error"] = _public_workflow_reason(
            "",
            failure_kind=failure_kind,
            local=local,
        )
    return result


def _task_snapshot(task: Any) -> dict[str, Any]:
    metadata = dict(getattr(task, "metadata", {}) or {})
    retry = dict(metadata.get("retry_policy") or {})
    return {
        "id": task.id,
        "title": task.title,
        "description": _public_task_description(task.description),
        "status": task.status.value,
        "parent_id": task.parent_id,
        "dependencies": list(task.depends_on),
        "agent_role": _role_name(task.role),
        "inputs": list(metadata.get("inputs") or ()),
        "outputs": list(metadata.get("outputs") or ()),
        "expected_files": list(metadata.get("expected_files") or ()),
        "acceptance_criteria": list(task.acceptance_criteria),
        "requirement_refs": list(metadata.get("requirement_refs") or ()),
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
                    "patch_available": False,
                }
            )
    return files


class CoreWebAdapter:
    """One narrow, auditable boundary between browser actions and the core."""

    def __init__(
        self,
        runtime: Any,
        *,
        on_execution_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.runtime = runtime
        self.store = runtime.store
        self.events = runtime.events
        self.session_id = runtime.session_id
        self._lock = RLock()
        self._requested_view = "plan"
        self._on_execution_requested = on_execution_requested

    def _goal(self) -> Any:
        goal = self.runtime.active_goal() or self.store.get_latest_goal(self.session_id)
        if goal is None:
            raise NotFoundError("this session does not have a plan yet")
        return goal

    def _workspace_goal(self) -> Any | None:
        """Return completed history unless Plan explicitly starts a new request."""

        active = self.runtime.active_goal()
        if active is not None:
            return active
        latest = self.store.get_latest_goal(self.session_id)
        if (
            latest is not None
            and latest.status in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}
            and self.runtime.interaction_mode.value == "plan"
        ):
            return None
        return latest

    def _session_goal(self, goal_id: str) -> Any:
        """Resolve a goal only when it belongs to this authenticated session."""
        goal = self.store.get_goal(str(goal_id))
        if self.store.goal_session_id(goal.id) != self.session_id:
            # Do not reveal whether another session's goal exists.
            raise NotFoundError("goal not found in this session")
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

    def _pending_question(self) -> dict[str, Any] | None:
        """Project the current user question without exposing model internals."""
        sources = (
            ("intake", "intake_questions"),
            ("plan", "plan_questions"),
            ("ultra", "ultra_questions"),
        )
        for source, method_name in sources:
            getter = getattr(self.runtime, method_name, None)
            if not callable(getter):
                continue
            try:
                questions = getter()
            except Exception:
                continue
            for raw in questions or ():
                if not isinstance(raw, Mapping):
                    continue
                answer = str(raw.get("answer") or "").strip()
                if answer:
                    continue
                question_id = str(raw.get("id") or raw.get("question_id") or "").strip()
                question = str(raw.get("question") or raw.get("prompt") or "").strip()
                if not question_id or not question:
                    continue
                options = []
                for option in raw.get("options") or ():
                    if isinstance(option, Mapping):
                        options.append({
                            "label": str(option.get("label") or option.get("value") or "")[:240],
                            "value": str(option.get("value") or option.get("label") or "")[:240],
                        })
                return {
                    "source": source,
                    "id": question_id,
                    "question": question[:2_000],
                    "options": options[:6],
                    "allow_freeform": bool(raw.get("allow_freeform", True)),
                }
        return None

    def request_view(self, view_name: str) -> None:
        if view_name in {"execution", "tree"}:
            view_name = "agents"
        elif view_name == "diff":
            view_name = "review"
        if view_name not in {"plan", "review", "agents", "history", "thread"}:
            raise ValueError(f"unknown workspace view: {view_name}")
        with self._lock:
            self._requested_view = view_name

    @staticmethod
    def _history_item(event: Any) -> dict[str, Any]:
        """Project durable events into safe, human-readable milestones."""
        payload = dict(getattr(event, "payload", {}) or {})
        kind = str(getattr(event, "event_type", "event"))
        phase = str(payload.get("phase") or payload.get("stage") or "")
        actor = str(payload.get("actor") or payload.get("source") or getattr(event, "entity_type", "harness"))
        summary = str(
            payload.get("summary")
            or payload.get("message")
            or payload.get("reason")
            or kind.replace(".", " ")
        )
        why = str(payload.get("why") or payload.get("reason") or "")
        if kind.startswith("provider.") or kind.startswith("model_"):
            summary = "Model activity recorded"
            why = "The provider reported an operational signal; hidden reasoning is not shown."
        evidence = payload.get("evidence") or payload.get("evidence_ids") or []
        if not isinstance(evidence, list):
            evidence = [evidence]
        # Never leak arguments, prompts, secrets, or model reasoning into the
        # durable history surface.  Only small evidence identifiers/summaries
        # are useful for a user inspecting a milestone.
        safe_evidence: list[Any] = []
        for item in evidence[:20]:
            if isinstance(item, Mapping):
                safe_evidence.append({
                    key: str(value)[:500]
                    for key, value in item.items()
                    if key in {"id", "summary", "kind", "verified", "path"}
                })
            elif isinstance(item, (str, int, float, bool)):
                safe_evidence.append(str(item)[:500])
        return {
            "sequence": int(getattr(event, "sequence", 0)),
            "event_id": str(getattr(event, "id", "")),
            "event_type": kind,
            "phase": phase,
            "actor": actor,
            "summary": summary[:1000],
            "why": why[:1000],
            "evidence": safe_evidence,
            "workspace_mutated": bool(payload.get("workspace_mutated") or payload.get("mutation")),
            "retry_count": int(payload.get("retry_count") or payload.get("attempt") or 0),
            "next_state": str(payload.get("next_state") or payload.get("to") or ""),
            "goal_id": getattr(event, "goal_id", None),
            "entity_id": getattr(event, "entity_id", None),
            "created_at": getattr(event, "created_at").isoformat(),
        }

    def history_snapshot(
        self,
        *,
        goal_id: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
        phase: str | None = None,
        actor: str | None = None,
        failures_only: bool = False,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        if goal_id:
            selected_goal = self._session_goal(goal_id)
            goal = selected_goal.id
        else:
            current_goal = self._workspace_goal()
            goal = current_goal.id if current_goal is not None else None
        events = self.store.list_events(goal, after_sequence=max(0, int(after_sequence)), limit=min(max(int(limit), 1), 200) + 1)
        items = [self._history_item(item) for item in events]
        if phase:
            items = [item for item in items if item["phase"] == phase or phase in item["event_type"]]
        if actor:
            items = [item for item in items if item["actor"] == actor]
        if entity_id:
            items = [item for item in items if item.get("entity_id") == entity_id]
        if failures_only:
            items = [item for item in items if any(token in item["event_type"].casefold() for token in ("fail", "error", "blocked", "paused", "uncertain"))]
        page_limit = min(max(int(limit), 1), 200)
        has_more = len(items) > page_limit
        page = items[:page_limit]
        next_cursor = page[-1]["sequence"] if has_more and page else None
        payload = {
            "session_id": self.session_id,
            "goal_id": goal,
            "items": page,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "filters": {"phase": phase, "actor": actor, "entity_id": entity_id, "failures_only": failures_only},
            "connection": "connected",
            "goals": [
                {
                    "id": item.id,
                    "objective": item.objective,
                    "status": item.status.value,
                    "plan_revision": item.active_plan_revision,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in self.store.list_goals(self.session_id, limit=50)
            ],
        }
        return HistorySnapshotPayload.model_validate(payload).model_dump()

    @staticmethod
    def _thread_item(
        item_type: str,
        item_id: str,
        sequence: int,
        content_revision: int,
        created_at: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one stable thread item.

        The browser reconciles these objects by ``item_id``.  Keeping the
        identifier independent from polling timestamps is what lets a live
        heartbeat update a status row without replacing the whole thread.
        ``kind`` is retained as a friendly alias for clients that used the
        earlier presentation vocabulary.
        """

        return {
            "item_id": str(item_id),
            "type": str(item_type),
            "kind": str(item_type),
            # The public cursor is one-based.  Durable event sequences are
            # already positive; synthetic rows (initial prompt/status/plan)
            # are lifted from zero so a small page can always advance past a
            # complete sequence group on reconnect.
            "sequence": max(1, int(sequence or 0)),
            "content_revision": max(0, int(content_revision or 0)),
            "created_at": str(created_at or _iso_now()),
            "payload": dict(payload or {}),
        }

    def thread_snapshot(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Project the durable transcript and workflow into one append-only feed.

        This is intentionally a projection, not another state machine.  Chat
        messages and durable events remain the sources of truth; the stable
        item IDs make reconnects and incremental rendering idempotent.
        """

        cursor = max(0, int(after_sequence))
        page_limit = min(max(int(limit), 1), 500)
        runtime_snapshot = self.runtime.workflow_runtime_snapshot()
        content_revision = int(runtime_snapshot.content_revision or 0)
        activity_sequence = int(
            max(
                int(runtime_snapshot.activity_sequence or 0),
                int(self.store.latest_event_sequence() or 0),
            )
        )
        goal = self._workspace_goal()
        items: list[dict[str, Any]] = []
        try:
            session_record = self.store.get_workflow_session(self.session_id)
        except Exception:
            session_record = {}

        def stable_timestamp(value: Any, fallback: str) -> str:
            if hasattr(value, "isoformat"):
                return str(value.isoformat())
            text = str(value or "").strip()
            return text or fallback

        session_created_at = stable_timestamp(
            session_record.get("created_at"),
            _iso_now(),
        )
        session_updated_at = stable_timestamp(
            session_record.get("updated_at"),
            session_created_at,
        )

        timeline = self.store.list_timeline_entries(self.session_id, limit=500)
        if not timeline:
            objective = str(
                runtime_snapshot.objective
                or (goal.objective if goal is not None else "")
                or ""
            ).strip()
            if objective:
                items.append(
                    self._thread_item(
                        "user_message",
                        f"user:{self.session_id}:{hashlib.sha256(objective.encode('utf-8')).hexdigest()[:16]}",
                        0,
                        content_revision,
                        session_created_at,
                        {"role": "user", "content": objective},
                    )
                )
        for entry in timeline:
            message = dict(entry.get("message") or {})
            role = str(message.get("role") or "assistant").casefold()
            if role == "user":
                item_type = "user_message"
            elif role == "tool":
                item_type = "tool_run"
            else:
                item_type = "assistant_message"
            sequence = int(entry.get("sequence") or 0)
            event_key = str(entry.get("event_key") or "")
            item_id = f"message:{event_key or f'{self.session_id}:{sequence}'}"
            # Model/tool messages may contain rich provider metadata.  Keep the
            # user-facing feed bounded and omit hidden reasoning fields.
            payload: dict[str, Any] = {
                "role": role,
                "content": str(message.get("content") or "")[:20_000],
            }
            for key in ("name", "id", "tool_call_id"):
                if message.get(key) is not None:
                    payload[key] = str(message[key])[:500]
            items.append(
                self._thread_item(
                    item_type,
                    item_id,
                    sequence,
                    content_revision,
                    str(entry.get("created_at") or _iso_now()),
                    payload,
                )
            )

        events = self.store.list_events(
            goal.id if goal is not None else None,
            after_sequence=0,
            limit=2_000,
        )
        plan_sequence = 0
        for event in events:
            event_type = str(event.event_type or "")
            payload = dict(event.payload or {})
            summary = str(
                payload.get("summary")
                or payload.get("message")
                or payload.get("reason")
                or event_type.replace(".", " ")
            )[:1_000]
            lower = event_type.casefold()
            if any(token in lower for token in ("plan", "goal", "semantic_turn")):
                plan_sequence = max(plan_sequence, int(event.sequence or 0))
            if any(token in lower for token in ("tool", "command", "process")):
                items.append(
                    self._thread_item(
                        "tool_run",
                        f"tool:{event.id or event.sequence}",
                        int(event.sequence or 0),
                        content_revision,
                        event.created_at.isoformat(),
                        {
                            "event_type": event_type,
                            "summary": summary,
                            "phase": str(payload.get("phase") or ""),
                            "status": str(payload.get("state") or payload.get("status") or "recorded"),
                            "tool": str(payload.get("tool") or payload.get("operation") or ""),
                            "entity_id": event.entity_id,
                        },
                    )
                )
            elif "approval" in lower:
                items.append(
                    self._thread_item(
                        "approval",
                        f"approval:{event.id or event.sequence}",
                        int(event.sequence or 0),
                        content_revision,
                        event.created_at.isoformat(),
                        {
                            "event_type": event_type,
                            "summary": summary,
                            "decision": str(payload.get("decision") or payload.get("resolution") or ""),
                            "action_fingerprint": str(payload.get("action_fingerprint") or ""),
                        },
                    )
                )

        # Context is the shared recovery authority for both the plan and the
        # thread.  Read it before projecting either item so a saved provider
        # boundary cannot produce a second, empty "Plan" checkpoint.
        context = self.workspace_context()
        recovery = context.get("provider_recovery")
        has_recovery = isinstance(recovery, Mapping) and bool(recovery)

        plan_payload: dict[str, Any] | None = None
        try:
            snapshot = self.plan_snapshot()
            if snapshot.get("goal_id") or snapshot.get("state") != "new_request":
                plan_payload = {
                    "goal_id": snapshot.get("goal_id"),
                    "revision": snapshot.get("revision"),
                    "status": snapshot.get("status"),
                    "goal_status": snapshot.get("goal_status"),
                    "summary": snapshot.get("summary") or snapshot.get("current_request") or "",
                    "tasks": list(snapshot.get("tasks") or []),
                    "expected_files": [
                        path
                        for task in snapshot.get("tasks") or []
                        for path in task.get("expected_files") or []
                    ],
                    "capability_envelope": dict(snapshot.get("capability_envelope") or {}),
                }
        except Exception:
            plan_payload = None
        plan_is_material = bool(
            plan_payload
            and (
                list(plan_payload.get("tasks") or [])
                or plan_payload.get("revision") not in {None, "", "pending"}
            )
        )
        # A pre-plan provider failure is represented by the recovery item
        # below.  Keeping the synthetic empty plan beside it made the same
        # checkpoint appear twice and invited users to reopen a dead surface.
        if plan_payload is not None and (not has_recovery or plan_is_material):
            revision = plan_payload.get("revision") or "pending"
            items.append(
                self._thread_item(
                    "plan",
                    f"plan:{plan_payload.get('goal_id') or self.session_id}:r{revision}",
                    plan_sequence or (activity_sequence if not timeline else 0),
                    content_revision,
                    stable_timestamp(
                        goal.created_at if goal is not None else None,
                        session_created_at,
                    ),
                    plan_payload,
                )
            )

        approval = context.get("tool_approval")
        if isinstance(approval, Mapping) and approval.get("action_fingerprint"):
            fingerprint = str(approval.get("action_fingerprint"))
            items.append(
                self._thread_item(
                    "approval",
                    f"approval:pending:{fingerprint}",
                    int(approval.get("requested_sequence") or activity_sequence),
                    content_revision,
                    _iso_now(),
                    {
                        "pending": True,
                        "tool": str(approval.get("tool") or ""),
                        "arguments": dict(approval.get("arguments") or {}),
                        "action_fingerprint": fingerprint,
                    },
                )
            )

        if isinstance(recovery, Mapping) and recovery:
            attempt = str(
                context.get("workflow_identity", {}).get("attempt_id")
                or recovery.get("attempt_id")
                or context.get("goal", {}).get("id")
                or self.session_id
            )
            items.append(
                self._thread_item(
                    "recovery",
                    f"recovery:{attempt}",
                    activity_sequence,
                    content_revision,
                    session_updated_at,
                    dict(recovery),
                )
            )

        review_payload: dict[str, Any] | None = None
        try:
            review = self.review_snapshot()
            if review.get("checkpoint_id"):
                review_payload = {
                    "checkpoint_id": review.get("checkpoint_id"),
                    "status": review.get("checkpoint_status"),
                    "plan_revision": review.get("plan_revision"),
                    "files": list(review.get("files") or []),
                }
        except Exception:
            review_payload = None
        if review_payload is not None:
            checkpoint = str(review_payload["checkpoint_id"])
            items.append(
                self._thread_item(
                    "change_set",
                    f"changes:{checkpoint}",
                    activity_sequence,
                    content_revision,
                    _iso_now(),
                    review_payload,
                )
            )
            items.append(
                self._thread_item(
                    "review",
                    f"review:{checkpoint}",
                    activity_sequence,
                    content_revision,
                    _iso_now(),
                    {
                        "checkpoint_id": checkpoint,
                        "status": review_payload.get("status"),
                        "file_count": len(review_payload.get("files") or []),
                    },
                )
            )

        recovery_state = str((recovery or {}).get("state") or "").casefold() if isinstance(recovery, Mapping) else ""
        raw_status_reason = str(runtime_snapshot.reason or "").strip()
        status_reason = _public_workflow_reason(
            raw_status_reason,
            failure_kind=str(
                runtime_snapshot.failure_kind
                or (context.get("workflow_identity") or {}).get("failure_kind")
                or ""
            ),
            local=str((context.get("runtime") or {}).get("execution_class") or "").casefold() == "local",
        ) if raw_status_reason else ""
        if has_recovery:
            status_reason = (
                "Retrying the saved checkpoint; your original request is preserved."
                if recovery_state in {"retrying", "waiting", "paused", "recovering"}
                else "The saved checkpoint is preserved and ready for recovery."
            )
        elif not status_reason:
            status_reason = f"Working with {runtime_snapshot.provider or 'the selected provider'}/{runtime_snapshot.model or 'model'}."
        status_payload = {
            "phase": runtime_snapshot.phase,
            "status": (context.get("goal") or {}).get("status") or runtime_snapshot.phase,
            "liveness": runtime_snapshot.liveness,
            "model": runtime_snapshot.model,
            "provider": runtime_snapshot.provider,
            "waiting_on": runtime_snapshot.waiting_on,
            "reason": status_reason,
            "activity_sequence": activity_sequence,
        }
        items.append(
            self._thread_item(
                "workflow_status",
                f"status:{self.session_id}:{content_revision}",
                activity_sequence,
                content_revision,
                session_updated_at,
                status_payload,
            )
        )
        if goal is not None and goal.status in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}:
            items.append(
                self._thread_item(
                    "completion",
                    f"completion:{goal.id}",
                    activity_sequence,
                    content_revision,
                    goal.updated_at.isoformat(),
                    {
                        "status": goal.status.value,
                        "summary": str(
                            goal.metadata.get("completion_summary")
                            or goal.metadata.get("outcome_summary")
                            or "Workflow finished."
                        ),
                    },
                )
            )

        # Stable ordering is part of the API contract.  Do not let a newly
        # projected status row reorder the conversation during a heartbeat.
        items.sort(key=lambda item: (int(item["sequence"]), str(item["item_id"])))
        filtered = [
            item
            for item in items
            if int(item["sequence"]) > cursor
            or (cursor == 0 and int(item["sequence"]) == 0)
        ]
        # Never cut a page through one activity sequence.  The cursor is a
        # sequence cursor (rather than an item offset), so doing so would
        # otherwise skip the remaining rows that share the boundary sequence
        # on the next reconnect.  A single large sequence is allowed to make
        # the page slightly larger than ``limit``; completeness is safer than
        # silently losing an approval or tool row.
        page_end = min(page_limit, len(filtered))
        if page_end and page_end < len(filtered):
            boundary_sequence = int(filtered[page_end - 1]["sequence"])
            while page_end < len(filtered) and int(filtered[page_end]["sequence"]) == boundary_sequence:
                page_end += 1
        page = filtered[:page_end]
        has_more = len(filtered) > len(page)
        next_sequence = max(
            [cursor, *(int(item["sequence"]) for item in page)]
        ) if page else cursor
        return ThreadSnapshotPayload.model_validate({
            "session_id": self.session_id,
            "items": page,
            "after_sequence": cursor,
            "next_sequence": next_sequence,
            "has_more": has_more,
            "activity_sequence": activity_sequence,
            "content_revision": content_revision,
            "connection": "connected",
        }).model_dump()

    def sessions_index_snapshot(self) -> dict[str, Any]:
        """Return the left-rail project/task list without exposing secrets."""

        session = self.store.get_workflow_session(self.session_id)
        goals = self.store.list_goals(self.session_id, limit=100)
        current = self._workspace_goal()
        updated_at = session.get("updated_at")
        if hasattr(updated_at, "isoformat"):
            updated_at = updated_at.isoformat()
        updated_at = str(updated_at or "")
        if not updated_at:
            updated_at = max(
                (goal.updated_at for goal in goals),
                default=datetime.now(timezone.utc),
            ).isoformat()
        return {
            "session_id": self.session_id,
            "projects": [
                {
                    "id": self.session_id,
                    "name": self.runtime.workspace.name,
                    "path": str(self.runtime.workspace),
                    "active": True,
                    "session_mode": str(session.get("session_mode") or "normal"),
                    "updated_at": updated_at,
                }
            ],
            "tasks": [
                {
                    "id": goal.id,
                    "title": goal.objective,
                    "status": goal.status.value,
                    "active": bool(current is not None and current.id == goal.id),
                    "plan_revision": goal.active_plan_revision,
                    "updated_at": goal.updated_at.isoformat(),
                }
                for goal in goals
            ],
            "pinned": [],
            "archived": [],
        }

    def inspector_snapshot(self, section: str | None = None) -> dict[str, Any]:
        """Project the durable environment details for the right Inspector."""

        settings = self.project_settings_snapshot()
        runtime_snapshot = self.runtime.workflow_runtime_snapshot()
        goal = self._workspace_goal()
        execution: dict[str, Any] = {
            "nodes": [],
            "tree": [],
            "agents": [],
            "result": {"artifacts": [], "changed_files": []},
        }
        try:
            execution = self.execution_snapshot()
        except NotFoundError:
            pass
        changes: list[dict[str, Any]] = []
        for path in execution.get("result", {}).get("changed_files", []) or []:
            changes.append({"path": str(path), "status": "modified"})
        try:
            review = self.review_snapshot()
            for file in review.get("files") or []:
                row = {"path": str(file.get("path") or ""), **dict(file)}
                if not any(item.get("path") == row.get("path") for item in changes):
                    changes.append(row)
        except Exception:
            pass
        artifacts = []
        try:
            artifacts = [
                {
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("suggested_name") or item.get("kind") or "artifact"),
                    "kind": str(item.get("kind") or "artifact"),
                    "size": int(item.get("byte_size") or 0),
                }
                for item in self.store.list_chat_artifacts(self.session_id)
            ]
        except Exception:
            artifacts = []
        runtime = runtime_snapshot.to_dict()
        environment = {
            "workspace": str(self.runtime.workspace),
            "name": self.runtime.workspace.name,
            "session_id": self.session_id,
            "platform": __import__("platform").system(),
            "read_only": False,
        }
        model = {
            **dict(settings.get("model") or {}),
            "capability_band": runtime.get("capability_band"),
            "attempt_id": runtime.get("attempt_id"),
            "attempt_state": runtime.get("attempt_state"),
            "local_adaptation_policy": dict(runtime.get("local_adaptation_policy") or {}),
        }
        protection = dict(settings.get("protection") or {})
        inspector = {
            "session_id": self.session_id,
            "content_revision": int(runtime_snapshot.content_revision or 0),
            "activity_sequence": int(runtime_snapshot.activity_sequence or 0),
            "selected_section": str(section or "environment"),
            "environment": environment,
            "model": model,
            "git": protection,
            "access": {
                "level": settings.get("access_level", "normal"),
                "safe_checkpoint": bool(settings.get("safe_checkpoint")),
            },
            "sleep": {
                "enabled": bool(self.runtime.sleep_mode_enabled()),
                "policy": self.runtime.sleep_mode_policy(),
            },
            "changes": changes,
            "processes": [
                {
                    "id": str(agent.get("id") or ""),
                    "name": str(agent.get("name") or agent.get("role") or "agent"),
                    "status": str(agent.get("status") or "unknown"),
                    "task": str(agent.get("task") or ""),
                    "elapsed_seconds": int(agent.get("elapsed_seconds") or 0),
                }
                for agent in execution.get("agents", []) or []
            ]
            + ([
                {
                    "id": "runtime",
                    "name": "GA3BAD runtime",
                    "status": str(runtime.get("phase") or "ready"),
                    "task": str(runtime.get("active_operation") or ""),
                    "elapsed_seconds": 0,
                }
            ] if runtime.get("active_operation") else []),
            "agents": list(execution.get("agents") or []),
            "tree": list(execution.get("tree") or []),
            "sources": {
                "artifacts": artifacts,
                "timeline_entries": len(self.store.list_timeline_entries(self.session_id, limit=500)),
                "goal_id": goal.id if goal is not None else None,
            },
            "project_settings": settings,
        }
        if section and section in inspector:
            return {
                "session_id": self.session_id,
                "content_revision": inspector["content_revision"],
                "activity_sequence": inspector["activity_sequence"],
                "selected_section": section,
                "section": inspector[section],
            }
        return inspector

    def plan_revisions_snapshot(self, goal_id: str | None = None) -> dict[str, Any]:
        goal = self._goal() if goal_id is None else self._session_goal(goal_id)
        current = self.store.get_latest_plan(goal.id)
        return {
            "goal_id": goal.id,
            "current_revision": current.revision if current else None,
            "revisions": [
                {
                    "revision": item.revision,
                    "status": item.status.value,
                    "summary": item.summary,
                    "fingerprint": item.fingerprint,
                    "execution_strategy": item.execution_strategy,
                    "task_count": len(item.tasks),
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                    "accepted_by": item.accepted_by,
                    "accepted_at": item.accepted_at.isoformat() if item.accepted_at else None,
                    "parent_revision": item.revision - 1 if item.revision > 1 else None,
                }
                for item in self.store.list_plan_revisions(goal.id)
            ],
        }

    def execution_snapshot(self) -> dict[str, Any]:
        snapshot = self.agents_snapshot()
        nodes = list(snapshot.get("nodes") or [])
        node_map = {str(item["id"]): {**item, "children": []} for item in nodes}
        roots: list[dict[str, Any]] = []
        for item in node_map.values():
            parent = str(item.get("parent_id") or "")
            if parent and parent in node_map:
                node_map[parent]["children"].append(item)
            else:
                roots.append(item)
        snapshot["tree"] = roots
        snapshot["read_only"] = True
        return snapshot

    def model_catalog_snapshot(self) -> dict[str, Any]:
        """Return a secret-free, on-demand model picker snapshot.

        Discovery stays outside the frequently-polled workspace payload so an
        offline local provider can never make the main control surface slow.
        Runtime replacement still enforces the authoritative safe-checkpoint
        and equal-or-stronger capability rules.
        """

        catalog = ModelCatalog(timeout=1.5)
        models = list(catalog.discover())
        current_id = (
            self.runtime.model_descriptor.id
            if getattr(self.runtime, "model_descriptor", None) is not None
            else ""
        )
        if not current_id:
            current_id = f"{self.runtime.provider_name}:{self.runtime.model_name}"
        goal = self._workspace_goal()
        safe_checkpoint = not bool(
            goal is not None
            and goal.status in {
                GoalStatus.RUNNING,
                GoalStatus.VERIFYING,
                GoalStatus.REVIEWING,
                GoalStatus.RECOVERING,
            }
        )
        ultra_session = getattr(self.runtime, "ultra_session", None)
        if (
            safe_checkpoint
            and ultra_session is not None
            and bool(getattr(ultra_session, "running", False))
            and not bool(getattr(ultra_session, "safe_for_reconfiguration", False))
        ):
            safe_checkpoint = False
        return {
            "current_id": current_id,
            "safe_checkpoint": safe_checkpoint,
            "models": [
                {
                    **item.to_dict(),
                    "display_name": item.display_name,
                    "selected": item.id == current_id,
                }
                for item in models
            ],
            "diagnostics": [
                {"source": item.source, "message": item.message}
                for item in catalog.diagnostics
            ],
        }

    def project_settings_snapshot(self) -> dict[str, Any]:
        """Expose the durable project profile used when reopening a workspace."""

        state = dict(self.runtime.session_snapshot())
        descriptor = self.runtime.model_descriptor
        model = (
            descriptor.to_dict()
            if descriptor is not None
            else {
                "provider": self.runtime.provider_name,
                "model": self.runtime.model_name,
                "execution_class": self.runtime.execution_class,
            }
        )
        manager = GitProtectionManager(self.runtime.workspace)
        config = manager.load_config()
        try:
            status = manager.inspect()
        except (OSError, RuntimeError, VersionControlError) as exc:
            # Git/GitHub diagnostics are informative, not a reason to make the
            # settings surface disappear.  The persisted provider remains the
            # source of truth until the user explicitly reconfigures it.
            status = None
        goal = self.runtime.active_goal()
        mode_lock = self.runtime.workflow_mode_lock()
        safe_checkpoint = not bool(
            goal is not None
            and goal.status in {
                GoalStatus.RUNNING,
                GoalStatus.VERIFYING,
                GoalStatus.REVIEWING,
                GoalStatus.RECOVERING,
            }
        )
        ultra_session = getattr(self.runtime, "ultra_session", None)
        if (
            safe_checkpoint
            and ultra_session is not None
            and bool(getattr(ultra_session, "running", False))
            and not bool(getattr(ultra_session, "safe_for_reconfiguration", False))
        ):
            safe_checkpoint = False
        return {
            "safe_checkpoint": safe_checkpoint,
            "mode_reconfigurable": not mode_lock.locked,
            "mode_lock_reason": mode_lock.reason if mode_lock.locked else "",
            "mode": "plan" if self.runtime.interaction_mode.value == "plan" else "working",
            "model": model,
            "access_level": self.runtime.access_level,
            "concurrency": int(state.get("concurrency") or self.runtime._workflow_concurrency_limit()),
            "protection": {
                "configured_provider": str(config.provider or "snapshot"),
                "tier": status.tier if status is not None else str(config.provider or "snapshot"),
                "github_connected": bool(status.github_connected) if status is not None else config.provider == "github",
                "dedicated_repository": bool(status.dedicated_repository) if status is not None else False,
                "git_available": bool(status.git_available) if status is not None else False,
                "branch": status.branch if status is not None else "",
                "dirty": bool(status.dirty) if status is not None else False,
                "commit_count": int(status.commit_count) if status is not None else 0,
                "gh_available": bool(status.gh_available) if status is not None else False,
                "gh_authenticated": bool(status.gh_authenticated) if status is not None else False,
                "detail": status.detail if status is not None else "Protection diagnostics are temporarily unavailable.",
            },
            "reopen_behavior": "These settings are reused when this project opens again. Change them here at a safe checkpoint.",
        }

    def apply_workspace_action(self, request: WorkspaceActionRequest) -> dict[str, Any]:
        """Route every browser action through one idempotent controller."""
        action = request.action
        if request.source == "terminal_fallback":
            # The runtime may expose a live web connection state.  Do not let
            # an accidental terminal action become a second approval surface.
            server = getattr(self.runtime, "local_web_server", None)
            web_connected = bool(
                getattr(self.runtime, "web_control_connected", None)
                if hasattr(self.runtime, "web_control_connected")
                else getattr(server, "running", False)
            )
            if web_connected:
                raise ValueError("Terminal fallback is available only while the Web workspace is disconnected.")
        goal = self._workspace_goal()
        before = self.store.latest_event_sequence()
        duplicate = False
        if goal is not None and request.action_fingerprint:
            pending = dict(goal.metadata.get("pending_tool_approval") or {})
            requested_sequence = (
                int(pending.get("requested_sequence") or 0)
                if action in {"allow_tool", "allow_tool_session", "deny_tool"}
                else 0
            )
            for event in self.store.list_recent_events(goal.id, limit=200):
                if (
                    str(event.payload.get("action_fingerprint") or "") == request.action_fingerprint
                    and event.event_type in {"workspace.action.accepted", "approval.received"}
                    and (not requested_sequence or int(event.sequence or 0) >= requested_sequence)
                ):
                    duplicate = True
                    break
        if duplicate:
            duplicate_message = (
                "Approval already applied. Workspace state has been refreshed."
                if action in {"allow_tool", "allow_tool_session", "deny_tool"}
                else "This action was already accepted; returning the original receipt."
            )
            return WorkspaceActionReceipt(
                accepted=True, action=action, source=request.source, duplicate=True,
                next_view=self._requested_view, next_phase=self.runtime.workflow_runtime_snapshot().phase,
                event_sequence=before, message=duplicate_message,
            ).model_dump()
        if request.expected_sequence is not None and request.expected_sequence != self.store.latest_event_sequence():
            raise ValueError("workspace action sequence is stale; refresh the Workspace and retry")
        action_message = "Workspace action accepted."
        if action == "approve_plan":
            revision = int(request.value or request.target_id or 0)
            self.approve_plan(revision)
            action_message = f"Plan r{revision} approved. Work is starting."
        elif action in {"allow_tool", "allow_tool_session", "deny_tool"}:
            decision = (
                "allow_session"
                if action == "allow_tool_session"
                else "allow_once" if action == "allow_tool" else "deny"
            )
            self.resolve_tool_approval(request.action_fingerprint or str(request.target_id or ""), decision)
            if action == "allow_tool_session":
                action_message = "Allowed for this session. Matching actions will continue without asking again."
        elif action == "retry":
            resume = getattr(self.runtime, "resume", None)
            if not callable(resume):
                raise ValueError("saved workflow cannot be retried")
            resume()
        elif action == "resume":
            self.runtime.resume()
        elif action == "pause":
            self.runtime.pause("paused from Workspace")
        elif action == "stop":
            self.runtime.stop_now()
        elif action in {"sleep_on", "sleep_full_on", "sleep_off"}:
            if action == "sleep_full_on" and str(request.value or "") != "FULL AUTO":
                raise ValueError("Type FULL AUTO to confirm unrestricted unattended approvals.")
            enabled = action != "sleep_off"
            policy = "full" if action == "sleep_full_on" else "safe"
            self.runtime.set_sleep_mode(enabled, policy=policy)
            resumed = False
            resolved_boundaries: tuple[str, ...] = ()
            if enabled:
                auto_resolve_boundary = getattr(self.runtime, "auto_resolve_full_auto_boundary", None)
                auto_resolve_tool = getattr(self.runtime, "auto_resolve_pending_sleep_approval", None)
                if policy == "full" and callable(auto_resolve_boundary):
                    candidate = auto_resolve_boundary()
                    resolved_boundaries = (
                        tuple(str(item) for item in candidate)
                        if isinstance(candidate, (tuple, list, set))
                        else ()
                    )
                elif callable(auto_resolve_tool):
                    resumed = bool(auto_resolve_tool())
                resumed = resumed or bool(resolved_boundaries)
                # Enabling Full Auto from Web must wake the owner even when
                # the terminal has not yet materialized an in-memory
                # AttentionRequest (for example, a provider failure is
                # already durable but the controller is between loop turns).
                # The wake is a coalesced read of durable state; it never
                # approves an action by itself and therefore cannot replay a
                # prompt or bypass the normal boundary policy.
                if self._on_execution_requested is not None:
                    self._on_execution_requested()
            action_message = (
                "Full Auto enabled. Critic-reviewed plans and every tool approval in this workspace will be accepted and audited."
                if action == "sleep_full_on"
                else "Safe Auto enabled. Reversible project checks and previews continue unattended."
                if action == "sleep_on"
                else "Unattended approvals are off."
            )
            if resumed:
                action_message += " The pending action was approved and resumed."
        elif action == "switch_model":
            descriptor_id = str(request.value or request.target_id or "").strip()
            if not descriptor_id:
                raise ValueError("Choose a model before switching.")
            catalog = ModelCatalog(timeout=1.5)
            descriptor = catalog.by_id(descriptor_id)
            if descriptor is None:
                raise ValueError("That model is no longer available. Refresh the model list.")
            provider = descriptor.create_provider()
            setattr(provider, "reasoning_effort", self.runtime.reasoning_effort)
            self.runtime.replace_provider(provider, descriptor)
            action_message = f"Model changed to {descriptor.provider}/{descriptor.model}."
        elif action == "reconfigure_protection":
            provider_name = str(request.value or request.target_id or "").strip().casefold()
            if provider_name not in {"github", "local_git", "snapshot"}:
                raise ValueError("Choose GitHub protection, local Git, or snapshot-only recovery.")
            active = self.runtime.active_goal()
            if active is not None and active.status in {
                GoalStatus.RUNNING,
                GoalStatus.VERIFYING,
                GoalStatus.REVIEWING,
                GoalStatus.RECOVERING,
            }:
                raise ValueError("Project protection can change only at a safe checkpoint.")
            manager = GitProtectionManager(self.runtime.workspace)
            try:
                if provider_name == "github":
                    status = manager.connect_github_private()
                elif provider_name == "local_git":
                    status = manager.ensure_local_history()
                    manager.configure(auto_checkpoint=True, auto_push=False, provider="local_git")
                    status = manager.inspect()
                else:
                    status = manager.use_snapshot_only()
            except VersionControlError as exc:
                raise ValueError(str(exc)) from exc
            self.store.append_event(
                "project.settings.changed",
                goal_id=active.id if active is not None else None,
                entity_type="project_settings",
                entity_id="protection",
                payload={"provider": provider_name, "tier": status.tier, "source": request.source},
            )
            action_message = {
                "github": "Project protection now uses GitHub with local checkpoints.",
                "local_git": "Project protection now uses local Git checkpoints without pushing to GitHub.",
                "snapshot": "Project protection now uses current-run snapshots only.",
            }[provider_name]
        elif action == "reconfigure_permissions":
            level = str(request.value or request.target_id or "").strip().casefold()
            if level not in {AccessLevel.NORMAL.value, AccessLevel.FULL.value}:
                raise ValueError("Permissions must be Normal or Full.")
            active = self.runtime.active_goal()
            if active is not None and active.status in {
                GoalStatus.RUNNING,
                GoalStatus.VERIFYING,
                GoalStatus.REVIEWING,
                GoalStatus.RECOVERING,
            }:
                raise ValueError("Permissions can change only at a safe checkpoint.")
            sandbox = (
                self.runtime.permission_adapter.sandbox
                if self.runtime.permission_adapter is not None
                else DockerSandbox()
            )
            adapter = PermissionAdapter(AccessLevel.parse(level), sandbox)
            self.runtime.replace_permission_adapter(adapter)
            action_message = f"Project permissions set to {adapter.access_level.value}."
        elif action == "reconfigure_mode":
            mode = str(request.value or request.target_id or "").strip().casefold()
            mode = {"working": "normal", "goal": "normal"}.get(mode, mode)
            if mode not in {"normal", "plan"}:
                raise ValueError("Workflow defaults must be Working or Plan.")
            try:
                actual = self.runtime.transition_mode(mode)
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
            action_message = f"Project workflow default set to {'Plan' if actual == 'plan' else 'Working'}."
        elif action == "reconfigure_concurrency":
            try:
                requested = int(str(request.value or request.target_id or "").strip())
            except ValueError as exc:
                raise ValueError("Agent capacity must be a whole number from 1 to 8.") from exc
            if requested < 1 or requested > 8:
                raise ValueError("Agent capacity must be a whole number from 1 to 8.")
            active = self.runtime.active_goal()
            if active is not None and active.status in {
                GoalStatus.RUNNING,
                GoalStatus.VERIFYING,
                GoalStatus.REVIEWING,
                GoalStatus.RECOVERING,
            }:
                raise ValueError("Agent capacity can change only at a safe checkpoint.")
            maximum = int(self.runtime.model_capability_envelope().max_concurrency or 1)
            if requested > maximum:
                raise ValueError(f"This model supports at most {maximum} concurrent agent(s).")
            config = self.runtime.config
            if self.runtime.execution_class == "local":
                config = replace(config, ultra_local_concurrency=requested)
            else:
                config = replace(config, ultra_cloud_concurrency=requested)
            self.runtime.replace_config(config)
            action_message = f"Project agent capacity set to {requested}."
        elif action == "continue_local_model":
            original_goal = self.runtime.active_goal()
            session_state = self.store.get_workflow_session(self.session_id).get(
                "state", {}
            )
            pending_raw = session_state.get("pending_semantic_turn")
            pending_semantic = (
                dict(pending_raw)
                if isinstance(pending_raw, Mapping)
                and str(pending_raw.get("status") or "").casefold()
                != "completed"
                else None
            )
            if original_goal is None and pending_semantic is None:
                raise ValueError("There is no active workflow to continue locally.")
            original_status = original_goal.status if original_goal is not None else None
            if original_status is GoalStatus.RUNNING:
                self.runtime.pause("preparing a safe local-model continuation")
            catalog = ModelCatalog(timeout=1.5)
            ranked: list[tuple[tuple[Any, ...], Any, Any]] = []
            for descriptor in catalog.discover():
                if (
                    descriptor.execution_class is not ExecutionClass.LOCAL
                    or not descriptor.supports_tools
                ):
                    continue
                provider = descriptor.create_provider()
                envelope = self.runtime._capability_envelope_for(provider, descriptor)
                score = (
                    envelope.level,
                    int(envelope.context_window_tokens or 0),
                    int(envelope.maximum_output_tokens or 0),
                    int(envelope.structured_output),
                    int(envelope.thinking),
                    float(envelope.parameter_count_billions or 0),
                    descriptor.id,
                )
                ranked.append((score, descriptor, provider))
            if not ranked:
                detail = "; ".join(item.message for item in catalog.diagnostics[:3])
                raise ValueError(
                    "No tool-capable local model is available."
                    + (f" {detail}" if detail else " Start Ollama and install a tool-capable model.")
                )
            _score, descriptor, provider = max(ranked, key=lambda item: item[0])
            setattr(provider, "reasoning_effort", self.runtime.reasoning_effort)
            continuation = self.runtime.continue_with_local_model(provider, descriptor)
            current_goal = self.runtime.active_goal()
            waiting_on = str((current_goal.metadata if current_goal is not None else {}).get("waiting_on") or "")
            provider_recovery = (
                dict(current_goal.metadata.get("provider_recovery") or {})
                if current_goal is not None
                else {}
            )
            provider_boundary = bool(
                current_goal is not None
                and (
                    original_status is GoalStatus.RUNNING
                    or waiting_on in {"provider", "model"}
                    or current_goal.metadata.get("provider_failure")
                    or (
                        original_status is GoalStatus.PAUSED
                        and current_goal.active_plan_revision is None
                        and bool(provider_recovery.get("error"))
                    )
                )
            )
            resumed = False
            if pending_semantic is not None and current_goal is None:
                self.runtime.resume()
                resumed = True
            elif current_goal is not None and current_goal.status is GoalStatus.PAUSED and provider_boundary:
                self.runtime.resume()
                resumed = True
            action_message = (
                f"Continuing with {descriptor.provider}/{descriptor.model}. "
                f"Remaining work uses {continuation['abstraction_level']} packets "
                f"of at most {continuation['max_cohesive_components_per_packet']} cohesive component(s). "
                "The accepted plan, executable evidence, and independent quality gates are unchanged."
                + (" Workflow resumed." if resumed else "")
            )
        elif action == "answer":
            question_id = str(request.target_id or "").strip()
            answer = str(request.value or "").strip()
            if not question_id or not answer:
                raise ValueError("An answer requires the current question id and a non-empty value.")
            question = self._pending_question()
            if question is None or question["id"] != question_id:
                raise ValueError("That question is stale. Refresh the Workspace and answer the current question.")
            if question["source"] == "intake":
                self.runtime.answer_intake_question(question_id, answer)
            elif question["source"] == "ultra":
                self.runtime.answer_ultra_question(question_id, answer)
            else:
                self.runtime.answer_plan_question(question_id, answer)
        else:
            raise ValueError(f"unsupported workspace action: {action}")
        after = self.events.latest_sequence
        if goal is not None:
            self.store.append_event(
                "workspace.action.accepted",
                goal_id=goal.id,
                entity_type="workspace",
                entity_id=request.target_id or action,
                payload={"action": action, "action_fingerprint": request.action_fingerprint, "source": request.source, "actor": "user"},
            )
        return WorkspaceActionReceipt(
            accepted=True, action=action, source=request.source, duplicate=False,
            next_view="execution" if action in {"retry", "resume", "pause", "stop", "continue_local_model"} else self._requested_view,
            next_phase=self.runtime.workflow_runtime_snapshot().phase,
            event_sequence=max(after, self.store.latest_event_sequence()),
            message=action_message if action in {
                "approve_plan", "allow_tool_session", "switch_model", "continue_local_model",
                "reconfigure_protection", "reconfigure_permissions", "reconfigure_mode",
                "reconfigure_concurrency",
                "sleep_on", "sleep_full_on", "sleep_off"
            } else (
                "Approved by Web · Harness resuming saved action"
                if action in {"allow_tool", "allow_tool_session", "retry", "resume"}
                else action_message
            ),
        ).model_dump()

    @staticmethod
    def _queued_prompt_snapshot(item: Any) -> dict[str, Any]:
        payload = {
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
        return payload

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
        normalized_text = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
        if normalized_text == re.sub(r"\s+", " ", goal.objective).strip().casefold():
            return {
                "queued": False,
                "duplicate": True,
                "duplicate_of": "active_request",
                "item": None,
                "queue": self.queue_snapshot(),
            }
        for existing in self.store.list_queued_prompts(
            self.session_id,
            include_terminal=False,
            limit=10,
        ):
            if normalized_text == re.sub(
                r"\s+", " ", existing.text
            ).strip().casefold():
                return {
                    "queued": False,
                    "duplicate": True,
                    "duplicate_of": "queued_prompt",
                    "item": self._queued_prompt_snapshot(existing),
                    "queue": self.queue_snapshot(),
                }
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
            "duplicate": False,
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
        goal = self._workspace_goal()
        session = self.store.get_workflow_session(self.session_id)
        session_state = dict(session.get("state") or {})
        pending_raw = session_state.get("pending_semantic_turn")
        pending_semantic = (
            dict(pending_raw)
            if isinstance(pending_raw, Mapping)
            and str(pending_raw.get("status") or "").casefold() != "completed"
            else None
        )
        pending_objective = str(
            (pending_semantic or {}).get("original_input") or ""
        ).strip()
        pending_provider_boundary = bool(
            pending_semantic
            and str(pending_semantic.get("status") or "").casefold()
            == "awaiting_provider"
        )
        sleep_policy = str(
            getattr(self.runtime, "sleep_mode_policy", lambda: "off")()
        ).casefold()
        retry_not_before_raw = (pending_semantic or {}).get("retry_not_before")
        try:
            retry_not_before = (
                float(retry_not_before_raw)
                if retry_not_before_raw is not None
                else None
            )
        except (TypeError, ValueError):
            retry_not_before = None
        full_auto_retry_scheduled = bool(
            pending_provider_boundary
            and sleep_policy == "full"
            and retry_not_before is not None
            and retry_not_before > datetime.now(timezone.utc).timestamp()
        )
        runtime_snapshot = self.runtime.workflow_runtime_snapshot()
        tool_approval: dict[str, Any] | None = None
        if goal is not None:
            try:
                pending = dict(goal.metadata.get("pending_tool_approval") or {})
                pending_fingerprint = str(pending.get("action_fingerprint") or "")
                pending_decision = str(pending.get("decision") or "").casefold()
                requested_sequence = int(pending.get("requested_sequence") or 0)
                recent = self.store.list_recent_events(goal.id, limit=200)
                accepted_fingerprints = {
                    str(event.payload.get("action_fingerprint") or "")
                    for event in recent
                    if (not requested_sequence or int(event.sequence or 0) >= requested_sequence)
                    and (
                        event.event_type == "approval.received"
                        or (
                            event.event_type == "workspace.action.accepted"
                            and str(event.payload.get("action") or "")
                            in {"allow_tool", "allow_tool_session", "deny_tool"}
                        )
                    )
                }
                accepted_fingerprints.discard("")
                if pending_fingerprint and pending_fingerprint not in accepted_fingerprints and pending_decision not in {
                    "allow", "allow_once", "allow_session", "deny", "reject"
                }:
                    arguments = pending.get("arguments")
                    tool_approval = {
                        "state": "waiting",
                        "tool": str(pending.get("tool") or "action"),
                        "risk": str(pending.get("risk") or "risky"),
                        "action_fingerprint": pending_fingerprint,
                        "arguments": arguments if isinstance(arguments, Mapping) else {},
                        "policy_group": str(pending.get("policy_group") or ""),
                        "policy_reason": str(pending.get("policy_reason") or ""),
                        "policy_scope": str(pending.get("policy_scope") or "project"),
                    }
                elif not pending_fingerprint:
                    # Compatibility for older saved sessions that emitted the
                    # request before durable pending approval metadata existed.
                    resolved_after: set[str] = set()
                    for event in reversed(recent):
                        if event.event_type == "approval.received":
                            fingerprint = str(event.payload.get("action_fingerprint") or "")
                            if fingerprint:
                                resolved_after.add(fingerprint)
                            continue
                        if event.event_type != "approval.requested":
                            continue
                        event_payload = dict(event.payload or {})
                        fingerprint = str(event_payload.get("action_fingerprint") or "")
                        if not fingerprint or fingerprint in resolved_after:
                            continue
                        tool_approval = {
                            "state": "waiting",
                            "tool": str(event_payload.get("tool") or event.entity_id or "action"),
                            "risk": str(event_payload.get("risk") or "risky"),
                            "action_fingerprint": fingerprint,
                            "arguments": {},
                        }
                        break
            except Exception:
                tool_approval = None
        queue = self.queue_snapshot()
        required_view: str | None = None
        checkpoint_id: str | None = None
        review_badge = 0
        plan_badge = 0
        if goal is None and pending_semantic is not None:
            # Semantic intake is already a durable request even before the
            # Goal row exists.  Route the browser to its saved planning
            # surface so a newly opened Web workspace cannot offer a second
            # empty composer for the same request.
            required_view = "plan"
            checkpoint_id = str(pending_semantic.get("turn_id") or "") or None
            plan_badge = 1
        elif goal is not None:
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
        # A durable RUNNING goal owns the truth.  A stale plan badge must not
        # make the browser claim that approval is still pending.
        pending_question = self._pending_question()
        pending_failure_kind = str(
            (pending_semantic or {}).get("failure_kind")
            or runtime_snapshot.failure_kind
            or ""
        ).casefold()
        pending_envelope = (pending_semantic or {}).get("model_capability_envelope")
        pending_is_local = isinstance(pending_envelope, Mapping) and str(
            pending_envelope.get("execution_class") or ""
        ).casefold() == "local"
        if tool_approval is not None:
            attention = {
                "state": "waiting",
                "eyebrow": "Approval required",
                "title": f"Allow {tool_approval['tool'].replace('_', ' ')}?",
                "body": "The harness is waiting before running this computer action. Allow it once, allow matching actions for this session, or deny it.",
                "action": {"label": "Open approval", "view": "agents"},
            }
        elif pending_question is not None:
            attention = {
                "state": "waiting",
                "eyebrow": "Your answer is needed",
                "title": "Answer one planning question",
                "body": pending_question["question"],
                "action": None,
            }
        elif required_view == "review":
            attention = {
                "state": "waiting",
                "eyebrow": "Your decision is needed",
                "title": "Changes need your attention",
                "body": "Resolve every changed file. Approved work continues; requested changes start a fixer.",
                "action": {"label": "Open changes", "view": "review"},
            }
        elif required_view == "plan" and goal is not None:
            attention = {
                "state": "waiting",
                "eyebrow": "Your decision is needed",
                "title": "Plan ready to start",
                "body": "You can edit this revision, save a draft, or approve it once.",
                "action": {"label": "Open plan", "view": "plan"},
            }
        elif pending_semantic is not None:
            if full_auto_retry_scheduled:
                attention = {
                    "state": "working",
                    "eyebrow": "Full Auto recovery",
                    "title": "The saved request will retry automatically",
                    "body": "The local provider is unavailable. The exact stage is saved and Full Auto will retry it after backoff; no prompt is needed.",
                    "action": {"label": "Open saved request", "view": "plan"},
                }
            elif pending_provider_boundary:
                if pending_failure_kind == "quota":
                    boundary_eyebrow = "Provider limit reached"
                    boundary_title = "The provider limit is exhausted"
                    boundary_body = (
                        "The exact request is saved. Change the model/provider or retry "
                        "after the limit resets; no workspace changes were made."
                    )
                elif pending_failure_kind == "rate_limit":
                    boundary_eyebrow = "Provider is rate limited"
                    boundary_title = "The provider asked us to wait"
                    boundary_body = (
                        "The exact request is saved and will not be duplicated. Retry "
                        "after the shown backoff or change the model."
                    )
                elif pending_failure_kind == "contract":
                    boundary_eyebrow = "Model response needs repair"
                    boundary_title = "The saved response was not structured correctly"
                    boundary_body = (
                        "The request is saved at the same stage. Retry uses one targeted "
                        "repair and a smaller packet; no incomplete plan is accepted."
                    )
                elif pending_failure_kind == "transport" and pending_is_local:
                    boundary_eyebrow = "Local runner unavailable"
                    boundary_title = "The local model is not reachable"
                    boundary_body = (
                        "The exact request is saved. Start the local runner or change "
                        "model; no workspace changes were made."
                    )
                else:
                    boundary_eyebrow = "Saved provider checkpoint"
                    boundary_title = "The saved request needs recovery"
                    boundary_body = _public_workflow_reason(
                        pending_semantic.get("last_error")
                        or runtime_snapshot.reason,
                        failure_kind=pending_failure_kind,
                        local=pending_is_local,
                    )
                attention = {
                    "state": "blocked",
                    "eyebrow": boundary_eyebrow,
                    "title": boundary_title,
                    "body": boundary_body,
                    "action": {"label": "Open saved request", "view": "plan"},
                }
            else:
                attention = {
                    "state": "working",
                    "eyebrow": "Request already submitted",
                    "title": "The saved request is being prepared",
                    "body": "The exact prompt is durable. Do not submit it again; this view updates when planning advances.",
                    "action": {"label": "Open saved request", "view": "plan"},
                }
        elif goal is None:
            attention = {
                "state": "idle",
                "eyebrow": "Plan your next request",
                "title": "Describe what you want to change",
                "body": "The request will be inspected and planned. Nothing runs until you choose Approve & work.",
                "action": {"label": "Write a plan request", "view": "plan"},
            }
        elif goal.status is GoalStatus.COMPLETED:
            attention = {
                "state": "complete",
                "eyebrow": "Project complete",
                "title": "The workflow has finished",
                "body": "The final plan and execution evidence remain available for inspection.",
                "action": None,
            }
        elif (
            goal.status in {GoalStatus.BLOCKED, GoalStatus.PAUSED}
            and runtime_snapshot.phase not in {"working", "starting"}
        ):
            blocker = _public_workflow_reason(
                goal.metadata.get("waiting_question")
                or goal.metadata.get("retry_reason")
                or runtime_snapshot.reason
                or "The saved workflow is paused at a safe checkpoint.",
                failure_kind=str(runtime_snapshot.failure_kind or ""),
                local=self.runtime.execution_class == "local",
            )
            attention = {
                "state": "blocked",
                "eyebrow": "Workflow needs attention",
                "title": "The workflow is paused",
                "body": blocker,
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
        required_action: dict[str, Any] | None = None
        action = attention.get("action") if isinstance(attention, Mapping) else None
        if tool_approval is not None:
            required_action = {
                "kind": "allow_tool",
                "label": "Allow once",
                "description": f"Allow {tool_approval['tool'].replace('_', ' ')} to run in the approved workspace.",
                "consequence": "Allow once resumes only this action. Session access covers matching actions until this session ends. Deny leaves the workflow paused.",
                "target_view": "execution",
                "fingerprint": tool_approval.get("action_fingerprint", ""),
                "owner": "user",
            }
        elif pending_question is not None:
            required_action = {
                "kind": "answer",
                "label": "Submit answer",
                "description": pending_question["question"],
                "consequence": "The saved planning stage resumes after this answer is recorded.",
                "target_view": "plan",
                "fingerprint": pending_question["id"],
                "owner": "user",
                "question": pending_question,
            }
        elif required_view == "plan" and goal is not None:
            pending_plan = self.store.get_latest_plan(goal.id) if goal is not None else None
            required_action = {
                "kind": "approve_plan",
                "label": "Approve & work",
                "description": "Accept the current plan revision and start the same workflow.",
                "consequence": "The revision becomes read-only and execution begins.",
                "target_view": "plan",
                "fingerprint": str((pending_plan.fingerprint if pending_plan else "") or (goal.metadata if goal else {}).get("plan_fingerprint") or ""),
                "owner": "user",
            }
        elif attention.get("state") == "blocked":
            required_action = {
                "kind": "retry",
                "label": "Retry saved stage",
                "description": str(attention.get("body") or "Retry the saved stage."),
                "consequence": "The same stage resumes without replaying accepted mutations.",
                "target_view": "execution",
                "fingerprint": "",
                "owner": "user",
                "alternatives": [
                    {"kind": "switch_model", "label": "Change model"},
                    {"kind": "stop", "label": "Stop safely"},
                ],
            }
        local_policy = (
            dict(goal.metadata.get("local_continuation_policy") or {})
            if goal is not None
            else {}
        )
        local_abstraction = dict(local_policy.get("abstraction") or {})
        local_quality_floor = dict(local_policy.get("quality_floor") or {})
        local_continuation = (
            {
                "active": True,
                "model": str(local_policy.get("model") or ""),
                "provider": str(local_policy.get("provider") or ""),
                "abstraction_level": str(local_abstraction.get("level") or "bounded"),
                "remaining_tasks": len(local_policy.get("remaining_task_packets") or ()),
                "quality_gates_unchanged": bool(
                    local_quality_floor.get("completion_gates_unchanged")
                ),
            }
            if local_policy
            else None
        )
        provider_recovery = (
            _public_provider_recovery(
                goal.metadata.get("provider_recovery") or {},
                failure_kind=pending_failure_kind or str(runtime_snapshot.failure_kind or ""),
                local=self.runtime.execution_class == "local",
            )
            if goal is not None and isinstance(goal.metadata.get("provider_recovery"), Mapping)
            else (
                _public_provider_recovery({
                    "state": "waiting" if full_auto_retry_scheduled else "paused",
                    "automatic_fallback": False,
                    "full_auto_retry": full_auto_retry_scheduled,
                    "full_auto_retry_attempts": int(
                        (pending_semantic or {}).get("full_auto_retry_attempts") or 0
                    ),
                    "retry_not_before": retry_not_before,
                    "error": str((pending_semantic or {}).get("last_error") or ""),
                }, failure_kind=pending_failure_kind or str(runtime_snapshot.failure_kind or ""), local=pending_is_local)
                if pending_provider_boundary
                else None
            )
        )
        with self._lock:
            requested_view = self._requested_view
        latest_plan = self.store.get_latest_plan(goal.id) if goal is not None else None
        runtime_payload = runtime_snapshot.to_dict()
        runtime_payload["execution_class"] = self.runtime.execution_class
        raw_runtime_reason = str(runtime_snapshot.reason or "").strip()
        runtime_payload["reason"] = (
            _public_workflow_reason(
                raw_runtime_reason,
                failure_kind=str(runtime_snapshot.failure_kind or pending_failure_kind or ""),
                local=self.runtime.execution_class == "local",
            )
            if raw_runtime_reason
            else ""
        )
        raw_active_operation = str(runtime_snapshot.active_operation or "").strip()
        runtime_payload["active_operation"] = (
            _public_workflow_reason(
                raw_active_operation,
                failure_kind=str(runtime_snapshot.failure_kind or pending_failure_kind or ""),
                local=self.runtime.execution_class == "local",
            )
            if raw_active_operation
            else ""
        )
        payload = {
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
                    "plan_revision": (
                        goal.active_plan_revision
                        or (latest_plan.revision if latest_plan is not None else None)
                    ),
                }
                if goal is not None
                else (
                    {
                        "id": str(pending_semantic.get("turn_id") or "pending-semantic"),
                        "objective": pending_objective,
                        "status": "paused" if pending_provider_boundary else "discovering",
                        "plan_revision": None,
                        "provisional": True,
                    }
                    if pending_semantic is not None
                    else None
                )
            ),
            "mode": self.runtime.interaction_mode.value,
            "runtime": runtime_payload,
            "route": runtime_snapshot.route,
            "execution_strategy": runtime_snapshot.execution_strategy,
            "phase": runtime_snapshot.phase,
            "waiting_on": runtime_snapshot.waiting_on,
            "resume_action": runtime_snapshot.resume_action,
            "tool_approval": tool_approval,
            "attention": attention,
            "navigation": {
                "thread": {"badge": max(plan_badge, review_badge)},
                "plan": {"badge": plan_badge},
                "review": {"badge": review_badge},
                "agents": {"badge": 0},
                "execution": {"badge": 0},
                "history": {"badge": 0},
            },
            "capabilities": {
                "can_manage_queue": (
                    goal is not None
                    and self._can_manage_queue(goal)
                ),
                "can_open_execution": goal is not None or pending_semantic is not None,
                "can_submit_plan_request": goal is None and pending_semantic is None,
            },
            "queue": queue,
            "updated_at": _iso_now(),
            "control_surface": "web",
            "required_action": required_action,
            "pending_question": pending_question,
            "workflow_identity": {
                "session_revision": runtime_snapshot.session_revision,
                "content_revision": runtime_snapshot.content_revision,
                "attempt_id": runtime_snapshot.attempt_id,
                "attempt_state": runtime_snapshot.attempt_state,
                "attempt_model": runtime_snapshot.attempt_model,
                "retry_at": runtime_snapshot.retry_at,
                "failure_kind": runtime_snapshot.failure_kind,
                "local_adaptation_policy": dict(
                    runtime_snapshot.local_adaptation_policy
                ),
                "goal_id": (
                    goal.id
                    if goal
                    else str(pending_semantic.get("turn_id") or "") or None
                    if pending_semantic
                    else None
                ),
                "plan_revision": (
                    goal.active_plan_revision
                    or (latest_plan.revision if latest_plan is not None else None)
                ) if goal else None,
                "request_fingerprint": str(
                    (goal.metadata if goal else pending_semantic or {}).get(
                        "request_fingerprint"
                    )
                    or ""
                ),
            },
            "history_cursor": int(self.store.latest_event_sequence()),
            "sleep_enabled": sleep_policy in {"safe", "full"},
            "sleep_policy": sleep_policy,
            "local_continuation": local_continuation,
            "provider_recovery": provider_recovery,
            "project_sessions": self.sessions_index_snapshot(),
        }
        return WorkspaceContextPayload.model_validate(payload).model_dump()

    def resolve_tool_approval(self, action_fingerprint: str, decision: str) -> dict[str, Any]:
        """Resolve one exact pending tool approval through the owning runtime."""

        fingerprint = str(action_fingerprint or "").strip()
        value = str(decision or "").strip().casefold()
        if value not in {"allow", "approve", "allow_once", "allow_session", "deny", "reject"}:
            raise ValueError("decision must be Allow once, Allow this session, or Deny")
        resolver = getattr(self.runtime, "resolve_tool_approval", None)
        if not fingerprint or not callable(resolver) or not resolver(fingerprint, value):
            raise ValueError(
                "This approval is stale or is not visible in the owning terminal. Refresh and choose the current request."
            )
        allowed = value in {"allow", "approve", "allow_once", "allow_session"}
        resolved_decision = "allow_session" if value == "allow_session" else "allow" if allowed else "deny"
        return {"resolved": True, "decision": resolved_decision}

    def plan_snapshot(self) -> dict[str, Any]:
        goal = self._workspace_goal()
        runtime_snapshot = self.runtime.workflow_runtime_snapshot()
        runtime_payload = runtime_snapshot.to_dict()
        runtime_payload["execution_class"] = self.runtime.execution_class
        raw_runtime_reason = str(runtime_snapshot.reason or "").strip()
        runtime_payload["reason"] = (
            _public_workflow_reason(
                raw_runtime_reason,
                failure_kind=str(runtime_snapshot.failure_kind or ""),
                local=self.runtime.execution_class == "local",
            )
            if raw_runtime_reason
            else ""
        )
        raw_active_operation = str(runtime_snapshot.active_operation or "").strip()
        runtime_payload["active_operation"] = (
            _public_workflow_reason(
                raw_active_operation,
                failure_kind=str(runtime_snapshot.failure_kind or ""),
                local=self.runtime.execution_class == "local",
            )
            if raw_active_operation
            else ""
        )
        workflow_identity = {
            "session_revision": runtime_snapshot.session_revision,
            "content_revision": runtime_snapshot.content_revision,
            "attempt_id": runtime_snapshot.attempt_id,
            "attempt_state": runtime_snapshot.attempt_state,
            "attempt_model": runtime_snapshot.attempt_model,
            "retry_at": runtime_snapshot.retry_at,
            "failure_kind": runtime_snapshot.failure_kind,
            "provider_state": runtime_snapshot.provider_request_state,
            "local_adaptation_policy": dict(runtime_snapshot.local_adaptation_policy),
        }
        session = self.store.get_workflow_session(self.session_id)
        session_state = dict(session.get("state") or {})
        pending_raw = session_state.get("pending_semantic_turn")
        pending_semantic = (
            dict(pending_raw)
            if isinstance(pending_raw, Mapping)
            and str(pending_raw.get("status") or "").casefold() != "completed"
            else None
        )
        if goal is None and pending_semantic is not None:
            provider_boundary = (
                str(pending_semantic.get("status") or "").casefold()
                == "awaiting_provider"
            )
            objective = str(pending_semantic.get("original_input") or "")
            envelope = pending_semantic.get("model_capability_envelope")
            if not isinstance(envelope, Mapping):
                envelope = self.runtime.model_capability_envelope().to_dict()
            return {
                "session_id": self.session_id,
                "session_short": self.session_id[:8],
                "state": "preparing_plan",
                "goal_id": str(
                    pending_semantic.get("turn_id") or "pending-semantic"
                ),
                "objective": objective,
                "current_request": objective,
                "request_fingerprint": str(
                    pending_semantic.get("request_fingerprint") or ""
                ),
                "revision": None,
                "status": "blocked" if provider_boundary else "in_progress",
                "goal_status": "paused" if provider_boundary else "discovering",
                "session_mode": str(session["session_mode"]),
                "interaction_mode": str(
                    pending_semantic.get("interaction_mode")
                    or self.runtime.interaction_mode.value
                ),
                "summary": (
                    "The saved request needs a targeted retry before a plan can be created."
                    if provider_boundary
                    else "GA3BAD is preparing the first durable plan revision."
                ),
                "tasks": [],
                "global_constraints": [],
                "protected_paths": [],
                "draft": None,
                "capability_envelope": dict(envelope),
                "provider_recovery": _public_provider_recovery(
                    pending_semantic.get("provider_recovery") or {},
                    failure_kind=str(
                        pending_semantic.get("failure_kind")
                        or runtime_snapshot.failure_kind
                        or ""
                    ),
                    local=str(runtime_payload.get("execution_class") or "").casefold() == "local",
                ),
                "task_demand": dict(pending_semantic.get("task_demand") or {}),
                "strategy_decision": dict(
                    pending_semantic.get("strategy_decision") or {}
                ),
                "capabilities": {
                    "can_edit": False,
                    "can_save_draft": False,
                    "can_create_revision": False,
                    "can_approve": False,
                    "can_manage_queue": False,
                    "can_submit_request": False,
                    "can_increase_depth": False,
                },
                "queue": self.queue_snapshot(),
                "updated_at": _iso_now(),
                "connection": "connected",
                "runtime": runtime_payload,
                "workflow_identity": workflow_identity,
            }
        if goal is None:
            envelope = self.runtime.model_capability_envelope()
            return {
                "session_id": self.session_id,
                "session_short": self.session_id[:8],
                "state": "new_request",
                "goal_id": None,
                "objective": "",
                "revision": None,
                "status": "draft",
                "goal_status": "idle",
                "session_mode": "plan",
                "interaction_mode": "plan",
                "summary": "",
                "tasks": [],
                "global_constraints": [],
                "protected_paths": [],
                "draft": None,
                "capability_envelope": envelope.to_dict(),
                "task_demand": None,
                "strategy_decision": None,
                "capabilities": {
                    "can_edit": False,
                    "can_save_draft": False,
                    "can_create_revision": False,
                    "can_approve": False,
                    "can_manage_queue": False,
                    "can_submit_request": True,
                    "can_increase_depth": False,
                },
                "queue": self.queue_snapshot(),
                "updated_at": _iso_now(),
                "connection": "connected",
                "runtime": runtime_payload,
                "workflow_identity": workflow_identity,
            }
        plan = self.store.get_latest_plan(goal.id)
        if plan is None:
            session = self.store.get_workflow_session(self.session_id)
            session_mode = str(session["session_mode"])
            goal_blocked = goal.status in {
                GoalStatus.PAUSED,
                GoalStatus.BLOCKED,
                GoalStatus.CANCELLED,
            }
            return {
                "session_id": self.session_id,
                "session_short": self.session_id[:8],
                "state": "preparing_plan",
                "goal_id": goal.id,
                "objective": goal.objective,
                "current_request": goal.objective,
                "request_fingerprint": str(goal.metadata.get("request_fingerprint") or ""),
                "revision": None,
                "status": "blocked" if goal_blocked else "in_progress",
                "goal_status": goal.status.value,
                "session_mode": session_mode,
                "interaction_mode": self.runtime.interaction_mode.value,
                "summary": "GA3BAD is preparing the first durable plan revision.",
                "tasks": [],
                "global_constraints": [],
                "protected_paths": [],
                "draft": None,
                "capability_envelope": dict(
                    goal.metadata.get("model_capability_envelope")
                    or self.runtime.model_capability_envelope().to_dict()
                ),
                "provider_recovery": _public_provider_recovery(
                    goal.metadata.get("provider_recovery") or {},
                    failure_kind=str(runtime_snapshot.failure_kind or ""),
                    local=str(runtime_payload.get("execution_class") or "").casefold() == "local",
                ),
                "task_demand": dict(goal.metadata.get("task_demand") or {}),
                "strategy_decision": dict(
                    goal.metadata.get("strategy_decision") or {}
                ),
                "capabilities": {
                    "can_edit": False,
                    "can_save_draft": False,
                    "can_create_revision": False,
                    "can_approve": False,
                    "can_manage_queue": self._can_manage_queue(goal),
                    "can_submit_request": False,
                    "can_increase_depth": False,
                },
                "queue": self.queue_snapshot(),
                "updated_at": _iso_now(),
                "connection": "connected",
                "runtime": runtime_payload,
                "workflow_identity": workflow_identity,
            }
        session_mode = str(session["session_mode"])
        strategy = str(
            goal.metadata.get("execution_strategy")
            or dict(goal.metadata.get("execution_policy") or {}).get("strategy")
            or ("recursive" if session_mode == "ultra" else "staged")
        )
        strategy_locked = bool(goal.metadata.get("strategy_locked"))
        semantic_goal = dict(goal.metadata.get("semantic_goal") or {})
        strategy_decision = dict(goal.metadata.get("strategy_decision") or {})
        execution_nodes: list[dict[str, Any]] = []
        run = self.store.get_active_ultra_run(goal.id)
        if run is not None:
            execution_nodes = [
                {
                    "id": node.id,
                    "title": node.title,
                    "objective": node.objective,
                    "parent_id": node.parent_id,
                    "dependencies": list(node.depends_on),
                    "assigned_role": node.assigned_role,
                    "status": node.status.value,
                }
                for node in self.store.list_work_nodes(run.id)
            ]
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
            "fingerprint": plan.fingerprint,
            "status": plan.status.value,
            "goal_status": goal.status.value,
            "session_mode": session_mode,
            "interaction_mode": self.runtime.interaction_mode.value,
            "execution_strategy": strategy,
            "strategy_locked": strategy_locked,
            "capability_envelope": dict(goal.metadata.get("model_capability_envelope") or {}),
            "provider_recovery": _public_provider_recovery(
                goal.metadata.get("provider_recovery") or {},
                failure_kind=str(runtime_snapshot.failure_kind or ""),
                local=str(runtime_payload.get("execution_class") or "").casefold() == "local",
            ),
            "workflow_identity": workflow_identity,
            "task_demand": dict(goal.metadata.get("task_demand") or {}),
            "strategy_decision": strategy_decision,
            "model_fit": self._model_fit_snapshot(strategy_decision),
            "revisions": [
                {
                    "revision": item.revision,
                    "status": item.status.value,
                    "summary": item.summary,
                    "fingerprint": item.fingerprint,
                    "created_at": item.created_at.isoformat(),
                    "accepted_at": item.accepted_at.isoformat() if item.accepted_at else None,
                }
                for item in self.store.list_plan_revisions(goal.id)
            ],
            "semantic_goal": semantic_goal,
            "execution_nodes": execution_nodes,
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
                ),
                "can_manage_queue": self._can_manage_queue(goal),
                "can_submit_request": False,
                "can_increase_depth": (
                    goal.status is GoalStatus.AWAITING_PLAN_APPROVAL
                    and plan.status is PlanStatus.PENDING_APPROVAL
                    and not strategy_locked
                    and strategy == "staged"
                ),
            },
            "queue": self.queue_snapshot(),
            "updated_at": plan.updated_at.isoformat(),
            "connection": "connected",
                "runtime": runtime_payload,
        }

    @staticmethod
    def _model_fit_snapshot(strategy_decision: Mapping[str, Any]) -> dict[str, Any]:
        """Explain model fit without turning a weaker model into a false blocker.

        Recursive execution is the runtime's explicit compensation mechanism for
        demand above a model's cohesive capability.  The browser should surface
        that trade-off before approval instead of claiming the run is unsafe.
        """

        reasons = [str(item) for item in strategy_decision.get("reasons", ())]
        lowered = " ".join(reasons).casefold()
        compensated = "exceeds" in lowered or "conservative minimal" in lowered
        incomplete = "metadata is incomplete" in lowered
        if compensated:
            status = "compensated"
            summary = (
                "This model can run the plan, but the harness will split the work "
                "into smaller verified units. Expect more hand-offs and a longer run."
            )
        else:
            status = "fit"
            summary = "The selected model fits the recorded task demand."
        return {
            "status": status,
            "summary": summary,
            "metadata_complete": not incomplete,
            "reasons": reasons,
        }

    def submit_plan_request(self, request: str) -> dict[str, Any]:
        with self._lock:
            if self.runtime.active_goal() is not None:
                raise ValueError("Finish or cancel the active workflow before starting another plan.")
            value = str(request)
            # The Build/change composer is already an explicit Goal surface.
            # For real local runners, avoid spending the first call on a
            # second semantic router; ``start_goal`` creates the durable Goal
            # and lets the staged planner perform its own semantic contract.
            # A pending semantic turn still owns the request, and scripted or
            # cloud providers keep the legacy routed path for compatibility.
            session_state = dict(self.runtime.session_snapshot() or {})
            pending = session_state.get("pending_semantic_turn")
            pending_active = isinstance(pending, Mapping) and str(
                pending.get("status") or ""
            ).casefold() != "completed"
            local_provider = str(getattr(self.runtime, "provider_name", "")).casefold() in {
                "ollama",
                "lmstudio",
                "llamacpp",
                "local",
            }
            if (
                str(getattr(self.runtime, "execution_class", "")).casefold() == "local"
                and local_provider
                and not pending_active
            ):
                self.runtime.start_goal(
                    value,
                    planning_only=True,
                    execution_mode="plan",
                    entry_surface="build",
                )
            else:
                self.runtime.transition_mode("plan")
                self.runtime.route_input(value)
            return self.plan_snapshot()

    def increase_execution_depth(self) -> dict[str, Any]:
        with self._lock:
            before = self.plan_snapshot()
            if not bool(before.get("capabilities", {}).get("can_increase_depth")):
                raise ValueError("Strategy depth can increase only on an unapproved staged plan.")
            self.runtime.increase_execution_depth()
            after = self.plan_snapshot()
            return {
                "increased": True,
                "previous_revision": before.get("revision"),
                "revision": after.get("revision"),
                "strategy": after.get("execution_strategy"),
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
            "requirement_refs": item.requirement_refs,
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
        semantic_goal = dict(goal.metadata.get("semantic_goal") or {})
        anchors = {
            str(item.get("id") or "").strip().upper()
            for item in semantic_goal.get("requirement_anchors", ())
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        }
        if anchors:
            covered = {
                str(ref).strip().upper()
                for task in preview
                for ref in task.metadata.get("requirement_refs", ())
                if str(ref).strip()
            }
            unknown = covered - anchors
            if unknown:
                raise ValueError(
                    "Plan revision references unknown requirement anchors: "
                    + ", ".join(sorted(unknown))
                )
            uncovered = anchors - covered
            if uncovered:
                raise ValueError(
                    "Plan revision would drop user requirement anchors: "
                    + ", ".join(sorted(uncovered))
                )
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
        draft_value = {
            **payload.model_dump(),
            "saved_at": _iso_now(),
            "stale": payload.base_revision != current["revision"],
        }
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": {
                    **dict(current_state.get("state") or {}),
                    "web_plan_draft": draft_value,
                }
            },
            expected_revision=int(session.get("revision") or 0),
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
                    "This recursive plan is governed by its specialist hierarchy. "
                    "Use the guided replan flow to change it before approval."
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
            change_summary = {
                "parent_revision": payload.base_revision,
                "revision": plan.revision,
                "timestamp": _iso_now(),
                "session_id": self.session_id,
                "tasks_added": added,
                "tasks_deleted": deleted,
                "tasks_modified": modified,
            }
            self.store.mutate_workflow_session(
                self.session_id,
                lambda current_state: {
                    "state": {
                        **{
                            key: value
                            for key, value in dict(current_state.get("state") or {}).items()
                            if key != "web_plan_draft"
                        },
                        "web_plan_change_summary": change_summary,
                    },
                    "goal_id": current_state.get("goal_id") or goal.id,
                },
                expected_revision=int(session.get("revision") or 0),
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
            execution_requested = False
            if self._on_execution_requested is not None:
                execution_requested = bool(self._on_execution_requested())
            self.events.publish(
                "plan.approved.local_web",
                "Approval accepted in Plan Studio; execution is starting.",
                goal_id=goal.id,
                session_id=self.session_id,
                revision=approved.revision,
                execution_requested=execution_requested,
                source="local_web",
            )
            return {
                "approved": True,
                "revision": approved.revision,
                "goal_status": self.store.get_goal(goal.id).status.value,
                "execution_requested": execution_requested,
            }

    def discard_plan_draft(self) -> dict[str, Any]:
        session = self.store.get_workflow_session(self.session_id)
        discarded = bool((session.get("state") or {}).get("web_plan_draft"))
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": {
                    key: value
                    for key, value in dict(current_state.get("state") or {}).items()
                    if key != "web_plan_draft"
                }
            },
            expected_revision=int(session.get("revision") or 0),
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
            item["patch_available"] = bool(item.get("diff") or item.get("hunks"))
            item["reason"] = reasons.get(item["path"], checkpoint.metadata.get("reason", "The artifact changed during this task; inspect the recorded evidence below."))
            item["patch_message"] = "" if item["patch_available"] else "No recorded patch is available yet. Artifact/evidence inspection is required."
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
                    f"Changes submitted · {counts['approved']} files approved · "
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
        runtime_snapshot = self.runtime.workflow_runtime_snapshot()
        projected_status = goal.status.value
        # A tool approval is a durable user boundary even while the worker's
        # goal remains RUNNING in order to keep the blocked call resumable.
        # The Web execution view must not expose that implementation detail as
        # active work: its badge and summary are projections of the same
        # runtime snapshot used by the workspace header.
        if runtime_snapshot.phase == "waiting_for_approval":
            projected_status = "waiting"
        elif runtime_snapshot.phase in {
            "paused",
            "retrying",
            "waiting_for_process",
        } and projected_status == GoalStatus.RUNNING.value:
            projected_status = "paused"
        run = self.store.get_active_ultra_run(goal.id)
        if run is None:
            # Staged/normal runs intentionally do not create an UltraRun.  The
            # result surface still needs to expose the durable files and
            # evidence produced by the coordinator; otherwise a successfully
            # completed Web workflow misleadingly renders an empty result.
            changed_files: list[str] = []
            for change_set in goal.metadata.get("goal_change_sets", ()):
                if not isinstance(change_set, Mapping):
                    continue
                for path in change_set.get("changed_files", ()) or ():
                    value = str(path or "").strip()
                    if value and value not in changed_files:
                        changed_files.append(value)
            evidence_by_path: dict[str, Any] = {}
            for evidence in self.store.list_evidence(goal.id):
                data = dict(evidence.data or {})
                path = str(data.get("path") or "").strip()
                if not path:
                    continue
                previous = evidence_by_path.get(path)
                if previous is None or (bool(evidence.verified) and not bool(previous.verified)):
                    evidence_by_path[path] = evidence
                if path not in changed_files:
                    changed_files.append(path)
            artifact_rows = []
            for path, evidence in evidence_by_path.items():
                data = dict(evidence.data or {})
                artifact_rows.append(
                    {
                        "id": evidence.id,
                        "kind": "file",
                        "path": path,
                        "content_hash": str(data.get("file_hash") or ""),
                        "preview_url": "",
                        "verified": bool(evidence.verified),
                        "created_at": evidence.created_at.isoformat(),
                    }
                )
            staged_summary = str(
                goal.metadata.get("completion_summary")
                or goal.metadata.get("outcome_summary")
                or goal.metadata.get("result_summary")
                or (
                    "The workflow completed with recorded evidence."
                    if goal.status is GoalStatus.COMPLETED
                    else "Outputs appear here as the workflow records them."
                )
            )
            return {
                "session_id": self.session_id,
                "session_short": self.session_id[:8],
                "plan_revision": goal.active_plan_revision,
                "core": {"id": "core", "name": "GA3BAD Core", "status": projected_status},
                "nodes": [],
                "agents": [],
                "result": {
                    "status": goal.status.value,
                    "summary": staged_summary,
                    "artifacts": artifact_rows,
                    "changed_files": changed_files,
                },
                "updated_at": _iso_now(),
                "connection": "connected",
            }
        nodes = self.store.list_work_nodes(run.id)
        node_by_id = {node.id: node for node in nodes}
        agents = self.store.list_agent_runs(run.id)
        brain = self.store.list_brain_entries(run.id, limit=500)
        recent = self.store.list_recent_events(goal.id, limit=200)
        artifact_rows = []
        for artifact in self.store.list_artifacts(run.id, limit=200):
            uri = str(artifact.uri or "").strip()
            preview_url = ""
            if uri:
                parsed = urlsplit(uri)
                if (
                    parsed.scheme in {"http", "https"}
                    and (parsed.hostname or "").casefold() in {"127.0.0.1", "localhost", "::1"}
                    and parsed.username is None
                    and parsed.password is None
                ):
                    preview_url = parsed._replace(query="", fragment="").geturl()
            evidence = dict(artifact.evidence or {})
            artifact_rows.append({
                "id": artifact.id,
                "kind": str(artifact.kind),
                "path": str(artifact.path or ""),
                "content_hash": artifact.content_hash,
                "preview_url": preview_url,
                "verified": bool(
                    evidence.get("verified")
                    or evidence.get("passed")
                    or evidence.get("success")
                ),
                "created_at": artifact.created_at.isoformat(),
            })
        change_sets = self.store.list_change_sets(run.id)
        changed_files = list(dict.fromkeys(
            path
            for change_set in change_sets
            for path in change_set.changed_files
        ))
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
                "objective": node.objective,
                "read_paths": list(node.contract.read_paths),
                "write_paths": list(node.contract.write_paths),
                "success_criteria": list(node.contract.success_criteria),
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
            "core": {"id": "core", "name": "GA3BAD Core", "status": projected_status},
            "nodes": node_rows,
            "agents": agent_rows,
            "result": {
                "status": goal.status.value,
                "summary": str(
                    goal.metadata.get("completion_summary")
                    or goal.metadata.get("outcome_summary")
                    or goal.metadata.get("result_summary")
                    or (
                        "The workflow completed with recorded evidence."
                        if goal.status is GoalStatus.COMPLETED
                        else "Outputs appear here as the workflow records them."
                    )
                ),
                "artifacts": artifact_rows,
                "changed_files": changed_files,
            },
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

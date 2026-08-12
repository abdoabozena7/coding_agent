"""Read-only, durable projections for the standalone Advanced Tracing page."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from .models import GoalStatus
from .safety import redact_data, redact_text


TERMINAL_TRACE_STATUSES = {
    GoalStatus.COMPLETED.value,
    GoalStatus.CANCELLED.value,
    GoalStatus.BLOCKED.value,
}
READ_TOOL_MARKERS = ("read", "list", "search", "find", "glob", "inspect")


@dataclass(frozen=True, slots=True)
class RepositoryContextCandidateV1:
    query: str
    path: str
    symbol: str
    rank: int
    score: float
    provenance: tuple[str, ...]
    decision: str
    reason: str
    stage: str


@dataclass(frozen=True, slots=True)
class ContextRotationV1:
    actor: str
    model: str
    before_chars: int
    after_chars: int
    budget_chars: int
    suspended_messages: int
    checkpoint_fingerprint: str
    reason: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class AdvancedTraceSnapshotV1:
    session_id: str
    goal_id: str
    run_id: str | None
    state: str
    status: str
    cutoff_sequence: int
    payload_hash: str = ""
    revision: int = 0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(value)


def _clean_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text or text in {".", "./", "/"} or "://" in text:
        return ""
    while text.startswith("./"):
        text = text[2:]
    try:
        normalized = str(PurePosixPath(text))
    except (TypeError, ValueError):
        return ""
    if normalized.startswith("../") or normalized == "..":
        return ""
    return normalized


def _mapping_paths(value: Any, *, depth: int = 0) -> tuple[str, ...]:
    if depth > 5:
        return ()
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            is_path_key = any(
                marker in lowered
                for marker in ("path", "file", "artifact", "target", "resource")
            )
            if is_path_key and isinstance(item, str):
                path = _clean_path(item)
                if path:
                    found.append(path)
            elif is_path_key and isinstance(item, (list, tuple, set)):
                for candidate in item:
                    if isinstance(candidate, str):
                        path = _clean_path(candidate)
                        if path:
                            found.append(path)
                    else:
                        found.extend(_mapping_paths(candidate, depth=depth + 1))
            elif isinstance(item, (Mapping, list, tuple, set)):
                found.extend(_mapping_paths(item, depth=depth + 1))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found.extend(_mapping_paths(item, depth=depth + 1))
    return tuple(dict.fromkeys(found))


def _preview(value: Any, limit: int = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(_jsonable(value), ensure_ascii=False, default=str)
    return redact_text(" ".join(text.split()), limit)


class AdvancedTraceProjection:
    """Fuse existing journals into one consistent developer-facing trace."""

    def __init__(
        self,
        adapter: Any,
        *,
        goal_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.store = adapter.store
        self.session_id = adapter.session_id
        self.goal = (
            adapter._session_goal(goal_id)
            if goal_id
            else adapter._workspace_goal() or self.store.get_latest_goal(self.session_id)
        )
        self.run = None
        if self.goal is not None:
            runs = self.store.list_ultra_runs(self.goal.id)
            if run_id:
                self.run = next((item for item in runs if item.id == run_id), None)
                if self.run is None:
                    raise ValueError("run does not belong to this session goal")
            elif runs:
                self.run = runs[-1]
        # Advanced Tracing is session-wide, not Goal-only.  Bounded Actions
        # (run/open/screenshot) intentionally create no Goal, but their visible
        # activity still belongs in the same ordered observer timeline.
        all_events = self.store.list_recent_events(limit=10_000)
        self.events = tuple(
            event
            for event in all_events
            if (
                (self.goal is not None and event.goal_id == self.goal.id)
                or (
                    str(event.entity_type or "") == "workflow_session"
                    and str(event.entity_id or "") == self.session_id
                )
                or str((event.payload or {}).get("session_id") or "") == self.session_id
            )
        )

    @property
    def run_id(self) -> str | None:
        return getattr(self.run, "id", None)

    @property
    def state(self) -> str:
        if self.goal is None:
            return "LIVE" if self.events else "EMPTY"
        return (
            "FROZEN"
            if str(self.goal.status.value) in TERMINAL_TRACE_STATUSES
            else "LIVE"
        )

    def available_runs(self) -> list[dict[str, Any]]:
        rows = []
        for goal in self.store.list_goals(self.session_id, limit=100):
            runs = self.store.list_ultra_runs(goal.id)
            if runs:
                for run in runs:
                    rows.append({
                        "goal_id": goal.id,
                        "run_id": run.id,
                        "status": run.status.value,
                        "provider": run.provider,
                        "model": run.model,
                        "created_at": run.created_at.isoformat(),
                        "objective": goal.objective,
                    })
            else:
                rows.append({
                    "goal_id": goal.id,
                    "run_id": None,
                    "status": goal.status.value,
                    "provider": getattr(self.adapter.runtime, "provider_name", ""),
                    "model": getattr(self.adapter.runtime, "model_name", ""),
                    "created_at": goal.created_at.isoformat(),
                    "objective": goal.objective,
                })
        return rows

    @staticmethod
    def _event_category(kind: str) -> str:
        lowered = kind.casefold()
        if lowered.startswith("context.") or "memory" in lowered:
            return "context"
        if "plan" in lowered or "intake" in lowered:
            return "plans"
        if "agent" in lowered or "node" in lowered or "delegation" in lowered:
            return "agents"
        if "quality" in lowered or "review" in lowered or "finding" in lowered:
            return "problems"
        if "mutation" in lowered or "artifact" in lowered or "tool" in lowered or "action" in lowered:
            return "files"
        return "workflow"

    def timeline(
        self,
        *,
        after: int = 0,
        limit: int = 250,
        category: str = "",
        query: str = "",
    ) -> dict[str, Any]:
        rows = []
        needle = str(query).strip().casefold()
        for event in self.events:
            if int(event.sequence or 0) <= max(0, int(after)):
                continue
            kind = str(event.event_type or "event")
            event_category = self._event_category(kind)
            payload = dict(event.payload or {})
            summary = str(
                payload.get("summary")
                or payload.get("message")
                or payload.get("reason")
                or payload.get("result")
                or kind.replace(".", " ")
            )
            haystack = " ".join(
                (kind, summary, str(event.entity_type or ""), str(event.entity_id or ""))
            ).casefold()
            if category and event_category != category:
                continue
            if needle and needle not in haystack:
                continue
            rows.append({
                "id": event.id,
                "sequence": event.sequence,
                "kind": kind,
                "category": event_category,
                "summary": redact_text(summary, 1_000),
                "actor": str(payload.get("actor") or payload.get("role") or payload.get("source") or event.entity_type or "harness"),
                "stage": str(payload.get("stage") or payload.get("phase") or ""),
                "status": str(payload.get("status") or payload.get("to") or "recorded"),
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "timestamp": event.created_at.isoformat() if event.created_at else "",
                "payload": redact_data(payload),
            })
            if len(rows) >= max(1, min(int(limit), 1_000)):
                break
        return {
            "items": rows,
            "next_cursor": rows[-1]["sequence"] if rows else max(0, int(after)),
            "has_more": bool(rows and rows[-1]["sequence"] < (self.events[-1].sequence if self.events else 0)),
        }

    def files(self) -> dict[str, Any]:
        records: dict[str, dict[str, Any]] = {}

        def mark(path_value: Any, state: str, source: str, detail: Any = "") -> None:
            path = _clean_path(path_value)
            if not path:
                return
            row = records.setdefault(path, {
                "path": path,
                "states": [],
                "records": [],
                "diff": "",
                "verified": False,
            })
            if state not in row["states"]:
                row["states"].append(state)
            row["records"].append({
                "state": state,
                "source": source,
                "detail": _preview(detail, 500),
            })
            if state == "verified":
                row["verified"] = True

        if self.goal is not None:
            for claim in self.goal.metadata.get("resource_claims", ()) or ():
                if isinstance(claim, Mapping):
                    for path in claim.get("resolved_paths", ()) or ():
                        mark(path, "requested", "resource claim", claim.get("reason", "accepted scope"))
            for plan in self.store.list_plan_revisions(self.goal.id):
                for task in self.store.list_tasks(self.goal.id, plan.revision):
                    metadata = dict(task.metadata or {})
                    for path in metadata.get("expected_files", ()) or ():
                        mark(path, "requested", f"plan r{plan.revision} / {task.task_id}", task.title)
            for action in self.store.list_actions(self.goal.id):
                try:
                    args = json.loads(str(action.get("args_json") or "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool = str(action.get("tool_name") or "tool")
                status = str(action.get("status") or "")
                for path in _mapping_paths(args):
                    mark(path, "requested", tool, args)
                    if any(marker in tool.casefold() for marker in READ_TOOL_MARKERS):
                        mark(
                            path,
                            "opened" if status == "completed" else "failed_access",
                            tool,
                            action.get("result_summary") or status,
                        )
            for evidence in self.store.list_evidence(self.goal.id):
                data = dict(evidence.data or {})
                for path in _mapping_paths(data):
                    mark(path, "verified" if evidence.verified else "evidence", evidence.kind, evidence.summary)

        retrievals = []
        for event in self.events:
            if event.event_type != "context.repository_retrieval":
                continue
            payload = dict(event.payload or {})
            for candidate in payload.get("candidates", ()) or ():
                if not isinstance(candidate, Mapping):
                    continue
                path = candidate.get("path")
                mark(path, "considered", "repository retrieval", payload.get("query", ""))
                outcome = str(candidate.get("outcome") or "excluded")
                mark(
                    path,
                    "selected_context" if outcome == "selected" else "excluded",
                    str(payload.get("stage") or "context"),
                    candidate.get("reason", ""),
                )
                retrievals.append(candidate)

        if self.run is not None:
            for artifact in self.store.list_artifacts(self.run.id, limit=10_000):
                if artifact.path:
                    mark(artifact.path, "artifact", artifact.kind, artifact.uri)
            for change in self.store.list_change_sets(self.run.id):
                for path in change.changed_files:
                    mark(path, "changed", change.responsible_agent_id, change.status.value)
                    row = records.get(_clean_path(path))
                    if row is not None and change.diff:
                        row["diff"] = change.diff
                for path in change.post_hashes:
                    if path in change.changed_files:
                        mark(path, "changed", "mutation ledger", change.post_hashes[path])
                if change.verification_evidence_ids:
                    for path in change.changed_files:
                        mark(path, "verified", "change set", change.verification_evidence_ids)
            for finding in self.store.list_quality_findings(self.run.id):
                mark(finding.path, "problem" if finding.status.value != "resolved" else "verified", "quality finding", finding.remediation)

        items = sorted(records.values(), key=lambda item: item["path"])
        inspect_next = []
        for item in items:
            states = set(item["states"])
            reason = ""
            if "changed" in states and "verified" not in states:
                reason = "Changed without fresh verification evidence"
            elif "problem" in states:
                reason = "An unresolved quality finding references this file"
            elif "failed_access" in states:
                reason = "A requested read or inspection failed"
            if reason:
                inspect_next.append({"path": item["path"], "reason": reason})
        return {"items": items, "inspect_next": inspect_next}

    def problems(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if self.run is not None:
            for finding in self.store.list_quality_findings(self.run.id):
                items.append({
                    "id": finding.id,
                    "kind": "quality_finding",
                    "status": finding.status.value,
                    "severity": finding.severity.value,
                    "category": finding.category.value,
                    "path": finding.path,
                    "location": finding.location,
                    "problem": _preview(finding.evidence, 1_000),
                    "solution": finding.remediation,
                    "verification": list(finding.verification),
                    "repair_node_id": finding.repair_node_id,
                    "timestamp": finding.created_at.isoformat(),
                })
            for cycle in self.store.list_quality_cycles(self.run.id):
                if str(cycle.result).casefold() not in {"passed", "complete", "completed", "success"} or cycle.blocker:
                    items.append({
                        "id": cycle.id,
                        "kind": "quality_cycle",
                        "status": cycle.result,
                        "severity": "warning",
                        "category": cycle.kind.value,
                        "path": "",
                        "problem": cycle.blocker or _preview(cycle.outputs),
                        "solution": _preview(cycle.inputs),
                        "verification": _jsonable(cycle.metrics),
                        "timestamp": cycle.started_at.isoformat(),
                    })
        if self.goal is not None:
            for action in self.store.list_actions(self.goal.id):
                status = str(action.get("status") or "")
                if status not in {"failed", "denied", "uncertain"}:
                    continue
                items.append({
                    "id": action.get("id"),
                    "kind": "tool_action",
                    "status": status,
                    "severity": "error" if status in {"failed", "uncertain"} else "warning",
                    "category": action.get("tool_name"),
                    "path": "",
                    "problem": action.get("result_summary") or status,
                    "solution": "Inspect the linked action, retry, or choose a different mechanism.",
                    "verification": [],
                    "timestamp": str(action.get("started_at") or ""),
                })
        return {"items": sorted(items, key=lambda item: str(item.get("timestamp") or ""), reverse=True)}

    def agents(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        agents: list[dict[str, Any]] = []
        scheduled: list[dict[str, Any]] = []
        registry: list[dict[str, Any]] = []
        orchestration_experiments: list[dict[str, Any]] = []
        worker_contributions: list[dict[str, Any]] = []
        model_stats: dict[str, dict[str, Any]] = {}
        if self.goal is not None:
            orchestration_experiments = [
                _jsonable(item)
                for item in self.store.list_orchestration_experiments(
                    goal_id=self.goal.id,
                    limit=2_000,
                )
            ]
            worker_contributions = [
                _jsonable(item)
                for item in self.store.list_worker_contributions(
                    goal_id=self.goal.id,
                    limit=5_000,
                )
            ]
        if self.run is not None:
            nodes = [_jsonable(item) for item in self.store.list_work_nodes(self.run.id)]
            agent_rows = self.store.list_agent_runs(self.run.id)
            agents = [_jsonable(item) for item in agent_rows]
            for agent in agent_rows:
                key = f"{agent.provider}/{agent.model}"
                stats = model_stats.setdefault(key, {
                    "provider": agent.provider,
                    "model": agent.model,
                    "calls": 0,
                    "completed": 0,
                    "failed": 0,
                    "attempts": 0,
                    "tokens": 0,
                })
                stats["calls"] += 1
                stats["attempts"] += int(agent.attempt or 0)
                status = agent.status.value
                if status == "completed":
                    stats["completed"] += 1
                if status in {"failed", "uncertain", "cancelled"}:
                    stats["failed"] += 1
                usage = dict(agent.usage or {})
                stats["tokens"] += sum(
                    int(value or 0)
                    for key_name, value in usage.items()
                    if "token" in str(key_name).casefold() and isinstance(value, (int, float))
                )
            scheduled = []
            for item in self.store.list_scheduled_agent_actions(self.run.id):
                value = dict(item)
                packet = value.pop("packet", {})
                value["packet_preview"] = _preview(packet, 500)
                value["packet_fingerprint"] = hashlib.sha256(
                    json.dumps(packet, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                scheduled.append(_jsonable(value))
            registry = [_jsonable(item) for item in self.store.list_agent_registry(self.run.id)]

        # Older and pre-run checkpoints may have durable lifecycle events before
        # the Ultra run projection exists. They are still authoritative evidence:
        # never hide a waiting/scheduled specialist just because no call started.
        agent_by_id = {
            str(item.get("id") or item.get("agent_run_id") or ""): item
            for item in agents
            if isinstance(item, Mapping)
        }
        scheduled_ids = {
            str(item.get("id") or item.get("action_id") or item.get("agent_id") or "")
            for item in scheduled
            if isinstance(item, Mapping)
        }
        event_agents: dict[str, dict[str, Any]] = {}
        status_for_event = {
            "agent.scheduled": "waiting",
            "agent.waiting": "waiting",
            "agent.started": "running",
            "agent.running": "running",
            "agent.completed": "completed",
            "agent.failed": "failed",
            "agent.cancelled": "cancelled",
        }
        for event in self.events:
            if not (
                str(event.event_type).startswith("agent.")
                or str(event.entity_type or "").casefold() == "agent"
            ):
                continue
            payload = dict(event.payload or {})
            agent_id = str(
                event.entity_id
                or payload.get("agent_run_id")
                or payload.get("agent_id")
                or payload.get("id")
                or event.id
            )
            current = event_agents.setdefault(
                agent_id,
                {
                    "id": agent_id,
                    "name": payload.get("name") or payload.get("role") or agent_id,
                    "role": payload.get("role") or payload.get("name") or "specialist",
                    "status": "waiting",
                    "provider": payload.get("provider") or "",
                    "model": payload.get("model") or "",
                    "phase": payload.get("phase") or "",
                    "attempt": int(payload.get("attempt") or 0),
                    "latency_ms": payload.get("latency_ms"),
                    "usage": payload.get("usage") or {},
                    "source": "durable_event",
                    "scheduled_at": event.created_at.isoformat() if event.created_at else "",
                },
            )
            current.update(
                {
                    key: value
                    for key, value in {
                        "name": payload.get("name") or current.get("name"),
                        "role": payload.get("role") or current.get("role"),
                        "provider": payload.get("provider") or current.get("provider"),
                        "model": payload.get("model") or current.get("model"),
                        "phase": payload.get("phase") or current.get("phase"),
                        "attempt": int(payload.get("attempt") or current.get("attempt") or 0),
                        "latency_ms": payload.get("latency_ms") or current.get("latency_ms"),
                        "usage": payload.get("usage") or current.get("usage") or {},
                    }.items()
                    if value not in (None, "")
                }
            )
            current["status"] = str(
                payload.get("status")
                or status_for_event.get(str(event.event_type), current.get("status") or "waiting")
            )
            current["last_event_sequence"] = event.sequence
            current["last_event_at"] = (
                event.created_at.isoformat() if event.created_at else ""
            )
            if str(event.event_type) in {"agent.scheduled", "agent.waiting"} and agent_id not in scheduled_ids:
                scheduled.append({
                    "id": agent_id,
                    "agent_id": agent_id,
                    "status": "waiting",
                    "name": current["name"],
                    "phase": current.get("phase", ""),
                    "source": "durable_event",
                    "created_at": current["scheduled_at"],
                })
                scheduled_ids.add(agent_id)
        for agent_id, item in event_agents.items():
            if agent_id not in agent_by_id:
                agents.append(item)

        # Include event-only agents in model comparisons without double-counting
        # multiple lifecycle events for the same specialist attempt.
        for item in event_agents.values():
            if str(item.get("id") or "") in agent_by_id:
                continue
            model = str(item.get("model") or "")
            provider = str(item.get("provider") or "")
            if not model:
                continue
            key = f"{provider}/{model}"
            stats = model_stats.setdefault(key, {
                "provider": provider,
                "model": model,
                "calls": 0,
                "completed": 0,
                "failed": 0,
                "attempts": 0,
                "tokens": 0,
            })
            stats["calls"] += 1
            stats["attempts"] += max(1, int(item.get("attempt") or 0))
            status = str(item.get("status") or "")
            if status == "completed":
                stats["completed"] += 1
            if status in {"failed", "uncertain", "cancelled"}:
                stats["failed"] += 1
            usage = dict(item.get("usage") or {})
            stats["tokens"] += sum(
                int(value or 0)
                for key_name, value in usage.items()
                if "token" in str(key_name).casefold() and isinstance(value, (int, float))
            )
        return {
            "nodes": nodes,
            "agents": agents,
            "scheduled": scheduled,
            "registry": registry,
            "orchestration_experiments": orchestration_experiments,
            "worker_contributions": worker_contributions,
            "worker_impact_summary": {
                "useful": sum(
                    str(item.get("outcome")) == "useful"
                    for item in worker_contributions
                ),
                "neutral": sum(
                    str(item.get("outcome")) == "neutral"
                    for item in worker_contributions
                ),
                "harmful": sum(
                    str(item.get("outcome")) == "harmful"
                    for item in worker_contributions
                ),
                "total_tokens": sum(
                    int(item.get("total_tokens") or 0)
                    for item in worker_contributions
                ),
                "total_latency_ms": sum(
                    int(item.get("latency_ms") or 0)
                    for item in worker_contributions
                ),
            },
            "models": sorted(model_stats.values(), key=lambda item: (-item["calls"], item["model"])),
        }

    def prompts(self, *, include_stored_text: bool = False) -> dict[str, Any]:
        plans = []
        traces = []
        if self.goal is not None:
            for plan in self.store.list_plan_revisions(self.goal.id):
                plans.append({
                    "id": plan.id,
                    "revision": plan.revision,
                    "status": plan.status.value,
                    "summary": plan.summary,
                    "fingerprint": plan.fingerprint,
                    "created_at": plan.created_at.isoformat(),
                    "updated_at": plan.updated_at.isoformat(),
                    "tasks": [_jsonable(item) for item in self.store.list_tasks(self.goal.id, plan.revision)],
                })
        if self.run is not None:
            for trace in self.store.list_prompt_traces(self.run.id, limit=1_000):
                item = {
                    "id": trace.id,
                    "role": trace.role,
                    "work_node_id": trace.work_node_id,
                    "agent_run_id": trace.agent_run_id,
                    "created_at": trace.created_at.isoformat(),
                    "redacted": True,
                    "truncated": trace.truncated,
                    "omitted_sections": list(trace.omitted_sections),
                    "reasoning_summary": trace.reasoning_summary,
                    "system_preview": _preview(trace.system_prompt),
                    "context_preview": _preview(trace.context_package),
                    "self_preview": _preview(trace.self_prompt),
                    "stored_chars": {
                        "system": len(trace.system_prompt),
                        "context": len(json.dumps(trace.context_package, ensure_ascii=False, default=str)),
                        "self": len(trace.self_prompt),
                    },
                    "chain_of_thought": "not stored",
                }
                if include_stored_text:
                    item.update({
                        "system_prompt": trace.system_prompt,
                        "context_package": redact_data(trace.context_package),
                        "self_prompt": trace.self_prompt,
                    })
                traces.append(item)
        return {"plans": plans, "traces": traces}

    def reveal_prompt(self, trace_id: str) -> dict[str, Any]:
        if self.run is None:
            raise ValueError("this trace has no Ultra prompt records")
        trace = self.store.get_prompt_trace(str(trace_id))
        if trace.ultra_run_id != self.run.id:
            raise ValueError("prompt trace does not belong to the selected run")
        return {
            "id": trace.id,
            "redacted": True,
            "system_prompt": trace.system_prompt,
            "context_package": redact_data(trace.context_package),
            "self_prompt": trace.self_prompt,
            "reasoning_summary": trace.reasoning_summary,
            "omitted_sections": list(trace.omitted_sections),
            "chain_of_thought": "not stored",
        }

    def context(self) -> dict[str, Any]:
        retrievals: list[dict[str, Any]] = []
        rotations: list[dict[str, Any]] = []
        for event in self.events:
            payload = dict(event.payload or {})
            if event.event_type == "context.repository_retrieval":
                candidates = []
                for item in payload.get("candidates", ()) or ():
                    if not isinstance(item, Mapping):
                        continue
                    candidates.append(asdict(RepositoryContextCandidateV1(
                        query=str(payload.get("query") or ""),
                        path=str(item.get("path") or ""),
                        symbol=str(item.get("name") or ""),
                        rank=int(item.get("rank") or 0),
                        score=float(item.get("score") or 0.0),
                        provenance=tuple(item.get("provenance") or ()),
                        decision=str(item.get("outcome") or "excluded"),
                        reason=str(item.get("reason") or ""),
                        stage=str(payload.get("stage") or ""),
                    )))
                retrievals.append({
                    "id": event.id,
                    "sequence": event.sequence,
                    "stage": payload.get("stage"),
                    "query": payload.get("query"),
                    "selected_count": payload.get("selected_count", 0),
                    "excluded_count": payload.get("excluded_count", 0),
                    "candidates": candidates,
                    "timestamp": event.created_at.isoformat() if event.created_at else "",
                })
            elif event.event_type == "context.rotated":
                rotations.append(asdict(ContextRotationV1(
                    actor=str(payload.get("actor") or "coordinator"),
                    model=str(payload.get("model") or ""),
                    before_chars=int(payload.get("before_chars") or 0),
                    after_chars=int(payload.get("after_chars") or 0),
                    budget_chars=int(payload.get("budget_chars") or 0),
                    suspended_messages=int(payload.get("suspended_messages") or 0),
                    checkpoint_fingerprint=str(payload.get("checkpoint_fingerprint") or ""),
                    reason=str(payload.get("reason") or "context budget reached"),
                    timestamp=event.created_at.isoformat() if event.created_at else "",
                )))
        memory = (
            [_jsonable(item) for item in self.store.list_memory_access(self.run.id, limit=1_000)]
            if self.run is not None
            else []
        )
        omitted = []
        if self.run is not None:
            for trace in self.store.list_prompt_traces(self.run.id, limit=1_000):
                if trace.omitted_sections:
                    omitted.append({
                        "trace_id": trace.id,
                        "agent_run_id": trace.agent_run_id,
                        "work_node_id": trace.work_node_id,
                        "sections": list(trace.omitted_sections),
                        "timestamp": trace.created_at.isoformat(),
                    })
        return {
            "retrievals": retrievals,
            "rotations": rotations,
            "memory_access": memory,
            "omitted_sections": omitted,
        }

    def changes(self) -> dict[str, Any]:
        rows = []
        if self.run is not None:
            for change in self.store.list_change_sets(self.run.id):
                rows.append({
                    "id": change.id,
                    "agent_id": change.responsible_agent_id,
                    "parent_id": change.parent_id,
                    "version": change.version,
                    "status": change.status.value,
                    "changed_files": list(change.changed_files),
                    "diff": redact_text(change.diff, 500_000),
                    "pre_hashes": redact_data(dict(change.pre_hashes)),
                    "post_hashes": redact_data(dict(change.post_hashes)),
                    "mutation_commands": list(change.mutation_commands),
                    "shell_created_files": list(change.shell_created_files),
                    "verification_evidence_ids": list(change.verification_evidence_ids),
                    "review_status": _jsonable(change.review_status),
                    "integration_status": change.integration_status,
                    "created_at": change.created_at.isoformat(),
                    "updated_at": change.updated_at.isoformat(),
                    "metadata": redact_data(_jsonable(change.metadata)),
                })
        return {"items": rows}

    def overview(self) -> dict[str, Any]:
        if self.goal is None:
            runtime_snapshot = self.adapter.runtime.workflow_runtime_snapshot()
            session = self.store.get_workflow_session(self.session_id)
            session_state = dict(session.get("state") or {})
            return {
                "session_id": self.session_id,
                "state": self.state,
                "status": str(runtime_snapshot.phase or runtime_snapshot.liveness or "idle"),
                "provider": getattr(self.adapter.runtime, "provider_name", ""),
                "model": getattr(self.adapter.runtime, "model_name", ""),
                "mode": getattr(getattr(self.adapter.runtime, "interaction_mode", None), "value", "working"),
                "objective": str(
                    session_state.get("session_title")
                    or session_state.get("original_objective")
                    or "Bounded workspace activity"
                ),
                "cutoff_sequence": self.events[-1].sequence if self.events else 0,
                "runs": self.available_runs(),
                "counts": {
                    "events": len(self.events),
                    "files": 0,
                    "problems": 0,
                    "nodes": 0,
                    "agents": 0,
                    "scheduled": 0,
                },
                "inspect_next": [],
            }
        files = self.files()
        problems = self.problems()
        agents = self.agents()
        cutoff = self.events[-1].sequence if self.events else 0
        run_status = self.run.status.value if self.run is not None else self.goal.status.value
        snapshots = self.store.list_advanced_trace_snapshots(self.goal.id)
        return {
            "session_id": self.session_id,
            "goal_id": self.goal.id,
            "run_id": self.run_id,
            "state": self.state,
            "status": run_status,
            "objective": self.goal.objective,
            "provider": getattr(self.run, "provider", getattr(self.adapter.runtime, "provider_name", "")),
            "model": getattr(self.run, "model", getattr(self.adapter.runtime, "model_name", "")),
            "mode": getattr(getattr(self.adapter.runtime, "interaction_mode", None), "value", "working"),
            "created_at": self.goal.created_at.isoformat(),
            "updated_at": self.goal.updated_at.isoformat(),
            "cutoff_sequence": cutoff,
            "counts": {
                "events": len(self.events),
                "files": len(files["items"]),
                "problems": len(problems["items"]),
                "nodes": len(agents["nodes"]),
                "agents": len(agents["agents"]),
                "scheduled": len(agents["scheduled"]),
                "worker_contributions": len(agents["worker_contributions"]),
            },
            "inspect_next": files["inspect_next"],
            "runs": self.available_runs(),
            "snapshots": snapshots,
        }

    def export_payload(self, *, include_stored_text: bool = False) -> dict[str, Any]:
        payload = {
            "schema": "advanced-trace/v1",
            "overview": self.overview(),
            "timeline": self.timeline(limit=10_000)["items"],
            "files": self.files(),
            "problems": self.problems(),
            "agents": self.agents(),
            "prompts": self.prompts(include_stored_text=include_stored_text),
            "context": self.context(),
            "changes": self.changes(),
            "privacy": {
                "stored_text_included": bool(include_stored_text),
                "secrets": "redacted",
                "chain_of_thought": "not stored",
            },
        }
        # Snapshot metadata is transport metadata, not part of its own hash.
        payload["overview"].pop("snapshots", None)
        return payload

    def ensure_frozen_snapshot(self) -> dict[str, Any] | None:
        if self.goal is None or self.state != "FROZEN":
            return None
        payload = self.export_payload(include_stored_text=False)
        cutoff = self.events[-1].sequence if self.events else 0
        return self.store.save_advanced_trace_snapshot(
            session_id=self.session_id,
            goal_id=self.goal.id,
            ultra_run_id=self.run_id,
            terminal_status=self.goal.status.value,
            cutoff_sequence=cutoff,
            payload=payload,
        )

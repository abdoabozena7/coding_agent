"""Focused retrieval facade over the durable ULTRA Project Brain.

The database remains the source of truth.  This class only implements the
context ordering and budget policy used by role agents, so callers never need
to load the whole project history into a model context window.
"""

from __future__ import annotations

from datetime import timedelta
import json
from typing import Any, Mapping

from .durable_memory import NextActionPacketV1
from .models import DomainError, utc_now
from .store import StateStore
from .ultra_models import (
    ArchitectureSpecV1,
    Artifact,
    BrainEntry,
    BrainSection,
    ContextPackageV1,
    GoalSpecV1,
    InsightV1,
    ResultPackageV1,
    WorkNode,
)


def _json_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    )


class ProjectBrain:
    """Read and write one ULTRA run's versioned project memory."""

    def __init__(self, store: StateStore, ultra_run_id: str) -> None:
        self.store = store
        self.ultra_run_id = ultra_run_id
        self.run = store.get_ultra_run(ultra_run_id)

    def write(
        self,
        section: BrainSection | str,
        title: str,
        content: str,
        *,
        data: Mapping[str, Any] | None = None,
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        role: str | None = None,
        ttl_seconds: int | None = None,
        expected_version: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BrainEntry:
        expires_at = None
        if ttl_seconds is not None:
            if ttl_seconds < 1:
                raise ValueError("memory TTL must be positive")
            expires_at = utc_now() + timedelta(seconds=min(ttl_seconds, 31_536_000))
        entry = BrainEntry(
            ultra_run_id=self.ultra_run_id,
            goal_id=self.run.goal_id,
            work_node_id=work_node_id,
            agent_run_id=agent_run_id,
            section=BrainSection(section),
            title=title,
            content=content,
            data=dict(data or {}),
            role=role,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
        )
        stored = self.store.put_brain_entry(entry, expected_version=expected_version)
        self.store.record_memory_access(
            self.ultra_run_id,
            direction="write",
            work_node_id=work_node_id,
            agent_run_id=agent_run_id,
            brain_entry_id=stored.id,
            query=title,
            metadata={"section": stored.section.value, "version": stored.version},
        )
        return stored

    put = write

    def set_north_star(self, spec: GoalSpecV1, *, expected_version: int | None = None) -> BrainEntry:
        return self.write(
            BrainSection.NORTH_STAR,
            "Project North Star",
            spec.objective,
            data=spec.to_dict(),
            expected_version=expected_version,
            metadata={"goal_fingerprint": spec.fingerprint},
        )

    def set_architecture(
        self, spec: ArchitectureSpecV1, *, expected_version: int | None = None
    ) -> BrainEntry:
        return self.write(
            BrainSection.ARCHITECTURE,
            "Current Architecture",
            spec.summary,
            data=spec.to_dict(),
            expected_version=expected_version,
        )

    def record_decision(
        self,
        title: str,
        decision: str,
        *,
        reason: str = "",
        alternatives: tuple[str, ...] = (),
        status: str = "accepted",
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        expected_version: int | None = None,
    ) -> BrainEntry:
        return self.write(
            BrainSection.DECISION,
            title,
            decision,
            data={
                "decision": decision,
                "reason": reason,
                "alternatives": list(alternatives),
                "status": status,
            },
            work_node_id=work_node_id,
            agent_run_id=agent_run_id,
            expected_version=expected_version,
        )

    def remember_for_role(
        self,
        role: str,
        title: str,
        content: str,
        *,
        ttl_seconds: int = 86_400,
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> BrainEntry:
        if not role.strip():
            raise DomainError("role memory requires a role")
        return self.write(
            BrainSection.ROLE_MEMORY,
            title,
            content,
            data=data,
            work_node_id=work_node_id,
            agent_run_id=agent_run_id,
            role=role,
            ttl_seconds=ttl_seconds,
        )

    def record_lesson(
        self,
        title: str,
        content: str,
        *,
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> BrainEntry:
        return self.write(
            BrainSection.LESSON,
            title,
            content,
            data=data,
            work_node_id=work_node_id,
            agent_run_id=agent_run_id,
        )

    def record_knowledge(
        self,
        title: str,
        content: str,
        *,
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        data: Mapping[str, Any] | None = None,
        confidence: float = 0.7,
        evidence_refs: tuple[str, ...] = (),
        promote: bool = True,
    ) -> BrainEntry:
        entry = self.write(
            BrainSection.KNOWLEDGE,
            title,
            content,
            data=data,
            work_node_id=work_node_id,
            agent_run_id=agent_run_id,
        )
        if promote:
            self.store.promote_brain_entry_to_project_memory(
                entry.id,
                confidence=confidence,
                evidence_refs=evidence_refs,
                metadata={"source": "record_knowledge", "promoted": True},
            )
        return entry

    def list(
        self,
        *,
        section: BrainSection | str | None = None,
        role: str | None = None,
        latest_only: bool = True,
        limit: int = 1_000,
    ) -> tuple[BrainEntry, ...]:
        return self.store.list_brain_entries(
            self.ultra_run_id,
            section=section,
            role=role,
            latest_only=latest_only,
            limit=limit,
        )

    def search(
        self,
        query: str,
        *,
        section: BrainSection | str | None = None,
        role: str | None = None,
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        limit: int = 20,
    ) -> tuple[BrainEntry, ...]:
        entries = self.store.search_brain(
            self.ultra_run_id, query, section=section, role=role, limit=limit
        )
        for position, entry in enumerate(entries):
            self.store.record_memory_access(
                self.ultra_run_id,
                direction="read",
                query=query,
                work_node_id=work_node_id,
                agent_run_id=agent_run_id,
                brain_entry_id=entry.id,
                score=float(position),
                metadata={"section": entry.section.value},
            )
        return entries

    @staticmethod
    def _node_payload(node: WorkNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "title": node.title,
            "objective": node.objective,
            "kind": node.kind.value,
            "status": node.status.value,
            "master_task_id": node.master_task_id,
            "depends_on": list(node.depends_on),
            "contract": node.contract.to_dict(),
            "checkpoint": node.checkpoint,
        }

    @staticmethod
    def _entry_payload(entry: BrainEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "section": entry.section.value,
            "title": entry.title,
            "content": entry.content,
            "data": dict(entry.data),
            "version": entry.version,
            "updated_at": entry.updated_at.isoformat(),
        }

    @staticmethod
    def _artifact_payload(artifact: Artifact) -> dict[str, Any]:
        return {
            "id": artifact.id,
            "kind": artifact.kind,
            "uri": artifact.uri,
            "path": artifact.path,
            "content_hash": artifact.content_hash,
            "pre_write_hash": artifact.pre_write_hash,
            "evidence": dict(artifact.evidence),
        }

    @staticmethod
    def _project_memory_payload(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "section": item["section"],
            "title": item["title"],
            "content": item["content"],
            "confidence": item["confidence"],
            "effective_confidence": item.get("effective_confidence", item["confidence"]),
            "evidence_refs": item["evidence_refs"],
            "reuse_count": item["reuse_count"],
        }

    def build_context(
        self,
        work_node_id: str,
        role: str,
        *,
        agent_run_id: str | None = None,
        query: str | None = None,
        budget_chars: int = 40_000,
    ) -> ContextPackageV1:
        """Build the ordered, focused context package defined by ULTRA v1."""
        node = self.store.get_work_node(work_node_id)
        if node.ultra_run_id != self.ultra_run_id:
            raise DomainError("context node belongs to another ULTRA run")
        budget_chars = max(1_000, min(int(budget_chars), 1_000_000))
        ancestors = self.store.work_node_ancestors(node.id)
        architecture = self.store.list_brain_entries(
            self.ultra_run_id, section=BrainSection.ARCHITECTURE, limit=10
        )
        decisions = self.store.list_brain_entries(
            self.ultra_run_id, section=BrainSection.DECISION, limit=50
        )
        constraints = self.store.list_brain_entries(
            self.ultra_run_id, section=BrainSection.CONSTRAINT, limit=30
        )
        dependency_artifacts: list[Artifact] = []
        for dependency in node.depends_on:
            dependency_artifacts.extend(
                self.store.list_artifacts(self.ultra_run_id, work_node_id=dependency, limit=100)
            )
        related_artifacts = self.store.list_artifacts(
            self.ultra_run_id, work_node_id=node.id, limit=100
        )
        role_memory = self.store.list_brain_entries(
            self.ultra_run_id,
            section=BrainSection.ROLE_MEMORY,
            role=role,
            work_node_id=None,
            limit=30,
        )
        focus_query = (query or node.objective).strip()
        knowledge = self.store.search_brain(
            self.ultra_run_id, focus_query, section=BrainSection.KNOWLEDGE, limit=20
        )
        lessons = self.store.search_brain(
            self.ultra_run_id, focus_query, section=BrainSection.LESSON, limit=20
        )
        project_lessons = self.store.search_project_memory(
            focus_query,
            section=BrainSection.LESSON,
            min_confidence=0.4,
            limit=12,
        )
        project_knowledge = self.store.search_project_memory(
            focus_query,
            section=BrainSection.KNOWLEDGE,
            min_confidence=0.4,
            limit=12,
        )
        for memory in (*project_lessons, *project_knowledge):
            self.store.record_project_memory_use(str(memory["id"]))
        previous_snapshot = self.store.latest_agent_memory_snapshot(
            self.ultra_run_id,
            work_node_id=node.id,
            role=role,
        )
        next_action = NextActionPacketV1(
            ultra_run_id=self.ultra_run_id,
            work_node_id=node.id,
            role=role,
            phase=node.checkpoint or node.assigned_role or "execute",
            objective=node.objective[:3_000],
            contract={
                "success_criteria": list(node.contract.success_criteria),
                "write_paths": list(node.contract.write_paths),
                "read_paths": list(node.contract.read_paths),
                "forbidden_changes": list(node.contract.forbidden_changes),
                "interfaces": dict(node.contract.interfaces),
            },
            checkpoint={
                "node_status": node.status.value,
                "checkpoint": node.checkpoint,
                "attempts": node.attempts,
                "previous_memory_revision": (
                    previous_snapshot.revision if previous_snapshot else 0
                ),
            },
            dependency_evidence=tuple(
                {
                    "id": item.id,
                    "kind": item.kind,
                    "path": item.path,
                    "content_hash": item.content_hash,
                }
                for item in dependency_artifacts[:12]
            ),
            relevant_memory=tuple(
                {
                    **self._project_memory_payload(item),
                    "content": str(item.get("content", ""))[:600],
                    "evidence_refs": list(item.get("evidence_refs", ()))[:4],
                }
                for item in (*project_lessons[:4], *project_knowledge[:4])
            ),
            required_outputs=tuple(node.contract.success_criteria),
            context_budget_chars=max(2_000, min(budget_chars, 120_000)),
        )
        candidates: list[tuple[str, Any, tuple[BrainEntry, ...]]] = [
            ("next_action_packet", next_action.to_dict(), ()),
            ("task", self._node_payload(node), ()),
            (
                "dependency_artifacts",
                [self._artifact_payload(item) for item in dependency_artifacts],
                (),
            ),
            (
                "previous_agent_memory",
                previous_snapshot.to_dict() if previous_snapshot else {},
                (),
            ),
            (
                "project_lessons",
                [self._project_memory_payload(item) for item in project_lessons],
                (),
            ),
            (
                "project_knowledge",
                [self._project_memory_payload(item) for item in project_knowledge],
                (),
            ),
            ("role_memory", [self._entry_payload(item) for item in role_memory], role_memory),
            ("knowledge", [self._entry_payload(item) for item in knowledge], knowledge),
            ("lessons", [self._entry_payload(item) for item in lessons], lessons),
            ("architecture", [self._entry_payload(item) for item in architecture], architecture),
            ("constraints", [self._entry_payload(item) for item in constraints], constraints),
            ("decisions", [self._entry_payload(item) for item in decisions], decisions),
            ("ancestor_contracts", [self._node_payload(item) for item in ancestors], ()),
            ("related_artifacts", [self._artifact_payload(item) for item in related_artifacts], ()),
        ]
        sections: dict[str, Any] = {}
        omitted: list[str] = []
        used = 0
        used_entries: list[BrainEntry] = []
        for name, value, entries in candidates:
            size = _json_size({name: value})
            if used + size <= budget_chars:
                sections[name] = value
                used += size
                used_entries.extend(entries)
            else:
                omitted.append(name)
        for entry in used_entries:
            self.store.record_memory_access(
                self.ultra_run_id,
                direction="read",
                query=focus_query,
                work_node_id=node.id,
                agent_run_id=agent_run_id,
                brain_entry_id=entry.id,
                metadata={"context_section": entry.section.value, "role": role},
            )
        return ContextPackageV1(
            ultra_run_id=self.ultra_run_id,
            work_node_id=node.id,
            role=role,
            sections=sections,
            omitted_sections=tuple(omitted),
            size_chars=used,
        )

    retrieve_context = build_context

    def write_back_result(
        self,
        work_node_id: str,
        result: ResultPackageV1,
        *,
        agent_run_id: str | None = None,
    ) -> tuple[BrainEntry, ...]:
        """Persist lessons and structured insights after a node checkpoint."""
        entries: list[BrainEntry] = []
        result_entry = self.record_lesson(
            f"Result: {work_node_id}",
            result.summary,
            work_node_id=work_node_id,
            agent_run_id=agent_run_id,
            data=result.to_dict(),
        )
        entries.append(result_entry)
        result_success = not result.issues and all(
            bool(item.get("passed", True)) for item in result.tests if isinstance(item, Mapping)
        )
        evidence_refs = tuple(
            str(item.get("id") or item.get("path") or item.get("uri") or item)
            if isinstance(item, Mapping)
            else str(item)
            for item in (*result.artifacts, *result.tests)
            if item
        )
        self.store.promote_brain_entry_to_project_memory(
            result_entry.id,
            confidence=0.85 if result_success else 0.55,
            evidence_refs=evidence_refs,
            metadata={"source": "write_back_result", "success": result_success},
        )
        for insight in result.insights:
            insight_entry = self.record_lesson(
                insight.summary,
                insight.details or insight.summary,
                work_node_id=work_node_id,
                agent_run_id=agent_run_id,
                data=insight.to_dict(),
            )
            entries.append(insight_entry)
            severity = str(insight.severity).casefold()
            confidence = 0.75 if severity in {"info", "warning"} else 0.6
            self.store.promote_brain_entry_to_project_memory(
                insight_entry.id,
                confidence=confidence,
                evidence_refs=evidence_refs,
                metadata={"source": "insight_writeback", "severity": insight.severity},
            )
        return tuple(entries)

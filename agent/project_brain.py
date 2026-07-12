"""Focused retrieval facade over the durable ULTRA Project Brain.

The database remains the source of truth.  This class only implements the
context ordering and budget policy used by role agents, so callers never need
to load the whole project history into a model context window.
"""

from __future__ import annotations

from datetime import timedelta
import json
from typing import Any, Mapping

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
        candidates: list[tuple[str, Any, tuple[BrainEntry, ...]]] = [
            ("task", self._node_payload(node), ()),
            ("ancestor_contracts", [self._node_payload(item) for item in ancestors], ()),
            ("architecture", [self._entry_payload(item) for item in architecture], architecture),
            ("decisions", [self._entry_payload(item) for item in decisions], decisions),
            ("constraints", [self._entry_payload(item) for item in constraints], constraints),
            (
                "dependency_artifacts",
                [self._artifact_payload(item) for item in dependency_artifacts],
                (),
            ),
            ("related_artifacts", [self._artifact_payload(item) for item in related_artifacts], ()),
            ("role_memory", [self._entry_payload(item) for item in role_memory], role_memory),
            ("knowledge", [self._entry_payload(item) for item in knowledge], knowledge),
            ("lessons", [self._entry_payload(item) for item in lessons], lessons),
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
        entries.append(
            self.record_lesson(
                f"Result: {work_node_id}",
                result.summary,
                work_node_id=work_node_id,
                agent_run_id=agent_run_id,
                data=result.to_dict(),
            )
        )
        for insight in result.insights:
            entries.append(
                self.record_lesson(
                    insight.summary,
                    insight.details or insight.summary,
                    work_node_id=work_node_id,
                    agent_run_id=agent_run_id,
                    data=insight.to_dict(),
                )
            )
        return tuple(entries)

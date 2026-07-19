"""Runnable integration between ULTRA orchestration and the v3 state store.

The provider-neutral engine is intentionally independent from the legacy
goal/plan runtime.  This module supplies the concrete adapters needed by the
CLI: real workspace tools, Docker-only Full shell access, durable v3 records,
legacy master-plan approval, file hashes, and resource leases.
"""

from __future__ import annotations

import fnmatch
import difflib
import hashlib
import json
import re
import shlex
import threading
from concurrent.futures import Future
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import tools
from .component_artifacts import (
    ComponentArtifactError,
    ComponentArtifactStore,
    MaterializedComponentPackageV2,
)
from .events import EventBus
from .evaluation import (
    learn_from_benchmark_trend,
    record_benchmark_trend,
    record_single_file_3d_html_benchmark,
    run_single_file_3d_html_benchmark,
)
from .model_catalog import ExecutionClass, ModelDescriptor
from .local_provider import (
    extract_first_json_object,
    normalize_action_proposal,
    normalize_generated_tool_args,
)
from .learning import GlobalLessonStore, LearnedLessonV1
from .models import DomainError, GoalStatus, Plan, PlanStatus, RoleProfile, TaskStatus, utc_now
from .providers.base import AssistantTurn, ToolCall
from .project_brain import ProjectBrain
from .safety import redact_data, redact_text
from .sandbox import AccessLevel, PermissionAdapter
from .scheduler import ResourceLease as RuntimeLease
from .scheduler import AdaptiveConcurrency, RateLimitError, ResourceLeaseManager, StaleWriteError
from .store import NotFoundError, StateStore, StateStoreError
from .swarm_coordinator import SwarmCoordinator
from .swarm_protocol import SwarmMessageType, SwarmMessageV1
from .ultra import (
    AgentRequest,
    AgentResponse,
    AgentRole,
    ArchitectureSpecV1 as EngineArchitectureSpec,
    BrainEntryV1,
    BrainSection as EngineBrainSection,
    ComponentPackageV1,
    GoalSpecV1 as EngineGoalSpec,
    ContextRequest,
    FocusedContextBuilder,
    InMemoryUltraState,
    InnerPhase,
    MasterPlanV1,
    NodeKind,
    NodeQualityTargetV1,
    NodeStatus,
    PromptTraceV1 as EnginePromptTrace,
    ResultPackageV1 as EngineResult,
    SpecialistProfileV1,
    UltraConfig,
    UltraOrchestrator,
    UltraPhase as EnginePhase,
    UltraRunResult,
    UltraRunV1,
    WorkNode as EngineWorkNode,
    _extract_json,
)
from .ultra_models import (
    AgentRun,
    AgentRunStatus,
    ArchitectureSpecV1,
    Artifact,
    BrainEntry,
    BrainSection,
    GoalSpecV1,
    InsightV1,
    PromptTraceV1,
    ResultPackageV1,
    TaskContractV1,
    UltraPhase,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    WorkNodeKind,
    WorkNodeStatus,
)
from .visual_judge import (
    VisualJudgeUnavailable,
    create_visual_judge,
    require_two_clean_acceptances,
    screenshot_anomalies,
)
from .workflow import AgentRegistryEntryV1, AgentState
from .quality import (
    ChangeSetStatus,
    ChangeSetV1,
    FindingSeverity,
    QualityCategory,
    QualityCycleKind,
    QualityCycleV1,
    QualityFindingV1,
    QualityPolicyV1,
)
from .reasoning import (
    evaluate_reasoning_artifact,
    repair_reasoning_artifact_graph,
    reasoning_debate_protocol_for,
    reasoning_scaffold_for,
)


_READ_TOOLS = tools.names(categories={"read"})
_WRITE_TOOLS = tools.names(categories={"write", "command", "install"})
_TOOL_RISK = tools.risk_map()
_STAGE_COMPONENT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "stage_component_file",
        "description": (
            "Stage exactly one real component file in harness-owned isolation. "
            "Call once per implementation, test, preview, or asset file before publishing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "role": {
                    "type": "string",
                    "enum": ["implementation", "preview", "test", "asset"],
                },
            },
            "required": ["path", "content", "role"],
        },
    },
}
_PUBLISH_COMPONENT_TOOL = {
    "type": "function",
    "function": {
        "name": "publish_component",
        "description": (
            "Finalize previously staged component files with an interface and preview manifest. "
            "This is the only valid completion path for component specialists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "interface": {
                    "type": "object",
                    "properties": {
                        "exports": {"type": "array", "items": {"type": "string"}},
                        "imports": {"type": "array", "items": {"type": "string"}},
                        "invariants": {"type": "array", "items": {"type": "string"}},
                        "integration_points": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["exports"],
                },
                "preview": {
                    "type": "object",
                    "properties": {"entrypoint": {"type": "string"}},
                    "required": ["entrypoint"],
                },
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "object"}},
                "quality": {"type": "object"},
            },
            "required": ["interface", "preview"],
        },
    },
}


def _schema_name(schema: Mapping[str, Any]) -> str:
    return str(schema.get("function", {}).get("name", ""))


def _schemas(names: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(names)
    return [schema for schema in tools.TOOL_SCHEMAS if _schema_name(schema) in wanted]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _hash_file(workspace: Path, value: str) -> str | None:
    parts = PurePosixPath(_normalized_path(value)).parts
    candidate = workspace.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _normalized_path(value: str) -> str:
    text = str(value or ".").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return str(PurePosixPath(text or ".")).rstrip("/") or "."


def _within_scope(path: str, scopes: Iterable[str]) -> bool:
    candidate = _normalized_path(path).casefold()
    for raw in scopes:
        scope = _normalized_path(raw).casefold()
        if scope in {".", "*", "**", "**/*"}:
            return True
        if any(character in scope for character in "*?["):
            if fnmatch.fnmatchcase(candidate, scope):
                return True
            continue
        if candidate == scope or candidate.startswith(scope + "/"):
            return True
    return False


def _snapshot_write_scopes(
    workspace: Path,
    scopes: Iterable[str],
    *,
    max_files: int = 50_000,
) -> dict[str, str | None]:
    """Capture file hashes protected by a node's write lease."""

    root = workspace.resolve()
    snapshot: dict[str, str | None] = {}
    seen: set[Path] = set()

    def remember(candidate: Path) -> None:
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            return
        if not resolved.is_file() or ".coding-agent" in relative.parts:
            return
        if resolved in seen:
            return
        if len(seen) >= max_files:
            raise RuntimeError(
                "write-scope snapshot exceeded 50000 files; narrow the module write_paths"
            )
        seen.add(resolved)
        key = _normalized_path(relative.as_posix())
        snapshot[key] = _hash_file(root, key)

    for raw_scope in scopes:
        scope = _normalized_path(raw_scope)
        if scope in {".", "*", "**", "**/*"}:
            for candidate in root.rglob("*"):
                remember(candidate)
            continue
        if any(character in scope for character in "*?["):
            for candidate in root.glob(scope):
                if candidate.is_dir():
                    for child in candidate.rglob("*"):
                        remember(child)
                else:
                    remember(candidate)
            continue
        candidate = root.joinpath(*PurePosixPath(scope).parts)
        if candidate.is_dir():
            for child in candidate.rglob("*"):
                remember(child)
        elif candidate.is_file():
            remember(candidate)
        else:
            snapshot.setdefault(scope, None)
    return snapshot


_PHASE_CONTRACTS: dict[str, Mapping[str, Any]] = {
    "goal_spec": {
        "payload": {
            "objective": "string",
            "success_criteria": ["observable criterion"],
            "constraints": ["constraint"],
            "in_scope": ["scope item"],
            "out_of_scope": ["non-goal"],
            "assumptions": ["assumption"],
            "questions": [
                {
                    "id": "stable id",
                    "header": "short label",
                    "question": "non-discoverable decision only",
                    "options": [
                        {"label": "choice", "description": "impact", "recommended": True}
                    ],
                    "allow_freeform": True,
                    "reason": "why this cannot be discovered",
                }
            ],
        }
    },
    "architecture": {
        "payload": {
            "summary": "architecture summary",
            "components": [{"name": "component", "responsibility": "..."}],
            "interfaces": [{"name": "interface", "contract": "..."}],
            "decisions": [{"decision": "...", "reason": "...", "alternatives": []}],
            "dependencies": [],
            "invariants": [],
        }
    },
    "master_plan": {
        "payload": {
            "summary": "complete master plan",
            "execution_strategy": "dependency waves and integration strategy",
            "milestones": [],
            "modules": [
                {
                    "id": "M001",
                    "title": "module title",
                    "objective": "bounded objective",
                    "acceptance_criteria": ["observable criterion"],
                    "verification": ["command or inspection"],
                    "depends_on": [],
                    "write_paths": ["workspace/relative/path"],
                    "forbidden_changes": [],
                    "owned_interfaces": [],
                    "metadata": {"external_dependencies": []},
                }
            ],
        }
    },
    InnerPhase.MINI_PLAN.value: {
        "payload": {"steps": ["step"], "research_required": False}
    },
    InnerPhase.DECOMPOSE.value: {
        "payload": {
            "children": [],
            "research_required": False,
        }
    },
    InnerPhase.REVIEW.value: {
        "payload": {"passed": True, "issues": [], "findings": [], "evidence": []}
    },
    InnerPhase.TEST.value: {
        "payload": {"passed": True, "issues": [], "test_results": [], "evidence": []}
    },
    InnerPhase.IMPLEMENT.value: {
        "payload": {"success": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.FIX.value: {
        "payload": {"success": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.INTEGRATE.value: {
        "payload": {"passed": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.GLOBAL_INTEGRATION.value: {
        "payload": {"passed": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.GLOBAL_REVIEW.value: {
        "payload": {"passed": True, "issues": [], "findings": [], "evidence": []}
    },
    InnerPhase.FINAL_EVIDENCE.value: {
        "payload": {"passed": True, "evidence": [], "test_results": [], "findings": []}
    },
}


class WorkspaceUltraAgent:
    """One isolated role conversation with real, policy-controlled tools."""

    def __init__(
        self,
        provider: Any,
        *,
        role: AgentRole,
        provider_name: str,
        model: str,
        executor: Callable[[ToolCall, AgentRequest], str],
        events: EventBus,
        max_steps: int = 16,
    ) -> None:
        self.provider = provider
        self.role = role
        self.provider_name = provider_name
        self.model = model
        self.executor = executor
        self.events = events
        self.max_steps = max(2, int(max_steps))

    def _allowed_tools(self) -> frozenset[str]:
        if self.role in {AgentRole.CODER, AgentRole.INTEGRATOR}:
            return _READ_TOOLS | _WRITE_TOOLS | {
                "stage_component_file",
                "publish_component",
            }
        if self.role in {AgentRole.TESTER, AgentRole.RESEARCHER}:
            return _READ_TOOLS | {"run_bash", "preview_html"}
        return _READ_TOOLS

    @staticmethod
    def _html_write_target(request: AgentRequest) -> str | None:
        contract = dict(request.task.get("contract", {})) if isinstance(request.task, Mapping) else {}
        for path in contract.get("write_paths", ()) or ():
            text = str(path).strip()
            if text.casefold().endswith((".html", ".htm")):
                return text
        return None

    @staticmethod
    def _is_full_html_quality_gate(request: AgentRequest) -> bool:
        contract = dict(request.task.get("contract", {})) if isinstance(request.task, Mapping) else {}
        text = " ".join(
            (
                str(contract.get("title", "")),
                str(contract.get("objective", "")),
                " ".join(str(item) for item in contract.get("acceptance_criteria", ()) or ()),
            )
        ).casefold()
        return any(
            marker in text
            for marker in ("browser qa", "visual refinement gate", "screenshot-based visual quality")
        )

    def _harness_html_preview(self, request: AgentRequest) -> dict[str, Any] | None:
        if request.phase != InnerPhase.TEST.value or self.role is not AgentRole.TESTER:
            return None
        target = self._html_write_target(request)
        if not target:
            return None
        result = str(
            self.executor(
                ToolCall(
                    "harness-html-preview",
                    "preview_html",
                    {"path": target, "open_browser": False, "verify": True, "settle_ms": 750},
                ),
                request,
            )
        )
        if result.startswith("Error:"):
            return {
                "status": "failed",
                "error": result,
                "console_errors": [result],
                "page_errors": [],
                "network_errors": [],
            }
        try:
            payload = json.loads(result)
        except Exception:
            return {
                "status": "failed",
                "error": f"preview_html returned malformed JSON: {redact_text(result, 500)}",
                "console_errors": [],
                "page_errors": [],
                "network_errors": [],
            }
        if not isinstance(payload, Mapping):
            return {"status": "failed", "error": "preview_html returned non-object payload"}
        evidence = dict(payload)
        html_result = str(
            self.executor(
                ToolCall("harness-html-readback", "read_file", {"path": target}),
                request,
            )
        )
        if html_result.startswith("Error:"):
            evidence["verification"] = "failed"
            evidence["page_errors"] = [
                *list(evidence.get("page_errors", ()) or ()),
                html_result,
            ]
            return evidence
        if not self._is_full_html_quality_gate(request):
            return evidence
        benchmark = run_single_file_3d_html_benchmark(html_result, preview=evidence)
        evidence["benchmark_scores"] = dict(benchmark.scores)
        evidence["benchmark_metrics"] = dict(benchmark.metrics)
        evidence["benchmark_findings"] = list(benchmark.findings)
        if benchmark.findings:
            evidence["verification"] = "failed"
            evidence["page_errors"] = [
                *list(evidence.get("page_errors", ()) or ()),
                *(f"HTML quality benchmark: {item}" for item in benchmark.findings),
            ]
        return evidence

    def execute(self, request: AgentRequest) -> AgentResponse:
        configured_effort = str(getattr(self.provider, "reasoning_effort", "medium"))
        deterministic_roles = {
            AgentRole.GOAL_UNDERSTANDING,
            AgentRole.ARCHITECT,
            AgentRole.PLANNER,
            AgentRole.DECOMPOSER,
            AgentRole.MEMORY,
            AgentRole.QUALITY_TRIAGER,
        }
        effective_effort = configured_effort
        if self.provider_name == "ollama" and configured_effort == "low" and self.role in deterministic_roles:
            effective_effort = "off"
            setattr(self.provider, "reasoning_effort", effective_effort)
        deterministic_budgets = {
            AgentRole.GOAL_UNDERSTANDING: 4096,
            AgentRole.ARCHITECT: 3072,
            AgentRole.PLANNER: 4096,
            AgentRole.DECOMPOSER: 4096,
            AgentRole.MEMORY: 768,
            AgentRole.QUALITY_TRIAGER: 1024,
        }
        if self.provider_name == "ollama" and self.role in deterministic_budgets:
            setattr(self.provider, "max_output_tokens", deterministic_budgets[self.role])
        if self.provider_name == "ollama":
            # JSON mode helps schema-only roles, but constraining tool-using
            # coders/integrators to JSON makes Ollama embed large source files
            # in a JSON string instead of issuing write tools, often producing
            # invalid escaping before any workspace mutation.
            setattr(self.provider, "force_json", self.role in deterministic_roles)
        self.events.publish(
            "ultra.reasoning_routed",
            f"[{self.role.value}] reasoning {configured_effort} -> {effective_effort}",
            run_id=request.run_id,
            node_id=request.node_id,
            role=self.role.value,
            phase=request.phase,
            configured=configured_effort,
            effective=effective_effort,
            max_output_tokens=getattr(self.provider, "max_output_tokens", None),
        )
        contract = _PHASE_CONTRACTS.get(
            request.phase,
            {"payload": {"success": True, "findings": [], "evidence": []}},
        )
        if self.role is AgentRole.CODER and request.phase in {
            InnerPhase.IMPLEMENT.value,
            InnerPhase.FIX.value,
        }:
            contract = {
                "payload": {
                    **dict(contract.get("payload", {})),
                    "proposed_write": {
                        "path": "one exact approved write_path",
                        "content": "complete replacement content; use only when a native write tool cannot be emitted",
                    },
                }
            }
        inspection_observed = False
        mutation_observed = False
        component_publication_passed = False
        harness_inspection: str | None = None
        harness_preview = self._harness_html_preview(request)
        if (
            harness_preview
            and request.phase == InnerPhase.TEST.value
            and self.role is AgentRole.TESTER
            and str(
                harness_preview.get("verification") or harness_preview.get("status") or ""
            ).casefold()
            not in {"passed", "ok", "success"}
        ):
            browser_details = [
                *[str(item) for item in harness_preview.get("benchmark_findings", ())],
                *[str(item) for item in harness_preview.get("console_errors", ())],
                *[str(item) for item in harness_preview.get("page_errors", ())],
                *[str(item) for item in harness_preview.get("network_errors", ())],
            ]
            findings = list(
                dict.fromkeys(
                    ["Harness browser verification failed for HTML output.", *browser_details]
                )
            )
            self.events.publish(
                "ultra.deterministic_test_gate",
                "Harness browser/readback benchmark failed; routing evidence to the fix loop",
                run_id=request.run_id,
                node_id=request.node_id,
                findings=findings,
                scores=dict(harness_preview.get("benchmark_scores", {})),
            )
            return AgentResponse.from_mapping(
                {
                    "payload": {
                        "passed": False,
                        "issues": findings,
                        "findings": findings,
                        "test_results": [
                            {
                                "name": "harness_html_browser_and_quality_gate",
                                "passed": False,
                                "scores": dict(harness_preview.get("benchmark_scores", {})),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                            }
                        ],
                        "evidence": [
                            {
                                "kind": "browser_preview",
                                "verification": harness_preview.get("verification"),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                            }
                        ],
                    },
                    "summary": "Harness browser/readback quality gate failed",
                    "reasoning_summary": "Observable browser and static benchmark evidence requires remediation.",
                },
                node_id=request.node_id,
                provider=self.provider_name,
                model=self.model,
            )
        debate_protocol = reasoning_debate_protocol_for(
            self.role.value,
            request.phase,
            request.task,
        )
        if request.phase == "goal_spec":
            inspection_call = ToolCall(
                "harness-goal-inspection",
                "list_files",
                {"path": "."},
            )
            harness_inspection = str(self.executor(inspection_call, request))
            if harness_inspection.startswith("Error:"):
                raise RuntimeError(
                    "GoalSpecV1 requires a successful harness workspace inspection: "
                    + harness_inspection
                )
            inspection_observed = True
            listed_paths = {
                line.strip()
                for line in harness_inspection.splitlines()
                if line.strip() and not line.strip().startswith("(")
            }
            if "index.html" in listed_paths:
                current_html = str(
                    self.executor(
                        ToolCall(
                            "harness-goal-index-readback",
                            "read_file",
                            {"path": "index.html"},
                        ),
                        request,
                    )
                )
                if not current_html.startswith("Error:"):
                    harness_inspection += (
                        "\n\nAUTHORITATIVE CURRENT index.html READBACK:\n"
                        + current_html[:40_000]
                    )
        write_target_state: list[dict[str, Any]] = []
        if self.role is AgentRole.CODER and request.phase in {
            InnerPhase.IMPLEMENT.value,
            InnerPhase.FIX.value,
        }:
            contract_payload = (
                dict(request.task.get("contract", {}))
                if isinstance(request.task, Mapping)
                else {}
            )
            for index, path in enumerate(contract_payload.get("write_paths", ()) or (), start=1):
                target = str(path).strip()
                if not target:
                    continue
                readback = str(
                    self.executor(
                        ToolCall(f"harness-write-target-{index}", "read_file", {"path": target}),
                        request,
                    )
                )
                write_target_state.append(
                    {
                        "path": target,
                        "exists": not readback.startswith("Error:"),
                        "content": readback[:40_000] if not readback.startswith("Error:") else "",
                        "read_error": readback[:500] if readback.startswith("Error:") else "",
                    }
                )
        request_contract = (
            dict(request.task.get("contract", {}))
            if isinstance(request.task, Mapping)
            else {}
        )
        request_component_only = bool(
            dict(request_contract.get("metadata", {})).get("component_package_only")
        )
        component_publication_phase = bool(
            request_component_only
            and self.role in {AgentRole.CODER, AgentRole.INTEGRATOR}
            and request.phase
            in {InnerPhase.IMPLEMENT.value, InnerPhase.FIX.value, InnerPhase.INTEGRATE.value}
        )
        if component_publication_phase:
            context_mapping = (
                dict(request.context) if isinstance(request.context, Mapping) else {}
            )
            north_star = (
                dict(context_mapping.get("north_star", {}))
                if isinstance(context_mapping.get("north_star"), Mapping)
                else {}
            )
            contract_metadata = (
                dict(request_contract.get("metadata", {}))
                if isinstance(request_contract.get("metadata"), Mapping)
                else {}
            )
            compact_contract = {
                key: request_contract.get(key)
                for key in (
                    "id",
                    "title",
                    "objective",
                    "acceptance_criteria",
                    "verification",
                    "owned_interfaces",
                )
                if request_contract.get(key) not in (None, "", (), [], {})
            }
            compact_contract["metadata"] = {
                key: contract_metadata.get(key)
                for key in (
                    "specialist_domain",
                    "owned_interfaces",
                    "component_leaf",
                    "component_assembler",
                )
                if contract_metadata.get(key) not in (None, "", (), [], {})
            }
            compact_context = {
                "north_star": {
                    key: north_star.get(key)
                    for key in ("objective", "success_criteria", "constraints")
                    if north_star.get(key)
                },
            }
            compact_task = {
                key: request.task.get(key)
                for key in (
                    "findings",
                    "attempt",
                    "change_approach",
                    "component_assembler",
                    "child_component_packages",
                    "prior_replan_guidance",
                    "prior_findings",
                )
                if request.task.get(key) not in (None, "", (), [], {})
            }
            compact_task["contract"] = compact_contract
            user_payload = {
                "component_task": compact_task,
                "integration_context": compact_context,
                "required_action": (
                    "Call publish_component with complete real files. If it rejects the "
                    "candidate, revise this component and call it again."
                ),
                "response_after_accepted_publication": {
                    "payload": {"success": True, "findings": [], "evidence": []},
                    "summary": "brief publication result",
                    "reasoning_summary": "brief evidence-based conclusion",
                },
            }
        else:
            user_payload = {
                "task": request.task,
                "focused_context": request.context,
                "harness_workspace_inspection": harness_inspection,
                "harness_html_preview": harness_preview,
                "harness_write_target_state": write_target_state,
                "harness_reasoning_scaffold": reasoning_scaffold_for(
                    self.role.value,
                    request.phase,
                    request.task,
                ).to_dict(),
                "harness_debate_protocol": debate_protocol.to_dict(),
                "response_contract": {
                    **contract,
                    "summary": "brief factual result summary",
                    "reasoning_summary": (
                        "brief conclusion, decisions, and evidence only; never hidden chain-of-thought"
                    ),
                    "reasoning_artifact": {
                        "claim": "short external claim being made",
                        "supporting_evidence": ["observable/tool/hash/browser/test evidence"],
                        "counterarguments": ["short objection or likely failure mode"],
                        "rejected_alternatives": ["alternative considered and why rejected"],
                        "verification_plan": ["concrete verification still required or already run"],
                        "reasoning_graph": {
                            "nodes": [
                                {
                                    "id": "chosen",
                                    "type": "decision",
                                    "summary": "chosen external decision",
                                    "status": "chosen",
                                    "evidence_refs": ["tool/test/hash/browser evidence"],
                                },
                                {
                                    "id": "rejected",
                                    "type": "option",
                                    "summary": "rejected alternative",
                                    "status": "rejected",
                                    "evidence_refs": [],
                                },
                            ],
                            "edges": [
                                {
                                    "from": "chosen",
                                    "to": "rejected",
                                    "relation": "rejects",
                                }
                            ],
                        },
                    },
                    "insights": [
                        {
                            "summary": "durable insight",
                            "severity": "info|warning|error",
                            "details": {},
                        }
                    ],
                },
            }
        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": _json(user_payload)}
        ]
        allowed_tools = self._allowed_tools()
        if component_publication_phase:
            if self.provider_name == "ollama":
                # Leaf quality comes from isolated previews, independent
                # judging, and revision loops. Keep the initial tool emission
                # bounded so a small thinking model cannot spend the whole
                # transport timeout before publishing the first candidate.
                setattr(self.provider, "reasoning_effort", "off")
                setattr(self.provider, "max_output_tokens", 2_048)
                self.events.publish(
                    "ultra.component_generation_routed",
                    (
                        f"[{request.node_id}] component emission think=off, output<=2048, "
                        f"prompt={len(conversation[0]['content'])} chars"
                    ),
                    run_id=request.run_id,
                    node_id=request.node_id,
                    role=self.role.value,
                    phase=request.phase,
                    prompt_chars=len(conversation[0]["content"]),
                )
            # A component specialist has no final workspace write ownership.
            # Exposing generic mutation tools encourages small local models to
            # bypass the typed package contract or merely describe a write.
            allowed_tools = allowed_tools & _READ_TOOLS
        if self.role is AgentRole.TESTER and self._html_write_target(request):
            # HTML verification is platform-neutral through preview_html plus
            # deterministic readback metrics. Avoid shell quoting/OS drift.
            allowed_tools = allowed_tools - {"run_bash", "run_command"}
        schemas = [] if request.phase == "goal_spec" else _schemas(allowed_tools)
        if component_publication_phase:
            schemas.extend(
                (dict(_STAGE_COMPONENT_FILE_TOOL), dict(_PUBLISH_COMPONENT_TOOL))
            )
        totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
        self.events.publish(
            "ultra.agent_started",
            f"[{self.role.value}] {request.phase}",
            run_id=request.run_id,
            node_id=request.node_id,
            role=self.role.value,
            phase=request.phase,
        )
        last_error: Exception | None = None
        invalid_json_attempts = 0
        max_invalid_json_attempts = 4 if self.provider_name.casefold() == "ollama" else 2
        for step in range(1, self.max_steps + 1):
            try:
                system_prompt = request.system_prompt
                contract_for_prompt = (
                    dict(request.task.get("contract", {}))
                    if isinstance(request.task, Mapping)
                    else {}
                )
                component_only = bool(
                    dict(contract_for_prompt.get("metadata", {})).get("component_package_only")
                )
                if self.role is AgentRole.CODER and request.phase in {
                    InnerPhase.IMPLEMENT.value,
                    InnerPhase.FIX.value,
                }:
                    if component_only:
                        system_prompt += (
                            "\n\nCOMPONENT PACKAGE PHASE: this specialist must not write the shared final "
                            "artifact. Stage exactly one complete file per stage_component_file call, "
                            "including implementation, test, and preview HTML files. Then call "
                            "publish_component with the interface and preview entrypoint. "
                            "A prose claim or payload.component_package without the tool call is invalid. "
                            "The interface must declare concrete exports/imports. "
                            "Every component preview must visibly demonstrate only this component in a "
                            "neutral, reviewable scene. Descriptions or summaries without full file content "
                            "are invalid. If the tool returns findings, revise only this component and call "
                            "publish_component again. Return the response_contract only after the tool says "
                            "passed=true. The parent FinalAssembler alone owns final_output_paths."
                        )
                    else:
                        system_prompt += (
                            "\n\nMUTATION PHASE: completion is impossible until at least one successful "
                            "workspace write/edit occurs inside contract.write_paths. Prefer write_file for "
                            "a complete HTML replacement. If native tool calling is unavailable, return the "
                            "complete artifact in payload.proposed_write; the harness will validate its exact "
                            "path and execute it. Never return success from a read-only state."
                        )
                if (
                    self.role is AgentRole.INTEGRATOR
                    and component_only
                    and not bool(request.task.get("final_assembler"))
                    and request.phase in {InnerPhase.INTEGRATE.value, InnerPhase.FIX.value}
                ):
                    system_prompt += (
                        "\n\nMATERIALIZED PARENT PACKAGE PHASE: integrate the exact child package "
                        "file_contents and exports. Stage each integrated file separately, then call "
                        "publish_component with a concrete interface and runnable preview entrypoint. "
                        "Do not summarize or "
                        "independently recreate child work. "
                        "Do not return final success until publish_component returns passed=true."
                    )
                if self.role is AgentRole.INTEGRATOR and bool(request.task.get("final_assembler")):
                    system_prompt += (
                        "\n\nFINAL ASSEMBLER PHASE: compose the supplied child_component_packages into "
                        "the approved final write_paths. You are the only owner of those final paths; perform "
                        "a real write, then read back and verify the integrated artifact. Consume the exact "
                        "materialized file_contents/exports and preserve their hashes where files remain "
                        "separate; do not recreate a child's implementation from its summary. The harness "
                        "will reject assembly when approved child bytes are neither copied nor inlined."
                    )
                turn = self.provider.call(conversation, schemas, system_prompt)
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status == 429 or "rate limit" in str(exc).casefold():
                    raise RateLimitError(str(exc)) from exc
                raise
            if not isinstance(turn, AssistantTurn):
                raise TypeError("ULTRA provider returned an invalid turn")
            if turn.usage:
                for key in totals:
                    totals[key] += int(getattr(turn.usage, key, 0) or 0)
            if not turn.tool_calls and schemas and turn.text:
                candidate = extract_first_json_object(turn.text)
                proposal = normalize_action_proposal(candidate) if candidate is not None else None
                if proposal is not None:
                    name, args = proposal
                    allowed = {_schema_name(schema) for schema in schemas}
                    if name in allowed:
                        turn.tool_calls.append(
                            ToolCall(
                                id=f"ultra-harness-{request.node_id or request.phase}-{step}",
                                name=name,
                                args=normalize_generated_tool_args(name, args),
                            )
                        )
                        self.events.publish(
                            "ultra.tool_proposal_normalized",
                            f"Normalized textual {name} proposal into a governed tool call",
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            tool=name,
                        )
            conversation.append(turn.to_message())
            if turn.tool_calls:
                if not schemas:
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Workspace inspection is already satisfied and tools are disabled for this phase. "
                                "Return the single JSON object required by response_contract now."
                            ),
                        }
                    )
                    continue
                for call in turn.tool_calls:
                    effective_call = ToolCall(
                        id=call.id,
                        name=call.name,
                        args=normalize_generated_tool_args(call.name, call.args),
                    )
                    if call.name == "edit_file":
                        new_text = str(call.args.get("new_str", ""))
                        old_text = str(call.args.get("old_str", ""))
                        complete_html = bool(
                            re.search(r"(?is)<!doctype\s+html|<html\b", new_text)
                        )
                        replacing_document = not old_text.strip() or bool(
                            re.search(r"(?is)<!doctype\s+html|<html\b", old_text)
                        )
                        if complete_html and replacing_document and "write_file" in self._allowed_tools():
                            effective_call = ToolCall(
                                id=call.id,
                                name="write_file",
                                args={"path": call.args.get("path", ""), "content": new_text},
                            )
                            self.events.publish(
                                "ultra.full_document_write_normalized",
                                "Normalized full-document edit_file proposal to write_file",
                                run_id=request.run_id,
                                node_id=request.node_id,
                                path=call.args.get("path", ""),
                            )
                    result = self.executor(effective_call, request)
                    if effective_call.name == "publish_component":
                        try:
                            publication_result = json.loads(str(result))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            publication_result = {}
                        component_publication_passed = bool(
                            isinstance(publication_result, Mapping)
                            and publication_result.get("passed")
                        )
                        if not component_publication_passed:
                            # The rejected source can be many thousands of
                            # tokens and is not evidence. Keep the typed
                            # finding/tool receipt, but compact the replayed
                            # assistant call so the next revision has room to
                            # generate a fresh candidate.
                            assistant_message = conversation[-1]
                            if isinstance(assistant_message, dict):
                                for historical_call in assistant_message.get(
                                    "tool_calls", ()
                                ):
                                    if (
                                        isinstance(historical_call, dict)
                                        and historical_call.get("id") == effective_call.id
                                    ):
                                        historical_call["args"] = {
                                            "rejected_candidate_omitted": True
                                        }
                            self.events.publish(
                                "ultra.component_revision_context_compacted",
                                f"[{request.node_id}] omitted rejected source from revision context",
                                run_id=request.run_id,
                                node_id=request.node_id,
                                phase=request.phase,
                            )
                    if effective_call.name in _READ_TOOLS and not str(result).startswith("Error:"):
                        inspection_observed = True
                    if effective_call.name in _WRITE_TOOLS and not str(result).startswith("Error:"):
                        mutation_observed = True
                    conversation.append(
                        {
                            "role": "tool",
                            "id": effective_call.id,
                            "name": effective_call.name,
                            "content": result,
                        }
                    )
                    if (
                        effective_call.name == "edit_file"
                        and str(result).startswith("Error:")
                        and any(
                            marker in str(result).casefold()
                            for marker in ("already exists", "old_str not found")
                        )
                    ):
                        path = str(effective_call.args.get("path", "")).strip()
                        fresh = str(
                            self.executor(
                                ToolCall(f"harness-edit-readback-{step}", "read_file", {"path": path}),
                                request,
                            )
                        )
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Authoritative readback for {path!r}:\n{fresh[:40_000]}\n\n"
                                    "For a localized edit, retry edit_file with an old_str copied exactly "
                                    "from this readback. For a complete artifact replacement, use write_file "
                                    "with the entire improved content. Do not guess old_str."
                                ),
                            }
                        )
                continue
            try:
                data = _extract_json(str(turn.text or ""))
                response = AgentResponse.from_mapping(
                    data,
                    node_id=request.node_id,
                    provider=self.provider_name,
                    model=self.model,
                    usage=totals,
                )
                repaired_reasoning, reasoning_repairs = repair_reasoning_artifact_graph(
                    response.payload.get("reasoning_artifact")
                )
                reasoning_evaluation = evaluate_reasoning_artifact(
                    repaired_reasoning,
                    debate_protocol,
                )
                if debate_protocol.required:
                    payload = dict(response.payload)
                    payload["reasoning_artifact"] = repaired_reasoning
                    if reasoning_repairs:
                        payload["harness_reasoning_repairs"] = list(reasoning_repairs)
                    payload["harness_reasoning_evaluation"] = reasoning_evaluation.to_dict()
                    response = AgentResponse.from_mapping(
                        {
                            "payload": payload,
                            "summary": response.summary,
                            "reasoning_summary": response.reasoning_summary,
                            "insights": [asdict(insight) for insight in response.insights],
                        },
                        node_id=request.node_id,
                        provider=self.provider_name,
                        model=self.model,
                        usage=totals,
                    )
                if request.phase == "goal_spec" and not inspection_observed:
                    last_error = RuntimeError(
                        "GoalSpecV1 requires repository inspection before questions or planning"
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Inspect the workspace with an available read tool before "
                                "returning GoalSpecV1. Do not ask for facts the repository can answer."
                            ),
                        }
                    )
                    continue
                if harness_preview and request.phase == InnerPhase.TEST.value:
                    preview_status = str(harness_preview.get("verification") or harness_preview.get("status") or "").casefold()
                    browser_failed = preview_status not in {"passed", "ok", "success"}
                    browser_findings = [
                        *[str(item) for item in harness_preview.get("console_errors", ())],
                        *[str(item) for item in harness_preview.get("page_errors", ())],
                        *[str(item) for item in harness_preview.get("network_errors", ())],
                    ]
                    if browser_failed:
                        payload = dict(response.payload)
                        payload["passed"] = False
                        existing_results = list(payload.get("test_results", ()) or ())
                        existing_results.append(
                            {
                                "name": "harness_html_preview",
                                "passed": False,
                                "status": preview_status or "failed",
                                "screenshot_path": harness_preview.get("screenshot_path"),
                            }
                        )
                        payload["test_results"] = existing_results
                        existing_evidence = list(payload.get("evidence", ()) or ())
                        existing_evidence.append(
                            {
                                "kind": "browser_preview",
                                "status": preview_status or "failed",
                                "title": harness_preview.get("title"),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                            }
                        )
                        payload["evidence"] = existing_evidence
                        existing_issues = list(payload.get("issues", ()) or ())
                        existing_issues.append(
                            "Harness browser verification failed for HTML output."
                        )
                        payload["issues"] = existing_issues
                        existing_findings = list(payload.get("findings", ()) or ())
                        existing_findings.extend(browser_findings or ["Harness browser verification failed."])
                        payload["findings"] = existing_findings
                        response = AgentResponse(
                            payload=payload,
                            summary=response.summary or "Harness browser verification failed",
                            insights=response.insights,
                            reasoning_summary=response.reasoning_summary,
                            usage=response.usage,
                            provider=response.provider,
                            model=response.model,
                        )
                contract_payload = (
                    dict(request.task.get("contract", {}))
                    if isinstance(request.task, Mapping)
                    else {}
                )
                requires_mutation = (
                    (
                        self.role is AgentRole.CODER
                        and request.phase in {InnerPhase.IMPLEMENT.value, InnerPhase.FIX.value}
                    )
                    or (
                        self.role is AgentRole.INTEGRATOR
                        and bool(request.task.get("final_assembler"))
                    )
                ) and bool(contract_payload.get("write_paths"))
                if requires_mutation and not mutation_observed:
                    proposed_write = response.payload.get("proposed_write")
                    if isinstance(proposed_write, Mapping):
                        proposed_path = str(proposed_write.get("path", "")).strip()
                        proposed_content = str(proposed_write.get("content", ""))
                        approved_paths = {
                            str(path).strip()
                            for path in contract_payload.get("write_paths", ()) or ()
                            if str(path).strip()
                        }
                        if proposed_path in approved_paths and proposed_content.strip():
                            write_result = str(
                                self.executor(
                                    ToolCall(
                                        f"harness-proposed-write-{step}",
                                        "write_file",
                                        {"path": proposed_path, "content": proposed_content},
                                    ),
                                    request,
                                )
                            )
                            if not write_result.startswith("Error:"):
                                mutation_observed = True
                                self.events.publish(
                                    "ultra.proposed_write_executed",
                                    "Validated and executed typed proposed_write fallback",
                                    run_id=request.run_id,
                                    node_id=request.node_id,
                                    path=proposed_path,
                                    characters=len(proposed_content),
                                )
                if requires_mutation and not mutation_observed:
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "No successful workspace mutation was observed. This phase owns "
                                f"write_paths={list(contract_payload.get('write_paths', ()))!r}. "
                                "Use an allowed write/edit tool now, then inspect the changed artifact "
                                "Use write_file for a complete artifact replacement; use edit_file only "
                                "with an exact old_str from harness_write_target_state. "
                                "and only afterward return the required JSON result. A prose or JSON-only "
                                "claim cannot complete an implementation/fix phase."
                            ),
                        }
                    )
                    continue
                if request_component_only and not component_publication_passed:
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Your final response is rejected because the harness has no successful "
                                "publish_component receipt. Stage each complete implementation/test/preview "
                                "file, then call publish_component. If a previous call returned findings, "
                                "revise the staged files and call it again. Do not return another prose "
                                "claim or JSON-only package."
                            ),
                        }
                    )
                    continue
                return response
            except Exception as exc:
                content_preview = redact_text(str(turn.text or ""), 800)
                last_error = RuntimeError(
                    f"{exc}; content_preview={content_preview!r}"
                )
                invalid_json_attempts += 1
                if invalid_json_attempts >= max_invalid_json_attempts:
                    raise RuntimeError(
                        f"{self.role.value} returned invalid structured JSON "
                        f"{invalid_json_attempts} times: {last_error}"
                    ) from exc
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not the single JSON object required by "
                            "response_contract. Return the corrected JSON object now."
                        ),
                    }
                )
        raise RuntimeError(
            f"{self.role.value} did not produce a valid structured result after "
            f"{self.max_steps} steps: {last_error or 'tool loop exhausted'}"
        )


class WorkspaceUltraAgentFactory:
    def __init__(
        self,
        descriptor: ModelDescriptor,
        executor: Callable[[ToolCall, AgentRequest], str],
        events: EventBus,
        *,
        max_steps: int,
        reasoning_effort: str = "medium",
    ) -> None:
        self.descriptor = descriptor
        self.executor = executor
        self.events = events
        self.max_steps = max_steps
        self.reasoning_effort = str(reasoning_effort)

    def create(
        self,
        role: AgentRole,
        *,
        run_id: str,
        node_id: str | None = None,
    ) -> WorkspaceUltraAgent:
        del run_id, node_id
        provider = self.descriptor.create_provider()
        setattr(provider, "reasoning_effort", self.reasoning_effort)
        return WorkspaceUltraAgent(
            provider,
            role=role,
            provider_name=self.descriptor.provider,
            model=self.descriptor.model,
            executor=self.executor,
            events=self.events,
            max_steps=self.max_steps,
        )


class DurableContextBuilder:
    """Prefer SQLite/FTS retrieval and fall back during not-yet-flushed expansion."""

    def __init__(self, store: StateStore, run_id: Callable[[], str | None], max_chars: int) -> None:
        self.store = store
        self.run_id = run_id
        self.max_chars = max_chars
        self.fallback = FocusedContextBuilder(max_chars)

    def build(self, request: ContextRequest) -> Mapping[str, Any]:
        run_id = self.run_id()
        if not run_id:
            return self.fallback.build(request)
        try:
            package = ProjectBrain(self.store, run_id).build_context(
                request.node.id,
                request.role.value,
                query=request.node.contract.objective,
                budget_chars=self.max_chars,
            )
        except (StateStoreError, DomainError):
            return self.fallback.build(request)
        sections = dict(package.sections)
        sections.setdefault("north_star", asdict(request.goal))
        sections.setdefault(
            "architecture_contract",
            {
                "summary": request.architecture.summary,
                "interfaces": list(request.architecture.interfaces),
                "invariants": list(request.architecture.invariants),
            },
        )
        sections["_omitted"] = list(package.omitted_sections)
        return sections


def _store_phase(phase: EnginePhase) -> UltraPhase:
    return {
        EnginePhase.NEW: UltraPhase.GOAL_INTERVIEW,
        EnginePhase.GOAL_SPEC: UltraPhase.GOAL_SPEC,
        EnginePhase.AWAITING_QUESTIONS: UltraPhase.GOAL_SPEC,
        EnginePhase.ARCHITECTURE: UltraPhase.ARCHITECTURE,
        EnginePhase.MASTER_PLAN: UltraPhase.MASTER_PLAN,
        EnginePhase.AWAITING_APPROVAL: UltraPhase.AWAITING_APPROVAL,
        EnginePhase.EXPANDING: UltraPhase.MODULE_WAVES,
        EnginePhase.MODULE_WAVES: UltraPhase.MODULE_WAVES,
        EnginePhase.INTEGRATION: UltraPhase.INTEGRATION,
        EnginePhase.GLOBAL_REVIEW: UltraPhase.GLOBAL_REVIEW,
        EnginePhase.FINAL_EVIDENCE: UltraPhase.EVIDENCE_GATE,
        EnginePhase.COMPLETED: UltraPhase.COMPLETED,
    }.get(phase, UltraPhase.MODULE_WAVES)


def _store_run_status(phase: EnginePhase) -> UltraRunStatus:
    if phase is EnginePhase.AWAITING_APPROVAL:
        return UltraRunStatus.AWAITING_APPROVAL
    if phase is EnginePhase.AWAITING_QUESTIONS:
        return UltraRunStatus.PAUSED
    if phase is EnginePhase.PAUSED:
        return UltraRunStatus.PAUSED
    if phase is EnginePhase.REVISION_REQUIRED:
        return UltraRunStatus.REVISION_REQUIRED
    if phase is EnginePhase.CANCELLED:
        return UltraRunStatus.CANCELLED
    if phase is EnginePhase.FAILED:
        return UltraRunStatus.BLOCKED
    if phase is EnginePhase.COMPLETED:
        return UltraRunStatus.COMPLETED
    if phase in {
        EnginePhase.EXPANDING,
        EnginePhase.MODULE_WAVES,
        EnginePhase.INTEGRATION,
        EnginePhase.GLOBAL_REVIEW,
        EnginePhase.FINAL_EVIDENCE,
    }:
        return UltraRunStatus.RUNNING
    return UltraRunStatus.DRAFT


def _store_node_status(status: NodeStatus) -> WorkNodeStatus:
    return {
        NodeStatus.PENDING: WorkNodeStatus.PENDING,
        # Foundation expansion plans every module before execution waves.
        # Marking PLANNING as IN_PROGRESS trips the durable dependency gate for
        # M002+ while M001 is intentionally not executed yet.
        NodeStatus.PLANNING: WorkNodeStatus.PENDING,
        NodeStatus.READY: WorkNodeStatus.READY,
        NodeStatus.RUNNING: WorkNodeStatus.IN_PROGRESS,
        NodeStatus.COMPLETED: WorkNodeStatus.COMPLETED,
        NodeStatus.FAILED: WorkNodeStatus.FAILED,
        NodeStatus.BLOCKED: WorkNodeStatus.BLOCKED,
        NodeStatus.CONFLICT: WorkNodeStatus.CONFLICT,
        NodeStatus.CANCELLED: WorkNodeStatus.CANCELLED,
        NodeStatus.UNCERTAIN: WorkNodeStatus.UNCERTAIN,
        NodeStatus.REVISION_REQUIRED: WorkNodeStatus.REVISION_REQUIRED,
    }[status]


def _store_kind(kind: NodeKind) -> WorkNodeKind:
    return WorkNodeKind(kind.value)


def _safe_task_id(value: str, index: int, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).upper()).strip("-._")
    base = (base or f"M{index:03d}")[:24]
    candidate = base
    suffix = 2
    while candidate in used:
        tail = f"-{suffix}"
        candidate = base[: 24 - len(tail)] + tail
        suffix += 1
    used.add(candidate)
    return candidate


class StateStoreUltraAdapter(InMemoryUltraState):
    """Mirror live engine state into the durable v3 schema."""

    def __init__(
        self,
        store: StateStore,
        goal_id: str,
        descriptor: ModelDescriptor,
        access_level: AccessLevel,
        config: UltraConfig,
        workspace: Path | None = None,
    ) -> None:
        super().__init__()
        self.store = store
        self.goal_id = goal_id
        self.descriptor = descriptor
        self.access_level = access_level
        self.config = config
        self.workspace = workspace
        self.run_id: str | None = None
        self.plan: Plan | None = None
        self.approved = False
        self.task_ids: dict[str, str] = {}
        self._pending_nodes: dict[str, EngineWorkNode] = {}
        self._result_cache: dict[str, EngineResult] = {}
        self._persisted_nodes: set[str] = set()
        self._persisted_agents: set[str] = set()
        self._persisted_traces: set[str] = set()
        self._pending_artifacts: list[Artifact] = []
        self._brain_results: set[str] = set()
        self._lease_ids: dict[str, list[str]] = {}
        self._lease_scopes: dict[str, tuple[str, ...]] = {}
        self._lease_hashes: dict[str, dict[str, str | None]] = {}
        self._used_project_lessons: dict[str, dict[str, Any]] = {}
        model_name = str(descriptor.model).casefold()
        self._global_memory_enabled = not model_name.startswith(("offline", "fake", "test"))
        self.global_lessons = GlobalLessonStore()
        self._used_global_lesson_ids: set[str] = set()
        self.component_artifacts = (
            ComponentArtifactStore(workspace)
            if workspace is not None
            else None
        )
        self._materialized_packages: dict[str, MaterializedComponentPackageV2] = {}
        self._component_previews: dict[str, str] = {}
        self._published_component_results: dict[str, Mapping[str, Any]] = {}
        self.visual_judge = create_visual_judge(
            builder_provider=descriptor.provider,
            builder_model=descriptor.model,
            ollama_host=descriptor.host or "http://127.0.0.1:11434",
        )
        self._adapter_lock = threading.RLock()

    def _run_config(self, run: UltraRunV1) -> dict[str, Any]:
        return {
            **dict(run.config_snapshot),
            "model_snapshot": dict(run.model_snapshot),
            "prompt": run.prompt,
            "engine_metadata": dict(run.metadata),
        }

    def save_ultra_run(self, run: UltraRunV1) -> None:
        super().save_ultra_run(run)
        with self._adapter_lock:
            self.run_id = run.id
            try:
                self.store.get_ultra_run(run.id)
            except NotFoundError:
                self.store.create_ultra_run(
                    UltraRun(
                        id=run.id,
                        goal_id=self.goal_id,
                        provider=self.descriptor.provider,
                        model=self.descriptor.model,
                        execution_class=self.descriptor.execution_class,
                        access_level=self.access_level,
                        concurrency=run.concurrency,
                        phase=_store_phase(run.phase),
                        status=_store_run_status(run.phase),
                        config=self._run_config(run),
                    )
                )
                self.store.update_goal_metadata(
                    self.goal_id,
                    ultra_run_id=run.id,
                    interaction_mode="ultra",
                )
                return
            self.store.update_ultra_run(
                run.id,
                provider=self.descriptor.provider,
                model=self.descriptor.model,
                execution_class=self.descriptor.execution_class,
                access_level=self.access_level,
                concurrency=(
                    1
                    if self.descriptor.execution_class is ExecutionClass.LOCAL
                    else max(1, min(8, run.concurrency))
                ),
                phase=_store_phase(run.phase),
                status=_store_run_status(run.phase),
                config=self._run_config(run),
                error=("ULTRA execution failed" if run.phase is EnginePhase.FAILED else None),
            )

    def save_specialist_profile(self, run_id: str, profile: SpecialistProfileV1) -> None:
        super().save_specialist_profile(run_id, profile)
        self.store.save_specialist_profile(
            {
                **asdict(profile),
                "ultra_run_id": run_id,
                "work_node_id": profile.node_id,
            }
        )

    def save_interface_contract(
        self,
        run_id: str,
        node_id: str,
        contract: Mapping[str, Any],
    ) -> None:
        self.store.save_interface_contract(run_id, node_id, contract)

    @staticmethod
    def _visual_rubric(node: EngineWorkNode) -> Mapping[str, Any]:
        domain = str(node.contract.metadata.get("specialist_domain") or "").casefold()
        dimensions: Mapping[str, tuple[str, ...]] = {
            "vehicles": (
                "silhouette", "proportions", "wheels_contact", "cabin_glass",
                "lights", "materials", "detail",
            ),
            "character": (
                "silhouette", "anatomy_stylization", "pose", "animation", "readability",
            ),
            "world": (
                "road_language", "depth", "environment_density", "lighting", "composition",
            ),
            "gameplay": (
                "responsiveness", "collisions", "pacing", "feedback", "progression",
            ),
            "presentation": (
                "camera", "hud_readability", "feedback", "polish", "accessibility",
            ),
            "qa": (
                "evidence_readability", "coverage", "runtime_health", "performance",
            ),
        }
        root = domain.split(".", 1)[0]
        selected = dimensions.get(
            root,
            ("task_fit", "composition", "readability", "polish", "integration_readiness"),
        )
        return {
            "domain": domain or node.contract.title,
            "dimensions": list(selected),
            "critical_minimum": 0.85,
            "zero_critical_findings": True,
        }

    def materialize_component_candidate(
        self,
        run_id: str,
        node: EngineWorkNode,
        candidate: AgentResponse,
        *,
        revision: int,
        child_packages: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if self.component_artifacts is None:
            raise ComponentArtifactError("component artifact store requires a workspace")
        published = self._published_component_results.pop(node.id, None)
        if published is not None:
            return dict(published)
        raw = candidate.payload.get("component_package")
        if not isinstance(raw, Mapping):
            raise ComponentArtifactError(
                "specialist response omitted payload.component_package"
            )
        component = dict(raw)
        designed = UltraOrchestrator._interface_contract(node)
        supplied_interface = component.get("interface")
        if isinstance(supplied_interface, Mapping):
            interface = {
                **dict(designed),
                **dict(supplied_interface),
            }
            if not interface.get("exports"):
                interface["exports"] = list(designed["exports"])
        else:
            interface = dict(designed)
        component["interface"] = interface
        package = self.component_artifacts.materialize(
            run_id=run_id,
            node_id=node.id,
            component=component,
            revision=revision,
            dependencies=node.depends_on,
            evidence=tuple(
                dict(item)
                for item in candidate.payload.get("evidence", ())
                if isinstance(item, Mapping)
            ),
            quality={"status": "pending_independent_evaluation"},
            parent_package_ids=tuple(
                str(value.get("id"))
                for value in child_packages.values()
                if value.get("id")
            ),
        )
        stored = self.store.put_materialized_component_package(package.to_dict())
        self._materialized_packages[package.id] = package
        preview = self.component_artifacts.verify_preview(package)
        screenshot = str(preview.get("screenshot_path") or "")
        findings: list[str] = []
        runtime_passed = str(preview.get("status")) == "passed" and bool(screenshot)
        if not runtime_passed:
            findings.append(
                "component preview failed runtime verification: "
                + str(preview.get("reason") or preview.get("status") or "unknown")
            )
        anomaly_findings = screenshot_anomalies(screenshot) if screenshot else ()
        findings.extend(f"visual anomaly gate: {item}" for item in anomaly_findings)
        verdict_values: list[Mapping[str, Any]] = []
        pairwise_value: Mapping[str, Any] | None = None
        status = "evaluated"
        if runtime_passed:
            try:
                verdicts = require_two_clean_acceptances(
                    self.visual_judge,
                    brief=node.contract.objective,
                    rubric=self._visual_rubric(node),
                    screenshot=screenshot,
                    runtime_evidence=preview,
                    nonce_prefix=f"{run_id}:{node.id}:r{revision}",
                )
                for verdict in verdicts:
                    value = verdict.to_dict()
                    verdict_values.append(value)
                    self.store.save_visual_evaluation(
                        run_id,
                        value,
                        work_node_id=node.id,
                        package_id=stored["id"],
                    )
                    findings.extend(item.message for item in verdict.findings)
                previous = self._component_previews.get(node.id)
                if previous and Path(previous).is_file():
                    comparison = self.visual_judge.compare(
                        brief=node.contract.objective,
                        rubric=self._visual_rubric(node),
                        candidate=screenshot,
                        baseline=previous,
                        clean_context_nonce=f"{run_id}:{node.id}:pairwise:r{revision}",
                    )
                    pairwise_value = comparison.to_dict()
                    self.store.save_pairwise_visual_comparison(
                        run_id,
                        pairwise_value,
                        work_node_id=node.id,
                    )
                    if not comparison.candidate_preferred:
                        findings.append(
                            "blind pairwise judge did not prefer this revision over its baseline"
                        )
                self._component_previews[node.id] = screenshot
            except VisualJudgeUnavailable as exc:
                status = "USER_REVIEW_REQUIRED"
                findings.append(str(exc))
                failure_value = {
                    "evaluator": str(
                        getattr(self.visual_judge, "evaluator", "unavailable_vision")
                    ),
                    "model": str(getattr(self.visual_judge, "model", "")),
                    "accepted": False,
                    "scores": {},
                    "findings": [
                        {
                            "severity": "critical",
                            "category": "evaluator_error",
                            "message": str(exc),
                            "evidence": screenshot,
                        }
                    ],
                    "summary": "Independent visual evaluation could not produce a valid verdict.",
                    "confidence": 0.0,
                    "screenshot_hash": hashlib.sha256(
                        Path(screenshot).read_bytes()
                    ).hexdigest(),
                    "context_fingerprint": hashlib.sha256(
                        f"{run_id}:{node.id}:visual-error:{revision}".encode("utf-8")
                    ).hexdigest(),
                    "status": "USER_REVIEW_REQUIRED",
                    "version": 1,
                }
                verdict_values.append(failure_value)
                self.store.save_visual_evaluation(
                    run_id,
                    failure_value,
                    work_node_id=node.id,
                    package_id=stored["id"],
                )
        accepted_twice = (
            len(verdict_values) == 2
            and all(bool(value.get("accepted")) for value in verdict_values)
        )
        pairwise_passed = (
            pairwise_value is None
            or bool(pairwise_value.get("candidate_preferred"))
        )
        passed = (
            runtime_passed
            and not anomaly_findings
            and accepted_twice
            and pairwise_passed
        )
        if status == "USER_REVIEW_REQUIRED":
            passed = False
        return {
            "passed": passed,
            "status": "accepted" if passed else status if status != "evaluated" else "rejected",
            "package": package.to_dict(include_content=True),
            "stored_package_id": stored["id"],
            "preview": preview,
            "visual_evaluations": verdict_values,
            "pairwise_comparison": pairwise_value,
            "findings": list(dict.fromkeys(findings)),
        }

    def publish_component_tool(
        self,
        run_id: str,
        node: EngineWorkNode,
        component: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        existing = self.store.list_component_packages(
            run_id,
            work_node_id=node.id,
        )
        revision = max(
            (int(item.get("version") or 0) for item in existing),
            default=0,
        ) + 1
        normalized_component = dict(component)
        staged_files = list(
            self.component_artifacts.draft_files(
                run_id=run_id,
                node_id=node.id,
            )
        ) if self.component_artifacts is not None else []
        preview = (
            dict(normalized_component.get("preview", {}))
            if isinstance(normalized_component.get("preview"), Mapping)
            else {}
        )
        preview_content = str(preview.pop("content", ""))
        preview_entrypoint = str(preview.get("entrypoint", "")).strip()
        implementation = (
            dict(normalized_component.get("implementation", {}))
            if isinstance(normalized_component.get("implementation"), Mapping)
            else {}
        )
        implementation_files = [
            dict(item)
            for item in implementation.get("files", ())
            if isinstance(item, Mapping)
        ]
        implementation_by_path = {
            str(item.get("path", "")): item
            for item in (
                *implementation_files,
                *(item for item in staged_files if str(item.get("role")) != "test"),
            )
            if str(item.get("path", ""))
        }
        implementation_files = list(implementation_by_path.values())
        staged_tests = [
            item for item in staged_files if str(item.get("role")) == "test"
        ]
        known_paths = {str(item.get("path", "")) for item in implementation_files}
        implementation["files"] = [
            *implementation_files,
            *(
                [
                    {
                        "path": preview_entrypoint,
                        "content": preview_content,
                        "role": "preview",
                    }
                ]
                if preview_entrypoint
                and preview_content.strip()
                and preview_entrypoint not in known_paths
                else []
            ),
        ]
        normalized_component["implementation"] = implementation
        normalized_component["preview"] = preview
        normalized_component["tests"] = [
            *(
                dict(item)
                for item in normalized_component.get("tests", ())
                if isinstance(item, Mapping)
            ),
            *staged_tests,
        ]
        response = AgentResponse(
            payload={"component_package": normalized_component},
            summary=f"Typed component publication for {node.id}",
            reasoning_summary="Files were submitted through publish_component.",
            provider="harness_tool",
            model="publish-component-v2",
        )
        self._published_component_results.pop(node.id, None)
        result = dict(
            self.materialize_component_candidate(
                run_id,
                node,
                response,
                revision=revision,
                child_packages={},
            )
        )
        self._published_component_results[node.id] = result
        return result

    def stage_component_file_tool(
        self,
        run_id: str,
        node: EngineWorkNode,
        *,
        path: str,
        content: str,
        role: str,
    ) -> Mapping[str, Any]:
        if self.component_artifacts is None:
            raise ComponentArtifactError("component artifact store requires a workspace")
        return self.component_artifacts.stage_draft_file(
            run_id=run_id,
            node_id=node.id,
            path=path,
            content=content,
            role=role,
        )

    def verify_package_consumption(
        self,
        run_id: str,
        node: EngineWorkNode,
        packages: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if self.component_artifacts is None or self.workspace is None:
            return {
                "passed": False,
                "findings": ["package consumption verification requires a workspace"],
            }
        materialized = tuple(
            self._materialized_packages[str(item.get("id"))]
            for item in packages
            if str(item.get("id")) in self._materialized_packages
        )
        missing = [
            str(item.get("id") or item.get("node_id") or "unknown")
            for item in packages
            if str(item.get("id")) not in self._materialized_packages
        ]
        target_paths = tuple(self.workspace / path for path in node.write_paths)
        evidence = self.component_artifacts.verify_consumption(
            assembler_node_id=node.id,
            packages=materialized,
            target_paths=target_paths,
        )
        for item in evidence:
            self.store.save_package_consumption_evidence(
                run_id,
                node.id,
                item.package_id,
                item.to_dict(),
            )
        findings = [
            finding
            for item in evidence
            for finding in item.findings
        ]
        findings.extend(
            f"child package {identifier} was not materialized in this run"
            for identifier in missing
        )
        passed = bool(packages) and not missing and bool(evidence) and all(
            item.passed for item in evidence
        )
        return {
            "passed": passed,
            "evidence": [item.to_dict() for item in evidence],
            "findings": findings,
        }

    def save_component_package(self, run_id: str, package: ComponentPackageV1) -> None:
        super().save_component_package(run_id, package)
        stored = self.store.put_component_package(
            {
                **asdict(package),
                "ultra_run_id": run_id,
                "work_node_id": package.node_id,
            }
        )
        node = self.store.get_work_node(package.node_id)
        self.store.post_swarm_message(
            SwarmMessageV1(
                ultra_run_id=run_id,
                sender_agent_id=f"specialist:{package.node_id}",
                recipient_agent_id=(
                    f"specialist:{node.parent_id}" if node.parent_id else "final-assembler"
                ),
                message_type=SwarmMessageType.PACKAGE_PUBLISHED,
                topic=f"component-package:{package.node_id}",
                payload={
                    "package_id": stored["id"],
                    "node_id": package.node_id,
                    "status": package.status,
                    "content_hash": stored["content_hash"],
                    "quality": dict(package.quality),
                },
                evidence=package.evidence,
                correlation_id=package.node_id,
            )
        )

    def save_node_quality_target(self, run_id: str, target: NodeQualityTargetV1) -> None:
        super().save_node_quality_target(run_id, target)
        self.store.save_node_quality_target(
            run_id,
            target.node_id,
            asdict(target),
            status="not_evaluated",
        )

    def foundation_project_lessons(
        self,
        run_id: str,
        query: str,
        *,
        phase: str,
        limit: int = 8,
    ) -> tuple[Mapping[str, Any], ...]:
        if self.run_id != run_id:
            return ()
        lesson_memories = self.store.search_project_memory(
            query,
            section=BrainSection.LESSON,
            min_confidence=0.4,
            limit=limit,
        )
        knowledge_memories = self.store.search_project_memory(
            query,
            section=BrainSection.KNOWLEDGE,
            min_confidence=0.4,
            limit=limit,
        )
        memories = tuple(
            sorted(
                (*lesson_memories, *knowledge_memories),
                key=lambda item: (
                    -float(item.get("effective_confidence", item.get("confidence", 0.0)) or 0.0),
                    -int(item.get("reuse_count", 0) or 0),
                    str(item.get("title") or ""),
                ),
            )[: max(1, int(limit))]
        )
        result: list[Mapping[str, Any]] = []
        for memory in memories:
            self.store.record_project_memory_use(str(memory["id"]))
            with self._adapter_lock:
                tracked = self._used_project_lessons.setdefault(
                    str(memory["id"]),
                    {
                        "id": str(memory["id"]),
                        "title": memory["title"],
                        "phases": [],
                        "queries": [],
                        "initial_confidence": memory["confidence"],
                        "initial_effective_confidence": memory.get("effective_confidence"),
                    },
                )
                if phase not in tracked["phases"]:
                    tracked["phases"].append(phase)
                query_text = str(query or "")[:500]
                if query_text and query_text not in tracked["queries"]:
                    tracked["queries"].append(query_text)
            result.append(
                {
                    "id": memory["id"],
                    "section": memory["section"],
                    "phase": phase,
                    "title": memory["title"],
                    "content": memory["content"],
                    "confidence": memory["confidence"],
                    "effective_confidence": memory.get("effective_confidence", memory["confidence"]),
                    "reuse_count": memory["reuse_count"],
                    "evidence_refs": memory["evidence_refs"],
                }
            )
        if self._global_memory_enabled and len(result) < max(1, int(limit)):
            for lesson in self.global_lessons.search(
                query,
                limit=max(1, int(limit)) - len(result),
            ):
                self._used_global_lesson_ids.add(lesson.id)
                result.append(
                    {
                        "id": lesson.id,
                        "section": BrainSection.LESSON.value,
                        "phase": phase,
                        "title": lesson.title,
                        "content": lesson.content,
                        "confidence": lesson.confidence,
                        "effective_confidence": lesson.confidence,
                        "reuse_count": lesson.successes + lesson.failures,
                        "evidence_refs": lesson.evidence_refs,
                        "scope": "global",
                    }
                )
        return tuple(result)

    def _record_global_lesson_evaluation_outcomes(
        self,
        *,
        passed: bool,
        benchmark_id: str,
        blocker: str,
        html_benchmark: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any], ...]:
        if not self._global_memory_enabled:
            return ()
        outcomes: list[Mapping[str, Any]] = []
        for lesson_id in tuple(self._used_global_lesson_ids):
            updated = self.global_lessons.record_outcome(lesson_id, succeeded=passed)
            if updated is not None:
                outcomes.append(
                    {"id": updated.id, "confidence": updated.confidence, "succeeded": passed}
                )
        visual = html_benchmark is not None
        content = (
            "Use recursive component isolation, FinalAssembler ownership, independent review, "
            "and evidence-backed consensus before accepting an Ultra result."
        )
        if visual:
            content += " Interactive HTML requires clean browser runtime, screenshots, and critical visual scores."
        if blocker:
            content += f" Last blocker pattern: {redact_text(blocker, 500)}"
        learned = self.global_lessons.put(
            LearnedLessonV1(
                title="Ultra recursive quality gate",
                content=content,
                applicability_tags=(
                    "ultra",
                    "recursive-specialists",
                    "visual" if visual else "integration",
                ),
                evidence_refs=(f"benchmark:{benchmark_id}",),
                successes=1 if passed else 0,
                failures=0 if passed else 1,
            )
        )
        outcomes.append(
            {"id": learned.id, "confidence": learned.confidence, "succeeded": passed, "recorded": True}
        )
        return tuple(outcomes)

    def _record_project_lesson_evaluation_outcomes(
        self,
        *,
        passed: bool,
        benchmark_id: str,
        html_benchmark_id: str | None = None,
        blocker: str = "",
    ) -> tuple[Mapping[str, Any], ...]:
        with self._adapter_lock:
            lessons = tuple(dict(item) for item in self._used_project_lessons.values())
        if not lessons:
            return ()
        evidence_ref = f"benchmark:{benchmark_id}"
        if html_benchmark_id:
            evidence_ref = f"{evidence_ref};html:{html_benchmark_id}"
        outcomes: list[Mapping[str, Any]] = []
        weight = 1.0 if passed else 1.5
        reason_prefix = "ULTRA global evaluation passed" if passed else "ULTRA global evaluation failed"
        for lesson in lessons:
            try:
                updated = self.store.record_project_memory_outcome(
                    str(lesson["id"]),
                    succeeded=passed,
                    evidence_ref=evidence_ref,
                    reason=(
                        f"{reason_prefix}; phases={','.join(lesson.get('phases', ()))}; "
                        f"blocker={blocker or 'none'}"
                    ),
                    weight=weight,
                )
            except (StateStoreError, ValueError) as exc:
                outcomes.append(
                    {
                        "id": lesson.get("id"),
                        "updated": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            outcomes.append(
                {
                    "id": updated["id"],
                    "updated": True,
                    "confidence": updated["confidence"],
                    "effective_confidence": updated.get("effective_confidence"),
                    "positive_outcomes": updated.get("metadata", {}).get("positive_outcomes", 0),
                    "negative_outcomes": updated.get("metadata", {}).get("negative_outcomes", 0),
                    "phases": lesson.get("phases", ()),
                }
            )
        return tuple(outcomes)

    @staticmethod
    def _remediation_steps_for_global_blocker(blocker: str) -> tuple[str, ...]:
        normalized = str(blocker or "").casefold()
        steps: list[str] = []
        if "consensus" in normalized:
            steps.extend(
                [
                    "Inspect every rejected, tied, or open quality vote and convert each rationale into a concrete fix task.",
                    "Do not accept the run until the same voters produce a fresh accepted consensus round.",
                ]
            )
        if "durable evidence" in normalized or "final evidence" in normalized:
            steps.extend(
                [
                    "Re-run the final evidence phase with concrete artifacts, test results, and observable proof instead of summaries.",
                    "Attach browser/runtime/test evidence that can be independently inspected after the run.",
                ]
            )
        if "html" in normalized or "3d" in normalized or "webgl" in normalized:
            steps.extend(
                [
                    "Run the single-file 3D HTML benchmark before completion and treat low visual/runtime scores as blocking.",
                    "Improve scene depth, lighting, interaction coverage, animation density, HUD clarity, and runtime error handling before retesting.",
                ]
            )
        if "regressed" in normalized or "regression" in normalized:
            steps.extend(
                [
                    "Compare the latest benchmark against the previous baseline and target the exact regressed score dimensions.",
                    "Prefer a smaller verified remediation over broad rewrites that risk new regressions.",
                ]
            )
        if "module" in normalized:
            steps.append("Re-open the failed module nodes and rerun their fix loop before global integration.")
        if not steps:
            steps.extend(
                [
                    "Treat the global evaluation blocker as a first-class remediation requirement, not a final summary.",
                    "Create a focused fix plan, rerun the relevant quality gate, then rerun global evaluation.",
                ]
            )
        return tuple(dict.fromkeys(steps))

    def _record_global_remediation_knowledge(
        self,
        *,
        passed: bool,
        benchmark_id: str,
        blocker: str,
        metrics: Mapping[str, Any],
        scores: Mapping[str, Any],
        html_benchmark: Mapping[str, Any] | None = None,
        global_trend: Mapping[str, Any] | None = None,
        html_trend: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        if passed or not self.run_id:
            return None
        blocker_text = str(blocker or "global evaluation gate failed").strip()
        title_seed = re.sub(r"[^a-z0-9]+", " ", blocker_text.casefold()).strip()
        title_seed = re.sub(r"\s+", " ", title_seed)[:90] or "global evaluation gate failed"
        steps = self._remediation_steps_for_global_blocker(blocker_text)
        evidence_refs = [f"benchmark:{benchmark_id}"]
        if html_benchmark and html_benchmark.get("id"):
            evidence_refs.append(f"html-benchmark:{html_benchmark['id']}")
        if global_trend and global_trend.get("id"):
            evidence_refs.append(f"trend:{global_trend['id']}")
        if html_trend and html_trend.get("id"):
            evidence_refs.append(f"html-trend:{html_trend['id']}")
        data = {
            "kind": "automatic_global_remediation",
            "blocker": blocker_text,
            "benchmark_id": benchmark_id,
            "html_benchmark_id": html_benchmark.get("id") if html_benchmark else None,
            "benchmark_trend_id": global_trend.get("id") if global_trend else None,
            "html_benchmark_trend_id": html_trend.get("id") if html_trend else None,
            "metrics": dict(metrics),
            "scores": dict(scores),
            "remediation_steps": steps,
            "reuse_policy": (
                "Inject this knowledge into future foundation planning whenever a similar "
                "goal, quality gate, benchmark, browser, evidence, or consensus query appears."
            ),
        }
        content = (
            f"Global evaluation failed because: {blocker_text}\n"
            "Required remediation steps:\n"
            + "\n".join(f"- {step}" for step in steps)
            + "\nEvidence and scores:\n"
            + _json(
                {
                    "benchmark_id": benchmark_id,
                    "html_benchmark_id": data["html_benchmark_id"],
                    "benchmark_trend_id": data["benchmark_trend_id"],
                    "html_benchmark_trend_id": data["html_benchmark_trend_id"],
                    "scores": dict(scores),
                    "metrics": dict(metrics),
                }
            )
        )
        try:
            entry = ProjectBrain(self.store, self.run_id).record_knowledge(
                f"Global remediation: {title_seed}",
                content,
                data=data,
                confidence=0.82,
                evidence_refs=tuple(evidence_refs),
                promote=True,
            )
        except (StateStoreError, DomainError, ValueError) as exc:
            return {
                "recorded": False,
                "error": f"{type(exc).__name__}: {exc}",
                "blocker": blocker_text,
                "remediation_steps": steps,
            }
        return {
            "recorded": True,
            "brain_entry_id": entry.id,
            "title": entry.title,
            "blocker": blocker_text,
            "remediation_steps": steps,
            "evidence_refs": tuple(evidence_refs),
        }

    def _record_benchmark_trend_if_possible(
        self,
        *,
        suite_name: str,
        scenario_name: str,
    ) -> Mapping[str, Any] | None:
        history = self.store.list_benchmark_results(
            suite_name=suite_name,
            scenario_name=scenario_name,
            limit=2,
        )
        if len(history) < 2:
            return None
        try:
            trend = record_benchmark_trend(
                self.store,
                suite_name=suite_name,
                scenario_name=scenario_name,
                provider=self.descriptor.provider,
                model=self.descriptor.model,
            )
            learning = learn_from_benchmark_trend(
                self.store,
                trend,
                ultra_run_id=self.run_id,
            )
            return {**dict(trend), "learning": learning}
        except (DomainError, StateStoreError, ValueError):
            return None

    @staticmethod
    def _trend_quality_regression(trend: Mapping[str, Any] | None) -> bool:
        if not trend or str(trend.get("result") or "") != "failed":
            return False
        metrics = trend.get("metrics")
        if not isinstance(metrics, Mapping):
            return False
        for key, value in metrics.items():
            if not str(key).startswith("score_delta:"):
                continue
            try:
                if float(value) < -0.01:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _goal_spec(value: EngineGoalSpec) -> GoalSpecV1:
        return GoalSpecV1(
            objective=value.objective,
            scope=value.in_scope,
            success_criteria=value.success_criteria,
            constraints=value.constraints,
            non_goals=value.out_of_scope,
        )

    @staticmethod
    def _architecture(value: EngineArchitectureSpec) -> ArchitectureSpecV1:
        interfaces: dict[str, Any] = {}
        for index, item in enumerate(value.interfaces, start=1):
            name = str(item.get("name") or item.get("id") or f"interface-{index}")
            interfaces[name] = dict(item)
        return ArchitectureSpecV1(
            summary=value.summary,
            components=value.components,
            interfaces=interfaces,
            decisions=value.decisions,
            constraints=value.invariants,
        )

    def checkpoint_questions(self, goal_spec: EngineGoalSpec) -> None:
        assert self.run_id
        self.store.update_ultra_run(
            self.run_id,
            phase=UltraPhase.GOAL_SPEC,
            status=UltraRunStatus.PAUSED,
            goal_spec=self._goal_spec(goal_spec),
            config={"pending_questions": list(goal_spec.questions)},
        )
        goal = self.store.get_goal(self.goal_id)
        self.store.update_goal_metadata(
            self.goal_id,
            ultra_run_id=self.run_id,
            plan_questions=list(goal_spec.questions),
            plan_answers={},
            waiting_question=(
                str(goal_spec.questions[0].get("question", "")) if goal_spec.questions else ""
            ),
            resume_status=GoalStatus.DISCOVERING.value,
            auto_retryable=False,
        )
        if goal.status != GoalStatus.PAUSED:
            self.store.transition_goal(
                self.goal_id,
                GoalStatus.PAUSED,
                reason="ULTRA goal decisions require user input",
            )

    def _plan_payload(self, master: MasterPlanV1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        used: set[str] = set()
        self.task_ids = {
            module.id: _safe_task_id(module.id, index, used)
            for index, module in enumerate(master.modules, start=1)
        }
        tasks: list[dict[str, Any]] = []
        changes: list[dict[str, Any]] = []
        for index, module in enumerate(master.modules, start=1):
            task_id = self.task_ids[module.id]
            dependencies = [
                self.task_ids[item]
                for item in module.depends_on
                if item in self.task_ids
            ]
            tasks.append(
                {
                    "id": task_id,
                    "title": module.title[:180],
                    "description": module.objective[:4_000],
                    "acceptance_criteria": list(module.acceptance_criteria),
                    "verification": list(module.verification),
                    "depends_on": dependencies,
                    "risk": str(module.metadata.get("risk", "high")).lower()
                    if str(module.metadata.get("risk", "high")).lower()
                    in {"low", "medium", "high", "critical"}
                    else "high",
                    "role": RoleProfile(
                        name="ULTRA module orchestrator",
                        mission=module.objective,
                        expertise=("planning", "implementation", "review", "testing", "integration"),
                        constraints=module.forbidden_changes,
                        deliverables=module.acceptance_criteria,
                        tool_policy={"write_paths": list(module.write_paths)},
                    ).to_dict(),
                    "priority": max(0, len(master.modules) - index),
                    "metadata": {"ultra_node_id": module.id},
                }
            )
            paths = module.write_paths or (".",)
            for path in paths:
                changes.append(
                    {
                        "path": path,
                        "intent": module.objective[:1_000],
                        "supports_tasks": [task_id],
                    }
                )
        return tasks, changes

    def bind_foundation(
        self,
        goal_spec: EngineGoalSpec,
        architecture: EngineArchitectureSpec,
        master: MasterPlanV1,
    ) -> Plan:
        assert self.run_id
        tasks, changes = self._plan_payload(master)
        all_ids = [item["id"] for item in tasks]
        strategy = master.execution_strategy or (
            "Execute dependency-safe module waves. Every node uses isolated planning, "
            "implementation, independent review, tests, bounded fixes, integration, and memory write-back."
        )
        self.plan = self.store.create_plan(
            self.goal_id,
            master.summary,
            tasks,
            applicability_evidence=[
                {
                    "fact": "ULTRA GoalSpec and Architecture were produced after repository inspection.",
                    "source": f"ultra:{self.run_id}:foundation",
                    "supports_tasks": all_ids,
                }
            ],
            execution_strategy=strategy,
            expected_changes=changes,
            proposed_by="ultra-planner",
            submit=True,
        )
        self.store.update_ultra_run(
            self.run_id,
            phase=UltraPhase.AWAITING_APPROVAL,
            status=UltraRunStatus.AWAITING_APPROVAL,
            goal_spec=self._goal_spec(goal_spec),
            architecture_spec=self._architecture(architecture),
            config={
                # Approval is performed against the persisted Plan record, so
                # this is the canonical fingerprint for every approval-bound
                # quality artifact as well.
                "master_plan_fingerprint": self.plan.fingerprint,
                "module_count": len(master.modules),
            },
        )
        brain = ProjectBrain(self.store, self.run_id)
        canonical_goal = self._goal_spec(goal_spec)
        canonical_architecture = self._architecture(architecture)
        brain.set_north_star(canonical_goal)
        brain.set_architecture(canonical_architecture)
        for index, decision in enumerate(architecture.decisions, start=1):
            choice = str(
                decision.get("decision")
                or decision.get("choice")
                or decision.get("summary")
                or _json(decision)
            )
            brain.record_decision(
                str(decision.get("title") or decision.get("name") or f"Architecture decision {index}"),
                choice,
                reason=str(decision.get("reason", "")),
                alternatives=tuple(str(item) for item in decision.get("alternatives", ())),
                status=str(decision.get("status", "accepted")),
            )
        for index, constraint in enumerate(goal_spec.constraints, start=1):
            brain.write(
                BrainSection.CONSTRAINT,
                f"Constraint {index}",
                constraint,
                data={"constraint": constraint, "source": "GoalSpecV1"},
            )
        goal = self.store.get_goal(self.goal_id)
        self.store.update_goal_metadata(
            self.goal_id,
            ultra_run_id=self.run_id,
            plan_questions=[],
            waiting_question="",
            auto_retryable=False,
        )
        if goal.status != GoalStatus.AWAITING_PLAN_APPROVAL:
            self.store.transition_goal(
                self.goal_id,
                GoalStatus.AWAITING_PLAN_APPROVAL,
                reason="ULTRA master plan awaits one user approval",
            )
        return self.plan

    def approve_master(self, master: MasterPlanV1) -> Plan:
        if not self.run_id or not self.plan:
            raise StateStoreError("ULTRA foundation is not bound to a durable master plan")
        accepted, _ = self.store.approve_plan(
            self.goal_id,
            self.plan.revision,
            approved_by="user",
            expected_fingerprint=self.plan.fingerprint,
        )
        self.store.approve_ultra_master(
            self.run_id,
            accepted.revision,
            accepted.fingerprint,
            approved_by="user",
        )
        self.plan = accepted
        self.approved = True
        self._flush_nodes()
        return accepted

    @staticmethod
    def _contract(node: EngineWorkNode, *, root: bool) -> TaskContractV1:
        interfaces = {name: {"owned": True} for name in node.contract.owned_interfaces}
        return TaskContractV1(
            objective=node.contract.objective,
            success_criteria=node.contract.acceptance_criteria,
            write_paths=node.contract.write_paths or ((".",) if root else ()),
            read_paths=(".",),
            forbidden_changes=node.contract.forbidden_changes,
            interfaces=interfaces,
            external_dependencies=tuple(
                str(item)
                for item in node.contract.metadata.get("external_dependencies", ())
            ),
            metadata={
                **dict(node.contract.metadata),
                "verification": list(node.contract.verification),
                "owned_interfaces": list(node.contract.owned_interfaces),
            },
        )

    def _stored_node(self, node: EngineWorkNode) -> WorkNode:
        root = node.parent_id is None and node.kind is NodeKind.MODULE
        result = self._result_cache.get(node.id)
        return WorkNode(
            id=node.id,
            ultra_run_id=self.run_id or "",
            title=node.contract.title,
            objective=node.contract.objective,
            contract=self._contract(node, root=root),
            kind=_store_kind(node.kind),
            status=_store_node_status(node.status),
            parent_id=node.parent_id,
            master_task_id=self.task_ids.get(node.id) if root else None,
            position=node.order,
            depends_on=node.depends_on,
            assigned_role=(node.phase.value if node.phase else "coder"),
            attempts=result.fix_attempts if result else 0,
            max_attempts=self.config.max_fix_attempts,
            result=self._result(result) if result else None,
            error=(result.summary if result and not result.success else None),
            checkpoint=(node.phase.value if node.phase else ""),
            metadata={"children": list(node.children)},
        )

    def _flush_nodes(self) -> None:
        if not self.approved:
            return
        while True:
            progressed = False
            for node_id, node in list(self._pending_nodes.items()):
                if node.parent_id and node.parent_id not in self._persisted_nodes:
                    continue
                if any(dep not in self._persisted_nodes for dep in node.depends_on):
                    continue
                stored = self._stored_node(node)
                self.store.create_work_node(stored)
                self._persisted_nodes.add(node_id)
                self._pending_nodes.pop(node_id, None)
                progressed = True
            if not progressed:
                break
        if not self._pending_nodes:
            self._flush_artifacts()

    def save_work_node(self, run_id: str, node: EngineWorkNode) -> None:
        super().save_work_node(run_id, node)
        with self._adapter_lock:
            self._pending_nodes[node.id] = node
            if node.id not in self._persisted_nodes:
                self._flush_nodes()
                return
            stored = self._stored_node(node)
            current = self.store.update_work_node_definition(
                node.id,
                contract=stored.contract,
                depends_on=stored.depends_on,
                assigned_role=stored.assigned_role,
                checkpoint=stored.checkpoint,
                metadata=stored.metadata,
            )
            target = _store_node_status(node.status)
            result = self._result_cache.get(node.id)
            if current.status != target or result is not None:
                clear_error_states = {
                    WorkNodeStatus.PENDING,
                    WorkNodeStatus.READY,
                    WorkNodeStatus.IN_PROGRESS,
                    WorkNodeStatus.COMPLETED,
                }
                self.store.transition_work_node(
                    node.id,
                    target,
                    result=self._result(result) if result else current.result,
                    error=(
                        result.summary
                        if result
                        and not result.success
                        and target
                        in {
                            WorkNodeStatus.FAILED,
                            WorkNodeStatus.REVISION_REQUIRED,
                            WorkNodeStatus.BLOCKED,
                        }
                        else None
                        if target in clear_error_states
                        else current.error
                    ),
                    checkpoint=node.phase.value if node.phase else current.checkpoint,
                )
            self._pending_nodes.pop(node.id, None)
            self._sync_master_task(node, result)

    @staticmethod
    def _insight(value: Any) -> InsightV1:
        kwargs: dict[str, Any] = {
            "summary": str(getattr(value, "summary", "Insight")),
            "category": str(getattr(value, "source", "agent")),
            "details": _json(getattr(value, "details", {})),
            "severity": str(getattr(value, "severity", "info"))
            if str(getattr(value, "severity", "info")) in {"info", "warning", "error", "critical"}
            else "info",
        }
        if getattr(value, "id", None):
            kwargs["id"] = value.id
        return InsightV1(**kwargs)

    @classmethod
    def _result(cls, value: EngineResult) -> ResultPackageV1:
        changed: list[str] = []
        artifact_uris: list[str] = []
        for item in value.artifacts:
            path = str(item.get("path", "")).strip()
            uri = str(item.get("uri", path)).strip()
            if path:
                changed.append(path)
            if uri:
                artifact_uris.append(uri)
        return ResultPackageV1(
            summary=value.summary or f"{value.node_id} result",
            changed_files=tuple(dict.fromkeys(changed)),
            tests=value.test_results,
            artifacts=tuple(dict.fromkeys(artifact_uris)),
            insights=tuple(cls._insight(item) for item in value.insights),
            issues=value.findings,
            metadata={
                "success": value.success,
                "status": value.status,
                "fix_attempts": value.fix_attempts,
                "evidence": list(value.evidence),
                "component_package": dict(value.component_package),
            },
        )

    def save_result_package(self, run_id: str, result: EngineResult) -> None:
        super().save_result_package(run_id, result)
        with self._adapter_lock:
            self._result_cache[result.node_id] = result
            if result.node_id in self._persisted_nodes:
                current = self.store.get_work_node(result.node_id)
                target = (
                    WorkNodeStatus.COMPLETED
                    if result.success
                    else WorkNodeStatus.REVISION_REQUIRED
                    if result.status == "revision_required"
                    else WorkNodeStatus.FAILED
                )
                self.store.transition_work_node(
                    result.node_id,
                    target,
                    result=self._result(result),
                    error=None if result.success else result.summary,
                )
                node = self.nodes.get(run_id, {}).get(result.node_id)
                if node:
                    self._sync_master_task(node, result)
                if result.node_id not in self._brain_results:
                    ProjectBrain(self.store, run_id).write_back_result(
                        result.node_id,
                        self._result(result),
                    )
                    self._brain_results.add(result.node_id)
            for item in result.artifacts:
                uri = str(item.get("uri") or item.get("path") or "").strip()
                if not uri:
                    continue
                self._queue_artifact(
                    Artifact(
                        ultra_run_id=run_id,
                        work_node_id=(result.node_id if result.node_id != "__global__" else None),
                        kind=str(item.get("kind", "result")),
                        uri=uri,
                        path=str(item.get("path") or "") or None,
                        content_hash=str(item.get("hash") or "") or None,
                        evidence={"result": result.summary},
                    )
                )

    def _master_node(self, node: EngineWorkNode) -> str | None:
        current = node
        seen: set[str] = set()
        while current.parent_id and current.parent_id not in seen:
            seen.add(current.id)
            parent = self.nodes.get(self.run_id or "", {}).get(current.parent_id)
            if parent is None:
                break
            current = parent
        return current.id if current.id in self.task_ids else None

    def master_task_for_node(self, node_id: str | None) -> str | None:
        if not node_id or not self.run_id:
            return None
        node = self.nodes.get(self.run_id, {}).get(node_id)
        root = self._master_node(node) if node else None
        return self.task_ids.get(root or "")

    def _sync_master_task(self, node: EngineWorkNode, result: EngineResult | None) -> None:
        if not self.plan:
            return
        task_id = self.master_task_for_node(node.id)
        if not task_id or self.task_ids.get(node.id) != task_id:
            return
        task = next((item for item in self.store.get_plan(self.goal_id, self.plan.revision).tasks if item.id == task_id), None)
        if task is None:
            return
        if node.status is NodeStatus.RUNNING and task.status in {TaskStatus.PENDING, TaskStatus.READY}:
            self.store.transition_task(
                self.goal_id,
                self.plan.revision,
                task_id,
                TaskStatus.IN_PROGRESS,
                actor="ultra-scheduler",
            )
        elif result and result.success and task.status != TaskStatus.COMPLETED:
            self.store.transition_task(
                self.goal_id,
                self.plan.revision,
                task_id,
                TaskStatus.COMPLETED,
                note=result.summary,
                evidence=(result.summary,),
                actor="ultra-evidence-gate",
            )
        elif result and not result.success and task.status not in {TaskStatus.FAILED, TaskStatus.BLOCKED}:
            self.store.transition_task(
                self.goal_id,
                self.plan.revision,
                task_id,
                TaskStatus.BLOCKED,
                note=result.summary or "ULTRA quality gate failed",
                actor="ultra-quality-gate",
            )

    def _record_swarm_run_update(
        self,
        item: Any,
        previous_registry: AgentRegistryEntryV1 | None = None,
    ) -> None:
        if item.status == "running":
            return
        if previous_registry is not None and previous_registry.state.value == item.status:
            return
        message_type = (
            SwarmMessageType.INFORM
            if item.status == "completed"
            else SwarmMessageType.BLOCKER
        )
        payload = {
            "agent_run_id": item.id,
            "node_id": item.node_id,
            "role": item.role.value,
            "phase": item.phase,
            "status": item.status,
            "summary": item.summary,
            "error": item.error,
            "usage": dict(item.usage),
            "prompt_trace_id": item.prompt_trace_id,
        }
        try:
            self.store.post_swarm_message(
                SwarmMessageV1(
                    ultra_run_id=item.run_id,
                    sender_agent_id=item.id,
                    recipient_agent_id="ultra-orchestrator",
                    message_type=message_type,
                    topic=f"agent_run:{item.node_id or '__global__'}:{item.role.value}:{item.phase}",
                    payload=payload,
                    confidence=1.0 if item.status == "completed" else 0.0,
                    correlation_id=item.node_id or item.phase or item.id,
                )
            )
        except StateStoreError:
            # Agent run persistence is the source of truth; swarm messages are
            # an auditable communication layer and must not make recovery worse.
            return

    def save_agent_run(self, item: Any) -> None:
        super().save_agent_run(item)
        with self._adapter_lock:
            status = {
                "running": AgentRunStatus.RUNNING,
                "completed": AgentRunStatus.COMPLETED,
                "failed": AgentRunStatus.FAILED,
                "cancelled": AgentRunStatus.CANCELLED,
                "rate_limited": AgentRunStatus.RATE_LIMITED,
                "uncertain": AgentRunStatus.UNCERTAIN,
            }.get(item.status, AgentRunStatus.FAILED)
            registry_state = {
                "running": AgentState.RUNNING,
                "completed": AgentState.COMPLETED,
                "failed": AgentState.FAILED,
                "cancelled": AgentState.CANCELLED,
                "rate_limited": AgentState.BLOCKED,
                "uncertain": AgentState.BLOCKED,
            }.get(item.status, AgentState.FAILED)
            existing_registry = {
                entry.runtime_id: entry
                for entry in self.store.list_agent_registry(item.run_id)
            }
            previous_registry = existing_registry.get(item.id)
            self.store.save_agent_registry_entry(
                AgentRegistryEntryV1(
                    runtime_id=item.id,
                    ultra_run_id=item.run_id,
                    display_index=(
                        previous_registry.display_index
                        if previous_registry is not None
                        else len(existing_registry) + 1
                    ),
                    role=item.role.value,
                    assigned_id=item.node_id,
                    state=registry_state,
                    provider=item.provider or self.descriptor.provider,
                    model=item.model or self.descriptor.model,
                    prompt_trace_refs=(item.prompt_trace_id,) if item.prompt_trace_id else (),
                    failure_reason=(item.error or None) if registry_state is AgentState.FAILED else None,
                    blocker=(item.error or None) if registry_state is AgentState.BLOCKED else None,
                    usage=item.usage,
                    started_at=(previous_registry.started_at if previous_registry else utc_now()),
                    ended_at=utc_now() if registry_state in {AgentState.COMPLETED, AgentState.FAILED, AgentState.CANCELLED} else None,
                )
            )
            self._record_swarm_run_update(item, previous_registry)
            if item.status == "completed":
                change_sets = self.store.list_change_sets(item.run_id)
                if item.role in {AgentRole.CODER, AgentRole.INTEGRATOR}:
                    for change_set in change_sets:
                        if (
                            change_set.responsible_agent_id == item.id
                            and change_set.status is ChangeSetStatus.OPEN
                        ):
                            self.store.save_change_set(
                                replace(change_set, status=ChangeSetStatus.CLOSED, updated_at=utc_now())
                            )
            if item.id in self._persisted_agents:
                self.store.update_agent_run(
                    item.id,
                    status,
                    usage=item.usage,
                    error=item.error or None,
                    prompt_trace_id=item.prompt_trace_id,
                    side_effects=item.role in {AgentRole.CODER, AgentRole.INTEGRATOR},
                )
                return
            self.store.create_agent_run(
                AgentRun(
                    id=item.id,
                    ultra_run_id=item.run_id,
                    work_node_id=item.node_id,
                    role=item.role.value,
                    provider=item.provider or self.descriptor.provider,
                    model=item.model or self.descriptor.model,
                    phase=item.phase,
                    status=status,
                    usage=item.usage,
                    error=item.error or None,
                    prompt_trace_id=item.prompt_trace_id,
                    side_effects=item.role in {AgentRole.CODER, AgentRole.INTEGRATOR},
                )
            )
            self._persisted_agents.add(item.id)

    def record_quality_review(self, node_id: str, category: str, passed: bool) -> None:
        if not self.run_id:
            return
        for change_set in self.store.list_change_sets(self.run_id):
            if change_set.parent_id != node_id or change_set.status not in {
                ChangeSetStatus.CLOSED,
                ChangeSetStatus.REVIEWING,
                ChangeSetStatus.APPROVED,
                ChangeSetStatus.BLOCKED,
            }:
                continue
            reviews = {**dict(change_set.review_status), category: "passed" if passed else "failed"}
            if any(value == "failed" for value in reviews.values()):
                target = ChangeSetStatus.BLOCKED
            elif all(reviews.get(key) == "passed" for key in ("clean_code", "security", "test_quality")):
                target = ChangeSetStatus.APPROVED
            else:
                target = ChangeSetStatus.REVIEWING
            self.store.save_change_set(
                replace(change_set, review_status=reviews, status=target, updated_at=utc_now())
            )
            if target is ChangeSetStatus.APPROVED:
                attempt = 1 + sum(
                    cycle.kind is QualityCycleKind.DELTA
                    for cycle in self.store.list_quality_cycles(self.run_id)
                )
                approach = hashlib.sha256(
                    _json({"change_set": change_set.id, "post_hashes": change_set.post_hashes, "reviews": reviews}).encode("utf-8")
                ).hexdigest()
                cycle = QualityCycleV1(
                    ultra_run_id=self.run_id,
                    kind=QualityCycleKind.DELTA,
                    attempt=attempt,
                    approach_fingerprint=approach,
                    inputs={"change_set_id": change_set.id, "post_hashes": dict(change_set.post_hashes)},
                    outputs={"reviews": reviews},
                    metrics={"changed_files": len(change_set.changed_files)},
                    result="passed",
                    ended_at=utc_now(),
                )
                self.store.save_quality_cycle(cycle)
                brain = ProjectBrain(self.store, self.run_id)
                brain.write(
                    BrainSection.CHANGE_SETS,
                    f"Change Set {change_set.id}",
                    f"Fresh clean-code, security, and test-quality reviews passed for {len(change_set.changed_files)} file(s).",
                    data={"change_set_id": change_set.id, "review_status": reviews, "cycle_id": cycle.id},
                )

    def record_quality_findings(
        self,
        node_id: str,
        category: str,
        records: Iterable[Mapping[str, Any]],
    ) -> None:
        if not self.run_id:
            return
        for record in records:
            path = str(record.get("path", "")).strip()
            file_hash = str(record.get("file_hash", "")).strip()
            principle = str(record.get("principle_id", "")).strip().lower()
            evidence = record.get("evidence", {})
            if not path or not file_hash or not principle or not isinstance(evidence, Mapping):
                continue  # unconfirmed prose is not a durable finding
            try:
                finding = QualityFindingV1(
                    ultra_run_id=self.run_id,
                    principle_id=principle,
                    category=QualityCategory(category),
                    severity=FindingSeverity(str(record.get("severity", "medium")).lower()),
                    path=path,
                    location=str(record.get("location", "")),
                    file_hash=file_hash,
                    evidence=dict(evidence),
                    remediation=str(record.get("remediation", "")).strip(),
                    acceptance_criteria=tuple(str(value) for value in record.get("acceptance_criteria", ()) if str(value).strip()),
                    verification=tuple(str(value) for value in record.get("verification", ()) if str(value).strip()),
                    repair_node_id=str(record.get("repair_node_id") or "") or None,
                )
            except (ValueError, DomainError):
                continue
            stored = self.store.put_quality_finding(finding)
            ProjectBrain(self.store, self.run_id).write(
                BrainSection.QUALITY_FINDINGS,
                f"Finding {stored.fingerprint[:12]}",
                f"{stored.severity.value} {stored.principle_id} finding at {stored.path}:{stored.location}",
                data={
                    "finding_id": stored.id,
                    "fingerprint": stored.fingerprint,
                    "status": stored.status.value,
                    "remediation": stored.remediation,
                },
            )

    def record_quality_consensus(
        self,
        node_id: str,
        votes: Iterable[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if not self.run_id:
            return {}
        vote_items = tuple(dict(item) for item in votes if isinstance(item, Mapping))
        if not vote_items:
            return {}

        def voter_id_for(item: Mapping[str, Any]) -> str:
            return str(item.get("voter_agent_id") or item.get("role") or "unknown")

        decisive = [
            item for item in vote_items if str(item.get("verdict", "")).casefold() in {"accept", "reject"}
        ]
        voter_ids = tuple(dict.fromkeys(voter_id_for(item) for item in vote_items))
        decisive_voter_ids = tuple(dict.fromkeys(voter_id_for(item) for item in decisive))
        topic = f"quality-gate:{node_id}"
        quorum = max(1, len(decisive_voter_ids) or len(voter_ids))
        coordinator = SwarmCoordinator(self.store)
        workflow = coordinator.propose(
            ultra_run_id=self.run_id,
            proposer_agent_id="ultra-orchestrator",
            topic=topic,
            proposal={
                "gate": "ultra-quality-consensus",
                "node_id": node_id,
                "vote_count": len(vote_items),
                "decisive_vote_count": len(decisive),
                "quorum": quorum,
            },
            voters=voter_ids,
            quorum=quorum,
            leader_agent_id="ultra-orchestrator",
        )
        current: Mapping[str, Any] = self.store.get_consensus_round(workflow.consensus_round_id)

        ordered_votes = tuple(item for item in vote_items if item not in decisive) + tuple(decisive)
        for item in ordered_votes:
            if current.get("status") in {"accepted", "rejected", "tied"}:
                break
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            current = coordinator.submit_vote(
                round_id=workflow.consensus_round_id,
                voter_agent_id=voter_id_for(item),
                verdict=str(item.get("verdict") or "abstain"),
                confidence=max(0.0, min(1.0, confidence)),
                rationale=str(item.get("rationale") or item.get("summary") or "")[:2_000],
                evidence={
                    "node_id": node_id,
                    "role": item.get("role"),
                    "passed": item.get("passed"),
                    **(
                        dict(item.get("evidence", {}))
                        if isinstance(item.get("evidence"), Mapping)
                        else {}
                    ),
                },
            )
        if current.get("status") in {"accepted", "rejected", "tied"}:
            self.store.post_swarm_message(
                SwarmMessageV1(
                    ultra_run_id=self.run_id,
                    sender_agent_id="ultra-orchestrator",
                    recipient_agent_id="ultra-orchestrator",
                    message_type=SwarmMessageType.DECISION,
                    topic=topic,
                    payload={
                        "consensus_round_id": current["id"],
                        "status": current["status"],
                        "decision": current.get("decision", {}),
                        "votes": current.get("votes", ()),
                        "swarm_workflow": {
                            "proposal_message_id": workflow.proposal_message_id,
                            "request_message_ids": workflow.request_message_ids,
                            "leader_agent_id": workflow.leader_agent_id,
                            "voter_agent_ids": workflow.voter_agent_ids,
                        },
                    },
                    confidence=1.0 if current.get("status") == "accepted" else 0.0,
                    correlation_id=node_id,
                )
            )
        return current

    def record_global_evaluation_gate(
        self,
        global_result: EngineResult,
        node_results: Iterable[EngineResult],
    ) -> Mapping[str, Any]:
        if not self.run_id:
            return {}
        nodes = tuple(node_results)
        agents = self.store.list_agent_runs(self.run_id)
        consensus_rounds = self.store.list_consensus_rounds(
            self.run_id,
            topic_prefix="quality-gate:",
        )

        def usage_total(*keys: str) -> float:
            total = 0.0
            for agent in agents:
                for key in keys:
                    try:
                        total += float(agent.usage.get(key, 0) or 0)
                    except (TypeError, ValueError):
                        continue
            return total

        accepted = sum(1 for item in consensus_rounds if item.get("status") == "accepted")
        rejected = sum(1 for item in consensus_rounds if item.get("status") == "rejected")
        tied = sum(1 for item in consensus_rounds if item.get("status") == "tied")
        open_rounds = sum(1 for item in consensus_rounds if item.get("status") == "open")
        node_successes = sum(1 for item in nodes if item.success)
        final_evidence = len(global_result.evidence)
        final_tests = len(global_result.test_results)
        metrics = {
            "agent_runs": float(len(agents)),
            "completed_agent_runs": float(sum(1 for item in agents if item.status is AgentRunStatus.COMPLETED)),
            "input_tokens": usage_total("input_tokens", "prompt_tokens"),
            "output_tokens": usage_total("output_tokens", "completion_tokens"),
            "cached_tokens": usage_total("cached_tokens"),
            "node_results": float(len(nodes)),
            "node_successes": float(node_successes),
            "final_evidence_items": float(final_evidence),
            "final_test_results": float(final_tests),
            "quality_consensus_rounds": float(len(consensus_rounds)),
            "quality_consensus_accepted": float(accepted),
            "quality_consensus_rejected": float(rejected),
            "quality_consensus_tied": float(tied),
            "quality_consensus_open": float(open_rounds),
        }
        scores = {
            "node_success_ratio": (node_successes / len(nodes)) if nodes else 0.0,
            "consensus_accept_ratio": (accepted / len(consensus_rounds)) if consensus_rounds else 1.0,
            "final_evidence_score": 1.0 if (final_evidence or final_tests) else 0.0,
            "global_success": 1.0 if global_result.success else 0.0,
        }
        blocker = ""
        if not global_result.success:
            blocker = "global integration/review/final evidence did not pass"
        elif nodes and node_successes != len(nodes):
            blocker = "not every module result succeeded"
        elif rejected or tied or open_rounds:
            blocker = "quality consensus is not unanimously accepted"
        elif not final_evidence and not final_tests:
            blocker = "final evidence gate produced no durable evidence or test results"
        html_benchmark: Mapping[str, Any] | None = self._record_html_benchmark_if_applicable(global_result, nodes)
        if html_benchmark and html_benchmark.get("result") != "passed":
            html_blocker = str(html_benchmark.get("blocker") or "single-file 3D HTML benchmark failed")
            blocker = blocker or html_blocker
            scores["html_3d_overall"] = float(html_benchmark.get("scores", {}).get("overall", 0.0))
            metrics["html_3d_benchmark_ran"] = 1.0
        elif html_benchmark:
            scores["html_3d_overall"] = float(html_benchmark.get("scores", {}).get("overall", 0.0))
            metrics["html_3d_benchmark_ran"] = 1.0
        html_trend = (
            self._record_benchmark_trend_if_possible(
                suite_name="weak-model-html",
                scenario_name="threejs-single-file",
            )
            if html_benchmark
            else None
        )
        if self._trend_quality_regression(html_trend):
            blocker = blocker or "HTML benchmark quality regressed against the previous baseline"
        visual_evaluations = self.store.list_visual_evaluations(self.run_id)
        materialized_packages = tuple(
            item
            for item in self.store.list_component_packages(self.run_id)
            if str(item.get("schema_name")) == "MaterializedComponentPackageV2"
        )
        latest_materialized: dict[str, Mapping[str, Any]] = {}
        for item in materialized_packages:
            node_id = str(item.get("work_node_id"))
            if (
                node_id not in latest_materialized
                or int(item.get("version") or 0)
                > int(latest_materialized[node_id].get("version") or 0)
            ):
                latest_materialized[node_id] = item
        accepted_visual_contexts = 0
        visual_score_values: list[float] = []
        visual_critical_findings = 0
        for package in latest_materialized.values():
            package_evaluations = [
                dict(item.get("verdict") or {})
                for item in visual_evaluations
                if str(item.get("package_id")) == str(package.get("id"))
            ]
            accepted_visual_contexts += sum(
                1 for value in package_evaluations if bool(value.get("accepted"))
            )
            for value in package_evaluations:
                visual_critical_findings += int(value.get("critical_findings") or 0)
                visual_score_values.extend(
                    float(score)
                    for score in dict(value.get("scores") or {}).values()
                )
            if len(package_evaluations) < 2 or not all(
                bool(value.get("accepted"))
                for value in package_evaluations[-2:]
            ):
                blocker = blocker or (
                    f"component {package.get('work_node_id')} lacks two clean-context "
                    "visual acceptances"
                )
        if html_benchmark and not latest_materialized:
            blocker = blocker or "USER_REVIEW_REQUIRED: no materialized visual packages were evaluated"
        if visual_critical_findings:
            blocker = blocker or "independent visual judge reported critical findings"
        metrics["independent_visual_evaluations"] = float(len(visual_evaluations))
        metrics["accepted_visual_contexts"] = float(accepted_visual_contexts)
        metrics["visual_critical_findings"] = float(visual_critical_findings)
        metrics["heuristic_visual_metrics_are_anomaly_only"] = 1.0
        weighted_dimensions = [
            scores["consensus_accept_ratio"],
            scores["final_evidence_score"],
            scores["global_success"],
        ]
        # A final-only evaluation has no component population to average. Keep
        # the metric at zero for trend visibility, but do not punish the score
        # for an inapplicable dimension.
        if nodes:
            weighted_dimensions.insert(0, scores["node_success_ratio"])
        if html_benchmark:
            html_scores = dict(html_benchmark.get("scores", {}))
            # Legacy HTML metrics remain useful as blank/runtime anomaly checks,
            # but cannot award visual acceptance.
            scores["legacy_heuristic_html_overall"] = float(
                html_scores.get("overall", 0.0)
            )
            visual_score = (
                min(visual_score_values)
                if visual_score_values
                else 0.0
            )
            scores["visual_critical"] = visual_score
            if visual_score < 0.85:
                blocker = blocker or "independent critical visual quality score is below 0.85"
        scores["overall"] = sum(weighted_dimensions) / max(1, len(weighted_dimensions))
        if scores["overall"] < 0.95:
            blocker = blocker or "overall quality score is below 0.95"
        passed = not blocker
        artifact_refs: list[str] = []
        for artifact in global_result.artifacts:
            if isinstance(artifact, Mapping):
                artifact_refs.append(str(artifact.get("uri") or artifact.get("path") or artifact))
            else:
                artifact_refs.append(str(artifact))
        recorded = self.store.record_benchmark_result(
            suite_name="ultra-automatic-evaluation",
            scenario_name="global-completion-gate",
            provider=self.descriptor.provider,
            model=self.descriptor.model,
            ultra_run_id=self.run_id,
            inputs={
                "global_result_status": global_result.status,
                "node_ids": [item.node_id for item in nodes],
                "evaluation_authority": "materialized_v9",
                "legacy_html_authority": "legacy_heuristic_anomaly_only",
            },
            metrics=metrics,
            scores=scores,
            result="passed" if passed else "failed",
            artifact_refs=artifact_refs,
            blocker=blocker or None,
        )
        global_trend = self._record_benchmark_trend_if_possible(
            suite_name="ultra-automatic-evaluation",
            scenario_name="global-completion-gate",
        )
        if passed and self._trend_quality_regression(global_trend):
            blocker = "global evaluation quality regressed against the previous baseline"
            passed = False
        lesson_outcomes = self._record_project_lesson_evaluation_outcomes(
            passed=passed,
            benchmark_id=str(recorded["id"]),
            html_benchmark_id=str(html_benchmark.get("id")) if html_benchmark else None,
            blocker=blocker,
        )
        global_lesson_outcomes = self._record_global_lesson_evaluation_outcomes(
            passed=passed,
            benchmark_id=str(recorded["id"]),
            blocker=blocker,
            html_benchmark=html_benchmark,
        )
        remediation_knowledge = self._record_global_remediation_knowledge(
            passed=passed,
            benchmark_id=str(recorded["id"]),
            blocker=blocker,
            metrics=metrics,
            scores=scores,
            html_benchmark=html_benchmark,
            global_trend=global_trend,
            html_trend=html_trend,
        )
        return {
            "passed": passed,
            "metrics": metrics,
            "scores": scores,
            "benchmark_id": recorded["id"],
            "html_benchmark_id": html_benchmark.get("id") if html_benchmark else None,
            "benchmark_trend_id": global_trend.get("id") if global_trend else None,
            "html_benchmark_trend_id": html_trend.get("id") if html_trend else None,
            "benchmark_trend_learning": global_trend.get("learning") if global_trend else None,
            "html_benchmark_trend_learning": html_trend.get("learning") if html_trend else None,
            "blocker": blocker,
            "project_lesson_outcomes": lesson_outcomes,
            "global_lesson_outcomes": global_lesson_outcomes,
            "remediation_knowledge": remediation_knowledge,
        }

    def _record_html_benchmark_if_applicable(
        self,
        global_result: EngineResult,
        node_results: Iterable[EngineResult],
    ) -> Mapping[str, Any] | None:
        if not self.run_id or self.workspace is None:
            return None
        candidates: list[str] = []
        for artifact in global_result.artifacts:
            if isinstance(artifact, Mapping):
                candidates.extend(str(artifact.get(key) or "") for key in ("path", "uri"))
            else:
                candidates.append(str(artifact))
        for result in node_results:
            for artifact in result.artifacts:
                if isinstance(artifact, Mapping):
                    candidates.extend(str(artifact.get(key) or "") for key in ("path", "uri"))
                else:
                    candidates.append(str(artifact))
        prompt = str(self.store.get_ultra_run(self.run_id).config.get("prompt", ""))
        should_check = any(value.casefold().endswith((".html", ".htm")) or "index.html" in value.casefold() for value in candidates)
        should_check = should_check or any(term in prompt.casefold() for term in ("html", "browser game", "three.js", "threejs", "3d game", "single-file"))
        index_path = (self.workspace / "index.html").resolve(strict=False)
        try:
            index_path.relative_to(self.workspace)
        except ValueError:
            return None
        if not should_check and not index_path.exists():
            return None
        if not index_path.is_file():
            recorded = self.store.record_benchmark_result(
                suite_name="weak-model-html",
                scenario_name="threejs-single-file",
                provider=self.descriptor.provider,
                model=self.descriptor.model,
                ultra_run_id=self.run_id,
                inputs={"artifact_hash": ""},
                metrics={"missing_index_html": 1.0},
                scores={"overall": 0.0},
                result="failed",
                artifact_refs=("workspace:index.html",),
                blocker="HTML benchmark target index.html was not created",
            )
            return recorded
        try:
            html = index_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            recorded = self.store.record_benchmark_result(
                suite_name="weak-model-html",
                scenario_name="threejs-single-file",
                provider=self.descriptor.provider,
                model=self.descriptor.model,
                ultra_run_id=self.run_id,
                inputs={"artifact_hash": ""},
                metrics={"read_error": 1.0},
                scores={"overall": 0.0},
                result="failed",
                artifact_refs=("workspace:index.html",),
                blocker=f"HTML benchmark could not read index.html: {type(exc).__name__}",
            )
            return recorded
        preview: Mapping[str, Any] = {}
        preview_id = ""
        try:
            preview_raw = tools.run_tool(
                "preview_html",
                {
                    "path": "index.html",
                    "open_browser": False,
                    "verify": True,
                    "settle_ms": 2500,
                },
            )
            parsed_preview = json.loads(preview_raw)
            if isinstance(parsed_preview, Mapping):
                preview = dict(parsed_preview)
                preview_id = str(preview.get("preview_id") or "")
        except (DomainError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            preview = {
                "status": "failed",
                "verification": "failed",
                "page_errors": [f"benchmark preview failed: {type(exc).__name__}: {exc}"],
            }
        finally:
            if preview_id:
                try:
                    tools.run_tool("stop_preview", {"preview_id": preview_id})
                except (DomainError, OSError, RuntimeError, ValueError):
                    pass
        return record_single_file_3d_html_benchmark(
            self.store,
            html,
            provider=self.descriptor.provider,
            model=self.descriptor.model,
            ultra_run_id=self.run_id,
            artifact_ref="workspace:index.html",
            preview=preview,
        )

    def save_prompt_trace(self, trace: EnginePromptTrace) -> None:
        super().save_prompt_trace(trace)
        with self._adapter_lock:
            if trace.id in self._persisted_traces:
                return
            self.store.add_prompt_trace(
                PromptTraceV1(
                    id=trace.id,
                    ultra_run_id=trace.run_id,
                    work_node_id=trace.node_id,
                    agent_run_id=trace.agent_run_id,
                    role=trace.role.value,
                    system_prompt=trace.system_prompt,
                    context_package=trace.context_package,
                    self_prompt=trace.self_prompt,
                    reasoning_summary=trace.reasoning_summary,
                    omitted_sections=trace.omitted_context,
                    redacted=True,
                    metadata={
                        "phase": trace.phase,
                        "chain_of_thought": "not stored",
                    },
                ),
                max_bytes=self.config.prompt_trace_chars,
            )
            self._persisted_traces.add(trace.id)

    @staticmethod
    def _brain_section(section: EngineBrainSection) -> BrainSection:
        return {
            EngineBrainSection.NORTH_STAR: BrainSection.NORTH_STAR,
            EngineBrainSection.ARCHITECTURE: BrainSection.ARCHITECTURE,
            EngineBrainSection.DECISION: BrainSection.DECISION,
            EngineBrainSection.CONSTRAINT: BrainSection.CONSTRAINT,
            EngineBrainSection.TASK_GRAPH: BrainSection.TASK_GRAPH,
            EngineBrainSection.ARTIFACT: BrainSection.ARTIFACT_INDEX,
            EngineBrainSection.KNOWLEDGE: BrainSection.KNOWLEDGE,
            EngineBrainSection.LESSON: BrainSection.LESSON,
            EngineBrainSection.ROLE_MEMORY: BrainSection.ROLE_MEMORY,
        }.get(section, BrainSection.KNOWLEDGE)

    def append_brain_entry(self, entry: BrainEntryV1) -> None:
        super().append_brain_entry(entry)
        role = entry.role.value if entry.role else None
        stored = self.store.put_brain_entry(
            BrainEntry(
                ultra_run_id=entry.run_id,
                goal_id=self.goal_id,
                work_node_id=entry.node_id,
                section=self._brain_section(entry.section),
                title=entry.key,
                content=_json(entry.value),
                data=entry.value,
                role=role,
                expires_at=entry.expires_at,
                metadata={"engine_section": entry.section.value},
            )
        )
        self.store.record_memory_access(
            entry.run_id,
            direction="write",
            work_node_id=entry.node_id,
            brain_entry_id=stored.id,
            query=entry.key,
            metadata={"section": stored.section.value, "role": role or ""},
        )

    def list_brain_entries(self, run_id: str) -> tuple[BrainEntryV1, ...]:
        live = super().list_brain_entries(run_id)
        result: list[BrainEntryV1] = []
        for item in self.store.list_brain_entries(run_id, latest_only=True):
            engine_name = str(item.metadata.get("engine_section", item.section.value))
            try:
                section = EngineBrainSection(engine_name)
            except ValueError:
                section = EngineBrainSection.KNOWLEDGE
            try:
                role = AgentRole(item.role) if item.role else None
            except ValueError:
                role = None
            result.append(
                BrainEntryV1(
                    section=section,
                    key=item.title,
                    value=dict(item.data),
                    run_id=run_id,
                    node_id=item.work_node_id,
                    role=role,
                    version=item.version,
                    expires_at=item.expires_at,
                    created_at=item.created_at,
                )
            )
        merged: dict[tuple[str, str, str], BrainEntryV1] = {
            (item.section.value, item.key, item.role.value if item.role else ""): item
            for item in result
        }
        for item in live:
            merged[(item.section.value, item.key, item.role.value if item.role else "")] = item
        return tuple(merged.values())

    def _queue_artifact(self, artifact: Artifact) -> None:
        self._pending_artifacts.append(artifact)
        self._flush_artifacts()

    def _flush_artifacts(self) -> None:
        remaining: list[Artifact] = []
        for artifact in self._pending_artifacts:
            if artifact.work_node_id and artifact.work_node_id not in self._persisted_nodes:
                remaining.append(artifact)
                continue
            try:
                stored = self.store.add_artifact(artifact)
            except StateStoreError:
                remaining.append(artifact)
                continue
            try:
                self.store.put_brain_entry(
                    BrainEntry(
                        ultra_run_id=stored.ultra_run_id,
                        goal_id=self.goal_id,
                        work_node_id=stored.work_node_id,
                        agent_run_id=stored.agent_run_id,
                        section=BrainSection.ARTIFACT_INDEX,
                        title=stored.path or stored.uri,
                        content=f"{stored.kind} artifact: {stored.uri}",
                        data={
                            "artifact_id": stored.id,
                            "kind": stored.kind,
                            "uri": stored.uri,
                            "path": stored.path,
                            "content_hash": stored.content_hash,
                            "pre_write_hash": stored.pre_write_hash,
                            "evidence": dict(stored.evidence),
                        },
                        metadata={"source": "artifact_index"},
                    )
                )
            except StateStoreError:
                # The artifacts table remains authoritative if the searchable
                # Project Brain mirror cannot be refreshed.
                pass
        self._pending_artifacts = remaining

    def record_file_artifact(
        self,
        node_id: str | None,
        path: str,
        pre_hash: str | None,
        post_hash: str | None,
        tool_name: str,
    ) -> None:
        if not self.run_id:
            return
        self._queue_artifact(
            Artifact(
                ultra_run_id=self.run_id,
                work_node_id=node_id,
                kind="file",
                uri=f"workspace:{_normalized_path(path)}",
                path=_normalized_path(path),
                content_hash=post_hash,
                pre_write_hash=pre_hash,
                evidence={"tool": tool_name},
            )
        )

    def lease_manager(self, workspace: Path) -> ResourceLeaseManager:
        def acquired(lease: RuntimeLease) -> None:
            if not self.run_id:
                return
            scopes = tuple(_normalized_path(path) for path in lease.paths)
            hashes = _snapshot_write_scopes(workspace, scopes)
            created: list[str] = []
            try:
                for path in lease.paths:
                    row = self.store.acquire_resource_lease(
                        self.run_id,
                        lease.owner,
                        path,
                        pre_write_hash=_hash_file(workspace, path),
                    )
                    created.append(row.id)
            except Exception:
                for lease_id in created:
                    self.store.release_resource_lease(lease_id)
                raise
            with self._adapter_lock:
                self._lease_ids[lease.owner] = created
                self._lease_scopes[lease.owner] = scopes
                self._lease_hashes[lease.owner] = hashes

        def released(lease: RuntimeLease) -> None:
            with self._adapter_lock:
                lease_ids = self._lease_ids.pop(lease.owner, [])
                self._lease_scopes.pop(lease.owner, None)
                self._lease_hashes.pop(lease.owner, None)
            for lease_id in lease_ids:
                self.store.release_resource_lease(lease_id)

        return ResourceLeaseManager(
            lambda path: _hash_file(workspace, path),
            on_acquire=acquired,
            on_release=released,
        )

    def lease_hash(self, owner: str, path: str) -> tuple[bool, str | None]:
        normalized = _normalized_path(path)
        with self._adapter_lock:
            scopes = self._lease_scopes.get(owner, ())
            if not scopes or not _within_scope(normalized, scopes):
                return False, None
            return True, self._lease_hashes.get(owner, {}).get(normalized)

    def advance_lease_hash(self, owner: str, path: str, value: str | None) -> None:
        normalized = _normalized_path(path)
        with self._adapter_lock:
            scopes = self._lease_scopes.get(owner, ())
            if scopes and _within_scope(normalized, scopes):
                self._lease_hashes.setdefault(owner, {})[normalized] = value


class UltraSession:
    """Interactive ULTRA profile owned by one :class:`AgentRuntime`."""

    def __init__(
        self,
        *,
        store: StateStore,
        workspace: Path,
        descriptor: ModelDescriptor,
        permission_adapter: PermissionAdapter,
        approval: Callable[[str, dict[str, Any], str], bool],
        events: EventBus,
        config: UltraConfig,
        agent_steps: int,
        reasoning_effort: str = "medium",
    ) -> None:
        self.store = store
        self.workspace = workspace
        self.descriptor = descriptor
        self.permission_adapter = permission_adapter
        self.approval = approval
        self.events = events
        self.config = config
        self.agent_steps = agent_steps
        self.reasoning_effort = str(reasoning_effort)
        self.goal_id: str | None = None
        self.adapter: StateStoreUltraAdapter | None = None
        self.orchestrator: UltraOrchestrator | None = None
        self.future: Future[UltraRunResult] | None = None
        self.answers: dict[str, str] = {}

    def _workspace_hashes(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for path in self.workspace.rglob("*"):
            if not path.is_file() or ".coding-agent" in path.parts:
                continue
            relative = path.relative_to(self.workspace).as_posix()
            try:
                values[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
        return values

    @property
    def running(self) -> bool:
        return bool(self.future and not self.future.done())

    def wait(self) -> UltraRunResult | None:
        """Wait for a live approved ULTRA run instead of closing its process."""
        return self.future.result() if self.future is not None else None

    @property
    def safe_for_reconfiguration(self) -> bool:
        if not self.running:
            return True
        if not self.orchestrator or not self.orchestrator.control.paused or not self.run_id:
            return False
        return not any(
            item.status is AgentRunStatus.RUNNING
            for item in self.store.list_agent_runs(self.run_id)
        )

    @property
    def run_id(self) -> str | None:
        return self.adapter.run_id if self.adapter else None

    def _node(self, node_id: str | None) -> EngineWorkNode | None:
        if not node_id or not self.orchestrator:
            return None
        return self.orchestrator.nodes.get(node_id)

    def _execute_tool(self, call: ToolCall, request: AgentRequest) -> str:
        allowed = WorkspaceUltraAgent(
            None,
            role=request.role,
            provider_name=self.descriptor.provider,
            model=self.descriptor.model,
            executor=lambda _call, _request: "",
            events=self.events,
        )._allowed_tools()
        if call.name not in allowed:
            return f"Error: role {request.role.value} cannot use {call.name}"
        args = call.args if isinstance(call.args, dict) else {}
        node = self._node(request.node_id)
        if call.name == "stage_component_file":
            if (
                node is None
                or not node.contract.metadata.get("component_package_only")
                or self.adapter is None
                or self.run_id is None
            ):
                return "Error: stage_component_file requires an active component specialist node"
            try:
                result = self.adapter.stage_component_file_tool(
                    self.run_id,
                    node,
                    path=str(args.get("path", "")),
                    content=str(args.get("content", "")),
                    role=str(args.get("role", "")),
                )
            except Exception as exc:
                rendered = f"Error: stage_component_file rejected the file: {exc}"
            else:
                rendered = _json({"status": "staged", **dict(result)})
            self.events.publish(
                "tool_result",
                rendered,
                tool=call.name,
                actor=request.role.value,
                node_id=request.node_id,
            )
            return rendered
        if call.name == "publish_component":
            if (
                node is None
                or not node.contract.metadata.get("component_package_only")
                or self.adapter is None
                or self.run_id is None
            ):
                return "Error: publish_component requires an active component specialist node"
            self.events.publish(
                "tool_call",
                call.name,
                args={"file_count": len(dict(args.get("implementation") or {}).get("files", ()))},
                actor=request.role.value,
                node_id=request.node_id,
            )
            try:
                result = self.adapter.publish_component_tool(
                    self.run_id,
                    node,
                    args,
                )
            except Exception as exc:
                rendered = f"Error: publish_component rejected the package: {exc}"
            else:
                rendered = _json(
                    {
                        "status": result.get("status"),
                        "passed": result.get("passed"),
                        "package_id": dict(result.get("package") or {}).get("id"),
                        "preview": result.get("preview"),
                        "findings": result.get("findings", ()),
                    }
                )
            self.events.publish(
                "tool_result",
                rendered,
                tool=call.name,
                actor=request.role.value,
                node_id=request.node_id,
            )
            return rendered
        if call.name == "apply_patch":
            patch_paths = [
                match.group(1).strip()
                for match in re.finditer(
                    r"(?m)^\+\+\+\s+(?:b/)?([^\t\r\n]+)", str(args.get("patch", ""))
                )
                if match.group(1).strip() != "/dev/null"
            ]
            scopes = node.write_paths if node else ()
            if not patch_paths or not scopes or any(not _within_scope(path, scopes) for path in patch_paths):
                return "Error: apply_patch contains a path outside this node's approved write scope"
        if call.name in {"write_file", "edit_file", "materialize_artifact"}:
            path = str(args.get("path", ""))
            scopes = node.write_paths if node else ()
            if not scopes or not _within_scope(path, scopes):
                return (
                    f"Error: path {path!r} is outside this node's approved write scope; "
                    "a master-plan revision is required"
                )
            normalized = _normalized_path(path)
            expected_known = False
            expected: str | None = None
            if node:
                for raw_path, value in node.pre_write_hashes.items():
                    if _normalized_path(raw_path) == normalized:
                        expected_known, expected = True, value
                        break
            if not expected_known and self.adapter and request.node_id:
                expected_known, expected = self.adapter.lease_hash(request.node_id, normalized)
            current = _hash_file(self.workspace, path)
            if expected_known and current != expected:
                raise StaleWriteError(
                    f"pre-write hash changed for {path!r}: expected {expected!r}, got {current!r}"
                )
        risk = _TOOL_RISK.get(call.name, "unknown")
        normal_requirement = tools.requires_approval(call.name, args)
        needs_approval = self.permission_adapter.requires_approval(normal_requirement)
        self.events.publish(
            "tool_call",
            call.name,
            args=redact_data(args),
            actor=request.role.value,
            node_id=request.node_id,
        )
        task_id = self.adapter.master_task_for_node(request.node_id) if self.adapter else None
        action_id = self.store.begin_action(
            self.goal_id or "",
            call.name,
            {
                "arguments": redact_data(args),
                "ultra_run_id": self.run_id,
                "node_id": request.node_id,
                "role": request.role.value,
                "phase": request.phase,
            },
            task_id=task_id,
            risk=risk,
            mutating=call.name in _WRITE_TOOLS,
        )
        if needs_approval and not self.approval(call.name, dict(args), risk):
            result = "Permission denied by the user. Do not repeat the same action."
            self.store.complete_action(action_id, result, status="denied")
            self.events.publish(
                "tool_result",
                result,
                tool=call.name,
                actor=request.role.value,
                node_id=request.node_id,
            )
            return result
        path = str(args.get("path", "")) if call.name in {"write_file", "edit_file", "materialize_artifact"} else ""
        pre_hash = _hash_file(self.workspace, path) if path else None
        pre_text = ""
        if path:
            try:
                pre_text = (self.workspace / path).read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                pre_text = ""
        before_workspace = self._workspace_hashes() if call.name in _WRITE_TOOLS and not path else {}
        try:
            with tools.workspace_context(self.workspace):
                if call.name in {"run_bash", "run_command"}:
                    assert self.orchestrator
                    with self.orchestrator.scheduler.leases.mutating_shell(request.node_id or request.role.value):
                        shell_command = str(args.get("command", ""))
                        if (
                            call.name == "run_command"
                            and str(args.get("cwd", ".")).strip() not in {"", "."}
                            and self.permission_adapter.access_level.value == "full"
                        ):
                            shell_command = f"cd -- {shlex.quote(str(args['cwd']))} && {shell_command}"
                        result = self.permission_adapter.run_shell(
                            shell_command,
                            self.workspace,
                            normal_runner=lambda command: tools.run_tool(
                                call.name, {**args, "command": command}
                            ),
                        )
                else:
                    result = tools.run_tool(call.name, args)
            result = redact_text(result, 50_000)
            status = "failed" if result.startswith("Error:") else "completed"
            self.store.complete_action(action_id, redact_text(result, 2_000), status=status)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            result = f"Error: ULTRA tool harness failure: {type(exc).__name__}: {redact_text(exc, 500)}"
            self.store.complete_action(action_id, result, status="failed")
        if path and not result.startswith("Error:") and self.adapter:
            post_hash = _hash_file(self.workspace, path)
            self.adapter.record_file_artifact(
                request.node_id,
                path,
                pre_hash,
                post_hash,
                call.name,
            )
            if request.node_id:
                self.adapter.advance_lease_hash(request.node_id, path, post_hash)
        if call.name in _WRITE_TOOLS and not result.startswith("Error:") and self.run_id:
            after_workspace = self._workspace_hashes()
            if path:
                current_post_hash = _hash_file(self.workspace, path)
                changed_files = (path,) if pre_hash != current_post_hash else ()
                pre_hashes = {path: pre_hash}
                post_hashes = {path: current_post_hash}
                try:
                    post_text = (self.workspace / path).read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    post_text = ""
                diff = "".join(
                    difflib.unified_diff(
                        pre_text.splitlines(keepends=True),
                        post_text.splitlines(keepends=True),
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}",
                    )
                )
                shell_created: tuple[str, ...] = ()
            else:
                changed_files = tuple(
                    sorted(
                        key
                        for key in set(before_workspace) | set(after_workspace)
                        if before_workspace.get(key) != after_workspace.get(key)
                    )
                )
                pre_hashes = {key: before_workspace.get(key) for key in changed_files}
                post_hashes = {key: after_workspace.get(key) for key in changed_files}
                shell_created = tuple(key for key in changed_files if key not in before_workspace)
                diff = "\n".join(
                    f"{key}: {pre_hashes[key] or '<absent>'} -> {post_hashes[key] or '<deleted>'}"
                    for key in changed_files
                )
            if not changed_files:
                self.events.publish(
                    "tool_result", result, tool=call.name,
                    actor=request.role.value, node_id=request.node_id,
                )
                return result
            responsible = request.agent_run_id or f"{request.role.value}:{request.node_id or request.phase}"
            existing = next(
                (
                    item for item in reversed(self.store.list_change_sets(self.run_id))
                    if item.responsible_agent_id == responsible and item.status is ChangeSetStatus.OPEN
                ),
                None,
            )
            change_set = existing or ChangeSetV1(
                ultra_run_id=self.run_id,
                responsible_agent_id=responsible,
                parent_id=request.node_id or request.phase,
            )
            combined_files = tuple(dict.fromkeys((*change_set.changed_files, *changed_files)))
            change_set = replace(
                change_set,
                changed_files=combined_files,
                pre_hashes={**dict(pre_hashes), **dict(change_set.pre_hashes)},
                post_hashes={**dict(change_set.post_hashes), **dict(post_hashes)},
                diff=(change_set.diff + ("\n" if change_set.diff and diff else "") + diff),
                mutation_commands=tuple(dict.fromkeys((*change_set.mutation_commands, str(args.get("command", call.name))))),
                shell_created_files=tuple(dict.fromkeys((*change_set.shell_created_files, *shell_created))),
                updated_at=utc_now(),
            )
            self.store.save_change_set(change_set)
            self.store.record_mutation(
                change_set.id,
                call.name,
                path=path or None,
                command=str(args.get("command", "")) or None,
                pre_hash=pre_hash,
                post_hash=_hash_file(self.workspace, path) if path else None,
                metadata={"action_id": action_id, "changed_files": list(changed_files)},
            )
        self.events.publish(
            "tool_result",
            result,
            tool=call.name,
            actor=request.role.value,
            node_id=request.node_id,
        )
        return result

    def start(self, objective: str) -> MasterPlanV1 | None:
        if self.orchestrator and self.orchestrator.phase not in {
            EnginePhase.COMPLETED,
            EnginePhase.CANCELLED,
            EnginePhase.FAILED,
        }:
            raise RuntimeError("an ULTRA run is already active")
        goal = self.store.create_goal(redact_text(objective, 20_000))
        self.store.transition_goal(goal.id, GoalStatus.DISCOVERING, reason="ULTRA foundation started")
        return self._prepare_existing_goal(goal.id, objective)

    def restart_foundation(self, goal_id: str, objective: str) -> MasterPlanV1 | None:
        """Create a fresh ULTRA run/revision while preserving the durable goal."""

        return self._prepare_existing_goal(goal_id, objective)

    def _prepare_existing_goal(self, goal_id: str, objective: str) -> MasterPlanV1 | None:
        self.goal_id = goal_id
        self.answers = {}
        self.adapter = StateStoreUltraAdapter(
            self.store,
            goal_id,
            self.descriptor,
            self.permission_adapter.access_level,
            self.config,
            workspace=self.workspace,
        )
        factory = WorkspaceUltraAgentFactory(
            self.descriptor,
            self._execute_tool,
            self.events,
            max_steps=self.agent_steps,
            reasoning_effort=self.reasoning_effort,
        )
        self.orchestrator = UltraOrchestrator(
            factory,
            execution_class=self.descriptor.execution_class,
            state=self.adapter,
            events=self.events,
            config=self.config,
            context_builder=DurableContextBuilder(
                self.store,
                lambda: self.adapter.run_id if self.adapter else None,
                self.config.context_chars,
            ),
            leases=self.adapter.lease_manager(self.workspace),
            model_snapshot=self.descriptor.to_dict(),
        )
        plan = self.orchestrator.prepare(objective)
        if plan is None:
            assert self.orchestrator.goal_spec
            self.adapter.checkpoint_questions(self.orchestrator.goal_spec)
            return None
        assert self.orchestrator.goal_spec and self.orchestrator.architecture
        self.adapter.bind_foundation(
            self.orchestrator.goal_spec,
            self.orchestrator.architecture,
            plan,
        )
        assert self.adapter.run_id
        durable_run = self.store.get_ultra_run(self.adapter.run_id)
        # The dedicated column is populated only when the user approves the
        # persisted master plan.  The policy baseline is created before that
        # approval boundary, so bind it to the generated (and persisted)
        # fingerprint in run configuration.  Falling back to the in-memory
        # master keeps older stores readable without conflating generation
        # with approval.
        durable_master_fingerprint = (
            durable_run.master_plan_fingerprint
            or str(durable_run.config.get("master_plan_fingerprint", ""))
            or plan.fingerprint
        )
        policy = QualityPolicyV1()
        self.store.save_quality_policy(
            self.adapter.run_id,
            policy,
            master_plan_fingerprint=durable_master_fingerprint,
        )
        inventory = sorted(
            path.relative_to(self.workspace).as_posix()
            for path in self.workspace.rglob("*")
            if path.is_file() and ".coding-agent" not in path.parts
        )
        baseline = QualityCycleV1(
            ultra_run_id=self.adapter.run_id,
            kind=QualityCycleKind.BASELINE,
            attempt=1,
            approach_fingerprint=hashlib.sha256(
                _json({"inventory": inventory, "master_plan": durable_master_fingerprint}).encode("utf-8")
            ).hexdigest(),
            inputs={"inventory": inventory, "master_plan_fingerprint": durable_master_fingerprint},
            outputs={"confirmed_findings": [], "quality_checklist": list(policy.required_reviews)},
            metrics={"project_files": len(inventory)},
            result="clean" if not inventory else "baseline_complete",
            ended_at=utc_now(),
        )
        self.store.save_quality_cycle(baseline)
        brain = ProjectBrain(self.store, self.adapter.run_id)
        brain.write(
            BrainSection.QUALITY_POLICY,
            "Quality Policy V1",
            "Approval-bound Ultra quality policy and completion severities.",
            data=policy.to_dict(),
            metadata={"master_plan_fingerprint": durable_master_fingerprint},
        )
        brain.write(
            BrainSection.QUALITY_CYCLES,
            "Goal-scoped baseline",
            f"Inspected {len(inventory)} project file(s); only confirmed evidence may become findings.",
            data={"cycle_id": baseline.id, "inventory": inventory},
        )
        return plan

    @staticmethod
    def _engine_node_status(status: WorkNodeStatus) -> NodeStatus:
        return {
            WorkNodeStatus.PENDING: NodeStatus.PENDING,
            WorkNodeStatus.READY: NodeStatus.READY,
            WorkNodeStatus.IN_PROGRESS: NodeStatus.RUNNING,
            WorkNodeStatus.REVIEWING: NodeStatus.RUNNING,
            WorkNodeStatus.TESTING: NodeStatus.RUNNING,
            WorkNodeStatus.FIXING: NodeStatus.RUNNING,
            WorkNodeStatus.INTEGRATING: NodeStatus.RUNNING,
            WorkNodeStatus.COMPLETED: NodeStatus.COMPLETED,
            WorkNodeStatus.FAILED: NodeStatus.FAILED,
            WorkNodeStatus.BLOCKED: NodeStatus.BLOCKED,
            WorkNodeStatus.CANCELLED: NodeStatus.CANCELLED,
            WorkNodeStatus.UNCERTAIN: NodeStatus.UNCERTAIN,
            WorkNodeStatus.REVISION_REQUIRED: NodeStatus.REVISION_REQUIRED,
        }[status]

    @staticmethod
    def _engine_result(node_id: str, value: ResultPackageV1) -> EngineResult:
        artifacts = tuple(
            {"path": path, "uri": f"workspace:{path}", "kind": "file"}
            for path in value.changed_files
        ) + tuple({"uri": uri, "kind": "result"} for uri in value.artifacts)
        return EngineResult(
            node_id=node_id,
            success=bool(value.metadata.get("success", True)),
            status=str(value.metadata.get("status", "completed")),
            summary=value.summary,
            artifacts=artifacts,
            evidence=tuple(value.metadata.get("evidence", ())),
            test_results=value.tests,
            findings=value.issues,
            component_package=dict(value.metadata.get("component_package", {})),
            fix_attempts=int(value.metadata.get("fix_attempts", 0) or 0),
        )

    def restore(self, run_id: str) -> Future[UltraRunResult] | Plan:
        """Rebuild the scheduler from durable evidence without replaying uncertainty."""

        run = self.store.get_ultra_run(run_id)
        nodes = self.store.list_work_nodes(run_id)
        uncertain_nodes = [item.id for item in nodes if item.status is WorkNodeStatus.UNCERTAIN]
        uncertain_agents = [
            item.id
            for item in self.store.list_agent_runs(run_id)
            if item.status is AgentRunStatus.UNCERTAIN
        ]
        uncertain_actions = [
            item["id"] for item in self.store.list_actions(run.goal_id, status="uncertain")
        ]
        if uncertain_nodes or uncertain_agents or uncertain_actions:
            values = [*uncertain_nodes, *uncertain_agents, *uncertain_actions]
            raise RuntimeError(
                "reconcile uncertain ULTRA work before resume: " + ", ".join(values[:12])
            )
        awaiting_approval = not run.master_approved or run.plan_revision is None
        if awaiting_approval:
            if run.status is not UltraRunStatus.AWAITING_APPROVAL:
                raise RuntimeError("the interrupted ULTRA run has no approved master plan")
            plan = self.store.get_latest_plan(run.goal_id)
            if plan is None or plan.status is not PlanStatus.PENDING_APPROVAL:
                raise RuntimeError("the interrupted ULTRA run has no pending master plan")
        else:
            plan = self.store.get_plan(run.goal_id, run.plan_revision)
        if run.goal_spec is None or run.architecture_spec is None:
            raise RuntimeError("the interrupted ULTRA foundation is incomplete; use /replan")

        self.goal_id = run.goal_id
        self.adapter = StateStoreUltraAdapter(
            self.store,
            run.goal_id,
            self.descriptor,
            self.permission_adapter.access_level,
            self.config,
            workspace=self.workspace,
        )
        self.adapter.run_id = run.id
        self.adapter.plan = plan
        self.adapter.approved = not awaiting_approval
        self.adapter.task_ids = {
            str(task.metadata.get("ultra_node_id", task.id)): task.id for task in plan.tasks
        }
        self.adapter._persisted_nodes = {item.id for item in nodes}

        factory = WorkspaceUltraAgentFactory(
            self.descriptor,
            self._execute_tool,
            self.events,
            max_steps=self.agent_steps,
            reasoning_effort=self.reasoning_effort,
        )
        self.orchestrator = UltraOrchestrator(
            factory,
            execution_class=self.descriptor.execution_class,
            state=self.adapter,
            events=self.events,
            config=self.config,
            context_builder=DurableContextBuilder(
                self.store,
                lambda: self.adapter.run_id if self.adapter else None,
                self.config.context_chars,
            ),
            leases=self.adapter.lease_manager(self.workspace),
            model_snapshot=self.descriptor.to_dict(),
        )
        goal_spec = EngineGoalSpec(
            objective=run.goal_spec.objective,
            success_criteria=run.goal_spec.success_criteria
            or ("Complete every approved module and final evidence gate.",),
            constraints=run.goal_spec.constraints,
            in_scope=run.goal_spec.scope,
            out_of_scope=run.goal_spec.non_goals,
            assumptions=tuple(
                f"{key}: {value}"
                for key, value in run.goal_spec.answered_questions.items()
            ),
        )
        interface_values = tuple(
            {"name": name, **(dict(value) if isinstance(value, Mapping) else {"contract": value})}
            for name, value in run.architecture_spec.interfaces.items()
        )
        architecture = EngineArchitectureSpec(
            summary=run.architecture_spec.summary,
            components=run.architecture_spec.components or ({"name": "restored-project"},),
            interfaces=interface_values,
            decisions=run.architecture_spec.decisions,
            invariants=run.architecture_spec.constraints,
        )
        stored_by_id = {item.id: item for item in nodes}
        top_level_ids = {
            item.id
            for item in nodes
            if item.parent_id is None and item.kind is WorkNodeKind.MODULE
        }
        engine_nodes: dict[str, EngineWorkNode] = {}
        module_contracts = []
        for item in nodes:
            verification = tuple(item.contract.metadata.get("verification", ()))
            if not verification and item.master_task_id:
                legacy = next(
                    (task for task in plan.tasks if task.id == item.master_task_id),
                    None,
                )
                verification = legacy.verification if legacy else ("Inspect the durable evidence.",)
            contract = self._engine_contract(item, verification)
            children = tuple(str(value) for value in item.metadata.get("children", ()))
            if not children:
                # Backward-compatible recovery for checkpoints written before
                # parent structure updates were persisted. Parent links are
                # independently durable, so reconstruct the exact child set
                # without invoking the planner or creating duplicate ids.
                children = tuple(
                    candidate.id
                    for candidate in nodes
                    if candidate.parent_id == item.id
                )
            dependencies = tuple(dict.fromkeys((*item.depends_on, *children)))
            try:
                phase = InnerPhase(item.checkpoint) if item.checkpoint else None
            except ValueError:
                phase = None
            engine = EngineWorkNode(
                contract=contract,
                parent_id=item.parent_id,
                depth=item.depth or 1,
                kind=NodeKind(item.kind.value),
                order=item.position,
                status=self._engine_node_status(item.status),
                phase=phase,
                children=children,
                pre_write_hashes={},
            )
            if dependencies != contract.depends_on:
                engine = EngineWorkNode(
                    contract=type(contract)(
                        id=contract.id,
                        title=contract.title,
                        objective=contract.objective,
                        acceptance_criteria=contract.acceptance_criteria,
                        verification=contract.verification,
                        depends_on=dependencies,
                        write_paths=contract.write_paths,
                        forbidden_changes=contract.forbidden_changes,
                        owned_interfaces=contract.owned_interfaces,
                        metadata=contract.metadata,
                    ),
                    parent_id=engine.parent_id,
                    depth=engine.depth,
                    kind=engine.kind,
                    order=engine.order,
                    status=engine.status,
                    phase=engine.phase,
                    children=engine.children,
                    pre_write_hashes=engine.pre_write_hashes,
                )
            engine_nodes[engine.id] = engine
            if item.parent_id is None and item.kind is WorkNodeKind.MODULE:
                # Dynamic children are scheduler dependencies of the parent,
                # not peers in the approval-bound top-level master plan.
                # Persisted graph checkpoints may include them in the engine
                # contract, so project only approved top-level dependencies
                # when rebuilding MasterPlanV1.
                module_contracts.append(
                    type(contract)(
                        id=contract.id,
                        title=contract.title,
                        objective=contract.objective,
                        acceptance_criteria=contract.acceptance_criteria,
                        verification=contract.verification,
                        depends_on=tuple(
                            dependency
                            for dependency in contract.depends_on
                            if dependency in top_level_ids
                        ),
                        write_paths=contract.write_paths,
                        forbidden_changes=contract.forbidden_changes,
                        owned_interfaces=contract.owned_interfaces,
                        metadata=contract.metadata,
                    )
                )
            if item.result:
                converted = self._engine_result(item.id, item.result)
                self.adapter._result_cache[item.id] = converted
                self.adapter.results[run_id][item.id] = converted

        # A foundation can be checkpointed between persisting the approval-bound
        # legacy Plan and materializing durable WorkNodes.  Rebuild only the
        # top-level module contracts from that immutable pending plan; execution
        # still cannot begin until the user approves its fingerprint.
        if not module_contracts and awaiting_approval:
            from .ultra import TaskContractV1 as EngineTaskContract

            paths_by_task: dict[str, list[str]] = {}
            for change in plan.expected_changes:
                path = str(change.get("path", "")).strip()
                for task_id in change.get("supports_tasks", ()):
                    if path:
                        paths_by_task.setdefault(str(task_id), []).append(path)
            task_to_node = {
                task.id: str(task.metadata.get("ultra_node_id", task.id))
                for task in plan.tasks
            }
            for position, task in enumerate(plan.tasks, start=1):
                node_id = task_to_node[task.id]
                contract = EngineTaskContract(
                    id=node_id,
                    title=task.title,
                    objective=task.description or task.title,
                    acceptance_criteria=task.acceptance_criteria,
                    verification=task.verification,
                    depends_on=tuple(
                        task_to_node.get(dependency, dependency)
                        for dependency in task.depends_on
                    ),
                    write_paths=tuple(dict.fromkeys(paths_by_task.get(task.id, ()))),
                    forbidden_changes=(),
                    owned_interfaces=(),
                    metadata={**dict(task.metadata), "restored_from_pending_plan": True},
                )
                node = EngineWorkNode(contract=contract, order=position)
                module_contracts.append(contract)
                engine_nodes[node.id] = node

        if not module_contracts:
            raise RuntimeError("the approved ULTRA run has no durable master modules")
        master = MasterPlanV1(
            summary=plan.summary,
            modules=tuple(module_contracts),
            execution_strategy=plan.execution_strategy,
            revision=plan.revision,
            fingerprint=run.master_plan_fingerprint
            or str(run.config.get("master_plan_fingerprint", "")),
        )
        prompt = str(run.config.get("prompt") or self.store.get_goal(run.goal_id).objective)
        engine_run = UltraRunV1(
            id=run.id,
            prompt=prompt,
            execution_class=self.descriptor.execution_class,
            phase=EnginePhase.AWAITING_APPROVAL,
            concurrency=run.concurrency,
            master_fingerprint=master.fingerprint,
            approved=not awaiting_approval,
            model_snapshot=self.descriptor.to_dict(),
            config_snapshot=dict(run.config),
            metadata={"restored": True},
            created_at=run.created_at,
            updated_at=run.updated_at,
        )
        self.orchestrator.run_state = engine_run
        self.orchestrator.goal_spec = goal_spec
        self.orchestrator.architecture = architecture
        self.orchestrator.master_plan = master
        self.orchestrator.nodes = engine_nodes
        self.orchestrator._results = dict(self.adapter.results[run_id])
        self.orchestrator._order = max((item.order for item in engine_nodes.values()), default=0)
        self.adapter.runs[run_id] = engine_run
        self.adapter.nodes[run_id] = dict(engine_nodes)
        if awaiting_approval:
            self.events.publish(
                "ultra.awaiting_approval",
                f"Restored ULTRA plan revision {plan.revision}; approval is still required",
                run_id=run_id,
                revision=plan.revision,
            )
            return plan
        self.store.update_ultra_run(
            run_id,
            phase=UltraPhase.MODULE_WAVES,
            status=UltraRunStatus.RUNNING,
            config={"restored_from_evidence": True},
        )
        self.future = self.orchestrator.background.start(self._run_and_finalize)
        return self.future

    @staticmethod
    def _engine_contract(item: WorkNode, verification: Sequence[str]) -> Any:
        from .ultra import TaskContractV1 as EngineTaskContract

        interface_names = tuple(item.contract.interfaces)
        return EngineTaskContract(
            id=item.id,
            title=item.title,
            objective=item.objective,
            acceptance_criteria=item.contract.success_criteria
            or ("Complete this durable node contract.",),
            verification=tuple(verification) or ("Inspect the durable evidence.",),
            depends_on=item.depends_on,
            write_paths=item.contract.write_paths,
            forbidden_changes=item.contract.forbidden_changes,
            owned_interfaces=interface_names,
            metadata={
                **dict(item.contract.metadata),
                "external_dependencies": list(item.contract.external_dependencies),
            },
        )

    def questions(self) -> tuple[Mapping[str, Any], ...]:
        if self.orchestrator and self.orchestrator.goal_spec:
            return tuple(self.orchestrator.goal_spec.questions)
        if self.run_id:
            return tuple(self.store.get_ultra_run(self.run_id).config.get("pending_questions", ()))
        return ()

    def add_guidance(self, text: str) -> None:
        if not self.adapter or not self.run_id:
            raise RuntimeError("there is no live ULTRA run for guidance")
        key = "user-guidance-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        self.adapter.append_brain_entry(
            BrainEntryV1(
                EngineBrainSection.KNOWLEDGE,
                key,
                {"summary": text, "source": "user", "priority": "high"},
                self.run_id,
            )
        )
        self.events.publish("ultra.guidance", "User guidance added to Project Brain")

    def switch_model(self, descriptor: ModelDescriptor) -> None:
        if not self.safe_for_reconfiguration:
            raise RuntimeError(
                "pause ULTRA and wait for active agents to reach a safe checkpoint before switching models"
            )
        self.descriptor = descriptor
        if self.adapter:
            self.adapter.descriptor = descriptor
        if not self.orchestrator:
            return
        factory = self.orchestrator.agent_factory
        if isinstance(factory, WorkspaceUltraAgentFactory):
            factory.descriptor = descriptor
        adaptive = AdaptiveConcurrency(
            descriptor.execution_class,
            cloud_default=self.config.cloud_concurrency,
            maximum=self.config.max_concurrency,
        )
        self.orchestrator.execution_class = descriptor.execution_class
        self.orchestrator.adaptive = adaptive
        self.orchestrator.scheduler.execution_class = descriptor.execution_class
        self.orchestrator.scheduler.adaptive = adaptive
        if self.orchestrator.run_state:
            self.orchestrator.run_state = replace(
                self.orchestrator.run_state,
                execution_class=descriptor.execution_class,
                concurrency=adaptive.current,
                model_snapshot=descriptor.to_dict(),
            )
            self.orchestrator.state.save_ultra_run(self.orchestrator.run_state)
        self.events.publish(
            "ultra.model_changed",
            f"ULTRA model changed to {descriptor.provider}/{descriptor.model}",
            execution_class=descriptor.execution_class.value,
            concurrency=adaptive.current,
        )

    def switch_permissions(self, adapter: PermissionAdapter) -> None:
        if not self.safe_for_reconfiguration:
            raise RuntimeError(
                "pause ULTRA and wait for active agents to reach a safe checkpoint before changing permissions"
            )
        self.permission_adapter = adapter
        if self.adapter:
            self.adapter.access_level = adapter.access_level
        if self.orchestrator and self.orchestrator.run_state:
            self.orchestrator.state.save_ultra_run(self.orchestrator.run_state)
        self.events.publish(
            "ultra.permissions_changed",
            f"ULTRA permissions changed to {adapter.access_level.value}",
            access_level=adapter.access_level.value,
        )

    def answer(self, question_id: str, value: str) -> MasterPlanV1 | None:
        if not self.orchestrator or not self.adapter or not self.goal_id:
            raise RuntimeError("there is no live ULTRA question round")
        pending = {str(item.get("id")): item for item in self.questions()}
        if question_id not in pending:
            raise ValueError(f"unknown ULTRA question id: {question_id}")
        self.answers[question_id] = str(value).strip()
        self.store.update_goal_metadata(self.goal_id, plan_answers=dict(self.answers))
        unanswered = set(pending) - set(self.answers)
        if unanswered:
            next_id = sorted(unanswered)[0]
            self.store.update_goal_metadata(
                self.goal_id,
                waiting_question=str(pending[next_id].get("question", "")),
            )
            return None
        goal = self.store.get_goal(self.goal_id)
        if goal.status is GoalStatus.PAUSED:
            self.store.transition_goal(
                self.goal_id,
                GoalStatus.DISCOVERING,
                reason="ULTRA goal questions answered",
            )
        plan = self.orchestrator.answer_questions(self.answers)
        assert self.orchestrator.goal_spec and self.orchestrator.architecture
        self.adapter.bind_foundation(
            self.orchestrator.goal_spec,
            self.orchestrator.architecture,
            plan,
        )
        return plan

    def approve(self, revision: int | None = None) -> Plan:
        if not self.orchestrator or not self.adapter or not self.orchestrator.master_plan:
            raise RuntimeError("there is no ULTRA master plan to approve")
        if revision is not None and self.adapter.plan and revision != self.adapter.plan.revision:
            raise ValueError(f"ULTRA is awaiting plan revision {self.adapter.plan.revision}")
        self.orchestrator.approve(self.orchestrator.master_plan.fingerprint)
        accepted = self.adapter.approve_master(self.orchestrator.master_plan)
        self.future = self.orchestrator.background.start(self._run_and_finalize)
        return accepted

    def _run_and_finalize(self) -> UltraRunResult:
        assert self.orchestrator
        try:
            result = self.orchestrator.run()
        except Exception as exc:
            self._record_engine_failure(exc)
            raise
        self._finalize_result(result)
        return result

    def _record_engine_failure(self, exc: Exception) -> None:
        if not self.goal_id:
            return
        try:
            goal = self.store.get_goal(self.goal_id)
            if goal.status not in {GoalStatus.BLOCKED, GoalStatus.CANCELLED}:
                self.store.transition_goal(
                    self.goal_id,
                    GoalStatus.BLOCKED,
                    reason=f"ULTRA engine failed: {redact_text(exc, 500)}",
                )
        except Exception:
            pass
        self.events.publish("error", f"ULTRA execution failed: {redact_text(exc, 500)}")

    def _finalize_result(self, result: UltraRunResult) -> None:
        if not self.goal_id:
            return
        try:
            goal = self.store.get_goal(self.goal_id)
            if result.successful:
                if self.run_id:
                    blocking_findings = [
                        item for item in self.store.list_quality_findings(self.run_id)
                        if item.severity.blocks_completion and item.status.value != "resolved"
                    ]
                    change_sets = self.store.list_change_sets(self.run_id)
                    unreviewed = [
                        item for item in change_sets
                        if item.status not in {ChangeSetStatus.APPROVED, ChangeSetStatus.INTEGRATED}
                    ]
                    if blocking_findings or unreviewed:
                        details = []
                        if blocking_findings:
                            details.append(f"{len(blocking_findings)} blocking quality finding(s)")
                        if unreviewed:
                            details.append(f"{len(unreviewed)} unreviewed Change Set(s)")
                        if goal.status is GoalStatus.RUNNING:
                            self.store.transition_goal(
                                self.goal_id,
                                GoalStatus.BLOCKED,
                                reason="ULTRA completion gate rejected: " + ", ".join(details),
                            )
                        return
                    for change_set in change_sets:
                        if change_set.status is ChangeSetStatus.APPROVED:
                            self.store.save_change_set(change_set.integrate())
                if goal.status is GoalStatus.RUNNING:
                    self.store.transition_goal(self.goal_id, GoalStatus.VERIFYING, reason="ULTRA module waves completed")
                goal = self.store.get_goal(self.goal_id)
                if goal.status is GoalStatus.VERIFYING:
                    self.store.transition_goal(self.goal_id, GoalStatus.REVIEWING, reason="ULTRA global review passed")
                goal = self.store.get_goal(self.goal_id)
                if goal.status is GoalStatus.REVIEWING:
                    self.store.transition_goal(self.goal_id, GoalStatus.COMPLETED, reason="ULTRA final evidence gate passed")
            elif result.run.phase is EnginePhase.CANCELLED:
                if goal.status is not GoalStatus.CANCELLED:
                    self.store.transition_goal(self.goal_id, GoalStatus.CANCELLED, reason="ULTRA cancelled")
            elif result.run.phase is EnginePhase.REVISION_REQUIRED:
                if goal.status is GoalStatus.RUNNING:
                    self.store.transition_goal(self.goal_id, GoalStatus.REVISING, reason="ULTRA requires master-plan revision")
                self.store.update_goal_metadata(
                    self.goal_id,
                    waiting_question="A quality or scope gate requires a revised master plan.",
                    auto_retryable=False,
                )
            elif goal.status is GoalStatus.RUNNING:
                self.store.transition_goal(self.goal_id, GoalStatus.BLOCKED, reason="ULTRA module wave failed")
        except Exception as exc:
            self.events.publish("error", f"ULTRA completion persistence failed: {redact_text(exc, 500)}")

    def pause(self) -> None:
        if not self.orchestrator:
            raise RuntimeError("there is no live ULTRA run")
        self.orchestrator.pause()

    def resume(self) -> None:
        if not self.orchestrator:
            raise RuntimeError("there is no live ULTRA run")
        self.orchestrator.resume()

    def cancel(self) -> None:
        if not self.orchestrator:
            raise RuntimeError("there is no live ULTRA run")
        if self.running:
            self.orchestrator.cancel()
        elif self.orchestrator.phase not in {
            EnginePhase.COMPLETED,
            EnginePhase.CANCELLED,
            EnginePhase.FAILED,
        }:
            self.orchestrator._set_phase(EnginePhase.CANCELLED, "ULTRA cancelled")

    def close(self) -> None:
        if self.orchestrator:
            self.orchestrator.background.close()


__all__ = [
    "StateStoreUltraAdapter",
    "UltraSession",
    "WorkspaceUltraAgent",
    "WorkspaceUltraAgentFactory",
]

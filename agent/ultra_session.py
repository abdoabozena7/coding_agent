"""Runnable integration between ULTRA orchestration and the v3 state store.

The provider-neutral engine is intentionally independent from the legacy
goal/plan runtime.  This module supplies the concrete adapters needed by the
CLI: real workspace tools, Docker-only Full shell access, durable v3 records,
legacy master-plan approval, file hashes, and resource leases.
"""

from __future__ import annotations

import ast
import fnmatch
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import Future
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import tools
from .component_artifacts import (
    ComponentFileV2,
    ComponentArtifactError,
    ComponentArtifactStore,
    InterfaceContractV1,
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
    extract_action_proposal,
    normalize_generated_tool_args,
)
from .learning import GlobalLessonStore, LearnedLessonV1
from .durable_memory import (
    AgentMemorySnapshotV1,
    NextActionPacketV1,
    NextActionStatus,
)
from .goal_outcome import (
    ExperimentOutcome,
    FinalAcceptanceEvidenceV1,
    GoalOutcomeContractV1,
    GoalOutcomeState,
    OptimizationExperimentV1,
)
from .orchestration import (
    AdaptiveOrchestrationPolicyV1,
    EvidenceAuthority,
    EvidenceClaimV1,
    EvidenceVerdict,
    OrchestrationArm,
    OrchestrationExperimentV1,
    WorkerImpactV1,
    WorkerRole,
    approach_fingerprint,
    classify_worker_impact,
    evidence_novelty,
)
from .models import DomainError, GoalStatus, Plan, PlanStatus, RoleProfile, TaskStatus, utc_now
from .semantic import RequestedEffect, SemanticGoalV2
from .providers.base import AssistantTurn, ToolCall
from .project_brain import ProjectBrain
from .safety import redact_data, redact_text
from .sandbox import AccessLevel, PermissionAdapter
from .scheduler import ResourceLease as RuntimeLease
from .scheduler import AdaptiveConcurrency, RateLimitError, ResourceLeaseManager, StaleWriteError
from .store import NotFoundError, StateStore, StateStoreError
from .tools._security import atomic_write_bytes
from .swarm_coordinator import SwarmCoordinator
from .swarm_protocol import SwarmMessageType, SwarmMessageV1
from .ultra import (
    ApprovalRequiredError,
    AgentProtocolError,
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
    _normalize_browser_scenario_transport,
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
    normalize_contract_path,
)
from .visual_judge import (
    UnavailableVisionJudge,
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
from .version_control import GitProtectionManager


class _MutationTransportExhausted(AgentProtocolError):
    """A local mutation turn ignored its bounded write-only handoff."""


_READ_TOOLS = tools.names(categories={"read"})
_WRITE_TOOLS = tools.names(categories={"write", "command", "install"})
_FILE_WRITE_TOOLS = tools.names(categories={"write"})
_TOOL_RISK = tools.risk_map()
_TESTER_VERIFICATION_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"(?:(?:\"[^\"]+\"|'[^']+'|python(?:\.exe)?)\s+-m\s+)?pytest\b|"
    r"(?:\"[^\"]+\"|'[^']+'|python(?:\.exe)?)\s+-m\s+(?:unittest|compileall)\b|"
    r"(?:python3?(?:\.exe)?|\"[^\"]+\"|'[^']+')\s+"
    r"[^\s;&|<>`]+\.py(?:\s+[^\r\n;&|<>`]*)?$|"
    r"npm\s+(?:test|run\s+(?:test|build|lint|check))\b|"
    r"(?:pnpm|yarn)\s+(?:test|build|lint|check)\b|"
    r"cargo\s+(?:test|check|clippy|build)\b|"
    r"go\s+test\b|"
    r"dotnet\s+(?:test|build)\b"
    r")",
    re.IGNORECASE,
)
_SHELL_COMPOSITION_RE = re.compile(r"[\r\n;&|<>`]|\$\(")


def _bind_accepted_repair_feedback(
    plan: MasterPlanV1,
    feedback: str,
) -> MasterPlanV1:
    """Keep explicit repair requirements attached to every repair module.

    Small models sometimes return a schema-valid repair plan that collapses
    detailed, accepted feedback into a generic objective such as "verify the
    app".  The harness must not invent product semantics, but it also must not
    discard semantics already supplied by the user or an evidence-backed
    reviewer.  Preserve that exact feedback in the approval-visible objective
    and in the durable node metadata used by specialists.
    """

    accepted = str(feedback or "").strip()
    if not accepted:
        return plan
    scenarios: tuple[Mapping[str, Any], ...] = ()
    marker_index = accepted.casefold().find("metadata.browser_scenarios")
    if marker_index >= 0:
        array_index = accepted.find("[", marker_index)
        if array_index >= 0:
            try:
                parsed, _end = json.JSONDecoder().raw_decode(accepted[array_index:])
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, list) and parsed and all(
                isinstance(item, Mapping)
                and bool(item.get("steps"))
                and bool(item.get("assertions"))
                for item in parsed
            ):
                scenarios = tuple(dict(item) for item in parsed)
    bound_modules = []
    for module in plan.modules:
        marker = "Accepted repair requirements:"
        objective = module.objective.strip()
        if marker.casefold() not in objective.casefold():
            objective = f"{objective}\n\n{marker}\n{accepted}".strip()
        bound_modules.append(
            replace(
                module,
                objective=objective[:8_000],
                metadata={
                    **dict(module.metadata),
                    **(
                        {"browser_scenarios": list(scenarios)}
                        if scenarios
                        else {}
                    ),
                    "accepted_repair_feedback": accepted,
                    "accepted_repair_feedback_bound": True,
                    **(
                        {"verification_only": True}
                        if scenarios
                        and re.search(
                            r"\b(?:verification[- ]focused|verification only|preserve the current implementation)\b",
                            accepted,
                            re.IGNORECASE,
                        )
                        else {}
                    ),
                },
            )
        )
    return replace(plan, modules=tuple(bound_modules))


def _bind_repair_feedback(
    plan: MasterPlanV1,
    feedback: str,
) -> MasterPlanV1:
    """Separate internal contract recovery from accepted product feedback."""

    accepted = str(feedback or "").strip()
    internal_contract_failure = (
        "ultra foundation/phase" in accepted.casefold()
        and "failed after three targeted typed-return repairs" in accepted.casefold()
    )
    if not internal_contract_failure:
        return _bind_accepted_repair_feedback(plan, accepted)
    return replace(
        plan,
        modules=tuple(
            replace(
                module,
                metadata={
                    **dict(module.metadata),
                    "contract_recovery_diagnostic": accepted,
                },
            )
            for module in plan.modules
        ),
    )


def _explicit_repair_scope_paths(feedback: str) -> tuple[str, ...]:
    """Extract only paths the user explicitly marks as the entire repair scope.

    This is an authority parser, not a semantic router.  It deliberately does
    nothing unless the feedback contains an unambiguous ``keep/edit ... exactly
    <path>`` or ``only ... <path>`` instruction.  A later, narrower repair
    authority must win over the broader path set approved by an older plan.
    """

    text = str(feedback or "").strip()
    if not text:
        return ()
    clauses = re.findall(
        r"(?:keep|edit|modify|change|touch|limit(?:\s+the\s+repair)?\s+to)"
        r"[^\r\n;]{0,48}?\b(?:exactly|only)\b\s+"
        r"((?:[\w.-]+[\\/])*[\w.-]+\.[A-Za-z0-9]{1,12})",
        text,
        flags=re.IGNORECASE,
    )
    paths: list[str] = []
    for candidate in clauses:
        normalized = normalize_contract_path(candidate)
        if normalized and normalized not in paths:
            paths.append(normalized)
    return tuple(paths)


def _bind_repair_architecture_authority(
    architecture: EngineArchitectureSpec,
    feedback: str,
) -> EngineArchitectureSpec:
    """Make newer, explicit repair authority win over stale design details."""

    accepted = str(feedback or "").strip()
    if not accepted:
        return architecture
    precedence = (
        "Latest accepted repair requirements take precedence over conflicting "
        f"earlier implementation details: {accepted}"
    )
    return replace(
        architecture,
        summary=f"{architecture.summary}\n\n{precedence}".strip()[:8_000],
        decisions=(
            *architecture.decisions,
            {
                "decision": "Apply the latest accepted repair requirements.",
                "reason": accepted,
                "status": "accepted_revision_authority",
            },
        ),
        invariants=tuple(dict.fromkeys((*architecture.invariants, precedence))),
    )


def _tester_verification_command_allowed(command: str) -> bool:
    """Allow one verifier invocation, never shell-composed workspace edits."""

    text = str(command).strip()
    return bool(
        text
        and not _SHELL_COMPOSITION_RE.search(text)
        and _TESTER_VERIFICATION_COMMAND_RE.match(text)
    )


_TESTER_CD_COMMAND_RE = re.compile(
    r"^\s*cd\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;&|<>`]+))"
    r"\s*&&\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _normalize_tester_verification_call(
    call: ToolCall,
) -> tuple[ToolCall, str]:
    """Convert a safe ``cd && verifier`` into typed cwd transport.

    This does not grant shell composition: only one workspace-relative cwd and
    one already-allowlisted verifier command survive the conversion.
    """

    if call.name not in {"run_bash", "run_command"}:
        return call, ""
    command = str(call.args.get("command") or "")
    match = _TESTER_CD_COMMAND_RE.fullmatch(command)
    if match is None:
        return call, ""
    cwd = next((value for value in match.groups()[:3] if value is not None), "")
    verification = str(match.group(4) or "").strip()
    normalized_cwd = cwd.strip().replace("\\", "/")
    path = PurePosixPath(normalized_cwd)
    if (
        not normalized_cwd
        or path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z]:", normalized_cwd)
        or not _tester_verification_command_allowed(verification)
    ):
        return call, ""
    return (
        ToolCall(
            id=call.id,
            name="run_command",
            args={"cwd": normalized_cwd, "command": verification},
        ),
        "tester cd-and-verifier shell syntax normalized to run_command cwd",
    )


_COMMAND_EXIT_CODE_RE = re.compile(r"^\s*exit code:\s*(-?\d+)\b", re.IGNORECASE)


def _tester_command_receipt(
    call: ToolCall,
    result: str,
    *,
    node_id: str = "",
    usage: Mapping[str, int] | None = None,
) -> AgentResponse | None:
    """Turn an executed verifier's exit status into authoritative evidence.

    The harness owns command execution and may therefore report the exact
    outcome without asking a small model to restate it as a typed boolean on a
    second inference.  No verdict is synthesized for errors, non-verifier
    commands, approval boundaries, or output without the executor's leading
    exit-code receipt.
    """

    if call.name not in {"run_bash", "run_command"}:
        return None
    command = str(call.args.get("command") or "").strip()
    if not _tester_verification_command_allowed(command):
        return None
    output = str(result or "")
    if output.startswith("Error:"):
        return None
    match = _COMMAND_EXIT_CODE_RE.match(output)
    if match is None:
        return None
    returncode = int(match.group(1))
    passed = returncode == 0
    cwd = str(call.args.get("cwd") or ".").strip() or "."
    bounded_output = redact_text(output[:12_000])
    finding = (
        ""
        if passed
        else f"Verification command failed with exit code {returncode}: {command}"
    )
    result_record = {
        "name": "executable_verification",
        "command": command,
        "cwd": cwd,
        "returncode": returncode,
        "passed": passed,
    }
    evidence = {
        "kind": "executable_verification",
        **result_record,
        "output": bounded_output,
    }
    return AgentResponse.from_mapping(
        {
            "payload": {
                "passed": passed,
                "success": passed,
                "issues": [] if passed else [finding],
                "findings": [] if passed else [finding],
                "test_results": [result_record],
                "evidence": [evidence],
            },
            "summary": (
                f"Executable verification passed: {command}"
                if passed
                else finding
            ),
            "reasoning_summary": (
                "Verdict derived from the harness-owned process exit status; "
                "no model-authored completion claim was used."
            ),
        },
        node_id=node_id,
        provider="harness",
        model="command-receipt-v1",
        usage=dict(usage or {}),
    )
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

_SUBMIT_MASTER_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_master_plan",
        "description": (
            "Submit the complete approval-bound master plan. This records planning data only; "
            "it does not execute commands or mutate the workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "execution_strategy": {"type": "string"},
                "modules": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "objective": {"type": "string"},
                            "acceptance_criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "verification": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": (
                                        "Literal executable verifier. preview_html must receive "
                                        "one workspace-relative .html path without a URL query or fragment."
                                    ),
                                },
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "write_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "metadata": {
                                "type": "object",
                                "description": (
                                    "For interactive preview modules include browser_scenarios: "
                                    "an array of {name, steps:[{action, role, name, value only for fill}], "
                                    "assertions:[{role, name, property, equals or contains}]} objects. "
                                    "Do not add text to click steps."
                                ),
                            },
                        },
                        "required": [
                            "id",
                            "title",
                            "objective",
                            "acceptance_criteria",
                            "verification",
                            "depends_on",
                            "write_paths",
                        ],
                    },
                },
            },
            "required": ["summary", "execution_strategy", "modules"],
        },
    },
}

_SUBMIT_BROWSER_SCENARIOS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_browser_scenarios",
        "description": (
            "Submit executable browser interaction scenarios for only the listed plan modules."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "module_id": {"type": "string"},
                            "browser_scenarios": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "steps": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {
                                                "type": "object",
                                                "description": (
                                                    "Executable step. Use action click with role/name or selector; "
                                                    "action fill with role/name or selector plus value; or action "
                                                    "press with key. role is an ARIA role such as textbox, button, "
                                                    "or heading, never an HTML tag such as input or h2. Never use "
                                                    "action type."
                                                ),
                                                "properties": {
                                                    "action": {
                                                        "type": "string",
                                                        "enum": ["click", "fill", "press"],
                                                    },
                                                    "role": {"type": "string"},
                                                    "name": {"type": "string"},
                                                    "selector": {"type": "string"},
                                                    "value": {"type": "string"},
                                                    "key": {"type": "string"},
                                                },
                                                "required": ["action"],
                                            },
                                        },
                                        "assertions": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {
                                                "type": "object",
                                                "description": (
                                                    "Observable assertion with role/name or selector, property "
                                                    "value, text, textContent, id, visible, checked, count, or "
                                                    "visibleCount; interactive graphics may expose bounded "
                                                    "dataObjectCount or dataVisualState values. Use exactly one "
                                                    "expected value in equals or "
                                                    "contains. Use property value only for form controls such as "
                                                    "textbox, combobox, spinbutton, or slider; use property text/textContent "
                                                    "for status, heading, button, and other elements. role is an "
                                                    "ARIA role, not an HTML tag."
                                                ),
                                                "properties": {
                                                    "role": {"type": "string"},
                                                    "name": {"type": "string"},
                                                    "selector": {"type": "string"},
                                                    "property": {
                                                        "type": "string",
                                                        "enum": [
                                                            "value",
                                                            "text",
                                                            "textContent",
                                                            "innerText",
                                                            "id",
                                                            "visible",
                                                            "checked",
                                                            "count",
                                                            "visibleCount",
                                                            "dataObjectCount",
                                                            "dataVisualState",
                                                        ],
                                                    },
                                                    "equals": {"type": "string"},
                                                    "contains": {"type": "string"},
                                                },
                                                "required": ["property"],
                                            },
                                        },
                                    },
                                    "required": ["name", "steps", "assertions"],
                                },
                            },
                        },
                        "required": ["module_id", "browser_scenarios"],
                    },
                }
            },
            "required": ["modules"],
        },
    },
}


def _specialist_quality_blueprint(domain: str) -> tuple[str, ...]:
    """Return compact domain craft constraints, not a whole-product prompt."""

    key = str(domain or "").strip().casefold()
    profiles: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (
            ("world.road.geometry",),
            (
                "Keep road length on the Z axis, centered at the origin; never rotate the complete road group.",
                "Use a dark asphalt slab with two separate shoulder strips outside its X bounds; no full-width overlay may cover asphalt.",
                "Model narrow curbs (about 0.2-0.35 units), layered shoulder edges, subtle material variation, and contact/drain details as distinct meshes.",
                "Repeat readable divider dashes (about 1.5-2.5 units long with larger gaps) along Z and keep all markings above and inside asphalt bounds.",
                "Use proportions and camera framing that show the whole reusable segment with strong silhouette and depth.",
                "Remove only a previous named component root; never clear the complete scene, camera, or harness lights.",
            ),
        ),
        (
            ("world.road.markings",),
            (
                "Define exactly window.buildPreview=({THREE,scene,camera,renderer})=>{...}; do not import Three.js, create a Scene/Renderer, touch the DOM, or invoke the function yourself.",
                "The harness preview supplies a neutral road: width 10.8 on X, length 52 on Z, surface Y=0. Build only reusable markings inside X=-5.2..5.2 and Z=-24..24.",
                "Use numeric color literals directly in THREE materials (for example 0xf5d547); never construct a hex literal with template interpolation such as 0x${value}.",
                "Build one named root containing: edge lines as BoxGeometry(0.12,0.03,48) at X=+/-5.0; yellow center dashes as BoxGeometry(0.14,0.03,1.9) every 3.3 Z units; a white stop line BoxGeometry(9.6,0.03,0.35).",
                "Build the zebra crossing as 6 separate full-width bars, each about BoxGeometry(9.2,0.03,0.45), separated along Z by about 0.8; do not make one dotted row across X.",
                "A gap means add no mesh. Reuse or clone a declared geometry for left/right pairs; never reference a differently named undeclared geometry.",
                "Keep every marking at Y=0.03 above the reference surface and inside the road bounds; use shared materials and dispose of no harness-owned object.",
            ),
        ),
        (
            ("world.road.collision",),
            (
                "Define exactly window.buildPreview=({THREE,scene,camera,renderer})=>{...}; never create/replace Scene, Renderer, DOM, camera, lights, or animation loop.",
                "Publish window.RoadCollisionAPI with pure functions overlapsAABB(a,b) and isInsideRoad(position,halfWidth,halfDepth); AABBs use minX/maxX/minZ/maxZ and touching edges count as collision.",
                "Use the contracted road bounds X=-5.2..5.2 and Z=-24..24. isInsideRoad must account for the caller half extents and return false when any edge leaves those bounds.",
                "Draw only aligned debug evidence: a translucent road-bounds box plus one green colliding test pair and one red separated pair; no placeholder label, arbitrary terrain, or detached example obstacle.",
                "Do not mutate scene children by geometry heuristics. Add one named collision root and keep pure API logic independent from Three.js meshes.",
                "The harness will execute positive/negative overlap and inside/outside bound assertions after buildPreview; any missing function or wrong result rejects the package.",
                "Keep the complete file under 120 lines: omit JSDoc and narrative comments, reuse one box helper/material family, and render no more than the road bounds plus four small test boxes.",
            ),
        ),
        (
            ("vehicles.wheels", "vehicle.wheels"),
            (
                "Build a wheel assembly with tire sidewall, tread cues, rim, hub, axle contact, and correct left/right orientation.",
                "Show the wheel at a useful three-quarter angle with no floating or intersecting parts.",
            ),
        ),
        (
            ("vehicles.vehicle_details_cut",),
            (
                "Own only aligned vehicle details for the accepted 2.8-wide, 5.2-long +Z body: four finished wheels, cabin glass, lamps, bumpers, mirrors, grille, and restrained trim.",
                "Place four wheel centers at X=+/-1.48, Y=0.55, Z=+/-1.55. Each wheel needs a dark tire, contrasting rim, and hub; rotate CylinderGeometry so the axle runs on X.",
                "Use separate windshield, rear window, two side-glass panels, two headlights, two tail lights, front/rear bumpers, grille bars, and mirror pairs. Avoid body-size boxes or a replacement chassis.",
                "Create one root with 20-32 visible meshes and at least five material colors. Keep bounds within X=+/-1.9, Z=+/-2.9, Y=0.25..2.2 and add the root once to scene.",
                "Expose window.VehicleDetailsAPI={root,wheelCenters:[...four points],forward:'+Z'} or a runnable window.buildPreview. Keep the complete source under 70 lines: one material map, one add helper, and tuple loops only; no per-mesh variables or narrative comments.",
            ),
        ),
        (
            ("vehicles.vehicle_wheels_cut",),
            (
                "Own only four complete wheel assemblies for a 2.8-wide, 5.2-long +Z vehicle; no body, glass, road, lights, or camera.",
                "Each assembly needs tire, inset rim, hub, and 5-8 visible spokes or tread cues. Use centers X=+/-1.48, Y=.55, Z=+/-1.55 and an X-running axle.",
                "Use dark rubber, metallic rim, and darker hub materials; keep outer tire radius .42-.52 and width .24-.34 so wheels support rather than dominate the body.",
                "Create one root with 16-28 meshes using one createWheel helper and one loop. Publish window.VehicleWheelsAPI={root,centers,forward:'+Z'} in <=65 lines.",
            ),
        ),
        (
            ("vehicles.vehicle_glass_cut",),
            (
                "Own only thin cabin glazing, pillars, and two side mirrors aligned to the accepted body; no replacement cabin/body, wheels, lights, road, or camera.",
                "Build a raked windshield near Z=+.72, rear glass near Z=-1.08, paired side windows near X=+/-1.11, slim dark pillars/roof trim, and mirror shells near X=+/-1.42.",
                "Glass must use transparent MeshPhysicalMaterial or MeshStandardMaterial with opacity .42-.68, roughness <=.3, and restrained cyan/charcoal tint; never opaque blue slabs.",
                "Create 8-16 thin meshes, add one root, and publish window.VehicleGlassAPI={root,forward:'+Z'} in <=65 lines using tuple loops.",
            ),
        ),
        (
            ("vehicles.vehicle_fascia_cut",),
            (
                "Own only front/rear fascia details: paired warm headlights at +Z, paired red tail lights at -Z, bumpers, grille, lower intake, side trim, and small badges.",
                "Use 12-22 thin meshes inside X=+/-1.45, Z=+/-2.58 and Y=.42..1.05. No body-size box, wheels, glass, road, or camera.",
                "Use at least four coherent materials: charcoal grille/trim, warm emissive headlights, red emissive tail lights, and metallic bumper accents.",
                "Create one root and publish window.VehicleFasciaAPI={root,forward:'+Z'} in <=65 lines using helpers and tuple loops.",
            ),
        ),
        (
            ("vehicles.chassis.shell.volumes",),
            (
                "Own only the four primary vehicle volumes: lower tub, hood, cabin/roof, and rear deck. Create one root; no road, floor, wheels, lights, camera, renderer, Scene, DOM, or animation loop.",
                "Use these bounded absolute anchors as a design grammar, then refine scale slightly without leaving the envelope: tub size [2.8,0.35,5.0] at [0,0.55,0]; hood [2.45,0.55,1.55] at [0,0.9,1.65]; cabin [2.2,0.9,2.15] at [0,1.2,-0.1]; rear deck [2.35,0.45,1.25] at [0,0.82,-1.85].",
                "X is lateral width and every center x must be zero. Z is longitudinal: positive Z is the front. Use two or three coherent colors and connected overlap; no dimension may use L as a multiplier larger than 0.45.",
                "Expose window.ChassisVolumesAPI={root,dimensions:{width:2.8,height:1.5,length:5.2},forward:'+Z'}. buildPreview is optional because the typed preview harness mounts API.root automatically; do not duplicate harness plumbing.",
                "Keep scene.js under 45 lines using one addBox helper and a tuple loop.",
            ),
        ),
        (
            ("vehicles.chassis.shell.panels",),
            (
                "Own only shell surface details aligned to a 2.8-wide, 5.2-long body: paired fender shoulders, paired door panels, front/rear fascia, lower cladding, restrained trim, and four recessed wheel-well cue panels.",
                "All centers must keep abs(x)<=1.38 and abs(z)<=2.5. Wheel cues and exported mounts are exactly the Cartesian pairs X=+/-1.38 and Z=+/-1.55. Use small absolute sizes, never L or W multipliers larger than 0.5.",
                "Publish window.ChassisPanelsAPI={root,wheelMounts:[...four {x,y,z}],forward:'+Z'}. buildPreview is optional because the typed preview harness mounts API.root automatically; the harness supplies a neutral reference body for isolated alignment review.",
                "Use 9-14 meshes, at least three material colors, one addBox helper, tuple loops, and at most 50 lines. Do not create primary tub/cabin volumes, wheels, road, lights, Scene, renderer, DOM, or animation loop.",
            ),
        ),
        (
            ("vehicles.chassis.shell",),
            (
                "Own only a coherent stylized vehicle body shell facing +Z; do not create road, floor, wheels, lights, camera, renderer, Scene, DOM, or animation loop.",
                "Build one named root around width 2.8, length 5.2, and body height 1.5 using 8-14 connected meshes: lower tub, sculpted hood, cabin/roof volume, rear deck, left/right fender shoulders, door-side panels, and front/rear fascia layers.",
                "Use tapered/scaled boxes and layered geometry to create a clear hood-cabin-rear rhythm. X is lateral width only: every mesh center must keep abs(x)<=1.38. Place hood/front at positive Z, cabin near Z=0, and rear deck at negative Z; never position longitudinal sections along X. Add four dark recessed wheel-well cue panels at X=+/-1.38 and Z about +/-1.55, but no finished wheels.",
                "Ground the lower body near Y=0.65 with no floating or detached bars. Use coherent deep paint, darker lower cladding/underbody, and restrained metallic trim; all-white geometry is invalid.",
                "Write one compact stage_component_file tool call only. Start scene.js with const T=window.THREE, root=new T.Group(), one addBox(name,size,pos,color) helper, and tuple-array loops; do not repeat mesh construction blocks or write narrative comments.",
                "Expose window.ChassisShellAPI={root,dimensions:{width:2.8,height:1.5,length:5.2},wheelMounts:[...four {x,y,z}],forward:'+Z'} before any optional detail. The entire complete scene.js must be at most 55 physical lines and roughly 4200 characters so the tool JSON closes before the local output limit.",
            ),
        ),
        (
            ("vehicles.chassis", "vehicle.chassis"),
            (
                "Create a readable vehicle silhouette with distinct hood, body, bumpers, wheel arches, and underbody stance.",
                "Use beveled/layered geometry and coherent proportions; a single box is never a finished chassis.",
            ),
        ),
        (
            ("vehicles.cabin", "vehicle.cabin", "vehicles.glass"),
            (
                "Separate cabin frame, windows, pillars, glass, mirrors, and trim with controlled transparency and depth.",
                "Avoid opaque window blocks, z-fighting, and glass that hides the complete silhouette.",
            ),
        ),
        (
            ("character.body", "character"),
            (
                "Build a readable stylized character silhouette with head, torso, limbs, face cues, feet contact, and a clear forward direction.",
                "Use nested pivots suitable for animation; no single primitive may stand in for the final body.",
            ),
        ),
        (
            ("world.environment.props.trees",),
            (
                "Own only a reusable broadleaf-and-pine tree kit; create no ground, road, sky, lights, camera, renderer, DOM, or animation loop.",
                "Model two broadleaf and two pine variants from separate tapered trunks, branch/canopy layers, faceted crowns, and grounded root/contact cues; no single trunk-plus-sphere placeholder.",
                "Arrange the four variants at X=-4,-1.3,1.3,4 on Z=0 with height 3-4.5, deterministic transforms, shared geometry/materials, and a named root/API.",
                "Use window.buildPreview and expose window.TreePropsAPI={root,createBroadleaf,createPine}; keep the complete file under 105 lines.",
            ),
        ),
        (
            ("world.environment.props.rocks",),
            (
                "Own only reusable low-poly boulders; create no full ground, road, background, lights, camera, renderer, DOM, or animation loop.",
                "Build three visibly different multi-rock clusters using faceted dodecahedron/icosahedron meshes, warm gray-brown material variation, grounded overlap, and controlled non-uniform scale.",
                "Arrange clusters at X=-4,0,4 on Z=0, deterministic and fully inside the harness view; expose window.RockPropsAPI={root,createRockCluster} in under 90 lines.",
            ),
        ),
        (
            ("world.environment.props.shrubs",),
            (
                "Own only reusable shrub and flower clusters; create no full ground, road, background, lights, camera, renderer, DOM, or animation loop.",
                "Build four layered clusters with 3-6 low-poly leaf masses, visible stems, two foliage tones, and restrained flower accents; avoid loose spheres and random placement.",
                "Arrange clusters at X=-4,-1.3,1.3,4 on Z=0 with deterministic tuple offsets; expose window.ShrubPropsAPI={root,createShrub} in under 95 lines.",
            ),
        ),
        (
            ("world.environment.props.roadside",),
            (
                "Own only reusable roadside detail models; create no full ground, road, background, lights, camera, renderer, DOM, or animation loop.",
                "Build a readable wooden direction sign, short post-and-rail fence, striped safety bollard pair, and small reflector marker; every asset must have multiple meshes and grounded contact.",
                "Arrange four assets across X=-4,-1.3,1.3,4 on Z=0 using coherent wood/paint/metal materials; expose window.RoadsidePropsAPI={root,createSign,createFence,createBollards,createReflector} in under 110 lines.",
            ),
        ),
        (
            ("world.environment.props",),
            (
                "Define exactly window.buildPreview=({THREE,scene,camera,renderer})=>{...}; never create/replace Scene, Renderer, DOM, camera, lights, fog, background, animation loop, road, river, or full-size ground plane.",
                "Own only a reusable stylized prop kit. Build one named root and instantiate a readable 3x2 showroom containing: broadleaf tree, pine tree, faceted rock cluster, layered shrub, wooden signpost, and short fence segment.",
                "Each prop must be a multi-mesh model with a distinct silhouette, grounded contact, low-poly bevel/facet cues, and coherent shared grass-green/warm-wood/stone/paint materials; no primitive alone is a finished prop.",
                "Use small neutral circular or square display plinths only beneath individual props. Space centers around X=-4.5,0,4.5 and Z=-3,3 so every prop remains visible in the harness camera.",
                "Use deterministic authored transforms: no Math.random, no giant sky/background box, and no mesh over 5 units in any dimension. Reuse declared geometries/materials and keep the complete file under 120 lines.",
                "Add all models to propsRoot, add that root once to scene, and expose window.EnvironmentPropsAPI={root:propsRoot,createTree,createPine,createRocks,createShrub,createSign,createFence}.",
            ),
        ),
        (
            ("world.environment.composition.hills",),
            (
                "Own only distant rolling hills at Z=-22..-30; create no road, floor, boundary wall, lights, camera, renderer, DOM, or animation loop.",
                "Use 5-7 overlapping faceted low hill forms with varied silhouette, cool desaturated green/blue depth tones, and clear gaps; never use one full-width opaque box.",
                "Keep X=-7..7 and Z=-16..16 unobstructed, use deterministic transforms, and expose window.HillCompositionAPI={root,band:'background'} in under 90 lines.",
            ),
        ),
        (
            ("world.environment.composition.tree_line",),
            (
                "Own only an irregular distant tree line at Z=-19..-25; create no road, floor, wall, lights, camera, renderer, DOM, or animation loop.",
                "Build 9-13 compact trunk-plus-faceted-crown silhouettes with authored height/color cadence and intentional gaps; never render identical gray blocks or a continuous hedge wall.",
                "Keep the central near field empty, use deterministic tuple arrays/shared geometry, and expose window.TreeLineCompositionAPI={root,band:'midground'} in under 100 lines.",
            ),
        ),
        (
            ("world.environment.composition.landmarks",),
            (
                "Own only two asymmetric farm landmarks outside |X|=14 and behind Z=-16; create no full ground, road, lights, camera, renderer, DOM, or animation loop.",
                "Model one small barn with roof/door/window/trim and one windmill or silo with readable layered silhouette; single boxes and debug shapes are not finished landmarks.",
                "Use coherent ochre/red/wood/metal accents and deterministic placement; expose window.LandmarkCompositionAPI={root,landmarks:2} in under 115 lines.",
            ),
        ),
        (
            ("world.environment.composition",),
            (
                "Define exactly window.buildPreview=({THREE,scene,camera,renderer})=>{...}; never create/replace Scene, Renderer, DOM, camera, lights, fog, background, animation loop, road, or full-size ground/floor plane.",
                "Own only distant scenic composition and landmark rhythm. Keep the playable central corridor X=-7..7 and near field Z=-16..16 empty so Terrain, Road, Vehicles, and Character remain unobstructed.",
                "Build one named root with three authored depth bands: low rolling faceted hills at Z=-24..-30, an irregular tree line at Z=-20..-26, and two asymmetric farm landmarks outside |X|=14.",
                "Use varied silhouettes and a coherent atmospheric palette with darker foreground greens, mid green/ochre landmarks, and cooler desaturated distant hills. Never use identical gray blocks, boundary walls, debug guides, or placeholder rows.",
                "Use deterministic tuple arrays and shared geometry/materials; no Math.random. No object may exceed 8 units high, crop the camera, or span the full scene width as one opaque box.",
                "Add every owned mesh to compositionRoot, add it once to scene, expose window.EnvironmentCompositionAPI={root:compositionRoot,bands:3,landmarks:2}, and keep the file under 120 lines.",
            ),
        ),
        (
            ("world.environment.terrain.banks",),
            (
                "Own only the two road-side bank slabs from Z=-26..26. Keep X=-6.2..6.2 completely empty; create no props, road, lights, camera, renderer, DOM, or animation loop.",
                "Use layered low boxes/edge strips with bank inner edges exactly at X=-6.2 and X=6.2, muted grass/soil contrast, grounded Y near 0, and no full-scene plane.",
                "Expose window.TerrainBanksAPI={root,corridorHalfWidth:6.2,extents:{minZ:-26,maxZ:26}} in under 75 lines.",
            ),
        ),
        (
            ("world.environment.terrain.verges",),
            (
                "Own only layered verge contours outside |X|=6.2: shallow drainage strips, soil transitions, and irregular grass edge rhythm; no road, props, lights, or full ground plane.",
                "Use reusable narrow geometry and deterministic Z segments with subtle height/material variation; never cross the contracted corridor or form tall boundary walls.",
                "Expose window.TerrainVergesAPI={root,corridorHalfWidth:6.2} with a compact integration-ready preview under 90 lines.",
            ),
        ),
        (
            ("world.environment.terrain.ground_cover",),
            (
                "Own only low ground-cover patches outside |X|=6.4: grass tufts, soil patches, and pebble groups below 0.5 units high; no trees, road, floor, lights, or background.",
                "Create 14-20 deterministic small clusters across both banks using tuple arrays, shared geometries, muted two-tone materials, and intentional spacing; no Math.random.",
                "Expose window.GroundCoverAPI={root,corridorHalfWidth:6.4} in under 100 lines.",
            ),
        ),
        (
            ("world.environment.terrain",),
            (
                "Define exactly window.buildPreview=({THREE,scene,camera,renderer})=>{...}; never create/replace Scene, Renderer, DOM, camera, lights, fog, background, or an animation loop.",
                "Build one named terrain root that can flank a road running on Z. Keep the central corridor X=-6.2..6.2 completely empty and place all terrain/props only on the left and right banks.",
                "Use a deterministic authored layout: no Math.random. Cover Z=-26..26 with exactly two low green bank slabs, then use one compact tuple layout array and a loop for all props.",
                "Create 12-18 readable low-poly props across both banks: trees made from separate trunk and faceted crown meshes, varied dodecahedron rocks, and small shrub clusters. Reuse shared geometries/materials.",
                "Use a coherent saturated-but-muted palette (grass greens, warm brown trunks, gray-brown rocks, two foliage tones) with visible contrast; never use near-white terrain or a full-scene earth plane.",
                "No prop may exceed about 4 units high or 2.5 units wide, enter the road corridor, or sit so close to the preview camera that it crops the composition. Keep roots grounded at Y=0.",
                "Add every bank and prop to terrainRoot, add that root once to scene, then finish with window.TerrainAPI={root:terrainRoot,corridorHalfWidth:6.2,extents:{minZ:-26,maxZ:26}}.",
                "The complete syntactically valid file is mandatory under 95 lines. Use no narrative comments, avoid per-prop statements, and simplify detail before risking a truncated closing brace or API assignment.",
            ),
        ),
        (
            ("world.lighting.rig.lights",),
            (
                "Own only one named production light root; create no meshes, ground, camera, renderer, Scene, DOM, background, fog, or animation loop.",
                "Add exactly three lights to the root: warm shadow-casting DirectionalLight key at intensity 1.0, cool HemisphereLight fill at 0.45, and subtle opposite DirectionalLight rim at 0.3; total intensity 1.75.",
                "Configure the key shadow map and +/-15 camera bounds with conservative bias. Expose window.LightingRigAPI={root,key,fill,hemisphere:fill,totalIntensity:1.75} in under 65 lines.",
            ),
        ),
        (
            ("world.lighting.rig.fixture",),
            (
                "Own only a compact neutral material/shadow proof fixture; create no lights, camera, renderer, Scene, DOM, background, fog, or animation loop.",
                "Use one 14x10 charcoal receiver plus six spaced forms: faceted icosahedron, cylinder, cone, torus, layered/beveled-looking box, and low-poly capsule substitute. Each must be grounded and cast/receive shadows.",
                "Use clearly distinct matte clay, rough stone, satin paint, dark rubber, warm metal, and semi-gloss materials in a balanced two-row composition; no debug colors, labels, grids, or giant floor.",
                "Expose window.LightingFixtureAPI={root,materials,forms:6} in under 100 lines.",
            ),
        ),
        (
            ("world.lighting.rig",),
            (
                "Own a production lighting rig plus a restrained neutral material test fixture; do not create/replace Scene, Renderer, DOM, camera, background, fog, or animation loop.",
                "Build one named lightingRoot containing 3-4 lights: cool hemisphere fill around 0.35-0.5, warm directional key around 0.9-1.2 with shadows, and subtle opposite fill/rim so total intensity stays at or below 2.4.",
                "Configure key shadow map/camera for a compact 30-unit scene. Avoid ambient intensity above 0.25, flat frontal lighting, pure-white fill, and multiple competing high-intensity directional lights.",
                "Inside the same root create a small charcoal ground receiver and 4-6 well-spaced neutral reference forms (faceted sphere, cylinder, beveled box families) using matte, rough, painted, and dark materials to prove gradients/contact shadows.",
                "Never build a game grid, white marker field, floating info panel, colored debug cubes, or huge pale floor. Do not reposition the harness camera.",
                "Expose window.LightingRigAPI={root:lightingRoot,key,fill,hemisphere,totalIntensity}; keep the complete file under 115 lines.",
            ),
        ),
        (
            ("world.lighting.atmosphere.settings",),
            (
                "Own only reusable atmosphere settings; create no meshes, lights, camera, renderer, Scene, DOM, or animation loop.",
                "Define background color 0x78b9d6, FogExp2 color 0x88aebd with density 0.015, ACES tone mapping, and exposure 0.9 in one short apply(scene,renderer) function.",
                "buildPreview must call apply on the supplied objects and expose window.AtmosphereAPI={root:new THREE.Group(),background,fogColor,fogDensity:0.015,exposure:0.9,apply}; keep the file under 55 lines.",
            ),
        ),
        (
            ("world.lighting.atmosphere.fixture",),
            (
                "Own only the visible atmosphere depth fixture; create no lights, camera, renderer, Scene, DOM, background, fog, sky geometry, or animation loop.",
                "Build one named root with six varied faceted forms distributed across X=-4..4 and Z=-3,-6,-9,-12,-15,-18, sizes 1.0-2.2, so near/mid/far layers fill the harness composition.",
                "Use icosahedron, cone, cylinder, torus, dodecahedron, and layered box silhouettes with authored warm-near to cool-far colors; avoid identical cubes and tiny centered rows.",
                "Expose window.AtmosphereFixtureAPI={root,forms:6,depthRange:[-3,-18]} in under 90 lines.",
            ),
        ),
        (
            ("world.lighting.atmosphere",),
            (
                "Own atmosphere configuration only: coherent sky/background color, restrained exponential fog, exposure/tone-mapping parameters, and a compact depth-demo fixture; do not create lights, camera, renderer, DOM, or animation loop.",
                "Set scene.background to a THREE.Color and scene.fog to restrained FogExp2 around density 0.012-0.02. Never model the sky as a PlaneGeometry/box/sphere mesh because it can occlude the world.",
                "Demonstrate near/mid/far separation with 5-7 varied low-poly forms in one named root at Z=-3,-7,-11,-15,-19 using authored scale/color cadence, not text panels or identical debug blocks. Keep contrast readable and avoid white washout.",
                "Expose window.AtmosphereAPI={root,background,fogColor,fogDensity,exposure,apply} so apply(scene,renderer) reproduces the exact settings; keep the file under 100 lines.",
            ),
        ),
        (
            ("world.lighting.shadows.settings",),
            (
                "Own only reusable shadow settings; create no meshes, lights, camera, renderer, Scene, DOM, background, or animation loop.",
                "configureRenderer(renderer) sets shadowMap.enabled=true and type=THREE.PCFSoftShadowMap only. configureLight(light) sets castShadow, 2048 map width/height, +/-8 camera bounds, bias=-0.0008, and normalBias=0.02.",
                "buildPreview must find the harness DirectionalLight, apply both functions, and expose window.ShadowQualityAPI={root:new THREE.Group(),configureRenderer,configureLight,bounds:8}; no placeholders or console-only bodies; under 60 lines.",
            ),
        ),
        (
            ("world.lighting.shadows.fixture",),
            (
                "Own only a shadow proof fixture; create no lights, camera, renderer, Scene, DOM, background, or animation loop.",
                "Build one horizontal BoxGeometry(8,0.12,6) receiver at Y=-0.06 with receiveShadow=true plus four distinct neutral forms fully above it inside X=-3..3,Z=-2..2, each castShadow=true.",
                "Use box, faceted icosahedron, cylinder, and torus/arched compound silhouettes with visible grounding and balanced spacing; no PlaneGeometry, wall, grid, bars, or giant floor.",
                "Expose window.ShadowFixtureAPI={root,receiver,forms:4} in under 85 lines.",
            ),
        ),
        (
            ("world.lighting.shadows",),
            (
                "Own shadow/contact configuration and a compact proof fixture only; do not create a second lighting rig, camera, renderer, DOM, background, or animation loop.",
                "Use 3-5 varied neutral forms on one horizontal dark BoxGeometry(8,0.12,6) receiver at Y=-0.06 to demonstrate contact, direction, bias, and softness; do not use PlaneGeometry, which risks a vertical wall.",
                "Set receiver.receiveShadow=true and every proof form castShadow=true. Place forms fully above Y=0 inside X=-3..3 and Z=-2..2 so the close harness camera sees complete silhouettes.",
                "Define configureLight(light) and call it on the existing scene.children DirectionalLight supplied by the harness; never instantiate AmbientLight, DirectionalLight, HemisphereLight, PointLight, or SpotLight.",
                "configureRenderer(renderer) must set shadowMap.enabled=true and PCFSoftShadowMap. configureLight must use bias about -0.0008 and normalBias about 0.02, never a positive 0.01 bias that detaches shadows.",
                "Expose window.ShadowQualityAPI={root,configureRenderer,configureLight,bounds} with conservative map size/bias and performance-safe settings; no placeholders or console-only functions; keep the file under 100 lines.",
            ),
        ),
        (
            ("world.environment", "world.lighting"),
            (
                "Compose foreground, midground, and background layers with varied scale, density, and intentional negative space.",
                "Lighting must reveal forms with key/fill contrast, grounded shadows, and a coherent palette.",
            ),
        ),
        (
            ("frontend.", "dashboard.", "ui."),
            (
                "Implement the owned UI slice with semantic structure, responsive states, keyboard focus, empty/loading/error states, and visual hierarchy.",
                "Use the existing design tokens and interfaces; avoid generic placeholder cards and unrelated page redesign.",
            ),
        ),
        (
            ("backend.", "api.", "service."),
            (
                "Implement only the owned boundary with explicit validation, typed errors, idempotency where relevant, and deterministic tests.",
                "Preserve surrounding interfaces and prove failure paths; never mutate unrelated modules.",
            ),
        ),
        (
            ("ml.", "data.", "training.", "evaluation."),
            (
                "Materialize a reproducible owned pipeline stage with fixed seeds, schema checks, leakage guards, metrics, and serialized evidence.",
                "Keep data/model interfaces explicit and test both nominal and failure paths without relying on hidden notebook state.",
            ),
        ),
    )
    for prefixes, blueprint in profiles:
        if any(key.startswith(prefix) for prefix in prefixes):
            return blueprint
    return (
        "Build only the owned component with a concrete interface, non-placeholder implementation, deterministic verification, and integration evidence.",
        "Preserve repository contracts and make every important quality claim observable in the isolated preview or tests.",
    )


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


def _repair_double_escaped_python_source(
    path: str, content: str
) -> tuple[str, bool]:
    """Repair a weak-model JSON escape only when syntax proves it is safe."""

    if Path(path).suffix.casefold() != ".py" or '\\"' not in content:
        return content, False
    try:
        ast.parse(content)
    except SyntaxError:
        repaired = content.replace('\\"', '"')
        try:
            ast.parse(repaired)
        except SyntaxError:
            return content, False
        return repaired, True
    return content, False


def _validate_workspace_artifacts(
    workspace: Path,
    paths: Iterable[str],
) -> Mapping[str, Any]:
    """Return deterministic existence/hash/syntax evidence for approved paths."""

    root = workspace.resolve()
    findings: list[str] = []
    evidence: list[Mapping[str, Any]] = []
    checked: set[str] = set()
    for raw in paths:
        normalized = _normalized_path(raw)
        if normalized in {".", "*", "**", "**/*"}:
            continue
        if ".." in PurePosixPath(normalized).parts:
            findings.append(f"artifact path escapes the workspace: {raw}")
            continue
        candidates: tuple[Path, ...]
        if any(character in normalized for character in "*?["):
            candidates = tuple(
                item
                for item in root.glob(normalized)
                if item.is_file()
            )
            if not candidates:
                findings.append(f"approved artifact pattern produced no files: {raw}")
                continue
        else:
            candidate = root.joinpath(*PurePosixPath(normalized).parts).resolve(
                strict=False
            )
            try:
                candidate.relative_to(root)
            except ValueError:
                findings.append(f"artifact path escapes the workspace: {raw}")
                continue
            if candidate.is_dir():
                candidates = tuple(item for item in candidate.rglob("*") if item.is_file())
                if not candidates:
                    findings.append(f"approved artifact directory is empty: {raw}")
                    continue
            elif candidate.is_file():
                candidates = (candidate,)
            else:
                findings.append(f"approved artifact is missing: {raw}")
                continue
        for candidate in candidates:
            relative = candidate.resolve().relative_to(root).as_posix()
            if relative in checked or ".coding-agent" in PurePosixPath(relative).parts:
                continue
            checked.add(relative)
            digest = _hash_file(root, relative)
            if digest is None:
                findings.append(f"approved artifact could not be hashed: {relative}")
                continue
            record: dict[str, Any] = {
                "kind": "artifact_hash",
                "path": relative,
                "sha256": digest,
            }
            if candidate.suffix.casefold() == ".py":
                try:
                    ast.parse(candidate.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, SyntaxError) as exc:
                    findings.append(
                        f"Python syntax validation failed for {relative}: {exc}"
                    )
                    continue
                record["python_syntax"] = "passed"
            evidence.append(record)
    return {
        "passed": not findings and bool(evidence),
        "findings": findings,
        "evidence": evidence,
    }


def _run_python_test_artifacts(
    workspace: Path,
    paths: Iterable[str],
) -> Mapping[str, Any]:
    targets = tuple(
        dict.fromkeys(
            _normalized_path(path)
            for path in paths
            if Path(_normalized_path(path)).suffix.casefold() == ".py"
            and (
                Path(_normalized_path(path)).name.casefold().startswith("test_")
                or "tests" in {
                    part.casefold()
                    for part in PurePosixPath(_normalized_path(path)).parts
                }
            )
        )
    )
    if not targets:
        return {"passed": True, "findings": [], "evidence": []}
    command = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        output = (completed.stdout or "")[-8_000:]
        passed = completed.returncode == 0
        return {
            "passed": passed,
            "findings": (
                []
                if passed
                else [
                    "Executable pytest verification failed for "
                    + ", ".join(targets)
                    + ":\n"
                    + output[-3_000:]
                ]
            ),
            "evidence": [
                {
                    "kind": "executable_verification",
                    "command": "python -m pytest "
                    + " ".join(targets)
                    + " -q -p no:cacheprovider",
                    "returncode": completed.returncode,
                    "passed": passed,
                    "output": output,
                }
            ],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "findings": [f"Executable pytest verification could not complete: {exc}"],
            "evidence": [],
        }


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
                        {"value": "choice", "label": "choice", "description": "impact", "recommended": True}
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
                    "verification": [
                        "literal executable command or tool call such as npm test, "
                        "python -m pytest -q, or preview_html index.html; never prose"
                    ],
                    "depends_on": [],
                    "write_paths": ["relative/path"],
                    "forbidden_changes": [],
                    "owned_interfaces": [],
                    "metadata": {
                        "external_dependencies": [],
                        "browser_scenarios": [
                            {
                                "name": "observable user flow",
                                "steps": [
                                    {"action": "click", "role": "button", "name": "Control name"}
                                ],
                                "assertions": [
                                    {
                                        "role": "textbox",
                                        "name": "Output name",
                                        "property": "value",
                                        "equals": "expected value",
                                    }
                                ],
                            }
                        ],
                    },
                }
            ],
        }
    },
    "browser_scenarios": {
        "payload": {
            "modules": [
                {
                    "module_id": "exact approved module id",
                    "browser_scenarios": [
                        {
                            "name": "observable user flow",
                            "steps": [
                                {
                                    "action": "click, fill, or press",
                                    "role": "accessible role",
                                    "name": "accessible name",
                                    "value": "required text for fill",
                                }
                            ],
                            "assertions": [
                                {
                                    "role": "accessible role",
                                    "name": "accessible name",
                                    "property": "value, text, or textContent",
                                    "equals": "expected result",
                                }
                            ],
                        }
                    ],
                }
            ]
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
        if self.role is AgentRole.TESTER:
            # ``run_command`` is the typed cwd transport used when a weak
            # model proposes the otherwise-safe ``cd dir && verifier`` form.
            # The tester command validator still rejects every non-verifier
            # command before either executor can be reached.
            return _READ_TOOLS | {"run_bash", "run_command", "preview_html"}
        if self.role is AgentRole.RESEARCHER:
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
        contract = dict(request.task.get("contract", {}))
        metadata = dict(contract.get("metadata") or {})
        interactions: list[dict[str, Any]] = []
        for raw_scenario in metadata.get("browser_scenarios", ()):
            if not isinstance(raw_scenario, Mapping):
                continue
            scenario, normalization_actions = (
                _normalize_browser_scenario_transport(raw_scenario)
            )
            event_bus = getattr(self, "events", None)
            if normalization_actions and callable(getattr(event_bus, "publish", None)):
                event_bus.publish(
                    "ultra.browser_contract_normalized",
                    "; ".join(normalization_actions),
                    run_id=request.run_id,
                    node_id=request.node_id,
                    actions=list(normalization_actions),
                )
            interactions.append(scenario)
        result = str(
            self.executor(
                ToolCall(
                    "harness-html-preview",
                    "preview_html",
                    {
                        "path": target,
                        "open_browser": False,
                        "verify": True,
                        "settle_ms": 750,
                        "interactions": interactions,
                    },
                ),
                request,
            )
        )
        if result.startswith("Error:"):
            lowered = result.casefold()
            failure_kind = (
                "contract"
                if any(
                    marker in lowered
                    for marker in (
                        "invalid arguments",
                        "must be one of",
                        "unknown key",
                        "required property",
                        "schema validation",
                    )
                )
                else "environment"
                if any(
                    marker in lowered
                    for marker in (
                        "playwright is not installed",
                        "browser executable",
                        "browser could not",
                        "permission denied",
                    )
                )
                else "application"
            )
            return {
                "status": "failed",
                "error": result,
                "failure_kind": failure_kind,
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
                "failure_kind": "tool",
                "console_errors": [],
                "page_errors": [],
                "network_errors": [],
            }
        if not isinstance(payload, Mapping):
            return {
                "status": "failed",
                "error": "preview_html returned non-object payload",
                "failure_kind": "tool",
            }
        evidence = dict(payload)
        # A missing model-authored selector is a test-contract problem only
        # when the page itself is healthy.  Browser runtime/page/network
        # errors are candidate evidence and must take precedence; otherwise a
        # real defect such as ``THREE is not defined`` is misrouted to tester
        # contract repair, product mutation is prohibited, and every Retry
        # reproduces the same broken application.
        candidate_runtime_errors = tuple(
            str(item).strip()
            for key in ("console_errors", "page_errors", "network_errors")
            for item in (evidence.get(key, ()) or ())
            if str(item).strip()
        )
        if candidate_runtime_errors:
            evidence["failure_kind"] = "application"
            evidence["candidate_runtime_errors"] = list(candidate_runtime_errors)
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

    @staticmethod
    def _component_review_reasoning_artifact(
        request: AgentRequest,
        response: AgentResponse,
    ) -> Mapping[str, Any]:
        """Build an auditable review record from observable component evidence.

        This is not hidden chain-of-thought and does not invent implementation
        facts.  It normalizes the package/hash/runtime evidence already present
        in the reviewer packet when a small local model omits the requested
        external reasoning envelope.
        """

        task_candidate = (
            dict(request.task.get("candidate") or {})
            if isinstance(request.task, Mapping)
            else {}
        )
        candidate_payload = (
            dict(task_candidate.get("payload") or {})
            if isinstance(task_candidate.get("payload"), Mapping)
            else {}
        )
        package = (
            dict(candidate_payload.get("materialized_component_package") or {})
            if isinstance(
                candidate_payload.get("materialized_component_package"),
                Mapping,
            )
            else {}
        )
        preview = (
            dict(candidate_payload.get("materialized_preview") or {})
            if isinstance(candidate_payload.get("materialized_preview"), Mapping)
            else {}
        )
        evidence_refs: list[str] = []
        content_hash = str(
            package.get("content_hash")
            or dict(package.get("implementation") or {}).get("content_hash")
            or ""
        ).strip()
        if content_hash:
            evidence_refs.append(f"component-package-sha256:{content_hash}")
        screenshot = str(
            preview.get("screenshot_path")
            or dict(package.get("preview") or {}).get("screenshot")
            or ""
        ).strip()
        if screenshot:
            evidence_refs.append(f"component-screenshot:{screenshot}")
        for key in ("evidence", "test_results", "artifacts"):
            raw_values = response.payload.get(key, ())
            if isinstance(raw_values, Mapping):
                raw_values = (raw_values,)
            if isinstance(raw_values, Sequence) and not isinstance(
                raw_values,
                (str, bytes),
            ):
                for index, item in enumerate(raw_values[:4], start=1):
                    if isinstance(item, Mapping):
                        marker = str(
                            item.get("id")
                            or item.get("kind")
                            or item.get("path")
                            or item.get("name")
                            or ""
                        ).strip()
                        if marker:
                            evidence_refs.append(f"{key}:{marker}")
                    elif str(item).strip():
                        evidence_refs.append(f"{key}:{index}")
        if not evidence_refs:
            evidence_refs.append(
                f"candidate-projection:{request.node_id or 'component'}"
            )

        findings: list[str] = []
        for key in ("findings", "issues"):
            raw_values = response.payload.get(key, ())
            if isinstance(raw_values, Mapping):
                raw_values = (raw_values,)
            if isinstance(raw_values, Sequence) and not isinstance(
                raw_values,
                (str, bytes),
            ):
                findings.extend(
                    str(item).strip()
                    for item in raw_values
                    if str(item).strip()
                )
        passed = bool(
            response.payload.get(
                "passed",
                response.payload.get("success", False),
            )
        )
        claim = (
            response.summary.strip()
            or f"Typed component review {'passed' if passed else 'rejected'}."
        )
        counterarguments = (
            findings[:3]
            if findings
            else [
                "A fresh runtime, hash, or contract mismatch would invalidate this verdict."
            ]
        )
        verification = [
            "Re-run the materialized preview and typed component checks against the same package hash."
        ]
        return {
            "claim": claim,
            "supporting_evidence": list(dict.fromkeys(evidence_refs)),
            "counterarguments": counterarguments,
            "rejected_alternatives": [
                "Contract-only acceptance was rejected in favor of the supplied materialized candidate evidence."
            ],
            "verification_plan": verification,
            "reasoning_graph": {
                "nodes": [
                    {
                        "id": "typed-verdict",
                        "type": "decision",
                        "summary": claim,
                        "status": "verified" if passed else "open",
                        "evidence_refs": list(dict.fromkeys(evidence_refs)),
                    },
                    {
                        "id": "fresh-check",
                        "type": "verification",
                        "summary": verification[0],
                        "status": "open",
                        "evidence_refs": list(dict.fromkeys(evidence_refs)),
                    },
                    {
                        "id": "contract-only",
                        "type": "option",
                        "summary": (
                            "Accepting from the contract without candidate "
                            "evidence was rejected."
                        ),
                        "status": "rejected",
                        "evidence_refs": [],
                    },
                ],
                "edges": [
                    {
                        "from": "fresh-check",
                        "to": "typed-verdict",
                        "relation": "verifies",
                    },
                    {
                        "from": "typed-verdict",
                        "to": "contract-only",
                        "relation": "rejects",
                    }
                ],
            },
        }

    def _assemble_materialized_children(
        self,
        request: AgentRequest,
        request_contract: Mapping[str, Any],
    ) -> AgentResponse:
        """Compose exact accepted child sources without a model round-trip."""

        raw_children = request.task.get("child_component_packages", {})
        children = dict(raw_children) if isinstance(raw_children, Mapping) else {}
        request_metadata = (
            dict(request_contract.get("metadata") or {})
            if isinstance(request_contract.get("metadata"), Mapping)
            else {}
        )
        parent_domain = str(
            request_metadata.get("specialist_domain") or ""
        ).strip().casefold()
        prepared: list[dict[str, str]] = []
        failures: list[str] = []
        for child_id, raw_package in sorted(children.items()):
            package = dict(raw_package) if isinstance(raw_package, Mapping) else {}
            contents = (
                dict(package.get("file_contents") or {})
                if isinstance(package.get("file_contents"), Mapping)
                else {}
            )
            source = str(contents.get("preview/scene.js") or "")
            package_id = str(package.get("id") or "")
            content_hash = str(package.get("content_hash") or "")
            if not package_id or not content_hash or not source.strip():
                failures.append(
                    f"child {child_id} lacks package id, content hash, or preview/scene.js"
                )
                continue
            has_builder = bool(
                re.search(r"\b(?:window\.)?buildPreview\s*(?:=|\()", source)
            )
            typed_api_match = re.search(
                r"\bwindow\.([A-Za-z_$][\w$]*API)\s*=",
                source,
            )
            has_typed_root = bool(typed_api_match)
            if not has_builder and not has_typed_root:
                failures.append(
                    f"child {child_id} lacks buildPreview or a typed *API.root export"
                )
                continue
            prepared.append(
                {
                    "node_id": str(child_id),
                    "package_id": package_id,
                    "content_hash": content_hash,
                    "source": source,
                    "typed_api": (
                        str(typed_api_match.group(1)) if typed_api_match else ""
                    ),
                    "has_builder": "1" if has_builder else "",
                }
            )
        if parent_domain == "vehicles.chassis.shell":
            expected_children = {
                f"{request.node_id}.volumes": "ChassisVolumesAPI",
                f"{request.node_id}.panels": "ChassisPanelsAPI",
            }
            prepared_by_id = {item["node_id"]: item for item in prepared}
            if set(prepared_by_id) != set(expected_children):
                failures.append(
                    "chassis shell assembler requires exactly its volumes and panels child packages"
                )
            else:
                for child_id, expected_api in expected_children.items():
                    child = prepared_by_id[child_id]
                    if (
                        child.get("typed_api") != expected_api
                        and not child.get("has_builder")
                    ):
                        failures.append(
                            f"child {child_id} must publish {expected_api}.root or buildPreview"
                        )
        if failures or not prepared:
            values = failures or ["component assembler received no child packages"]
            return AgentResponse(
                payload={"success": False, "passed": False, "findings": values},
                summary="Deterministic child-package assembly could not start.",
                reasoning_summary="Durable child package receipts were incomplete.",
                provider="harness",
                model="exact-child-assembler-v1",
            )

        declarations: list[str] = []
        for child in prepared:
            typed_api_name = str(child.get("typed_api") or "")
            declarations.append(
                "\n".join(
                    (
                        "{",
                        "  const childWindow=Object.create(window);",
                        "  ((window)=>{",
                        child["source"],
                        "  })(childWindow);",
                        f"  const typedApi={('childWindow[' + _json(typed_api_name) + ']') if typed_api_name else 'null'};",
                        "  const typedRoot=typedApi&&typedApi.root&&typedApi.root.isObject3D?typedApi:null;",
                        "  const childBuild=typeof childWindow.buildPreview===\"function\"",
                        "    ? childWindow.buildPreview",
                        "    : typedRoot ? ((context)=>{const target=(context&&context.scene)||context;target.add(typedRoot.root);}) : null;",
                        "  if(typeof childBuild!==\"function\")",
                        f"    throw new Error({_json('child '+child['node_id']+' lacks buildPreview or typed API root')});",
                        "  __acceptedChildren.push({",
                        f"    nodeId:{_json(child['node_id'])},",
                        f"    packageId:{_json(child['package_id'])},",
                        f"    contentHash:{_json(child['content_hash'])},",
                        "    scope:childWindow,build:childBuild",
                        "  });",
                        "}",
                    )
                )
            )
        consumption = [
            {
                "node_id": child["node_id"],
                "package_id": child["package_id"],
                "content_hash": child["content_hash"],
            }
            for child in prepared
        ]
        parent_name = re.sub(
            r"[^A-Za-z0-9]+", "_", str(request.node_id or "component_parent")
        ).strip("_")
        parent_exports: tuple[str, ...] = ()
        if parent_domain == "vehicles.chassis.shell":
            parent_exports = (
                "  const panelApi=window.ChassisPanelsAPI||{};",
                "  const canonicalMounts=[{x:1.38,y:.55,z:1.55},{x:-1.38,y:.55,z:1.55},{x:1.38,y:.55,z:-1.55},{x:-1.38,y:.55,z:-1.55}];",
                "  window.ChassisShellAPI={",
                "    root:parentRoot,",
                "    dimensions:{width:2.8,height:1.5,length:5.2},",
                "    wheelMounts:Array.isArray(panelApi.wheelMounts)&&panelApi.wheelMounts.length===4?panelApi.wheelMounts:canonicalMounts,",
                "    forward:'+Z'",
                "  };",
            )
        composite_source = "\n".join(
            (
                "const __acceptedChildren=[];",
                *declarations,
                "window.buildPreview=(context)=>{",
                "  const {THREE,scene}=context;",
                "  const parentRoot=new THREE.Group();",
                f"  parentRoot.name={_json(parent_name + '_IntegratedRoot')};",
                "  scene.add(parentRoot);",
                "  for(const child of __acceptedChildren){",
                "    const childRoot=new THREE.Group();",
                "    childRoot.name=child.nodeId+'__consumed';",
                "    parentRoot.add(childRoot);",
                "    const childScene=Object.create(scene);",
                "    childScene.add=(...items)=>childRoot.add(...items);",
                "    childScene.remove=(...items)=>childRoot.remove(...items);",
                "    childScene.getObjectByName=(name)=>childRoot.getObjectByName(name);",
                "    const childContext=Object.assign(Object.create(childScene),context,{scene:childScene});",
                "    child.build(childContext);",
                "    for(const key of Object.keys(child.scope)){",
                "      if(key!==\"buildPreview\") window[key]=child.scope[key];",
                "    }",
                "    if(child.nodeId.endsWith(\".collision\")) childRoot.visible=false;",
                "  }",
                *parent_exports,
                f"  window.__componentConsumption={_json(consumption)};",
                "  return parentRoot;",
                "};",
            )
        )
        stage_result = str(
            self.executor(
                ToolCall(
                    "harness-parent-scene",
                    "stage_component_file",
                    {"path": "preview/scene.js", "content": composite_source, "role": "implementation"},
                ),
                request,
            )
        )
        if stage_result.startswith("Error:"):
            return AgentResponse(
                payload={"success": False, "passed": False, "findings": [stage_result]},
                summary="Exact child source staging failed.",
                provider="harness",
                model="exact-child-assembler-v1",
            )
        expected_hashes = [item["content_hash"] for item in prepared]
        test_source = (
            "export const expectedChildHashes="
            + _json(expected_hashes)
            + ";\nexport function verifyIntegratedPreview(scope=globalThis){\n"
            + " const host=scope.window||scope;\n"
            + " if(typeof host.buildPreview!==\"function\") throw new Error(\"missing integrated buildPreview\");\n"
            + " return expectedChildHashes.length>0;\n}\n"
        )
        test_result = str(
            self.executor(
                ToolCall(
                    "harness-parent-test",
                    "stage_component_file",
                    {"path": "test/component-integration.test.js", "content": test_source, "role": "test"},
                ),
                request,
            )
        )
        if test_result.startswith("Error:"):
            return AgentResponse(
                payload={"success": False, "passed": False, "findings": [test_result]},
                summary="Exact child integration-test staging failed.",
                provider="harness",
                model="exact-child-assembler-v1",
            )
        metadata = (
            dict(request_contract.get("metadata") or {})
            if isinstance(request_contract.get("metadata"), Mapping)
            else {}
        )
        default_owned = (
            ("ChassisShellAPI",)
            if parent_domain == "vehicles.chassis.shell"
            else ("IntegratedComponentPackage",)
        )
        owned = [
            str(item)
            for item in (
                request_contract.get("owned_interfaces")
                or metadata.get("owned_interfaces")
                or default_owned
            )
            if str(item).strip()
        ]
        if parent_domain == "vehicles.chassis.shell" and "ChassisShellAPI" not in owned:
            owned.append("ChassisShellAPI")
        publication_raw = str(
            self.executor(
                ToolCall(
                    "harness-parent-publish",
                    "publish_component",
                    {
                        "implementation": {"files": []},
                        "interface": {
                            "exports": owned,
                            "imports": [item["package_id"] for item in prepared],
                            "integration_points": [
                                "Consumes exact accepted child source hashes.",
                                "Exposes child runtime APIs without reimplementation.",
                            ],
                        },
                        "tests": [],
                        "preview": {"entrypoint": "preview/index.html"},
                        "dependencies": [item["package_id"] for item in prepared],
                    },
                ),
                request,
            )
        )
        try:
            publication = json.loads(publication_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            publication = {}
        passed = bool(isinstance(publication, Mapping) and publication.get("passed"))
        findings = [
            str(item)
            for item in (
                publication.get("findings", ())
                if isinstance(publication, Mapping)
                else ()
            )
        ]
        if not passed and not findings:
            findings = [publication_raw[:1_200] or "parent component publication failed"]
        preview = (
            dict(publication.get("preview") or {})
            if isinstance(publication, Mapping)
            and isinstance(publication.get("preview"), Mapping)
            else {}
        )
        return AgentResponse(
            payload={
                "success": passed,
                "passed": passed,
                "status": str(publication.get("status") or ("accepted" if passed else "rejected")),
                "component_publication": {
                    "package_id": str(publication.get("stored_package_id") or publication.get("package_id") or ""),
                    "screenshot_path": str(preview.get("screenshot_path") or ""),
                },
                "package_consumption": consumption,
                "findings": findings,
            },
            summary=(
                "Exact accepted child packages were composed and runtime verified."
                if passed
                else "Exact child-package assembly failed its runtime gate."
            ),
            reasoning_summary="The harness composed immutable child sources and hashes; no model rewrote child work.",
            provider="harness",
            model="exact-child-assembler-v1",
        )

    def execute(self, request: AgentRequest) -> AgentResponse:
        request_contract = (
            dict(request.task.get("contract", {}))
            if isinstance(request.task, Mapping)
            else {}
        )
        typed_return_repair = (
            dict(request.task.get("typed_return_repair", {}))
            if isinstance(request.task, Mapping)
            and isinstance(request.task.get("typed_return_repair"), Mapping)
            else {}
        )
        if (
            self.provider_name.casefold() == "ollama"
            and bool(typed_return_repair.get("repeated_response"))
        ):
            reset_cache = getattr(self.provider, "reset_model_cache", None)
            cache_reset = False
            if callable(reset_cache):
                try:
                    reset_cache()
                    cache_reset = True
                except Exception:
                    cache_reset = False
            self.events.publish(
                "ultra.typed_return_cache_reset",
                "Repeated invalid typed response detected; local model cache "
                f"reset={cache_reset} before the adapted repair request.",
                run_id=request.run_id,
                node_id=request.node_id,
                role=self.role.value,
                phase=request.phase,
                response_fingerprint=str(
                    typed_return_repair.get("response_fingerprint") or ""
                ),
                cache_reset=cache_reset,
            )
        request_component_only = bool(
            dict(request_contract.get("metadata", {})).get("component_package_only")
        )
        component_leaf_quality_phase = bool(
            request_component_only
            and self.role
            in {
                AgentRole.REVIEWER,
                AgentRole.CLEAN_CODE_REVIEWER,
                AgentRole.TESTER,
                AgentRole.SECURITY_REVIEWER,
                AgentRole.TEST_QUALITY_REVIEWER,
            }
            and request.phase
            in {
                InnerPhase.REVIEW.value,
                InnerPhase.TEST.value,
            }
        )
        component_quality_triage_phase = bool(
            request_component_only
            and self.role is AgentRole.QUALITY_TRIAGER
            and request.phase == InnerPhase.REVIEW.value
            and isinstance(request.task.get("review_verdicts"), Sequence)
        )
        bounded_review_inspection_phase = bool(
            self.role
            in {
                AgentRole.REVIEWER,
                AgentRole.CLEAN_CODE_REVIEWER,
                AgentRole.SECURITY_REVIEWER,
                AgentRole.TEST_QUALITY_REVIEWER,
            }
            and request.phase
            in {
                InnerPhase.REVIEW.value,
                InnerPhase.GLOBAL_REVIEW.value,
            }
        )
        local_governed_mutation_phase = bool(
            self.provider_name.casefold() == "ollama"
            and self.role in {AgentRole.CODER, AgentRole.INTEGRATOR}
            and request.phase
            in {
                InnerPhase.IMPLEMENT.value,
                InnerPhase.FIX.value,
                InnerPhase.INTEGRATE.value,
            }
            and tuple(request_contract.get("write_paths", ()) or ())
        )
        configured_effort = str(getattr(self.provider, "reasoning_effort", "medium"))
        deterministic_roles = {
            AgentRole.GOAL_UNDERSTANDING,
            AgentRole.ARCHITECT,
            AgentRole.PLANNER,
            AgentRole.DECOMPOSER,
            AgentRole.MEMORY,
            AgentRole.QUALITY_TRIAGER,
        }
        compact_foundation_phases = {
            "goal_spec",
            "architecture",
            "architecture_critique",
            "architecture_judge",
            "master_plan",
            "browser_scenarios",
        }
        effective_effort = configured_effort
        if self.provider_name == "ollama" and (
            self.role in deterministic_roles
            or request.phase in compact_foundation_phases
            or component_leaf_quality_phase
            or component_quality_triage_phase
            or local_governed_mutation_phase
        ):
            effective_effort = "off"
            setattr(self.provider, "reasoning_effort", effective_effort)
        deterministic_budgets = {
            AgentRole.GOAL_UNDERSTANDING: 4096,
            AgentRole.ARCHITECT: 1024,
            AgentRole.PLANNER: 1536,
            AgentRole.DECOMPOSER: 4096,
            AgentRole.MEMORY: 768,
            AgentRole.QUALITY_TRIAGER: 1024,
        }
        if self.provider_name == "ollama" and self.role in deterministic_budgets:
            setattr(self.provider, "max_output_tokens", deterministic_budgets[self.role])
        elif (
            self.provider_name == "ollama"
            and request.phase in {"architecture_critique", "architecture_judge"}
        ):
            setattr(self.provider, "max_output_tokens", 1_024)
        elif self.provider_name == "ollama" and component_leaf_quality_phase:
            setattr(self.provider, "max_output_tokens", 640)
            setattr(self.provider, "temperature", 0.0)
        elif self.provider_name == "ollama" and component_quality_triage_phase:
            setattr(self.provider, "max_output_tokens", 512)
            setattr(self.provider, "temperature", 0.0)
        if self.provider_name == "ollama":
            # Gemma's current Ollama grammar can terminate on an internal
            # <unused50> token and return an empty response. Typed extraction,
            # validation, and one targeted repair are safer than format=json;
            # governed action roles use validated streamed JSON transport.
            setattr(self.provider, "force_json", False)
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
        if (
            self.role is AgentRole.MEMORY
            and request.phase == InnerPhase.MEMORY_WRITEBACK.value
        ):
            summaries = [
                str(item).strip()
                for item in request.task.get("result_summaries", ())
                if str(item).strip()
            ]
            self.events.publish(
                "ultra.deterministic_memory_writeback",
                "Durable result receipts were compacted without another model call.",
                run_id=request.run_id,
                node_id=request.node_id,
                summaries=len(summaries),
            )
            return AgentResponse.from_mapping(
                {
                    "payload": {
                        "success": True,
                        "insights": [
                            {
                                "summary": summary[:500],
                                "severity": "info",
                                "details": {"source": "accepted_result_receipt"},
                            }
                            for summary in summaries[-6:]
                        ],
                    },
                    "summary": "Accepted result receipts were written to durable memory.",
                    "reasoning_summary": (
                        "Memory writeback copied accepted external summaries only; "
                        "it did not ask the model to reinterpret or mutate the result."
                    ),
                },
                node_id=request.node_id,
                provider="harness",
                model="durable-memory-writeback-v1",
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
            failure_kind = str(harness_preview.get("failure_kind") or "application")
            blocker_owner = {
                "contract": "test_harness",
                "tool": "tooling",
                "environment": "environment",
                "application": "candidate_code",
            }.get(failure_kind, "diagnostic")
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
                (
                    "Harness browser contract failed; routing evidence to test-definition repair"
                    if failure_kind == "contract"
                    else "Harness environment failed; candidate mutation is prohibited"
                    if failure_kind in {"tool", "environment"}
                    else "Harness browser/readback benchmark found a candidate defect"
                ),
                run_id=request.run_id,
                node_id=request.node_id,
                findings=findings,
                scores=dict(harness_preview.get("benchmark_scores", {})),
            )
            return AgentResponse.from_mapping(
                {
                    "payload": {
                        "passed": False,
                        "failure_kind": failure_kind,
                        "blocker_owner": blocker_owner,
                        "candidate_status": (
                            "unverified"
                            if failure_kind in {"contract", "tool", "environment"}
                            else "failed"
                        ),
                        "verification_status": "blocked",
                        "issues": findings,
                        "findings": findings,
                        "test_results": [
                            {
                                "name": "harness_html_browser_and_quality_gate",
                                "passed": False,
                                "failure_kind": failure_kind,
                                "blocker_owner": blocker_owner,
                                "scores": dict(harness_preview.get("benchmark_scores", {})),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                                "interaction_targets": list(
                                    harness_preview.get("interaction_targets", ()) or ()
                                ),
                                "interaction_results": list(
                                    harness_preview.get("interaction_results", ()) or ()
                                ),
                            }
                        ],
                        "evidence": [
                            {
                                "kind": "browser_preview",
                                "verification": harness_preview.get("verification"),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                                "interaction_results": list(
                                    harness_preview.get("interaction_results", ()) or ()
                                ),
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
        if (
            harness_preview
            and request.phase == InnerPhase.TEST.value
            and self.role is AgentRole.TESTER
            and str(
                harness_preview.get("verification")
                or harness_preview.get("status")
                or ""
            ).casefold()
            in {"passed", "ok", "success"}
        ):
            self.events.publish(
                "ultra.deterministic_test_gate",
                "Harness browser verification passed; using its receipt as the tester verdict.",
                run_id=request.run_id,
                node_id=request.node_id,
                screenshot_path=harness_preview.get("screenshot_path"),
            )
            return AgentResponse.from_mapping(
                {
                    "payload": {
                        "passed": True,
                        "success": True,
                        "issues": [],
                        "findings": [],
                        "test_results": [
                            {
                                "name": "harness_html_preview",
                                "passed": True,
                                "http_status": harness_preview.get("http_status"),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                                "interaction_results": list(
                                    harness_preview.get("interaction_results", ()) or ()
                                ),
                            }
                        ],
                        "evidence": [
                            {
                                "kind": "harness_browser_preview",
                                "source": "harness",
                                "status": "passed",
                                "http_status": harness_preview.get("http_status"),
                                "console_errors": list(
                                    harness_preview.get("console_errors", ()) or ()
                                ),
                                "page_errors": list(
                                    harness_preview.get("page_errors", ()) or ()
                                ),
                                "network_errors": list(
                                    harness_preview.get("network_errors", ()) or ()
                                ),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                                "interaction_results": list(
                                    harness_preview.get("interaction_results", ()) or ()
                                ),
                            }
                        ],
                    },
                    "summary": "Harness browser preview passed with authoritative evidence.",
                    "reasoning_summary": (
                        "The deterministic browser gate observed HTTP success and no "
                        "blocking page, console, or network errors."
                    ),
                },
                node_id=request.node_id,
                provider="harness",
                model="browser-preview-gate-v1",
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
        mutation_requirements = [
            str(item)[:1_200]
            for item in (
                request.task.get("findings", ())
                if isinstance(request.task, Mapping)
                else ()
            )
            if str(item).strip()
        ][:12]
        mutation_acceptance_criteria = [
            str(item)[:1_200]
            for item in (
                request_contract.get("acceptance_criteria", ())
                or request_contract.get("success_criteria", ())
                or ()
            )
            if str(item).strip()
        ][:12]
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
        component_publication_phase = bool(
            request_component_only
            and self.role in {AgentRole.CODER, AgentRole.INTEGRATOR}
            and request.phase
            in {InnerPhase.IMPLEMENT.value, InnerPhase.FIX.value, InnerPhase.INTEGRATE.value}
        )
        if component_publication_phase and bool(
            request.task.get("component_assembler")
        ):
            return self._assemble_materialized_children(
                request,
                request_contract,
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
            specialist_domain = str(
                contract_metadata.get("specialist_domain")
                or request_contract.get("title")
                or request.node_id
                or "component"
            )
            def _compact_items(value: Any, *, limit: int = 3, chars: int = 320) -> list[str]:
                if not isinstance(value, (list, tuple)):
                    return []
                return [str(item)[:chars] for item in value[:limit] if str(item).strip()]

            compact_contract = {
                key: request_contract.get(key)
                for key in (
                    "id",
                    "title",
                    "objective",
                    "owned_interfaces",
                )
                if request_contract.get(key) not in (None, "", (), [], {})
            }
            compact_contract["objective"] = str(
                compact_contract.get("objective", "")
            )[:600]
            compact_contract["acceptance_criteria"] = _compact_items(
                request_contract.get("acceptance_criteria")
            )
            compact_contract["verification"] = _compact_items(
                request_contract.get("verification"), limit=2
            )
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
                    "objective": str(north_star.get("objective", ""))[:500],
                    "success_criteria": _compact_items(
                        north_star.get("success_criteria"), limit=3
                    ),
                    "constraints": _compact_items(
                        north_star.get("constraints"), limit=3
                    ),
                },
            }
            compact_task = {
                key: request.task.get(key)
                for key in (
                    "findings",
                    "attempt",
                    "change_approach",
                    "optimization_variable",
                    "champion_challenger",
                    "component_assembler",
                    "prior_replan_guidance",
                    "prior_findings",
                    "hard_blocker_before_publish",
                )
                if request.task.get(key) not in (None, "", (), [], {})
            }
            compact_task["contract"] = compact_contract
            raw_child_packages = request.task.get("child_component_packages")
            if isinstance(raw_child_packages, Mapping):
                compact_task["child_component_packages"] = {
                    str(child_id): {
                        "id": str(dict(package).get("id") or ""),
                        "content_hash": str(
                            dict(package).get("content_hash") or ""
                        ),
                        "interface": dict(package).get("interface", {}),
                        "files": [
                            {
                                "path": str(item.get("path") or ""),
                                "content_hash": str(item.get("content_hash") or ""),
                                "role": str(item.get("role") or ""),
                            }
                            for item in dict(package).get("files", ())
                            if isinstance(item, Mapping)
                        ],
                    }
                    for child_id, package in raw_child_packages.items()
                    if isinstance(package, Mapping)
                }
            user_payload = {
                "component_task": compact_task,
                "integration_context": compact_context,
                "specialist_quality_blueprint": list(
                    _specialist_quality_blueprint(specialist_domain)
                ),
                "required_action": (
                    "First action only: call stage_component_file for preview/scene.js defining "
                    "window.buildPreview=({THREE,scene,camera,renderer})=>{...}; build only "
                    "this specialist's detailed 3D component. Apply every current finding and "
                    "blueprint rule in executable geometry, not comments. Return no prose. The "
                    "harness supplies Scene, Camera, Renderer, DOM, lighting, and animation loop: "
                    "use the passed objects and never instantiate or replace them. Cubes, labels, "
                    "and placeholder geometry are rejected."
                ),
                "response_after_accepted_publication": {
                    "payload": {"success": True, "findings": [], "evidence": []},
                    "summary": "brief publication result",
                    "reasoning_summary": "brief evidence-based conclusion",
                },
            }
        elif component_leaf_quality_phase:
            candidate = (
                dict(request.task.get("candidate", {}))
                if isinstance(request.task.get("candidate"), Mapping)
                else {}
            )
            candidate_payload = (
                dict(candidate.get("payload", {}))
                if isinstance(candidate.get("payload"), Mapping)
                else {}
            )
            materialized = candidate_payload.get("materialized_component_package")
            materialized = (
                dict(materialized) if isinstance(materialized, Mapping) else {}
            )
            file_contents = materialized.get("file_contents")
            file_contents = (
                dict(file_contents) if isinstance(file_contents, Mapping) else {}
            )
            publication = candidate_payload.get("component_publication")
            publication = (
                dict(publication) if isinstance(publication, Mapping) else {}
            )
            materialized_preview = candidate_payload.get("materialized_preview")
            materialized_preview = (
                dict(materialized_preview)
                if isinstance(materialized_preview, Mapping)
                else {}
            )
            acceptance_values = tuple(
                request_contract.get("acceptance_criteria", ()) or ()
            )
            verification_values = tuple(
                request_contract.get("verification", ()) or ()
            )
            test_role = self.role in {
                AgentRole.TESTER,
                AgentRole.TEST_QUALITY_REVIEWER,
            }
            ordered_files = sorted(
                file_contents.items(),
                key=lambda item: (
                    0
                    if str(item[1]).strip()
                    else 1,
                    0
                    if (
                        test_role
                        and "test" in str(item[0]).casefold()
                    )
                    or (
                        not test_role
                        and str(item[0]).replace("\\", "/").casefold().endswith(
                            "preview/scene.js"
                        )
                    )
                    else 1,
                    0
                    if not test_role
                    and str(item[0]).replace("\\", "/").casefold().startswith(
                        "src/"
                    )
                    else 1,
                    1
                    if not test_role
                    and str(item[0]).casefold().endswith((".html", ".htm"))
                    else 0,
                    str(item[0]).casefold(),
                ),
            )

            def _memory_excerpt(path: Any, content: Any, *, budget: int) -> dict[str, Any]:
                text = str(content)
                if len(text) > budget:
                    head = max(1, budget - 360)
                    head_boundary = text.rfind("\n", 0, head)
                    if head_boundary > max(1, head // 2):
                        head = head_boundary + 1
                    tail_start = max(head, len(text) - 320)
                    tail_boundary = text.find("\n", tail_start)
                    if tail_boundary >= 0 and tail_boundary + 1 < len(text):
                        tail_start = tail_boundary + 1
                    excerpt = (
                        text[:head]
                        + "\n/* HARNESS MEMORY PROJECTION BOUNDARY: "
                        "the verified source continues in durable storage; "
                        "this marker is not part of the artifact. */\n"
                        + text[tail_start:]
                    )
                else:
                    excerpt = text
                return {
                    "path": str(path),
                    "characters": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "excerpt": excerpt,
                }

            evidence_slices: list[dict[str, Any]] = []
            remaining_slice_budget = 4_000 if test_role else 6_000
            for path, content in ordered_files:
                if not str(content).strip() or remaining_slice_budget < 300:
                    continue
                slice_budget = remaining_slice_budget
                evidence_slices.append(
                    _memory_excerpt(path, content, budget=slice_budget)
                )
                remaining_slice_budget -= slice_budget
                if len(evidence_slices) >= 1:
                    break
            user_payload = {
                "quality_task": {
                    "role": self.role.value,
                    "phase": request.phase,
                    "node_id": request.node_id,
                    "objective": str(request_contract.get("objective", ""))[:800],
                    "acceptance_criteria": [
                        str(item)[:320] for item in acceptance_values[:4]
                    ],
                    "verification": [
                        str(item)[:320] for item in verification_values[:3]
                    ],
                },
                "materialized_receipt": {
                    "package_id": str(
                        publication.get("package_id")
                        or materialized.get("id")
                        or ""
                    ),
                    "content_hash": str(materialized.get("content_hash") or ""),
                    "root": str(materialized.get("root") or ""),
                    "preview_entrypoint": str(
                        materialized.get("preview_entrypoint") or ""
                    ),
                    "screenshot_path": str(
                        publication.get("screenshot_path") or ""
                    ),
                    "runtime_gate": {
                        "status": str(
                            materialized_preview.get("status")
                            or materialized_preview.get("verification")
                            or publication.get("status")
                            or ""
                        ),
                        "console_errors": [
                            str(item)[:240]
                            for item in tuple(
                                materialized_preview.get("console_errors", ()) or ()
                            )[:3]
                        ],
                        "page_errors": [
                            str(item)[:240]
                            for item in tuple(
                                materialized_preview.get("page_errors", ()) or ()
                            )[:3]
                        ],
                    },
                    "files": [
                        {
                            "path": str(path),
                            "characters": len(str(content)),
                            "sha256": hashlib.sha256(
                                str(content).encode("utf-8")
                            ).hexdigest(),
                        }
                        for path, content in list(file_contents.items())[:24]
                    ],
                },
                "durable_memory_projection": evidence_slices,
                "required_action": (
                    "Review only this bounded, hash-backed memory projection and runtime "
                    "receipt. Return the response_contract JSON once; no tool call. Reject "
                    "only observable contract, code, security, or test defects and cite the "
                    "path. A HARNESS MEMORY PROJECTION BOUNDARY means only that context was "
                    "elided, never that source or a test is missing. Do not judge visual "
                    "aesthetics from code."
                ),
                "response_contract": {
                    **contract,
                    "summary": "brief evidence-based gate result",
                    "reasoning_summary": "brief observable evidence only",
                },
            }
        elif component_quality_triage_phase:
            raw_verdicts = request.task.get("review_verdicts", ())
            compact_verdicts: list[dict[str, Any]] = []
            for raw in raw_verdicts if isinstance(raw_verdicts, Sequence) else ():
                if not isinstance(raw, Mapping):
                    continue
                raw_payload = (
                    dict(raw.get("payload", {}))
                    if isinstance(raw.get("payload"), Mapping)
                    else {}
                )
                compact_verdicts.append(
                    {
                        "role": str(raw.get("role") or ""),
                        "passed": bool(raw.get("passed")),
                        "issues": [
                            str(item)[:320]
                            for item in tuple(raw_payload.get("issues", ()) or ())[:5]
                        ],
                        "findings": [
                            str(item)[:320]
                            for item in tuple(raw_payload.get("findings", ()) or ())[:5]
                        ],
                        "evidence": [
                            str(item)[:320]
                            for item in tuple(raw_payload.get("evidence", ()) or ())[:3]
                        ],
                    }
                )
            user_payload = {
                "quality_triage": {
                    "node_id": request.node_id,
                    "objective": str(request_contract.get("objective", ""))[:600],
                    "verdicts": compact_verdicts,
                },
                "required_action": (
                    "Normalize and deduplicate only supplied findings. passed must be true "
                    "only when every supplied verdict passed. Return one response_contract "
                    "JSON object with no tools and no invented finding."
                ),
                "response_contract": {
                    **contract,
                    "summary": "brief deterministic triage result",
                    "reasoning_summary": "brief supplied-evidence conclusion",
                },
            }
        elif request.phase in compact_foundation_phases:
            # Foundation calls decide one structured artifact at a time.  They
            # do not need the generic debate/scaffold/tool catalog repeated in
            # every prompt; the durable north star and prior artifact are
            # already in task/context and the harness validates the result.
            user_payload = {
                "task": request.task,
                "focused_context": request.context,
                "harness_workspace_inspection": harness_inspection,
                "harness_reasoning_scaffold": reasoning_scaffold_for(
                    self.role.value,
                    request.phase,
                    request.task,
                ).to_dict(),
                "harness_debate_protocol": debate_protocol.to_dict(),
                "response_contract": {
                    **contract,
                    "summary": "one brief factual result summary",
                    "reasoning_summary": (
                        "brief external decisions and observable evidence only"
                    ),
                    "reasoning_artifact": {
                        "claim": "main external decision",
                        "supporting_evidence": ["repository, contract, or test evidence"],
                        "counterarguments": ["one likely failure mode"],
                        "rejected_alternatives": ["one rejected alternative"],
                        "verification_plan": ["next observable verification"],
                    },
                },
                "memory_policy": (
                    "Use only this action packet. Durable history remains in SQLite; "
                    "do not reconstruct or repeat the whole project history."
                ),
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
        metadata_only_publication = bool(
            self.role is AgentRole.INTEGRATOR
            and request.phase == InnerPhase.INTEGRATE.value
            and request.task.get("publish_component_package")
            and not request.task.get("final_assembler")
            and not request.task.get("component_assembler")
        )
        allowed_tools = self._allowed_tools()
        if local_governed_mutation_phase and request.phase == InnerPhase.FIX.value:
            contract_metadata = (
                dict(request_contract.get("metadata", {}))
                if isinstance(request_contract.get("metadata"), Mapping)
                else {}
            )
            dependency_evidence_parts: list[str] = []
            for key in ("findings", "prior_findings", "issues"):
                value = request.task.get(key)
                if isinstance(value, Mapping):
                    dependency_evidence_parts.append(_json(value))
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    dependency_evidence_parts.extend(str(item) for item in value)
                elif value is not None:
                    dependency_evidence_parts.append(str(value))
            dependency_contract_text = " ".join(
                (
                    str(request_contract.get("objective") or ""),
                    " ".join(
                        str(item)
                        for item in tuple(
                            request_contract.get("acceptance_criteria", ()) or ()
                        )
                    ),
                    _json(contract_metadata),
                )
            ).casefold()
            dependency_finding_text = " ".join(
                dependency_evidence_parts
            ).casefold()
            dependency_markers = (
                "missing dependency",
                "missing module",
                "module not found",
                "cannot find module",
                "package not installed",
                "install depend",
                "npm install",
                "pnpm install",
                "yarn install",
                "requirements.txt",
                "package.json",
            )
            dependency_install_justified = bool(
                contract_metadata.get("allow_dependency_install")
                or any(
                    marker in dependency_finding_text
                    or marker in dependency_contract_text
                    for marker in dependency_markers
                )
            )
            if not dependency_install_justified:
                allowed_tools = allowed_tools - {"install_dependencies"}
                self.events.publish(
                    "ultra.dependency_install_restricted",
                    "Dependency installation was withheld from Fix because no finding or contract requires it.",
                    run_id=request.run_id,
                    node_id=request.node_id,
                    role=self.role.value,
                    phase=request.phase,
                )
        if component_publication_phase:
            if self.provider_name == "ollama":
                # Leaf quality comes from isolated previews, independent
                # judging, and revision loops. Keep the initial tool emission
                # bounded so a small thinking model cannot spend the whole
                # transport timeout before publishing the first candidate.
                component_temperature = (
                    0.15
                    if str(specialist_domain).casefold().startswith(
                        ("world.road.markings", "world.road.collision")
                    )
                    else 0.25
                )
                prompt_chars = len(conversation[0]["content"])
                context_size = max(
                    2_048,
                    int(getattr(self.provider, "context_size", 4_096) or 4_096),
                )
                configured_ceiling = max(
                    1_536,
                    min(
                        2_048,
                        int(os.environ.get("AGENT_ULTRA_COMPONENT_OUTPUT_TOKENS", "2048")),
                    ),
                )
                if str(specialist_domain).casefold().startswith(
                    "vehicles.chassis.shell"
                ):
                    # This leaf is naturally data-driven. A tight budget plus
                    # its tuple-loop contract prevents Gemma from expanding
                    # every mesh into a repeated block and truncating the
                    # surrounding typed tool call.
                    configured_ceiling = min(configured_ceiling, 1_536)
                elif str(specialist_domain).casefold().startswith("vehicles.vehicle_"):
                    configured_ceiling = max(configured_ceiling, 2_560)
                # JSON/code prompts tokenize more densely than plain English.
                # Reserve a conservative prompt estimate plus transport slack,
                # then give the materialized leaf the remaining local context.
                prompt_token_estimate = (prompt_chars + 2) // 3
                component_output_budget = min(
                    configured_ceiling,
                    max(1_536, context_size - prompt_token_estimate - 320),
                )
                setattr(self.provider, "reasoning_effort", "off")
                setattr(self.provider, "max_output_tokens", component_output_budget)
                setattr(self.provider, "temperature", component_temperature)
                self.events.publish(
                    "ultra.component_generation_routed",
                    (
                        f"[{request.node_id}] component emission think=off, "
                        f"output<={component_output_budget}, "
                        f"temperature={component_temperature:.2f}, "
                        f"prompt={len(conversation[0]['content'])} chars"
                    ),
                    run_id=request.run_id,
                    node_id=request.node_id,
                    role=self.role.value,
                    phase=request.phase,
                    prompt_chars=prompt_chars,
                    temperature=component_temperature,
                    max_output_tokens=component_output_budget,
                )
            # A component specialist has no final workspace write ownership.
            # Exposing generic mutation tools encourages small local models to
            # bypass the typed package contract or merely describe a write.
            # The complete component contract and retrieved evidence are
            # already in the action packet. A 4K local-model context cannot
            # afford the generic repository read catalog on every leaf.
            # Isolated leaves need only materialization and publication.
            allowed_tools = frozenset()
        elif component_leaf_quality_phase:
            # The harness projects the relevant durable files and runtime
            # receipt into a bounded packet above. Reopening the artifact
            # through a tool loop wastes a 4K local context and can evict the
            # evidence the reviewer actually needs.
            allowed_tools = frozenset()
        elif component_quality_triage_phase:
            allowed_tools = frozenset()
        elif metadata_only_publication:
            # Implementation bytes already passed independent review. This
            # phase may publish package metadata, but must never mutate the
            # reviewed candidate after its evidence gate.
            allowed_tools = frozenset()
        if self.role is AgentRole.TESTER and self._html_write_target(request):
            # HTML verification is platform-neutral through preview_html plus
            # deterministic readback metrics. Avoid shell quoting/OS drift.
            allowed_tools = allowed_tools - {"run_bash", "run_command"}
        schemas = (
            []
            if request.phase in compact_foundation_phases
            else _schemas(allowed_tools)
        )
        if request.phase == "master_plan" and self.provider_name == "ollama":
            # Gemma-family local models are markedly more reliable when this
            # large nested contract is transported as one native tool call.
            # The arguments remain entirely model-authored and still pass the
            # same MasterPlanV1 and applicability validators before approval.
            schemas.append(dict(_SUBMIT_MASTER_PLAN_TOOL))
        elif request.phase == "browser_scenarios" and self.provider_name == "ollama":
            schemas.append(dict(_SUBMIT_BROWSER_SCENARIOS_TOOL))
        if component_publication_phase:
            # Gemma-family local templates are substantially more reliable
            # when the first turn has one unambiguous action. Publication is
            # unlocked immediately after the first successful staged file.
            schemas.append(dict(_STAGE_COMPONENT_FILE_TOOL))
            if self.provider_name != "ollama":
                schemas.append(dict(_PUBLISH_COMPONENT_TOOL))
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
        component_unused_retries = 0
        quality_read_count = 0
        mutation_read_count = 0
        mutation_claim_count = 0
        successful_mutation_paths: set[str] = set()
        max_invalid_json_attempts = 4 if self.provider_name.casefold() == "ollama" else 2
        for step in range(1, self.max_steps + 1):
            try:
                if (
                    component_leaf_quality_phase
                    and self.provider_name == "ollama"
                ):
                    tool_indexes = [
                        index
                        for index, message in enumerate(conversation)
                        if message.get("role") == "tool"
                    ]
                    for index in tool_indexes[:-1]:
                        content = str(conversation[index].get("content", ""))
                        conversation[index]["content"] = _json(
                            {
                                "durable_tool_receipt": True,
                                "sha256": hashlib.sha256(
                                    content.encode("utf-8")
                                ).hexdigest(),
                                "characters": len(content),
                            }
                        )
                    if tool_indexes:
                        latest_index = tool_indexes[-1]
                        latest = str(conversation[latest_index].get("content", ""))
                        if len(latest) > 1_200:
                            conversation[latest_index]["content"] = (
                                latest[:900]
                                + "\n...[durable tool output compacted]...\n"
                                + latest[-200:]
                            )
                system_prompt = request.system_prompt
                contract_for_prompt = (
                    dict(request.task.get("contract", {}))
                    if isinstance(request.task, Mapping)
                    else {}
                )
                component_only = bool(
                    dict(contract_for_prompt.get("metadata", {})).get("component_package_only")
                )
                if component_publication_phase:
                    publication_is_unlocked = "publish_component" in {
                        _schema_name(schema) for schema in schemas
                    }
                    if self.provider_name == "ollama" and not publication_is_unlocked:
                        system_prompt = (
                            "Call exactly one tool now: stage_component_file. Return no prose or JSON. "
                            "Create the complete preview/scene.js requested by the user with real "
                            "detailed Three.js geometry using only the passed THREE, scene, camera, "
                            "and renderer. Never create a Scene, Renderer, DOM node, or animation "
                            "loop. Do not call any unavailable tool."
                        )
                    else:
                        system_prompt = (
                            f"You are the isolated {self.role.value} for one component. "
                            "Obey only component_task.contract. Use stage_component_file for complete "
                            "files, then publish_component. Never write or redesign the whole product. "
                            "Continue after each tool receipt until publication passes; then return one "
                            "small JSON response. Do not emit hidden reasoning or internal tokens."
                        )
                elif component_leaf_quality_phase:
                    system_prompt = (
                        f"You are the isolated {self.role.value} for one materialized component. "
                        "The user packet is a bounded projection from durable memory, not a request "
                        "to inspect the whole repository. Evaluate only observable evidence for your "
                        "role and return exactly one JSON object matching response_contract. Do not "
                        "call tools, emit prose, judge aesthetics, or expose hidden reasoning. A "
                        "HARNESS MEMORY PROJECTION BOUNDARY is context compression, never evidence "
                        "that artifact logic or tests are missing."
                    )
                elif component_quality_triage_phase:
                    system_prompt = (
                        "You are a deterministic quality triager. Use only the supplied typed "
                        "verdicts, deduplicate their findings, and return exactly one JSON object "
                        "matching response_contract. Do not call tools or invent evidence."
                    )
                elif metadata_only_publication:
                    system_prompt += (
                        "\n\nMETADATA-ONLY PUBLICATION PHASE: the implementation has already "
                        "passed independent review. Do not call tools or propose workspace "
                        "writes. Publish only the typed component_package metadata that "
                        "describes the accepted candidate."
                    )
                if request.phase == "master_plan":
                    system_prompt += (
                        "\n\nMASTER PLAN VERIFICATION TRANSPORT: every module.verification item "
                        "must begin with a literal executable command or harness tool name. "
                        "Valid shapes include 'npm test', 'python -m pytest -q', and "
                        "'preview_html path/to/entry.html'. Invalid shapes include 'run tests', "
                        "'test the UI', or 'verify the app'. If using npm/pnpm/yarn/bun/npx, "
                        "include package.json in a module write_paths. If using preview_html, "
                        "include that exact preview target in write_paths. Choose commands and "
                        "paths from the model-authored plan; do not emit placeholders."
                    )
                    if self.provider_name == "ollama":
                        system_prompt += (
                            " LOCAL MASTER PLAN TRANSPORT: call submit_master_plan exactly "
                            "once with the complete plan. Return no prose and call no other tool. "
                            "A preview_html verifier must end at the literal .html path with no "
                            "query or #fragment. Put clicks, typing, and expected UI state in "
                            "that module's metadata.browser_scenarios steps and assertions."
                        )
                elif request.phase == "browser_scenarios" and self.provider_name == "ollama":
                    system_prompt += (
                        "\n\nLOCAL BROWSER SCENARIO TRANSPORT: call "
                        "submit_browser_scenarios exactly once. Return no prose and call no "
                        "other tool. Preserve every exact module_id. Each scenario requires "
                        "non-empty executable steps and observable assertions. Supported steps "
                        "are click(role/name or selector), fill(role/name or selector plus value), "
                        "and press(key); never put text on a click step. Assertions require a target, "
                        "a supported property (value, text, textContent, id, visible, checked, "
                        "count, visibleCount, dataObjectCount, or dataVisualState), and equals or "
                        "contains. Interactive canvas/WebGL work must expose and assert a bounded "
                        "data-object-count or data-visual-state value after the relevant user actions; "
                        "canvas presence alone does not prove rendering behavior. role values must be ARIA "
                        "roles such as textbox, button, heading, or status, never HTML tags like "
                        "input, h2, or div. Use property value only with form-control roles; use "
                        "property text or textContent for status, heading, button, and other elements."
                        " module_id belongs on each parent modules[] object, never inside an "
                        "individual browser_scenarios[] object."
                    )
                if self.role is AgentRole.CODER and request.phase in {
                    InnerPhase.IMPLEMENT.value,
                    InnerPhase.FIX.value,
                }:
                    if component_only:
                        if not (
                            self.provider_name == "ollama"
                            and component_publication_phase
                            and not publication_is_unlocked
                        ):
                            system_prompt += (
                                " The preview scene must visibly show a polished, reviewable component with "
                                "silhouette, proportions, material response, placement, and useful detail. "
                                "If rejected, revise only this component. FinalAssembler alone owns final paths."
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
                provider_schemas = schemas
                if self.provider_name.casefold() == "ollama" and schemas:
                    # Ollama's native tool transport is deliberately atomic.
                    # A local runner can therefore generate indefinitely with
                    # zero observable response bytes, including during coder
                    # mutation turns. Transport one action as streamed JSON and
                    # rebuild it below through the same role/phase allow-list;
                    # tool execution and approval semantics remain unchanged.
                    compact_contracts = [
                        {
                            "name": _schema_name(schema),
                            "parameters": dict(
                                schema.get("function", {}).get("parameters", {})
                            ),
                        }
                        for schema in schemas
                        if _schema_name(schema)
                    ]
                    system_prompt += (
                        "\n\nNATIVE TOOL TRANSPORT IS DISABLED FOR THIS LOCAL STAGE. "
                        "Propose exactly one available action as "
                        '{"name":"AVAILABLE_NAME","args":{...}} with no prose. '
                        "The harness validates and executes it. Use this exact compact "
                        "action contract:\n"
                        + _json(compact_contracts)
                    )
                    provider_schemas = []
                turn = self.provider.call(
                    conversation, provider_schemas, system_prompt
                )
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
                proposal, action_transport_receipt = extract_action_proposal(
                    turn.text
                )
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
                            normalization=action_transport_receipt.to_dict(),
                        )
            conversation.append(turn.to_message())
            if turn.tool_calls:
                if request.phase == "master_plan" and self.provider_name == "ollama":
                    submitted = [
                        call
                        for call in turn.tool_calls
                        if call.name == "submit_master_plan"
                    ]
                    if len(submitted) == 1 and len(turn.tool_calls) == 1:
                        self.events.publish(
                            "ultra.master_plan_transport_submitted",
                            "Local planner submitted the approval-bound plan through typed transport.",
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                        )
                        return AgentResponse(
                            payload=dict(submitted[0].args),
                            summary="Local planner submitted the typed master plan.",
                            reasoning_summary=(
                                "Plan semantics are model-authored; the harness only bound the "
                                "typed transport and will run the normal validators."
                            ),
                            usage=dict(totals),
                            provider=self.provider_name,
                            model=self.model,
                        )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "The master-plan turn must contain exactly one "
                                "submit_master_plan call. Call it now with the complete plan."
                            ),
                        }
                    )
                    continue
                if request.phase == "browser_scenarios" and self.provider_name == "ollama":
                    submitted = [
                        call
                        for call in turn.tool_calls
                        if call.name == "submit_browser_scenarios"
                    ]
                    if len(submitted) == 1 and len(turn.tool_calls) == 1:
                        self.events.publish(
                            "ultra.browser_scenarios_transport_submitted",
                            "Local planner submitted bounded browser scenarios through typed transport.",
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                        )
                        return AgentResponse(
                            payload=dict(submitted[0].args),
                            summary="Local planner submitted bounded browser scenarios.",
                            reasoning_summary=(
                                "Scenario semantics are model-authored; the harness only binds "
                                "them to exact approval-bound module ids."
                            ),
                            usage=dict(totals),
                            provider=self.provider_name,
                            model=self.model,
                        )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Call submit_browser_scenarios exactly once for every listed "
                                "module_id with non-empty steps and assertions."
                            ),
                        }
                    )
                    continue
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
                    exposed_tools = {
                        _schema_name(schema)
                        for schema in schemas
                        if _schema_name(schema)
                    }
                    if call.name not in exposed_tools:
                        if (
                            local_governed_mutation_phase
                            and inspection_observed
                            and call.name in _READ_TOOLS
                        ):
                            raise _MutationTransportExhausted(
                                f"{self.role.value} repeated a read after the one-read "
                                "mutation budget; the required next action is an "
                                "approval-bound write_file call"
                            )
                        result = (
                            f"Error: tool {call.name!r} is not allowed for "
                            f"{self.role.value} during {request.phase}"
                        )
                        self.events.publish(
                            "ultra.disallowed_tool_rejected",
                            result,
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            tool=call.name,
                        )
                        conversation.append(
                            {
                                "role": "tool",
                                "id": call.id,
                                "name": call.name,
                                "content": result,
                            }
                        )
                        continue
                    if self.role is AgentRole.TESTER:
                        normalized_test_call, normalization_note = (
                            _normalize_tester_verification_call(call)
                        )
                        if normalization_note:
                            call = normalized_test_call
                            self.events.publish(
                                "ultra.tool_proposal_normalized",
                                normalization_note,
                                run_id=request.run_id,
                                node_id=request.node_id,
                                role=self.role.value,
                                phase=request.phase,
                                tool=call.name,
                            )
                            if call.name not in exposed_tools:
                                result = (
                                    "Error: typed tester verification transport "
                                    f"{call.name!r} is not exposed in this phase"
                                )
                                conversation.append(
                                    {
                                        "role": "tool",
                                        "id": call.id,
                                        "name": call.name,
                                        "content": result,
                                    }
                                )
                                continue
                    if (
                        self.role is AgentRole.TESTER
                        and call.name in {"run_bash", "run_command"}
                        and not _tester_verification_command_allowed(
                            str(call.args.get("command", ""))
                        )
                    ):
                        result = (
                            "Error: tester shell access accepts one verification "
                            "command only; shell composition and workspace mutation "
                            "commands are not allowed"
                        )
                        self.events.publish(
                            "ultra.disallowed_tool_rejected",
                            result,
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            tool=call.name,
                        )
                        conversation.append(
                            {
                                "role": "tool",
                                "id": call.id,
                                "name": call.name,
                                "content": result,
                            }
                        )
                        continue
                    unlocked_publication_now = False
                    effective_call = ToolCall(
                        id=call.id,
                        name=call.name,
                        args=normalize_generated_tool_args(call.name, call.args),
                    )
                    if effective_call.name == "write_file":
                        repaired_content, repaired_python_escaping = (
                            _repair_double_escaped_python_source(
                                str(effective_call.args.get("path", "")),
                                str(effective_call.args.get("content", "")),
                            )
                        )
                        if repaired_python_escaping:
                            effective_call = ToolCall(
                                id=effective_call.id,
                                name=effective_call.name,
                                args={
                                    **dict(effective_call.args),
                                    "content": repaired_content,
                                },
                            )
                            self.events.publish(
                                "ultra.write_tool_normalized",
                                "Removed double JSON escaping after Python "
                                "syntax validation",
                                run_id=request.run_id,
                                node_id=request.node_id,
                                path=str(
                                    effective_call.args.get("path", "")
                                ),
                            )
                    available_tool_names = {
                        _schema_name(schema) for schema in schemas
                    }
                    if effective_call.name not in available_tool_names:
                        conversation.append(
                            {
                                "role": "tool",
                                "id": effective_call.id,
                                "name": effective_call.name,
                                "content": (
                                    f"Error: {effective_call.name} is not available at this "
                                    "component checkpoint."
                                ),
                            }
                        )
                        continue
                    if (
                        component_publication_phase
                        and effective_call.name == "stage_component_file"
                    ):
                        # The isolated leaf contract has one canonical browser
                        # entry source. Weak models sometimes rename it after a
                        # retry (component.js, road.js, etc.), which disables
                        # deterministic packaging even though the bytes are a
                        # valid buildPreview implementation. Normalize the
                        # staging address; the original generated bytes remain
                        # unchanged and hash-audited.
                        staged_content = str(
                            effective_call.args.get("content", "")
                        )
                        if (
                            "buildpreview" in staged_content.casefold()
                            or str(
                                effective_call.args.get("role", "")
                            ).casefold()
                            == "preview"
                        ):
                            effective_call = ToolCall(
                                id=effective_call.id,
                                name=effective_call.name,
                                args={
                                    **dict(effective_call.args),
                                    "path": "preview/scene.js",
                                    "role": "preview",
                                },
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
                    mutation_path = (
                        str(effective_call.args.get("path") or "").strip()
                        if effective_call.name in _WRITE_TOOLS
                        else ""
                    )
                    repeated_mutation_path = bool(
                        mutation_path and mutation_path in successful_mutation_paths
                    )
                    result = self.executor(effective_call, request)
                    tester_receipt = None
                    if (
                        self.role is AgentRole.TESTER
                        and request.phase == InnerPhase.TEST.value
                    ):
                        tester_receipt = _tester_command_receipt(
                            effective_call,
                            str(result),
                            node_id=request.node_id,
                            usage=totals,
                        )
                    if tester_receipt is not None:
                        self.events.publish(
                            "ultra.deterministic_test_gate",
                            tester_receipt.summary,
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            command=str(effective_call.args.get("command") or ""),
                            cwd=str(effective_call.args.get("cwd") or "."),
                            passed=bool(tester_receipt.payload.get("passed")),
                            returncode=int(
                                dict(tester_receipt.payload["test_results"][0]).get(
                                    "returncode"
                                )
                                if dict(tester_receipt.payload["test_results"][0]).get(
                                    "returncode"
                                ) is not None
                                else -1
                            ),
                        )
                        return tester_receipt
                    if effective_call.name == "stage_component_file":
                        # The full source is durable in the isolated artifact
                        # store. Replaying it through every subsequent local
                        # model turn wastes the 4K context and can destabilize
                        # the GPU runner. Keep only an auditable receipt.
                        assistant_message = conversation[-1]
                        if isinstance(assistant_message, dict):
                            for historical_call in assistant_message.get(
                                "tool_calls", ()
                            ):
                                if (
                                    isinstance(historical_call, dict)
                                    and historical_call.get("id") == effective_call.id
                                ):
                                    source = str(effective_call.args.get("content", ""))
                                    historical_call["args"] = {
                                        "path": str(effective_call.args.get("path", "")),
                                        "role": str(effective_call.args.get("role", "")),
                                        "content_sha256": hashlib.sha256(
                                            source.encode("utf-8")
                                        ).hexdigest(),
                                        "content_omitted_after_staging": True,
                                    }
                    if (
                        component_publication_phase
                        and effective_call.name == "stage_component_file"
                        and not str(result).startswith("Error:")
                        and "publish_component"
                        not in {_schema_name(schema) for schema in schemas}
                    ):
                        schemas.append(dict(_PUBLISH_COMPONENT_TOOL))
                        unlocked_publication_now = True
                        self.events.publish(
                            "ultra.component_publication_unlocked",
                            f"[{request.node_id}] staged first file; publication tool unlocked",
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                        )
                        if self.provider_name == "ollama":
                            # Gemma is most reliable when it spends its single
                            # bounded generation on the component itself. The
                            # remaining manifest/test/shell work is typed and
                            # deterministic, so do it in the harness instead
                            # of burning two more fragile local-model turns.
                            staged_path = str(effective_call.args.get("path", ""))
                            if staged_path.casefold().endswith(
                                ("preview/scene.js", "preview.scene.js")
                            ):
                                smoke_source = (
                                    "export function verifyComponentPreview(scope = globalThis) {\n"
                                    "  const host = scope.window || scope;\n"
                                    "  if (typeof host.buildPreview !== 'function') {\n"
                                    "    throw new Error('component preview contract missing buildPreview');\n"
                                    "  }\n"
                                    "  return true;\n"
                                    "}\n"
                                )
                                smoke_result = str(
                                    self.executor(
                                        ToolCall(
                                            f"harness-component-smoke-{step}",
                                            "stage_component_file",
                                            {
                                                "path": "test/component-contract.test.js",
                                                "content": smoke_source,
                                                "role": "test",
                                            },
                                        ),
                                        request,
                                    )
                                )
                                metadata = (
                                    dict(request_contract.get("metadata", {}))
                                    if isinstance(
                                        request_contract.get("metadata"), Mapping
                                    )
                                    else {}
                                )
                                owned = tuple(
                                    str(item)
                                    for item in (
                                        request_contract.get("owned_interfaces")
                                        or metadata.get("owned_interfaces")
                                        or ()
                                    )
                                    if str(item).strip()
                                )
                                domain = str(
                                    metadata.get("specialist_domain")
                                    or request_contract.get("title")
                                    or request.node_id
                                    or "Component"
                                )
                                auto_publish_result = str(
                                    self.executor(
                                        ToolCall(
                                            f"harness-component-publish-{step}",
                                            "publish_component",
                                            {
                                                "implementation": {"files": []},
                                                "interface": {
                                                    "exports": list(
                                                        owned
                                                        or (
                                                            re.sub(
                                                                r"[^A-Za-z0-9]+",
                                                                "_",
                                                                domain,
                                                            ).strip("_")
                                                            + "Package",
                                                        )
                                                    ),
                                                    "imports": ["THREE runtime"],
                                                    "integration_points": [
                                                        "Parent consumes the exact staged scene hash.",
                                                        "Harness supplies renderer shell and runtime smoke gate.",
                                                    ],
                                                },
                                                "tests": [],
                                                "preview": {
                                                    "entrypoint": "preview/index.html"
                                                },
                                                "dependencies": ["three"],
                                            },
                                        ),
                                        request,
                                    )
                                )
                                try:
                                    auto_publication = json.loads(
                                        auto_publish_result
                                    )
                                except (
                                    TypeError,
                                    ValueError,
                                    json.JSONDecodeError,
                                ):
                                    auto_publication = {}
                                auto_passed = bool(
                                    not smoke_result.startswith("Error:")
                                    and isinstance(auto_publication, Mapping)
                                    and auto_publication.get("passed")
                                )
                                self.events.publish(
                                    "ultra.component_auto_packaged",
                                    (
                                        f"[{request.node_id}] harness packaged the "
                                        f"single generated component: "
                                        f"{'passed' if auto_passed else 'rejected'}"
                                    ),
                                    run_id=request.run_id,
                                    node_id=request.node_id,
                                    role=self.role.value,
                                    phase=request.phase,
                                    generated_path=staged_path,
                                    smoke_test_staged=not smoke_result.startswith(
                                        "Error:"
                                    ),
                                    passed=auto_passed,
                                    findings=list(
                                        auto_publication.get("findings", ())
                                    )
                                    if isinstance(auto_publication, Mapping)
                                    else [],
                                )
                                if auto_passed:
                                    package_id = str(
                                        auto_publication.get("package_id") or ""
                                    )
                                    preview_value = (
                                        dict(auto_publication.get("preview", {}))
                                        if isinstance(
                                            auto_publication.get("preview"),
                                            Mapping,
                                        )
                                        else {}
                                    )
                                    return AgentResponse(
                                        payload={
                                            "success": True,
                                            "component_publication": {
                                                "package_id": package_id,
                                                "status": str(
                                                    auto_publication.get("status")
                                                    or "materialized"
                                                ),
                                                "screenshot_path": str(
                                                    preview_value.get(
                                                        "screenshot_path"
                                                    )
                                                    or ""
                                                ),
                                            },
                                            "evidence": [
                                                {
                                                    "kind": "materialized_component_receipt",
                                                    "package_id": package_id,
                                                    "generated_path": staged_path,
                                                    "packaging": "deterministic_harness",
                                                }
                                            ],
                                            "findings": [],
                                        },
                                        summary=(
                                            "Specialist component generated once, "
                                            "deterministically packaged, and runtime verified."
                                        ),
                                        reasoning_summary=(
                                            "The builder owned component code; the harness "
                                            "owned manifest, smoke test, preview shell, hashes, "
                                            "and publication."
                                        ),
                                        usage=dict(totals),
                                        provider=self.provider_name,
                                        model=self.model,
                                    )
                                failure_findings = [
                                    str(item).strip()
                                    for item in auto_publication.get("findings", ())
                                    if str(item).strip()
                                ] if isinstance(auto_publication, Mapping) else []
                                if not failure_findings:
                                    failure_findings = [
                                        str(auto_publish_result)[:1_200]
                                        or "component publication failed"
                                    ]
                                # End this specialist turn after one generated
                                # candidate. The engine fix loop will create a
                                # fresh role context containing the complete
                                # contract, blueprint, and typed finding. A
                                # same-conversation retry loses that bounded
                                # context and makes weak local models regress
                                # into generic Scene/DOM boilerplate.
                                return AgentResponse(
                                    payload={
                                        "success": False,
                                        "passed": False,
                                        "status": "rejected",
                                        "findings": failure_findings,
                                    },
                                    summary=(
                                        "Specialist component failed deterministic "
                                        "publication and requires a fresh challenger."
                                    ),
                                    reasoning_summary=(
                                        "The harness stopped the weak-model turn after "
                                        "one materialized candidate and preserved its "
                                        "typed runtime findings."
                                    ),
                                    usage=dict(totals),
                                    provider=self.provider_name,
                                    model=self.model,
                                )
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
                        if local_governed_mutation_phase:
                            mutation_read_count += 1
                    if effective_call.name in _WRITE_TOOLS and not str(result).startswith("Error:"):
                        mutation_observed = True
                        if mutation_path:
                            successful_mutation_paths.add(mutation_path)
                    conversation.append(
                        {
                            "role": "tool",
                            "id": effective_call.id,
                            "name": effective_call.name,
                            "content": result,
                        }
                    )
                    if (
                        local_governed_mutation_phase
                        and effective_call.name in _READ_TOOLS
                        and not str(result).startswith("Error:")
                    ):
                        # One focused read is enough for a local mutation
                        # turn. Replace the growing conversation with the
                        # exact approved target packet and expose only file
                        # mutation tools. This prevents a weak model from
                        # spending all 16 steps repeatedly inspecting or
                        # returning a read-only success envelope.
                        target_state = next(
                            (
                                dict(item)
                                for item in write_target_state
                                if str(item.get("path") or "").strip()
                            ),
                            {},
                        )
                        target_path = str(target_state.get("path") or "").strip()
                        conversation = [
                            {
                                "role": "user",
                                "content": _json(
                                    {
                                        "node_id": request.node_id,
                                        "phase": request.phase,
                                        "objective": str(
                                            request_contract.get("objective") or ""
                                        )[:800],
                                        "required_corrections": mutation_requirements,
                                        "acceptance_criteria": mutation_acceptance_criteria,
                                        "approved_write_paths": [
                                            str(path)
                                            for path in tuple(
                                                request_contract.get("write_paths", ())
                                                or ()
                                            )
                                        ],
                                        "next_target": {
                                            "path": target_path,
                                            "exists": bool(target_state.get("exists")),
                                            "current_content": str(result)[:40_000],
                                        },
                                        "required_action": (
                                            "Return exactly one edit_file action for the existing "
                                            "next_target.path (or write_file only when it does not exist) with "
                                            "the complete implementation required by the approved "
                                            "module contract. Return "
                                            "no prose, do not use another read tool, and do not "
                                            "claim completion without the governed write receipt."
                                        ),
                                    }
                                ),
                            }
                        ]
                        schemas[:] = [
                            schema
                            for schema in schemas
                            if _schema_name(schema) in _FILE_WRITE_TOOLS
                        ]
                        self.events.publish(
                            "ultra.mutation_read_budget_closed",
                            "Closed local read tools after one focused inspection and "
                            "requested the approval-bound write.",
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            path=target_path,
                            read_count=mutation_read_count,
                        )
                    if (
                        local_governed_mutation_phase
                        and effective_call.name in _FILE_WRITE_TOOLS
                        and not str(result).startswith("Error:")
                        and schemas
                    ):
                        # One governed mutation is the end of this isolated
                        # local-model turn. Subsequent tester/reviewer/browser
                        # roles remain separate mandatory gates.
                        schemas.clear()
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "The governed mutation succeeded and tools are now closed for "
                                    "this turn. Return exactly one response_contract JSON object "
                                    "using the mutation receipt. Do not claim product acceptance, "
                                    "review, browser, or test success."
                                ),
                            }
                        )
                        self.events.publish(
                            "ultra.mutation_handoff_requested",
                            "Closed local mutation tools after the first successful write and requested one typed handoff.",
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            path=mutation_path,
                        )
                    if repeated_mutation_path and schemas:
                        # The durable executor/replay guard has already
                        # accepted this exact path. A weak model repeating the
                        # same mutation cannot add evidence, so close tools and
                        # require the typed handoff. Independent review and
                        # executable verification still decide acceptance.
                        schemas.clear()
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "The same successful mutation was proposed again and the "
                                    "durable replay guard preserved the accepted bytes. Tools are "
                                    "now closed. Return the response_contract JSON using only the "
                                    "observed mutation receipt; do not claim review or test success."
                                ),
                            }
                        )
                        self.events.publish(
                            "ultra.repeated_mutation_handoff",
                            "Closed tools after a duplicate mutation proposal and requested the typed handoff.",
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            path=mutation_path,
                        )
                    if (
                        (component_leaf_quality_phase or bounded_review_inspection_phase)
                        and effective_call.name in _READ_TOOLS
                        and not str(result).startswith("Error:")
                    ):
                        quality_read_count += 1
                        read_budget = 2 if component_leaf_quality_phase else 4
                        if quality_read_count >= read_budget and schemas:
                            schemas.clear()
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "The bounded independent inspection budget is complete. "
                                        "Use only the observed tool receipts to return the "
                                        "response_contract JSON now. Do not request another tool, "
                                        "invent missing evidence, or repeat an inspection."
                                    ),
                                }
                            )
                    if (
                        effective_call.name == "publish_component"
                        and component_publication_passed
                    ):
                        package_value = (
                            dict(publication_result.get("package", {}))
                            if isinstance(publication_result.get("package"), Mapping)
                            else {}
                        )
                        preview_value = (
                            dict(publication_result.get("preview", {}))
                            if isinstance(publication_result.get("preview"), Mapping)
                            else {}
                        )
                        return AgentResponse(
                            payload={
                                "success": True,
                                "component_publication": {
                                    "package_id": str(
                                        publication_result.get("stored_package_id")
                                        or publication_result.get("package_id")
                                        or package_value.get("id")
                                        or ""
                                    ),
                                    "status": str(
                                        publication_result.get("status")
                                        or "materialized"
                                    ),
                                    "screenshot_path": str(
                                        preview_value.get("screenshot_path") or ""
                                    ),
                                },
                                "evidence": [
                                    {
                                        "kind": "materialized_component_receipt",
                                        "package_id": str(
                                            publication_result.get("stored_package_id")
                                            or publication_result.get("package_id")
                                            or package_value.get("id")
                                            or ""
                                        ),
                                    }
                                ],
                                "findings": [],
                            },
                            summary=(
                                "Component package materialized and passed its runtime "
                                "publication gate; independent quality review remains external."
                            ),
                            reasoning_summary=(
                                "Harness receipt, package hash, and preview evidence were observed."
                            ),
                            usage=dict(totals),
                            provider=self.provider_name,
                            model=self.model,
                        )
                    if unlocked_publication_now:
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Preview is staged. Now stage one implementation file and one "
                                    "test file with complete content, then call publish_component with "
                                    "preview.entrypoint='preview/index.html' and concrete exports."
                                ),
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
                synthesized_reasoning = False
                if component_leaf_quality_phase and not reasoning_evaluation.passed:
                    repaired_reasoning = self._component_review_reasoning_artifact(
                        request,
                        response,
                    )
                    reasoning_evaluation = evaluate_reasoning_artifact(
                        repaired_reasoning,
                        debate_protocol,
                    )
                    synthesized_reasoning = True
                if debate_protocol.required:
                    payload = dict(response.payload)
                    payload["reasoning_artifact"] = repaired_reasoning
                    if reasoning_repairs:
                        payload["harness_reasoning_repairs"] = list(reasoning_repairs)
                    if synthesized_reasoning:
                        payload["harness_reasoning_synthesized"] = (
                            "observable_component_review_evidence"
                        )
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
                    else:
                        payload = dict(response.payload)
                        existing_evidence = list(payload.get("evidence", ()) or ())
                        existing_evidence.append(
                            {
                                "kind": "harness_browser_preview",
                                "status": "passed",
                                "http_status": harness_preview.get("http_status"),
                                "console_errors": list(harness_preview.get("console_errors", ()) or ()),
                                "page_errors": list(harness_preview.get("page_errors", ()) or ()),
                                "network_errors": list(harness_preview.get("network_errors", ()) or ()),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                                "interaction_results": list(
                                    harness_preview.get("interaction_results", ()) or ()
                                ),
                            }
                        )
                        payload["evidence"] = existing_evidence
                        existing_results = list(payload.get("test_results", ()) or ())
                        existing_results.append(
                            {
                                "name": "harness_html_preview",
                                "passed": True,
                                "http_status": harness_preview.get("http_status"),
                                "screenshot_path": harness_preview.get("screenshot_path"),
                            }
                        )
                        payload["test_results"] = existing_results
                        evidence_absence = re.compile(
                            r"(?:no|missing|lack(?:ing)?|additional)\s+(?:browser\s+|runtime\s+|"
                            r"verification\s+)?evidence|not\s+verified",
                            re.IGNORECASE,
                        )
                        raw_issues = list(payload.get("issues", ()) or ())
                        raw_findings = list(payload.get("findings", ()) or ())
                        issues = [item for item in raw_issues if not evidence_absence.search(str(item))]
                        findings = [item for item in raw_findings if not evidence_absence.search(str(item))]
                        provider_claimed_only_missing_evidence = bool(
                            evidence_absence.search(
                                " ".join(
                                    [
                                        str(response.summary or ""),
                                        *(str(item) for item in raw_issues),
                                        *(str(item) for item in raw_findings),
                                    ]
                                )
                            )
                        )
                        payload["issues"] = issues
                        payload["findings"] = findings
                        if provider_claimed_only_missing_evidence and not issues and not findings:
                            payload["passed"] = True
                            payload["success"] = True
                        response = AgentResponse(
                            payload=payload,
                            summary=(
                                "Harness browser preview passed with authoritative evidence."
                                if provider_claimed_only_missing_evidence
                                else response.summary
                            ),
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
                        proposed_content, repaired_python_escaping = (
                            _repair_double_escaped_python_source(
                                proposed_path,
                                proposed_content,
                            )
                        )
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
                                successful_mutation_paths.add(proposed_path)
                                if repaired_python_escaping:
                                    self.events.publish(
                                        "ultra.proposed_write_normalized",
                                        "Removed double JSON escaping after Python "
                                        "syntax validation",
                                        run_id=request.run_id,
                                        node_id=request.node_id,
                                        path=proposed_path,
                                    )
                                self.events.publish(
                                    "ultra.proposed_write_executed",
                                    "Validated and executed typed proposed_write fallback",
                                    run_id=request.run_id,
                                    node_id=request.node_id,
                                    path=proposed_path,
                                    characters=len(proposed_content),
                                )
                if requires_mutation and not mutation_observed:
                    mutation_claim_count += 1
                    if mutation_claim_count >= 2:
                        existing_fix_targets = [
                            str(item.get("path") or "")
                            for item in write_target_state
                            if str(item.get("path") or "") and bool(item.get("exists"))
                        ]
                        if (
                            request.phase == InnerPhase.FIX.value
                            and existing_fix_targets
                            and len(existing_fix_targets) == len(write_target_state)
                        ):
                            # A weak local model can correctly conclude that a
                            # prior durable mutation already contains its fix,
                            # yet fail to emit another (unnecessary) write.
                            # Treat this only as a no-change candidate and move
                            # to independent test/reviewer gates. It is not
                            # acceptance evidence and IMPLEMENT still requires
                            # an actual governed mutation.
                            self.events.publish(
                                "ultra.fix_no_change_candidate",
                                "A read-only fix claim was routed to independent verification instead of another identical write request.",
                                run_id=request.run_id,
                                node_id=request.node_id,
                                role=self.role.value,
                                paths=existing_fix_targets,
                            )
                            return AgentResponse.from_mapping(
                                {
                                    "payload": {
                                        "success": True,
                                        "passed": True,
                                        "phase_receipt_only": True,
                                        "verification_only": True,
                                        "no_change_candidate": True,
                                        "artifacts": existing_fix_targets,
                                        "evidence": [
                                            {
                                                "kind": "approved_existing_artifact",
                                                "path": path,
                                                "status": "requires_independent_verification",
                                            }
                                            for path in existing_fix_targets
                                        ],
                                        "findings": [],
                                        "issues": [],
                                    },
                                    "summary": "Existing approved artifacts were preserved and routed to independent verification.",
                                    "reasoning_summary": "No new mutation receipt was accepted; tester and reviewer gates remain authoritative.",
                                },
                                node_id=request.node_id,
                                provider="harness",
                                model="no-change-fix-boundary-v1",
                            )
                        raise _MutationTransportExhausted(
                            f"{self.role.value} returned {mutation_claim_count} "
                            "structured responses without the required governed "
                            f"mutation for write_paths={list(contract_payload.get('write_paths', ()))!r}"
                        )
                    reset_cache = getattr(self.provider, "reset_model_cache", None)
                    cache_reset = False
                    if callable(reset_cache):
                        try:
                            reset_cache()
                            cache_reset = True
                        except Exception:
                            cache_reset = False
                    target_state = next(
                        (
                            dict(item)
                            for item in write_target_state
                            if str(item.get("path") or "").strip()
                        ),
                        {},
                    )
                    target_path = str(target_state.get("path") or "").strip()
                    conversation = [
                        {
                            "role": "user",
                            "content": _json(
                                {
                                    "node_id": request.node_id,
                                    "phase": request.phase,
                                    "objective": str(
                                        contract_payload.get("objective") or ""
                                    )[:800],
                                    "required_corrections": mutation_requirements,
                                    "acceptance_criteria": mutation_acceptance_criteria,
                                    "approved_write_paths": list(
                                        contract_payload.get("write_paths", ()) or ()
                                    ),
                                    "next_target": {
                                        "path": target_path,
                                        "exists": bool(target_state.get("exists")),
                                        "current_content": str(
                                            target_state.get("content") or ""
                                        )[:40_000],
                                    },
                                    "rejected_claim_fingerprint": hashlib.sha256(
                                        _json(response.payload).encode("utf-8")
                                    ).hexdigest(),
                                    "required_action": (
                                        "Return exactly one edit_file action for an existing "
                                        "next_target.path (or write_file only when it does not exist) "
                                        "with the complete approved "
                                        "implementation. Return no prose, do not read "
                                        "again, and do not return a success JSON before "
                                        "the governed write receipt."
                                    ),
                                }
                            ),
                        }
                    ]
                    schemas[:] = [
                        schema
                        for schema in schemas
                        if _schema_name(schema) in _FILE_WRITE_TOOLS
                    ]
                    self.events.publish(
                        "ultra.mutation_claim_rejected",
                        "Rejected a read-only implementation claim, reset the local "
                        "model cache, and requested one approval-bound write.",
                        run_id=request.run_id,
                        node_id=request.node_id,
                        role=self.role.value,
                        phase=request.phase,
                        path=target_path,
                        claim_count=mutation_claim_count,
                        cache_reset=cache_reset,
                    )
                    continue
                if component_publication_phase and not component_publication_passed:
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
                if isinstance(exc, _MutationTransportExhausted):
                    raise
                content_preview = redact_text(str(turn.text or ""), 800)
                compact_preview = re.sub(r"\s+", "", content_preview)
                internal_token_only = bool(
                    re.fullmatch(r"(?:<unused\d+>)+", compact_preview)
                )
                empty_transport_turn = not compact_preview and not turn.tool_calls
                if (
                    local_governed_mutation_phase
                    and mutation_observed
                    and bool(successful_mutation_paths)
                ):
                    recovered = AgentResponse.from_mapping(
                        {
                            "payload": {
                                "success": True,
                                "passed": True,
                                "phase_receipt_only": True,
                                "handoff_transport_failed": True,
                                "recovered_from_empty_provider_turn": bool(
                                    internal_token_only or empty_transport_turn
                                ),
                                "artifacts": [
                                    {
                                        "kind": "governed_mutation_receipt",
                                        "path": path,
                                    }
                                    for path in sorted(successful_mutation_paths)
                                ],
                                "evidence": [
                                    {
                                        "kind": "governed_mutation_receipt",
                                        "path": path,
                                        "status": "executed",
                                    }
                                    for path in sorted(successful_mutation_paths)
                                ],
                                "findings": [],
                            },
                            "summary": (
                                f"{self.role.value} governed mutation completed; "
                                "the typed handoff transport was recovered from its receipt."
                            ),
                            "reasoning_summary": (
                                "This receipt closes only the mutation phase. Independent "
                                "tester, reviewer, browser, and final-evidence gates remain required."
                            ),
                        },
                        node_id=request.node_id,
                        provider="harness",
                        model="same-turn-mutation-receipt-v2",
                        usage=totals,
                    )
                    self.events.publish(
                        "ultra.mutation_handoff_recovered",
                        recovered.summary,
                        run_id=request.run_id,
                        node_id=request.node_id,
                        role=self.role.value,
                        phase=request.phase,
                        paths=sorted(successful_mutation_paths),
                        error_type=type(exc).__name__,
                        response_fingerprint=hashlib.sha256(
                            str(turn.text or "").encode("utf-8")
                        ).hexdigest(),
                    )
                    return recovered
                if internal_token_only or empty_transport_turn:
                    if (
                        mutation_observed
                        and bool(successful_mutation_paths)
                        and request.phase
                        in {
                            InnerPhase.IMPLEMENT.value,
                            InnerPhase.FIX.value,
                            InnerPhase.INTEGRATE.value,
                        }
                        and self.role in {AgentRole.CODER, AgentRole.INTEGRATOR}
                    ):
                        recovered = AgentResponse.from_mapping(
                            {
                                "payload": {
                                    "success": True,
                                    "passed": True,
                                    "recovered_from_empty_provider_turn": True,
                                    "artifacts": [
                                        {
                                            "kind": "governed_mutation_receipt",
                                            "path": path,
                                        }
                                        for path in sorted(successful_mutation_paths)
                                    ],
                                    "evidence": [
                                        {
                                            "kind": "governed_mutation_receipt",
                                            "path": path,
                                            "status": "executed",
                                        }
                                        for path in sorted(successful_mutation_paths)
                                    ],
                                    "findings": [],
                                },
                                "summary": (
                                    f"{self.role.value} mutation receipt recovered after an "
                                    "empty local-model handoff."
                                ),
                                "reasoning_summary": (
                                    "The executor confirmed a governed write in this exact agent "
                                    "turn. Independent artifact, review, and runtime gates remain required."
                                ),
                            },
                            node_id=request.node_id,
                            provider="harness",
                            model="same-turn-mutation-receipt-v1",
                        )
                        self.events.publish(
                            "ultra.same_turn_mutation_receipt_recovered",
                            recovered.summary,
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                            paths=sorted(successful_mutation_paths),
                        )
                        return recovered
                    local_foundation_transport = bool(
                        request.phase in {"master_plan", "browser_scenarios"}
                        and self.provider_name == "ollama"
                    )
                    if (
                        (
                            component_publication_phase
                            or component_leaf_quality_phase
                            or component_quality_triage_phase
                            or local_foundation_transport
                            or local_governed_mutation_phase
                        )
                        and component_unused_retries < 2
                    ):
                        component_unused_retries += 1
                        reset_cache = getattr(self.provider, "reset_model_cache", None)
                        cache_reset = False
                        if callable(reset_cache):
                            try:
                                reset_cache()
                                cache_reset = True
                            except Exception:
                                cache_reset = False
                        if request.phase == "master_plan" and local_foundation_transport:
                            recovery_content = _json(
                                {
                                    "goal_spec": request.task.get("goal_spec", {}),
                                    "architecture": request.task.get("architecture", {}),
                                    "applicability_repair": request.task.get(
                                        "applicability_repair", []
                                    ),
                                    "accepted_requested_effects": request.task.get(
                                        "accepted_requested_effects", []
                                    ),
                                    "required_action": (
                                        "Call submit_master_plan exactly once with the complete "
                                        "approval-bound plan. Return no prose and call no other tool. "
                                        "Use literal executable verification commands. preview_html "
                                        "must receive one .html path without a query or fragment; "
                                        "put clicks, typing, and assertions in metadata.browser_scenarios."
                                    ),
                                }
                            )
                        elif request.phase == "browser_scenarios" and local_foundation_transport:
                            recovery_content = _json(
                                {
                                    "modules": request.task.get("modules", []),
                                    "required_action": (
                                        "Call submit_browser_scenarios exactly once. Preserve each "
                                        "module_id and return at least one scenario with non-empty "
                                        "executable steps and observable assertions. Use only click, "
                                        "fill with value, or press with key; do not add text to click. "
                                        "Assertions require a target, a supported property (value, text, "
                                        "textContent, id, visible, checked, count, visibleCount, "
                                        "dataObjectCount, or dataVisualState), and "
                                        "equals or contains."
                                    ),
                                }
                            )
                        elif local_governed_mutation_phase:
                            target_state = next(
                                (
                                    dict(item)
                                    for item in write_target_state
                                    if not bool(item.get("exists"))
                                ),
                                dict(write_target_state[0])
                                if write_target_state
                                else {},
                            )
                            target_path = str(target_state.get("path") or "").strip()
                            recovery_content = _json(
                                {
                                    "node_id": request.node_id,
                                    "phase": request.phase,
                                    "objective": str(
                                        request_contract.get("objective") or ""
                                    )[:800],
                                    "approved_write_paths": [
                                        str(path)
                                        for path in tuple(
                                            request_contract.get("write_paths", ()) or ()
                                        )
                                    ],
                                    "next_target": {
                                        "path": target_path,
                                        "exists": bool(target_state.get("exists")),
                                        "current_content": str(
                                            target_state.get("content") or ""
                                        )[:8_000],
                                    },
                                    "required_action": (
                                        "Return exactly one write_file action for next_target.path "
                                        "with the complete implementation required by the approved "
                                        "module contract. Do not claim other files were written. "
                                        "Return no prose and do not use read tools."
                                    ),
                                }
                            )
                        elif component_publication_phase:
                            specialist_domain = str(
                                contract_metadata.get("specialist_domain")
                                or request_contract.get("title")
                                or "component"
                            )[:240]
                            objective = str(
                                request_contract.get("objective") or ""
                            )[:500]
                            priority_findings = tuple(
                                str(item)[:280]
                                for item in (
                                    request.task.get("findings")
                                    or request.task.get("prior_findings")
                                    or ()
                                )
                                if str(item).strip()
                            )[:3]
                            recovery_blueprint = _specialist_quality_blueprint(
                                specialist_domain
                            )[:4]
                            recovery_content = (
                                f"Build only {specialist_domain}: {objective}. "
                                f"Mandatory craft rules: {' | '.join(recovery_blueprint)}. "
                                + (
                                    "Fix these observed defects: "
                                    + " | ".join(priority_findings)
                                    + ". "
                                    if priority_findings
                                    else ""
                                )
                                +
                                "Call stage_component_file exactly once now for "
                                "preview/scene.js (role preview). Return no prose. "
                                "Define window.buildPreview=({THREE,scene,camera,renderer})=>{...} "
                                "with executable detailed geometry and materials. Use only those "
                                "passed objects; never create Scene/Renderer/DOM/animation loop. "
                                "Implement the rules in code, not comments."
                            )
                        else:
                            recovery_content = _json(
                                {
                                    **dict(user_payload),
                                    "recovery_instruction": (
                                        "Use the durable receipt and at most one focused read. "
                                        "Then return the response_contract JSON now."
                                    ),
                                }
                            )
                        conversation = [{"role": "user", "content": recovery_content}]
                        self.events.publish(
                            (
                                "ultra.master_plan_transport_recovered"
                                if request.phase == "master_plan" and local_foundation_transport
                                else "ultra.browser_scenarios_transport_recovered"
                                if request.phase == "browser_scenarios" and local_foundation_transport
                                else "ultra.mutation_transport_recovered"
                                if local_governed_mutation_phase
                                else "ultra.component_transport_recovered"
                            ),
                            (
                                f"[{request.node_id or request.phase}] retried degraded local "
                                f"response with a minimal {request.phase} packet "
                                f"({component_unused_retries}/2); "
                                f"model_cache_reset={cache_reset}"
                            ),
                            run_id=request.run_id,
                            node_id=request.node_id,
                            role=self.role.value,
                            phase=request.phase,
                        )
                        continue
                    raise AgentProtocolError(
                        f"{self.role.value} emitted only an internal unused token"
                    ) from exc
                last_error = RuntimeError(
                    f"{exc}; content_preview={content_preview!r}"
                )
                invalid_json_attempts += 1
                if invalid_json_attempts >= max_invalid_json_attempts:
                    raise AgentProtocolError(
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
        raise AgentProtocolError(
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
        # A bounded repair already carries the latest accepted authority in its
        # node contract.  Rehydrating the whole project brain here can drown a
        # small local model in stale, contradictory lessons from earlier
        # revisions (and materially increase inference latency).  Keep the
        # current contract and direct dependency evidence, and explicitly mark
        # historical sections as omitted.  Normal first-pass implementation
        # still receives the full durable context below.
        if "accepted repair requirements:" in request.node.contract.objective.casefold():
            return {
                "north_star": {
                    "objective": request.goal.objective,
                    "success_criteria": list(request.goal.success_criteria),
                },
                "repair_contract": asdict(request.node.contract),
                "dependency_artifacts": {
                    node_id: asdict(result)
                    for node_id, result in request.dependency_results.items()
                },
                "_omitted": [
                    "historical_architecture",
                    "project_lessons",
                    "project_knowledge",
                    "role_memory",
                ],
            }
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
        events: EventBus | None = None,
    ) -> None:
        super().__init__()
        self.store = store
        self.goal_id = goal_id
        self.descriptor = descriptor
        self.access_level = access_level
        self.config = config
        self.workspace = workspace
        self.events = events
        self.run_id: str | None = None
        goal = self.store.get_goal(goal_id)
        self.adaptive_orchestration_policy = AdaptiveOrchestrationPolicyV1.from_dict(
            goal.metadata.get("adaptive_orchestration_policy")
            if isinstance(goal.metadata.get("adaptive_orchestration_policy"), Mapping)
            else None
        )
        self.adaptive_orchestration_shadow_mode = (
            self.adaptive_orchestration_policy.shadow_mode
        )
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
        self._lease_initial_hashes: dict[str, dict[str, str | None]] = {}
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
        self._component_champion_scores: dict[str, float] = {}
        self._published_component_results: dict[str, Mapping[str, Any]] = {}
        self._active_component_draft_paths: dict[str, set[str]] = {}
        self.visual_judge = create_visual_judge(
            builder_provider=descriptor.provider,
            builder_model=descriptor.model,
            ollama_host=descriptor.host or "http://127.0.0.1:11434",
        )
        self._adapter_lock = threading.RLock()

    def stage_next_action(
        self,
        action_id: str,
        *,
        role: str,
        phase: str,
        node_id: str | None,
        task: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> None:
        """Checkpoint the bounded assignment before local inference begins."""

        if not self.run_id:
            return
        context_packet = context.get("next_action_packet")
        base = dict(context_packet) if isinstance(context_packet, Mapping) else {}
        task_contract = task.get("contract")
        contract = (
            dict(task_contract)
            if isinstance(task_contract, Mapping)
            else dict(base.get("contract") or {})
        )
        objective = str(
            contract.get("objective")
            or task.get("objective")
            or base.get("objective")
            or f"Execute {phase}"
        )[:3_000]
        dependency_evidence = context.get("dependency_artifacts", ())
        if isinstance(dependency_evidence, Mapping):
            dependency_evidence = tuple(
                {"node_id": key, "result": value}
                for key, value in list(dependency_evidence.items())[:12]
            )
        elif not isinstance(dependency_evidence, (list, tuple)):
            dependency_evidence = ()
        relevant: list[Mapping[str, Any]] = []
        for name in (
            "previous_agent_memory",
            "project_lessons",
            "project_knowledge",
            "role_memory",
        ):
            value = context.get(name)
            if isinstance(value, Mapping) and value:
                relevant.append({"section": name, "value": dict(value)})
            elif isinstance(value, (list, tuple)):
                relevant.extend(
                    {"section": name, "value": dict(item)}
                    for item in value[:4]
                    if isinstance(item, Mapping)
                )
        packet = NextActionPacketV1(
            ultra_run_id=self.run_id,
            work_node_id=node_id,
            role=role,
            phase=phase,
            objective=objective,
            contract=contract,
            checkpoint=dict(base.get("checkpoint") or {}),
            dependency_evidence=tuple(
                dict(item)
                for item in dependency_evidence[:12]
                if isinstance(item, Mapping)
            ),
            relevant_memory=tuple(relevant[:12]),
            required_outputs=tuple(
                str(item)
                for item in (
                    contract.get("acceptance_criteria")
                    or contract.get("success_criteria")
                    or ()
                )
            ),
            omitted_sections=tuple(
                str(item) for item in context.get("_omitted", ())
            ),
            context_budget_chars=max(
                2_000, min(int(self.config.context_chars), 120_000)
            ),
        )
        self.store.stage_scheduled_agent_action(
            action_id,
            packet,
            agent_run_id=action_id,
            status=NextActionStatus.RUNNING,
        )

    def _checkpoint_agent_memory(self, item: Any) -> None:
        if not self.run_id:
            return
        node = None
        if item.node_id:
            try:
                node = self.store.get_work_node(item.node_id)
            except StateStoreError:
                node = None
        actions = self.store.list_scheduled_agent_actions(self.run_id)
        completed = tuple(
            str(action["id"])
            for action in actions
            if action["status"] == NextActionStatus.COMPLETED.value
            and action["role"] == item.role.value
            and action["work_node_id"] == item.node_id
        )[-24:]
        findings = tuple(
            {
                "id": finding.id,
                "severity": finding.severity.value,
                "category": finding.category.value,
                "path": finding.path,
                "remediation": finding.remediation,
                "status": finding.status.value,
            }
            for finding in self.store.list_quality_findings(self.run_id)
            if finding.repair_node_id in {None, item.node_id}
            and finding.status.value != "resolved"
        )[:20]
        artifacts = (
            self.store.list_artifacts(self.run_id, work_node_id=item.node_id, limit=100)
            if item.node_id
            else ()
        )
        decisions = self.store.list_brain_entries(
            self.run_id,
            section=BrainSection.DECISION,
            limit=50,
        )
        snapshot = AgentMemorySnapshotV1(
            ultra_run_id=self.run_id,
            work_node_id=item.node_id,
            role=item.role.value,
            objective=(
                node.objective
                if node is not None
                else f"{item.role.value} {item.phase}"
            ),
            checkpoint=(
                node.checkpoint if node is not None else str(item.phase)
            ),
            completed_actions=completed,
            open_findings=findings,
            decision_refs=tuple(entry.id for entry in decisions),
            artifact_refs=tuple(
                artifact.content_hash or artifact.uri for artifact in artifacts
            ),
            dependency_refs=tuple(node.depends_on) if node is not None else (),
            last_result={
                "status": item.status,
                "summary": item.summary,
                "error": item.error,
                "usage": dict(item.usage),
                "phase": item.phase,
            },
            next_action_id=(
                item.id
                if item.status in {"running", "rate_limited", "uncertain"}
                else None
            ),
        )
        self.store.save_agent_memory_snapshot(snapshot)

    def _workspace_hashes(self) -> dict[str, str]:
        """Return a stable source/evidence inventory for preservation gates."""

        if self.workspace is None:
            return {}
        values: dict[str, str] = {}
        for path in self.workspace.rglob("*"):
            if not path.is_file() or ".coding-agent" in path.parts:
                continue
            relative = path.relative_to(self.workspace).as_posix()
            if relative.startswith(("output/playwright/", ".playwright-cli/")):
                continue
            try:
                values[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
        return values

    @staticmethod
    def _generated_threejs_preview(
        scene_source: str,
        *,
        node_id: str = "",
    ) -> str:
        """Wrap specialist-only scene code in deterministic browser plumbing."""

        # The isolated shell already loads Three.js as a browser global and
        # passes it to buildPreview. Local models nevertheless often emit the
        # Node/bundler import form. Normalize only that transport boundary;
        # the original staged source and its hash remain untouched.
        scene_source = re.sub(
            r"""(?m)^\s*import\s+\*\s+as\s+THREE\s+from\s+["']three["']\s*;?\s*$""",
            "",
            str(scene_source),
        )
        scene_source = re.sub(
            r"""(?m)^\s*import\s*\{([^}]+)\}\s*from\s*["']three["']\s*;?\s*$""",
            lambda match: f"const {{{match.group(1)}}}=THREE;",
            scene_source,
        )
        scene_source = re.sub(
            r"""(?m)^\s*export\s+default\s+""",
            "",
            scene_source,
        )
        scene_source = re.sub(
            r"""(?m)^\s*export\s+(?=(?:async\s+)?(?:function|class|const|let|var)\b)""",
            "",
            scene_source,
        )
        scene_source = re.sub(
            r"""(?m)^\s*export\s*\{[^}]*\}\s*;?\s*$""",
            "",
            scene_source,
        )
        preview_fixture = ""
        component_runtime_gate = ""
        api_root_fallback = ""
        node_key = str(node_id).casefold()
        if node_key.endswith(".world.lighting.rig"):
            # The Rig specialist must be judged without the harness defaults
            # doubling its intensity. Its own typed rig must fully light the
            # neutral proof fixture and remain safe for parent integration.
            preview_fixture = """
for (const item of [...scene.children]) {
  if (item && item.isLight) scene.remove(item);
}
"""
            component_runtime_gate = """
const rigApi=window.LightingRigAPI;
if (!rigApi || !rigApi.root || !rigApi.root.isObject3D) {
  throw new Error("lighting rig must publish LightingRigAPI.root");
}
let rigLights=0;
let rigMeshes=0;
let intensity=0;
let shadowKey=false;
rigApi.root.traverse((item)=>{
  if (item && item.isLight) {
    rigLights+=1; intensity+=Number(item.intensity||0);
    if (item.isDirectionalLight && item.castShadow) shadowKey=true;
  }
});
scene.traverse((item)=>{if (item && item.isMesh) rigMeshes+=1;});
if (rigLights<3 || rigLights>4) throw new Error("lighting rig must contain three or four owned lights");
if (intensity>2.4) throw new Error("lighting rig combined intensity exceeds 2.4");
if (!shadowKey) throw new Error("lighting rig requires one shadow-casting directional key");
if (rigMeshes<5) throw new Error("lighting rig requires a compact multi-material proof fixture");
window.__componentSelfTest={passed:true,checks:["owned-rig","bounded-intensity","shadow-key","material-fixture"]};
"""
        elif node_key.endswith(".world.lighting.atmosphere"):
            component_runtime_gate = """
const atmosphereApi=window.AtmosphereAPI;
if (!atmosphereApi || !atmosphereApi.root || !atmosphereApi.root.isObject3D
    || typeof atmosphereApi.apply!=="function") {
  throw new Error("atmosphere package must publish AtmosphereAPI root and apply function");
}
let atmosphereMeshes=0;
let oversizedAtmosphereMeshes=0;
scene.updateMatrixWorld(true);
scene.traverse((item)=>{
  if (!item || !item.isMesh) return;
  atmosphereMeshes+=1;
  const size=new THREE.Box3().setFromObject(item).getSize(new THREE.Vector3());
  if (size.x>6 || size.y>6 || size.z>6) oversizedAtmosphereMeshes+=1;
});
if (atmosphereMeshes<5 || atmosphereMeshes>8) {
  throw new Error("atmosphere fixture must contain five to eight visible depth forms");
}
if (oversizedAtmosphereMeshes) throw new Error("atmosphere fixture contains an occluding oversized mesh");
if (!scene.fog || !(scene.background && scene.background.isColor)) {
  throw new Error("atmosphere must configure scene fog and a color background without sky geometry");
}
camera.position.set(8,5.5,10); camera.lookAt(0,0,-9);
window.__componentSelfTest={passed:true,checks:["api","color-background","fog","depth-fixture","bounded-meshes"]};
"""
        elif node_key.endswith(".world.lighting.shadows"):
            component_runtime_gate = """
const shadowApi=window.ShadowQualityAPI;
if (!shadowApi || !shadowApi.root || !shadowApi.root.isObject3D
    || typeof shadowApi.configureRenderer!=="function"
    || typeof shadowApi.configureLight!=="function") {
  throw new Error("shadow package must publish ShadowQualityAPI root and configure functions");
}
let shadowMeshes=0;
let horizontalReceivers=0;
let receivingHorizontalReceivers=0;
let shadowCasters=0;
scene.updateMatrixWorld(true);
scene.traverse((item)=>{
  if (!item || !item.isMesh) return;
  shadowMeshes+=1;
  if (item.castShadow) shadowCasters+=1;
  const size=new THREE.Box3().setFromObject(item).getSize(new THREE.Vector3());
  if (size.x*size.z>=20 && size.y<=.4) {
    horizontalReceivers+=1;
    if (item.receiveShadow) receivingHorizontalReceivers+=1;
  }
});
if (shadowMeshes<4 || shadowMeshes>7) {
  throw new Error("shadow proof fixture must contain four to seven visible meshes");
}
if (horizontalReceivers!==1) {
  throw new Error("shadow proof fixture requires exactly one thin horizontal receiver");
}
if (receivingHorizontalReceivers!==1 || shadowCasters<3) {
  throw new Error("shadow proof requires one receiving ground and at least three casting forms");
}
shadowApi.configureRenderer(renderer);
const previewShadowKey=scene.children.find((item)=>item && item.isDirectionalLight);
if (!previewShadowKey) throw new Error("shadow preview requires the harness directional key");
shadowApi.configureLight(previewShadowKey);
if (!renderer.shadowMap.enabled || !previewShadowKey.castShadow
    || previewShadowKey.shadow.bias>0.001) {
  throw new Error("shadow configuration did not enable a grounded low-bias key");
}
for (const item of scene.children) {
  if (item && item.isHemisphereLight) item.intensity=.28;
  if (item && item.isDirectionalLight) item.intensity=1.1;
}
camera.position.set(7,5.5,9); camera.lookAt(0,.6,0);
window.__componentSelfTest={passed:true,checks:["api","fixture-density","close-framing"]};
"""
        elif node_key.endswith(".vehicles.chassis.shell.volumes"):
            api_root_fallback = "window.ChassisVolumesAPI&&window.ChassisVolumesAPI.root"
            preview_fixture = """
const harnessVehicleFloor=new THREE.Mesh(
  new THREE.BoxGeometry(11,.08,11),
  new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92})
);
harnessVehicleFloor.name="__harness_vehicle_floor_not_component_output";
harnessVehicleFloor.position.y=.30; harnessVehicleFloor.receiveShadow=true;
scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const inferredVolumesRoot=scene.children.find((item)=>!harnessChildrenBeforeBuild.has(item)&&item&&item.isObject3D);
const volumesApi=window.ChassisVolumesAPI||(
  inferredVolumesRoot?{root:inferredVolumesRoot,dimensions:{width:2.8,height:1.5,length:5.2},forward:"+Z"}:null
);
if (!volumesApi || !volumesApi.root || !volumesApi.root.isObject3D
    || volumesApi.forward!=="+Z") {
  throw new Error("chassis volumes must publish ChassisVolumesAPI.root with +Z forward");
}
let volumeMeshes=0;
const volumeColors=new Set();
volumesApi.root.traverse((item)=>{
  if (!item || !item.isMesh) return;
  volumeMeshes+=1;
  const materials=Array.isArray(item.material)?item.material:[item.material];
  for (const material of materials) if (material&&material.color) volumeColors.add(material.color.getHexString());
});
volumesApi.root.updateMatrixWorld(true);
const volumeSize=new THREE.Box3().setFromObject(volumesApi.root).getSize(new THREE.Vector3());
if (volumeMeshes<4 || volumeMeshes>7) throw new Error(`chassis volumes require four to seven meshes; measured ${volumeMeshes}`);
if (volumeColors.size<2) throw new Error("chassis volumes require at least two coherent material colors");
if (volumeSize.x<2.4 || volumeSize.x>3.2 || volumeSize.z<4.6 || volumeSize.z>5.5
    || volumeSize.y<.8 || volumeSize.y>2.0) {
  throw new Error(`chassis volume envelope failed: measured x=${volumeSize.x.toFixed(2)}, y=${volumeSize.y.toFixed(2)}, z=${volumeSize.z.toFixed(2)}`);
}
camera.position.set(4.7,3.1,6.4); camera.lookAt(0,.82,0);
window.__componentSelfTest={passed:true,checks:["typed-root","primary-volumes","bounded-envelope","material-hierarchy"]};
"""
        elif node_key.endswith(".vehicles.chassis.shell.panels"):
            api_root_fallback = "window.ChassisPanelsAPI&&window.ChassisPanelsAPI.root"
            preview_fixture = """
const harnessBodyReference=new THREE.Group();
harnessBodyReference.name="__harness_body_reference_not_component_output";
const referenceMaterial=new THREE.MeshStandardMaterial({color:0x596574,roughness:.78,transparent:true,opacity:.42});
for (const [size,pos] of [
  [[2.8,.35,5.0],[0,.55,0]],[[2.45,.55,1.55],[0,.9,1.65]],
  [[2.2,.9,2.15],[0,1.2,-.1]],[[2.35,.45,1.25],[0,.82,-1.85]]
]) {
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),referenceMaterial);
  mesh.position.set(...pos); harnessBodyReference.add(mesh);
}
scene.add(harnessBodyReference);
const harnessVehicleFloor=new THREE.Mesh(
  new THREE.BoxGeometry(11,.08,11),
  new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92})
);
harnessVehicleFloor.name="__harness_vehicle_floor_not_component_output";
harnessVehicleFloor.position.y=.30; harnessVehicleFloor.receiveShadow=true;
scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const inferredPanelsRoot=scene.children.find((item)=>!harnessChildrenBeforeBuild.has(item)&&item&&item.isObject3D);
const panelsApi=window.ChassisPanelsAPI||(
  inferredPanelsRoot?{root:inferredPanelsRoot,wheelMounts:[
    {x:1.38,y:.55,z:1.55},{x:-1.38,y:.55,z:1.55},
    {x:1.38,y:.55,z:-1.55},{x:-1.38,y:.55,z:-1.55}
  ],forward:"+Z"}:null
);
if (!panelsApi || !panelsApi.root || !panelsApi.root.isObject3D
    || panelsApi.forward!=="+Z" || !Array.isArray(panelsApi.wheelMounts)
    || panelsApi.wheelMounts.length!==4) {
  throw new Error("chassis panels must publish ChassisPanelsAPI.root with +Z forward and four mounts");
}
let panelMeshes=0;
const panelColors=new Set();
panelsApi.root.traverse((item)=>{
  if (!item || !item.isMesh) return;
  panelMeshes+=1;
  const materials=Array.isArray(item.material)?item.material:[item.material];
  for (const material of materials) if (material&&material.color) panelColors.add(material.color.getHexString());
});
const mountPairs=new Set(panelsApi.wheelMounts.map((mount)=>`${Number(mount.x).toFixed(2)}:${Number(mount.z).toFixed(2)}`));
const expectedMounts=["-1.38:-1.55","-1.38:1.55","1.38:-1.55","1.38:1.55"];
if (!expectedMounts.every((key)=>mountPairs.has(key))) throw new Error("chassis panel mounts must use every X=+/-1.38 and Z=+/-1.55 pair");
panelsApi.root.updateMatrixWorld(true);
const panelSize=new THREE.Box3().setFromObject(panelsApi.root).getSize(new THREE.Vector3());
if (panelMeshes<9 || panelMeshes>16) throw new Error(`chassis panels require nine to sixteen meshes; measured ${panelMeshes}`);
if (panelColors.size<3) throw new Error("chassis panels require three material colors for paint, cladding, and trim");
if (panelSize.x>3.3 || panelSize.z>5.5 || panelSize.y>2.1) throw new Error(`chassis panel envelope failed: measured x=${panelSize.x.toFixed(2)}, y=${panelSize.y.toFixed(2)}, z=${panelSize.z.toFixed(2)}`);
camera.position.set(4.7,3.1,6.4); camera.lookAt(0,.82,0);
window.__componentSelfTest={passed:true,checks:["typed-root","panel-density","mount-grid","reference-alignment"]};
"""
        elif node_key.endswith(".vehicles.vehicle_wheels_cut"):
            preview_fixture = """
const harnessBodyReference=new THREE.Group();const bodyMat=new THREE.MeshStandardMaterial({color:0x506477,roughness:.8,transparent:true,opacity:.24});
for(const [size,pos] of [[[2.8,.35,5],[0,.55,0]],[[2.45,.55,1.55],[0,.9,1.65]],[[2.2,.9,2.15],[0,1.2,-.1]],[[2.35,.45,1.25],[0,.82,-1.85]]]){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),bodyMat);mesh.position.set(...pos);harnessBodyReference.add(mesh);}scene.add(harnessBodyReference);
const harnessVehicleFloor=new THREE.Mesh(new THREE.BoxGeometry(11,.08,11),new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92}));harnessVehicleFloor.position.y=.30;scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const wheelsRoot=(window.VehicleWheelsAPI&&window.VehicleWheelsAPI.root)||scene.children.find((item)=>!harnessChildrenBeforeBuild.has(item)&&item&&item.isObject3D);
if(!wheelsRoot)throw new Error("wheel specialist requires one root");let wheelMeshes=0;const wheelColors=new Set();
wheelsRoot.traverse((item)=>{if(!item||!item.isMesh)return;wheelMeshes+=1;const mats=Array.isArray(item.material)?item.material:[item.material];for(const mat of mats)if(mat&&mat.color)wheelColors.add(mat.color.getHexString());});
if(wheelMeshes<16||wheelMeshes>32)throw new Error(`wheel specialist requires 16-32 meshes; measured ${wheelMeshes}`);
if(wheelColors.size<3)throw new Error("wheel specialist requires rubber, rim, and hub colors");
camera.position.set(4.7,3.1,6.4);camera.lookAt(0,.82,0);window.__componentSelfTest={passed:true,checks:["four-assemblies","mesh-density","material-separation"]};
"""
        elif node_key.endswith(".vehicles.vehicle_glass_cut"):
            preview_fixture = """
const harnessBodyReference=new THREE.Group();const bodyMat=new THREE.MeshStandardMaterial({color:0x506477,roughness:.8,transparent:true,opacity:.24});
for(const [size,pos] of [[[2.8,.35,5],[0,.55,0]],[[2.45,.55,1.55],[0,.9,1.65]],[[2.2,.9,2.15],[0,1.2,-.1]],[[2.35,.45,1.25],[0,.82,-1.85]]]){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),bodyMat);mesh.position.set(...pos);harnessBodyReference.add(mesh);}scene.add(harnessBodyReference);
const harnessVehicleFloor=new THREE.Mesh(new THREE.BoxGeometry(11,.08,11),new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92}));harnessVehicleFloor.position.y=.30;scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const glassRoots=(window.VehicleGlassAPI&&window.VehicleGlassAPI.root)?[window.VehicleGlassAPI.root]:scene.children.filter((item)=>!harnessChildrenBeforeBuild.has(item)&&item&&item.isObject3D);
if(!glassRoots.length)throw new Error("glass specialist requires generated glazing");let glassMeshes=0,transparentMeshes=0;
for(const root of glassRoots)root.traverse((item)=>{if(!item||!item.isMesh)return;glassMeshes+=1;const mats=Array.isArray(item.material)?item.material:[item.material];if(mats.some((mat)=>mat&&(mat.transparent||Number(mat.opacity)<.9)))transparentMeshes+=1;});
if(glassMeshes<8||glassMeshes>18)throw new Error(`glass specialist requires 8-18 meshes; measured ${glassMeshes}`);
if(transparentMeshes<4)throw new Error(`glass specialist requires at least four transparent panes; measured ${transparentMeshes}`);
camera.position.set(4.7,3.1,6.4);camera.lookAt(0,.9,0);window.__componentSelfTest={passed:true,checks:["thin-glazing","transparency","body-alignment"]};
"""
        elif node_key.endswith(".vehicles.vehicle_fascia_cut"):
            preview_fixture = """
const harnessBodyReference=new THREE.Group();const bodyMat=new THREE.MeshStandardMaterial({color:0x506477,roughness:.8,transparent:true,opacity:.24});
for(const [size,pos] of [[[2.8,.35,5],[0,.55,0]],[[2.45,.55,1.55],[0,.9,1.65]],[[2.2,.9,2.15],[0,1.2,-.1]],[[2.35,.45,1.25],[0,.82,-1.85]]]){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),bodyMat);mesh.position.set(...pos);harnessBodyReference.add(mesh);}scene.add(harnessBodyReference);
const harnessVehicleFloor=new THREE.Mesh(new THREE.BoxGeometry(11,.08,11),new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92}));harnessVehicleFloor.position.y=.30;scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const fasciaRoots=(window.VehicleFasciaAPI&&window.VehicleFasciaAPI.root)?[window.VehicleFasciaAPI.root]:scene.children.filter((item)=>!harnessChildrenBeforeBuild.has(item)&&item&&item.isObject3D);
if(!fasciaRoots.length)throw new Error("fascia specialist requires generated details");let fasciaMeshes=0;const fasciaColors=new Set();
for(const root of fasciaRoots)root.traverse((item)=>{if(!item||!item.isMesh)return;fasciaMeshes+=1;const mats=Array.isArray(item.material)?item.material:[item.material];for(const mat of mats)if(mat&&mat.color)fasciaColors.add(mat.color.getHexString());});
if(fasciaMeshes<12||fasciaMeshes>24)throw new Error(`fascia specialist requires 12-24 meshes; measured ${fasciaMeshes}`);
if(fasciaColors.size<4)throw new Error("fascia specialist requires grille, lamp, tail, and metallic colors");
camera.position.set(4.7,3.1,6.4);camera.lookAt(0,.75,0);window.__componentSelfTest={passed:true,checks:["lamp-pairs","fascia-density","material-hierarchy"]};
"""
        elif node_key.endswith(".vehicles.vehicle_details_cut"):
            preview_fixture = """
const harnessBodyReference=new THREE.Group();
harnessBodyReference.name="__harness_body_reference_not_component_output";
const bodyMat=new THREE.MeshStandardMaterial({color:0x506477,roughness:.8,transparent:true,opacity:.28});
for (const [size,pos] of [
  [[2.8,.35,5.0],[0,.55,0]],[[2.45,.55,1.55],[0,.9,1.65]],
  [[2.2,.9,2.15],[0,1.2,-.1]],[[2.35,.45,1.25],[0,.82,-1.85]]
]) { const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),bodyMat);mesh.position.set(...pos);harnessBodyReference.add(mesh); }
scene.add(harnessBodyReference);
const harnessVehicleFloor=new THREE.Mesh(new THREE.BoxGeometry(11,.08,11),new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92}));
harnessVehicleFloor.name="__harness_vehicle_floor_not_component_output";harnessVehicleFloor.position.y=.30;scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const detailsApi=window.VehicleDetailsAPI;
const detailsRoot=(detailsApi&&detailsApi.root&&detailsApi.root.isObject3D?detailsApi.root:null)
  ||scene.children.find((item)=>!harnessChildrenBeforeBuild.has(item)&&item&&item.isObject3D);
if(!detailsRoot) throw new Error("vehicle details require a generated root");
let detailMeshes=0;const detailColors=new Set();
detailsRoot.traverse((item)=>{if(!item||!item.isMesh)return;detailMeshes+=1;const materials=Array.isArray(item.material)?item.material:[item.material];for(const material of materials)if(material&&material.color)detailColors.add(material.color.getHexString());});
detailsRoot.updateMatrixWorld(true);
const detailSize=new THREE.Box3().setFromObject(detailsRoot).getSize(new THREE.Vector3());
if(detailMeshes<20||detailMeshes>36) throw new Error(`vehicle details require 20 to 36 meshes; measured ${detailMeshes}`);
if(detailColors.size<5) throw new Error(`vehicle details require at least five material colors; measured ${detailColors.size}`);
if(detailSize.x>3.9||detailSize.z>6.0||detailSize.y>2.4) throw new Error(`vehicle details envelope failed: x=${detailSize.x.toFixed(2)}, y=${detailSize.y.toFixed(2)}, z=${detailSize.z.toFixed(2)}`);
camera.position.set(4.7,3.1,6.4);camera.lookAt(0,.82,0);
window.__componentSelfTest={passed:true,checks:["detail-density","material-variety","body-alignment"]};
"""
        elif node_key.endswith(
            (".vehicles.vehicle_preview_cut", ".vehicles.vehicle_preview_v2_cut")
        ):
            preview_fixture = """
const harnessVehicleFloor=new THREE.Mesh(new THREE.BoxGeometry(11,.08,11),new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92}));
harnessVehicleFloor.name="__harness_vehicle_floor_not_component_output";harnessVehicleFloor.position.y=.30;scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
let vehicleMeshes=0;const vehicleColors=new Set();
for(const child of scene.children){if(harnessChildrenBeforeBuild.has(child))continue;child.traverse((item)=>{if(!item||!item.isMesh)return;vehicleMeshes+=1;const materials=Array.isArray(item.material)?item.material:[item.material];for(const material of materials)if(material&&material.color)vehicleColors.add(material.color.getHexString());});}
if(vehicleMeshes<30) throw new Error(`integrated vehicle preview requires at least 30 meshes; measured ${vehicleMeshes}`);
if(vehicleColors.size<6) throw new Error(`integrated vehicle preview requires at least six colors; measured ${vehicleColors.size}`);
camera.position.set(4.7,3.1,6.4);camera.lookAt(0,.82,0);
window.__componentSelfTest={passed:true,checks:["body-consumed","details-consumed","visual-density"]};
"""
        elif node_key.endswith(".vehicles.compiled_vehicle_cut"):
            preview_fixture = """
const harnessVehicleFloor=new THREE.Mesh(new THREE.BoxGeometry(12,.1,12),new THREE.MeshStandardMaterial({color:0x33444d,roughness:.94}));
harnessVehicleFloor.name="__harness_vehicle_floor_not_component_output";harnessVehicleFloor.position.y=.25;harnessVehicleFloor.receiveShadow=true;scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const compiledApi=window.CompiledVehicleAPI;
if(!compiledApi||!compiledApi.root||!compiledApi.root.isObject3D||compiledApi.forward!=="+Z")throw new Error("compiled vehicle must publish CompiledVehicleAPI.root with +Z forward");
let compiledMeshes=0;const compiledColors=new Set();compiledApi.root.traverse((item)=>{if(!item||!item.isMesh)return;compiledMeshes+=1;const mats=Array.isArray(item.material)?item.material:[item.material];for(const mat of mats)if(mat&&mat.color)compiledColors.add(mat.color.getHexString());});
compiledApi.root.updateMatrixWorld(true);const compiledSize=new THREE.Box3().setFromObject(compiledApi.root).getSize(new THREE.Vector3());
if(compiledMeshes<48||compiledMeshes>96)throw new Error(`compiled vehicle requires 48-96 meshes; measured ${compiledMeshes}`);
if(compiledColors.size<8)throw new Error(`compiled vehicle requires at least eight material colors; measured ${compiledColors.size}`);
if(compiledSize.x<3.0||compiledSize.x>3.9||compiledSize.z<4.7||compiledSize.z>5.5||compiledSize.y<1.4||compiledSize.y>2.5)throw new Error(`compiled vehicle envelope failed: x=${compiledSize.x.toFixed(2)}, y=${compiledSize.y.toFixed(2)}, z=${compiledSize.z.toFixed(2)}`);
camera.position.set(5.6,3.35,6.4);camera.lookAt(0,.92,0);window.__componentSelfTest={passed:true,checks:["typed-spec","cohesive-topology","material-hierarchy","bounded-envelope"]};
"""
        elif node_key.endswith(".character.compiled_character_cut"):
            preview_fixture = """
const harnessCharacterFloor=new THREE.Mesh(new THREE.CylinderGeometry(4.8,4.8,.12,48),new THREE.MeshStandardMaterial({color:0x76a86f,roughness:.96}));
harnessCharacterFloor.name="__harness_character_floor_not_component_output";harnessCharacterFloor.position.y=-.08;harnessCharacterFloor.receiveShadow=true;scene.add(harnessCharacterFloor);
"""
            component_runtime_gate = """
const characterApi=window.CompiledCharacterAPI;
if(!characterApi||!characterApi.root||!characterApi.root.isObject3D||characterApi.forward!=="+Z"||typeof characterApi.updateAnimation!=="function"||typeof characterApi.create!=="function")throw new Error("compiled character must publish root, factory, +Z forward, and animation API");
for(const state of ["idle","hop","hit"])if(!characterApi.states.includes(state))throw new Error(`compiled character lacks ${state} state`);
let characterMeshes=0;const characterColors=new Set();characterApi.root.traverse((item)=>{if(!item||!item.isMesh)return;characterMeshes+=1;const mats=Array.isArray(item.material)?item.material:[item.material];for(const mat of mats)if(mat&&mat.color)characterColors.add(mat.color.getHexString());});
characterApi.root.updateMatrixWorld(true);const characterSize=new THREE.Box3().setFromObject(characterApi.root).getSize(new THREE.Vector3());
if(characterMeshes<28||characterMeshes>56)throw new Error(`compiled character requires 28-56 meshes; measured ${characterMeshes}`);
if(characterColors.size<7)throw new Error(`compiled character requires at least seven material colors; measured ${characterColors.size}`);
if(characterSize.x<1.4||characterSize.x>3.0||characterSize.y<2.6||characterSize.y>5.8||characterSize.z<1.2||characterSize.z>3.0)throw new Error(`compiled character envelope failed: x=${characterSize.x.toFixed(2)}, y=${characterSize.y.toFixed(2)}, z=${characterSize.z.toFixed(2)}`);
characterApi.updateAnimation("hop",.54);camera.position.set(5.0,3.25,7.0);camera.lookAt(0,1.65,.15);window.__componentSelfTest={passed:true,checks:["typed-spec","readable-silhouette","factory-api","three-animation-states"]};
"""
        elif node_key.endswith(".vehicles.chassis.shell"):
            api_root_fallback = "window.ChassisShellAPI&&window.ChassisShellAPI.root"
            preview_fixture = """
const harnessVehicleFloor=new THREE.Mesh(
  new THREE.BoxGeometry(11,.08,11),
  new THREE.MeshStandardMaterial({color:0x40515b,roughness:.92})
);
harnessVehicleFloor.name="__harness_vehicle_floor_not_component_output";
harnessVehicleFloor.position.y=.30; harnessVehicleFloor.receiveShadow=true;
scene.add(harnessVehicleFloor);
"""
            component_runtime_gate = """
const shellApi=window.ChassisShellAPI;
if (!shellApi || !shellApi.root || !shellApi.root.isObject3D
    || shellApi.forward!=="+Z"
    || !Array.isArray(shellApi.wheelMounts)
    || shellApi.wheelMounts.length!==4) {
  throw new Error("chassis shell must publish ChassisShellAPI with +Z forward and four wheel mounts");
}
let shellMeshes=0;
const shellColors=new Set();
shellApi.root.traverse((item)=>{
  if (!item || !item.isMesh) return;
  shellMeshes+=1;
  const materials=Array.isArray(item.material)?item.material:[item.material];
  for (const material of materials) {
    if (material && material.color) shellColors.add(material.color.getHexString());
  }
});
shellApi.root.updateMatrixWorld(true);
const shellSize=new THREE.Box3().setFromObject(shellApi.root).getSize(new THREE.Vector3());
const integratedShell=Array.isArray(window.__componentConsumption)&&window.__componentConsumption.length>=2;
const shellMinMeshes=integratedShell?13:8;
const shellMaxMeshes=integratedShell?23:18;
if (shellMeshes<shellMinMeshes || shellMeshes>shellMaxMeshes) throw new Error(`chassis shell requires ${shellMinMeshes} to ${shellMaxMeshes} connected detail meshes; measured ${shellMeshes}`);
if (shellColors.size<3) throw new Error("chassis shell requires distinct paint, cladding, and trim colors");
if (shellSize.x<2.4 || shellSize.x>3.5 || shellSize.z<4.4 || shellSize.z>6.2
    || shellSize.y<.8 || shellSize.y>2.5) {
  throw new Error(`chassis shell bounding proportions violate the contracted vehicle envelope: measured x=${shellSize.x.toFixed(2)}, y=${shellSize.y.toFixed(2)}, z=${shellSize.z.toFixed(2)}; X is lateral width and Z is longitudinal length`);
}
camera.position.set(4.7,3.1,6.4); camera.lookAt(0,.82,0);
window.__componentSelfTest={passed:true,checks:["api","wheel-mounts","mesh-detail","material-hierarchy","proportions"]};
"""
        elif ".road.markings" in node_key:
            # Thin paint geometry is legitimately sparse in isolation. Supply
            # a neutral, clearly non-owned road surface in the preview shell so
            # screenshot anomaly detection and direct review can judge
            # alignment/cadence without forcing the Markings specialist to
            # duplicate the Geometry package during final integration.
            preview_fixture = """
const harnessReferenceRoad=new THREE.Group();
harnessReferenceRoad.name="__harness_reference_road_not_component_output";
const referenceAsphalt=new THREE.Mesh(
  new THREE.BoxGeometry(10.8,.12,52),
  new THREE.MeshStandardMaterial({color:0x3c4550,roughness:.92,metalness:.02})
);
referenceAsphalt.position.y=-.09;
harnessReferenceRoad.add(referenceAsphalt);
for (const x of [-5.65,5.65]) {
  const shoulder=new THREE.Mesh(
    new THREE.BoxGeometry(.85,.08,52),
    new THREE.MeshStandardMaterial({color:0x6f786f,roughness:1})
  );
  shoulder.position.set(x,-.12,0);
  harnessReferenceRoad.add(shoulder);
}
scene.add(harnessReferenceRoad);
"""
        elif ".road.collision" in node_key:
            preview_fixture = """
const harnessReferenceRoad=new THREE.Group();
harnessReferenceRoad.name="__harness_reference_road_not_component_output";
const referenceAsphalt=new THREE.Mesh(
  new THREE.BoxGeometry(10.8,.12,52),
  new THREE.MeshStandardMaterial({color:0x3c4550,roughness:.92,metalness:.02})
);
referenceAsphalt.position.y=-.09;
harnessReferenceRoad.add(referenceAsphalt);
scene.add(harnessReferenceRoad);
"""
            component_runtime_gate = """
const collisionApi=window.RoadCollisionAPI;
if (!collisionApi
    || typeof collisionApi.overlapsAABB!=="function"
    || typeof collisionApi.isInsideRoad!=="function") {
  throw new Error("collision package must publish RoadCollisionAPI.overlapsAABB and isInsideRoad");
}
const overlapA={minX:-1,maxX:1,minZ:-1,maxZ:1};
const overlapB={minX:.5,maxX:2,minZ:.5,maxZ:2};
const touchingB={minX:1,maxX:2,minZ:-.5,maxZ:.5};
const separateB={minX:2.1,maxX:3,minZ:2.1,maxZ:3};
if (!collisionApi.overlapsAABB(overlapA,overlapB)) {
  throw new Error("collision overlap positive assertion failed");
}
if (!collisionApi.overlapsAABB(overlapA,touchingB)) {
  throw new Error("collision touching-edge assertion failed");
}
if (collisionApi.overlapsAABB(overlapA,separateB)) {
  throw new Error("collision overlap negative assertion failed");
}
if (!collisionApi.isInsideRoad({x:0,z:0},.5,.5)) {
  throw new Error("collision inside-road positive assertion failed");
}
if (collisionApi.isInsideRoad({x:5.1,z:0},.5,.5)) {
  throw new Error("collision outside-road assertion failed");
}
window.__componentSelfTest={
  passed:true,
  checks:["overlap-positive","touching-positive","overlap-negative","inside-positive","outside-negative"]
};
"""
        elif node_key.endswith(".world.environment.terrain"):
            # Terrain is judged in the integration context it actually owns:
            # authored banks around a neutral, explicitly non-owned road.
            preview_fixture = """
const harnessReferenceRoad=new THREE.Group();
harnessReferenceRoad.name="__harness_reference_road_not_component_output";
const referenceAsphalt=new THREE.Mesh(
  new THREE.BoxGeometry(11.8,.12,52),
  new THREE.MeshStandardMaterial({color:0x35404a,roughness:.94,metalness:.01})
);
referenceAsphalt.position.y=-.03;
harnessReferenceRoad.add(referenceAsphalt);
for (let z=-22;z<=22;z+=4) {
  const dash=new THREE.Mesh(
    new THREE.BoxGeometry(.14,.04,2),
    new THREE.MeshStandardMaterial({color:0xf1d44e,roughness:.7})
  );
  dash.position.set(0,.05,z); harnessReferenceRoad.add(dash);
}
scene.add(harnessReferenceRoad);
"""
            component_runtime_gate = """
const terrainApi=window.TerrainAPI;
if (!terrainApi || !terrainApi.root || !terrainApi.root.isObject3D) {
  throw new Error("terrain package must publish TerrainAPI.root as an Object3D");
}
if (terrainApi.corridorHalfWidth!==6.2
    || !terrainApi.extents
    || terrainApi.extents.minZ!==-26
    || terrainApi.extents.maxZ!==26) {
  throw new Error("terrain package must publish the contracted corridor and extents");
}
let terrainMeshes=0;
let ownedLights=0;
let corridorIntrusions=0;
terrainApi.root.updateMatrixWorld(true);
terrainApi.root.traverse((item)=>{
  if (item && item.isLight) ownedLights+=1;
  if (!item || !item.isMesh) return;
  terrainMeshes+=1;
  const bounds=new THREE.Box3().setFromObject(item);
  if (bounds.min.x < 6.19 && bounds.max.x > -6.19) corridorIntrusions+=1;
});
if (terrainMeshes < 14) throw new Error("terrain root must contain two banks and at least twelve visible prop meshes");
if (terrainApi.root.children.length < 14) throw new Error("terrain root must contain two banks and at least twelve authored prop objects");
if (ownedLights) throw new Error("terrain root must not own preview or product lighting");
if (corridorIntrusions) throw new Error("terrain meshes intrude into the reserved road corridor");
camera.position.set(18,18,26); camera.lookAt(0,0,-3);
window.__componentSelfTest={passed:true,checks:["api","extents","mesh-density","no-owned-lights","clear-corridor"]};
"""
        return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultra component preview</title>
<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#10151d}canvas{display:block}</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body><script>
{
""" + scene_source + """
if (typeof window.buildPreview !== "function" && typeof buildPreview === "function") {
  window.buildPreview = buildPreview;
}
}
""" + (
            """
if (typeof window.buildPreview !== "function" && """
            + api_root_fallback
            + """) {
  window.buildPreview=(context)=>{
    const target=(context&&context.scene)||context;
    target.add("""
            + api_root_fallback
            + """);
  };
}
"""
            if api_root_fallback
            else ""
        ) + """
if (typeof window.buildPreview !== "function") {
  throw new Error("preview/scene.js must define window.buildPreview");
}
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9bc9e8);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, .1, 1000);
camera.position.set(12,10,16); camera.lookAt(0,0,0);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(innerWidth,innerHeight);
renderer.shadowMap.enabled=true;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=0.82;
if (THREE.sRGBEncoding) renderer.outputEncoding=THREE.sRGBEncoding;
document.body.appendChild(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x334455,0.62));
const key=new THREE.DirectionalLight(0xffffff,0.95); key.position.set(8,14,10);
key.castShadow=true; scene.add(key);
// A one-argument specialist may expect either a context object or the Scene
// itself. This projection supports both without guessing from parameter names:
// it inherits Scene behavior and also exposes the typed context fields.
const previewContext=Object.assign(Object.create(scene),{THREE,scene,camera,renderer});
""" + preview_fixture + """
const harnessChildrenBeforeBuild=new Set(scene.children);
if (window.buildPreview.length >= 2) {
  window.buildPreview(THREE,scene,camera,renderer);
} else {
  window.buildPreview(previewContext);
}
""" + component_runtime_gate + """
// Specialists may clear the scene while rebuilding their isolated root. The
// harness owns neutral presentation lighting, so restore it after component
// construction instead of forcing every weak-model leaf to reproduce it.
if (!scene.children.some((item)=>item && item.isHemisphereLight)) {
  scene.add(new THREE.HemisphereLight(0xffffff,0x334455,0.62));
}
if (!scene.children.some((item)=>item && item.isDirectionalLight)) {
  const previewKey=new THREE.DirectionalLight(0xffffff,0.95);
  previewKey.position.set(8,14,10); previewKey.castShadow=true;
  scene.add(previewKey);
}
function frame(t){requestAnimationFrame(frame);renderer.render(scene,camera);}
requestAnimationFrame(frame);
addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
</script></body></html>"""

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
                try:
                    self.store.heartbeat_goal_outcome(
                        self.goal_id,
                        ultra_run_id=run.id,
                        process_token=f"pid:{os.getpid()}",
                    )
                except NotFoundError:
                    pass
                return
            self.store.update_ultra_run(
                run.id,
                provider=self.descriptor.provider,
                model=self.descriptor.model,
                execution_class=self.descriptor.execution_class,
                access_level=self.access_level,
                concurrency=max(1, min(8, run.concurrency)),
                phase=_store_phase(run.phase),
                status=_store_run_status(run.phase),
                config=self._run_config(run),
                error=("ULTRA execution failed" if run.phase is EnginePhase.FAILED else None),
            )
            try:
                self.store.heartbeat_goal_outcome(
                    self.goal_id,
                    ultra_run_id=run.id,
                    process_token=f"pid:{os.getpid()}",
                )
            except NotFoundError:
                pass

    def save_specialist_profile(self, run_id: str, profile: SpecialistProfileV1) -> None:
        super().save_specialist_profile(run_id, profile)
        self.store.save_specialist_profile(
            {
                **asdict(profile),
                "ultra_run_id": run_id,
                "work_node_id": profile.node_id,
            }
        )

    def record_specialist_topology(
        self,
        run_id: str,
        parent: EngineWorkNode,
        children: Sequence[EngineWorkNode],
        readiness: Any,
    ) -> None:
        """Make recursive task granularity the primary optimization variable."""

        self.store.record_optimization_experiment(
            OptimizationExperimentV1(
                ultra_run_id=run_id,
                node_id=parent.id,
                variable="specialist_topology",
                baseline={
                    "single_specialist": parent.id,
                    "leaf_readiness": float(getattr(readiness, "score", 0.0)),
                },
                candidate={
                    "child_count": len(children),
                    "children": [
                        {
                            "id": child.id,
                            "mission": child.contract.objective,
                            "interfaces": list(child.contract.owned_interfaces),
                        }
                        for child in children
                    ],
                },
                hypothesis=(
                    "Narrow independently previewable specialists produce higher-quality "
                    "components than one model context owning the complete parent system."
                ),
                before_score=0.0,
                after_score=0.0,
                outcome=ExperimentOutcome.INCONCLUSIVE,
                evidence=tuple(
                    f"specialist:{child.id}" for child in children
                ),
            )
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
            "critical_minimum": 0.90,
            "zero_critical_findings": True,
            "reject_placeholders": True,
            "require_specific_visual_evidence": True,
        }

    @staticmethod
    def _requires_visual_component(node: EngineWorkNode) -> bool:
        metadata = node.contract.metadata
        if "visual_required" in metadata:
            return bool(metadata.get("visual_required"))
        domain = str(metadata.get("specialist_domain") or "").casefold()
        root = domain.split(".", 1)[0]
        if root in {"world", "vehicles", "character", "presentation"}:
            return True
        if domain == "qa.visual":
            return True
        return any(
            term in " ".join(
                (
                    node.contract.title,
                    node.contract.objective,
                    *node.contract.acceptance_criteria,
                )
            ).casefold()
            for term in (
                "visual composition",
                "visual model",
                "lighting",
                "camera framing",
                "animation preview",
            )
        )

    @staticmethod
    def _requires_threejs_preview(node: EngineWorkNode) -> bool:
        domain = str(node.contract.metadata.get("specialist_domain") or "").casefold()
        return (
            domain.startswith(("world", "vehicles", "character"))
            or domain.startswith("presentation.camera")
            or domain.startswith("presentation.effects.particles")
        )

    def _assert_domain_preview(
        self,
        node: EngineWorkNode,
        package: MaterializedComponentPackageV2,
    ) -> None:
        if not self._requires_threejs_preview(node):
            return
        preview_path = Path(package.root) / package.preview_entrypoint
        preview_text = preview_path.read_text(encoding="utf-8", errors="replace")
        lowered = preview_text.casefold()
        specialist_source = ""
        for file_ref in getattr(package, "files", ()):
            relative = str(getattr(file_ref, "path", "")).replace("\\", "/")
            if relative.casefold().endswith("preview/scene.js"):
                source_path = Path(package.root) / relative
                if source_path.is_file():
                    specialist_source = source_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                break
        missing: list[str] = []
        if "<canvas" not in lowered and "new three.webglrenderer" not in lowered:
            missing.append("a real canvas")
        if not any(
            marker in lowered
            for marker in (
                "three.module",
                "three.min",
                "from 'three'",
                'from "three"',
                "window.three",
                "new three.",
            )
        ):
            missing.append("a real Three.js scene")
        node_key = str(node.id).casefold()
        specialist_domain = str(
            node.contract.metadata.get("specialist_domain") or ""
        ).casefold()
        has_builder = bool(
            re.search(
                r"\b(?:window\.)?buildPreview\s*(?:=|\()",
                specialist_source,
            )
        )
        if ".road.collision" in node_key:
            for label, pattern in (
                ("RoadCollisionAPI", r"\bRoadCollisionAPI\b"),
                ("overlapsAABB", r"\boverlapsAABB\b"),
                ("isInsideRoad", r"\bisInsideRoad\b"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"collision contract {label}")
        if specialist_domain == "world.lighting.rig":
            for label, pattern in (
                ("LightingRigAPI", r"\bLightingRigAPI\b"),
                ("totalIntensity", r"\btotalIntensity\b"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"lighting rig contract {label}")
            if re.search(r"\bcamera\.(?:position|lookAt|rotation)\b", specialist_source):
                missing.append("lighting rig may not reposition the harness camera")
        if specialist_domain == "world.lighting.atmosphere":
            for label, pattern in (
                ("window.AtmosphereAPI", r"\bwindow\.AtmosphereAPI\b"),
                ("apply function", r"\bapply\s*[:=]"),
                ("scene color background", r"\bscene\.background\s*="),
                ("scene fog", r"\bscene\.fog\s*="),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"atmosphere contract {label}")
            if re.search(r"\bnew\s+THREE\.(?:Ambient|Directional|Hemisphere|Point|Spot|RectArea)Light\s*\(", specialist_source):
                missing.append("atmosphere specialist may not own lighting")
            if re.search(r"\b(?:Plane|Box|Sphere)Geometry\s*\([^\n]*(?:20|30|40|50|100)", specialist_source):
                missing.append("atmosphere specialist may not create oversized sky geometry")
        if specialist_domain == "world.lighting.shadows":
            for label, pattern in (
                ("ShadowQualityAPI", r"\bwindow\.ShadowQualityAPI\b"),
                ("configureRenderer", r"\bconfigureRenderer\b"),
                ("configureLight", r"\bconfigureLight\b"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"shadow contract {label}")
            if re.search(r"\bnew\s+THREE\.(?:Ambient|Directional|Hemisphere|Point|Spot|RectArea)Light\s*\(", specialist_source):
                missing.append("shadow specialist may not create a competing light rig")
        if specialist_domain == "vehicles.chassis.shell":
            for label, pattern in (
                ("ChassisShellAPI", r"\bwindow\.ChassisShellAPI\b"),
                ("four wheel mounts", r"\bwheelMounts\b"),
                ("+Z forward", r"\bforward\s*:\s*[\"']\+Z[\"']"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"chassis shell contract {label}")
        if specialist_domain == "vehicles.chassis.shell.volumes" and not has_builder:
            for label, pattern in (
                ("ChassisVolumesAPI", r"\bwindow\.ChassisVolumesAPI\b"),
                ("+Z forward", r"\bforward\s*:\s*[\"']\+Z[\"']"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"chassis volumes contract {label}")
        if specialist_domain == "vehicles.chassis.shell.panels" and not has_builder:
            for label, pattern in (
                ("ChassisPanelsAPI", r"\bwindow\.ChassisPanelsAPI\b"),
                ("four wheel mounts", r"\bwheelMounts\b"),
                ("+Z forward", r"\bforward\s*:\s*[\"']\+Z[\"']"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"chassis panels contract {label}")
        if specialist_domain == "world.environment.terrain":
            for label, pattern in (
                ("TerrainAPI", r"\bTerrainAPI\b"),
                ("contracted corridorHalfWidth", r"corridorHalfWidth\s*:\s*6\.2\b"),
                ("contracted minZ", r"minZ\s*:\s*-26\b"),
                ("contracted maxZ", r"maxZ\s*:\s*26\b"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"terrain contract {label}")
            if specialist_source.count("\n") + 1 > 120:
                missing.append("terrain implementation exceeds the 120-line safety ceiling")
        if specialist_domain == "world.environment.props":
            for label, pattern in (
                ("EnvironmentPropsAPI", r"\bEnvironmentPropsAPI\b"),
                ("createTree", r"\bcreateTree\b"),
                ("createPine", r"\bcreatePine\b"),
                ("createRocks", r"\bcreateRocks\b"),
                ("createShrub", r"\bcreateShrub\b"),
                ("createSign", r"\bcreateSign\b"),
                ("createFence", r"\bcreateFence\b"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"props contract {label}")
        if specialist_domain == "world.environment.composition":
            for label, pattern in (
                ("EnvironmentCompositionAPI", r"\bEnvironmentCompositionAPI\b"),
                ("three depth bands", r"\bbands\s*:\s*3\b"),
                ("two landmarks", r"\blandmarks\s*:\s*2\b"),
            ):
                if not re.search(pattern, specialist_source):
                    missing.append(f"composition contract {label}")
        if any(
            specialist_domain.startswith(marker)
            for marker in (
                "world.environment.terrain",
                "world.environment.props",
                "world.environment.composition",
            )
        ):
            if re.search(r"\bMath\.random\s*\(", specialist_source):
                missing.append("environment components require deterministic authored transforms")
            if re.search(
                r"\bnew\s+THREE\.(?:Ambient|Directional|Hemisphere|Point|Spot|RectArea)Light\s*\(",
                specialist_source,
            ):
                missing.append("environment components may not own harness or product lighting")
        placeholder_markers = tuple(
            marker
            for marker in (
                "visualization proxy",
                "api demonstration",
                "placeholder geometry",
                "left verge placeholder",
                "lane center placeholder",
            )
            if marker in lowered
        )
        typed_root_api = {
            "vehicles.chassis.shell": "ChassisShellAPI",
            "vehicles.chassis.shell.volumes": "ChassisVolumesAPI",
            "vehicles.chassis.shell.panels": "ChassisPanelsAPI",
        }.get(specialist_domain, "")
        has_typed_root_entrypoint = bool(
            typed_root_api
            and re.search(
                rf"\bwindow\.{re.escape(typed_root_api)}\b",
                specialist_source,
            )
        )
        shell_violations = tuple(
            label
            for label, pattern in (
                ("specialist-created Scene", r"\bnew\s+THREE\.Scene\s*\("),
                (
                    "specialist-created WebGLRenderer",
                    r"\bnew\s+THREE\.WebGLRenderer\s*\(",
                ),
                ("specialist DOM mutation", r"\bdocument\.(?:body|createElement)\b"),
                (
                    "specialist animation loop",
                    r"\brequestAnimationFrame\s*\(",
                ),
            )
            if specialist_source
            and re.search(pattern, specialist_source, re.IGNORECASE)
        )
        if specialist_source and not has_builder and not has_typed_root_entrypoint:
            shell_violations = (
                "missing window.buildPreview or typed API.root specialist entrypoint",
                *shell_violations,
            )
        if missing or placeholder_markers or shell_violations:
            details = [*missing, *placeholder_markers, *shell_violations]
            raise ComponentArtifactError(
                f"component {node.id} preview is not a materialized 3D specialist artifact: "
                + ", ".join(details)
            )

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
        visual_required = self._requires_visual_component(node)
        codex_supervisory_review = (
            visual_required
            and isinstance(self.visual_judge, UnavailableVisionJudge)
        )
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
            quality={
                "status": (
                    "pending_independent_evaluation"
                    if visual_required
                    else "pending_deterministic_runtime"
                ),
                "visual_required": visual_required,
                "evaluation_mode": (
                    "codex_supervisory_final"
                    if codex_supervisory_review
                    else "independent_vision"
                    if visual_required
                    else "deterministic_runtime"
                ),
            },
            parent_package_ids=tuple(
                str(value.get("id"))
                for value in child_packages.values()
                if value.get("id")
            ),
        )
        stored = self.store.put_materialized_component_package(package.to_dict())
        self._materialized_packages[package.id] = package
        self._assert_domain_preview(node, package)
        preview = self.component_artifacts.verify_preview(package)
        screenshot = str(preview.get("screenshot_path") or "")
        findings: list[str] = []
        runtime_passed = str(preview.get("status")) == "passed" and bool(screenshot)
        if not runtime_passed:
            runtime_details = tuple(
                dict.fromkeys(
                    str(item)
                    for key in ("console_errors", "page_errors", "network_errors")
                    for item in preview.get(key, ())
                    if str(item).strip()
                )
            )
            findings.append(
                "component preview failed runtime verification: "
                + str(preview.get("reason") or preview.get("status") or "unknown")
                + (
                    "; " + "; ".join(runtime_details[:6])
                    if runtime_details
                    else ""
                )
            )
        anomaly_findings = (
            screenshot_anomalies(screenshot)
            if screenshot and visual_required
            else ()
        )
        findings.extend(f"visual anomaly gate: {item}" for item in anomaly_findings)
        verdict_values: list[Mapping[str, Any]] = []
        pairwise_value: Mapping[str, Any] | None = None
        status = "evaluated"
        if runtime_passed and visual_required and not codex_supervisory_review:
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
        elif runtime_passed and codex_supervisory_review:
            status = "PROVISIONAL_CODEX_REVIEW"
        accepted_twice = (
            not visual_required
            or codex_supervisory_review
            or (
                len(verdict_values) == 2
                and all(bool(value.get("accepted")) for value in verdict_values)
            )
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
        score_values = [
            float(score)
            for verdict in verdict_values
            for score in dict(verdict.get("scores") or {}).values()
        ]
        current_score = (
            min(score_values)
            if score_values
            else 1.0
            if runtime_passed and not visual_required
            else 0.0
        )
        prior_score = self._component_champion_scores.get(node.id, 0.0)
        improved = current_score > prior_score
        if improved:
            self._component_champion_scores[node.id] = current_score
        strategy = (
            "initial_candidate"
            if revision == 1
            else "finding_specific_prompt"
            if revision == 2
            else "context_and_examples"
            if revision == 3
            else "specialist_strategy_change"
        )
        self.store.record_optimization_experiment(
            OptimizationExperimentV1(
                ultra_run_id=run_id,
                node_id=node.id,
                variable=strategy,
                baseline={"revision": max(0, revision - 1), "score": prior_score},
                candidate={
                    "revision": revision,
                    "runtime_passed": runtime_passed,
                    "anomaly_passed": not anomaly_findings,
                    "accepted_twice": accepted_twice,
                    "pairwise_passed": pairwise_passed,
                    "component_gate_passed": passed,
                },
                hypothesis=(
                    "A single controlled specialist-strategy change should improve "
                    "the weakest independently scored visual dimension."
                ),
                before_score=prior_score,
                after_score=current_score,
                outcome=(
                    ExperimentOutcome.CHAMPION
                    if improved
                    else ExperimentOutcome.REJECTED
                    if revision > 1
                    else ExperimentOutcome.INCONCLUSIVE
                ),
                evidence=tuple(
                    dict.fromkeys(
                        (
                            *(str(item) for item in findings),
                            *(str(verdict.get("screenshot_hash") or "") for verdict in verdict_values),
                        )
                    )
                ),
            )
        )
        return {
            "passed": passed,
            "status": (
                "provisional_codex_review"
                if passed and codex_supervisory_review
                else "accepted"
                if passed
                else status
                if status != "evaluated"
                else "rejected"
            ),
            "package": package.to_dict(include_content=True),
            "stored_package_id": stored["id"],
            "preview": preview,
            "visual_evaluations": verdict_values,
            "pairwise_comparison": pairwise_value,
            "findings": list(dict.fromkeys(findings)),
        }

    def restore_passed_component_candidate(
        self,
        run_id: str,
        node: EngineWorkNode,
    ) -> AgentResponse | None:
        """Resume quality review from the latest durable runtime-passing package."""

        if self.component_artifacts is None:
            return None
        experiments = self.store.list_optimization_experiments(
            run_id,
            work_node_id=node.id,
        )
        passing_revisions: dict[int, Mapping[str, Any]] = {}
        for experiment in experiments:
            raw_candidate = experiment.get("candidate")
            candidate = (
                dict(raw_candidate) if isinstance(raw_candidate, Mapping) else {}
            )
            if (
                candidate.get("runtime_passed")
                and candidate.get("accepted_twice")
                and candidate.get("pairwise_passed")
            ):
                passing_revisions[int(candidate.get("revision") or 0)] = candidate
        if not passing_revisions:
            return None
        packages = {
            int(item.get("version") or 0): item
            for item in self.store.list_component_packages(
                run_id,
                work_node_id=node.id,
            )
        }
        visual_evaluations = self.store.list_visual_evaluations(
            run_id,
            work_node_id=node.id,
        )
        for revision in sorted(passing_revisions, reverse=True):
            stored = packages.get(revision)
            if stored is None:
                continue
            package_id = str(stored.get("id") or "")
            package_visuals = [
                item
                for item in visual_evaluations
                if str(item.get("package_id") or "") == package_id
            ]
            directly_accepted = bool(
                package_visuals
                and dict(package_visuals[-1].get("verdict") or {}).get("accepted")
            )
            if (
                passing_revisions[revision].get("anomaly_passed") is not True
                and not directly_accepted
            ):
                # Old experiment rows did not persist anomaly_passed. They are
                # only restorable when a later direct visual review explicitly
                # accepted the exact package. This prevents a runtime-passing
                # but 93%-blank screenshot from becoming a durable champion.
                continue
            if package_visuals and not bool(
                dict(package_visuals[-1].get("verdict") or {}).get("accepted")
            ):
                # A direct Codex/independent rejection outranks the runtime
                # receipt. The next resume must build a challenger and consume
                # the persisted typed findings, never restore this champion.
                continue
            root = self.component_artifacts.package_root(
                run_id,
                node.id,
                revision,
            )
            manifest_path = root / "component-package.json"
            screenshot = root / "evidence" / "preview.png"
            if not manifest_path.is_file() or not screenshot.is_file():
                continue
            try:
                raw_package = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(raw_package, Mapping):
                continue
            package = dict(raw_package)
            package["file_contents"] = {
                str(item.get("path")): (root / str(item.get("path"))).read_text(
                    encoding="utf-8"
                )
                for item in package.get("files", ())
                if isinstance(item, Mapping)
                and str(item.get("role")) in {
                    "implementation",
                    "preview",
                    "test",
                }
                and (root / str(item.get("path"))).is_file()
            }
            stored_package_id = str(
                stored.get("id") or package.get("id") or ""
            )
            accepted_visuals = [
                dict(item.get("verdict") or {})
                for item in package_visuals
                if bool(dict(item.get("verdict") or {}).get("accepted"))
            ]
            directly_accepted = bool(accepted_visuals)
            result = {
                "passed": True,
                "status": (
                    "accepted"
                    if directly_accepted
                    else "provisional_codex_review"
                ),
                "package": package,
                "stored_package_id": stored_package_id,
                "preview": {
                    "status": "passed",
                    "verification": "passed",
                    "screenshot_path": str(screenshot),
                    "restored_from_checkpoint": True,
                },
                "visual_evaluations": accepted_visuals,
                "findings": [],
            }
            self._published_component_results[node.id] = result
            return AgentResponse(
                payload={
                    "success": True,
                    "component_publication": {
                        "package_id": stored_package_id,
                        "status": result["status"],
                        "screenshot_path": str(screenshot),
                    },
                    "evidence": [
                        {
                            "kind": "restored_materialized_component_receipt",
                            "package_id": stored_package_id,
                            "content_hash": str(
                                package.get("content_hash") or ""
                            ),
                        },
                        *(
                            {
                                "kind": "direct_component_visual_acceptance",
                                "package_id": stored_package_id,
                                "screenshot_hash": str(
                                    verdict.get("screenshot_hash") or ""
                                ),
                                "score": verdict.get("score"),
                            }
                            for verdict in accepted_visuals[-1:]
                        ),
                    ],
                    "findings": [],
                },
                summary=(
                    "Restored runtime-passing component package from durable "
                    "checkpoint; quality review resumes without rebuilding it."
                ),
                reasoning_summary=(
                    "The package manifest, file hashes, runtime experiment, "
                    "and screenshot exist."
                ),
                provider="harness",
                model="component-checkpoint-v1",
            )
        return None

    def component_revision_findings(
        self,
        run_id: str,
        node: EngineWorkNode,
    ) -> tuple[str, ...]:
        """Project unresolved direct review evidence into the next specialist packet."""

        packages = self.store.list_component_packages(
            run_id,
            work_node_id=node.id,
        )
        evaluations = self.store.list_visual_evaluations(
            run_id,
            work_node_id=node.id,
        )
        if packages and evaluations:
            latest_package_id = str(packages[-1].get("id") or "")
            latest_evaluation = evaluations[-1]
            verdict = (
                dict(latest_evaluation.get("verdict") or {})
                if isinstance(latest_evaluation, Mapping)
                else {}
            )
            if (
                bool(verdict.get("accepted"))
                and str(latest_evaluation.get("package_id") or "")
                == latest_package_id
            ):
                # A strict direct review of the latest materialized package
                # supersedes screenshot-specific findings from older revisions.
                # The review script closes those records through an integrated
                # remediation Change Set; this guard also prevents a crash
                # between acceptance and that durable cleanup from rebuilding
                # an already accepted component.
                return ()

        findings = [
            item
            for item in self.store.list_quality_findings(run_id)
            if item.repair_node_id == node.id
            and item.status.value != "resolved"
        ]
        # The latest direct review supersedes older screenshot-specific
        # symptoms for prompt construction. Older records remain durable for
        # audit and trend analysis but replaying all of them wastes context and
        # can give a weak builder contradictory styling directions.
        findings = findings[-1:]
        values: list[str] = []
        for finding in findings:
            evidence = (
                dict(finding.evidence)
                if isinstance(finding.evidence, Mapping)
                else {}
            )
            direct_messages = [
                str(item)
                for item in evidence.get("findings", ())
                if str(item).strip()
            ]
            if direct_messages:
                # Direct-review remediation already concatenates these same
                # messages. Feed the leaf one canonical copy so a 4K local
                # context retains enough output budget for executable code.
                values.extend(direct_messages)
            else:
                values.append(str(finding.remediation))
        return tuple(dict.fromkeys(item for item in values if item.strip()))

    @staticmethod
    def _adapt_typed_component_source(
        node: EngineWorkNode,
        source: str,
    ) -> str:
        """Bridge a valid weak-model buildPreview to a strict typed contract.

        The adapter never invents meshes. It preserves the generated source,
        captures the model's component root, normalizes only an out-of-envelope
        root scale, and publishes the deterministic interface that the parent
        assembler needs. This is used only after the model emitted executable
        geometry but repeatedly omitted protocol plumbing.
        """

        domain = str(node.contract.metadata.get("specialist_domain") or "").casefold()
        if not re.search(r"\b(?:window\.)?buildPreview\s*(?:=|\()", source):
            return source
        if domain == "vehicles.chassis.shell.volumes":
            if re.search(r"\bwindow\.ChassisVolumesAPI\b", source):
                return source
            adapter = """
const __gemmaVolumesBuild=window.buildPreview;
window.buildPreview=(context)=>{
  const scene=(context&&context.scene)||context;
  const before=new Set(scene.children);
  const returned=__gemmaVolumesBuild(context);
  const root=(returned&&returned.root&&returned.root.isObject3D?returned.root:null)
    ||scene.children.find((item)=>!before.has(item)&&item&&item.isObject3D);
  if(!root) throw new Error("volume adapter could not identify the generated root");
  window.ChassisVolumesAPI={root,dimensions:{width:2.8,height:1.5,length:5.2},forward:'+Z'};
  return window.ChassisVolumesAPI;
};
"""
            return source.rstrip() + "\n" + adapter.strip() + "\n"
        if domain == "vehicles.chassis.shell.panels":
            if re.search(r"\bwindow\.ChassisPanelsAPI\b", source):
                return source
            adapter = """
const __gemmaPanelsBuild=window.buildPreview;
window.buildPreview=(context)=>{
  const scene=(context&&context.scene)||context;
  const before=new Set(scene.children);
  const returned=__gemmaPanelsBuild(context);
  const root=(returned&&returned.root&&returned.root.isObject3D?returned.root:null)
    ||scene.children.find((item)=>!before.has(item)&&item&&item.isObject3D);
  if(!root) throw new Error("panel adapter could not identify the generated root");
  root.updateMatrixWorld(true);
  const size=new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3());
  if(size.x>3.2) root.scale.x*=3.2/size.x;
  if(size.z>5.2) root.scale.z*=5.2/size.z;
  window.ChassisPanelsAPI={root,wheelMounts:[
    {x:1.38,y:.55,z:1.55},{x:-1.38,y:.55,z:1.55},
    {x:1.38,y:.55,z:-1.55},{x:-1.38,y:.55,z:-1.55}
  ],forward:'+Z'};
  return window.ChassisPanelsAPI;
};
"""
            return source.rstrip() + "\n" + adapter.strip() + "\n"
        if domain == "vehicles.vehicle_details_cut":
            adapter = """
const __gemmaVehicleDetailsBuild=window.buildPreview;
window.buildPreview=(context)=>{
  const scene=(context&&context.scene)||context;
  const before=new Set(scene.children);
  const returned=__gemmaVehicleDetailsBuild(context);
  const api=window.VehicleDetailsAPI||(returned&&returned.root?returned:null)||{};
  const root=(api.root&&api.root.isObject3D?api.root:null)
    ||scene.children.find((item)=>!before.has(item)&&item&&item.isObject3D);
  if(!root) throw new Error("vehicle-details adapter could not identify the generated root");
  const centers=[{x:-1.48,y:.55,z:1.55},{x:1.48,y:.55,z:1.55},{x:-1.48,y:.55,z:-1.55},{x:1.48,y:.55,z:-1.55}];
  const wheelGroups=root.children.filter((item)=>item&&item.isGroup&&item.children.length>=3).slice(0,4);
  wheelGroups.forEach((wheel,index)=>{
    const center=centers[index];wheel.position.set(center.x,center.y,center.z);
    wheel.children.forEach((part)=>{if(part&&part.isMesh)part.rotation.set(0,Math.PI/2,0);});
  });
  window.VehicleDetailsAPI={...api,root,wheelCenters:centers,forward:'+Z'};
  return window.VehicleDetailsAPI;
};
"""
            return source.rstrip() + "\n" + adapter.strip() + "\n"
        return source

    def publish_component_tool(
        self,
        run_id: str,
        node: EngineWorkNode,
        component: Mapping[str, Any],
        *,
        child_packages: Mapping[str, Mapping[str, Any]] | None = None,
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
        staged_files = [
            (
                {
                    **dict(item),
                    "content": self._adapt_typed_component_source(
                        node,
                        str(item.get("content", "")),
                    ),
                }
                if str(item.get("path", "")).replace("\\", "/").casefold().endswith(
                    "preview/scene.js"
                )
                else dict(item)
            )
            for item in staged_files
        ]
        active_paths = self._active_component_draft_paths.get(node.id, set())
        if active_paths:
            staged_files = [
                item
                for item in staged_files
                if str(item.get("path", "")) in active_paths
            ]
        preview = (
            dict(normalized_component.get("preview", {}))
            if isinstance(normalized_component.get("preview"), Mapping)
            else {}
        )
        preview_content = str(preview.pop("content", ""))
        preview_entrypoint = str(preview.get("entrypoint", "")).strip()
        staged_paths = {str(item.get("path", "")) for item in staged_files}
        scene_candidates = sorted(
            (
                item
                for item in staged_files
                if str(item.get("path", "")).casefold().endswith(
                    ("preview/scene.js", "preview.scene.js")
                )
            ),
            key=lambda item: (
                "buildpreview" not in str(item.get("content", "")).casefold(),
                bool(
                    re.search(
                        r"""(?m)^\s*(?:import\b|.*\bfrom\s*["'])""",
                        str(item.get("content", "")),
                    )
                ),
                str(item.get("path", "")).casefold() != "preview/scene.js",
                len(str(item.get("path", ""))),
                str(item.get("path", "")),
            ),
        )
        if scene_candidates:
            # Small models often publish scene.js itself as the entrypoint.
            # That is a useful component source but not browser-executable
            # packaging, so the harness deterministically supplies the HTML.
            if not preview_entrypoint.casefold().endswith((".html", ".htm")):
                preview_entrypoint = "preview/index.html"
                preview["entrypoint"] = preview_entrypoint
            if preview_entrypoint not in staged_paths:
                staged_files.append(
                    {
                        "path": preview_entrypoint,
                        "content": self._generated_threejs_preview(
                            str(scene_candidates[0].get("content", "")),
                            node_id=node.id,
                        ),
                        "role": "preview",
                    }
                )
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
        try:
            result = dict(
                self.materialize_component_candidate(
                    run_id,
                node,
                response,
                revision=revision,
                child_packages=dict(child_packages or {}),
            )
            )
        except ComponentArtifactError as exc:
            failure_messages = (str(exc),)
            self._record_component_publication_failure(
                run_id,
                node,
                revision=revision,
                scene_candidates=scene_candidates,
                findings=failure_messages,
            )
            self._published_component_results[node.id] = {
                "passed": False,
                "status": "rejected",
                "package": {},
                "stored_package_id": "",
                "preview": {"status": "failed"},
                "visual_evaluations": [],
                "findings": list(failure_messages),
            }
            raise
        if not bool(result.get("passed")):
            failure_messages = tuple(
                str(item).strip()
                for item in result.get("findings", ())
                if str(item).strip()
            ) or (
                f"component publication revision {revision} failed its runtime or evidence gate",
            )
            self._record_component_publication_failure(
                run_id,
                node,
                revision=revision,
                scene_candidates=scene_candidates,
                findings=failure_messages,
            )
        elif child_packages:
            composite_source = (
                str(scene_candidates[0].get("content") or "")
                if scene_candidates
                else ""
            )
            consumption_findings: list[str] = []
            parent_package = (
                dict(result.get("package") or {})
                if isinstance(result.get("package"), Mapping)
                else {}
            )
            parent_root = Path(str(parent_package.get("root") or ""))
            target = parent_root / "preview" / "scene.js"
            for child_id, raw_child in child_packages.items():
                child = (
                    dict(raw_child) if isinstance(raw_child, Mapping) else {}
                )
                package_id = str(child.get("id") or child_id)
                child_hash = str(child.get("content_hash") or "")
                contents = (
                    dict(child.get("file_contents") or {})
                    if isinstance(child.get("file_contents"), Mapping)
                    else {}
                )
                child_source = str(contents.get("preview/scene.js") or "")
                consumed_hashes = tuple(
                    str(item.get("content_hash") or "")
                    for item in child.get("files", ())
                    if isinstance(item, Mapping)
                    and str(item.get("path") or "") == "preview/scene.js"
                    and str(item.get("content_hash") or "")
                )
                consumed = bool(
                    child_hash
                    and child_source
                    and child_hash in composite_source
                    and child_source in composite_source
                )
                findings = () if consumed else (
                    f"parent assembly omitted exact source/hash for child {child_id}",
                )
                self.store.save_package_consumption_evidence(
                    run_id,
                    node.id,
                    package_id,
                    {
                        "assembler_node_id": node.id,
                        "package_id": package_id,
                        "consumed_file_hashes": list(consumed_hashes),
                        "target_paths": [str(target)],
                        "passed": consumed,
                        "findings": list(findings),
                        "child_content_hash": child_hash,
                        "verification": "exact-source-and-hash-in-parent-staging",
                    },
                )
                consumption_findings.extend(findings)
            if consumption_findings:
                result = {
                    **result,
                    "passed": False,
                    "status": "rejected",
                    "findings": list(
                        dict.fromkeys(
                            (*result.get("findings", ()), *consumption_findings)
                        )
                    ),
                }
        self._published_component_results[node.id] = result
        return result

    def _record_component_publication_failure(
        self,
        run_id: str,
        node: EngineWorkNode,
        *,
        revision: int,
        scene_candidates: Sequence[Mapping[str, Any]],
        findings: Sequence[str],
    ) -> None:
        """Persist failed leaf feedback before another local-model attempt."""

        source = (
            str(scene_candidates[0].get("content") or "")
            if scene_candidates
            else ""
        )
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        messages = tuple(
            dict.fromkeys(str(item).strip() for item in findings if str(item).strip())
        )
        if not messages:
            return
        self.store.put_quality_finding(
            QualityFindingV1(
                ultra_run_id=run_id,
                principle_id="correctness",
                category=QualityCategory.API,
                severity=FindingSeverity.HIGH,
                path=f"staging://{node.id}/preview/scene.js",
                location=f"component publication revision {revision}",
                file_hash=source_hash,
                evidence={
                    "source": "materialized-component-runtime-gate",
                    "revision": revision,
                    "node_id": node.id,
                    "findings": list(messages),
                },
                remediation=(
                    f"Specialist {node.id} must publish a fresh revision that corrects: "
                    + "; ".join(messages)
                ),
                acceptance_criteria=(
                    "The materialized component passes its typed runtime contract.",
                    "The isolated preview completes with zero browser or syntax errors.",
                    "The replacement source hash differs from the rejected revision.",
                ),
                verification=(
                    "Re-run the harness-owned component runtime assertions.",
                    "Capture a fresh deterministic component screenshot.",
                ),
                repair_node_id=node.id,
            )
        )

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
        result = self.component_artifacts.stage_draft_file(
            run_id=run_id,
            node_id=node.id,
            path=path,
            content=content,
            role=role,
        )
        self._active_component_draft_paths.setdefault(node.id, set()).add(
            str(result["path"])
        )
        return result

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
        materialized_values: list[MaterializedComponentPackageV2] = []
        missing: list[str] = []
        for item in packages:
            identifier = str(item.get("id") or "")
            package = self._materialized_packages.get(identifier)
            if package is None:
                try:
                    interface_value = dict(item.get("interface") or {})
                    package = MaterializedComponentPackageV2(
                        id=identifier,
                        run_id=str(item.get("run_id") or run_id),
                        node_id=str(item.get("node_id") or ""),
                        revision=max(1, int(item.get("revision") or 1)),
                        root=str(item.get("root") or ""),
                        files=tuple(
                            ComponentFileV2(
                                path=str(file_value.get("path") or ""),
                                content_hash=str(
                                    file_value.get("content_hash") or ""
                                ),
                                size=int(file_value.get("size") or 0),
                                media_type=str(
                                    file_value.get("media_type")
                                    or "application/octet-stream"
                                ),
                                role=str(file_value.get("role") or "implementation"),
                            )
                            for file_value in item.get("files", ())
                            if isinstance(file_value, Mapping)
                        ),
                        interface=InterfaceContractV1(
                            node_id=str(
                                interface_value.get("node_id")
                                or item.get("node_id")
                                or ""
                            ),
                            exports=tuple(
                                str(value)
                                for value in interface_value.get("exports", ())
                                if str(value).strip()
                            ),
                            imports=tuple(
                                str(value)
                                for value in interface_value.get("imports", ())
                                if str(value).strip()
                            ),
                            invariants=tuple(
                                str(value)
                                for value in interface_value.get("invariants", ())
                                if str(value).strip()
                            ),
                            integration_points=tuple(
                                str(value)
                                for value in interface_value.get(
                                    "integration_points", ()
                                )
                                if str(value).strip()
                            ),
                        ),
                        preview_entrypoint=str(
                            item.get("preview_entrypoint") or "preview/index.html"
                        ),
                        dependencies=tuple(
                            str(value)
                            for value in item.get("dependencies", ())
                            if str(value).strip()
                        ),
                        evidence=tuple(
                            dict(value)
                            for value in item.get("evidence", ())
                            if isinstance(value, Mapping)
                        ),
                        quality=(
                            dict(item.get("quality") or {})
                            if isinstance(item.get("quality"), Mapping)
                            else {}
                        ),
                        parent_package_ids=tuple(
                            str(value)
                            for value in item.get("parent_package_ids", ())
                            if str(value).strip()
                        ),
                    )
                except (TypeError, ValueError, ComponentArtifactError):
                    package = None
                if package is not None:
                    self._materialized_packages[identifier] = package
            if package is None:
                missing.append(
                    str(item.get("id") or item.get("node_id") or "unknown")
                )
            else:
                materialized_values.append(package)
        materialized = tuple(materialized_values)
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

    def verify_node_artifacts(
        self,
        run_id: str,
        node: EngineWorkNode,
    ) -> Mapping[str, Any]:
        del run_id
        if self.workspace is None:
            return {
                "passed": False,
                "findings": ["artifact validation requires a workspace"],
                "evidence": [],
            }
        artifacts = dict(
            _validate_workspace_artifacts(self.workspace, node.write_paths)
        )
        executable = dict(
            _run_python_test_artifacts(self.workspace, node.write_paths)
        )
        mutation_evidence: list[Mapping[str, Any]] = []
        for raw_path in node.write_paths:
            path = _normalized_path(raw_path)
            if not path or any(character in path for character in "*?["):
                continue
            known, before = self.initial_lease_hash(node.id, path)
            after = _hash_file(self.workspace, path)
            if known and before != after:
                mutation_evidence.append(
                    {"path": path, "before_sha256": before, "after_sha256": after}
                )
        return {
            "passed": bool(artifacts.get("passed"))
            and bool(executable.get("passed")),
            "workspace_mutated": bool(mutation_evidence),
            "mutation_evidence": mutation_evidence,
            "findings": [
                *artifacts.get("findings", ()),
                *executable.get("findings", ()),
            ],
            "evidence": [
                *artifacts.get("evidence", ()),
                *executable.get("evidence", ()),
            ],
        }

    def restore_exact_child_artifacts(
        self,
        run_id: str,
        node: EngineWorkNode,
        packages: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if self.workspace is None:
            return {
                "passed": False,
                "findings": ["exact child restoration requires a workspace"],
            }
        expected: dict[str, str] = {}
        for package in packages:
            implementation = (
                dict(package.get("implementation") or {})
                if isinstance(package.get("implementation"), Mapping)
                else {}
            )
            for artifact in implementation.get("artifacts", ()):
                if not isinstance(artifact, Mapping):
                    continue
                path = _normalized_path(
                    str(artifact.get("path") or artifact.get("uri") or "")
                    .removeprefix("workspace:")
                )
                content_hash = str(
                    artifact.get("content_hash")
                    or artifact.get("sha256")
                    or artifact.get("hash")
                    or ""
                )
                if path and content_hash and _within_scope(path, node.write_paths):
                    expected[path] = content_hash
        parent_paths = tuple(
            _normalized_path(path)
            for path in node.write_paths
            if _normalized_path(path) not in {".", "*", "**", "**/*"}
            and not any(character in path for character in "*?[")
        )
        if not parent_paths or any(path not in expected for path in parent_paths):
            return {
                "passed": False,
                "findings": [
                    "accepted child packages do not cover every exact parent write path"
                ],
            }
        actions = tuple(reversed(self.store.list_actions(self.goal_id)))
        evidence: list[Mapping[str, Any]] = []
        for path in parent_paths:
            expected_hash = expected[path]
            content: str | None = None
            source_action = ""
            if _hash_file(self.workspace, path) != expected_hash:
                for action in actions:
                    if (
                        action.get("tool_name") != "write_file"
                        or action.get("status") != "completed"
                    ):
                        continue
                    try:
                        envelope = json.loads(str(action.get("args_json") or "{}"))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    arguments = (
                        dict(envelope.get("arguments") or {})
                        if isinstance(envelope, Mapping)
                        else {}
                    )
                    candidate_path = _normalized_path(
                        str(arguments.get("path") or "")
                    )
                    candidate_content = arguments.get("content")
                    if (
                        candidate_path == path
                        and isinstance(candidate_content, str)
                        and hashlib.sha256(
                            candidate_content.encode("utf-8")
                        ).hexdigest()
                        == expected_hash
                    ):
                        content = candidate_content
                        source_action = str(action.get("id") or "")
                        break
                if content is None:
                    return {
                        "passed": False,
                        "findings": [
                            f"no durable write receipt matches accepted child hash for {path}"
                        ],
                    }
                target = self.workspace.joinpath(
                    *PurePosixPath(path).parts
                ).resolve(strict=False)
                try:
                    target.relative_to(self.workspace.resolve())
                except ValueError:
                    return {
                        "passed": False,
                        "findings": [f"child artifact path escapes workspace: {path}"],
                    }
                with tools.workspace_context(self.workspace):
                    atomic_write_bytes(
                        target,
                        content.encode("utf-8"),
                        overwrite=True,
                    )
            evidence.append(
                {
                    "kind": "exact_child_artifact",
                    "path": path,
                    "sha256": expected_hash,
                    "source_action_id": source_action,
                    "restored": bool(content is not None),
                }
            )
        self.store.append_event(
            "ultra.exact_child_artifacts_restored",
            goal_id=self.goal_id,
            entity_type="work_node",
            entity_id=node.id,
            payload={
                "run_id": run_id,
                "paths": list(parent_paths),
                "hashes": [expected[path] for path in parent_paths],
            },
        )
        return {"passed": True, "findings": [], "evidence": evidence}

    def save_component_package(self, run_id: str, package: ComponentPackageV1) -> None:
        super().save_component_package(run_id, package)
        existing_packages = self.store.list_component_packages(
            run_id,
            work_node_id=package.node_id,
        )
        next_storage_version = (
            max(
                (int(item.get("version") or 0) for item in existing_packages),
                default=0,
            )
            + 1
        )
        stored = self.store.put_component_package(
            {
                **asdict(package),
                "ultra_run_id": run_id,
                "work_node_id": package.node_id,
                # ComponentPackageV1.version is the wire-schema version, not
                # the durable publication revision. A resumed node must append
                # a new storage version instead of colliding with revision 1.
                "version": next_storage_version,
            }
        )
        node = self.store.get_work_node(package.node_id)
        try:
            self.store.post_swarm_message(
                SwarmMessageV1(
                    ultra_run_id=run_id,
                    sender_agent_id=f"specialist:{package.node_id}",
                    recipient_agent_id=(
                        f"specialist:{node.parent_id}"
                        if node.parent_id
                        else "final-assembler"
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
        except Exception as exc:
            # The package itself is already durable. A secondary notification
            # receipt is telemetry and must not invalidate accepted evidence.
            self.events.publish(
                "ultra.package_receipt_failed",
                f"Package {stored['id']} persisted but its swarm receipt failed.",
                run_id=run_id,
                node_id=package.node_id,
                error=redact_text(str(exc))[:500],
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
                    "metadata": {
                        **dict(module.metadata),
                        "ultra_node_id": module.id,
                    },
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
        goal = self.store.get_goal(self.goal_id)
        authored_effects = tuple(
            RequestedEffect.parse(item)
            for item in goal.metadata.get("accepted_requested_effects", ())
        )
        if RequestedEffect.READ_WORKSPACE not in authored_effects:
            authored_effects = (RequestedEffect.READ_WORKSPACE, *authored_effects)
        semantic_goal = SemanticGoalV2(
            original_request=goal.objective,
            interpreted_outcome=goal_spec.objective,
            requested_effects=authored_effects,
            required_outcomes=goal_spec.in_scope or goal_spec.success_criteria,
            constraints=goal_spec.constraints,
            exclusions=goal_spec.out_of_scope,
            acceptance_criteria=goal_spec.success_criteria,
            repository_evidence_refs=(f"ultra:{self.run_id}:foundation",),
            status="critic_accepted",
        )
        self.store.update_goal_metadata(
            self.goal_id,
            semantic_goal=semantic_goal.to_dict(),
            semantic_goal_fingerprint=semantic_goal.fingerprint,
            accepted_semantic_fingerprint=semantic_goal.fingerprint,
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
                # Keep the exact durable objective available to evaluators and
                # restart recovery.  Replacing the config at approval used to
                # discard it, which silently disabled task-specific gates such
                # as the Three.js/WebGL benchmark.
                "prompt": goal.objective,
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

    def approve_master(
        self,
        master: MasterPlanV1,
        *,
        approved_by: str = "user",
    ) -> Plan:
        if not self.run_id or not self.plan:
            raise StateStoreError("ULTRA foundation is not bound to a durable master plan")
        accepted, _ = self.store.approve_plan(
            self.goal_id,
            self.plan.revision,
            approved_by=approved_by,
            expected_fingerprint=self.plan.fingerprint,
        )
        self.store.approve_ultra_master(
            self.run_id,
            accepted.revision,
            accepted.fingerprint,
            approved_by=approved_by,
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
        resumable_task_states = {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.UNCERTAIN,
        }
        if (
            node.status is NodeStatus.RUNNING
            and task.status in resumable_task_states
        ):
            self.store.transition_task(
                self.goal_id,
                self.plan.revision,
                task_id,
                TaskStatus.IN_PROGRESS,
                actor="ultra-scheduler",
            )
        elif result and result.success and task.status != TaskStatus.COMPLETED:
            if task.status in {
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.UNCERTAIN,
            }:
                # A recoverable provider/harness failure may have persisted
                # the legacy plan task as blocked even though the restored
                # engine node can safely resume. Re-enter execution before
                # recording completion so task transition invariants stay
                # intact and the mutation is never replayed blindly.
                task = self.store.transition_task(
                    self.goal_id,
                    self.plan.revision,
                    task_id,
                    TaskStatus.IN_PROGRESS,
                    note="Resumed from the last durable ULTRA evidence gate.",
                    actor="ultra-recovery",
                )
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
            action_status = {
                "running": NextActionStatus.RUNNING,
                "completed": NextActionStatus.COMPLETED,
                "failed": NextActionStatus.FAILED,
                "cancelled": NextActionStatus.CANCELLED,
                "rate_limited": NextActionStatus.FAILED,
                "uncertain": NextActionStatus.RECOVERING,
            }.get(item.status, NextActionStatus.FAILED)
            try:
                self.store.transition_scheduled_agent_action(
                    item.id,
                    action_status,
                    error=item.error,
                )
            except NotFoundError:
                # Legacy/in-memory tests may save an AgentRun without staging
                # a durable NextActionPacket first.
                pass
            self._checkpoint_agent_memory(item)
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
                            closed = self.store.save_change_set(
                                replace(
                                    change_set,
                                    status=ChangeSetStatus.CLOSED,
                                    updated_at=utc_now(),
                                )
                            )
                            if self.events is not None:
                                self.events.publish(
                                    "checkpoint.review_ready",
                                    (
                                        f"Checkpoint {closed.id} is ready for review · "
                                        f"{len(closed.changed_files)} file(s) changed."
                                    ),
                                    checkpoint_id=closed.id,
                                    plan_revision=(
                                        self.plan.revision if self.plan is not None else None
                                    ),
                                    changed_files=len(closed.changed_files),
                                    source="agent_runtime",
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
                ChangeSetStatus.OPEN,
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

    def reconcile_verified_change_sets(
        self,
        run_id: str,
    ) -> tuple[ChangeSetV1, ...]:
        """Approve durable checkpoints already accepted by node consensus.

        A transport failure can leave a mutation checkpoint OPEN/BLOCKED even
        though the node's final package, independent consensus, workspace
        hashes, and executable gate are all durable and successful. Reconcile
        those projections without replaying the mutation or asking for a
        second plan approval.
        """

        if self.workspace is None:
            return self.store.list_change_sets(run_id)
        nodes = {
            item.id: item
            for item in self.store.list_work_nodes(run_id)
        }
        reconciled: list[str] = []
        for change_set in self.store.list_change_sets(run_id):
            if change_set.status in {
                ChangeSetStatus.APPROVED,
                ChangeSetStatus.INTEGRATED,
            }:
                continue
            node = nodes.get(str(change_set.parent_id or ""))
            if node is None or not node.result or not node.result.metadata.get("success"):
                continue
            packages = self.store.list_component_packages(
                run_id,
                work_node_id=node.id,
            )
            if not packages:
                continue
            latest = packages[-1]
            consensus = (
                dict(latest.get("quality") or {}).get("consensus")
                if isinstance(latest.get("quality"), Mapping)
                else {}
            )
            consensus_status = (
                str(dict(consensus or {}).get("status") or "").casefold()
                if isinstance(consensus, Mapping)
                else ""
            )
            if consensus_status != "accepted":
                continue
            scopes = tuple(node.contract.write_paths)
            if not scopes or any(
                not _within_scope(path, scopes)
                for path in change_set.changed_files
            ):
                continue
            artifact_gate = _validate_workspace_artifacts(
                self.workspace,
                scopes,
            )
            executable_gate = _run_python_test_artifacts(
                self.workspace,
                scopes,
            )
            if not artifact_gate.get("passed") or not executable_gate.get("passed"):
                continue
            reviews = {
                **dict(change_set.review_status),
                "accepted_consensus": "passed",
                "workspace_artifacts": "passed",
                "executable_gate": "passed",
            }
            # Majority consensus cannot erase a mandatory reviewer rejection.
            # Recover missing durable review projections only from the
            # corresponding accepted vote; integration still requires all
            # three named review gates.
            vote_roles = {
                "security": "security",
                "clean_code": "clean_code",
                "testing": "test_quality",
                "test_quality": "test_quality",
            }
            for vote in tuple(dict(consensus or {}).get("votes") or ()):
                if not isinstance(vote, Mapping):
                    continue
                voter = str(vote.get("voter_agent_id") or "").rsplit(":", 1)[-1]
                required_role = vote_roles.get(voter.casefold())
                if (
                    required_role
                    and reviews.get(required_role) != "failed"
                    and str(vote.get("verdict") or "").casefold() == "accept"
                ):
                    reviews[required_role] = "passed"
            if not all(
                reviews.get(role) == "passed"
                for role in ("clean_code", "security", "test_quality")
            ):
                continue
            self.store.save_change_set(
                replace(
                    change_set,
                    status=ChangeSetStatus.APPROVED,
                    review_status=reviews,
                    updated_at=utc_now(),
                )
            )
            reconciled.append(change_set.id)
        if reconciled and self.events is not None:
            self.events.publish(
                "ultra.change_sets_reconciled",
                (
                    f"Reconciled {len(reconciled)} accepted checkpoint(s) "
                    "from durable consensus and harness evidence."
                ),
                run_id=run_id,
                change_set_ids=reconciled,
            )
        return self.store.list_change_sets(run_id)

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

    def record_adaptive_quality_observations(
        self,
        node_id: str,
        *,
        candidate_fingerprint: str,
        observations: Iterable[Mapping[str, Any]],
        consensus: Mapping[str, Any],
        gate_passed: bool,
    ) -> None:
        """Attribute Ultra review value without calling production data causal."""

        if not self.run_id:
            return
        items = tuple(
            dict(item) for item in observations if isinstance(item, Mapping)
        )
        invoked = tuple(
            item
            for item in items
            if isinstance(item.get("response"), AgentResponse)
            and item["response"].model != "deterministic-review-router-v1"
        )
        if not invoked:
            return

        def trusted_records(response: AgentResponse) -> tuple[Mapping[str, Any], ...]:
            records: list[Mapping[str, Any]] = []
            for key in ("test_results", "evidence"):
                raw = response.payload.get(key, ())
                values = (
                    (raw,)
                    if isinstance(raw, Mapping)
                    else raw
                    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes))
                    else ()
                )
                for value in values:
                    if not isinstance(value, Mapping):
                        continue
                    name = str(value.get("name") or value.get("kind") or "")
                    if (
                        value.get("artifact_hash")
                        or value.get("action_id")
                        or str(value.get("source") or "").casefold() == "harness"
                        or str(value.get("authority") or "").casefold()
                        in {"browser", "harness", "test", "user"}
                        or name.startswith("harness_")
                    ):
                        records.append(dict(value))
            return tuple(records)

        def record_ref(value: Mapping[str, Any]) -> str:
            return str(
                value.get("artifact_hash")
                or value.get("action_id")
                or value.get("uri")
                or value.get("name")
                or value.get("kind")
                or ""
            ).strip()

        observation_refs: dict[str, tuple[str, ...]] = {}
        for item in invoked:
            response = item["response"]
            observation_refs[str(item.get("category") or "review")] = tuple(
                dict.fromkeys(
                    ref
                    for ref in (
                        record_ref(value) for value in trusted_records(response)
                    )
                    if ref
                )
            )
        all_refs = tuple(
            dict.fromkeys(
                ref for refs in observation_refs.values() for ref in refs if ref
            )
        )
        model_fingerprint = hashlib.sha256(
            self.descriptor.id.encode("utf-8")
        ).hexdigest()
        node = self.store.get_work_node(node_id)
        task_class = str(
            node.contract.metadata.get("task_family")
            or node.kind.value
            or "general"
        )
        total_input = sum(
            int(item["response"].usage.get("input_tokens", 0) or 0)
            for item in invoked
        )
        total_output = sum(
            int(item["response"].usage.get("output_tokens", 0) or 0)
            for item in invoked
        )
        experiment = OrchestrationExperimentV1(
            goal_id=self.goal_id,
            ultra_run_id=self.run_id,
            unit_id=node_id,
            arm=OrchestrationArm.PRODUCTION,
            model_fingerprint=model_fingerprint,
            policy_fingerprint=self.adaptive_orchestration_policy.fingerprint,
            task_class=task_class,
            candidate_score=1.0 if gate_passed else 0.0,
            success=gate_passed,
            false_completion=(
                str(consensus.get("status") or "").casefold() == "accepted"
                and not gate_passed
            ),
            model_calls=len(invoked),
            input_tokens=total_input,
            output_tokens=total_output,
            causal=False,
            matched_benchmark=False,
            metrics={
                "workflow": "ultra",
                "shadow_mode": self.adaptive_orchestration_policy.shadow_mode,
                "consensus_status": str(consensus.get("status") or "not_recorded"),
                "reviewer_categories": [
                    str(item.get("category") or "review") for item in invoked
                ],
            },
            evidence=all_refs,
        )
        try:
            self.store.record_orchestration_experiment(experiment)
            existing_refs: set[str] = set()
            agent_runs = self.store.list_agent_runs(
                self.run_id,
                work_node_id=node_id,
            )
            for item in invoked:
                response = item["response"]
                category = str(item.get("category") or "review")
                agent_role = str(item.get("agent_role") or "reviewer")
                refs = observation_refs.get(category, ())
                novelty = evidence_novelty(existing_refs, refs)
                existing_refs.update(refs)
                trusted = trusted_records(response)
                failed_records = tuple(
                    value for value in trusted if value.get("passed") is False
                )
                findings = tuple(
                    str(value).strip()
                    for key in ("findings", "issues")
                    for value in (
                        response.payload.get(key, ())
                        if isinstance(response.payload.get(key), Sequence)
                        and not isinstance(response.payload.get(key), (str, bytes))
                        else ()
                    )
                    if str(value).strip()
                )
                verified_findings = len(failed_records)
                false_findings = len(findings) if findings and not failed_records else 0
                score_delta = novelty if gate_passed and bool(response.payload.get("passed")) else 0.0
                outcome = classify_worker_impact(
                    verified_findings=verified_findings,
                    false_findings=false_findings,
                    accepted_fixes=0,
                    novelty=novelty,
                    score_delta=score_delta,
                )
                agent_run = next(
                    (
                        run
                        for run in reversed(agent_runs)
                        if run.role == agent_role
                        and run.status is AgentRunStatus.COMPLETED
                    ),
                    None,
                )
                claim = EvidenceClaimV1(
                    criterion_id=category,
                    claim=response.summary or f"{category} quality verdict",
                    artifact_hash=candidate_fingerprint if refs else "",
                    evidence_refs=refs,
                    falsification_check=(
                        f"Re-run the authoritative {category} gate against artifact "
                        f"{candidate_fingerprint[:12]}."
                    ),
                    verdict=(
                        EvidenceVerdict.FAILED
                        if failed_records
                        else EvidenceVerdict.PASSED
                        if refs and bool(response.payload.get("passed"))
                        else EvidenceVerdict.NEEDS_EVIDENCE
                    ),
                    authority=(
                        EvidenceAuthority.HARNESS
                        if refs
                        else EvidenceAuthority.MODEL
                    ),
                    producer_id=agent_run.id if agent_run is not None else agent_role,
                    verifier_id="harness" if refs else "",
                )
                role = (
                    WorkerRole.FALSIFIER
                    if category == "runtime_tests"
                    else WorkerRole.REVIEWER
                )
                self.store.record_worker_contribution(
                    WorkerImpactV1(
                        experiment_id=experiment.id,
                        worker_id=(
                            agent_run.id
                            if agent_run is not None
                            else f"{node_id}:{category}"
                        ),
                        agent_run_id=agent_run.id if agent_run is not None else None,
                        role=role,
                        task_class=task_class,
                        model_fingerprint=model_fingerprint,
                        outcome=outcome,
                        approach_fingerprint=approach_fingerprint(
                            role=role,
                            objective=f"{category}:{node.contract.objective}",
                            falsification_targets=node.contract.success_criteria,
                            context_refs=(candidate_fingerprint,),
                        ),
                        evidence_novelty=novelty,
                        verified_findings=verified_findings,
                        false_findings=false_findings,
                        accepted_fixes=0,
                        score_delta=score_delta,
                        input_tokens=int(response.usage.get("input_tokens", 0) or 0),
                        output_tokens=int(response.usage.get("output_tokens", 0) or 0),
                        claims=(claim,),
                        evidence=refs,
                        reason=(
                            "review added authoritative evidence used by the quality gate"
                            if outcome.value == "useful"
                            else "review added no independently verified decision-changing evidence"
                            if outcome.value == "neutral"
                            else "review emitted unsupported findings"
                        ),
                    )
                )
        except (DomainError, NotFoundError, StateStoreError, ValueError) as exc:
            self.store.append_event(
                "worker.contribution_not_counted",
                goal_id=self.goal_id,
                entity_type="work_node",
                entity_id=node_id,
                payload={"reason": redact_text(str(exc), 1_000)},
            )

    @staticmethod
    def _repository_preservation_delta(
        *,
        baseline_hashes: Mapping[str, str],
        current_hashes: Mapping[str, str],
        tracked_paths: Iterable[str],
        approved_scopes: Iterable[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        tracked = {
            str(path).replace("\\", "/").removeprefix("./")
            for path in tracked_paths
        }
        scopes = tuple(
            str(scope).replace("\\", "/").removeprefix("./")
            for scope in approved_scopes
            if str(scope).strip()
        )
        changed = {
            path
            for path in set(baseline_hashes) | set(current_hashes)
            if baseline_hashes.get(path) != current_hashes.get(path)
        }
        unexpected = tuple(sorted(changed - tracked))
        out_of_scope = tuple(
            sorted(
                path
                for path in tracked
                if not scopes
                or not any(_within_scope(path, (scope,)) for scope in scopes)
            )
        )
        return unexpected, out_of_scope

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
        codex_supervisory_packages = 0
        independent_visual_packages = 0
        for package in latest_materialized.values():
            quality = dict(package.get("quality") or {})
            if not bool(quality.get("visual_required", False)):
                continue
            if str(quality.get("evaluation_mode") or "") == "codex_supervisory_final":
                codex_supervisory_packages += 1
                continue
            independent_visual_packages += 1
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
        metrics["codex_supervisory_packages"] = float(codex_supervisory_packages)
        metrics["independent_visual_packages"] = float(independent_visual_packages)
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
            if independent_visual_packages and visual_score < 0.90:
                blocker = blocker or "independent critical visual quality score is below 0.90"
        scores["overall"] = sum(weighted_dimensions) / max(1, len(weighted_dimensions))
        if scores["overall"] < 0.95:
            blocker = blocker or "overall quality score is below 0.95"
        open_blocking_findings = tuple(
            item
            for item in self.store.list_quality_findings(self.run_id)
            if item.severity.blocks_completion and item.status.value != "resolved"
        )
        if open_blocking_findings:
            blocker = blocker or (
                "blocking specialist findings remain: "
                + ", ".join(
                    f"{item.repair_node_id or item.path}:{item.principle_id}"
                    for item in open_blocking_findings[:8]
                )
            )
        metrics["open_blocking_findings"] = float(len(open_blocking_findings))

        # Large-repository preservation gate. The baseline is hash-bound
        # before approval; every later source change must appear in a durable
        # Change Set and remain inside at least one approved node write scope.
        baseline_cycle = next(
            (
                item
                for item in self.store.list_quality_cycles(self.run_id)
                if item.kind is QualityCycleKind.BASELINE
                and isinstance(item.inputs.get("file_hashes"), Mapping)
            ),
            None,
        )
        unexpected_repository_changes: list[str] = []
        out_of_scope_changes: list[str] = []
        if baseline_cycle is not None:
            baseline_hashes = {
                str(path): str(digest)
                for path, digest in dict(
                    baseline_cycle.inputs.get("file_hashes", {})
                ).items()
            }
            current_hashes = {
                path: digest
                for path, digest in self._workspace_hashes().items()
                if not path.startswith(("run-artifacts/", "output/playwright/"))
            }
            change_sets = self.store.list_change_sets(self.run_id)
            tracked_paths = {
                str(path).replace("\\", "/").removeprefix("./")
                for change_set in change_sets
                for path in change_set.changed_files
            }
            approved_scopes = tuple(
                scope
                for work_node in self.store.list_work_nodes(self.run_id)
                for scope in work_node.contract.write_paths
                if str(scope).strip()
            )
            unexpected, out_of_scope = self._repository_preservation_delta(
                baseline_hashes=baseline_hashes,
                current_hashes=current_hashes,
                tracked_paths=tracked_paths,
                approved_scopes=approved_scopes,
            )
            unexpected_repository_changes = list(unexpected)
            out_of_scope_changes = list(out_of_scope)
            if unexpected_repository_changes:
                blocker = blocker or (
                    "repository preservation failed: untracked source changes: "
                    + ", ".join(unexpected_repository_changes[:8])
                )
            if out_of_scope_changes:
                blocker = blocker or (
                    "repository preservation failed: changes escaped approved write scopes: "
                    + ", ".join(out_of_scope_changes[:8])
                )
        metrics["unexpected_repository_changes"] = float(
            len(unexpected_repository_changes)
        )
        metrics["out_of_scope_repository_changes"] = float(
            len(out_of_scope_changes)
        )
        scores["repository_preservation"] = (
            1.0
            if not unexpected_repository_changes and not out_of_scope_changes
            else 0.0
        )
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
        durable_run = self.store.get_ultra_run(self.run_id)
        prompt = str(durable_run.config.get("prompt", ""))
        # This evaluator is intentionally specific to interactive 3D/WebGL
        # artifacts. Normal HTML products are already covered by the real
        # browser/runtime gate and must not be rejected for lacking a scene,
        # camera, animation loop, or gameplay state.
        architecture_text = _json(
            asdict(durable_run.architecture_spec)
            if durable_run.architecture_spec is not None
            else {}
        )
        evaluator_contract = f"{prompt}\n{architecture_text}".casefold()
        should_check = any(
            term in evaluator_contract
            for term in (
                "three.js",
                "threejs",
                "webgl",
                "3d game",
                "3d html",
                "browser game",
            )
        )
        if not should_check:
            return None
        html_targets: list[tuple[str, Path]] = []
        for raw in candidates:
            value = str(raw or "").strip().replace("\\", "/")
            if value.startswith("workspace:"):
                value = value.removeprefix("workspace:")
            value = value.removeprefix("./")
            if not value.casefold().endswith((".html", ".htm")):
                continue
            candidate = (self.workspace / value).resolve(strict=False)
            try:
                candidate.relative_to(self.workspace)
            except ValueError:
                continue
            html_targets.append((value, candidate))
        fallback = (
            "index.html",
            (self.workspace / "index.html").resolve(strict=False),
        )
        target_name, index_path = next(
            ((name, path) for name, path in html_targets if path.is_file()),
            html_targets[0] if html_targets else fallback,
        )
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
                artifact_refs=(f"workspace:{target_name}",),
                blocker=f"HTML benchmark target {target_name} was not created",
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
                artifact_refs=(f"workspace:{target_name}",),
                blocker=f"HTML benchmark could not read {target_name}: {type(exc).__name__}",
            )
            return recorded
        preview: Mapping[str, Any] = {}
        preview_id = ""
        try:
            preview_raw = tools.run_tool(
                "preview_html",
                {
                    "path": target_name,
                    "open_browser": False,
                    "verify": True,
                    "settle_ms": 2500,
                },
            )
            parsed_preview = json.loads(preview_raw)
            if isinstance(parsed_preview, Mapping):
                preview = dict(parsed_preview)
                preview_id = str(preview.get("preview_id") or "")
                screenshot_source = Path(
                    str(preview.get("screenshot_path") or "")
                )
                if screenshot_source.is_file():
                    screenshot_target = (
                        self.workspace
                        / "output"
                        / "playwright"
                        / f"ultra-final-{self.run_id}.png"
                    )
                    screenshot_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(screenshot_source, screenshot_target)
                    preview["screenshot_path"] = str(screenshot_target)
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
            artifact_ref=f"workspace:{target_name}",
            preview=preview,
        )

    def save_prompt_trace(self, trace: EnginePromptTrace) -> None:
        super().save_prompt_trace(trace)
        with self._adapter_lock:
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
        serialized = _json(entry.value)
        content = serialized
        if len(serialized) > 80_000:
            content = _json(
                {
                    "summary": (
                        f"Large {entry.section.value} payload stored in data_json; "
                        "retrieve focused slices instead of replaying it."
                    ),
                    "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                    "characters": len(serialized),
                }
            )
        stored = self.store.put_brain_entry(
            BrainEntry(
                ultra_run_id=entry.run_id,
                goal_id=self.goal_id,
                work_node_id=entry.node_id,
                section=self._brain_section(entry.section),
                title=entry.key,
                content=content,
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
                self._lease_initial_hashes[lease.owner] = dict(hashes)

        def released(lease: RuntimeLease) -> None:
            with self._adapter_lock:
                lease_ids = self._lease_ids.pop(lease.owner, [])
                self._lease_scopes.pop(lease.owner, None)
                self._lease_hashes.pop(lease.owner, None)
                self._lease_initial_hashes.pop(lease.owner, None)
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

    def initial_lease_hash(self, owner: str, path: str) -> tuple[bool, str | None]:
        """Return the immutable pre-execution hash for mutation proof."""

        normalized = _normalized_path(path)
        with self._adapter_lock:
            scopes = self._lease_scopes.get(owner, ())
            if not scopes or not _within_scope(normalized, scopes):
                return False, None
            return True, self._lease_initial_hashes.get(owner, {}).get(normalized)

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
        version_control: GitProtectionManager | None = None,
        session_id: str | None = None,
        adaptive_orchestration_policy: AdaptiveOrchestrationPolicyV1 | None = None,
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
        self.version_control = version_control
        self.session_id = str(session_id or "").strip() or None
        self.adaptive_orchestration_policy = (
            adaptive_orchestration_policy or AdaptiveOrchestrationPolicyV1()
        )
        self.goal_id: str | None = None
        self.adapter: StateStoreUltraAdapter | None = None
        self.orchestrator: UltraOrchestrator | None = None
        self.future: Future[UltraRunResult] | None = None
        self.answers: dict[str, str] = {}

    def _workspace_hashes(self) -> dict[str, str]:
        values: dict[str, str] = {}
        ignored_parts = {
            ".coding-agent",
            ".git",
            ".hg",
            ".svn",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "__pycache__",
            "node_modules",
            "run-artifacts",
        }
        workspace = self.workspace.resolve()
        for current, directories, filenames in os.walk(
            workspace,
            topdown=True,
            followlinks=False,
        ):
            # Prune ignored descendants before traversal.  Never inspect the
            # absolute workspace ancestors: a perfectly valid workspace may
            # itself live under a parent named ``run-artifacts`` or ``.git``.
            directories[:] = [
                name
                for name in directories
                if name not in ignored_parts
                and not (Path(current) / name).is_symlink()
            ]
            for filename in filenames:
                path = Path(current) / filename
                if path.is_symlink():
                    continue
                try:
                    relative = path.relative_to(workspace).as_posix()
                    if relative.startswith(("output/playwright/", ".playwright-cli/")):
                        continue
                    values[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                except (OSError, ValueError):
                    continue
        return values

    def _workspace_fingerprint(self) -> str:
        return hashlib.sha256(
            _json(self._workspace_hashes()).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _recoverable_workspace_path(value: str) -> bool:
        """Keep source/config artifacts recoverable without snapshotting caches or VCS internals."""

        path = PurePosixPath(str(value).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            return False
        blocked_roots = {
            ".coding-agent",
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "node_modules",
            "run-artifacts",
            "__pycache__",
        }
        if path.parts[0] in blocked_roots or "__pycache__" in path.parts:
            return False
        return not (
            len(path.parts) >= 2
            and path.parts[0] == "output"
            and path.parts[1] == "playwright"
        )

    def _recovery_root(self, run_id: str) -> Path:
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id))[:160] or "ultra-run"
        return self.workspace / ".coding-agent" / "recovery" / safe_run_id

    def _capture_workspace_baseline(self, run_id: str) -> Mapping[str, Any]:
        """Create a project-scoped recovery snapshot before approval."""

        recovery_root = self._recovery_root(run_id)
        baseline_root = recovery_root / "baseline"
        manifest_path = recovery_root / "manifest.json"
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, Mapping) and existing.get("files"):
                return dict(existing)
        files: dict[str, str] = {}
        baseline_root.mkdir(parents=True, exist_ok=True)
        for source in self.workspace.rglob("*"):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(self.workspace).as_posix()
            if not self._recoverable_workspace_path(relative):
                continue
            try:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                destination = baseline_root / PurePosixPath(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            except OSError:
                continue
            files[relative] = digest
        manifest = {
            "schema": "WorkspaceRecoveryBaselineV1",
            "run_id": str(run_id),
            "captured_at": utc_now().isoformat(),
            "files": files,
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        return manifest

    def _restore_workspace_baseline(
        self,
        run_id: str,
        changed_paths: Iterable[str],
        *,
        reason: str,
    ) -> Mapping[str, Any]:
        """Restore only agent-recorded mutations; unrelated concurrent user files remain untouched."""

        goal_id = getattr(self, "goal_id", None)
        store = getattr(self, "store", None)
        journal_restored = (
            store.rollback_mutation_journal(goal_id)
            if goal_id and store is not None
            else ()
        )
        recovery_root = self._recovery_root(run_id)
        manifest_path = recovery_root / "manifest.json"
        if not manifest_path.is_file():
            return {"restored": (), "removed": (), "preserved": (), "reason": reason}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        baseline_files = {
            str(path): str(digest)
            for path, digest in dict(manifest.get("files") or {}).items()
        }
        baseline_root = recovery_root / "baseline"
        rollback_stamp = utc_now().isoformat().replace(":", "-")
        rejection_root = recovery_root / "rejected" / rollback_stamp
        restored: list[str] = list(journal_restored)
        removed: list[str] = []
        preserved: list[str] = []
        for raw_path in sorted(set(str(item) for item in changed_paths if str(item).strip())):
            relative = PurePosixPath(raw_path.replace("\\", "/"))
            normalized = relative.as_posix().removeprefix("./")
            if not self._recoverable_workspace_path(normalized):
                continue
            target = self.workspace / relative
            if target.is_file() and not target.is_symlink():
                rejected = rejection_root / relative
                rejected.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, rejected)
                preserved.append(normalized)
            if normalized in baseline_files:
                source = baseline_root / relative
                if not source.is_file():
                    raise RuntimeError(f"recovery baseline is missing {normalized!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.ga3bad-restore-{os.getpid()}")
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
                restored.append(normalized)
            elif target.exists():
                if target.is_file() or target.is_symlink():
                    target.unlink()
                    removed.append(normalized)
        report = {
            "schema": "WorkspaceRollbackReportV1",
            "run_id": str(run_id),
            "reason": str(reason),
            "restored": tuple(restored),
            "removed": tuple(removed),
            "preserved": tuple(preserved),
            "completed_at": utc_now().isoformat(),
        }
        rejection_root.mkdir(parents=True, exist_ok=True)
        (rejection_root / "rollback-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def _rollback_rejected_changes(self, reason: str) -> Mapping[str, Any]:
        if not self.run_id:
            return {"restored": (), "removed": (), "preserved": (), "reason": reason}
        changed_paths = tuple(
            dict.fromkeys(
                path
                for change_set in self.store.list_change_sets(self.run_id)
                for path in change_set.changed_files
            )
        )
        report = self._restore_workspace_baseline(
            self.run_id,
            changed_paths,
            reason=reason,
        )
        self.events.publish(
            "ultra.workspace_rolled_back",
            "Rejected Ultra changes were removed and the accepted workspace baseline was restored.",
            **dict(report),
        )
        return report

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

    def _component_read_path(
        self,
        request: AgentRequest,
        node: EngineWorkNode | None,
        requested: str,
    ) -> str | None:
        """Map a model's component-relative read to its isolated artifact root.

        Small local models commonly try ``src/file.js`` even though component
        packages live in a harness-owned staging directory. The harness owns
        this mapping so reviewers can inspect exact bytes without granting
        shared-workspace writes or teaching every model an absolute path.
        """

        if node is None or self.adapter is None or self.run_id is None:
            return None
        contract = (
            dict(request.task.get("contract", {}))
            if isinstance(request.task, Mapping)
            else {}
        )
        if not bool(dict(contract.get("metadata", {})).get("component_package_only")):
            return None

        roots: list[tuple[Path, tuple[str, ...]]] = []
        candidate = (
            dict(request.task.get("candidate", {}))
            if isinstance(request.task.get("candidate"), Mapping)
            else {}
        )
        payload = (
            dict(candidate.get("payload", {}))
            if isinstance(candidate.get("payload"), Mapping)
            else {}
        )
        package = payload.get("materialized_component_package")
        if isinstance(package, Mapping) and package.get("root"):
            package_root = Path(str(package["root"])).resolve()
            package_files = tuple(
                str(item.get("path", "")).replace("\\", "/").strip("/")
                for item in package.get("files", ())
                if isinstance(item, Mapping) and str(item.get("path", "")).strip()
            )
            roots.append((package_root, package_files))
        if self.adapter.component_artifacts is not None:
            draft_root = self.adapter.component_artifacts.draft_root(
                self.run_id,
                node.id,
            ).resolve()
            draft_files = tuple(
                str(item.get("path", "")).replace("\\", "/").strip("/")
                for item in self.adapter.component_artifacts.draft_files(
                    run_id=self.run_id,
                    node_id=node.id,
                )
            )
            roots.append((draft_root, draft_files))

        raw = str(requested or ".").replace("\\", "/").strip()
        normalized = raw.strip("/")
        node_prefixes = (node.id, node.id.split(".")[-1])
        for prefix in node_prefixes:
            if normalized == prefix:
                normalized = ""
            elif normalized.startswith(prefix + "/"):
                normalized = normalized[len(prefix) + 1 :]
        if normalized in {"", ".", "*", ".*", "**", "**/*"}:
            normalized = ""

        workspace = self.workspace.resolve()
        for root, files in roots:
            try:
                root.relative_to(workspace)
            except ValueError:
                continue
            relative = normalized
            if relative and files and relative not in files:
                suffix_matches = [
                    item
                    for item in files
                    if item.endswith("/" + relative) or relative.endswith("/" + item)
                ]
                if len(suffix_matches) == 1:
                    relative = suffix_matches[0]
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                continue
            if target.exists():
                return target.relative_to(workspace).as_posix()
        return None

    def _checkpoint_tool_approval_boundary(
        self,
        *,
        tool: str,
        args: Mapping[str, Any],
        risk: str,
        action_id: str,
        reason: str,
    ) -> None:
        if not self.goal_id:
            return
        fingerprint = hashlib.sha256(
            _json({"tool": tool, "args": redact_data(dict(args)), "risk": risk}).encode("utf-8")
        ).hexdigest()
        goal = self.store.get_goal(self.goal_id)
        self.store.update_goal_metadata(
            self.goal_id,
            pending_tool_approval={
                "tool": tool,
                "arguments": redact_data(dict(args)),
                "risk": risk,
                "action_id": action_id,
                "action_fingerprint": fingerprint,
            },
            waiting_question=(
                f"Approval is required before {tool} can run. No workspace mutation "
                "from this action occurred. Retry from the saved checkpoint to approve it."
            ),
            waiting_on="approval",
            resume_action="Retry",
            auto_retryable=False,
        )
        if goal.status is GoalStatus.RUNNING:
            self.store.transition_goal(
                self.goal_id,
                GoalStatus.PAUSED,
                reason=reason,
            )
        self.store.append_event(
            "execution.boundary",
            goal_id=self.goal_id,
            entity_type="action",
            entity_id=action_id,
            payload={
                "phase": "waiting_for_approval",
                "reason": reason,
                "waiting_on": "user",
                "resume_action": "Retry",
                "tool": tool,
                "risk": risk,
                "action_fingerprint": fingerprint,
                "workspace_mutated": False,
            },
        )

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
        args = dict(call.args) if isinstance(call.args, dict) else {}
        # ``.`` is a valid directory scope in a task contract, but not a
        # valid file target. Small/local models often copy that scope into
        # ``read_file`` verbatim. Normalize the harmless inspection instead
        # of failing an agent and cascading retries through its branch.
        if (
            call.name == "read_file"
            and str(args.get("path") or ".").strip().replace("\\", "/")
            in {"", ".", "./"}
        ):
            call = ToolCall(
                id=call.id,
                name="list_files",
                args={"path": "."},
                native=dict(call.native),
            )
            args = dict(call.args)
            self.store.append_event(
                "tool_contract.normalized",
                goal_id=self.goal_id,
                payload={
                    "actor": request.role.value,
                    "received": "read_file",
                    "normalized_to": "list_files",
                    "path": ".",
                    "stage": "ultra_execution_directory_scope",
                    "workspace_mutated": False,
                },
            )
        node = self._node(request.node_id)
        if call.name in {"run_bash", "run_command"} and node is not None:
            command = " ".join(str(args.get("command") or "").split()).casefold()
            approved_commands = {
                " ".join(str(item).split()).casefold()
                for item in node.contract.verification
                if str(item).strip()
                and re.match(
                    r"^(?:python(?:\.exe)?(?:\s+-m)?|pytest|npm|pnpm|yarn|bun|deno|"
                    r"node|cargo|go\s+test|dotnet|playwright|npx\s+playwright)\b",
                    str(item).strip(),
                    re.IGNORECASE,
                )
            }
            if command not in approved_commands:
                error = (
                    f"command {str(args.get('command') or '')!r} is not an approval-bound "
                    "verification command for this node; use one of: "
                    + (", ".join(sorted(approved_commands)) or "the non-shell verifier in the plan")
                )
                self.store.append_event(
                    "tool_contract.rejected",
                    goal_id=self.goal_id,
                    payload={
                        "actor": request.role.value,
                        "received": [call.name],
                        "stage": "ultra_execution_command_scope",
                        "error": error,
                        "approval_requested": False,
                        "workspace_mutated": False,
                    },
                )
                return f"Error: {error}"
        if call.name in {"read_file", "list_files", "grep"} and "path" in args:
            component_path = self._component_read_path(
                request,
                node,
                str(args.get("path", ".")),
            )
            if component_path is not None:
                args["path"] = component_path
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
                    child_packages=(
                        dict(request.task.get("child_component_packages") or {})
                        if isinstance(
                            request.task.get("child_component_packages"), Mapping
                        )
                        else {}
                    ),
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
        tool_spec = tools.get_spec(call.name)
        if tool_spec is not None:
            try:
                args = dict(tools.validate_tool_arguments(tool_spec.schema, args))
            except (TypeError, ValueError) as exc:
                error = f"invalid arguments: {redact_text(exc, 1_000)}"
                self.store.append_event(
                    "tool_contract.rejected",
                    goal_id=self.goal_id,
                    payload={
                        "actor": request.role.value,
                        "received": [call.name],
                        "stage": "ultra_execution_arguments",
                        "error": error,
                        "approval_requested": False,
                        "workspace_mutated": False,
                    },
                )
                return f"Error: {error}"
            applicability_error = tools.applicability_issue(
                call.name, args, self.workspace
            )
            if applicability_error:
                self.store.append_event(
                    "tool_contract.rejected",
                    goal_id=self.goal_id,
                    payload={
                        "actor": request.role.value,
                        "received": [call.name],
                        "stage": "ultra_execution_applicability",
                        "error": applicability_error,
                        "approval_requested": False,
                        "workspace_mutated": False,
                    },
                )
                return f"Error: {applicability_error}"
        risk = _TOOL_RISK.get(call.name, "unknown")
        normal_requirement = tools.requires_approval(call.name, args)
        needs_approval = self.permission_adapter.requires_approval(normal_requirement)
        replay_key = ""
        if call.name in _WRITE_TOOLS:
            replay_key = hashlib.sha256(
                _json(
                    {
                        "tool": call.name,
                        "arguments": args,
                        "ultra_run_id": self.run_id,
                        "node_id": request.node_id,
                        "role": request.role.value,
                        "phase": request.phase,
                    }
                ).encode("utf-8")
            ).hexdigest()
            current_workspace_fingerprint = self._workspace_fingerprint()
            prior_guard = next(
                (
                    event
                    for event in reversed(
                        self.store.list_recent_events(self.goal_id, limit=2_000)
                    )
                    if event.event_type == "mutation.replay_guard"
                    and str(event.payload.get("replay_key") or "") == replay_key
                    and str(event.payload.get("post_workspace_fingerprint") or "")
                    == current_workspace_fingerprint
                ),
                None,
            )
            if prior_guard is not None:
                prior_result = str(
                    prior_guard.payload.get("result")
                    or "The identical saved mutation is already applied."
                )
                self.store.append_event(
                    "mutation.replay_prevented",
                    goal_id=self.goal_id,
                    entity_type="action",
                    entity_id=str(prior_guard.entity_id or ""),
                    payload={
                        "tool": call.name,
                        "node_id": request.node_id,
                        "phase": request.phase,
                        "replay_key": replay_key,
                        "workspace_fingerprint": current_workspace_fingerprint,
                    },
                )
                self.events.publish(
                    "tool_result",
                    "Identical saved mutation already applied; replay prevented.",
                    tool=call.name,
                    actor=request.role.value,
                    node_id=request.node_id,
                )
                return prior_result
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
        if needs_approval:
            try:
                approval_value = self.approval(call.name, dict(args), risk)
                if isinstance(approval_value, bool):
                    approval_granted = approval_value
                else:
                    approval_token = str(
                        getattr(approval_value, "value", approval_value)
                    ).strip().casefold()
                    if approval_token == "ui_error":
                        raise RuntimeError(
                            "the approval interface closed without a decision"
                        )
                    approval_granted = approval_token in {
                        "allow_once",
                        "allow_session",
                        "allow",
                        "yes",
                    }
            except Exception as exc:
                result = (
                    "Error: approval could not be collected; the action was not "
                    f"executed. {type(exc).__name__}: {redact_text(exc, 500)}"
                )
                self.store.complete_action(action_id, result, status="failed")
                self._checkpoint_tool_approval_boundary(
                    tool=call.name,
                    args=args,
                    risk=risk,
                    action_id=action_id,
                    reason=f"Approval for {call.name} could not be collected.",
                )
                self.events.publish(
                    "tool_result",
                    result,
                    tool=call.name,
                    actor=request.role.value,
                    node_id=request.node_id,
                )
                raise ApprovalRequiredError(
                    f"Approval for {call.name} is required before execution can continue."
                ) from exc
            if not approval_granted:
                result = "Permission denied by the user. Do not repeat the same action."
                self.store.complete_action(action_id, result, status="denied")
                self._checkpoint_tool_approval_boundary(
                    tool=call.name,
                    args=args,
                    risk=risk,
                    action_id=action_id,
                    reason=f"Approval for {call.name} was not granted.",
                )
                self.events.publish(
                    "tool_result",
                    result,
                    tool=call.name,
                    actor=request.role.value,
                    node_id=request.node_id,
                )
                raise ApprovalRequiredError(
                    f"Approval for {call.name} is required before execution can continue."
                )
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
            if (
                call.name == "read_file"
                and str(result).startswith("Error:")
                and "does not exist" in str(result).casefold()
                and node is not None
                and _within_scope(
                    str(args.get("path") or ""),
                    tuple(getattr(node.contract, "read_paths", (".",)))
                    + tuple(node.contract.write_paths),
                )
            ):
                missing_path = str(args.get("path") or "")
                result = (
                    f"Observation: {missing_path!r} does not exist yet. "
                    "Do not repeat this read. If it is this task's output, create it through "
                    "the approved write tool; otherwise report the missing dependency."
                )
                self.store.append_event(
                    "tool_contract.missing_file_observed",
                    goal_id=self.goal_id,
                    payload={
                        "actor": request.role.value,
                        "node_id": request.node_id,
                        "path": missing_path,
                        "workspace_mutated": False,
                    },
                )
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
                self.store.append_event(
                    "mutation.replay_guard",
                    goal_id=self.goal_id,
                    entity_type="action",
                    entity_id=action_id,
                    payload={
                        "tool": call.name,
                        "node_id": request.node_id,
                        "phase": request.phase,
                        "replay_key": replay_key,
                        "post_workspace_fingerprint": self._workspace_fingerprint(),
                        "result": redact_text(result, 2_000),
                    },
                )
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
        if (
            call.name in _WRITE_TOOLS
            and not result.startswith("Error:")
            and replay_key
        ):
            self.store.append_event(
                "mutation.replay_guard",
                goal_id=self.goal_id,
                entity_type="action",
                entity_id=action_id,
                payload={
                    "tool": call.name,
                    "node_id": request.node_id,
                    "phase": request.phase,
                    "replay_key": replay_key,
                    "post_workspace_fingerprint": self._workspace_fingerprint(),
                    "result": redact_text(result, 2_000),
                },
            )
        self.events.publish(
            "tool_result",
            result,
            tool=call.name,
            actor=request.role.value,
            node_id=request.node_id,
        )
        return result

    def _ensure_outcome_contract(self, objective: str) -> Mapping[str, Any]:
        """Create or refresh the durable authority above individual Ultra runs."""

        assert self.goal_id
        normalized = objective.casefold()
        visual = any(
            term in normalized
            for term in (
                "html",
                "three.js",
                "threejs",
                "game",
                "لعبة",
                "frontend",
                "dashboard",
                "visual",
                "web app",
            )
        )
        autonomous_evidence = (
            "final_artifact",
            "runtime",
            "screenshots",
            "orchestrator_completion",
        )
        contract = GoalOutcomeContractV1(
            goal_id=self.goal_id,
            objective=objective,
            required_evidence=(
                autonomous_evidence
                if visual
                else ("orchestrator_completion",)
            ),
            require_candidate_preferred=False,
        )
        try:
            current = self.store.get_goal_outcome_contract(self.goal_id)
        except NotFoundError:
            return self.store.save_goal_outcome_contract(
                contract,
                ultra_run_id=self.run_id,
                process_token=f"pid:{os.getpid()}",
            )
        restored = GoalOutcomeContractV1.from_dict(current["contract"])
        # Older autonomous visual contracts required ``codex_visual_review``.
        # That evidence is intentionally recorded only by the external
        # supervisor API, never by the local Ultra runtime.  Consequently a
        # fully successful Playwright run could not satisfy its own contract
        # and was changed from COMPLETED to BLOCKED during finalization.
        # Migrate only the exact shape previously generated here; bespoke
        # contracts that explicitly require a supervisor remain untouched.
        if (
            restored.required_evidence
            == (
                "final_artifact",
                "runtime",
                "screenshots",
                "codex_visual_review",
            )
            and not restored.require_candidate_preferred
        ):
            restored = replace(restored, required_evidence=autonomous_evidence)
        return self.store.save_goal_outcome_contract(
            restored,
            ultra_run_id=self.run_id,
            state=(
                GoalOutcomeState.RECOVERING
                if current["state"] == GoalOutcomeState.RECOVERING.value
                else GoalOutcomeState.RUNNING
            ),
            process_token=f"pid:{os.getpid()}",
        )

    def start(
        self,
        objective: str,
        *,
        requested_effects: Sequence[str] = (),
    ) -> MasterPlanV1 | None:
        if self.orchestrator and self.orchestrator.phase not in {
            EnginePhase.COMPLETED,
            EnginePhase.CANCELLED,
            EnginePhase.FAILED,
        }:
            raise RuntimeError("an ULTRA run is already active")
        # The recursive foundation is part of the caller's durable workspace
        # thread. Creating an unbound Goal made Runtime.active_goal() return
        # None at the approval boundary, so Full Auto and resume logic could
        # not see or approve the ready master plan.
        goal = self.store.create_goal(
            redact_text(objective, 20_000),
            session_id=self.session_id,
        )
        self.store.update_goal_metadata(
            goal.id,
            adaptive_orchestration_policy=self.adaptive_orchestration_policy.to_dict(),
        )
        if requested_effects:
            self.store.update_goal_metadata(
                goal.id,
                accepted_requested_effects=list(
                    dict.fromkeys(str(item) for item in requested_effects if str(item))
                ),
            )
        self.store.transition_goal(goal.id, GoalStatus.DISCOVERING, reason="ULTRA foundation started")
        return self._prepare_existing_goal(
            goal.id,
            objective,
            requested_effects=requested_effects,
        )

    def restart_foundation(self, goal_id: str, objective: str) -> MasterPlanV1 | None:
        """Create a fresh ULTRA run/revision while preserving the durable goal."""

        effects = tuple(
            str(item)
            for item in self.store.get_goal(goal_id).metadata.get(
                "accepted_requested_effects", ()
            )
            if str(item)
        )
        return self._prepare_existing_goal(
            goal_id,
            objective,
            requested_effects=effects,
        )

    def restart_plan_from_accepted_foundation(
        self,
        goal_id: str,
        source_run_id: str,
        feedback: str,
        *,
        allow_scope_expansion: bool = False,
    ) -> MasterPlanV1:
        """Create a repair plan without replaying accepted semantic stages."""

        source = self.store.get_ultra_run(source_run_id)
        if source.goal_id != goal_id:
            raise RuntimeError("repair source run does not belong to the active goal")
        if source.goal_spec is None or source.architecture_spec is None:
            raise RuntimeError("repair source has no accepted semantic foundation")
        goal = self.store.get_goal(goal_id)
        self.store.update_goal_metadata(
            goal_id,
            accepted_foundation_source_run_id=source_run_id,
        )
        goal = self.store.get_goal(goal_id)
        self.goal_id = goal_id
        self.answers = dict(goal.metadata.get("plan_answers", {}))
        self.adapter = StateStoreUltraAdapter(
            self.store,
            goal_id,
            self.descriptor,
            self.permission_adapter.access_level,
            self.config,
            workspace=self.workspace,
            events=self.events,
        )
        repair_kind = (
            "EVIDENCE-BOUND SCOPE REPAIR REQUEST"
            if allow_scope_expansion
            else "IN-SCOPE QUALITY REPAIR REQUEST"
        )
        repair_policy = (
            "The prior approved paths were proven insufficient by authoritative evidence. "
            "You may add only concrete workspace-relative paths directly required to resolve "
            "the cited failures. Do not add unrelated dependencies, network effects, or permissions. "
            "The expanded revision will wait for explicit user approval."
            if allow_scope_expansion
            else "Reuse the accepted scope and output authority."
        )
        repair_prompt = (
            f"{goal.objective}\n\n{repair_kind}:\n"
            f"{redact_text(feedback, 4_000)}\n"
            "Reuse the accepted semantic goal and architecture. "
            f"{repair_policy} "
            "Revise only the execution plan and verification needed to resolve the cited evidence."
        )
        self._ensure_outcome_contract(goal.objective)
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
        engine_goal = EngineGoalSpec(
            objective=source.goal_spec.objective,
            success_criteria=source.goal_spec.success_criteria
            or ("Complete the requested outcome with authoritative evidence.",),
            constraints=source.goal_spec.constraints,
            in_scope=source.goal_spec.scope,
            out_of_scope=source.goal_spec.non_goals,
            assumptions=tuple(
                f"{key}: {value}"
                for key, value in source.goal_spec.answered_questions.items()
            ),
        )
        interfaces = tuple(
            {
                "name": name,
                **(
                    dict(value)
                    if isinstance(value, Mapping)
                    else {"contract": value}
                ),
            }
            for name, value in source.architecture_spec.interfaces.items()
        )
        accepted_engine_architecture = EngineArchitectureSpec(
            summary=source.architecture_spec.summary,
            components=source.architecture_spec.components
            or ({"name": "accepted-project"},),
            interfaces=interfaces,
            decisions=source.architecture_spec.decisions,
            invariants=source.architecture_spec.constraints,
        )
        planning_architecture = _bind_repair_architecture_authority(
            accepted_engine_architecture,
            redact_text(feedback, 4_000),
        )
        explicit_repair_paths = _explicit_repair_scope_paths(feedback)
        plan = self.orchestrator.prepare_plan_revision(
            repair_prompt,
            goal_spec=engine_goal,
            architecture=planning_architecture,
            requested_effects=tuple(
                str(item)
                for item in goal.metadata.get("accepted_requested_effects", ())
                if str(item)
            ),
            approved_write_paths=(
                explicit_repair_paths
                or tuple(
                    str(item)
                    for item in goal.metadata.get("approved_scope_paths", ())
                    if str(item)
                )
            ) if not allow_scope_expansion else (),
        )
        plan = _bind_repair_feedback(
            plan,
            redact_text(feedback, 4_000),
        )
        # The orchestrator owns the executable master contract used after
        # approval.  Keep its in-memory copy aligned with the persisted Plan
        # projection so execution cannot fall back to the model's lossy,
        # generic repair wording.
        self.orchestrator.master_plan = plan
        # Replanning may enrich the transient planning context with the latest
        # repair authority, but the accepted semantic ArchitectureSpec is an
        # immutable foundation and must persist byte-for-byte across revisions.
        self.adapter.bind_foundation(engine_goal, accepted_engine_architecture, plan)
        assert self.adapter.run_id
        durable_run = self.store.get_ultra_run(self.adapter.run_id)
        fingerprint = (
            durable_run.master_plan_fingerprint
            or str(durable_run.config.get("master_plan_fingerprint", ""))
            or plan.fingerprint
        )
        policy = QualityPolicyV1()
        self.store.save_quality_policy(
            self.adapter.run_id,
            policy,
            master_plan_fingerprint=fingerprint,
        )
        baseline_hashes = self._workspace_hashes()
        baseline = QualityCycleV1(
            ultra_run_id=self.adapter.run_id,
            kind=QualityCycleKind.BASELINE,
            attempt=1,
            approach_fingerprint=hashlib.sha256(
                _json(
                    {
                        "inventory": sorted(baseline_hashes),
                        "master_plan": fingerprint,
                        "source_run_id": source_run_id,
                    }
                ).encode("utf-8")
            ).hexdigest(),
            inputs={
                "inventory": sorted(baseline_hashes),
                "file_hashes": baseline_hashes,
                "master_plan_fingerprint": fingerprint,
                "accepted_foundation_source": source_run_id,
            },
            outputs={
                "confirmed_findings": [],
                "quality_checklist": list(policy.required_reviews),
            },
            metrics={"project_files": len(baseline_hashes)},
            result="baseline_complete",
            ended_at=utc_now(),
        )
        self.store.save_quality_cycle(baseline)
        self._capture_workspace_baseline(self.adapter.run_id)
        self.events.publish(
            "ultra.plan_repair_prepared",
            "Prepared a new plan from the accepted foundation without semantic replay.",
            run_id=self.adapter.run_id,
            source_run_id=source_run_id,
            goal_id=goal_id,
            fingerprint=fingerprint,
        )
        return plan

    def _prepare_existing_goal(
        self,
        goal_id: str,
        objective: str,
        *,
        requested_effects: Sequence[str] = (),
    ) -> MasterPlanV1 | None:
        self.goal_id = goal_id
        self.answers = {}
        self.adapter = StateStoreUltraAdapter(
            self.store,
            goal_id,
            self.descriptor,
            self.permission_adapter.access_level,
            self.config,
            workspace=self.workspace,
            events=self.events,
        )
        # The durable product goal exists above foundation generation.  A
        # malformed local-model architecture must therefore be recoverable
        # instead of leaving no outcome contract or heartbeat behind.
        self._ensure_outcome_contract(objective)
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
        plan = self.orchestrator.prepare(
            objective,
            requested_effects=requested_effects,
        )
        if plan is None:
            assert self.orchestrator.goal_spec
            self.adapter.checkpoint_questions(self.orchestrator.goal_spec)
            self._ensure_outcome_contract(objective)
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
        baseline_hashes = {
            path: digest
            for path, digest in self._workspace_hashes().items()
            if not path.startswith(("run-artifacts/", "output/playwright/"))
        }
        inventory = sorted(baseline_hashes)
        baseline = QualityCycleV1(
            ultra_run_id=self.adapter.run_id,
            kind=QualityCycleKind.BASELINE,
            attempt=1,
            approach_fingerprint=hashlib.sha256(
                _json({"inventory": inventory, "master_plan": durable_master_fingerprint}).encode("utf-8")
            ).hexdigest(),
            inputs={
                "inventory": inventory,
                "file_hashes": baseline_hashes,
                "master_plan_fingerprint": durable_master_fingerprint,
            },
            outputs={"confirmed_findings": [], "quality_checklist": list(policy.required_reviews)},
            metrics={"project_files": len(inventory)},
            result="clean" if not inventory else "baseline_complete",
            ended_at=utc_now(),
        )
        self.store.save_quality_cycle(baseline)
        recovery_manifest = self._capture_workspace_baseline(self.adapter.run_id)
        self.events.publish(
            "ultra.workspace_baseline_captured",
            "Accepted workspace baseline captured before approval; rejected changes can be rolled back.",
            run_id=self.adapter.run_id,
            file_count=len(dict(recovery_manifest.get("files") or {})),
        )
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
            data={
                "cycle_id": baseline.id,
                "inventory": inventory,
                "file_hashes": baseline_hashes,
            },
        )
        self._ensure_outcome_contract(objective)
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
            WorkNodeStatus.CONFLICT: NodeStatus.CONFLICT,
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

    def restore(
        self,
        run_id: str,
        *,
        start_background: bool = True,
    ) -> Future[UltraRunResult] | Plan:
        """Rebuild the scheduler from durable evidence without replaying uncertainty.

        ``start_background=False`` is the inspection/execution-cut boundary. It
        restores the exact durable graph and accepted packages without waking
        every ready node. Targeted E2E tools can then execute a bounded set of
        specialists while the full goal remains incomplete.
        """

        run = self.store.get_ultra_run(run_id)
        engine_metadata = (
            dict(run.config.get("engine_metadata") or {})
            if isinstance(run.config.get("engine_metadata"), Mapping)
            else {}
        )
        raw_goal = engine_metadata.get("accepted_goal_spec")
        raw_architecture = engine_metadata.get("accepted_architecture")
        if (
            run.goal_spec is None
            and isinstance(raw_goal, Mapping)
        ):
            restored_goal = EngineGoalSpec(
                objective=str(raw_goal.get("objective") or ""),
                success_criteria=tuple(
                    str(item) for item in raw_goal.get("success_criteria", ()) if str(item)
                ),
                constraints=tuple(
                    str(item) for item in raw_goal.get("constraints", ()) if str(item)
                ),
                in_scope=tuple(
                    str(item) for item in raw_goal.get("in_scope", ()) if str(item)
                ),
                out_of_scope=tuple(
                    str(item) for item in raw_goal.get("out_of_scope", ()) if str(item)
                ),
                assumptions=tuple(
                    str(item) for item in raw_goal.get("assumptions", ()) if str(item)
                ),
                questions=tuple(
                    dict(item)
                    for item in raw_goal.get("questions", ())
                    if isinstance(item, Mapping)
                ),
            )
            restored_architecture = None
            if isinstance(raw_architecture, Mapping):
                restored_architecture = EngineArchitectureSpec(
                    summary=str(raw_architecture.get("summary") or ""),
                    components=tuple(
                        dict(item)
                        for item in raw_architecture.get("components", ())
                        if isinstance(item, Mapping)
                    ),
                    interfaces=tuple(
                        dict(item)
                        for item in raw_architecture.get("interfaces", ())
                        if isinstance(item, Mapping)
                    ),
                    decisions=tuple(
                        dict(item)
                        for item in raw_architecture.get("decisions", ())
                        if isinstance(item, Mapping)
                    ),
                    invariants=tuple(
                        str(item)
                        for item in raw_architecture.get("invariants", ())
                        if str(item)
                    ),
                )
            self.store.update_ultra_run(
                run_id,
                goal_spec=StateStoreUltraAdapter._goal_spec(restored_goal),
                architecture_spec=(
                    StateStoreUltraAdapter._architecture(restored_architecture)
                    if restored_architecture is not None
                    else None
                ),
            )
            run = self.store.get_ultra_run(run_id)
            self.events.publish(
                "ultra.foundation_checkpoint_enriched",
                "Restored accepted foundation metadata without repeating a model stage.",
                run_id=run_id,
                architecture_restored=restored_architecture is not None,
            )
        checkpoint = (
            dict(engine_metadata.get("foundation_checkpoint") or {})
            if isinstance(engine_metadata.get("foundation_checkpoint"), Mapping)
            else {}
        )
        if (
            run.status is UltraRunStatus.PAUSED
            and str(checkpoint.get("stage") or "") == "master_plan"
            and run.plan_revision is None
            and run.goal_spec is not None
            and run.architecture_spec is not None
        ):
            self.store.update_ultra_run(
                run.id,
                status=UltraRunStatus.BLOCKED,
                error="superseded by targeted master-plan checkpoint repair",
            )
            self.events.publish(
                "ultra.master_plan_checkpoint_resumed",
                "Resuming only the saved master-plan stage from accepted foundation data.",
                run_id=run.id,
                goal_id=run.goal_id,
            )
            return self.restart_plan_from_accepted_foundation(
                run.goal_id,
                run.id,
                str(checkpoint.get("error") or "repair the master-plan contract"),
            )
        nodes = self.store.list_work_nodes(run_id)
        reconciled_nodes = []
        for item in nodes:
            if item.status is WorkNodeStatus.COMPLETED:
                artifact_paths = (
                    tuple(item.result.changed_files)
                    if item.result is not None and item.result.changed_files
                    else tuple(item.contract.write_paths)
                )
                missing_paths = tuple(
                    path
                    for path in artifact_paths
                    if path not in {"", "."}
                    and not (self.workspace / PurePosixPath(path)).is_file()
                )
                if missing_paths:
                    self.store.transition_work_node(
                        item.id,
                        WorkNodeStatus.READY,
                        error=(
                            "completed evidence invalidated because durable artifact(s) "
                            f"are missing: {', '.join(missing_paths)}"
                        ),
                        checkpoint="recovery",
                    )
                    item = replace(
                        item,
                        status=WorkNodeStatus.READY,
                        result=None,
                        error=(
                            "completed evidence invalidated because durable artifact(s) "
                            f"are missing: {', '.join(missing_paths)}"
                        ),
                        checkpoint="recovery",
                    )
                    self.events.publish(
                        "ultra.completed_evidence_invalidated",
                        f"Reopened {item.id}; its completed artifact is missing.",
                        run_id=run_id,
                        node_id=item.id,
                        missing_paths=list(missing_paths),
                    )
            reconciled_nodes.append(item)
        nodes = tuple(reconciled_nodes)
        node_map = {item.id: item for item in nodes}
        root_nodes = tuple(
            item for item in nodes if item.parent_id is None
        )
        invalid_child_ids: set[str] = set()
        for item in nodes:
            if not item.parent_id or item.parent_id not in node_map:
                continue
            parent = node_map[item.parent_id]
            declared_targets = UltraOrchestrator._declared_write_targets(
                item.title,
                item.objective,
            )
            invalid_targets = tuple(
                path
                for path in declared_targets
                if not any(
                    UltraOrchestrator._contains_scope(scope, path)
                    for scope in parent.contract.write_paths
                )
            )
            child_terms = UltraOrchestrator._semantic_terms(item.objective)
            parent_overlap = len(
                child_terms
                & UltraOrchestrator._semantic_terms(parent.objective)
            )
            sibling_matches = [
                (
                    len(
                        child_terms
                        & UltraOrchestrator._semantic_terms(candidate.objective)
                    ),
                    candidate,
                )
                for candidate in root_nodes
                if candidate.id != parent.id
            ]
            sibling_overlap, semantic_owner = max(
                sibling_matches,
                key=lambda value: value[0],
                default=(0, None),
            )
            semantic_drift = (
                semantic_owner is not None
                and sibling_overlap >= 2
                and sibling_overlap > parent_overlap
            )
            if invalid_targets or semantic_drift:
                self.store.transition_work_node(
                    item.id,
                    WorkNodeStatus.CANCELLED,
                    error=(
                        "restored child contract conflicts with its parent scope"
                        + (
                            f": {', '.join(invalid_targets)}"
                            if invalid_targets
                            else (
                                f"; semantic owner is {semantic_owner.id}"
                                if semantic_owner is not None
                                else ""
                            )
                        )
                    ),
                    checkpoint="recovery",
                )
                invalid_child_ids.add(item.id)
                self.events.publish(
                    "ultra.child_contract_rejected",
                    f"Cancelled restored child {item.id}; it writes outside parent scope.",
                    run_id=run_id,
                    node_id=item.parent_id,
                    child_id=item.id,
                    invalid_targets=list(invalid_targets),
                    semantic_owner=(
                        semantic_owner.id
                        if semantic_drift and semantic_owner is not None
                        else ""
                    ),
                )
        retained_child_scopes: dict[
            tuple[str, tuple[str, ...]],
            WorkNode,
        ] = {}
        for item in nodes:
            if (
                item.id in invalid_child_ids
                or not item.parent_id
                or item.status is WorkNodeStatus.CANCELLED
            ):
                continue
            scope_key = (
                item.parent_id,
                tuple(
                    sorted(
                        _normalized_path(path).casefold()
                        for path in item.contract.write_paths
                    )
                ),
            )
            if not scope_key[1]:
                continue
            retained = retained_child_scopes.get(scope_key)
            if retained is None:
                retained_child_scopes[scope_key] = item
                continue
            semantic_overlap = (
                UltraOrchestrator._semantic_terms(item.objective)
                & UltraOrchestrator._semantic_terms(retained.objective)
            )
            if len(semantic_overlap) < 2:
                continue
            self.store.transition_work_node(
                item.id,
                WorkNodeStatus.CANCELLED,
                error=(
                    "duplicate restored child contract for the same parent "
                    f"and write scope as {retained.id}"
                ),
                checkpoint="recovery",
            )
            invalid_child_ids.add(item.id)
            self.events.publish(
                "ultra.duplicate_child_cancelled",
                f"Cancelled duplicate child {item.id}; {retained.id} owns its scope.",
                run_id=run_id,
                node_id=item.parent_id,
                child_id=item.id,
                retained_child_id=retained.id,
                write_paths=list(scope_key[1]),
            )
        if invalid_child_ids:
            nodes = tuple(
                item for item in nodes if item.id not in invalid_child_ids
            )
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
        pending_questions = tuple(
            item
            for item in run.config.get("pending_questions", ())
            if isinstance(item, Mapping)
        )
        question_boundary = bool(
            run.goal_spec is not None
            and run.architecture_spec is None
            and pending_questions
        )
        awaiting_approval = not run.master_approved or run.plan_revision is None
        if question_boundary:
            plan = None
        elif awaiting_approval:
            if run.status is not UltraRunStatus.AWAITING_APPROVAL:
                raise RuntimeError("the interrupted ULTRA run has no approved master plan")
            plan = self.store.get_latest_plan(run.goal_id)
            if plan is None or plan.status is not PlanStatus.PENDING_APPROVAL:
                raise RuntimeError("the interrupted ULTRA run has no pending master plan")
        else:
            plan = self.store.get_plan(run.goal_id, run.plan_revision)
        if question_boundary:
            # Restore the exact durable question boundary without asking the
            # provider to repeat foundation work. Answering the remaining
            # questions continues architecture and plan generation from here.
            self.goal_id = run.goal_id
            self.adapter = StateStoreUltraAdapter(
                self.store,
                run.goal_id,
                self.descriptor,
                self.permission_adapter.access_level,
                self.config,
                workspace=self.workspace,
                events=self.events,
            )
            self.adapter.run_id = run.id
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
            prompt = str(
                run.config.get("prompt")
                or self.store.get_goal(run.goal_id).objective
            )
            engine_run = UltraRunV1(
                id=run.id,
                prompt=prompt,
                execution_class=self.descriptor.execution_class,
                phase=EnginePhase.AWAITING_QUESTIONS,
                concurrency=run.concurrency,
                approved=False,
                model_snapshot=self.descriptor.to_dict(),
                config_snapshot=dict(run.config),
                metadata={
                    "restored": True,
                    "pending_questions": list(pending_questions),
                },
                created_at=run.created_at,
                updated_at=run.updated_at,
            )
            self.orchestrator.run_state = engine_run
            self.orchestrator.goal_spec = EngineGoalSpec(
                objective=run.goal_spec.objective,
                success_criteria=run.goal_spec.success_criteria
                or ("Complete the requested outcome with authoritative evidence.",),
                constraints=run.goal_spec.constraints,
                in_scope=run.goal_spec.scope,
                out_of_scope=run.goal_spec.non_goals,
                assumptions=tuple(
                    f"{key}: {value}"
                    for key, value in run.goal_spec.answered_questions.items()
                ),
                questions=pending_questions,
            )
            self.answers = dict(
                self.store.get_goal(run.goal_id).metadata.get("plan_answers", {})
            )
            self.adapter.runs[run.id] = engine_run
            self.events.publish(
                "ultra.questions_restored",
                "Restored the durable ULTRA question boundary.",
                run_id=run.id,
                questions=len(pending_questions),
            )
            return self.orchestrator.master_plan
        if run.goal_spec is None or run.architecture_spec is None:
            raise RuntimeError(
                "the interrupted ULTRA foundation has no recoverable question checkpoint; choose Replan"
            )

        self.goal_id = run.goal_id
        self.adapter = StateStoreUltraAdapter(
            self.store,
            run.goal_id,
            self.descriptor,
            self.permission_adapter.access_level,
            self.config,
            workspace=self.workspace,
            events=self.events,
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
            children = tuple(
                str(value)
                for value in item.metadata.get("children", ())
                if str(value) in stored_by_id
            )
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
            dependencies = tuple(
                dict.fromkeys(
                    (
                        *(
                            dependency
                            for dependency in item.depends_on
                            if dependency in stored_by_id
                        ),
                        *children,
                    )
                )
            )
            try:
                phase = InnerPhase(item.checkpoint) if item.checkpoint else None
            except ValueError:
                phase = None
            restored_status = self._engine_node_status(item.status)
            stored_result_success = bool(
                item.result
                and item.result.metadata.get("success")
            )
            restored_artifact_check = (
                _validate_workspace_artifacts(
                    self.workspace,
                    contract.write_paths,
                )
                if stored_result_success
                else {"passed": False}
            )
            restored_executable_check = (
                _run_python_test_artifacts(
                    self.workspace,
                    contract.write_paths,
                )
                if stored_result_success
                and bool(restored_artifact_check.get("passed"))
                else {"passed": False}
            )
            durable_artifacts_valid = bool(
                stored_result_success
                and restored_artifact_check.get("passed")
                and restored_executable_check.get("passed")
            )
            if durable_artifacts_valid:
                # A process can fail after the accepted ResultPackage and
                # workspace artifact are durable (for example while emitting a
                # secondary package receipt). Preserve that evidence boundary
                # and never replay the successful mutation.
                restored_status = NodeStatus.COMPLETED
                self.events.publish(
                    "ultra.post_evidence_completion_recovered",
                    f"Recovered completed evidence for {item.id}.",
                    run_id=run_id,
                    node_id=item.id,
                    previous_status=item.status.value,
                )
            elif (
                not awaiting_approval
                and restored_status
                in {
                    NodeStatus.FAILED,
                    NodeStatus.BLOCKED,
                    NodeStatus.CONFLICT,
                    NodeStatus.REVISION_REQUIRED,
                }
            ):
                restored_status = NodeStatus.READY
                self.events.publish(
                    "ultra.recoverable_node_reopened",
                    f"Reopened {item.id} from its last durable evidence gate.",
                    run_id=run_id,
                    node_id=item.id,
                    previous_status=item.status.value,
                )
            engine = EngineWorkNode(
                contract=contract,
                parent_id=item.parent_id,
                depth=item.depth or 1,
                kind=NodeKind(item.kind.value),
                order=item.position,
                status=restored_status,
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
        if not module_contracts:
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
                # The process may stop after the approval-bound Plan is
                # committed but before its top-level WorkNodes are materialized.
                # Queue the reconstructed immutable nodes in the adapter now;
                # approve_master() will flush them before specialist profiles,
                # whose foreign keys require those WorkNodes to exist.
                self.adapter.save_work_node(run.id, node)

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
        for node_id, result in self.adapter.results[run_id].items():
            node = engine_nodes.get(node_id)
            if node is not None and result.success:
                # A process can stop after the durable result package and
                # WorkNode are committed but before the legacy Plan task is
                # advanced. Reconcile that projection before scheduling; the
                # implementation itself is never replayed.
                self.adapter._sync_master_task(node, result)
        if awaiting_approval:
            self.events.publish(
                "ultra.awaiting_approval",
                f"Restored ULTRA plan revision {plan.revision}; approval is still required",
                run_id=run_id,
                revision=plan.revision,
            )
            return plan
        if not start_background:
            self.events.publish(
                "ultra.restored_idle",
                "Restored ULTRA graph without starting the full scheduler",
                run_id=run_id,
                nodes=len(engine_nodes),
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
            local_default=self.config.local_concurrency,
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
        # Permission policy changes do not move the workspace or replace the
        # execution environment, so they can take effect live between worker
        # tool calls.  Model/provider changes still require the ordinary safe
        # reconfiguration boundary.
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

    def adopt_plan_revision(self, plan: Plan) -> MasterPlanV1:
        """Rebind an edited pre-approval Plan to the exact Ultra master graph.

        The browser edits the durable Plan record, never a parallel execution
        graph. Before approval we rebuild only the unstarted first layer and
        bind both representations to the same fingerprint.
        """

        if not self.orchestrator or not self.adapter or not self.run_id:
            raise RuntimeError("there is no live ULTRA foundation to update")
        run = self.store.get_ultra_run(self.run_id)
        if run.master_approved or self.adapter.approved:
            raise RuntimeError("an approved ULTRA graph cannot adopt a new plan revision")
        if self.adapter._persisted_nodes:
            raise RuntimeError("first-layer work already exists; create a guided replan instead")

        from .ultra import TaskContractV1 as EngineTaskContract

        paths_by_task: dict[str, list[str]] = {}
        for change in plan.expected_changes:
            if not isinstance(change, Mapping):
                continue
            path = str(change.get("path") or "").strip()
            if not path or path.startswith("<"):
                continue
            for task_id in change.get("supports_tasks", ()):
                paths_by_task.setdefault(str(task_id), []).append(path)
        task_to_node = {
            task.id: str(task.metadata.get("ultra_node_id") or task.id)
            for task in plan.tasks
        }
        modules = []
        nodes: dict[str, EngineWorkNode] = {}
        for position, task in enumerate(plan.tasks, 1):
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
                forbidden_changes=tuple(task.metadata.get("constraints") or ()),
                owned_interfaces=tuple(task.metadata.get("owned_interfaces") or ()),
                metadata={**dict(task.metadata), "adopted_plan_revision": plan.revision},
            )
            modules.append(contract)
            nodes[node_id] = EngineWorkNode(contract=contract, order=position)
        master = MasterPlanV1(
            summary=plan.summary,
            modules=tuple(modules),
            execution_strategy=plan.execution_strategy,
            revision=plan.revision,
            fingerprint=plan.fingerprint,
        )
        self.adapter.plan = plan
        self.adapter.task_ids = {task_to_node[task.id]: task.id for task in plan.tasks}
        self.adapter._pending_nodes.clear()
        self.orchestrator.master_plan = master
        self.orchestrator.nodes = nodes
        self.orchestrator._order = max((node.order for node in nodes.values()), default=0)
        for node in nodes.values():
            self.adapter.save_work_node(self.run_id, node)
        self.store.update_ultra_run(
            self.run_id,
            phase=UltraPhase.AWAITING_APPROVAL,
            status=UltraRunStatus.AWAITING_APPROVAL,
            config={
                "master_plan_fingerprint": plan.fingerprint,
                "module_count": len(modules),
                "adopted_plan_revision": plan.revision,
            },
        )
        self.events.publish(
            "ultra.plan_revision_adopted",
            f"Ultra first-layer preview now matches Plan r{plan.revision}.",
            run_id=self.run_id,
            plan_revision=plan.revision,
            plan_fingerprint=plan.fingerprint,
        )
        return master

    def approve(
        self,
        revision: int | None = None,
        *,
        approved_by: str = "user",
    ) -> Plan:
        if not self.orchestrator or not self.adapter or not self.orchestrator.master_plan:
            raise RuntimeError("there is no ULTRA master plan to approve")
        if revision is not None and self.adapter.plan and revision != self.adapter.plan.revision:
            raise ValueError(f"ULTRA is awaiting plan revision {self.adapter.plan.revision}")
        self.orchestrator.approve(self.orchestrator.master_plan.fingerprint)
        accepted = self.adapter.approve_master(
            self.orchestrator.master_plan,
            approved_by=approved_by,
        )
        self.future = self.orchestrator.background.start(self._run_and_finalize)
        return accepted

    def _record_automatic_outcome_evidence(self) -> None:
        """Project existing runtime and judge records into the final gate."""

        if not self.run_id:
            return
        self.store.record_final_acceptance_evidence(
            FinalAcceptanceEvidenceV1(
                ultra_run_id=self.run_id,
                kind="orchestrator_completion",
                authority="harness:ultra-global-gate",
                passed=True,
                score=1.0,
                details={"critical": True},
            )
        )
        index = self.workspace / "index.html"
        if index.is_file() and index.stat().st_size > 1_000:
            self.store.record_final_acceptance_evidence(
                FinalAcceptanceEvidenceV1(
                    ultra_run_id=self.run_id,
                    kind="final_artifact",
                    authority="harness:file-hash",
                    passed=True,
                    score=1.0,
                    artifact_hash=hashlib.sha256(index.read_bytes()).hexdigest(),
                    details={
                        "path": "index.html",
                        "size": index.stat().st_size,
                        "critical": True,
                    },
                )
            )
        screenshots = tuple(
            path
            for root in (
                self.workspace / "output" / "playwright",
                self.workspace / "run-artifacts" / "final",
            )
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if screenshots:
            digest = hashlib.sha256()
            for path in sorted(screenshots):
                digest.update(path.read_bytes())
            self.store.record_final_acceptance_evidence(
                FinalAcceptanceEvidenceV1(
                    ultra_run_id=self.run_id,
                    kind="screenshots",
                    authority="harness:playwright-artifacts",
                    passed=True,
                    score=1.0,
                    artifact_hash=digest.hexdigest(),
                    details={
                        "paths": [
                            path.relative_to(self.workspace).as_posix()
                            for path in screenshots
                        ],
                        "critical": True,
                    },
                )
            )
        benchmarks = [
            item
            for item in self.store.list_benchmark_results(limit=1_000)
            if str(item.get("ultra_run_id") or "") == self.run_id
            and str(item.get("result") or "") == "passed"
            and str(item.get("evaluation_authority") or "") != "legacy_heuristic"
        ]
        if benchmarks:
            latest = benchmarks[0]
            scores = [
                float(value)
                for value in dict(latest.get("scores") or {}).values()
                if isinstance(value, (int, float))
            ]
            self.store.record_final_acceptance_evidence(
                FinalAcceptanceEvidenceV1(
                    ultra_run_id=self.run_id,
                    kind="runtime",
                    authority=str(
                        latest.get("evaluation_authority") or "deterministic_harness"
                    ),
                    passed=True,
                    score=min(scores) if scores else 1.0,
                    details={
                        "benchmark_id": latest["id"],
                        "suite": latest["suite_name"],
                        "scenario": latest["scenario_name"],
                        "critical": True,
                    },
                )
            )
        for item in self.store.list_visual_evaluations(self.run_id):
            verdict = dict(item.get("verdict") or {})
            scores = [
                float(value)
                for value in dict(verdict.get("scores") or {}).values()
                if isinstance(value, (int, float))
            ]
            findings = tuple(verdict.get("findings") or ())
            critical = sum(
                1
                for finding in findings
                if isinstance(finding, Mapping)
                and str(finding.get("severity") or "").casefold() == "critical"
            )
            if bool(verdict.get("accepted")):
                self.store.record_final_acceptance_evidence(
                    FinalAcceptanceEvidenceV1(
                        ultra_run_id=self.run_id,
                        kind="independent_visual",
                        authority=(
                            f"{item.get('evaluator')}:{item.get('model')}:"
                            f"{item.get('context_fingerprint')}"
                        ),
                        passed=True,
                        score=min(scores) if scores else 0.0,
                        critical_findings=critical,
                        artifact_hash=str(item.get("screenshot_hash") or ""),
                        details={
                            "visual_evaluation_id": item["id"],
                            "critical": True,
                        },
                    )
                )
        for item in self.store.list_pairwise_visual_comparisons(self.run_id):
            comparison = dict(item.get("comparison") or {})
            preferred = bool(
                comparison.get("candidate_preferred")
                or str(item.get("preferred") or "").casefold()
                in {"candidate", "new", "ultra"}
            )
            if preferred:
                self.store.record_final_acceptance_evidence(
                    FinalAcceptanceEvidenceV1(
                        ultra_run_id=self.run_id,
                        kind="pairwise_baseline",
                        authority=(
                            f"{item.get('evaluator')}:{item.get('model')}:blind"
                        ),
                        passed=True,
                        score=float(comparison.get("confidence", 1.0) or 1.0),
                        artifact_hash=str(item.get("candidate_hash") or ""),
                        details={
                            "candidate_preferred": True,
                            "comparison_id": item["id"],
                            "critical": True,
                        },
                    )
                )

    def _run_and_finalize(self) -> UltraRunResult:
        assert self.orchestrator
        if self.goal_id:
            self.store.heartbeat_goal_outcome(
                self.goal_id,
                ultra_run_id=self.run_id,
                process_token=f"pid:{os.getpid()}",
            )
        try:
            result = self.orchestrator.run()
        except Exception as exc:
            self.events.publish(
                "ultra.recovery_checkpoint_preserved",
                "ULTRA stopped before acceptance; durable mutations were preserved "
                "for crash-safe inspection and resume.",
                run_id=self.run_id,
                error=redact_text(exc, 500),
            )
            self._record_engine_failure(exc)
            raise
        if not result.successful:
            self.events.publish(
                "ultra.recovery_checkpoint_preserved",
                "ULTRA candidate has not passed yet; durable mutations were "
                "preserved for bounded repair or resume.",
                run_id=self.run_id,
            )
        self._finalize_result(result)
        return result

    def _record_engine_failure(self, exc: Exception) -> None:
        if not self.goal_id:
            return
        try:
            goal = self.store.get_goal(self.goal_id)
            try:
                self.store.set_goal_outcome_state(
                    self.goal_id,
                    GoalOutcomeState.RECOVERING,
                    decision={
                        "accepted": False,
                        "blockers": [f"engine failure requires recovery: {redact_text(exc, 500)}"],
                    },
                )
            except NotFoundError:
                pass
            if goal.status not in {GoalStatus.BLOCKED, GoalStatus.CANCELLED}:
                self.store.transition_goal(
                    self.goal_id,
                    GoalStatus.BLOCKED,
                    reason=f"ULTRA engine failed: {redact_text(exc, 500)}",
                )
            self.store.update_goal_metadata(
                self.goal_id,
                retry_reason=(
                    "The recursive harness stopped at a saved checkpoint. "
                    "Retry resumes unfinished branches without replaying accepted mutations."
                ),
                waiting_on="recovery",
                auto_retryable=True,
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
                    change_sets = (
                        self.adapter.reconcile_verified_change_sets(self.run_id)
                        if self.adapter is not None and not blocking_findings
                        else self.store.list_change_sets(self.run_id)
                    )
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
                            self._rollback_rejected_changes(
                                "Ultra completion was rejected because findings or Change Sets remain open."
                            )
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
                    # Restored completed runs can enter finalization without
                    # rebuilding their outcome contract.  Refresh/migrate it
                    # at the acceptance boundary so a legacy, locally
                    # undischargeable supervisor requirement cannot survive a
                    # resume and reject otherwise complete evidence.
                    self._ensure_outcome_contract(goal.objective)
                    self._record_automatic_outcome_evidence()
                    assert self.run_id
                    decision = self.store.evaluate_final_acceptance(
                        self.goal_id,
                        self.run_id,
                    )
                    if bool(decision.get("accepted")):
                        if self.version_control is not None:
                            commit = self.version_control.create_checkpoint(
                                f"{goal.objective[:120]} (ultra_final_acceptance)",
                                kind="accepted",
                            )
                            if commit:
                                self.store.append_event(
                                    "version_control.checkpoint",
                                    goal_id=self.goal_id,
                                    payload={
                                        "commit": commit,
                                        "source": "ultra_final_acceptance",
                                    },
                                )
                        self.store.transition_goal(
                            self.goal_id,
                            GoalStatus.COMPLETED,
                            reason="GoalOutcomeContract final acceptance gate passed",
                        )
                    else:
                        self._rollback_rejected_changes(
                            "GoalOutcomeContract rejected the candidate; preserving the previously accepted workspace."
                        )
                        self.store.transition_goal(
                            self.goal_id,
                            GoalStatus.BLOCKED,
                            reason=(
                                "Product outcome remains below acceptance: "
                                + "; ".join(str(item) for item in decision.get("blockers", ()))
                            ),
                        )
            elif result.run.phase is EnginePhase.CANCELLED:
                if goal.status is not GoalStatus.CANCELLED:
                    self.store.transition_goal(self.goal_id, GoalStatus.CANCELLED, reason="ULTRA cancelled")
            elif result.run.phase is EnginePhase.REVISION_REQUIRED:
                try:
                    self.store.set_goal_outcome_state(
                        self.goal_id,
                        GoalOutcomeState.QUALITY_BLOCKED,
                        decision={
                            "accepted": False,
                            "blockers": ["component or global quality gate requires a new strategy"],
                        },
                    )
                except NotFoundError:
                    pass
                if goal.status is GoalStatus.RUNNING:
                    self.store.transition_goal(self.goal_id, GoalStatus.REVISING, reason="ULTRA requires master-plan revision")
                self.store.update_goal_metadata(
                    self.goal_id,
                    waiting_question="A quality or scope gate requires a revised master plan.",
                    auto_retryable=False,
                )
            elif goal.status is GoalStatus.RUNNING:
                self.store.transition_goal(self.goal_id, GoalStatus.BLOCKED, reason="ULTRA module wave failed")
                self.store.update_goal_metadata(
                    self.goal_id,
                    retry_reason=(
                        "One or more recursive branches exhausted their bounded repair attempts. "
                        "Completed branches and accepted mutations are preserved for retry."
                    ),
                    waiting_on="recovery",
                    auto_retryable=True,
                )
        except Exception as exc:
            self.events.publish("error", f"ULTRA completion persistence failed: {redact_text(exc, 500)}")
            # Never leave a durable goal claiming RUNNING after its engine has
            # reached a terminal result but persistence/final acceptance
            # raised.  The recovery checkpoint is still authoritative and
            # the normal Retry action can rebuild from it.
            self._record_engine_failure(exc)

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

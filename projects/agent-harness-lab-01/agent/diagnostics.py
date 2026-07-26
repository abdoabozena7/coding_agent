"""Evidence-driven error signatures and bounded repair decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import re
import urllib.request
from typing import Any, Mapping

from .hardware import HardwareProbeResult, probe_local_gpu
from .model_catalog import DEFAULT_OLLAMA_HOST


class FailureDomain(str, Enum):
    SYNTAX = "syntax"; IMPORT = "import_or_dependency"; RUNTIME = "runtime_exception"
    TEST = "test_assertion"; TYPE = "type_or_contract"; BROWSER = "browser_console"
    RENDERING = "rendering"; NETWORK = "network"; PROVIDER = "provider_protocol"
    TOOL = "tool_availability"; PERMISSION = "permission"; SANDBOX = "sandbox"
    STALE_FILE = "stale_file"; FILE_CONFLICT = "file_conflict"; MODEL_OUTPUT = "invalid_model_output"
    QUALITY_REGRESSION = "quality_regression"


@dataclass(frozen=True, slots=True)
class ErrorSignature:
    domain: FailureDomain
    operation: str
    command: str = ""
    exit_code: int | None = None
    http_status: int | None = None
    exception_type: str = ""
    stack_frames: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    normalized_message: str = ""
    file_hashes: Mapping[str, str] = None
    change_set_id: str | None = None

    @property
    def fingerprint(self) -> str:
        value = {"domain": self.domain.value, "operation": self.operation, "command": self.command, "exit": self.exit_code,
                 "http": self.http_status, "exception": self.exception_type, "frames": self.stack_frames[:5],
                 "paths": self.paths, "message": self.normalized_message, "hashes": dict(self.file_hashes or {}), "change_set": self.change_set_id}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalize_error_message(message: str) -> str:
    value = re.sub(r"0x[0-9a-f]+", "<addr>", str(message), flags=re.I)
    value = re.sub(r"\b\d+(?:\.\d+)?(?:ms|s)?\b", "<n>", value)
    return re.sub(r"\s+", " ", value).strip()


class RepairHistory:
    def __init__(self, limit: int = 3): self.limit, self.attempts = limit, {}
    def record(self, signature: ErrorSignature, approach: str) -> str:
        key = (signature.fingerprint, approach)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return "different_approach_required" if self.attempts[key] >= self.limit else "repair"


@dataclass(frozen=True, slots=True)
class CapabilityCheck:
    name: str
    passed: bool
    evidence: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentReadinessReport:
    passed: bool
    require_gpu: bool
    gpu: HardwareProbeResult
    checks: tuple[CapabilityCheck, ...] = field(default_factory=tuple)

    @property
    def failed_checks(self) -> tuple[CapabilityCheck, ...]:
        return tuple(item for item in self.checks if not item.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "require_gpu": self.require_gpu,
            "gpu": self.gpu.to_dict(),
            "checks": [item.to_dict() for item in self.checks],
            "failed_checks": [item.to_dict() for item in self.failed_checks],
        }


def readiness_report_benchmark_payload(report: AgentReadinessReport) -> dict[str, Any]:
    """Convert a readiness report into stable benchmark-run fields.

    The diagnostics layer can run as a one-off doctor command or as a recurring
    quality gate.  This payload keeps both uses comparable by recording the
    same metrics, scores, and failure context for structural, behavioral, and
    live model probes.
    """

    total_checks = len(report.checks)
    failed_checks = report.failed_checks
    passed_checks = total_checks - len(failed_checks)
    pass_ratio = (passed_checks / total_checks) if total_checks else 0.0
    failed_summaries = tuple(
        {
            "name": item.name,
            "message": item.message,
            "missing": item.missing,
        }
        for item in failed_checks
    )
    blocker = "; ".join(
        f"{item.name}: {item.message or 'failed'}" for item in failed_checks
    ) or None
    return {
        "inputs": {
            "require_gpu": report.require_gpu,
            "gpu": report.gpu.to_dict(),
            "checks": [item.to_dict() for item in report.checks],
            "failed_checks": failed_summaries,
        },
        "metrics": {
            "checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": len(failed_checks),
            "gpu_available": 1 if report.gpu.gpu_available else 0,
            "gpu_required": 1 if report.require_gpu else 0,
        },
        "scores": {
            "pass_ratio": pass_ratio,
            "all_passed": 1.0 if report.passed else 0.0,
        },
        "result": "passed" if report.passed else "failed",
        "blocker": blocker,
    }


def record_agent_readiness_report(
    store: Any,
    report: AgentReadinessReport,
    *,
    suite_name: str = "agent-readiness",
    scenario_name: str = "diagnostics",
    provider: str = "diagnostics",
    model: str = "agent",
    artifact_refs: tuple[str, ...] = (),
    ultra_run_id: str | None = None,
) -> dict[str, Any]:
    """Persist a readiness/diagnostic report into ``benchmark_runs``."""

    payload = readiness_report_benchmark_payload(report)
    return store.record_benchmark_result(
        suite_name=suite_name,
        scenario_name=scenario_name,
        provider=provider,
        model=model,
        inputs=payload["inputs"],
        metrics=payload["metrics"],
        scores=payload["scores"],
        result=payload["result"],
        artifact_refs=artifact_refs,
        ultra_run_id=ultra_run_id,
        blocker=payload["blocker"],
    )


def _has_attrs(target: object, *names: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present = tuple(name for name in names if hasattr(target, name))
    missing = tuple(name for name in names if not hasattr(target, name))
    return present, missing


def _capability(name: str, target: object, required: tuple[str, ...], *, message: str) -> CapabilityCheck:
    present, missing = _has_attrs(target, *required)
    return CapabilityCheck(
        name=name,
        passed=not missing,
        evidence=present,
        missing=missing,
        message=message if not missing else f"{message}; missing: {', '.join(missing)}",
    )


def audit_agent_readiness(
    *,
    require_gpu: bool = False,
    environ: Mapping[str, str] | None = None,
) -> AgentReadinessReport:
    """Audit the weak-model agent architecture against the critical capability map.

    This check is intentionally structural and deterministic.  It verifies that
    the runtime contains the orchestration surfaces that keep small local models
    from acting alone: code graphs, hybrid retrieval, explicit reasoning
    artifacts, swarm consensus, durable learning, and benchmark gates.  When
    ``require_gpu`` is true, it also fails closed if a supported local GPU is not
    detected.
    """

    from . import evaluation, reasoning
    from .project_brain import ProjectBrain
    from .repository_index import EmbeddingProvider, RepositoryIndex
    from .store import StateStore
    from .swarm_coordinator import SwarmCoordinator
    from .swarm_protocol import SwarmMessageV1
    from .ultra_session import StateStoreUltraAdapter

    gpu = probe_local_gpu(environ)
    checks = [
        _capability(
            "code_understanding",
            RepositoryIndex,
            (
                "dependency_graph",
                "resolved_dependency_graph",
                "resolved_call_graph",
                "ownership_graph",
                "semantic_map",
                "context_slice",
                "save_cache",
                "load_cache",
            ),
            message="RepositoryIndex must understand structure beyond regex search",
        ),
        _capability(
            "retrieval",
            RepositoryIndex,
            ("embedding_search", "search_with_scores", "hybrid_search", "context_slice"),
            message="Retrieval must combine lexical, semantic/embedding, hybrid, and focused context slicing",
        ),
        CapabilityCheck(
            name="embedding_provider",
            passed=hasattr(EmbeddingProvider, "__mro_entries__") or hasattr(EmbeddingProvider, "__dict__"),
            evidence=("EmbeddingProvider",),
            message="Repository retrieval exposes an embedding-provider protocol",
        ),
        _capability(
            "reasoning_harness",
            reasoning,
            ("reasoning_scaffold_for", "reasoning_debate_protocol_for", "evaluate_reasoning_artifact"),
            message="Critical phases require external reasoning/debate artifacts instead of hidden CoT trust",
        ),
        _capability(
            "evaluation_benchmarks",
            evaluation,
            (
                "run_repository_retrieval_benchmark",
                "record_repository_retrieval_benchmark",
                "run_single_file_3d_html_benchmark",
                "record_benchmark_trend",
                "learn_from_benchmark_trend",
            ),
            message="Quality must be measured with benchmarks and trend/regression records",
        ),
        _capability(
            "swarm_intelligence",
            SwarmCoordinator,
            ("propose", "submit_vote"),
            message="Agents need formal proposal, voting, and decision workflow",
        ),
        _capability(
            "communication_protocol",
            SwarmMessageV1,
            (
                "encode_frame",
                "decode_frame",
                "encode_dsl_frame",
                "decode_dsl_frame",
                "encode_binary_frame",
                "decode_binary_frame",
                "decode_any_frame",
            ),
            message="Agent communication must support JSON, compact DSL, and checksummed binary frames",
        ),
        _capability(
            "knowledge_and_learning",
            ProjectBrain,
            ("record_knowledge", "build_context", "write_back_result"),
            message="Project Brain must persist knowledge and feed it back into context",
        ),
        _capability(
            "project_memory_confidence",
            StateStore,
            ("promote_brain_entry_to_project_memory", "search_project_memory", "record_project_memory_outcome"),
            message="Cross-run memory must support confidence scoring and outcome-based learning",
        ),
        _capability(
            "ultra_evaluation_learning",
            StateStoreUltraAdapter,
            (
                "foundation_project_lessons",
                "record_global_evaluation_gate",
                "_record_global_remediation_knowledge",
            ),
            message="ULTRA must inject cross-run lessons and record remediation knowledge from failed gates",
        ),
    ]
    if require_gpu:
        checks.append(
            CapabilityCheck(
                name="required_local_gpu",
                passed=gpu.gpu_available,
                evidence=tuple(device.get("name", "") for device in gpu.devices if device.get("name")),
                missing=() if gpu.gpu_available else ("usable_local_gpu",),
                message=gpu.message or "GPU-required mode needs a usable local GPU",
            )
        )
    passed = all(item.passed for item in checks)
    return AgentReadinessReport(
        passed=passed,
        require_gpu=require_gpu,
        gpu=gpu,
        checks=tuple(checks),
    )


def _behavioral_check(name: str, callback: Any) -> CapabilityCheck:
    try:
        evidence = callback()
    except Exception as exc:
        return CapabilityCheck(
            name=name,
            passed=False,
            missing=(type(exc).__name__,),
            message=str(exc),
        )
    if isinstance(evidence, CapabilityCheck):
        return evidence
    items = tuple(str(item) for item in (evidence or ()))
    return CapabilityCheck(name=name, passed=True, evidence=items)


def benchmark_agent_readiness(
    *,
    require_gpu: bool = False,
    environ: Mapping[str, str] | None = None,
) -> AgentReadinessReport:
    """Run deterministic behavioral probes for the critical agent architecture.

    Unlike :func:`audit_agent_readiness`, this creates tiny throwaway projects
    and state records to prove that the orchestration pieces actually cooperate:
    repository graph retrieval, external reasoning-graph enforcement, swarm
    proposal/vote/decision flow, and evaluation-to-cross-run-learning.
    """

    from pathlib import Path
    import tempfile

    from .evaluation import RetrievalBenchmarkCase, run_repository_retrieval_benchmark
    from .model_catalog import ExecutionClass as CatalogExecutionClass, ModelDescriptor
    from .models import GoalStatus, RoleProfile, Task
    from .project_brain import ProjectBrain
    from .reasoning import evaluate_reasoning_artifact, reasoning_debate_protocol_for
    from .repository_index import RepositoryIndex
    from .sandbox import AccessLevel as PermissionAccessLevel
    from .store import StateStore
    from .swarm_bus import SwarmBus
    from .swarm_coordinator import SwarmCoordinator
    from .swarm_protocol import SwarmMessageType
    from .ultra import ResultPackageV1 as EngineResult, UltraConfig
    from .ultra_models import (
        ArchitectureSpecV1,
        ExecutionClass,
        GoalSpecV1,
        UltraRun,
    )
    from .ultra_session import StateStoreUltraAdapter

    gpu = probe_local_gpu(environ)

    def code_retrieval_probe() -> tuple[str, ...]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            workspace = Path(directory)
            (workspace / "utils.py").write_text(
                "def normalize_score(value):\n"
                "    return max(0, min(100, int(value)))\n",
                encoding="utf-8",
            )
            (workspace / "service.py").write_text(
                "from utils import normalize_score\n\n"
                "class GameService:\n"
                "    def run(self, raw_score):\n"
                "        return normalize_score(raw_score)\n",
                encoding="utf-8",
            )
            index = RepositoryIndex(workspace)
            index.update_all()
            retrieval = run_repository_retrieval_benchmark(
                index,
                (
                    RetrievalBenchmarkCase(
                        query="normalize score value",
                        expected_names=("normalize_score",),
                        k=5,
                    ),
                ),
            )
            call_graph = index.resolved_call_graph()
            slice_result = index.context_slice("GameService run normalize score", max_entries=6)
            has_call = any(
                any("normalize_score" in target for target in targets)
                for targets in call_graph.values()
            )
            if not retrieval.passed or not has_call or not slice_result.entries:
                raise AssertionError(
                    "repository benchmark did not prove retrieval + call graph + context slice"
                )
            return (
                f"retrieval_mrr={retrieval.metrics['mean_reciprocal_rank']:.2f}",
                f"call_graph_sources={len(call_graph)}",
                f"context_entries={len(slice_result.entries)}",
            )

    def reasoning_probe() -> tuple[str, ...]:
        protocol = reasoning_debate_protocol_for(
            "architect",
            "architecture",
            {"contract": {"objective": "Build verified orchestration"}},
        )
        shallow = evaluate_reasoning_artifact({"claim": "looks fine"}, protocol)
        strong = evaluate_reasoning_artifact(
            {
                "claim": "Use benchmark-gated remediation",
                "supporting_evidence": ["benchmark:synthetic"],
                "counterarguments": ["A prose-only pass can hide missing evidence."],
                "rejected_alternatives": ["Accepting model confidence without tests."],
                "verification_plan": ["Run retrieval, reasoning, swarm, and learning probes."],
                "reasoning_graph": {
                    "nodes": [
                        {
                            "id": "chosen",
                            "type": "decision",
                            "summary": "Benchmark-gated remediation",
                            "status": "chosen",
                            "evidence_refs": ["benchmark:synthetic"],
                        },
                        {
                            "id": "rejected",
                            "type": "option",
                            "summary": "Trust prose-only output",
                            "status": "rejected",
                            "evidence_refs": [],
                        },
                    ],
                    "edges": [{"from": "chosen", "to": "rejected", "relation": "rejects"}],
                },
            },
            protocol,
        )
        if shallow.passed or not strong.passed:
            raise AssertionError("reasoning harness did not reject shallow and accept graph artifact")
        return (f"shallow_score={shallow.score}", f"strong_score={strong.score}")

    def _create_minimal_ultra_store(workspace: Path) -> tuple[StateStore, str, str]:
        store = StateStore(workspace)
        goal = store.create_goal("Behavioral readiness benchmark")
        store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        plan = store.create_plan(
            goal.id,
            "Benchmark architecture readiness",
            (
                Task(
                    id="CORE",
                    title="Core benchmark task",
                    description="Exercise readiness architecture",
                    acceptance_criteria=("Readiness probe records observable evidence.",),
                    verification=("Run deterministic diagnostics benchmark.",),
                    role=RoleProfile(name="benchmark", mission="Exercise readiness"),
                ),
            ),
            applicability_evidence=(
                {
                    "fact": "Synthetic benchmark workspace exists",
                    "source": "diagnostics",
                    "supports_tasks": ["CORE"],
                },
            ),
            execution_strategy="Run deterministic readiness probes.",
            expected_changes=(
                {
                    "path": "diagnostics/readiness.json",
                    "intent": "Record deterministic readiness benchmark evidence",
                    "supports_tasks": ["CORE"],
                },
            ),
        )
        plan, _ = store.approve_plan(goal.id, plan.revision, expected_fingerprint=plan.fingerprint)
        run = store.create_ultra_run(
            UltraRun(
                goal_id=goal.id,
                provider="ollama",
                model="gemma4",
                execution_class=ExecutionClass.LOCAL,
                concurrency=1,
                goal_spec=GoalSpecV1(
                    objective="Behavioral readiness benchmark",
                    scope=("diagnostics",),
                    success_criteria=("readiness probes pass",),
                ),
                architecture_spec=ArchitectureSpecV1(
                    summary="Synthetic readiness architecture",
                    components=({"name": "diagnostics"},),
                    interfaces={"probe": {"run": "None -> report"}},
                ),
            )
        )
        run = store.approve_ultra_master(run.id, plan.revision, plan.fingerprint)
        return store, goal.id, run.id

    def swarm_probe() -> tuple[str, ...]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            store, _goal_id, run_id = _create_minimal_ultra_store(Path(directory))
            try:
                bus = SwarmBus(store)
                decisions: list[dict[str, object]] = []
                bus.subscribe(run_id, decisions.append, recipient_agent_id="swarm", message_type=SwarmMessageType.DECISION)
                coordinator = SwarmCoordinator(store, bus)
                workflow = coordinator.propose(
                    ultra_run_id=run_id,
                    proposer_agent_id="planner-1",
                    topic="readiness-consensus",
                    proposal={"gate": "readiness"},
                    voters=("critic-1", "tester-1"),
                    quorum=2,
                    leader_agent_id="planner-1",
                )
                coordinator.submit_vote(
                    round_id=workflow.consensus_round_id,
                    voter_agent_id="critic-1",
                    verdict="accept",
                    confidence=0.8,
                    rationale="Protocol is wired.",
                )
                closed = coordinator.submit_vote(
                    round_id=workflow.consensus_round_id,
                    voter_agent_id="tester-1",
                    verdict="accept",
                    confidence=0.9,
                    rationale="Decision emitted.",
                )
                if closed["status"] != "accepted" or not decisions:
                    raise AssertionError("swarm consensus did not close and publish a decision")
                return (
                    f"round={workflow.consensus_round_id}",
                    f"requests={len(workflow.request_message_ids)}",
                    f"decision={closed['status']}",
                )
            finally:
                store.close()

    def learning_probe() -> tuple[str, ...]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            workspace = Path(directory)
            store, goal_id, run_id = _create_minimal_ultra_store(workspace)
            try:
                brain = ProjectBrain(store, run_id)
                entry = brain.record_knowledge(
                    "Readiness runtime evidence",
                    "Future runs must require durable evidence before completion.",
                    confidence=0.81,
                    evidence_refs=("diagnostic:seed",),
                )
                memory = store.search_project_memory(
                    "durable evidence completion",
                    section="knowledge",
                )[0]
                adapter = StateStoreUltraAdapter(
                    store,
                    goal_id,
                    ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
                    PermissionAccessLevel.NORMAL,
                    UltraConfig(),
                    workspace=workspace,
                )
                adapter.run_id = run_id
                context = adapter.foundation_project_lessons(
                    run_id,
                    "durable evidence completion",
                    phase="master_plan",
                )
                gate = adapter.record_global_evaluation_gate(
                    EngineResult(
                        node_id="global",
                        success=True,
                        summary="claimed completion without durable evidence",
                        evidence=(),
                        test_results=(),
                    ),
                    (),
                )
                if gate["passed"] or not gate.get("remediation_knowledge"):
                    raise AssertionError("failed evaluation did not create remediation knowledge")
                updated = store.search_project_memory("durable evidence completion", section="knowledge", min_confidence=0.0)[0]
                if updated["metadata"].get("negative_outcomes", 0) < 1:
                    raise AssertionError("used project memory was not penalized by failed evaluation")
                return (
                    f"knowledge_entry={entry.id}",
                    f"memory={memory['id']}",
                    f"context_items={len(context)}",
                    f"remediation={gate['remediation_knowledge']['recorded']}",
                )
            finally:
                store.close()

    checks = [
        _behavioral_check("behavioral_code_retrieval", code_retrieval_probe),
        _behavioral_check("behavioral_reasoning", reasoning_probe),
        _behavioral_check("behavioral_swarm_consensus", swarm_probe),
        _behavioral_check("behavioral_learning_evaluation", learning_probe),
    ]
    if require_gpu:
        checks.append(
            CapabilityCheck(
                name="behavioral_required_local_gpu",
                passed=gpu.gpu_available,
                evidence=tuple(device.get("name", "") for device in gpu.devices if device.get("name")),
                missing=() if gpu.gpu_available else ("usable_local_gpu",),
                message=gpu.message or "GPU-required benchmark needs a usable local GPU",
            )
        )
    return AgentReadinessReport(
        passed=all(item.passed for item in checks),
        require_gpu=require_gpu,
        gpu=gpu,
        checks=tuple(checks),
    )


def _ollama_http_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 120.0,
) -> Mapping[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method=method.upper(),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = response.read().decode("utf-8")
    result = json.loads(decoded or "{}")
    if not isinstance(result, Mapping):
        raise ValueError("Ollama returned a non-object JSON response")
    return result


def probe_ollama_model_live(
    model: str,
    *,
    host: str | None = None,
    require_gpu: bool = False,
    environ: Mapping[str, str] | None = None,
    http_json: Any | None = None,
    timeout: float = 120.0,
) -> AgentReadinessReport:
    """Probe a real local Ollama model with a bounded structured-output request."""

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("live Ollama probe requires a model name")
    base_url = str(host or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    caller = http_json or _ollama_http_json
    gpu = probe_local_gpu(environ)

    def structured_json_probe() -> tuple[str, ...]:
        response = caller(
            "POST",
            f"{base_url}/api/chat",
            payload={
                "model": model_name,
                "stream": False,
                "format": "json",
                "think": False,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return exactly valid JSON with keys ok, model, verification. "
                            f"ok must be true and model must be {model_name!r}. No prose."
                        ),
                    }
                ],
                "options": {"num_predict": 192},
            },
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        message = response.get("message")
        content = ""
        if isinstance(message, Mapping):
            content = str(message.get("content") or "")
        parsed = json.loads(content)
        if not isinstance(parsed, Mapping):
            raise ValueError("model JSON content was not an object")
        if parsed.get("ok") is not True:
            raise AssertionError("model did not return ok=true")
        if str(parsed.get("model") or "").strip() != model_name:
            raise AssertionError("model echoed a different model name")
        usage = {
            key: response.get(key)
            for key in ("prompt_eval_count", "eval_count", "total_duration", "load_duration")
            if response.get(key) is not None
        }
        evidence = [
            f"model={model_name}",
            f"json_keys={','.join(sorted(str(key) for key in parsed.keys()))}",
        ]
        for key, value in usage.items():
            evidence.append(f"{key}={value}")
        return tuple(evidence)

    checks = [_behavioral_check("live_ollama_structured_json", structured_json_probe)]
    if require_gpu:
        checks.append(
            CapabilityCheck(
                name="live_required_local_gpu",
                passed=gpu.gpu_available,
                evidence=tuple(device.get("name", "") for device in gpu.devices if device.get("name")),
                missing=() if gpu.gpu_available else ("usable_local_gpu",),
                message=gpu.message or "Live local model probe requires a usable GPU",
            )
        )
    return AgentReadinessReport(
        passed=all(item.passed for item in checks),
        require_gpu=require_gpu,
        gpu=gpu,
        checks=tuple(checks),
    )


def _ollama_structured_probe_payload(model_name: str, *, controlled: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return exactly valid JSON with keys ok, model, verification. "
                    f"ok must be true and model must be {model_name!r}. No prose."
                ),
            }
        ],
        "options": {"num_predict": 192},
    }
    if controlled:
        # Ollama's native chat API expects this at top level.  Putting it under
        # options lets some thinking models spend the entire output budget in a
        # hidden/native thinking field and return empty content.
        payload["think"] = False
    return payload


def _parse_live_ollama_structured_response(response: Mapping[str, Any], model_name: str) -> tuple[bool, tuple[str, ...], str]:
    message = response.get("message")
    content = str(message.get("content") or "") if isinstance(message, Mapping) else ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        thinking = str(message.get("thinking") or "") if isinstance(message, Mapping) else ""
        return (
            False,
            (
                f"json_error={type(exc).__name__}",
                f"content_chars={len(content)}",
                f"thinking_chars={len(thinking)}",
                f"done_reason={response.get('done_reason', '')}",
            ),
            str(exc),
        )
    if not isinstance(parsed, Mapping):
        return False, ("json_type=non_object",), "model JSON content was not an object"
    if parsed.get("ok") is not True:
        return False, (f"ok={parsed.get('ok')!r}",), "model did not return ok=true"
    if str(parsed.get("model") or "").strip() != model_name:
        return False, (f"model={parsed.get('model')!r}",), "model echoed a different model name"
    usage = tuple(
        f"{key}={response.get(key)}"
        for key in ("prompt_eval_count", "eval_count", "total_duration", "load_duration")
        if response.get(key) is not None
    )
    return (
        True,
        (
            f"json_keys={','.join(sorted(str(key) for key in parsed.keys()))}",
            *usage,
        ),
        "",
    )


def probe_ollama_orchestration_delta_live(
    model: str,
    *,
    host: str | None = None,
    require_gpu: bool = False,
    environ: Mapping[str, str] | None = None,
    http_json: Any | None = None,
    timeout: float = 120.0,
) -> AgentReadinessReport:
    """Compare raw Ollama structured output against the agent-controlled request.

    This is a small live proof that weak-model orchestration is not decorative:
    the controlled request must pass, and the report records whether raw output
    failed or already passed.
    """

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("live Ollama delta probe requires a model name")
    base_url = str(host or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    caller = http_json or _ollama_http_json
    gpu = probe_local_gpu(environ)

    def delta_probe() -> CapabilityCheck:
        raw_response = caller(
            "POST",
            f"{base_url}/api/chat",
            payload=_ollama_structured_probe_payload(model_name, controlled=False),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        raw_passed, raw_evidence, raw_error = _parse_live_ollama_structured_response(
            raw_response,
            model_name,
        )
        controlled_response = caller(
            "POST",
            f"{base_url}/api/chat",
            payload=_ollama_structured_probe_payload(model_name, controlled=True),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        controlled_passed, controlled_evidence, controlled_error = _parse_live_ollama_structured_response(
            controlled_response,
            model_name,
        )
        evidence = (
            f"model={model_name}",
            f"raw_passed={raw_passed}",
            *tuple(f"raw:{item}" for item in raw_evidence),
            f"controlled_passed={controlled_passed}",
            *tuple(f"controlled:{item}" for item in controlled_evidence),
            f"orchestration_improved={bool((not raw_passed) and controlled_passed)}",
        )
        if not controlled_passed:
            return CapabilityCheck(
                name="live_ollama_orchestration_delta",
                passed=False,
                evidence=evidence,
                missing=("controlled_structured_json",),
                message=controlled_error or "controlled Ollama request did not produce valid structured JSON",
            )
        return CapabilityCheck(
            name="live_ollama_orchestration_delta",
            passed=True,
            evidence=evidence,
            message=("raw failed and controlled request passed" if not raw_passed else "raw and controlled requests both passed"),
        )

    checks = [_behavioral_check("live_ollama_orchestration_delta", delta_probe)]
    if require_gpu:
        checks.append(
            CapabilityCheck(
                name="live_delta_required_local_gpu",
                passed=gpu.gpu_available,
                evidence=tuple(device.get("name", "") for device in gpu.devices if device.get("name")),
                missing=() if gpu.gpu_available else ("usable_local_gpu",),
                message=gpu.message or "Live local model delta probe requires a usable GPU",
            )
        )
    return AgentReadinessReport(
        passed=all(item.passed for item in checks),
        require_gpu=require_gpu,
        gpu=gpu,
        checks=tuple(checks),
    )


def probe_ollama_html_microtask_live(
    model: str,
    *,
    host: str | None = None,
    require_gpu: bool = False,
    require_quality: bool = False,
    refine_attempts: int = 0,
    environ: Mapping[str, str] | None = None,
    http_json: Any | None = None,
    timeout: float = 180.0,
) -> AgentReadinessReport:
    """Ask a live Ollama model for a tiny single-file 3D HTML artifact and benchmark it.

    By default this proves that the model output can be captured and evaluated;
    set ``require_quality=True`` to turn the HTML benchmark threshold into a
    strict pass/fail gate.
    """

    from .evaluation import run_single_file_3d_html_benchmark

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("live Ollama HTML microtask probe requires a model name")
    base_url = str(host or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    caller = http_json or _ollama_http_json
    gpu = probe_local_gpu(environ)

    def html_probe() -> CapabilityCheck:
        def request_html(prompt: str) -> tuple[str, Mapping[str, Any]]:
            response = caller(
                "POST",
                f"{base_url}/api/chat",
                payload={
                    "model": model_name,
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"num_predict": 4096},
                },
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
            message = response.get("message")
            content = str(message.get("content") or "") if isinstance(message, Mapping) else ""
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{exc}; content_chars={len(content)}; "
                    f"thinking_chars={len(str(message.get('thinking') or '')) if isinstance(message, Mapping) else 0}"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise ValueError("HTML microtask JSON was not an object")
            return str(parsed.get("html") or ""), response

        base_prompt = (
            "Return exactly one JSON object with keys ok, model, html. "
            f"ok must be true and model must be {model_name!r}. "
            "html must be a complete single-file 3D browser game demo using canvas/WebGL or THREE-style code. "
            "Do not use any external URLs, CDNs, imports, images, fonts, scripts, stylesheets, src, or href resources; all code and styles must be inline. "
            "Include explicit THREE.Scene or scene setup, PerspectiveCamera/camera, WebGLRenderer or webgl context, "
            "player input, animation loop, score/health HUD, enemies/projectiles/collision words, lighting/materials, "
            "fog/particles/trails/bloom or equivalent visual richness, responsive resize handling, and no markdown."
        )
        attempts: list[tuple[str, Any]] = []
        try:
            html, _response = request_html(base_prompt)
        except ValueError as exc:
            return CapabilityCheck(
                name="live_ollama_html_microtask",
                passed=False,
                evidence=(f"model={model_name}",),
                missing=("valid_json_artifact",),
                message=str(exc),
            )
        benchmark = run_single_file_3d_html_benchmark(html)
        attempts.append((html, benchmark))

        for attempt in range(max(0, int(refine_attempts))):
            if benchmark.passed:
                break
            refinement_prompt = (
                base_prompt
                + "\nThe previous candidate failed the automatic benchmark. "
                + "Fix these blocker findings exactly: "
                + "; ".join(benchmark.findings or ("quality score below threshold",))
                + ". Current scores: "
                + json.dumps(dict(benchmark.scores), sort_keys=True)
                + ". Return a fresh complete HTML file in the html field, not a patch."
            )
            try:
                candidate_html, _response = request_html(refinement_prompt)
            except ValueError:
                continue
            candidate_benchmark = run_single_file_3d_html_benchmark(candidate_html)
            attempts.append((candidate_html, candidate_benchmark))
            if (
                candidate_benchmark.passed
                and not benchmark.passed
                or candidate_benchmark.scores.get("overall", 0.0) >= benchmark.scores.get("overall", 0.0)
            ):
                html, benchmark = candidate_html, candidate_benchmark

        best_index = max(
            range(len(attempts)),
            key=lambda index: (
                1 if attempts[index][1].passed else 0,
                attempts[index][1].scores.get("overall", 0.0),
            ),
        )
        html, benchmark = attempts[best_index]
        initial = attempts[0][1]
        improved = benchmark.scores.get("overall", 0.0) > initial.scores.get("overall", 0.0)
        evidence = (
            f"model={model_name}",
            f"attempts={len(attempts)}",
            f"best_attempt={best_index + 1}",
            f"html_chars={len(html)}",
            f"benchmark_passed={benchmark.passed}",
            f"overall={benchmark.scores.get('overall', 0.0):.3f}",
            f"initial_overall={initial.scores.get('overall', 0.0):.3f}",
            f"refinement_improved={improved}",
            *tuple(f"score:{key}={value:.3f}" for key, value in sorted(benchmark.scores.items())),
            *tuple(f"finding:{item}" for item in benchmark.findings[:8]),
        )
        if require_quality and not benchmark.passed:
            return CapabilityCheck(
                name="live_ollama_html_microtask",
                passed=False,
                evidence=evidence,
                missing=("html_quality_threshold",),
                message="live HTML microtask did not pass the single-file 3D benchmark",
            )
        return CapabilityCheck(
            name="live_ollama_html_microtask",
            passed=True,
            evidence=evidence,
            message=("HTML benchmark passed" if benchmark.passed else "HTML benchmark executed and caught quality gaps"),
        )

    checks = [_behavioral_check("live_ollama_html_microtask", html_probe)]
    if require_gpu:
        checks.append(
            CapabilityCheck(
                name="live_html_required_local_gpu",
                passed=gpu.gpu_available,
                evidence=tuple(device.get("name", "") for device in gpu.devices if device.get("name")),
                missing=() if gpu.gpu_available else ("usable_local_gpu",),
                message=gpu.message or "Live HTML microtask requires a usable GPU",
            )
        )
    return AgentReadinessReport(
        passed=all(item.passed for item in checks),
        require_gpu=require_gpu,
        gpu=gpu,
        checks=tuple(checks),
    )

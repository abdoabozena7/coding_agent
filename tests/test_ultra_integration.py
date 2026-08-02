from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import pytest

from agent.commands import parse_command
from agent.events import EventBus
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.models import GoalStatus, TaskStatus
from agent.providers.base import AssistantTurn, ToolCall, Usage
from agent.quality import ChangeSetStatus, ChangeSetV1
from agent.runtime import AgentRuntime, RuntimeStateError
from agent.sandbox import AccessLevel, DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.ultra import (
    ApprovalRequiredError,
    AgentProtocolError,
    AgentRequest,
    AgentResponse,
    AgentRole,
    InnerPhase,
    NodeStatus,
    UltraConfig,
    WorkNode,
    ArchitectureSpecV1 as EngineArchitectureSpec,
    ContextRequest,
    GoalSpecV1 as EngineGoalSpec,
    UltraRunV1 as EngineUltraRun,
)
from agent.ultra import MasterPlanV1, TaskContractV1, UltraOrchestrator, _with_quality_milestone
from agent.ultra_models import BrainSection, UltraPhase, UltraRunStatus
from agent.ultra_session import (
    UltraSession,
    WorkspaceUltraAgent,
    WorkspaceUltraAgentFactory,
    StateStoreUltraAdapter,
    DurableContextBuilder,
    _bind_accepted_repair_feedback,
    _bind_repair_architecture_authority,
    _explicit_repair_scope_paths,
    _repair_double_escaped_python_source,
    _run_python_test_artifacts,
    _normalize_tester_verification_call,
    _tester_command_receipt,
    _store_node_status,
    _validate_workspace_artifacts,
)
from agent.ultra_models import WorkNodeStatus
from agent.ui_state import ApprovalDecision


def test_ultra_planning_is_not_persisted_as_execution_in_progress():
    assert _store_node_status(NodeStatus.PLANNING) is WorkNodeStatus.PENDING


def test_accepted_repair_feedback_survives_lossy_small_model_plan_wording():
    plan = MasterPlanV1(
        summary="Verify the app.",
        modules=(
            TaskContractV1(
                id="M001",
                title="Final verification",
                objective="Verify the current app.",
                acceptance_criteria=("The app works.",),
                verification=("preview_html src/index.html",),
                write_paths=("src/index.html",),
            ),
        ),
    )
    feedback = "Remove the duplicate result and add keyboard focus states."

    bound = _bind_accepted_repair_feedback(plan, feedback)

    assert feedback in bound.modules[0].objective
    assert bound.modules[0].metadata["accepted_repair_feedback"] == feedback
    assert bound.modules[0].metadata["accepted_repair_feedback_bound"] is True


def test_typed_browser_scenarios_survive_lossy_small_model_plan_wording():
    plan = MasterPlanV1(
        summary="Verify the app.",
        modules=(
            TaskContractV1(
                id="M001",
                title="Verify",
                objective="Open the page.",
                acceptance_criteria=("No console errors.",),
                verification=("preview_html src/index.html",),
                write_paths=("src/index.html",),
            ),
        ),
    )
    feedback = (
        'Put this in metadata.browser_scenarios: [{"name":"addition",'
        '"steps":[{"action":"click","role":"button","name":"7"}],'
        '"assertions":[{"role":"textbox","name":"Display",'
        '"property":"value","equals":"7"}]}].'
    )

    bound = _bind_accepted_repair_feedback(plan, feedback)

    assert bound.modules[0].metadata["browser_scenarios"][0]["name"] == "addition"


def test_plan_projection_preserves_approval_bound_module_metadata():
    scenario = {
        "name": "addition",
        "steps": [{"action": "click", "role": "button", "name": "7"}],
        "assertions": [
            {"role": "textbox", "name": "Display", "property": "value", "equals": "7"}
        ],
    }
    master = MasterPlanV1(
        summary="Verify",
        modules=(
            TaskContractV1(
                id="M001",
                title="Verify",
                objective="Verify interactions.",
                acceptance_criteria=("Click 7 and display 7.",),
                verification=("preview_html src/index.html",),
                write_paths=("src/index.html",),
                metadata={"browser_scenarios": [scenario]},
            ),
        ),
    )
    adapter = object.__new__(StateStoreUltraAdapter)

    tasks, _changes = adapter._plan_payload(master)

    assert tasks[0]["metadata"]["browser_scenarios"] == [scenario]


def test_latest_repair_feedback_overrides_stale_architecture_detail():
    from agent.ultra import ArchitectureSpecV1

    architecture = ArchitectureSpecV1(
        summary="Use React for the calculator UI.",
        components=({"name": "calculator"},),
        decisions=({"decision": "Use React."},),
        invariants=("Keep the React component split.",),
    )
    feedback = "Use one self-contained vanilla JavaScript HTML file."

    bound = _bind_repair_architecture_authority(architecture, feedback)

    assert feedback in bound.summary
    assert feedback in bound.invariants[-1]
    assert bound.decisions[-1]["status"] == "accepted_revision_authority"


def test_explicit_repair_scope_narrows_an_older_broader_plan_scope():
    feedback = (
        "Reject the broad revision. Keep exactly src/index.html. "
        "No extra files or dependencies."
    )

    assert _explicit_repair_scope_paths(feedback) == ("src/index.html",)


def test_repair_scope_parser_does_not_infer_paths_without_exact_authority():
    feedback = "Inspect src/index.html and decide whether package.json also needs a change."

    assert _explicit_repair_scope_paths(feedback) == ()


def test_bounded_repair_context_omits_stale_cross_revision_memory():
    contract = TaskContractV1(
        id="M001",
        title="Repair calculator",
        objective=(
            "Fix the current calculator.\n\nAccepted repair requirements:\n"
            "Keep exactly src/index.html and make 7+5 equal 12."
        ),
        acceptance_criteria=("7+5=12",),
        verification=("preview_html src/index.html",),
        write_paths=("src/index.html",),
    )
    node = WorkNode(contract=contract)
    goal = EngineGoalSpec("Repair calculator", ("7+5=12",))
    architecture = EngineArchitectureSpec(
        summary="Stale React architecture that must not override repair authority.",
        components=({"name": "old-react-ui"},),
    )
    plan = MasterPlanV1(summary="Repair", modules=(contract,))
    request = ContextRequest(
        run=EngineUltraRun("Repair calculator", ExecutionClass.LOCAL),
        node=node,
        role=AgentRole.CODER,
        goal=goal,
        architecture=architecture,
        plan=plan,
        nodes={node.id: node},
        brain=(),
        dependency_results={},
    )

    context = DurableContextBuilder(mock.Mock(), lambda: "run-1", 43_000).build(request)

    assert context["repair_contract"]["write_paths"] == ("src/index.html",)
    assert "architecture_contract" not in context
    assert "project_lessons" in context["_omitted"]
    assert "React" not in json.dumps(context)


def test_artifact_receipt_requires_a_real_hash_change_from_lease_baseline(tmp_path):
    target = tmp_path / "src" / "index.html"
    target.parent.mkdir(parents=True)
    target.write_text("before", encoding="utf-8")
    before = hashlib.sha256(b"before").hexdigest()
    adapter = object.__new__(StateStoreUltraAdapter)
    adapter.workspace = tmp_path
    adapter._adapter_lock = threading.RLock()
    adapter._lease_scopes = {"M001": ("src/index.html",)}
    adapter._lease_initial_hashes = {"M001": {"src/index.html": before}}
    node = mock.Mock(id="M001", write_paths=("src/index.html",))
    passed = {"passed": True, "findings": [], "evidence": []}

    with mock.patch("agent.ultra_session._validate_workspace_artifacts", return_value=passed), \
         mock.patch("agent.ultra_session._run_python_test_artifacts", return_value=passed):
        unchanged = adapter.verify_node_artifacts("run", node)
        target.write_text("after", encoding="utf-8")
        changed = adapter.verify_node_artifacts("run", node)

    assert unchanged["passed"] is True
    assert unchanged["workspace_mutated"] is False
    assert changed["workspace_mutated"] is True
    assert changed["mutation_evidence"][0]["before_sha256"] == before


def test_ultra_restore_maps_every_durable_work_node_status():
    mapped = {
        status: UltraSession._engine_node_status(status)
        for status in WorkNodeStatus
    }

    assert set(mapped) == set(WorkNodeStatus)
    assert mapped[WorkNodeStatus.CONFLICT] is NodeStatus.CONFLICT


def test_double_escaped_python_fallback_is_repaired_only_when_syntax_proves_it():
    repaired, changed = _repair_double_escaped_python_source(
        "formatter.py",
        'def format_sum(a, b):\n    \\"\\"\\"Add values.\\"\\"\\"\n    return f\\"{a + b}\\"\n',
    )
    assert changed
    assert repaired == (
        'def format_sum(a, b):\n    """Add values."""\n    return f"{a + b}"\n'
    )


def test_tester_cd_verifier_is_normalized_to_typed_cwd_without_shell_expansion():
    normalized, receipt = _normalize_tester_verification_call(
        ToolCall(
            "verify",
            "run_bash",
            {"command": "cd workspace && npm test"},
        )
    )
    assert normalized.name == "run_command"
    assert normalized.args == {"cwd": "workspace", "command": "npm test"}
    assert "normalized" in receipt

    unsafe, unsafe_receipt = _normalize_tester_verification_call(
        ToolCall(
            "unsafe",
            "run_bash",
            {"command": "cd ../outside && npm test"},
        )
    )
    assert unsafe.name == "run_bash"
    assert unsafe_receipt == ""

    invalid, invalid_changed = _repair_double_escaped_python_source(
        "formatter.py",
        'def broken(:\n    return \\"still broken\\"\n',
    )
    assert not invalid_changed
    assert '\\"still broken\\"' in invalid

    javascript, javascript_changed = _repair_double_escaped_python_source(
        "app.js",
        'const value = \\"literal\\";\n',
    )
    assert not javascript_changed
    assert javascript == 'const value = \\"literal\\";\n'


def test_tester_command_receipt_uses_only_authoritative_exit_status():
    passed = _tester_command_receipt(
        ToolCall("verify", "run_command", {"cwd": "web", "command": "npm test"}),
        "exit code: 0\nstdout:\n12 tests passed\n",
        node_id="M001",
    )
    assert passed is not None
    assert passed.provider == "harness"
    assert passed.model == "command-receipt-v1"
    assert passed.payload["passed"] is True
    assert passed.payload["test_results"][0]["returncode"] == 0

    failed = _tester_command_receipt(
        ToolCall("verify", "run_bash", {"command": "python -m pytest -q"}),
        "exit code: 1\nstdout:\n1 failed\n",
    )
    assert failed is not None
    assert failed.payload["passed"] is False
    assert "exit code 1" in failed.payload["findings"][0]

    assert (
        _tester_command_receipt(
            ToolCall("verify", "run_bash", {"command": "python -m pytest -q"}),
            "pytest probably passed",
        )
        is None
    )
    assert (
        _tester_command_receipt(
            ToolCall("read", "read_file", {"path": "test.py"}),
            "exit code: 0",
        )
        is None
    )
def test_workspace_artifact_gate_hashes_files_and_rejects_python_syntax_errors():
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "formatter.py").write_text(
            "def format_sum(a, b):\n    return a + b\n",
            encoding="utf-8",
        )
        passed = _validate_workspace_artifacts(workspace, ("formatter.py",))
        assert passed["passed"]
        assert passed["evidence"][0]["python_syntax"] == "passed"
        assert len(passed["evidence"][0]["sha256"]) == 64

        (workspace / "formatter.py").write_text(
            "def format_sum(:\n",
            encoding="utf-8",
        )
        failed = _validate_workspace_artifacts(workspace, ("formatter.py",))
        assert not failed["passed"]
        assert any(
            "Python syntax validation failed" in finding
            for finding in failed["findings"]
        )


def test_workspace_python_test_gate_executes_pytest_without_leaking_cache():
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        test_path = workspace / "tests" / "test_gate.py"
        test_path.parent.mkdir(parents=True)
        test_path.write_text("def test_gate():\n    assert False\n", encoding="utf-8")

        failed = _run_python_test_artifacts(
            workspace,
            ("tests/test_gate.py",),
        )
        assert not failed["passed"]
        assert failed["evidence"][0]["returncode"] != 0

        test_path.write_text("def test_gate():\n    assert True\n", encoding="utf-8")
        passed = _run_python_test_artifacts(
            workspace,
            ("tests/test_gate.py",),
        )
        assert passed["passed"]
        assert passed["evidence"][0]["returncode"] == 0
        assert not (workspace / ".pytest_cache").exists()
        assert not any(workspace.rglob("__pycache__"))


def test_missing_master_plan_is_rejected_without_harness_authored_modules():
    normalized, actions = UltraOrchestrator._normalize_typed_payload(
        "master_plan",
        {},
        {
            "goal_spec": {
                "objective": (
                    "Create mathlib.py and formatter.py, then create "
                    "tests/test_components.py and run pytest."
                ),
                "in_scope": [
                    "mathlib.py",
                    "formatter.py",
                    "tests/test_components.py",
                ],
                "success_criteria": ["tests/test_components.py passes pytest."],
            },
            "architecture": {
                "summary": "Two implementation components plus verification.",
                "components": [
                    {
                        "name": "mathlib.py",
                        "responsibility": "Implement add in mathlib.py.",
                    },
                    {
                        "name": "formatter.py",
                        "responsibility": "Implement format_sum in formatter.py.",
                    },
                ],
            },
            "protocol_node_namespace": "run123",
        },
    )

    assert normalized == {}
    assert actions == ()
    with pytest.raises(AgentProtocolError, match="summary and modules"):
        UltraOrchestrator._validate_typed_response("master_plan", normalized)


def test_final_evidence_missing_boolean_uses_only_authoritative_operational_gate():
    normalized, actions = UltraOrchestrator._normalize_typed_payload(
        "final_evidence",
        {"summary": "All requested behavior is complete."},
        {
            "integration": {"success": True},
            "review": {"passed": True},
            "authoritative_operational_evidence": {
                "kind": "authoritative_operational_evidence",
                "passed": True,
                "hashed_paths": ["module.py", "tests/test_module.py"],
                "executable_checks": [
                    {
                        "kind": "executable_verification",
                        "passed": True,
                        "returncode": 0,
                    }
                ],
            },
        },
    )
    assert normalized["success"]
    assert normalized["passed"]
    assert normalized["evaluator_capability"] == "harness_operational_evidence"
    assert any("authoritative artifact hashes" in item for item in actions)

    rejected, _ = UltraOrchestrator._normalize_typed_payload(
        "final_evidence",
        {"summary": "Claims completion without evidence."},
        {
            "integration": {"success": True},
            "review": {"passed": True},
            "authoritative_operational_evidence": {"passed": False},
        },
    )
    assert "success" not in rejected
    assert "passed" not in rejected
    with pytest.raises(AgentProtocolError, match="must be boolean"):
        UltraOrchestrator._validate_typed_response("final_evidence", rejected)


@pytest.mark.parametrize(
    ("phase", "field", "raw", "expected"),
    [
        ("review", "passed", "true", True),
        ("review", "passed", "FAILED", False),
        ("global_review", "passed", "pass", True),
        ("integrate", "success", "false", False),
    ],
)
def test_exact_string_verdict_is_mechanically_normalized(
    phase: str,
    field: str,
    raw: str,
    expected: bool,
):
    normalized, actions = UltraOrchestrator._normalize_typed_payload(
        phase,
        {field: raw},
        {},
    )

    assert normalized[field] is expected
    assert any("exact string verdict normalized" in item for item in actions)
    UltraOrchestrator._validate_typed_response(phase, normalized)


def test_ambiguous_string_verdict_is_not_invented():
    normalized, actions = UltraOrchestrator._normalize_typed_payload(
        "review",
        {"passed": "looks mostly fine"},
        {},
    )

    assert normalized["passed"] == "looks mostly fine"
    assert not any("string verdict normalized" in item for item in actions)
    with pytest.raises(AgentProtocolError, match="passed must be boolean"):
        UltraOrchestrator._validate_typed_response("review", normalized)


def test_exact_child_artifact_restoration_replays_matching_durable_write():
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        expected_content = (
            "from formatter import format_sum\n\n"
            "def test_format_sum():\n"
            "    assert format_sum(2, 3) == 'Sum is 5'\n"
        )
        expected_hash = hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
        target = workspace / "tests" / "test_components.py"
        target.parent.mkdir(parents=True)
        target.write_text("def test_drifted():\n    assert False\n", encoding="utf-8")

        store = StateStore(workspace)
        try:
            goal = store.create_goal("Restore accepted child artifact")
            action_id = store.begin_action(
                goal.id,
                "write_file",
                {
                    "arguments": {
                        "path": "tests/test_components.py",
                        "content": expected_content,
                    }
                },
                mutating=True,
            )
            store.complete_action(action_id, "accepted test artifact")
            adapter = StateStoreUltraAdapter(
                store,
                goal.id,
                ModelDescriptor("ollama", "offline-ultra", ExecutionClass.LOCAL),
                AccessLevel.NORMAL,
                UltraConfig(),
                workspace=workspace,
            )
            contract = TaskContractV1(
                id="M003",
                title="Tests",
                objective="Create tests.",
                acceptance_criteria=("Tests exist.",),
                verification=("Run pytest.",),
                write_paths=("tests/test_components.py",),
            )
            result = adapter.restore_exact_child_artifacts(
                "ultra-test-run",
                WorkNode(
                    contract=contract,
                    status=NodeStatus.RUNNING,
                    children=("M003.1",),
                ),
                (
                    {
                        "implementation": {
                            "artifacts": [
                                {
                                    "path": "tests/test_components.py",
                                    "content_hash": expected_hash,
                                }
                            ]
                        }
                    },
                ),
            )
        finally:
            store.close()

        assert result["passed"]
        assert result["evidence"][0]["restored"]
        assert target.read_text(encoding="utf-8") == expected_content


def test_independent_reviews_approve_open_durable_mutation_checkpoint():
    change_set = ChangeSetV1(
        ultra_run_id="run",
        responsible_agent_id="failed-transport-coder",
        parent_id="M001",
        changed_files=("module.py",),
        status=ChangeSetStatus.OPEN,
    )
    saved = [change_set]
    store = mock.Mock()
    store.list_change_sets.side_effect = lambda _run_id: tuple(saved)
    store.save_change_set.side_effect = lambda item: saved.__setitem__(0, item) or item
    store.list_quality_cycles.return_value = ()
    adapter = object.__new__(StateStoreUltraAdapter)
    adapter.run_id = "run"
    adapter.store = store

    with mock.patch("agent.ultra_session.ProjectBrain") as brain:
        adapter.record_quality_review("M001", "clean_code", True)
        adapter.record_quality_review("M001", "security", True)
        adapter.record_quality_review("M001", "test_quality", True)

    assert saved[0].status is ChangeSetStatus.APPROVED
    assert saved[0].review_status == {
        "clean_code": "passed",
        "security": "passed",
        "test_quality": "passed",
    }
    store.save_quality_cycle.assert_called_once()
    brain.return_value.write.assert_called_once()


class PhaseProvider:
    """Offline provider that follows every ULTRA phase and performs one edit."""

    model = "offline-ultra"

    def __init__(
        self,
        *,
        ask_question: bool = False,
        malformed_optional_question: bool = False,
    ) -> None:
        self.calls = 0
        self.ask_question = ask_question
        self.malformed_optional_question = malformed_optional_question

    @staticmethod
    def _phase(system: str) -> str:
        return system.split("phase ", 1)[1].split(".", 1)[0]

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        del conversation, tools, on_text, on_thought
        self.calls += 1
        phase = self._phase(system)
        if phase == "goal_spec":
            if self.calls == 1:
                return AssistantTurn(
                    tool_calls=[ToolCall("inspect-workspace", "list_files", {"path": "."})],
                    usage=Usage(1, 0, 1),
                )
            questions = (
                [
                    {
                        "id": "platform",
                        "header": "Platform",
                        "question": "Which target platform should own the first release?",
                        "options": [
                            {
                                "value": "desktop",
                                "label": "Desktop",
                                "description": "Keyboard-first desktop build.",
                                "recommended": True,
                            },
                            {
                                "value": "web",
                                "label": "Web",
                                "description": "Browser-first deployment.",
                                "recommended": False,
                            },
                            {
                                "value": "both",
                                "label": "Both",
                                "description": "Support desktop and browser.",
                                "recommended": False,
                            },
                        ],
                        "allow_freeform": True,
                        "reason": "The product target is not encoded in the repository.",
                        "decision_need": {
                            "impact": "Selects the external release target and deployment authority.",
                            "affected_scope": ["release target"],
                            "affected_effects": ["external_side_effect"],
                            "reversible": False,
                            "requires_user_authority": True,
                            "reason": "The original request does not authorize a release target.",
                            "evidence_refs": ["goal:original_request"],
                        },
                    }
                ]
                if self.ask_question
                else (
                    [
                        {
                            "id": "optional_style",
                            "header": "Style",
                            "question": "Use the model's suggested visual style?",
                            "options": [{"label": "Use it"}],
                            "allow_freeform": True,
                            "reason": "This is optional presentation polish.",
                        }
                    ]
                    if self.malformed_optional_question
                    else []
                )
            )
            payload = {
                "objective": "Build the demo",
                "success_criteria": ["game.txt exists"],
                "constraints": [],
                "in_scope": ["demo"],
                "out_of_scope": [],
                "assumptions": [],
                "questions": questions,
            }
        elif phase == "architecture":
            payload = {
                "summary": "One-file demo architecture",
                "components": [{"name": "demo"}],
                "interfaces": [],
                "decisions": [],
                "dependencies": [],
                "invariants": [],
            }
        elif phase == "master_plan":
            payload = {
                "summary": "Build and verify the demo",
                "execution_strategy": "Execute one safe module and every quality gate.",
                "modules": [
                    {
                        "id": "M001",
                        "title": "Demo",
                        "objective": "Create game.txt",
                        "acceptance_criteria": ["game.txt exists"],
                        "verification": ["Read game.txt"],
                        "depends_on": [],
                        "write_paths": ["game.txt"],
                        "forbidden_changes": [],
                        "owned_interfaces": [],
                        "metadata": {},
                    }
                ],
            }
        elif phase == "mini_plan":
            payload = {"steps": ["Create the file"], "research_required": False}
        elif phase == "decompose":
            payload = {"children": [], "research_required": False}
        elif phase in {"implement", "fix"} and self.calls == 1:
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        "write-game",
                        "write_file",
                        {"path": "game.txt", "content": "ready\n"},
                    )
                ],
                usage=Usage(1, 0, 1),
            )
        elif phase in {
            "review",
            "test",
            "integrate",
            "global_integration",
            "global_review",
            "final_evidence",
        }:
            payload = {
                "passed": True,
                "issues": [],
                "findings": [],
                "evidence": [{"kind": "check", "value": "ok"}],
                "test_results": [{"passed": True}],
            }
        else:
            payload = {
                "success": True,
                "passed": True,
                "artifacts": [{"path": "game.txt", "uri": "workspace:game.txt"}],
                "evidence": [{"kind": "done"}],
                "findings": [],
            }
        payload.setdefault(
            "reasoning_artifact",
            {
                "claim": f"{phase} satisfies the current contract",
                "supporting_evidence": ["offline fixture evidence"],
                "counterarguments": ["fixture could miss integration regressions"],
                "rejected_alternatives": ["manual-only verification"],
                "verification_plan": ["use harness integration assertions"],
                "reasoning_graph": {
                    "nodes": [
                        {
                            "id": "fixture-evidence",
                            "type": "verification",
                            "summary": "Offline fixture evidence supports the current phase claim.",
                            "status": "verified",
                            "evidence_refs": ["offline fixture evidence"],
                        },
                        {
                            "id": "manual-only",
                            "type": "option",
                            "summary": "Manual-only verification is insufficient for the harness.",
                            "status": "rejected",
                            "evidence_refs": [],
                        },
                    ],
                    "edges": [
                        {"from": "fixture-evidence", "to": "manual-only", "relation": "rejects"}
                    ],
                },
            },
        )
        return AssistantTurn(
            text=json.dumps(
                {
                    "payload": payload,
                    "summary": f"{phase} complete",
                    "reasoning_summary": "Verified against explicit evidence.",
                    "insights": [],
                }
            ),
            usage=Usage(2, 0, 2),
        )

    def summarize(self, messages):
        del messages
        return "summary"


class EmptyThenValidImplementProvider(PhaseProvider):
    """Emit one transport-empty implementation turn, then recover normally."""

    def __init__(self) -> None:
        super().__init__()
        self.empty_implement_sent = False
        self.implementation_write_sent = False

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        phase = self._phase(system)
        if phase != "implement":
            return super().call(
                conversation,
                tools,
                system,
                on_text=on_text,
                on_thought=on_thought,
            )
        del conversation, tools, on_text, on_thought
        self.calls += 1
        if not self.empty_implement_sent:
            self.empty_implement_sent = True
            return AssistantTurn(usage=Usage(1, 0, 1))
        if not self.implementation_write_sent:
            self.implementation_write_sent = True
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        "write-game-after-empty",
                        "write_file",
                        {"path": "game.txt", "content": "ready\n"},
                    )
                ],
                usage=Usage(1, 0, 1),
            )
        payload = {
            "success": True,
            "passed": True,
            "artifacts": [{"path": "game.txt", "uri": "workspace:game.txt"}],
            "evidence": [{"kind": "done"}],
            "findings": [],
            "reasoning_artifact": {
                "claim": "The implementation satisfies the current contract.",
                "supporting_evidence": ["game.txt was written"],
                "counterarguments": ["The first transport turn was empty"],
                "rejected_alternatives": ["Treating an empty turn as semantic failure"],
                "verification_plan": ["Use the ordinary independent review gates"],
                "reasoning_graph": {
                    "nodes": [
                        {
                            "id": "written",
                            "type": "verification",
                            "summary": "The approved artifact was written.",
                            "status": "verified",
                            "evidence_refs": ["game.txt was written"],
                        }
                    ],
                    "edges": [],
                },
            },
        }
        return AssistantTurn(
            text=json.dumps(
                {
                    "payload": payload,
                    "summary": "implement complete after transport repair",
                    "reasoning_summary": "The empty turn was retried before mutation.",
                    "insights": [],
                }
            ),
            usage=Usage(2, 0, 2),
        )


class HtmlGameProvider(PhaseProvider):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.html = html

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        phase = self._phase(system)
        # Recursive single-artifact runs reserve the final path for the parent
        # FinalAssembler. The fixture must therefore perform its write during
        # the final assembler's integrate phase, not only in a leaf coder.
        if phase == "integrate" and "FINAL ASSEMBLER PHASE" in system and self.calls == 0:
            self.calls += 1
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        "assemble-html",
                        "write_file",
                        {"path": "index.html", "content": self.html},
                    )
                ],
                usage=Usage(1, 0, 1),
            )
        if phase == "master_plan":
            self.calls += 1
            payload = {
                "summary": "Build and verify the single-file 3D HTML game",
                "execution_strategy": "Create index.html, run browser and benchmark gates.",
                "modules": [
                    {
                        "id": "M001",
                        "title": "Single-file 3D browser game",
                        "objective": "Create index.html",
                        "acceptance_criteria": ["index.html contains a playable 3D browser game"],
                        "verification": ["Preview index.html and run deterministic 3D HTML benchmark"],
                        "depends_on": [],
                        "write_paths": ["index.html"],
                        "forbidden_changes": [],
                        "owned_interfaces": [],
                        "metadata": {},
                    }
                ],
            }
            payload.setdefault(
                "reasoning_artifact",
                {
                    "claim": "The plan targets the requested 3D HTML artifact.",
                    "supporting_evidence": ["write path index.html"],
                    "counterarguments": ["A static page could masquerade as a game."],
                    "rejected_alternatives": ["Separate JS/CSS assets"],
                    "verification_plan": ["Run the single-file 3D HTML benchmark"],
                    "reasoning_graph": {
                        "nodes": [
                            {
                                "id": "single-file-html",
                                "type": "decision",
                                "summary": "Use one index.html artifact and benchmark it.",
                                "status": "chosen",
                                "evidence_refs": ["write path index.html"],
                            },
                            {
                                "id": "split-assets",
                                "type": "option",
                                "summary": "Separate assets violate the single-file benchmark goal.",
                                "status": "rejected",
                                "evidence_refs": [],
                            },
                        ],
                        "edges": [
                            {"from": "single-file-html", "to": "split-assets", "relation": "rejects"}
                        ],
                    },
                },
            )
            return AssistantTurn(text=json.dumps({"payload": payload, "summary": "master plan complete", "reasoning_summary": "Planned benchmarked HTML output."}), usage=Usage(2, 0, 2))
        if phase in {"implement", "fix"} and self.calls == 0:
            self.calls += 1
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        "write-html",
                        "write_file",
                        {"path": "index.html", "content": self.html},
                    )
                ],
                usage=Usage(1, 0, 1),
            )
        turn = super().call(conversation, tools, system, on_text, on_thought)
        if getattr(turn, "text", ""):
            data = json.loads(turn.text)
            payload = dict(data.get("payload", {}))
            if phase in {"implement", "integrate", "global_integration", "final_evidence"}:
                artifacts = list(payload.get("artifacts", []) or [])
                artifacts.append({"path": "index.html", "uri": "workspace:index.html"})
                payload["artifacts"] = artifacts
                evidence = list(payload.get("evidence", []) or [])
                evidence.append({"kind": "artifact", "path": "index.html"})
                payload["evidence"] = evidence
            data["payload"] = payload
            return AssistantTurn(text=json.dumps(data), usage=turn.usage)
        return turn


class StaleWriteProvider(PhaseProvider):
    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        if self._phase(system) == "implement" and self.calls == 0:
            (self.workspace / "game.txt").write_text("external update\n")
        return super().call(conversation, tools, system, on_text, on_thought)


class BlockingProvider(PhaseProvider):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.started = started
        self.release = release

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        if self._phase(system) == "implement" and self.calls == 0:
            self.started.set()
            if not self.release.wait(5):
                raise TimeoutError("test did not release the implement agent")
        return super().call(conversation, tools, system, on_text, on_thought)


class ConsensusRejectProvider(PhaseProvider):
    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        turn = super().call(conversation, tools, system, on_text, on_thought)
        phase = self._phase(system)
        if phase != "review" or not getattr(turn, "text", ""):
            return turn
        data = json.loads(turn.text)
        payload = dict(data.get("payload", {}))
        payload.update(
            {
                "passed": True,
                "consensus_vote": "reject",
                "confidence": 0.95,
                "findings": [],
                "issues": [],
            }
        )
        data["payload"] = payload
        data["summary"] = "review claims pass but consensus rejects evidence"
        data["reasoning_summary"] = "Evidence is insufficient for release."
        return AssistantTurn(text=json.dumps(data), usage=turn.usage)


class EmptyFinalEvidenceProvider(PhaseProvider):
    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        turn = super().call(conversation, tools, system, on_text, on_thought)
        phase = self._phase(system)
        if phase != "final_evidence" or not getattr(turn, "text", ""):
            return turn
        data = json.loads(turn.text)
        payload = dict(data.get("payload", {}))
        payload.update({"passed": True, "evidence": [], "test_results": []})
        data["payload"] = payload
        data["summary"] = "final evidence claims pass without durable proof"
        return AssistantTurn(text=json.dumps(data), usage=turn.usage)


class MissingReasoningReviewProvider(PhaseProvider):
    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        turn = super().call(conversation, tools, system, on_text, on_thought)
        phase = self._phase(system)
        if phase != "review" or not getattr(turn, "text", ""):
            return turn
        data = json.loads(turn.text)
        payload = dict(data.get("payload", {}))
        payload.pop("reasoning_artifact", None)
        payload.update({"passed": True, "findings": [], "issues": []})
        data["payload"] = payload
        data["summary"] = "review claims pass without debate artifact"
        return AssistantTurn(text=json.dumps(data), usage=turn.usage)


class PlanningQuestionProvider:
    model = "offline-plan"

    def __init__(self) -> None:
        self.planner_calls = 0

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        del conversation, system, on_text, on_thought
        names = {item["function"]["name"] for item in tools}
        if "submit_plan_review" in names:
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        "critic",
                        "submit_plan_review",
                        {"verdict": "pass", "summary": "Complete plan", "issues": []},
                    )
                ]
            )
        self.planner_calls += 1
        if self.planner_calls in {1, 3}:
            call_id = "inspect-before-question" if self.planner_calls == 1 else "inspect-after-answer"
            return AssistantTurn(
                tool_calls=[ToolCall(call_id, "list_files", {"path": "."})]
            )
        if self.planner_calls == 2:
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        "ask-platform",
                        "request_plan_input",
                        {
                            "questions": [
                                {
                                    "id": "platform",
                                    "header": "Platform",
                                    "question": "Which platform owns the first release?",
                                    "options": [
                                        {
                                            "value": "desktop",
                                            "label": "Desktop",
                                            "description": "Desktop application.",
                                            "recommended": True,
                                        },
                                        {
                                            "value": "web",
                                            "label": "Web",
                                            "description": "Browser application.",
                                            "recommended": False,
                                        },
                                        {
                                            "value": "cross_platform",
                                            "label": "Cross-platform",
                                            "description": "Support desktop and browser releases.",
                                            "recommended": False,
                                        },
                                    ],
                                    "allow_freeform": True,
                                    "reason": "Product scope is not discoverable from this empty workspace.",
                                    "decision_need": {
                                        "impact": "Selects the product platform and artifact type.",
                                        "affected_scope": ["release platform"],
                                        "affected_effects": ["workspace artifacts"],
                                        "reversible": False,
                                        "requires_user_authority": True,
                                        "reason": "The empty workspace cannot establish the intended release platform.",
                                        "evidence_refs": ["inspection:I001"],
                                    },
                                }
                            ]
                        },
                    )
                ]
            )
        return AssistantTurn(
            tool_calls=[
                ToolCall(
                    "plan",
                    "propose_plan",
                    {
                        "semantic_goal": {
                            "original_request": "Create an application",
                            "interpreted_outcome": "Create the user-selected application target.",
                            "requested_effects": ["read_workspace", "mutate_workspace"],
                            "required_outcomes": ["The selected application entry point exists."],
                            "constraints": ["Preserve the selected platform decision."],
                            "exclusions": [],
                            "acceptance_criteria": ["app.py exists"],
                            "unresolved_decisions": [],
                            "repository_evidence_refs": ["inspection:I001"],
                        },
                        "summary": "Create the selected platform entry point",
                        "applicability_evidence": [
                            {
                                "fact": "The workspace was inspected and is ready for app.py.",
                                "source": "tool:inspect-after-answer",
                                "supports_tasks": ["T001"],
                            }
                        ],
                        "execution_strategy": "Create app.py, verify it, and preserve the selected platform decision.",
                        "expected_changes": [
                            {
                                "path": "app.py",
                                "intent": "Add the selected platform entry point.",
                                "basis": "model_selected_new_layout",
                                "evidence_refs": ["inspection:I001"],
                                "supports_tasks": ["T001"],
                            }
                        ],
                        "tasks": [
                            {
                                "id": "T001",
                                "title": "Create entry point",
                                "description": "Create the selected platform entry point.",
                                "acceptance_criteria": ["app.py exists"],
                                "verification": ["Read app.py"],
                                "depends_on": [],
                                "risk": "low",
                            }
                        ],
                    },
                )
            ]
        )

    def summarize(self, messages):
        del messages
        return "summary"


class FinalOnlyGoalProvider:
    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        del conversation, tools, system, on_text, on_thought
        return AssistantTurn(
            text=json.dumps(
                {
                    "payload": {
                        "objective": "Build the demo",
                        "success_criteria": ["Done"],
                        "questions": [],
                    },
                    "summary": "Uninspected goal",
                }
            )
        )


class PassingTesterProvider:
    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        del conversation, tools, system, on_text, on_thought
        return AssistantTurn(
            text=json.dumps(
                {
                    "payload": {
                        "passed": True,
                        "issues": [],
                        "findings": [],
                        "evidence": [{"kind": "model-claim", "status": "passed"}],
                        "test_results": [{"name": "model_claim", "passed": True}],
                    },
                    "summary": "tester passed",
                }
            )
        )


class CapturingGoalProvider:
    def __init__(self) -> None:
        self.user_payload = {}

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        del tools, system, on_text, on_thought
        self.user_payload = json.loads(conversation[0]["content"])
        return AssistantTurn(
            text=json.dumps(
                {
                    "payload": {
                        "objective": "Build the demo",
                        "success_criteria": ["Done"],
                        "questions": [],
                    },
                    "summary": "captured",
                }
            )
        )


class UltraIntegrationTests(unittest.TestCase):
    def test_task_contract_derives_missing_proof_fields_only_from_nonempty_objective(self):
        contract = TaskContractV1.from_mapping(
            {"title": "Polish", "objective": "Add final browser polish"},
            fallback_id="M001",
        )
        self.assertIn("Add final browser polish", contract.acceptance_criteria[0])
        self.assertIn("Add final browser polish", contract.verification[0])
        with self.assertRaises(Exception):
            TaskContractV1.from_mapping({"title": "Empty"}, fallback_id="M001")

    def test_master_plan_normalizes_weak_model_module_ids_and_dependencies(self):
        plan = MasterPlanV1.from_mapping(
            {
                "summary": "Build in waves",
                "modules": [
                    {
                        "id": "M01",
                        "title": "Base",
                        "objective": "Create base",
                        "acceptance_criteria": ["Base exists"],
                        "verification": ["Inspect base"],
                    },
                    {
                        "id": "module-two",
                        "title": "Polish",
                        "objective": "Polish base",
                        "acceptance_criteria": ["Polish exists"],
                        "verification": ["Inspect polish"],
                        "depends_on": ["M1"],
                    },
                ],
            }
        )
        self.assertEqual([item.id for item in plan.modules], ["M001", "M002"])
        self.assertEqual(plan.modules[1].depends_on, ("M001",))

    def test_master_plan_normalizes_dependency_with_human_label_suffix(self):
        plan = MasterPlanV1.from_mapping(
            {
                "summary": "Build",
                "modules": [
                    {"id": "M001", "title": "Renderer", "objective": "Render"},
                    {
                        "id": "M002",
                        "title": "Gameplay",
                        "objective": "Play",
                        "depends_on": ["M001: Renderer Core"],
                    },
                ],
            }
        )
        self.assertEqual(plan.modules[1].depends_on, ("M001",))

        sparse = MasterPlanV1.from_mapping(
            {
                "summary": "Sparse weak-model plan",
                "modules": [
                    {
                        "acceptance_criteria": ["Browser QA passes"],
                        "verification": ["Run browser QA"],
                    }
                ],
            }
        )
        self.assertEqual(sparse.modules[0].title, "Module M001")
        self.assertEqual(sparse.modules[0].objective, "Browser QA passes")

    def test_child_scope_parser_distinguishes_write_targets_from_read_references(self):
        self.assertEqual(
            UltraOrchestrator._declared_write_targets(
                "Implement format_sum in formatter.py."
            ),
            ("formatter.py",),
        )
        self.assertEqual(
            UltraOrchestrator._declared_write_targets(
                "Create tests/test_components.py which validates mathlib.py "
                "and formatter.py."
            ),
            ("tests/test_components.py",),
        )
        child = UltraOrchestrator._semantic_terms(
            "Implement formatting logic format_sum which calls add from mathlib."
        )
        math_parent = UltraOrchestrator._semantic_terms(
            "Create the minimal mathematical utility."
        )
        formatter_sibling = UltraOrchestrator._semantic_terms(
            "Implement formatting logic dependent on the core math utility."
        )
        self.assertGreater(
            len(child & formatter_sibling),
            len(child & math_parent),
        )

    def test_noncomponent_review_missing_boolean_uses_typed_evidence(self):
        normalized, actions = UltraOrchestrator._normalize_typed_payload(
            "review",
            {
                "findings": [],
                "issues": [],
                "evidence": [{"path": "mathlib.py", "check": "signature present"}],
            },
            {"contract": {"metadata": {}}},
        )
        self.assertTrue(normalized["passed"])
        self.assertTrue(any("finding-free typed evidence" in item for item in actions))

        abstained, abstention_actions = UltraOrchestrator._normalize_typed_payload(
            "review",
            {"findings": [], "issues": [], "evidence": []},
            {"contract": {"metadata": {}}},
        )
        self.assertTrue(abstained["passed"])
        self.assertTrue(abstained["abstained"])
        self.assertTrue(any("reviewer abstained" in item for item in abstention_actions))
        UltraOrchestrator._validate_typed_response("review", abstained)

        missing_test_verdict, _ = UltraOrchestrator._normalize_typed_payload(
            "test",
            {"findings": [], "issues": [], "evidence": []},
            {"contract": {"metadata": {}}},
        )
        self.assertNotIn("passed", missing_test_verdict)
        with self.assertRaisesRegex(AgentProtocolError, "passed must be boolean"):
            UltraOrchestrator._validate_typed_response("test", missing_test_verdict)

    def test_local_integrator_missing_boolean_abstains_after_quality_gate(self):
        normalized, actions = UltraOrchestrator._normalize_typed_payload(
            "integrate",
            {"passed": None, "findings": [], "issues": []},
            {"publish_component_package": True},
        )
        self.assertTrue(normalized["success"])
        self.assertTrue(normalized["passed"])
        self.assertTrue(normalized["abstained"])
        self.assertTrue(any("publisher abstained" in item for item in actions))
        self.assertTrue(
            UltraOrchestrator._passed(
                AgentResponse(payload=normalized, summary="publisher abstained")
            )
        )

        blocked, blocked_actions = UltraOrchestrator._normalize_typed_payload(
            "integrate",
            {"findings": ["formatter.py imports an unapproved dependency"]},
            {"publish_component_package": True},
        )
        self.assertNotIn("success", blocked)
        self.assertFalse(blocked.get("abstained", False))
        self.assertEqual(blocked_actions, ())
        with self.assertRaisesRegex(AgentProtocolError, "must be boolean"):
            UltraOrchestrator._validate_typed_response("integrate", blocked)

    def test_informational_structured_findings_do_not_trigger_repairs(self):
        response = AgentResponse(
            payload={
                "passed": True,
                "findings": [
                    {
                        "severity": "info",
                        "summary": "Dependency contract is satisfied.",
                    },
                    {
                        "severity": "high",
                        "summary": "Approved dependency was bypassed.",
                    },
                    {
                        "blocking": False,
                        "summary": "Optional naming suggestion.",
                    },
                ],
            },
            summary="structured review",
        )
        self.assertEqual(
            UltraOrchestrator._findings(response),
            ("Approved dependency was bypassed.",),
        )

    def test_quality_milestone_normalizes_object_milestones_without_hashing_dicts(self):
        milestones = _with_quality_milestone(
            [{"title": "Playable Core"}, {"name": "Visual Polish"}]
        )
        self.assertEqual(milestones[0]["title"], "Playable Core")
        self.assertEqual(milestones[-1]["kind"], "quality_gate")
        self.assertEqual(len(_with_quality_milestone(milestones)), len(milestones))

    def _descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            "ollama",
            "offline-ultra",
            ExecutionClass.LOCAL,
            capabilities=("tools",),
        )

    def test_workspace_hashes_use_relative_ignore_rules_and_prune_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "run-artifacts" / "project"
            workspace.mkdir(parents=True)
            (workspace / "artifact.txt").write_text("accepted", encoding="utf-8")
            dependencies = workspace / "node_modules" / "package"
            dependencies.mkdir(parents=True)
            (dependencies / "ignored.js").write_text("ignored", encoding="utf-8")
            store = StateStore(workspace)
            session = UltraSession(
                store=store,
                workspace=workspace,
                descriptor=self._descriptor(),
                permission_adapter=PermissionAdapter("normal", DockerSandbox()),
                approval=lambda *_args: True,
                events=EventBus(),
                config=UltraConfig(),
                agent_steps=2,
            )
            try:
                hashes = session._workspace_hashes()
                self.assertEqual(set(hashes), {"artifact.txt"})
                self.assertEqual(
                    hashes["artifact.txt"],
                    hashlib.sha256(b"accepted").hexdigest(),
                )
            finally:
                session.close()
                store.close()

    def _runtime(self, workspace: Path, store: StateStore, *, ask_question: bool = False):
        descriptor = self._descriptor()
        provider = PhaseProvider(ask_question=ask_question)
        return AgentRuntime(
            provider,
            store,
            workspace,
            model_descriptor=descriptor,
            permission_adapter=PermissionAdapter("normal", DockerSandbox()),
            approval=lambda *_args: True,
            events=EventBus(),
        )

    def test_ultra_rejects_inapplicable_preview_before_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "component.js").write_text(
                "export const value = 1;", encoding="utf-8"
            )
            store = StateStore(workspace)
            approvals: list[tuple[str, dict, str]] = []
            session = UltraSession(
                store=store,
                workspace=workspace,
                descriptor=self._descriptor(),
                permission_adapter=PermissionAdapter("normal", DockerSandbox()),
                approval=lambda name, args, risk: approvals.append(
                    (name, args, risk)
                ) or True,
                events=EventBus(),
                config=UltraConfig(),
                agent_steps=2,
            )
            try:
                result = session._execute_tool(
                    ToolCall(
                        "invalid-preview",
                        "preview_html",
                        {"path": "component.js", "open_browser": False},
                    ),
                    AgentRequest(
                        run_id="run",
                        role=AgentRole.TESTER,
                        phase="test",
                        system_prompt="Verify the accepted artifact.",
                        context={},
                        task={},
                        node_id=None,
                    ),
                )
                self.assertIn("requires an existing .html or .htm file", result)
                self.assertEqual(approvals, [])
                self.assertEqual(store.list_session_actions("workspace-session"), ())
            finally:
                session.close()
                store.close()

    def test_ultra_replays_completed_identical_mutation_receipt_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            goal = store.create_goal("Write one approved artifact")
            approvals: list[str] = []
            session = UltraSession(
                store=store,
                workspace=workspace,
                descriptor=self._descriptor(),
                permission_adapter=PermissionAdapter("normal", DockerSandbox()),
                approval=lambda name, _args, _risk: approvals.append(name) or True,
                events=EventBus(),
                config=UltraConfig(),
                agent_steps=2,
            )
            adapter = mock.Mock()
            adapter.run_id = None
            adapter.master_task_for_node.return_value = None
            adapter.lease_hash.return_value = (False, None)
            session.goal_id = goal.id
            session.adapter = adapter
            node = mock.Mock(write_paths=("artifact.txt",), pre_write_hashes={})
            request = AgentRequest(
                run_id="run",
                role=AgentRole.CODER,
                phase="implement",
                system_prompt="Implement the accepted artifact.",
                context={},
                task={},
                node_id="node",
            )
            try:
                with mock.patch.object(session, "_node", return_value=node):
                    first = session._execute_tool(
                        ToolCall(
                            "write-first",
                            "write_file",
                            {"path": "artifact.txt", "content": "accepted\n"},
                        ),
                        request,
                    )
                    second = session._execute_tool(
                        ToolCall(
                            "write-repeat",
                            "write_file",
                            {"path": "artifact.txt", "content": "accepted\n"},
                        ),
                        request,
                    )

                self.assertIn("Wrote", first)
                self.assertEqual(second, first)
                self.assertEqual(len(store.list_actions(goal.id)), 1)
                self.assertEqual(approvals, [])
                self.assertTrue(
                    any(
                        event.event_type == "mutation.replay_prevented"
                        for event in store.list_recent_events(goal.id, limit=100)
                    )
                )
            finally:
                session.close()
                store.close()

    def test_ultra_rejects_unplanned_shell_command_without_user_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            goal = store.create_goal("Verify one approved artifact")
            approvals: list[str] = []
            session = UltraSession(
                store=store,
                workspace=workspace,
                descriptor=self._descriptor(),
                permission_adapter=PermissionAdapter("normal", DockerSandbox()),
                approval=lambda name, _args, _risk: approvals.append(name) or True,
                events=EventBus(),
                config=UltraConfig(),
                agent_steps=2,
            )
            session.goal_id = goal.id
            node = mock.Mock()
            node.contract.verification = ("preview_html index.html",)
            request = AgentRequest(
                run_id="run",
                role=AgentRole.CODER,
                phase="fix",
                system_prompt="Repair using the approval-bound verifier.",
                context={},
                task={},
                node_id="node",
            )
            try:
                with mock.patch.object(session, "_node", return_value=node):
                    result = session._execute_tool(
                        ToolCall(
                            "unplanned-command",
                            "run_command",
                            {"command": "npm start", "cwd": "."},
                        ),
                        request,
                    )

                self.assertIn("not an approval-bound verification command", result)
                self.assertEqual(approvals, [])
                self.assertEqual(store.list_actions(goal.id), ())
                self.assertTrue(
                    any(
                        event.event_type == "tool_contract.rejected"
                        and event.payload.get("stage") == "ultra_execution_command_scope"
                        for event in store.list_recent_events(goal.id, limit=20)
                    )
                )
            finally:
                session.close()
                store.close()

    def test_ultra_explicit_denial_is_not_truthy_and_leaves_no_running_action(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            goal = store.create_goal("Verify the approved project")
            session = UltraSession(
                store=store,
                workspace=workspace,
                descriptor=self._descriptor(),
                permission_adapter=PermissionAdapter("normal", DockerSandbox()),
                approval=lambda *_args: ApprovalDecision.DENY,
                events=EventBus(),
                config=UltraConfig(),
                agent_steps=2,
            )
            adapter = mock.Mock()
            adapter.master_task_for_node.return_value = None
            session.goal_id = goal.id
            session.adapter = adapter
            try:
                with self.assertRaisesRegex(
                    ApprovalRequiredError,
                    "Approval for run_command is required",
                ):
                    session._execute_tool(
                        ToolCall(
                            "denied-command",
                            "run_command",
                            {"command": "echo must-not-run", "cwd": "."},
                        ),
                        AgentRequest(
                            run_id="run",
                            role=AgentRole.CODER,
                            phase="implement",
                            system_prompt="Verify the accepted artifact.",
                            context={},
                            task={},
                            node_id=None,
                        ),
                    )

                actions = store.list_actions(goal.id)
                self.assertEqual(len(actions), 1)
                self.assertEqual(actions[0]["status"], "denied")
                paused = store.get_goal(goal.id)
                self.assertEqual(paused.metadata["waiting_on"], "approval")
                self.assertEqual(
                    paused.metadata["pending_tool_approval"]["tool"],
                    "run_command",
                )
            finally:
                session.close()
                store.close()

    def test_ultra_factory_propagates_session_reasoning_effort_to_every_role_provider(self):
        descriptor = self._descriptor()
        provider = PhaseProvider()
        with mock.patch.object(ModelDescriptor, "create_provider", return_value=provider):
            factory = WorkspaceUltraAgentFactory(
                descriptor,
                lambda _call, _request: "ok",
                EventBus(),
                max_steps=2,
                reasoning_effort="low",
            )
            agent = factory.create(AgentRole.CODER, run_id="run", node_id="node")
        self.assertEqual(agent.provider.reasoning_effort, "low")

    def test_ultra_retries_one_internal_unused_token_before_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            provider = EmptyThenValidImplementProvider()
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    return_value=provider,
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    runtime.approve_ultra()
                    result = runtime.ultra_session.future.result(timeout=10)

                self.assertTrue(provider.empty_implement_sent)
                self.assertTrue(provider.implementation_write_sent)
                self.assertTrue(result.successful)
                self.assertEqual((workspace / "game.txt").read_text(), "ready\n")
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_routes_low_reasoning_off_for_deterministic_foundation_roles(self):
        provider = FinalOnlyGoalProvider()
        provider.reasoning_effort = "low"
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.GOAL_UNDERSTANDING,
            provider_name="ollama",
            model="offline",
            executor=lambda _call, _request: "ok",
            events=EventBus(),
            max_steps=2,
        )
        agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.GOAL_UNDERSTANDING,
                phase="goal_spec",
                system_prompt="Build GoalSpecV1.",
                context={},
                task={"prompt": "Build it"},
            )
        )
        self.assertEqual(provider.reasoning_effort, "off")
        self.assertEqual(provider.max_output_tokens, 4096)
        # Ollama's grammar-constrained format can crash Gemma-family runners
        # on internal tokens such as <unused50>. The harness still validates
        # and repairs the typed response without enabling that grammar.
        self.assertFalse(provider.force_json)

    def test_ultra_goal_spec_runs_harness_workspace_inspection_before_provider(self):
        calls = []
        agent = WorkspaceUltraAgent(
            FinalOnlyGoalProvider(),
            role=AgentRole.GOAL_UNDERSTANDING,
            provider_name="offline",
            model="final-only",
            executor=lambda call, _request: calls.append(call) or "(no files under '.')",
            events=EventBus(),
            max_steps=2,
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.GOAL_UNDERSTANDING,
                phase="goal_spec",
                system_prompt="Build GoalSpecV1.",
                context={},
                task={"prompt": "Build it"},
            )
        )
        self.assertEqual(calls[0].name, "list_files")
        self.assertEqual(calls[0].args, {"path": "."})
        self.assertEqual(response.payload["objective"], "Build the demo")

    def test_memory_writeback_compacts_accepted_receipts_without_model_call(self):
        provider = mock.Mock()
        provider.call.side_effect = AssertionError("memory must not call the provider")
        events = EventBus()
        captured = []
        events.subscribe(captured.append)
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.MEMORY,
            provider_name="offline",
            model="unused",
            executor=lambda _call, _request: "unexpected",
            events=events,
            max_steps=2,
        )

        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.MEMORY,
                phase=InnerPhase.MEMORY_WRITEBACK.value,
                system_prompt="Write memory.",
                context={},
                task={"result_summaries": ["Artifact hash and browser receipt accepted."]},
                node_id="M001",
            )
        )

        provider.call.assert_not_called()
        self.assertTrue(response.payload["success"])
        self.assertEqual(
            response.payload["insights"][0]["summary"],
            "Artifact hash and browser receipt accepted.",
        )
        self.assertTrue(
            any(item.kind == "ultra.deterministic_memory_writeback" for item in captured)
        )

    def test_post_review_component_publication_is_metadata_only(self):
        class PublicationProvider:
            def __init__(self):
                self.tools = None
                self.system = ""

            def call(self, conversation, tools, system, **_kwargs):
                del conversation
                self.tools = list(tools)
                self.system = system
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {
                                "passed": True,
                                "success": True,
                                "component_package": {
                                    "implementation": {
                                        "summary": "accepted bytes",
                                        "artifacts": [],
                                    }
                                },
                            },
                            "summary": "Package metadata published.",
                        }
                    )
                )

        provider = PublicationProvider()
        executed = []
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.INTEGRATOR,
            provider_name="offline",
            model="publisher",
            executor=lambda call, _request: executed.append(call) or "unexpected",
            events=EventBus(),
            max_steps=2,
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.INTEGRATOR,
                phase="integrate",
                system_prompt="Publish the component.",
                context={},
                task={
                    "publish_component_package": True,
                    "contract": {
                        "id": "M001",
                        "title": "Module",
                        "objective": "Publish reviewed metadata.",
                        "write_paths": ["module.py"],
                    },
                },
                node_id="M001",
            )
        )

        self.assertEqual(provider.tools, [])
        self.assertEqual(executed, [])
        self.assertIn("METADATA-ONLY PUBLICATION PHASE", provider.system)
        self.assertTrue(response.payload["passed"])

    def test_tester_native_write_call_is_rejected_before_executor(self):
        class DisallowedWriteProvider:
            def __init__(self):
                self.calls = 0

            def call(self, conversation, tools, system, **_kwargs):
                del conversation, tools, system
                self.calls += 1
                if self.calls == 1:
                    return AssistantTurn(
                        tool_calls=[
                            ToolCall(
                                "forbidden-write",
                                "write_file",
                                {
                                    "path": "tests/test_components.py",
                                    "content": "def test_bypass(): assert True\n",
                                },
                            )
                        ]
                    )
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {
                                "passed": True,
                                "issues": [],
                                "findings": [],
                                "test_results": [],
                            },
                            "summary": "Read-only test review complete.",
                        }
                    )
                )

        executed = []
        agent = WorkspaceUltraAgent(
            DisallowedWriteProvider(),
            role=AgentRole.TESTER,
            provider_name="offline",
            model="tester",
            executor=lambda call, _request: executed.append(call) or "unexpected",
            events=EventBus(),
            max_steps=3,
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.TESTER,
                phase="test",
                system_prompt="Test without mutation.",
                context={},
                task={
                    "contract": {
                        "id": "M001",
                        "title": "Module",
                        "objective": "Review tests.",
                        "write_paths": ["module.py"],
                    }
                },
                node_id="M001",
            )
        )

        self.assertEqual(executed, [])
        self.assertTrue(response.payload["passed"])

    def test_tester_shell_composed_write_is_rejected_before_executor(self):
        class ShellWriteProvider:
            def __init__(self):
                self.calls = 0

            def call(self, conversation, tools, system, **_kwargs):
                del conversation, tools, system
                self.calls += 1
                if self.calls == 1:
                    return AssistantTurn(
                        tool_calls=[
                            ToolCall(
                                "forbidden-shell-write",
                                "run_bash",
                                {
                                    "command": (
                                        "mkdir -p tests && echo \"def test_bypass(): "
                                        "assert True\" > tests/test_components.py"
                                    )
                                },
                            )
                        ]
                    )
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {
                                "passed": True,
                                "issues": [],
                                "findings": [],
                                "test_results": [],
                            },
                            "summary": "Mutation attempt rejected.",
                        }
                    )
                )

        executed = []
        events = EventBus()
        rejected = []
        events.subscribe(
            lambda event: (
                rejected.append(event)
                if event.kind == "ultra.disallowed_tool_rejected"
                else None
            )
        )
        agent = WorkspaceUltraAgent(
            ShellWriteProvider(),
            role=AgentRole.TESTER,
            provider_name="offline",
            model="tester",
            executor=lambda call, _request: executed.append(call) or "unexpected",
            events=events,
            max_steps=3,
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.TESTER,
                phase="test",
                system_prompt="Test without mutation.",
                context={},
                task={
                    "contract": {
                        "id": "M001",
                        "title": "Module",
                        "objective": "Review tests.",
                        "write_paths": ["module.py"],
                    }
                },
                node_id="M001",
            )
        )

        self.assertEqual(executed, [])
        self.assertEqual(len(rejected), 1)
        self.assertTrue(response.payload["passed"])

    def test_tester_single_pytest_command_reaches_executor(self):
        class PytestProvider:
            def __init__(self):
                self.calls = 0

            def call(self, conversation, tools, system, **_kwargs):
                del conversation, tools, system
                self.calls += 1
                if self.calls == 1:
                    return AssistantTurn(
                        tool_calls=[
                            ToolCall(
                                "allowed-pytest",
                                "run_bash",
                                {"command": "python -m pytest tests/test_components.py -q"},
                            )
                        ]
                    )
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {
                                "passed": True,
                                "issues": [],
                                "findings": [],
                                "test_results": [
                                    {"name": "pytest", "passed": True}
                                ],
                            },
                            "summary": "Verification passed.",
                        }
                    )
                )

        executed = []
        agent = WorkspaceUltraAgent(
            PytestProvider(),
            role=AgentRole.TESTER,
            provider_name="offline",
            model="tester",
            executor=lambda call, _request: executed.append(call) or "passed",
            events=EventBus(),
            max_steps=3,
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.TESTER,
                phase="test",
                system_prompt="Run verification.",
                context={},
                task={"contract": {"write_paths": ["module.py"]}},
                node_id="M001",
            )
        )

        self.assertEqual([call.name for call in executed], ["run_bash"])
        self.assertTrue(response.payload["passed"])

    def test_tester_cd_verifier_returns_harness_receipt_without_second_inference(self):
        class CdVerifierProvider:
            def __init__(self):
                self.calls = 0

            def call(self, conversation, tools, system, **_kwargs):
                del conversation, tools, system
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("executor receipt must end the tester turn")
                return AssistantTurn(
                    tool_calls=[
                        ToolCall(
                            "allowed-npm-test",
                            "run_bash",
                            {"command": "cd web && npm test"},
                        )
                    ]
                )

        provider = CdVerifierProvider()
        executed = []
        events = EventBus()
        deterministic_gates = []
        events.subscribe(
            lambda event: (
                deterministic_gates.append(event)
                if event.kind == "ultra.deterministic_test_gate"
                else None
            )
        )
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.TESTER,
            provider_name="offline",
            model="tester",
            executor=lambda call, _request: (
                executed.append(call) or "exit code: 0\nstdout:\nall tests passed\n"
            ),
            events=events,
            max_steps=3,
        )

        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.TESTER,
                phase="test",
                system_prompt="Run verification.",
                context={},
                task={"contract": {"write_paths": ["web/app.js"]}},
                node_id="M001",
            )
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0].name, "run_command")
        self.assertEqual(executed[0].args, {"cwd": "web", "command": "npm test"})
        self.assertTrue(response.payload["passed"])
        self.assertEqual(response.provider, "harness")
        self.assertEqual(len(deterministic_gates), 1)

    def test_reviewer_inspection_budget_forces_structured_verdict(self):
        class ReadingReviewerProvider:
            def __init__(self):
                self.calls = 0
                self.tool_counts = []

            def call(self, conversation, tools, system, **_kwargs):
                del conversation, system
                self.calls += 1
                self.tool_counts.append(len(tools))
                if tools:
                    return AssistantTurn(
                        tool_calls=[
                            ToolCall(
                                f"read-{self.calls}",
                                "read_file",
                                {"path": "module.py"},
                            )
                        ]
                    )
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {
                                "passed": True,
                                "issues": [],
                                "findings": [],
                                "evidence": [
                                    {"kind": "bounded_read_review", "path": "module.py"}
                                ],
                            },
                            "summary": "Bounded review passed.",
                        }
                    )
                )

        provider = ReadingReviewerProvider()
        executed = []
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.CLEAN_CODE_REVIEWER,
            provider_name="offline",
            model="reviewer",
            executor=lambda call, _request: (
                executed.append(call) or "def implementation():\n    return True\n"
            ),
            events=EventBus(),
            max_steps=16,
        )

        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.CLEAN_CODE_REVIEWER,
                phase="review",
                system_prompt="Review the implementation.",
                context={},
                task={"contract": {"write_paths": ["module.py"]}},
                node_id="M001",
            )
        )

        self.assertEqual(len(executed), 4)
        self.assertEqual(provider.calls, 5)
        self.assertEqual(provider.tool_counts[-1], 0)
        self.assertTrue(response.payload["passed"])

    def test_duplicate_coder_mutation_closes_tools_and_forces_handoff(self):
        class RepeatingWriterProvider:
            def __init__(self):
                self.calls = 0
                self.tool_counts = []

            def call(self, conversation, tools, system, **_kwargs):
                del conversation, system
                self.calls += 1
                self.tool_counts.append(len(tools))
                if tools:
                    return AssistantTurn(
                        tool_calls=[
                            ToolCall(
                                f"write-{self.calls}",
                                "write_file",
                                {"path": "module.py", "content": "VALUE = 1\n"},
                            )
                        ]
                    )
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {
                                "success": True,
                                "artifacts": [{"path": "module.py"}],
                                "evidence": [{"kind": "mutation_receipt"}],
                                "findings": [],
                            },
                            "summary": "Mutation handoff complete.",
                        }
                    )
                )

        provider = RepeatingWriterProvider()
        executed = []
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.CODER,
            provider_name="offline",
            model="coder",
            executor=lambda call, _request: (
                executed.append(call) or "Wrote 10 characters to module.py"
            ),
            events=EventBus(),
            max_steps=8,
        )

        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.CODER,
                phase="implement",
                system_prompt="Implement the module.",
                context={},
                task={"contract": {"write_paths": ["module.py"]}},
                node_id="M001",
            )
        )

        self.assertEqual(
            [call.name for call in executed].count("write_file"),
            2,
        )
        self.assertEqual(provider.calls, 3)
        self.assertEqual(provider.tool_counts[-1], 0)
        self.assertTrue(response.payload["success"])

    def test_ultra_injects_harness_reasoning_scaffold_without_hidden_cot(self):
        provider = CapturingGoalProvider()
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.GOAL_UNDERSTANDING,
            provider_name="offline",
            model="capture",
            executor=lambda _call, _request: "(no files under '.')",
            events=EventBus(),
            max_steps=2,
        )
        agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.GOAL_UNDERSTANDING,
                phase="goal_spec",
                system_prompt="Build GoalSpecV1.",
                context={},
                task={"prompt": "Build it"},
            )
        )
        scaffold = provider.user_payload["harness_reasoning_scaffold"]
        self.assertEqual(scaffold["mode"], "external_structured_summary")
        self.assertIn("verification_plan", scaffold["required_summary_fields"])
        self.assertIn("Do not reveal hidden chain-of-thought", scaffold["privacy_rule"])
        debate = provider.user_payload["harness_debate_protocol"]
        self.assertEqual(debate["output_key"], "reasoning_artifact")
        self.assertIn("counterarguments", debate["required_fields"])
        self.assertFalse(debate["external_reasoning_graph"]["required"])
        self.assertEqual(debate["external_reasoning_graph"]["output_key"], "reasoning_graph")
        self.assertIn("Do not expose hidden chain-of-thought", debate["privacy_rule"])

    def test_ultra_tester_forces_failed_result_when_harness_browser_preview_fails(self):
        calls = []

        def executor(call, _request):
            calls.append(call)
            if call.name == "preview_html":
                return json.dumps(
                    {
                        "status": "running",
                        "verification": "failed",
                        "console_errors": ["THREE is not defined"],
                        "page_errors": [],
                        "network_errors": ["HTTP 404 cdn"],
                        "screenshot_path": "preview.png",
                    }
                )
            return "ok"

        agent = WorkspaceUltraAgent(
            PassingTesterProvider(),
            role=AgentRole.TESTER,
            provider_name="offline",
            model="tester",
            executor=executor,
            events=EventBus(),
            max_steps=2,
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.TESTER,
                phase="test",
                system_prompt="Run tests.",
                context={},
                task={
                    "contract": {
                        "id": "M001",
                        "title": "HTML Build",
                        "objective": "Build index.html",
                        "write_paths": ["index.html"],
                    }
                },
                node_id="M001",
            )
        )
        self.assertEqual(calls[0].name, "preview_html")
        self.assertFalse(response.payload["passed"])
        self.assertTrue(
            any(
                "Harness browser verification failed" in issue
                for issue in response.payload["issues"]
            )
        )
        self.assertIn("THREE is not defined", response.payload["findings"])
        self.assertEqual(
            response.payload["test_results"][-1]["name"],
            "harness_html_browser_and_quality_gate",
        )

    def test_ultra_tester_returns_passed_browser_receipt_without_model_handoff(self):
        class ProviderMustNotRun:
            def call(self, *_args, **_kwargs):
                raise AssertionError("passed deterministic preview must end tester turn")

        calls = []

        def executor(call, _request):
            calls.append(call)
            if call.name == "preview_html":
                return json.dumps(
                    {
                        "status": "running",
                        "verification": "passed",
                        "http_status": 200,
                        "console_errors": [],
                        "page_errors": [],
                        "network_errors": [],
                        "screenshot_path": "preview.png",
                    }
                )
            return "ok"

        agent = WorkspaceUltraAgent(
            ProviderMustNotRun(),
            role=AgentRole.TESTER,
            provider_name="offline",
            model="tester",
            executor=executor,
            events=EventBus(),
            max_steps=2,
        )

        scenario = {
            "name": "addition",
            "steps": [{"action": "click", "role": "button", "name": "7"}],
            "assertions": [
                {
                    "role": "textbox",
                    "name": "Display",
                    "property": "value",
                    "equals": "7",
                }
            ],
        }
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.TESTER,
                phase="test",
                system_prompt="Run tests.",
                context={},
                task={
                    "contract": {
                        "id": "M001",
                        "title": "HTML Build",
                        "objective": "Build index.html",
                        "write_paths": ["index.html"],
                        "metadata": {"browser_scenarios": [scenario]},
                    }
                },
                node_id="M001",
            )
        )

        self.assertEqual(calls[0].name, "preview_html")
        self.assertEqual(calls[0].args["interactions"], [scenario])
        self.assertTrue(response.payload["passed"])
        self.assertEqual(response.provider, "harness")
        self.assertEqual(
            response.payload["test_results"][0]["name"],
            "harness_html_preview",
        )

    def test_ultra_preserves_valid_model_authored_question_without_keyword_filtering(self):
        questions = UltraOrchestrator._validated_questions(
            [
                {
                    "id": "platform",
                    "header": "Platform",
                    "question": "Which target platform should own the first release?",
                    "reason": "This changes product behavior and deployment.",
                    "options": [
                        {"value": "web", "label": "Web", "description": "Browser release.", "recommended": True},
                        {"value": "desktop", "label": "Desktop", "description": "Native release.", "recommended": False},
                        {"value": "both", "label": "Both", "description": "Support both releases.", "recommended": False},
                    ],
                    "allow_freeform": True,
                    "decision_need": {
                        "impact": "Selects the external release target and deployment authority.",
                        "affected_scope": ["release target"],
                        "affected_effects": ["external_side_effect"],
                        "reversible": False,
                        "requires_user_authority": True,
                        "reason": "The original request does not authorize a release target.",
                        "evidence_refs": ["goal:original_request"],
                    },
                },
            ]
        )
        self.assertEqual([item["id"] for item in questions], ["platform"])
        self.assertEqual(
            UltraOrchestrator._validated_questions(
                [{"id": "one", "question": "Use the only viable fallback?", "options": [{"label": "Yes"}]}]
            ),
            (),
        )

    def test_ultra_does_not_fill_missing_goal_objective_from_context(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "goal_spec",
            {"objective": "", "success_criteria": ["Playable game exists"]},
            {"prompt": "Build the requested game"},
        )
        self.assertEqual(payload["objective"], "")
        self.assertEqual(actions, ())
        with self.assertRaises(AgentProtocolError):
            UltraOrchestrator._validate_typed_response("goal_spec", payload)

        untouched, no_actions = UltraOrchestrator._normalize_typed_payload(
            "goal_spec",
            {"objective": "", "success_criteria": []},
            {},
        )
        self.assertEqual(untouched["objective"], "")
        self.assertEqual(no_actions, ())

    def test_ultra_does_not_restore_empty_master_plan_or_add_product_qa(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "master_plan",
            {"summary": "", "modules": []},
            {
                "goal_spec": {"objective": "Build a browser game with screenshot visual quality review"},
                "architecture": {
                    "summary": "Game architecture",
                    "components": [
                        {"name": "Renderer", "responsibility": "Render the 3D scene"},
                        {"name": "Gameplay", "responsibility": "Run combat and waves"},
                    ],
                },
            },
        )
        self.assertEqual(payload, {"summary": "", "modules": []})
        self.assertEqual(actions, ())
        with self.assertRaises(AgentProtocolError):
            UltraOrchestrator._validate_typed_response("master_plan", payload)

    def test_ultra_does_not_inject_browser_qa_from_screenshot_keywords(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "master_plan",
            {
                "summary": "Build and polish",
                "modules": [
                    {
                        "id": "M001",
                        "title": "Game State and Visual Polish",
                        "objective": "Finish the game and capture a screenshot",
                        "acceptance_criteria": ["The game looks polished"],
                        "verification": ["Capture a 1280x720 screenshot"],
                        "depends_on": [],
                        "write_paths": ["index.html"],
                    }
                ],
            },
            {"goal_spec": {"objective": "Build a browser game with screenshot visual quality review"}},
        )
        self.assertEqual(len(payload["modules"]), 1)
        self.assertEqual(payload["modules"][0]["title"], "Game State and Visual Polish")
        self.assertEqual(actions, ())

    def test_ultra_does_not_rewrite_model_authored_paths_from_keywords(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "master_plan",
            {
                "summary": "Build split game",
                "modules": [
                    {
                        "id": "M001",
                        "title": "Game",
                        "objective": "Implement game logic",
                        "acceptance_criteria": ["Game runs"],
                        "verification": ["Open browser"],
                        "depends_on": [],
                        "write_paths": ["index.html", "js/game.js"],
                    }
                ],
            },
            {
                "goal_spec": {
                    "objective": "Build one self-contained single-file index.html Three.js game using the jsDelivr CDN",
                    "constraints": ["No separate JavaScript or CSS files"],
                },
                "architecture": {},
            },
        )

        self.assertEqual(payload["modules"][0]["write_paths"], ["index.html", "js/game.js"])
        self.assertEqual(payload["modules"][0]["acceptance_criteria"], ["Game runs"])
        self.assertEqual(actions, ())

    def test_ultra_binds_literal_verifier_prerequisites_without_guessing_paths(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "master_plan",
            {
                "summary": "Build and verify the local calculator",
                "modules": [
                    {
                        "id": "M001",
                        "title": "Calculator",
                        "objective": "Implement the approved calculator",
                        "acceptance_criteria": ["Calculator works"],
                        "verification": [
                            "npm test",
                            "preview_html index.html",
                        ],
                        "depends_on": [],
                        "write_paths": ["src/calculator.js"],
                    }
                ],
            },
            {
                "goal_spec": {"objective": "Build the approved calculator"},
                "architecture": {},
            },
        )

        self.assertEqual(
            payload["modules"][0]["write_paths"],
            ["src/calculator.js", "package.json", "index.html"],
        )
        self.assertIn(
            "master_plan bound package.json prerequisite to its package verification module",
            actions,
        )
        self.assertIn(
            "master_plan bound literal preview target index.html to its verification module",
            actions,
        )

    def test_ultra_does_not_create_browser_qa_module_from_goal_text(self):
        payload, _ = UltraOrchestrator._normalize_typed_payload(
            "master_plan",
            {
                "summary": "Build",
                "modules": [{"id": "M001", "title": "Build", "objective": "Implement game"}],
            },
            {
                "goal_spec": {
                    "objective": "Implement the approved game",
                    "constraints": ["Pass browser QA and provide a real 1280x720 screenshot"],
                }
            },
        )
        self.assertEqual([item["title"] for item in payload["modules"]], ["Build"])

    def test_ultra_namespaces_sparse_child_without_inventing_its_contract(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "decompose",
            {"children": [{"id": "M001_Refinement", "finding": "Improve canyon depth and lighting contrast"}]},
            {"contract": {"id": "M001", "title": "Environment Generation & Visual Effects"}},
        )
        child = payload["children"][0]
        self.assertEqual(child["id"], "M001.1")
        self.assertEqual(child["finding"], "Improve canyon depth and lighting contrast")
        self.assertNotIn("title", child)
        self.assertNotIn("objective", child)
        self.assertNotIn("acceptance_criteria", child)
        self.assertNotIn("verification", child)
        self.assertEqual(actions, ("decompose child ids isolated under the parent contract",))

    def test_ultra_rejects_sparse_child_inside_typed_repair_boundary(self):
        payload, _ = UltraOrchestrator._normalize_typed_payload(
            "decompose",
            {
                "children": [
                    {
                        "id": "M001_Refinement",
                        "finding": "Read the component source before integration",
                    }
                ]
            },
            {"contract": {"id": "M001", "title": "Integration"}},
        )

        with self.assertRaisesRegex(
            AgentProtocolError,
            "requires a title and objective",
        ):
            UltraOrchestrator._validate_typed_response("decompose", payload)

        UltraOrchestrator._validate_typed_response(
            "decompose",
            {
                "children": [
                    {
                        "id": "M001.1",
                        "title": "Inspect component contracts",
                        "objective": "Inspect both approved component contracts",
                        "acceptance_criteria": ["Both contracts are understood"],
                        "verification": ["Compare both approved interfaces"],
                    }
                ]
            },
        )

    def test_ultra_drops_question_that_reopens_explicit_no_placeholder_constraint(self):
        question = {
            "header": "Asset Detail",
            "question": "Should placeholder geometry be used because this is a single-file build?",
            "reason": "Choose asset fidelity.",
        }
        prompt = "Build a production-quality single-file game that is not a placeholder."
        self.assertTrue(
            UltraOrchestrator._question_reopens_explicit_prompt_constraint(question, prompt)
        )
        self.assertFalse(
            UltraOrchestrator._question_reopens_explicit_prompt_constraint(
                {"question": "Which target platform should own the release?"}, prompt
            )
        )
        self.assertTrue(
            UltraOrchestrator._question_reopens_explicit_prompt_constraint(
                {"question": "What primary aspect ratio should the viewport use?"},
                "Capture a 1280x720 screenshot and remain responsive.",
            )
        )

    def test_ultra_drops_malformed_optional_questions_instead_of_stopping_goal(self):
        self.assertEqual(
            UltraOrchestrator._validated_questions(
            [
                {
                    "id": "particle_system_scope",
                    "header": "Particle System Scope",
                    "question": "Should the implementation focus on emissive GPU shaders for explosions or are basic THREE.Points sufficient?",
                    "reason": "This limits scope creep for particle fidelity.",
                    "options": [],
                },
                {
                    "id": "environmental_interaction",
                    "header": "Environmental Interactivity",
                    "question": "Are animated rails purely decorative or traversable collision geometry that affects player pathing?",
                    "reason": "This changes physics integration complexity.",
                    "options": [],
                },
            ]
            ),
            (),
        )

    def test_ultra_requires_targeted_repair_for_malformed_authority_question(self):
        with self.assertRaisesRegex(AgentProtocolError, "Repair this question only"):
            UltraOrchestrator._validated_questions(
            [
                {
                    "id": "enemy_precision",
                    "header": "Enemy Precision",
                    "question": "Must the melee enemy use exact hitbox timing or is proximity collision sufficient?",
                    "reason": "Clarify implementation detail.",
                    "options": [],
                    "decision_need": {
                        "impact": "The collision contract changes observable game behavior.",
                        "affected_scope": ["collision behavior"],
                        "affected_effects": ["workspace artifacts"],
                        "reversible": False,
                        "requires_user_authority": True,
                        "reason": "The requested behavior does not choose either contract.",
                        "evidence_refs": ["goal:original_request"],
                    },
                }
            ]
        )

    def test_ultra_drops_preferences_without_typed_authority_proof(self):
        questions = UltraOrchestrator._validated_questions(
            [
                {
                    "id": "touch",
                    "header": "Touch Input Method",
                    "question": "Should mobile controls use buttons, swipes, or both?",
                    "reason": "This changes UI layout and implementation complexity.",
                    "options": [{"label": "Both"}, {"label": "Buttons"}],
                },
                {
                    "id": "audio",
                    "header": "Optional Audio",
                    "question": "Should the game include procedural sound effects?",
                    "reason": "Audio is optional polish and can use a safe default.",
                    "options": [{"label": "Include"}, {"label": "Omit"}],
                },
            ]
        )
        self.assertEqual(questions, ())

    def test_ultra_edits_workspace_and_persists_every_quality_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(),
                ):
                    runtime = self._runtime(workspace, store)
                    master = runtime.start_ultra("Build the demo")
                    self.assertIsNotNone(master)
                    accepted = runtime.approve_ultra()
                    result = runtime.ultra_session.future.result(timeout=10)

                run = runtime.active_ultra_run()
                self.assertTrue(result.successful)
                self.assertEqual(store.get_goal(accepted.goal_id).status, GoalStatus.COMPLETED)
                self.assertEqual((workspace / "game.txt").read_text(), "ready\n")
                self.assertEqual(store.list_work_nodes(run.id)[0].status.value, "completed")
                agents = store.list_agent_runs(run.id)
                traces = store.list_prompt_traces(run.id)
                self.assertGreaterEqual(len(agents), 10)
                self.assertGreaterEqual(len(traces), 10)
                trace_ids = {trace.id for trace in traces}
                self.assertTrue(
                    all(
                        agent.prompt_trace_id in trace_ids
                        for agent in agents
                        if agent.status.value == "completed"
                    )
                )
                self.assertTrue(all(trace.agent_run_id for trace in traces))
                self.assertTrue(store.list_artifacts(run.id))
                policy, policy_fingerprint = store.get_quality_policy(run.id)
                self.assertEqual(policy.version, 1)
                self.assertEqual(policy_fingerprint, run.master_plan_fingerprint)
                change_sets = store.list_change_sets(run.id)
                self.assertTrue(change_sets)
                self.assertTrue(all(item.status.value == "integrated" for item in change_sets))
                self.assertTrue(
                    all(
                        item.review_status
                        == {"clean_code": "passed", "security": "passed", "test_quality": "passed"}
                        for item in change_sets
                    )
                )
                self.assertTrue(store.list_mutations(change_sets[0].id))
                cycles = store.list_quality_cycles(run.id)
                self.assertTrue(any(item.kind.value == "baseline" for item in cycles))
                self.assertTrue(any(item.kind.value == "delta" for item in cycles))
                registry = store.list_agent_registry(run.id)
                self.assertEqual(len(registry), len(agents))
                self.assertTrue(all(item.runtime_id for item in registry))
                swarm_updates = store.list_swarm_messages(
                    run.id,
                    recipient_agent_id="ultra-orchestrator",
                )
                agent_updates = [item for item in swarm_updates if item["message_type"] == "inform"]
                completed_agents = [item for item in agents if item.status.value == "completed"]
                self.assertEqual(len(agent_updates), len(completed_agents))
                self.assertTrue(all(item["payload"]["status"] == "completed" for item in agent_updates))
                self.assertTrue(any(item["message_type"] == "decision" for item in swarm_updates))
                benchmarks = store.list_benchmark_results(
                    suite_name="ultra-automatic-evaluation",
                    scenario_name="global-completion-gate",
                )
                self.assertEqual(benchmarks[0]["result"], "passed")
                self.assertEqual(benchmarks[0]["scores"]["global_success"], 1.0)
                self.assertGreater(benchmarks[0]["metrics"]["agent_runs"], 0)
                self.assertTrue(store.list_brain_entries(run.id))
                self.assertTrue(
                    store.list_brain_entries(run.id, section=BrainSection.TASK_GRAPH)
                )
                self.assertTrue(
                    store.list_brain_entries(run.id, section=BrainSection.ARTIFACT_INDEX)
                )
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_automatic_evaluation_gate_blocks_empty_final_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: EmptyFinalEvidenceProvider(),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    runtime.approve_ultra()
                    result = runtime.ultra_session.future.result(timeout=10)

                run = runtime.active_ultra_run()
                refreshed = store.get_ultra_run(run.id)
                self.assertFalse(result.successful)
                self.assertEqual(refreshed.status, UltraRunStatus.REVISION_REQUIRED)
                benchmarks = store.list_benchmark_results(
                    suite_name="ultra-automatic-evaluation",
                    scenario_name="global-completion-gate",
                )
                self.assertEqual(benchmarks[0]["result"], "failed")
                self.assertEqual(benchmarks[0]["scores"]["final_evidence_score"], 0.0)
                self.assertIn("no durable evidence", benchmarks[0]["blocker"])
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_automatically_blocks_weak_single_file_3d_html_benchmark(self):
        weak_html = "<!doctype html><html><title>3D Game</title><body><h1>3D Game</h1><p>Coming soon</p></body></html>"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: HtmlGameProvider(weak_html),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build a production-quality single-file 3D HTML game")
                    runtime.approve_ultra()
                    result = runtime.ultra_session.future.result(timeout=30)

                run = runtime.active_ultra_run()
                self.assertFalse(result.successful)
                self.assertEqual(
                    store.get_ultra_run(run.id).status,
                    UltraRunStatus.REVISION_REQUIRED,
                )
                html_benchmarks = store.list_benchmark_results(
                    suite_name="weak-model-html",
                    scenario_name="threejs-single-file",
                )
                # v9 rejects the provider before final assembly because it did
                # not publish runnable materialized specialist packages. The
                # old heuristic HTML benchmark is therefore not an acceptance
                # authority and must not manufacture a result for this run.
                self.assertTrue(
                    all(item["result"] == "failed" for item in html_benchmarks)
                )
                self.assertFalse(
                    any(
                        package.get("schema_name") == "MaterializedComponentPackageV2"
                        for package in store.list_component_packages(run.id)
                    )
                )
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_rejects_code_rich_but_nonfunctional_3d_html_benchmark(self):
        rich_html = """
<!doctype html><html><head><title>Neon Rift Arena</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:radial-gradient(circle,#102,#001);overflow:hidden}.hud{position:fixed;color:white;filter:drop-shadow(0 0 8px cyan)}</style></head>
<body><canvas id="game" aria-label="Neon 3D arena" role="img"></canvas><div class="hud">score health level</div>
<script>
const THREE = {
  Scene: class { constructor(){this.items=[]} add(x){this.items.push(x)} },
  Fog: class { constructor(){} },
  PerspectiveCamera: class { constructor(){this.aspect=1} updateProjectionMatrix(){} },
  WebGLRenderer: class { constructor(){this.shadowMap={enabled:false}} setSize(){} render(){} },
  AmbientLight: class { constructor(){} },
  PointLight: class { constructor(){} },
  MeshStandardMaterial: class { constructor(){} },
  BoxGeometry: class { constructor(){} },
  SphereGeometry: class { constructor(){} },
  Mesh: class { constructor(){this.rotation={y:0};this.position={distanceTo(){return 9}};this.castShadow=false} }
};
const scene = new THREE.Scene(); scene.fog = new THREE.Fog(0x020014, 10, 90);
const camera = new THREE.PerspectiveCamera(70, innerWidth/innerHeight, .1, 1000);
const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('game'), antialias:true});
renderer.setSize(innerWidth, innerHeight); renderer.shadowMap.enabled = true;
scene.add(new THREE.AmbientLight(0x3344ff, .5)); scene.add(new THREE.PointLight(0xff44cc, 2));
const material = new THREE.MeshStandardMaterial({color:0x33ffee, emissive:0x112244, roughness:.25, metalness:.7});
for(let i=0;i<30;i++){ const mesh = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), material); mesh.castShadow=true; scene.add(mesh); }
const enemies=[], projectiles=[], particles=[], trail=[]; let score=0, health=100, level=1, velocity={x:0,z:0}, bloom=true;
addEventListener('keydown', e => { velocity.x = e.key === 'ArrowRight' ? 1 : velocity.x; });
addEventListener('keyup', e => { velocity.x = 0; });
function collision(a,b){ return a.position && b.position && a.position.distanceTo(b.position) < 1.2; }
function spawnEnemy(){ enemies.push(new THREE.Mesh(new THREE.SphereGeometry(.5), material)); }
function fireProjectile(){ projectiles.push({hit:false, velocity:2}); }
function lerp(a,b,t){return a+(b-a)*t}
function animate(){ requestAnimationFrame(animate); enemies.forEach(e=>e.rotation.y+=.03); projectiles.forEach(p=>p.hit = p.hit || false); renderer.render(scene,camera); }
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
animate();
</script></body></html>
"""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: HtmlGameProvider(rich_html),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build a production-quality single-file 3D HTML game")
                    runtime.approve_ultra()
                    # Real browser startup can exceed ten seconds on a busy Windows host.
                    # This test asserts fail-closed quality behavior, not browser cold-start
                    # latency, so use the same bounded allowance as the weak-HTML case.
                    result = runtime.ultra_session.future.result(timeout=45)

                run = runtime.active_ultra_run()
                self.assertFalse(result.successful)
                self.assertEqual(
                    store.get_ultra_run(run.id).status,
                    UltraRunStatus.REVISION_REQUIRED,
                )
                html_benchmarks = store.list_benchmark_results(
                    suite_name="weak-model-html",
                    scenario_name="threejs-single-file",
                )
                self.assertTrue(
                    all(item["result"] == "failed" for item in html_benchmarks)
                )
                self.assertFalse(
                    any(
                        package.get("schema_name") == "MaterializedComponentPackageV2"
                        for package in store.list_component_packages(run.id)
                    )
                )
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_missing_reasoning_artifact_blocks_superficial_quality_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: MissingReasoningReviewProvider(),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    runtime.approve_ultra()
                    result = runtime.ultra_session.future.result(timeout=10)

                run = runtime.active_ultra_run()
                self.assertFalse(result.successful)
                self.assertEqual(store.get_ultra_run(run.id).status, UltraRunStatus.REVISION_REQUIRED)
                decisions = [
                    item
                    for item in store.list_swarm_messages(
                        run.id,
                        recipient_agent_id="ultra-orchestrator",
                        include_consumed=True,
                    )
                    if item["message_type"] == "decision"
                ]
                self.assertTrue(any(item["payload"]["status"] == "rejected" for item in decisions))
                rejected_votes = [
                    vote
                    for decision in decisions
                    for vote in decision["payload"].get("votes", ())
                    if vote["verdict"] == "reject"
                ]
                self.assertTrue(
                    any(
                        not vote["evidence"]["harness_reasoning_evaluation"]["passed"]
                        for vote in rejected_votes
                    )
                )
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_quality_consensus_rejection_blocks_superficial_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: ConsensusRejectProvider(),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    runtime.approve_ultra()
                    result = runtime.ultra_session.future.result(timeout=10)

                run = runtime.active_ultra_run()
                refreshed = store.get_ultra_run(run.id)
                self.assertFalse(result.successful)
                self.assertEqual(refreshed.status, UltraRunStatus.REVISION_REQUIRED)
                self.assertTrue(
                    any(item.status == "revision_required" for item in result.node_results)
                )
                decisions = [
                    item
                    for item in store.list_swarm_messages(
                        run.id,
                        recipient_agent_id="ultra-orchestrator",
                        include_consumed=True,
                    )
                    if item["message_type"] == "decision"
                ]
                self.assertTrue(decisions)
                self.assertTrue(any(item["payload"]["status"] == "rejected" for item in decisions))
                self.assertTrue(any("swarm_workflow" in item["payload"] for item in decisions))
                swarm_messages = store.list_swarm_messages(
                    run.id,
                    include_consumed=True,
                    limit=1000,
                )
                self.assertTrue(
                    any(
                        item["message_type"] == "proposal"
                        and item["topic"].startswith("quality-gate:")
                        for item in swarm_messages
                    )
                )
                self.assertTrue(
                    any(
                        item["message_type"] == "request"
                        and item["topic"].startswith("consensus-vote:")
                        for item in swarm_messages
                    )
                )
                self.assertTrue(
                    any(
                        item["message_type"] == "decision"
                        and item["recipient_agent_id"] == "swarm"
                        and item["topic"].startswith("consensus-decision:")
                        for item in swarm_messages
                    )
                )
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_lease_snapshot_blocks_external_stale_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "game.txt").write_text("original\n")
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: StaleWriteProvider(workspace),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    runtime.approve_ultra()
                    result = runtime.ultra_session.future.result(timeout=10)

                run = runtime.active_ultra_run()
                self.assertFalse(result.successful)
                self.assertEqual((workspace / "game.txt").read_text(), "external update\n")
                self.assertIn(
                    "conflict",
                    {node.status.value for node in store.list_work_nodes(run.id)},
                )
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_running_agent_is_visible_before_its_prompt_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            started = threading.Event()
            release = threading.Event()
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: BlockingProvider(started, release),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    runtime.approve_ultra()
                    self.assertTrue(started.wait(5))
                    run = runtime.active_ultra_run()
                    active = [
                        agent
                        for agent in store.list_agent_runs(run.id)
                        if agent.status.value == "running"
                    ]
                    self.assertTrue(active)
                    self.assertTrue(any(agent.phase == "implement" for agent in active))
                    release.set()
                    self.assertTrue(runtime.ultra_session.future.result(timeout=10).successful)
            finally:
                release.set()
                if runtime:
                    runtime.close()
                store.close()

    def test_paused_ultra_can_switch_model_after_agents_reach_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            started = threading.Event()
            release = threading.Event()
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: BlockingProvider(started, release),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    runtime.approve_ultra()
                    self.assertTrue(started.wait(5))
                    runtime.pause()
                    cloud = ModelDescriptor(
                        "openai",
                        "offline-cloud",
                        ExecutionClass.CLOUD,
                        capabilities=("tools",),
                    )
                    replacement = PhaseProvider()
                    replacement.model = "offline-cloud"
                    with self.assertRaises(RuntimeStateError):
                        runtime.replace_provider(replacement, cloud)

                    release.set()
                    deadline = time.monotonic() + 5
                    while (
                        not runtime.ultra_session.safe_for_reconfiguration
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    self.assertTrue(runtime.ultra_session.safe_for_reconfiguration)
                    runtime.replace_provider(replacement, cloud)
                    stored = runtime.active_ultra_run()
                    self.assertEqual(stored.execution_class, ExecutionClass.CLOUD)
                    # Cloud is a location, not a capability claim. This
                    # descriptor has no documented provider/hardware
                    # concurrency, so the conservative worker limit remains 1.
                    self.assertEqual(stored.concurrency, 1)
                    runtime.resume()
                    self.assertTrue(runtime.ultra_session.future.result(timeout=10).successful)
            finally:
                release.set()
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_question_answer_is_bound_into_master_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(ask_question=True),
                ):
                    runtime = self._runtime(workspace, store, ask_question=True)
                    self.assertIsNone(runtime.start_ultra("Build the demo"))
                    self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
                    master = runtime.answer_ultra_question("platform", "Desktop")

                self.assertIn('"platform":"desktop"', master.execution_strategy)
                self.assertEqual(
                    runtime.latest_plan().fingerprint,
                    store.get_latest_plan(runtime.active_goal().id).fingerprint,
                )
                self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_malformed_optional_question_does_not_pause_foundation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            provider = PhaseProvider(malformed_optional_question=True)
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    return_value=provider,
                ):
                    runtime = self._runtime(workspace, store)
                    plan = runtime.start_ultra("Build the demo")

                self.assertIsNotNone(plan)
                self.assertEqual(
                    runtime.active_goal().status,
                    GoalStatus.AWAITING_PLAN_APPROVAL,
                )
                self.assertEqual(runtime.ultra_session.orchestrator.goal_spec.questions, ())
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_question_boundary_restores_in_a_new_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_store = StateStore(workspace)
            first = second = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(ask_question=True),
                ):
                    first = self._runtime(
                        workspace,
                        first_store,
                        ask_question=True,
                    )
                    self.assertIsNone(first.start_ultra("Build the demo"))
                    goal_id = first.active_goal().id
                    run_id = first.active_ultra_run().id
                    self.assertEqual(
                        first.active_goal().status,
                        GoalStatus.PAUSED,
                    )
                    first.close()
                    first_store.close()
                    first = None

                    second_store = StateStore(workspace)
                    try:
                        second = self._runtime(
                            workspace,
                            second_store,
                            ask_question=True,
                        )
                        self.assertIsNone(second.ultra_session)
                        master = second.answer_ultra_question(
                            "platform",
                            "Desktop",
                        )
                        self.assertIsNotNone(master)
                        self.assertIn('"platform":"desktop"', master.execution_strategy)
                        self.assertEqual(second.active_goal().id, goal_id)
                        self.assertEqual(second.active_ultra_run().id, run_id)
                        self.assertEqual(
                            second.active_goal().status,
                            GoalStatus.AWAITING_PLAN_APPROVAL,
                        )
                    finally:
                        if second:
                            second.close()
                        second_store.close()
                        second = None
            finally:
                if first:
                    first.close()
                try:
                    first_store.close()
                except Exception:
                    pass

    def test_pending_ultra_plan_restart_materializes_nodes_before_specialists(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_store = StateStore(workspace)
            first = second = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(),
                ):
                    first = self._runtime(workspace, first_store)
                    first.start_ultra("Build the demo")
                    plan = first.latest_plan()
                    self.assertIsNotNone(plan)
                    goal_id = first.active_goal().id
                    run_id = first.active_ultra_run().id
                    self.assertIsNotNone(plan)
                    self.assertEqual(
                        first.active_goal().status,
                        GoalStatus.AWAITING_PLAN_APPROVAL,
                    )
                    first.close()
                    first_store.close()
                    first = None

                    second_store = StateStore(workspace)
                    try:
                        second = self._runtime(workspace, second_store)
                        accepted = second.approve_ultra()
                        result = second.ultra_session.future.result(timeout=10)

                        self.assertEqual(accepted.revision, plan.revision)
                        self.assertTrue(result.successful)
                        self.assertEqual(
                            second_store.get_goal(goal_id).status,
                            GoalStatus.COMPLETED,
                        )
                        self.assertTrue(second_store.list_work_nodes(run_id))
                        self.assertTrue(
                            second_store.list_specialist_profiles(run_id)
                        )
                    finally:
                        if second:
                            second.close()
                        second_store.close()
                        second = None
            finally:
                if first:
                    first.close()
                try:
                    first_store.close()
                except Exception:
                    pass

    def test_approved_ultra_restart_rebuilds_missing_top_level_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_store = StateStore(workspace)
            first = second = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(),
                ):
                    first = self._runtime(workspace, first_store)
                    first.start_ultra("Build the demo")
                    plan = first.latest_plan()
                    self.assertIsNotNone(plan)
                    goal_id = first.active_goal().id
                    run_id = first.active_ultra_run().id
                    accepted, _ = first_store.approve_plan(
                        goal_id,
                        plan.revision,
                        approved_by="user",
                        expected_fingerprint=plan.fingerprint,
                    )
                    first_store.approve_ultra_master(
                        run_id,
                        accepted.revision,
                        accepted.fingerprint,
                        approved_by="user",
                    )
                    first_store.update_goal_metadata(
                        goal_id,
                        resume_status=GoalStatus.RUNNING.value,
                    )
                    first_store.transition_goal(
                        goal_id,
                        GoalStatus.PAUSED,
                        reason="simulated stop before WorkNode materialization",
                    )
                    self.assertFalse(first_store.list_work_nodes(run_id))
                    first.close()
                    first_store.close()
                    first = None

                    second_store = StateStore(workspace)
                    try:
                        second = self._runtime(workspace, second_store)
                        second.resume()
                        result = second.ultra_session.future.result(timeout=10)

                        self.assertTrue(result.successful)
                        self.assertTrue(second_store.list_work_nodes(run_id))
                        self.assertTrue(
                            second_store.list_specialist_profiles(run_id)
                        )
                    finally:
                        if second:
                            second.close()
                        second_store.close()
                        second = None
            finally:
                if first:
                    first.close()
                try:
                    first_store.close()
                except Exception:
                    pass

    def test_ultra_replan_creates_a_new_master_approval_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(),
                ):
                    runtime = self._runtime(workspace, store)
                    runtime.start_ultra("Build the demo")
                    old_run = runtime.active_ultra_run()
                    revised = runtime.replan_ultra("Target a revised public interface")

                new_run = runtime.active_ultra_run()
                self.assertNotEqual(old_run.id, new_run.id)
                self.assertEqual(store.get_ultra_run(old_run.id).status, UltraRunStatus.BLOCKED)
                self.assertEqual(
                    new_run.goal_spec.to_dict(),
                    old_run.goal_spec.to_dict(),
                )
                self.assertEqual(
                    new_run.architecture_spec.to_dict(),
                    old_run.architecture_spec.to_dict(),
                )
                self.assertTrue(
                    bool(new_run.config.get("accepted_foundation_reused"))
                )
                self.assertEqual(runtime.latest_plan().revision, 2)
                self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
                self.assertIsNotNone(revised)
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_approved_run_rebuilds_from_sqlite_evidence_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            first = second = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(),
                ):
                    first = self._runtime(workspace, store)
                    first.start_ultra("Build the demo")
                    orchestrator = first.ultra_session.orchestrator
                    adapter = first.ultra_session.adapter
                    orchestrator.approve(orchestrator.master_plan.fingerprint)
                    accepted = adapter.approve_master(orchestrator.master_plan)
                    run_id = adapter.run_id
                    store.update_ultra_run(
                        run_id,
                        status=UltraRunStatus.RECOVERING,
                        phase=UltraPhase.MODULE_WAVES,
                    )
                    store.update_goal_metadata(
                        accepted.goal_id,
                        ultra_run_id=run_id,
                        resume_status=GoalStatus.RUNNING.value,
                    )
                    store.transition_goal(
                        accepted.goal_id,
                        GoalStatus.PAUSED,
                        reason="simulated restart",
                    )
                    first.close()
                    first = None

                    second = self._runtime(workspace, store)
                    second.resume()
                    result = second.ultra_session.future.result(timeout=10)

                self.assertTrue(result.successful)
                self.assertEqual(store.get_goal(accepted.goal_id).status, GoalStatus.COMPLETED)
                self.assertEqual((workspace / "game.txt").read_text(), "ready\n")
            finally:
                if first:
                    first.close()
                if second:
                    second.close()
                store.close()

    def test_recoverable_blocked_ultra_resumes_from_sqlite_checkpoint(self):
        """A fixed harness/provider failure must not force foundation replay."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            first = second = None
            try:
                with mock.patch.object(
                    ModelDescriptor,
                    "create_provider",
                    lambda _self: PhaseProvider(),
                ):
                    first = self._runtime(workspace, store)
                    first.start_ultra("Build the demo")
                    orchestrator = first.ultra_session.orchestrator
                    adapter = first.ultra_session.adapter
                    orchestrator.approve(orchestrator.master_plan.fingerprint)
                    accepted = adapter.approve_master(orchestrator.master_plan)
                    run_id = adapter.run_id
                    store.update_ultra_run(
                        run_id,
                        status=UltraRunStatus.BLOCKED,
                        phase=UltraPhase.MODULE_WAVES,
                    )
                    task = store.get_plan(
                        accepted.goal_id,
                        accepted.revision,
                    ).tasks[0]
                    store.transition_task(
                        accepted.goal_id,
                        accepted.revision,
                        task.id,
                        TaskStatus.BLOCKED,
                        note="simulated recoverable module-wave failure",
                        actor="test",
                    )
                    root_node = next(
                        item
                        for item in store.list_work_nodes(run_id)
                        if item.parent_id is None
                    )
                    store.transition_work_node(
                        root_node.id,
                        WorkNodeStatus.BLOCKED,
                        error="simulated recoverable module-wave failure",
                    )
                    store.transition_goal(
                        accepted.goal_id,
                        GoalStatus.BLOCKED,
                        reason="simulated recoverable provider/parser failure",
                    )
                    first.close()
                    first = None

                    second = self._runtime(workspace, store)
                    second.resume()
                    result = second.ultra_session.future.result(timeout=10)

                self.assertTrue(result.successful)
                self.assertEqual(store.get_goal(accepted.goal_id).status, GoalStatus.COMPLETED)
                self.assertEqual((workspace / "game.txt").read_text(), "ready\n")
                self.assertEqual(
                    store.get_plan(accepted.goal_id, accepted.revision).tasks[0].status,
                    TaskStatus.COMPLETED,
                )
            finally:
                if first:
                    first.close()
                if second:
                    second.close()
                store.close()

    def test_plan_mode_questions_are_durable_and_fingerprint_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = AgentRuntime(
                PlanningQuestionProvider(),
                store,
                workspace,
                events=EventBus(),
            )
            try:
                self.assertIsNone(runtime.start_goal("Create an application"))
                self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
                self.assertEqual(runtime.plan_questions()[0]["id"], "platform")
                plan = runtime.answer_plan_question("platform", "Desktop")
                self.assertIsNotNone(plan)
                self.assertIn('"platform":"desktop"', plan.execution_strategy)
                self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
            finally:
                runtime.close()
                store.close()

    def test_plain_text_answers_the_visible_plan_question(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            runtime = AgentRuntime(
                PlanningQuestionProvider(),
                store,
                workspace,
                events=EventBus(),
            )
            try:
                self.assertIsNone(runtime.start_goal("Create an application"))

                plan = runtime.apply_command(parse_command("Desktop"))

                self.assertIsNotNone(plan)
                self.assertIn('"platform":"desktop"', plan.execution_strategy)
                self.assertEqual(
                    runtime.active_goal().status,
                    GoalStatus.AWAITING_PLAN_APPROVAL,
                )
            finally:
                runtime.close()
                store.close()


if __name__ == "__main__":
    unittest.main()

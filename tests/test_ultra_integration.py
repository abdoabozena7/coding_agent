from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from agent.events import EventBus
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.models import GoalStatus
from agent.providers.base import AssistantTurn, ToolCall, Usage
from agent.runtime import AgentRuntime, RuntimeStateError
from agent.sandbox import DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.ultra import AgentRequest, AgentRole
from agent.ultra_models import BrainSection, UltraPhase, UltraRunStatus
from agent.ultra_session import WorkspaceUltraAgent


class PhaseProvider:
    """Offline provider that follows every ULTRA phase and performs one edit."""

    model = "offline-ultra"

    def __init__(self, *, ask_question: bool = False) -> None:
        self.calls = 0
        self.ask_question = ask_question

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
                                "label": "Desktop",
                                "description": "Keyboard-first desktop build.",
                                "recommended": True,
                            },
                            {
                                "label": "Web",
                                "description": "Browser-first deployment.",
                                "recommended": False,
                            },
                        ],
                        "allow_freeform": False,
                        "reason": "The product target is not encoded in the repository.",
                    }
                ]
                if self.ask_question
                else []
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
                                            "label": "Desktop",
                                            "description": "Desktop application.",
                                            "recommended": True,
                                        },
                                        {
                                            "label": "Web",
                                            "description": "Browser application.",
                                            "recommended": False,
                                        },
                                    ],
                                    "allow_freeform": False,
                                    "reason": "Product scope is not discoverable from this empty workspace.",
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


class UltraIntegrationTests(unittest.TestCase):
    def _descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            "ollama",
            "offline-ultra",
            ExecutionClass.LOCAL,
            capabilities=("tools",),
        )

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

    def test_ultra_goal_spec_requires_workspace_inspection(self):
        agent = WorkspaceUltraAgent(
            FinalOnlyGoalProvider(),
            role=AgentRole.GOAL_UNDERSTANDING,
            provider_name="offline",
            model="final-only",
            executor=lambda _call, _request: "ok",
            events=EventBus(),
            max_steps=2,
        )
        with self.assertRaisesRegex(RuntimeError, "repository inspection"):
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
                    self.assertEqual(stored.concurrency, 4)
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

                self.assertIn("Desktop", master.execution_strategy)
                self.assertEqual(
                    runtime.latest_plan().fingerprint,
                    store.get_latest_plan(runtime.active_goal().id).fingerprint,
                )
                self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
            finally:
                if runtime:
                    runtime.close()
                store.close()

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
                self.assertIn("Desktop", plan.execution_strategy)
                self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
            finally:
                runtime.close()
                store.close()


if __name__ == "__main__":
    unittest.main()

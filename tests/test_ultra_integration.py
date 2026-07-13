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
from agent.ultra import AgentRequest, AgentRole, NodeStatus
from agent.ultra import MasterPlanV1, TaskContractV1, UltraOrchestrator, _with_quality_milestone
from agent.ultra_models import BrainSection, UltraPhase, UltraRunStatus
from agent.ultra_session import WorkspaceUltraAgent, WorkspaceUltraAgentFactory, _store_node_status
from agent.ultra_models import WorkNodeStatus


def test_ultra_planning_is_not_persisted_as_execution_in_progress():
    assert _store_node_status(NodeStatus.PLANNING) is WorkNodeStatus.PENDING


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


class HtmlGameProvider(PhaseProvider):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.html = html

    def call(self, conversation, tools, system, on_text=None, on_thought=None):
        phase = self._phase(system)
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
        self.assertEqual(provider.max_output_tokens, 2048)
        self.assertTrue(provider.force_json)

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
        self.assertIn("Harness browser verification failed", response.payload["issues"][-1])
        self.assertIn("THREE is not defined", response.payload["findings"])
        self.assertEqual(response.payload["test_results"][-1]["name"], "harness_html_preview")

    def test_ultra_drops_verification_mechanics_questions_but_keeps_product_decisions(self):
        questions = UltraOrchestrator._validated_questions(
            [
                {
                    "id": "verify",
                    "header": "Verification Mechanism",
                    "question": "Should read-back verify content or also file metadata?",
                    "reason": "Choose the verification method.",
                    "options": [],
                },
                {
                    "id": "combat",
                    "header": "Combat Complexity Balance",
                    "question": "Should enemy behavior use ranged and melee threat vectors?",
                    "reason": "This guides implementation complexity and AI state machine depth.",
                    "options": [],
                },
                {
                    "id": "platform",
                    "header": "Platform",
                    "question": "Which target platform should own the first release?",
                    "reason": "This changes product behavior and deployment.",
                    "options": [
                        {"label": "Web", "description": "Browser release."},
                        {"label": "Desktop", "description": "Native release."},
                    ],
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

    def test_ultra_restores_missing_goal_objective_only_from_authoritative_prompt(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "goal_spec",
            {"objective": "", "success_criteria": ["Playable game exists"]},
            {"prompt": "Build the requested game"},
        )
        self.assertEqual(payload["objective"], "Build the requested game")
        self.assertTrue(actions)

        untouched, no_actions = UltraOrchestrator._normalize_typed_payload(
            "goal_spec",
            {"objective": "", "success_criteria": []},
            {},
        )
        self.assertEqual(untouched["objective"], "")
        self.assertEqual(no_actions, ())

    def test_ultra_restores_empty_master_plan_from_architecture_and_adds_browser_qa_gate(self):
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
        self.assertEqual(payload["summary"], "Game architecture")
        self.assertEqual(payload["modules"][-1]["title"], "Browser QA and Visual Refinement Gate")
        self.assertIn("1280x720", payload["modules"][-1]["acceptance_criteria"][0])
        self.assertTrue(actions)

    def test_ultra_does_not_treat_screenshot_inside_build_module_as_browser_qa_gate(self):
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
        self.assertEqual(len(payload["modules"]), 2)
        self.assertEqual(payload["modules"][-1]["title"], "Browser QA and Visual Refinement Gate")
        self.assertIn("Browser QA gate added", " ".join(actions))

    def test_ultra_finds_browser_qa_requirement_outside_terse_objective(self):
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
        self.assertEqual(payload["modules"][-1]["title"], "Browser QA and Visual Refinement Gate")

    def test_ultra_restores_sparse_decompose_refinement_child_contract(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "decompose",
            {"children": [{"id": "M001_Refinement", "finding": "Improve canyon depth and lighting contrast"}]},
            {"contract": {"id": "M001", "title": "Environment Generation & Visual Effects"}},
        )
        child = payload["children"][0]
        self.assertEqual(child["id"], "M001_Refinement")
        self.assertIn("Environment Generation & Visual Effects", child["title"])
        self.assertIn("Improve canyon depth", child["objective"])
        self.assertTrue(child["acceptance_criteria"])
        self.assertTrue(child["verification"])
        self.assertTrue(actions)

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

    def test_ultra_drops_particle_and_environment_scope_questions_as_implementation_policy(self):
        questions = UltraOrchestrator._validated_questions(
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
        )
        self.assertEqual(questions, ())

    def test_ultra_drops_open_ended_questions_without_bounded_choices(self):
        questions = UltraOrchestrator._validated_questions(
            [
                {
                    "id": "enemy_precision",
                    "header": "Enemy Precision",
                    "question": "Must the melee enemy use exact hitbox timing or is proximity collision sufficient?",
                    "reason": "Clarify implementation detail.",
                    "options": [],
                }
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
                self.assertEqual(store.get_ultra_run(run.id).status, UltraRunStatus.REVISION_REQUIRED)
                html_benchmarks = store.list_benchmark_results(
                    suite_name="weak-model-html",
                    scenario_name="threejs-single-file",
                )
                self.assertEqual(html_benchmarks[0]["result"], "failed")
                self.assertLess(html_benchmarks[0]["scores"]["overall"], 0.8)
                self.assertIn("3D/WebGL", html_benchmarks[0]["blocker"])
            finally:
                if runtime:
                    runtime.close()
                store.close()

    def test_ultra_accepts_rich_single_file_3d_html_benchmark(self):
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
                    result = runtime.ultra_session.future.result(timeout=10)

                run = runtime.active_ultra_run()
                self.assertTrue(result.successful)
                self.assertEqual(store.get_ultra_run(run.id).status, UltraRunStatus.COMPLETED)
                html_benchmarks = store.list_benchmark_results(
                    suite_name="weak-model-html",
                    scenario_name="threejs-single-file",
                )
                self.assertEqual(html_benchmarks[0]["result"], "passed")
                self.assertGreaterEqual(html_benchmarks[0]["scores"]["overall"], 0.8)
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

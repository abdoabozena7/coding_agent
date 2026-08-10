from __future__ import annotations

import threading
import time
import unittest
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from agent.events import EventBus
from agent.scheduler import (
    AdaptiveConcurrency,
    DeterministicWaveScheduler,
    ExecutionClass,
    RateLimitError,
    ResourceLeaseManager,
    ScheduleStatus,
)
from agent.reasoning import evaluate_reasoning_artifact, repair_reasoning_artifact_graph, reasoning_debate_protocol_for
from agent.ultra import (
    AgentProtocolError,
    AgentRequest,
    AgentResponse,
    AgentRole,
    InMemoryUltraState,
    InnerPhase,
    MasterPlanV1,
    NodeStatus,
    ResultPackageV1,
    TaskContractV1,
    UltraConfig,
    UltraOrchestrator,
    UltraPhase,
    _bind_explicit_browser_scenarios,
)


def test_runtime_effect_plan_requires_concrete_applicable_verification() -> None:
    vague = MasterPlanV1(
        summary="Build the app.",
        modules=(
            TaskContractV1(
                id="M001",
                title="App",
                objective="Build it.",
                acceptance_criteria=("The app runs.",),
                verification=("Run tests on the app",),
                write_paths=("src/app.js",),
            ),
        ),
    )
    issues = UltraOrchestrator._master_plan_applicability_issues(
        vague,
        requested_effects=("run", "preview"),
    )
    assert any("concrete executable" in item for item in issues)
    assert any("preview_html" in item for item in issues)

    executable = MasterPlanV1(
        summary="Build and verify the app.",
        modules=(
            TaskContractV1(
                id="M001",
                title="Runnable app",
                objective="Build it.",
                acceptance_criteria=("The app runs.",),
                verification=("npm test", "preview_html index.html"),
                write_paths=("package.json", "index.html", "tests/app.test.js"),
            ),
        ),
    )
    assert UltraOrchestrator._master_plan_applicability_issues(
        executable,
        requested_effects=("run", "preview"),
    ) == ()
    scope_issues = UltraOrchestrator._master_plan_applicability_issues(
        executable,
        requested_effects=("run", "preview"),
        approved_write_paths=("package.json", "index.html"),
    )
    assert any("tests/app.test.js" in item for item in scope_issues)

    placeholder = MasterPlanV1(
        summary="Build with placeholder paths.",
        modules=(
            TaskContractV1(
                id="M001",
                title="App",
                objective="Build it.",
                acceptance_criteria=("The app runs.",),
                verification=("npm test",),
                write_paths=("path/to/App/index.js", "package.json"),
            ),
        ),
    )
    placeholder_issues = UltraOrchestrator._master_plan_applicability_issues(
        placeholder,
        requested_effects=("run",),
    )
    assert any("contain placeholders" in item for item in placeholder_issues)

    broad_root = MasterPlanV1(
        summary="Broad workspace root plan.",
        modules=(
            TaskContractV1(
                id="M001",
                title="Integration",
                objective="Integrate the approved artifacts.",
                acceptance_criteria=("Integration passes.",),
                verification=("pytest -q",),
                write_paths=(".",),
            ),
        ),
    )
    root_issues = UltraOrchestrator._master_plan_applicability_issues(
        broad_root,
        requested_effects=("run",),
    )
    assert any("contain placeholders" in item and "." in item for item in root_issues)

    non_html_preview = MasterPlanV1(
        summary="Invalid JavaScript preview target.",
        modules=(
            TaskContractV1(
                id="M001",
                title="UI source",
                objective="Create the UI source module.",
                acceptance_criteria=("The UI source exists.",),
                verification=("preview_html src/index.js",),
                write_paths=("src/index.js",),
            ),
        ),
    )
    preview_issues = UltraOrchestrator._master_plan_applicability_issues(
        non_html_preview,
        requested_effects=("preview",),
    )
    assert any("must be an HTML artifact" in item for item in preview_issues)

    missing_scope = MasterPlanV1(
        summary="Module without a write lease.",
        modules=(
            TaskContractV1(
                id="M001",
                title="Integration",
                objective="Integrate the runnable application.",
                acceptance_criteria=("Integration passes.",),
                verification=("pytest -q",),
                write_paths=(),
            ),
        ),
    )
    missing_scope_issues = UltraOrchestrator._master_plan_applicability_issues(
        missing_scope,
        requested_effects=("run", "write"),
    )
    assert any("requires concrete approval-bound write_paths" in item for item in missing_scope_issues)


def test_master_plan_normalizes_conceptual_workspace_root() -> None:
    contract = TaskContractV1(
        id="M001",
        title="Repair app",
        objective="Repair the approved app.",
        acceptance_criteria=("The repair passes.",),
        verification=("npm test",),
        write_paths=("workspace/src/App.js", "workspace/package.json"),
    )
    assert contract.write_paths == ("src/App.js", "package.json")


def test_interactive_html_acceptance_requires_typed_browser_scenarios() -> None:
    base = TaskContractV1(
        id="M001",
        title="Calculator",
        objective="Repair calculator behavior.",
        acceptance_criteria=("Click 7 + 5 and display 12; keyboard Enter works.",),
        verification=("preview_html src/index.html",),
        write_paths=("src/index.html",),
    )
    missing = MasterPlanV1(summary="Repair", modules=(base,))

    issues = UltraOrchestrator._master_plan_applicability_issues(
        missing,
        requested_effects=("preview",),
    )
    assert any("browser_scenarios" in item for item in issues)

    scenario = {
        "name": "addition",
        "steps": [
            {"action": "click", "role": "button", "name": "7"},
            {"action": "click", "role": "button", "name": "Add"},
            {"action": "click", "role": "button", "name": "5"},
            {"action": "click", "role": "button", "name": "Equals"},
        ],
        "assertions": [
            {
                "role": "textbox",
                "name": "Calculator display",
                "property": "value",
                "equals": "12",
            }
        ],
    }
    bound = MasterPlanV1(
        summary="Repair with executable interaction evidence.",
        modules=(replace(base, metadata={"browser_scenarios": [scenario]}),),
    )
    assert UltraOrchestrator._master_plan_applicability_issues(
        bound,
        requested_effects=("preview",),
    ) == ()


def test_explicit_browser_scenarios_are_bound_before_plan_applicability_review() -> None:
    plan = MasterPlanV1(
        summary="Lossy small-model plan",
        modules=(
            TaskContractV1(
                id="M001",
                title="Verify",
                objective="Open the page.",
                acceptance_criteria=("No errors.",),
                verification=("preview_html src/index.html",),
                write_paths=("src/index.html",),
            ),
        ),
    )
    authority = (
        'metadata.browser_scenarios: [{"name":"addition",'
        '"steps":[{"action":"click","role":"button","name":"7"}],'
        '"assertions":[{"role":"textbox","name":"Display",'
        '"property":"value","equals":"7"}]}]'
    )

    bound = _bind_explicit_browser_scenarios(plan, authority)

    assert bound.modules[0].metadata["browser_scenarios"][0]["name"] == "addition"

    verification_bound = _bind_explicit_browser_scenarios(
        plan,
        authority + " Preserve the current implementation; this is verification-focused.",
    )
    assert verification_bound.modules[0].metadata["verification_only"] is True


def test_master_plan_separates_preview_scenario_from_literal_html_target() -> None:
    payload, actions = UltraOrchestrator._normalize_typed_payload(
        "master_plan",
        {
            "summary": "Build and verify.",
            "modules": [
                {
                    "id": "M001",
                    "title": "Calculator",
                    "objective": "Build it.",
                    "acceptance_criteria": ["Calculator works."],
                    "verification": [
                        "preview_html src/index.html plus 7+5=12",
                        "preview_html src/index.html divide-by-zero check",
                    ],
                    "write_paths": ["src/index.html"],
                }
            ],
        },
        {"protocol_node_namespace": "rtest"},
    )

    module_payload = payload["modules"][0]
    assert module_payload["verification"] == ["preview_html src/index.html"]
    assert module_payload["write_paths"] == ["src/index.html"]
    assert "Browser verification scenario: plus 7+5=12" in module_payload[
        "acceptance_criteria"
    ]
    assert "Browser verification scenario: divide-by-zero check" in module_payload[
        "acceptance_criteria"
    ]
    assert any("separated preview scenario prose" in item for item in actions)


def test_browser_scenario_transport_normalizes_executor_aliases() -> None:
    payload, actions = UltraOrchestrator._normalize_typed_payload(
        "browser_scenarios",
        {
            "modules": [
                {
                    "browser_scenarios": [
                        {
                            "module_id": "M001",
                            "name": "addition",
                            "steps": [
                                {
                                    "action": "type",
                                    "role": "textbox",
                                    "accessible_name": "First number",
                                    "text": "123",
                                }
                            ],
                            "assertions": [
                                {
                                    "role": "h2",
                                    "accessible name": "Result",
                                    "property": "text",
                                    "value": "123",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        {},
    )

    scenario = payload["modules"][0]["browser_scenarios"][0]
    assert payload["modules"][0]["module_id"] == "M001"
    assert "module_id" not in scenario
    assert scenario["steps"][0] == {
        "action": "fill",
        "role": "textbox",
        "name": "First number",
        "value": "123",
    }
    assert scenario["assertions"][0] == {
        "role": "heading",
        "name": "Result",
        "property": "text",
        "equals": "123",
    }
    assert any("type action normalized to fill" in item for item in actions)
    UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_project_136_browser_transport_removes_click_noise_and_preserves_dom_assertions() -> None:
    payload, actions = UltraOrchestrator._normalize_typed_payload(
        "browser_scenarios",
        {
            "modules": [
                {
                    "module_id": "r0bec63abc011.M001",
                    "browser_scenarios": [
                        {
                            "name": "Verify basic HTML load",
                            "steps": [
                                {
                                    "action": "click",
                                    "role": "button",
                                    "name": "Initial Button Check",
                                    "text": "",
                                }
                            ],
                            "assertions": [
                                {
                                    "role": "textbox",
                                    "name": "Canvas Existence",
                                    "property": "id",
                                    "equals": "threeD-canvas",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        {"modules": [{"module_id": "r0bec63abc011.M001"}]},
    )

    scenario = payload["modules"][0]["browser_scenarios"][0]
    assert "text" not in scenario["steps"][0]
    assert scenario["assertions"][0]["property"] == "id"
    assert any("empty text transport key removed" in item for item in actions)
    UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_visible_element_count_alias_is_bounded_observable_property() -> None:
    payload, _actions = UltraOrchestrator._normalize_typed_payload(
        "browser_scenarios",
        {
            "modules": [
                {
                    "module_id": "M003",
                    "browser_scenarios": [
                        {
                            "name": "3D state count",
                            "steps": [{"action": "click", "selector": "#equals"}],
                            "assertions": [
                                {
                                    "selector": "#scene [data-object]",
                                    "property": "visible_element_count",
                                    "equals": "3",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        {"modules": [{"module_id": "M003"}]},
    )

    assert (
        payload["modules"][0]["browser_scenarios"][0]["assertions"][0][
            "property"
        ]
        == "visibleCount"
    )
    UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_graphics_data_observable_aliases_are_bounded_properties() -> None:
    for supplied, expected in (
        ("object_count", "dataObjectCount"),
        ("visual_state", "dataVisualState"),
    ):
        payload, _actions = UltraOrchestrator._normalize_typed_payload(
            "browser_scenarios",
            {
                "modules": [{
                    "module_id": "M003",
                    "browser_scenarios": [{
                        "name": "3D state",
                        "steps": [{"action": "click", "selector": "#equals"}],
                        "assertions": [{
                            "selector": "#threeD-canvas",
                            "property": supplied,
                            "equals": "3" if expected == "dataObjectCount" else "5+3=8",
                        }],
                    }],
                }],
            },
            {"modules": [{"module_id": "M003"}]},
        )
        assertion = payload["modules"][0]["browser_scenarios"][0]["assertions"][0]
        assert assertion["property"] == expected
        UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_browser_assertion_expected_alias_and_duplicate_comparator_are_normalized() -> None:
    payload, actions = UltraOrchestrator._normalize_typed_payload(
        "browser_scenarios",
        {
            "modules": [{
                "module_id": "M003",
                "browser_scenarios": [{
                    "name": "result",
                    "steps": [{"action": "click", "selector": "#equals"}],
                    "assertions": [{
                        "selector": "#display",
                        "property": "textContent",
                        "expected": 8,
                        "contains": "8",
                    }],
                }],
            }],
        },
        {"modules": [{"module_id": "M003"}]},
    )
    assertion = payload["modules"][0]["browser_scenarios"][0]["assertions"][0]
    assert assertion == {
        "selector": "#display",
        "property": "textContent",
        "equals": "8",
    }
    assert any("expected value alias" in item for item in actions)
    assert any("duplicate contains" in item for item in actions)
    UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_browser_assertion_rejects_conflicting_comparators() -> None:
    payload = {
        "modules": [{
            "module_id": "M003",
            "browser_scenarios": [{
                "name": "conflict",
                "steps": [{"action": "click", "selector": "#equals"}],
                "assertions": [{
                    "selector": "#display",
                    "property": "textContent",
                    "equals": "8",
                    "contains": "result",
                }],
            }],
        }],
    }
    with unittest.TestCase().assertRaisesRegex(AgentProtocolError, "exactly one"):
        UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_browser_inventory_repair_rebinds_only_an_exact_named_id() -> None:
    scenarios = [{
        "name": "addition",
        "steps": [{"action": "click", "role": "button", "name": "5"}],
        "assertions": [{
            "role": "textbox",
            "name": "Calculation display value",
            "property": "textContent",
            "expected": 8,
            "contains": "8",
        }],
    }]
    repaired, receipts = UltraOrchestrator._browser_inventory_contract_repair(
        scenarios,
        [{
            "interaction_targets": [
                {"id": "display", "tag": "div", "name": "0"},
                {"id": "five", "tag": "button", "name": "5"},
            ]
        }],
    )
    assert repaired[0]["steps"] == scenarios[0]["steps"]
    assert repaired[0]["assertions"] == [{
        "selector": "#display",
        "property": "textContent",
        "equals": "8",
    }]
    assert any("#display" in item for item in receipts)


def test_browser_inventory_repair_does_not_guess_an_unnamed_target() -> None:
    scenarios = [{
        "name": "result",
        "steps": [{"action": "click", "selector": "#equals"}],
        "assertions": [{
            "role": "status",
            "name": "Final answer",
            "property": "textContent",
            "equals": "8",
        }],
    }]
    repaired, receipts = UltraOrchestrator._browser_inventory_contract_repair(
        scenarios,
        [{"interaction_targets": [{"id": "display", "tag": "div"}]}],
    )
    assert repaired == scenarios
    assert not any("target rebound" in item for item in receipts)


def test_browser_scenario_validation_rejects_fill_without_value() -> None:
    payload = {
        "modules": [
            {
                "module_id": "M001",
                "browser_scenarios": [
                    {
                        "name": "addition",
                        "steps": [{"action": "fill", "role": "textbox", "name": "Input"}],
                        "assertions": [
                            {
                                "role": "status",
                                "name": "Result",
                                "property": "text",
                                "equals": "123",
                            }
                        ],
                    }
                ],
            }
        ]
    }

    try:
        UltraOrchestrator._validate_typed_response("browser_scenarios", payload)
    except AgentProtocolError as exc:
        assert "fill step requires value" in str(exc)
    else:
        raise AssertionError("fill without value must not reach browser execution")


def test_browser_scenario_validation_rejects_canvas_as_aria_role() -> None:
    payload = {
        "modules": [
            {
                "module_id": "M003",
                "browser_scenarios": [
                    {
                        "name": "canvas state",
                        "steps": [
                            {"action": "click", "role": "button", "name": "2"}
                        ],
                        "assertions": [
                            {
                                "role": "canvas",
                                "name": "Canvas element visible for 3D rendering",
                                "property": "visible",
                                "equals": "true",
                            }
                        ],
                    }
                ],
            }
        ]
    }

    try:
        UltraOrchestrator._validate_typed_response("browser_scenarios", payload)
    except AgentProtocolError as exc:
        assert "ARIA role, not an HTML tag" in str(exc)
    else:
        raise AssertionError("canvas HTML tag must not be accepted as an ARIA role")


def test_browser_scenario_transport_normalizes_canvas_tag_to_selector() -> None:
    payload, actions = UltraOrchestrator._normalize_typed_payload(
        "browser_scenarios",
        {
            "modules": [
                {
                    "module_id": "M003",
                    "browser_scenarios": [
                        {
                            "name": "canvas state",
                            "steps": [
                                {"action": "click", "role": "button", "name": "2"}
                            ],
                            "assertions": [
                                {
                                    "role": "canvas",
                                    "name": "Canvas element visible for 3D rendering",
                                    "property": "visible",
                                    "equals": "true",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        {"modules": [{"module_id": "M003"}]},
    )

    assertion = payload["modules"][0]["browser_scenarios"][0]["assertions"][0]
    assert assertion == {
        "selector": "canvas",
        "property": "visible",
        "equals": "true",
    }
    assert any("canvas HTML tag normalized" in item for item in actions)
    UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_browser_scenario_root_array_binds_to_exact_single_requested_module() -> None:
    payload, actions = UltraOrchestrator._normalize_typed_payload(
        "browser_scenarios",
        {
            "browser_scenarios": [
                {
                    "name": "addition",
                    "steps": [
                        {"action": "click", "role": "button", "name": "Add"}
                    ],
                    "assertions": [
                        {
                            "role": "status",
                            "name": "Result",
                            "property": "text",
                            "equals": "579",
                        }
                    ],
                }
            ]
        },
        {"modules": [{"module_id": "M004"}]},
    )

    assert payload["modules"][0]["module_id"] == "M004"
    assert payload["modules"][0]["browser_scenarios"][0]["name"] == "addition"
    assert any("one exact requested module" in item for item in actions)
    UltraOrchestrator._validate_typed_response("browser_scenarios", payload)


def test_browser_scenario_value_assertion_requires_a_form_control() -> None:
    payload = {
        "modules": [
            {
                "module_id": "M004",
                "browser_scenarios": [
                    {
                        "name": "result",
                        "steps": [{"action": "click", "role": "button", "name": "Run"}],
                        "assertions": [
                            {
                                "role": "status",
                                "name": "Result",
                                "property": "value",
                                "equals": "579",
                            }
                        ],
                    }
                ],
            }
        ]
    }

    try:
        UltraOrchestrator._validate_typed_response("browser_scenarios", payload)
    except AgentProtocolError as exc:
        assert "property value requires a form-control role" in str(exc)
    else:
        raise AssertionError("status.value must not reach Playwright input_value")


def test_browser_404_repair_constraints_are_scope_preserving_for_small_models() -> None:
    constraints = UltraOrchestrator._harness_repair_constraints(
        write_paths=("index.html",),
        verification=("preview_html index.html",),
        findings=(
            "HTTP 404 http://127.0.0.1:1234/styles.css",
            "GET http://127.0.0.1:1234/app.js: net::ERR_ABORTED",
        ),
    )

    rendered = " ".join(constraints)
    assert "complete self-contained implementation" in rendered
    assert "inline its required CSS and JavaScript" in rendered
    assert "preview_html index.html" in rendered
    assert "do not invent files" in rendered


@dataclass(frozen=True)
class Item:
    id: str
    depends_on: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    pre_write_hashes: Mapping[str, str | None] = field(default_factory=dict)
    order: int = 0


class FakeAgent:
    def __init__(self, handler: Callable[[AgentRequest], AgentResponse]):
        self.handler = handler

    def execute(self, request: AgentRequest) -> AgentResponse:
        return self.handler(request)


class FakeFactory:
    def __init__(self, handler: Callable[[AgentRequest], AgentResponse]):
        self.handler = handler
        self.created: list[tuple[AgentRole, str | None]] = []
        self.requests: list[AgentRequest] = []
        self._lock = threading.Lock()

    def create(self, role: AgentRole, *, run_id: str, node_id: str | None = None):
        del run_id
        with self._lock:
            self.created.append((role, node_id))

        def execute(request: AgentRequest) -> AgentResponse:
            with self._lock:
                self.requests.append(request)
            return self.handler(request)

        return FakeAgent(execute)


class ProjectMemoryState(InMemoryUltraState):
    def __init__(self) -> None:
        super().__init__()
        self.lookups: list[tuple[str, str]] = []

    def foundation_project_lessons(self, run_id: str, query: str, *, phase: str):
        self.lookups.append((phase, query))
        return (
            {
                "id": f"lesson-{phase}",
                "title": "Avoid shallow visual acceptance",
                "content": "Require browser/runtime evidence before planning completion.",
                "confidence": 0.91,
                "reuse_count": 3,
                "evidence_refs": ["bench:visual"],
            },
        )


def module(module_id: str, path: str, depends_on=()):
    return {
        "id": module_id,
        "title": f"Module {module_id}",
        "objective": f"Implement {module_id}",
        "acceptance_criteria": [f"{module_id} works"],
        "verification": [f"test {module_id}"],
        "depends_on": list(depends_on),
        "write_paths": [path],
        "forbidden_changes": ["do not change public scope"],
    }


def standard_handler(
    request: AgentRequest,
    *,
    modules: list[dict[str, Any]] | None = None,
) -> AgentResponse:
    modules = modules or [
        module("M1", "src/one"),
        module("M2", "src/two"),
        module("M3", "src/three"),
        module("M4", "src/four"),
    ]
    if request.phase == "goal_spec":
        payload = {
            "objective": "Build a complete system",
            "success_criteria": ["all modules pass their tests"],
            "constraints": ["preserve compatibility"],
        }
    elif request.phase == "architecture":
        payload = {
            "summary": "A modular architecture",
            "components": [{"name": item["id"]} for item in modules],
            "interfaces": [{"name": "stable-api"}],
        }
    elif request.phase == "master_plan":
        payload = {
            "summary": "Implement all modules and verify them",
            "execution_strategy": "dependency-safe waves",
            "modules": modules,
        }
    elif request.phase == InnerPhase.MINI_PLAN.value:
        payload = {"steps": ["implement", "verify"], "research_required": False}
    elif request.phase == InnerPhase.DECOMPOSE.value:
        payload = {"children": []}
    elif request.phase in {
        InnerPhase.REVIEW.value,
        InnerPhase.TEST.value,
        InnerPhase.INTEGRATE.value,
        InnerPhase.GLOBAL_INTEGRATION.value,
        InnerPhase.GLOBAL_REVIEW.value,
        InnerPhase.FINAL_EVIDENCE.value,
    }:
        payload = {
            "passed": True,
            "evidence": [{"kind": request.phase, "verified": True}],
            "test_results": [{"name": request.phase, "passed": True}],
        }
    elif request.phase == InnerPhase.IMPLEMENT.value:
        payload = {"artifacts": [{"path": f"{request.node_id}.py"}]}
    else:
        payload = {}
    return AgentResponse(
        payload=payload,
        summary=f"{request.phase}:{request.node_id or 'global'}",
        reasoning_summary="Decision based on the supplied contract and evidence.",
        provider="fake",
        model="scripted",
    )


def prepared_engine(
    execution_class=ExecutionClass.LOCAL,
    *,
    handler: Callable[[AgentRequest], AgentResponse] | None = None,
    modules: list[dict[str, Any]] | None = None,
    state: InMemoryUltraState | None = None,
):
    selected = handler or (lambda request: standard_handler(request, modules=modules))
    factory = FakeFactory(selected)
    engine = UltraOrchestrator(
        factory,
        execution_class=execution_class,
        state=state,
        config=UltraConfig(
            min_top_modules=1,
            max_top_modules=12,
            cloud_concurrency=4,
            provider_retries=2,
        ),
    )
    plan = engine.prepare("build the whole product")
    engine.approve(plan.fingerprint)
    return engine, factory, plan


class SchedulerTests(unittest.TestCase):
    def test_browser_scenario_repair_is_a_model_authored_micro_step(self):
        def handler(request: AgentRequest) -> AgentResponse:
            if request.phase == "browser_scenarios":
                self.assertEqual(request.task["modules"][0]["module_id"], "M004")
                return AgentResponse(
                    payload={
                        "modules": [
                            {
                                "module_id": "M004",
                                "browser_scenarios": [
                                    {
                                        "name": "Enter a calculation",
                                        "steps": [
                                            {
                                                "action": "click",
                                                "role": "button",
                                                "name": "7",
                                            }
                                        ],
                                        "assertions": [
                                            {
                                                "role": "textbox",
                                                "name": "Display",
                                                "property": "value",
                                                "equals": "7",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    },
                    summary="Bound one browser flow.",
                    provider="fake",
                    model="scripted",
                )
            return standard_handler(request)

        factory = FakeFactory(handler)
        engine = UltraOrchestrator(
            factory,
            execution_class=ExecutionClass.LOCAL,
            config=UltraConfig(
                min_top_modules=1,
                max_top_modules=12,
                provider_retries=1,
            ),
        )
        engine.prepare("Build the base project")
        plan = MasterPlanV1(
            summary="Build and verify the calculator.",
            modules=(
                TaskContractV1(
                    id="M004",
                    title="Browser verification",
                    objective="Click calculator buttons and verify the display.",
                    acceptance_criteria=("Click 7 and show 7.",),
                    verification=("preview_html index.html",),
                    write_paths=("index.html",),
                ),
            ),
        )

        bound = engine._bind_model_authored_browser_scenarios(
            "Build an interactive calculator",
            plan,
            (
                "module 'M004' has interactive acceptance criteria but "
                "metadata.browser_scenarios has no executable steps and assertions",
            ),
        )

        scenarios = bound.modules[0].metadata["browser_scenarios"]
        self.assertEqual(scenarios[0]["name"], "Enter a calculation")
        self.assertEqual(
            bound.modules[0].metadata["browser_scenarios_authorship"],
            "model_typed_micro_step",
        )
        self.assertEqual(
            [request.phase for request in factory.requests].count("browser_scenarios"),
            1,
        )

    def test_local_browser_scenario_repair_uses_one_model_call_per_module(self):
        def handler(request: AgentRequest) -> AgentResponse:
            if request.phase != "browser_scenarios":
                return standard_handler(request)
            self.assertEqual(len(request.task["modules"]), 1)
            module_id = request.task["modules"][0]["module_id"]
            return AgentResponse(
                payload={
                    "modules": [
                        {
                            "module_id": module_id,
                            "browser_scenarios": [
                                {
                                    "name": f"Verify {module_id}",
                                    "steps": [
                                        {"action": "click", "role": "button", "name": "Run"}
                                    ],
                                    "assertions": [
                                        {
                                            "role": "status",
                                            "name": "Result",
                                            "property": "text",
                                            "contains": "ready",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
                summary=f"Bound {module_id}",
                provider="fake",
                model="scripted",
            )

        factory = FakeFactory(handler)
        engine = UltraOrchestrator(
            factory,
            execution_class=ExecutionClass.LOCAL,
            config=UltraConfig(min_top_modules=1, max_top_modules=12, provider_retries=1),
        )
        engine.prepare("Build the base project")
        modules = tuple(
            TaskContractV1(
                id=module_id,
                title=f"Browser verification {module_id}",
                objective="Verify one interaction.",
                acceptance_criteria=("Click Run and observe ready.",),
                verification=("preview_html index.html",),
                write_paths=("index.html",),
            )
            for module_id in ("M004", "M005")
        )
        plan = MasterPlanV1(summary="Verify both browser flows.", modules=modules)

        bound = engine._bind_model_authored_browser_scenarios(
            "Build an interactive calculator",
            plan,
            tuple(
                f"module '{module.id}' has interactive acceptance criteria but "
                "metadata.browser_scenarios has no executable steps and assertions"
                for module in modules
            ),
        )

        scenario_requests = [
            request for request in factory.requests if request.phase == "browser_scenarios"
        ]
        self.assertEqual(
            [request.task["modules"][0]["module_id"] for request in scenario_requests],
            ["M004", "M005"],
        )
        self.assertTrue(
            all(module.metadata.get("browser_scenarios") for module in bound.modules)
        )

    def test_malformed_architecture_candidate_stops_after_targeted_repairs_without_fallback(self):
        def handler(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request)
            if request.phase == "architecture":
                return AgentResponse(
                    payload={"summary": "", "components": []},
                    summary="malformed architecture envelope",
                    provider="fake",
                    model="scripted",
                )
            return response

        factory = FakeFactory(handler)
        engine = UltraOrchestrator(
            factory,
            execution_class=ExecutionClass.LOCAL,
            config=UltraConfig(
                min_top_modules=1,
                max_top_modules=12,
                provider_retries=1,
            ),
        )
        with self.assertRaisesRegex(
            AgentProtocolError,
            "architecture failed after three targeted typed-return repairs",
        ):
            engine.prepare("Build a polished Three.js browser game")
        architecture_calls = [
            request for request in factory.requests if request.phase == "architecture"
        ]
        self.assertEqual(len(architecture_calls), 4)
        self.assertFalse(
            architecture_calls[1].task["typed_return_repair"]["repeated_response"]
        )
        self.assertTrue(
            architecture_calls[2].task["typed_return_repair"]["repeated_response"]
        )
        self.assertIn(
            "do not replay it",
            architecture_calls[2].task["typed_return_repair"]["instruction"],
        )
        self.assertIsNone(engine.architecture)

    def test_cross_run_lessons_are_injected_into_foundation_planning(self):
        state = ProjectMemoryState()
        engine, factory, _plan = prepared_engine(state=state)
        del engine
        by_phase = {request.phase: request for request in factory.requests}

        for phase in ("goal_spec", "architecture", "master_plan"):
            lessons = by_phase[phase].context["cross_run_project_lessons"]
            self.assertEqual(lessons[0]["title"], "Avoid shallow visual acceptance")
            self.assertEqual(lessons[0]["confidence"], 0.91)
        self.assertEqual([phase for phase, _query in state.lookups], ["goal_spec", "architecture", "master_plan"])

    def test_waves_are_dependency_safe_and_write_disjoint(self):
        items = [
            Item("A", write_paths=("src/shared",), order=1),
            Item("B", write_paths=("src/shared/file.py",), order=2),
            Item("C", write_paths=("src/other",), order=3),
            Item("D", depends_on=("A",), write_paths=("tests",), order=4),
        ]
        scheduler = DeterministicWaveScheduler(ExecutionClass.CLOUD, cloud_default=4)
        report = scheduler.run(items, lambda item: item.id)

        self.assertEqual(report.waves[0], ("A", "C"))
        self.assertNotIn("B", report.waves[0])
        self.assertGreaterEqual(len(report.waves), 2)
        self.assertTrue(report.successful)

    def test_empty_restart_wave_is_successful_when_all_work_is_already_durable(self):
        scheduler = DeterministicWaveScheduler(ExecutionClass.LOCAL)
        report = scheduler.run(
            (),
            lambda _item: self.fail("no worker should run"),
            initially_completed=("A", "B"),
        )

        self.assertEqual(report.outcomes, ())
        self.assertEqual(report.waves, ())
        self.assertTrue(report.successful)

    def test_user_authority_boundary_is_not_converted_to_worker_failure(self):
        class UserBoundary(RuntimeError):
            user_boundary = True

        scheduler = DeterministicWaveScheduler(ExecutionClass.LOCAL)

        with self.assertRaisesRegex(UserBoundary, "approval required"):
            scheduler.run(
                [Item("A", write_paths=("src/a.py",))],
                lambda _item: (_ for _ in ()).throw(
                    UserBoundary("approval required")
                ),
            )

    def test_prewrite_hash_conflict_blocks_before_worker(self):
        called: list[str] = []
        leases = ResourceLeaseManager(lambda _path: "new-hash")
        scheduler = DeterministicWaveScheduler(ExecutionClass.LOCAL, leases=leases)
        report = scheduler.run(
            [
                Item(
                    "A",
                    write_paths=("src/a.py",),
                    pre_write_hashes={"src/a.py": "old-hash"},
                )
            ],
            lambda item: called.append(item.id),
        )

        self.assertEqual(called, [])
        self.assertEqual(report.outcomes[0].status, ScheduleStatus.CONFLICT)
        self.assertIn("pre-write hash changed", report.outcomes[0].error)

    def test_rate_limit_retries_and_adapts_cloud_concurrency(self):
        attempts = Counter()
        adaptive = AdaptiveConcurrency(ExecutionClass.CLOUD, cloud_default=4, recover_after=99)
        scheduler = DeterministicWaveScheduler(
            ExecutionClass.CLOUD,
            adaptive=adaptive,
            rate_limit_retries=2,
        )

        def worker(item):
            attempts[item.id] += 1
            if item.id == "A" and attempts[item.id] == 1:
                raise RateLimitError("429")
            return item.id

        report = scheduler.run(
            [Item("A", write_paths=("a",)), Item("B", write_paths=("b",))],
            worker,
        )

        self.assertTrue(report.successful)
        self.assertEqual(attempts["A"], 2)
        self.assertEqual(adaptive.current, 3)


class UltraEngineTests(unittest.TestCase):
    def test_browser_repair_cannot_replace_calculation_with_canvas_presence(self):
        previous = [
            {
                "name": "3D calculation",
                "steps": [
                    {"action": "click", "role": "button", "name": "2"},
                    {"action": "click", "role": "button", "name": "*"},
                    {"action": "click", "role": "button", "name": "4"},
                ],
                "assertions": [
                    {
                        "role": "textbox",
                        "name": "3D Object State Check",
                        "property": "visibleCount",
                        "equals": "3",
                    }
                ],
            }
        ]
        weakened = [
            {
                "name": "canvas presence",
                "steps": [{"action": "click", "role": "button", "name": "."}],
                "assertions": [
                    {"selector": "canvas", "property": "visible", "equals": "true"}
                ],
            }
        ]

        reasons = UltraOrchestrator._browser_repair_weakening(previous, weakened)

        self.assertIn("repair reduced the accepted user-action sequence", reasons)
        self.assertIn("repair weakened an observable outcome to a presence check", reasons)
        self.assertIn(
            "repair replaced a value assertion with a boolean presence assertion",
            reasons,
        )

    def test_global_summary_excludes_superseded_failed_attempt_evidence(self):
        result = ResultPackageV1(
            node_id="M001",
            success=True,
            status="completed",
            summary="Old model summary still says the canvas is missing.",
            evidence=(
                {
                    "kind": "browser_preview",
                    "verification": "failed",
                    "interaction_results": [{"passed": False}],
                },
                {
                    "kind": "harness_browser_preview",
                    "status": "passed",
                    "interaction_results": [{"passed": True}],
                },
                {"kind": "artifact_hash", "path": "index.html", "sha256": "abc"},
            ),
        )

        summary = UltraOrchestrator._global_node_summary(result)

        self.assertEqual(
            summary["summary"],
            "M001 completed with an accepted node quality gate.",
        )
        self.assertEqual(len(summary["evidence"]), 2)
        self.assertTrue(all(item.get("verification") != "failed" for item in summary["evidence"]))

    def test_replan_checkpoint_resumes_verification_without_replaying_coder(self):
        engine, factory, _ = prepared_engine()
        node_id = next(iter(engine.nodes))
        node = replace(
            engine.nodes[node_id],
            status=NodeStatus.REVISION_REQUIRED,
            phase=InnerPhase.REPLAN,
        )
        engine.nodes[node_id] = node
        engine._prepared.clear()
        engine._results[node_id] = ResultPackageV1(
            node_id=node_id,
            success=False,
            status="revision_required",
            summary="Browser contract requires repair.",
            component_package={
                "candidate": {"payload": {"mutation_receipt": True}},
                "replan": {
                    "verification_only": True,
                    "preserve_candidate": True,
                    "failure_kind": "contract",
                },
            },
        )

        request_count = len(factory.requests)
        engine._ensure_expanded(node_id)
        self.assertEqual(engine.nodes[node_id].status, NodeStatus.READY)
        result = engine._execute_node(engine.nodes[node_id])

        resumed_requests = factory.requests[request_count:]
        self.assertTrue(result.success)
        self.assertFalse(
            any(
                request.role is AgentRole.CODER
                and request.phase == InnerPhase.IMPLEMENT.value
                for request in resumed_requests
            )
        )

    def test_authoritative_hashes_skip_model_authored_final_evidence_formatter(self):
        modules = [module("M1", "src/app.py")]

        def handler(request: AgentRequest) -> AgentResponse:
            if request.phase == InnerPhase.FINAL_EVIDENCE.value:
                self.fail(
                    "authoritative operational evidence must not be handed back to "
                    "the model for schema formatting"
                )
            response = standard_handler(request, modules=modules)
            if request.phase == InnerPhase.IMPLEMENT.value:
                return replace(
                    response,
                    payload={
                        "artifacts": [
                            {
                                "path": "src/app.py",
                                "sha256": "a" * 64,
                            }
                        ]
                    },
                )
            return response

        events = EventBus()
        captured = []
        events.subscribe(captured.append)
        factory = FakeFactory(handler)
        engine = UltraOrchestrator(
            factory,
            execution_class=ExecutionClass.LOCAL,
            events=events,
            config=UltraConfig(min_top_modules=1, max_top_modules=4),
        )
        plan = engine.prepare("build the whole product")
        engine.approve(plan.fingerprint)

        result = engine.run()

        self.assertTrue(result.successful)
        self.assertEqual(
            result.global_result.evidence[0]["kind"],
            "authoritative_operational_evidence",
        )
        self.assertTrue(
            any(item.kind == "ultra.deterministic_final_evidence" for item in captured)
        )
        self.assertFalse(
            any(
                request.phase
                in {
                    InnerPhase.GLOBAL_INTEGRATION.value,
                    InnerPhase.GLOBAL_REVIEW.value,
                    InnerPhase.FINAL_EVIDENCE.value,
                }
                for request in factory.requests
            )
        )

    def test_live_progress_events_include_real_graph_counts_and_current_assignment(self):
        events = EventBus()
        captured = []
        events.subscribe(captured.append)
        modules = [module("M1", "src/one")]
        factory = FakeFactory(lambda request: standard_handler(request, modules=modules))
        engine = UltraOrchestrator(
            factory,
            execution_class=ExecutionClass.LOCAL,
            events=events,
            config=UltraConfig(min_top_modules=1, max_top_modules=4),
        )
        plan = engine.prepare("build the whole product")
        engine.approve(plan.fingerprint)
        result = engine.run()

        self.assertTrue(result.successful)
        graph = next(item for item in captured if item.kind == "ultra.graph_ready")
        self.assertGreaterEqual(graph.data["total_nodes"], 1)
        started = next(
            item
            for item in captured
            if item.kind == "ultra.agent_started" and item.data.get("node_id")
        )
        self.assertEqual(started.data["total_nodes"], graph.data["total_nodes"])
        self.assertTrue(started.data["current_node_title"])
        completed = [
            item
            for item in captured
            if item.kind == "ultra.node" and item.data.get("status") == "completed"
        ]
        self.assertEqual(completed[-1].data["completed_nodes"], graph.data["total_nodes"])

    def test_quality_consensus_uses_typed_passed_over_contradictory_declaration(self):
        engine, _factory, _plan = prepared_engine()
        node = next(iter(engine.nodes.values()))
        records = engine._quality_vote_records(
            node,
            (
                AgentResponse(
                    payload={"passed": True, "verdict": "reject"},
                    summary="Typed review passed.",
                ),
                AgentResponse(
                    payload={"passed": False, "verdict": "accept"},
                    summary="Typed review failed.",
                ),
            ),
        )

        self.assertEqual(records[0]["verdict"], "accept")
        self.assertEqual(records[0]["evidence"]["declared_verdict"], "reject")
        self.assertEqual(records[1]["verdict"], "reject")
        self.assertEqual(records[1]["evidence"]["declared_verdict"], "accept")

    def test_authoritative_runtime_evidence_prevents_reasoning_shape_false_rejection(self):
        engine, _factory, _plan = prepared_engine()
        node = next(iter(engine.nodes.values()))
        missing_reasoning = {
            "passed": False,
            "missing_fields": ["reasoning_artifact"],
        }
        records = engine._quality_vote_records(
            node,
            (
                AgentResponse(
                    payload={
                        "passed": True,
                        "harness_reasoning_evaluation": missing_reasoning,
                    },
                    summary="Typed reviewer passed without optional prose.",
                ),
                AgentResponse(
                    payload={
                        "passed": True,
                        "test_results": [
                            {"name": "harness_html_preview", "passed": True}
                        ],
                        "harness_reasoning_evaluation": missing_reasoning,
                    },
                    summary="Authoritative preview passed.",
                ),
            ),
        )

        self.assertEqual([item["verdict"] for item in records], ["accept", "accept"])
        self.assertFalse(
            records[0]["evidence"]["harness_reasoning_evaluation"]["passed"]
        )

    def test_passed_browser_evidence_satisfies_duplicate_preview_request_only(self):
        response = AgentResponse(
            payload={
                "passed": False,
                "issues": [
                    {
                        "summary": "The application must be functional in a real browser.",
                        "description": "Preview the HTML file to verify functionality.",
                    }
                ],
                "findings": [],
            },
            summary="Browser verification is required.",
        )
        evidence = {
            "test_results": [
                {"name": "harness_html_preview", "passed": True, "http_status": 200}
            ]
        }

        reconciled = UltraOrchestrator._reconcile_satisfied_evidence_request(
            response, evidence
        )

        self.assertTrue(reconciled.payload["passed"])
        self.assertFalse(reconciled.payload["issues"])

        concrete = replace(
            response,
            payload={
                **response.payload,
                "issues": ["Browser preview failed with console error 404."],
            },
        )
        unresolved = UltraOrchestrator._reconcile_satisfied_evidence_request(
            concrete, evidence
        )
        self.assertFalse(unresolved.payload["passed"])
        self.assertTrue(unresolved.payload["issues"])

    def test_agent_response_preserves_typed_envelope_reasoning_artifact(self):
        artifact = {
            "claim": "candidate passes",
            "supporting_evidence": ["review:candidate"],
            "counterarguments": ["runtime still pending"],
            "rejected_alternatives": ["contract-only review"],
            "verification_plan": ["run the candidate"],
        }
        response = AgentResponse.from_mapping(
            {
                "summary": "reviewed",
                "payload": {"passed": True},
                "reasoning_artifact": artifact,
            }
        )

        self.assertTrue(response.payload["passed"])
        self.assertEqual(response.payload["reasoning_artifact"], artifact)

    def test_small_model_reasoning_graph_ids_are_repaired_without_inventing_evidence(self):
        artifact = {
            "claim": "candidate passes",
            "supporting_evidence": ["candidate:api"],
            "counterarguments": ["browser pending"],
            "rejected_alternatives": ["contract-only review"],
            "verification_plan": ["run candidate"],
            "reasoning_graph": {
                "nodes": [
                    {"id": "choice", "summary": "candidate passes"},
                    {"id": "alt", "summary": "contract-only review"},
                ],
                "edges": [{"from": "missing", "to": "alt", "relation": "supports"}],
            },
        }
        repaired, actions = repair_reasoning_artifact_graph(artifact)
        protocol = reasoning_debate_protocol_for("reviewer", "review", {})

        self.assertTrue(actions)
        self.assertEqual(repaired["supporting_evidence"], artifact["supporting_evidence"])
        self.assertTrue(evaluate_reasoning_artifact(repaired, protocol).passed)

    def test_fix_candidate_is_composed_with_prior_implementation(self):
        base = AgentResponse(
            payload={
                "implementation": {"code": "class Environment {}", "api": {"setup": True}},
                "evidence": [{"id": "initial"}],
            },
            summary="initial",
        )
        fix = AgentResponse(
            payload={
                "implementation": {"api": {"update": True}},
                "evidence": [{"id": "fix"}],
            },
            summary="fixed",
        )

        combined = UltraOrchestrator._merge_candidate_response(base, fix)

        self.assertEqual(combined.payload["implementation"]["code"], "class Environment {}")
        self.assertEqual(
            combined.payload["implementation"]["api"],
            {"setup": True, "update": True},
        )
        self.assertEqual(combined.payload["evidence"], [{"id": "fix"}])

    def test_single_html_artifact_does_not_invent_recursive_specialists(self):
        state = InMemoryUltraState()
        html_module = module("GAME", "index.html")
        html_module.update(
            {
                "title": "Three.js vehicle game",
                "objective": "Build a polished 3D vehicle, road, character, gameplay, and presentation",
            }
        )
        engine, factory, _plan = prepared_engine(
            state=state,
            modules=[html_module],
        )

        result = engine.run()

        self.assertTrue(result.successful)
        root = next(node for node in engine.nodes.values() if node.parent_id is None)
        children = [node for node in engine.nodes.values() if node.parent_id]
        self.assertEqual(children, [])
        run_id = engine.run_state.id
        self.assertGreaterEqual(len(state.specialists[run_id]), 1)
        self.assertIn(root.id, state.component_packages[run_id])
        assembler_requests = [
            request
            for request in factory.requests
            if request.phase == InnerPhase.INTEGRATE.value
            and request.task.get("publish_component_package")
        ]
        self.assertEqual(len(assembler_requests), 1)
        self.assertFalse(assembler_requests[0].task.get("final_assembler", False))

    def test_decomposer_contract_scales_only_real_work_boundaries(self):
        engine, factory, _plan = prepared_engine()

        result = engine.run()

        self.assertTrue(result.successful)
        request = next(
            item for item in factory.requests
            if item.phase == InnerPhase.DECOMPOSE.value
        )
        self.assertIn("size the work tree from the approved contract", request.system_prompt)
        self.assertIn("Return children: []", request.system_prompt)
        self.assertIn("quality roles", request.system_prompt)
        component_assemblers = [
            request
            for request in factory.requests
            if request.phase == InnerPhase.INTEGRATE.value
            and request.task.get("component_assembler")
        ]
        self.assertEqual(component_assemblers, [])
        self.assertTrue(all(not request.task.get("final_assembler") for request in component_assemblers))
        review_requests = [
            request
            for request in factory.requests
            if request.phase in {InnerPhase.REVIEW.value, InnerPhase.TEST.value}
            and request.task.get("fresh_review", request.task.get("fresh_test_context", False))
        ]
        self.assertTrue(review_requests)
        self.assertTrue(all("candidate" in request.task for request in review_requests))

    def test_foundation_is_sequential_fingerprint_bound_and_traced_without_cot(self):
        state = InMemoryUltraState()
        engine, factory, plan = prepared_engine(state=state)

        self.assertEqual(engine.phase, UltraPhase.AWAITING_APPROVAL)
        self.assertTrue(engine.run_state.approved)
        self.assertEqual(
            [request.phase for request in factory.requests[:6]],
            [
                "goal_spec",
                "architecture",
                "architecture",
                "architecture_critique",
                "architecture_judge",
                "master_plan",
            ],
        )
        self.assertEqual(len(factory.created), 6)
        self.assertEqual(plan.fingerprint, engine.run_state.master_fingerprint)
        self.assertTrue(all("chain-of-thought" not in trace.reasoning_summary for trace in state.traces))
        self.assertTrue(all(not hasattr(trace, "chain_of_thought") for trace in state.traces))

    def test_local_and_cloud_execute_identical_pipeline_with_different_parallelism(self):
        local, local_factory, _ = prepared_engine(ExecutionClass.LOCAL)
        cloud, cloud_factory, _ = prepared_engine(ExecutionClass.CLOUD)

        local_result = local.run()
        cloud_result = cloud.run()

        self.assertTrue(local_result.successful)
        self.assertTrue(cloud_result.successful)
        self.assertEqual(local_result.schedule.peak_concurrency, 1)
        self.assertEqual(cloud_result.schedule.peak_concurrency, 4)
        local_pipeline = Counter(request.phase for request in local_factory.requests)
        cloud_pipeline = Counter(request.phase for request in cloud_factory.requests)
        self.assertEqual(local_pipeline, cloud_pipeline)
        self.assertEqual(
            [result.node_id.split(".", 1)[-1] for result in local_result.node_results],
            [result.node_id.split(".", 1)[-1] for result in cloud_result.node_results],
        )
        self.assertNotEqual(
            [result.node_id for result in local_result.node_results],
            [result.node_id for result in cloud_result.node_results],
        )

    def test_dynamic_children_inherit_contract_and_out_of_scope_child_requires_revision(self):
        parent_module = [module("M1", "src/feature")]

        def contained(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request, modules=parent_module)
            if request.phase == InnerPhase.DECOMPOSE.value and request.node_id.endswith(".M001"):
                return AgentResponse(
                    payload={
                        "children": [
                            module("M1.child", "src/feature/child.py"),
                        ]
                    },
                    summary="contained child",
                )
            return response

        engine, _, _ = prepared_engine(handler=contained, modules=parent_module)
        result = engine.run()
        self.assertTrue(result.successful)
        parent_id = next(node_id for node_id in engine.nodes if node_id.endswith(".M001"))
        child_id = engine.nodes[parent_id].children[0]
        child = engine.nodes[child_id]
        self.assertEqual(child.parent_id, parent_id)
        self.assertIn("do not change public scope", child.contract.forbidden_changes)
        self.assertIn(child_id, engine.nodes[parent_id].depends_on)

        def escaped(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request, modules=parent_module)
            if request.phase == InnerPhase.DECOMPOSE.value and request.node_id.endswith(".M001"):
                return AgentResponse(
                    payload={"children": [module("M1.escape", "outside/file.py")]},
                    summary="bad child",
                )
            return response

        escaped_engine, _, _ = prepared_engine(handler=escaped, modules=parent_module)
        escaped_result = escaped_engine.run()
        self.assertFalse(escaped_result.successful)
        self.assertEqual(escaped_engine.phase, UltraPhase.REVISION_REQUIRED)

    def test_identical_quality_failure_breaker_stops_the_third_code_mutation(self):
        modules = [module("M1", "src/one")]
        phases = Counter()

        def failing_quality(request: AgentRequest) -> AgentResponse:
            phases[request.phase] += 1
            response = standard_handler(request, modules=modules)
            if request.phase in {InnerPhase.REVIEW.value, InnerPhase.TEST.value}:
                return AgentResponse(
                    payload={"passed": False, "issues": ["still broken"]},
                    summary="quality failed",
                )
            if request.phase == InnerPhase.REPLAN.value:
                return AgentResponse(payload={"revision": "change approach"}, summary="replan now")
            return response

        engine, _, _ = prepared_engine(handler=failing_quality, modules=modules)
        result = engine.run()

        self.assertFalse(result.successful)
        self.assertEqual(engine.phase, UltraPhase.REVISION_REQUIRED)
        self.assertEqual(phases[InnerPhase.FIX.value], 2)
        self.assertEqual(phases[InnerPhase.REPLAN.value], 1)
        self.assertEqual(result.node_results[0].fix_attempts, 2)
        self.assertEqual(
            result.node_results[0].component_package["status"],
            "best_candidate_below_target",
        )
        self.assertEqual(
            result.node_results[0].component_package["replan"]["revision"],
            "change approach",
        )
        diagnostic = result.node_results[0].component_package["failure_diagnostic"]
        self.assertTrue(diagnostic["mutation_prohibited"])
        self.assertEqual(diagnostic["blocker_owner"], "diagnostic")
        self.assertEqual(diagnostic["occurrences"], 3)

    def test_browser_contract_failure_routes_to_tester_repair_without_coder_fix(self):
        browser_module = module("M1", "index.html")
        browser_module["metadata"] = {
            "browser_scenarios": [
                {
                    "name": "calculator flow",
                    "steps": [{"action": "click", "selector": "#missing"}],
                    "assertions": [
                        {
                            "selector": "#display",
                            "property": "text",
                            "equals": "1",
                        }
                    ],
                }
            ]
        }
        phases = Counter()

        def contract_repair(request: AgentRequest) -> AgentResponse:
            phases[(request.role.value, request.phase)] += 1
            if request.phase == "browser_scenarios":
                return AgentResponse(
                    payload={
                        "modules": [
                            {
                                "module_id": request.node_id,
                                "browser_scenarios": [
                                    {
                                        "name": "calculator flow",
                                        "steps": [
                                            {"action": "click", "selector": "#increment"}
                                        ],
                                        "assertions": [
                                            {
                                                "selector": "#display",
                                                "property": "textContent",
                                                "equals": "1",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    },
                    summary="tester repaired only the browser contract",
                )
            if request.phase == InnerPhase.TEST.value and request.role is AgentRole.TESTER:
                scenarios = request.task["contract"]["metadata"]["browser_scenarios"]
                repaired = scenarios[0]["steps"][0].get("selector") == "#increment"
                if not repaired:
                    return AgentResponse(
                        payload={
                            "passed": False,
                            "failure_kind": "contract",
                            "blocker_owner": "test_harness",
                            "issues": ["interaction target matched no elements: #missing"],
                            "test_results": [
                                {
                                    "name": "browser",
                                    "passed": False,
                                    "failure_kind": "contract",
                                    "interaction_targets": [
                                        {"selector": "#increment", "role": "button"}
                                    ],
                                }
                            ],
                        },
                        summary="browser contract failed",
                    )
            return standard_handler(request, modules=[browser_module])

        engine, factory, _ = prepared_engine(
            handler=contract_repair,
            modules=[browser_module],
        )
        result = engine.run()

        self.assertTrue(result.successful)
        self.assertFalse(
            any(
                "matched no elements" in finding
                for node_result in result.node_results
                for finding in node_result.findings
            )
        )
        self.assertFalse(
            any(
                test_result.get("passed") is False
                for node_result in result.node_results
                for test_result in node_result.test_results
            )
        )
        self.assertEqual(phases[(AgentRole.TESTER.value, "browser_scenarios")], 1)
        self.assertEqual(phases[(AgentRole.CODER.value, InnerPhase.FIX.value)], 0)
        repaired_request = next(
            request
            for request in factory.requests
            if request.phase == InnerPhase.TEST.value
            and request.role is AgentRole.TESTER
            and request.task["contract"]["metadata"]["browser_scenarios"][0]["steps"][0].get("selector")
            == "#increment"
        )
        self.assertEqual(
            repaired_request.task["contract"]["metadata"]["browser_scenarios_authorship"],
            "tester_contract_repair",
        )

    def test_browser_environment_failure_retries_verification_without_coder_fix(self):
        browser_module = module("M1", "index.html")
        test_attempts = 0
        phases = Counter()

        def transient_environment_failure(request: AgentRequest) -> AgentResponse:
            nonlocal test_attempts
            phases[(request.role.value, request.phase)] += 1
            if request.phase == InnerPhase.TEST.value and request.role is AgentRole.TESTER:
                test_attempts += 1
                if test_attempts <= 2:
                    return AgentResponse(
                        payload={
                            "passed": False,
                            "failure_kind": "environment",
                            "blocker_owner": "environment",
                            "issues": ["browser executable is temporarily unavailable"],
                        },
                        summary="browser environment unavailable",
                    )
            return standard_handler(request, modules=[browser_module])

        engine, _, _ = prepared_engine(
            handler=transient_environment_failure,
            modules=[browser_module],
        )
        result = engine.run()

        self.assertTrue(result.successful)
        self.assertEqual(test_attempts, 3)
        self.assertEqual(phases[(AgentRole.CODER.value, InnerPhase.FIX.value)], 0)

    def test_leaf_review_does_not_fail_for_downstream_plan_artifacts(self):
        modules = [
            module("M1", "mathlib.py"),
            module("M2", "formatter.py", depends_on=("M1",)),
        ]
        fix_nodes = []

        def downstream_only_review(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request, modules=modules)
            if (
                request.node_id
                and request.node_id.endswith(".M001")
                and request.phase in {InnerPhase.REVIEW.value, InnerPhase.TEST.value}
                and request.role is not AgentRole.SECURITY_REVIEWER
            ):
                return AgentResponse(
                    payload={
                        "passed": False,
                        "issues": [
                            "formatter.py is not implemented yet; it is required downstream."
                        ],
                    },
                    summary="Future module is pending.",
                )
            if request.phase == InnerPhase.FIX.value:
                fix_nodes.append(request.node_id)
            return response

        engine, _, _ = prepared_engine(
            handler=downstream_only_review,
            modules=modules,
        )
        result = engine.run()

        self.assertTrue(result.successful)
        self.assertNotIn(
            next(node_id for node_id in engine.nodes if node_id.endswith(".M001")),
            fix_nodes,
        )

    def test_leaf_review_keeps_current_path_failure_blocking(self):
        modules = [
            module("M1", "mathlib.py"),
            module("M2", "formatter.py", depends_on=("M1",)),
        ]
        engine, _, _ = prepared_engine(modules=modules)
        node = next(
            node
            for node in engine.nodes.values()
            if node.id.endswith(".M001")
        )
        scoped = engine._scope_leaf_review(
            node,
            AgentResponse(
                payload={
                    "passed": False,
                    "issues": ["mathlib.py returns subtraction instead of addition."],
                },
                summary="Current component is incorrect.",
            ),
        )

        self.assertFalse(scoped.payload["passed"])
        self.assertFalse(scoped.payload.get("abstained", False))
        self.assertEqual(
            scoped.payload["issues"],
            ["mathlib.py returns subtraction instead of addition."],
        )

    def test_recovered_fix_checkpoint_does_not_redecompose_leaf(self):
        modules = [module("M1", "mathlib.py")]
        engine, factory, _ = prepared_engine(modules=modules)
        node_id = next(
            node_id for node_id in engine.nodes if node_id.endswith(".M001")
        )
        engine.nodes[node_id] = replace(
            engine.nodes[node_id],
            status=type(engine.nodes[node_id].status).PENDING,
            phase=InnerPhase.FIX,
            children=(),
        )
        before = len(factory.requests)

        engine._ensure_expanded(node_id)

        self.assertEqual(len(factory.requests), before)
        self.assertEqual(engine.nodes[node_id].status.value, "ready")
        self.assertEqual(engine.nodes[node_id].phase, InnerPhase.FIX)

    def test_replan_refines_contract_without_expanding_scope(self):
        original = TaskContractV1.from_mapping(module("M1", "src/one"))
        refined = UltraOrchestrator._refine_contract_from_replan(
            original,
            {
                "reasoning_artifact": {
                    "claim": "The integration contract must expose `masterUpdate(deltaTime): void`.",
                    "findings": [
                        "The current package has no stable parent-callable update entrypoint."
                    ],
                    "verification_plan": [
                        "Call `masterUpdate(deltaTime): void` twice and verify deterministic state."
                    ],
                }
            },
        )

        self.assertIn("masterUpdate(deltaTime): void", refined.owned_interfaces)
        self.assertIn(
            "Call `masterUpdate(deltaTime): void` twice and verify deterministic state.",
            refined.verification,
        )
        self.assertTrue(
            any("integration contract" in item for item in refined.acceptance_criteria)
        )
        self.assertEqual(refined.write_paths, original.write_paths)
        self.assertEqual(refined.depends_on, original.depends_on)
        self.assertEqual(refined.forbidden_changes, original.forbidden_changes)
        self.assertTrue(refined.metadata["replan_refinement_requirements"])

    def test_background_cancel_stops_at_safe_checkpoint(self):
        modules = [module("M1", "src/one")]
        entered = threading.Event()
        release = threading.Event()

        def blocking(request: AgentRequest) -> AgentResponse:
            if request.phase == InnerPhase.IMPLEMENT.value:
                entered.set()
                release.wait(timeout=2)
            return standard_handler(request, modules=modules)

        engine, _, _ = prepared_engine(handler=blocking, modules=modules)
        engine.start_background()
        self.assertTrue(entered.wait(timeout=2))
        engine.cancel()
        release.set()
        result = engine.background.result(timeout=3)

        self.assertFalse(result.successful)
        self.assertEqual(engine.phase, UltraPhase.CANCELLED)
        self.assertTrue(engine.control.cancelled)


if __name__ == "__main__":
    unittest.main()

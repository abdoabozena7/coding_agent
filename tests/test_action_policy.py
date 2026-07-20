from __future__ import annotations

from types import SimpleNamespace

from agent.action_policy import (
    ApprovalRequirement,
    classify_action,
    plan_review_reasons,
    should_surface_question,
)


def test_project_reads_and_writes_do_not_interrupt_simple_workflow(tmp_path):
    for tool in ("read_file", "list_files", "write_file", "edit_file", "apply_patch"):
        decision = classify_action(tool, {"path": "app.py"}, workspace=tmp_path, sandboxed=False)
        assert decision.requirement is ApprovalRequirement.AUTO


def test_project_checks_are_once_per_session_without_sandbox_and_auto_inside_it(tmp_path):
    args = {"command": "python -m pytest -q"}
    normal = classify_action("run_bash", args, workspace=tmp_path, sandboxed=False)
    isolated = classify_action("run_bash", args, workspace=tmp_path, sandboxed=True)
    assert normal.requirement is ApprovalRequirement.SESSION
    assert normal.group == "project_checks"
    assert isolated.requirement is ApprovalRequirement.AUTO


def test_dependencies_deletion_and_network_stay_explicit(tmp_path):
    for command in ("pip install rich", "Remove-Item output.txt", "curl https://example.com"):
        decision = classify_action(
            "run_bash", {"command": command}, workspace=tmp_path, sandboxed=True
        )
        assert decision.requirement is ApprovalRequirement.ONCE


def test_only_consequential_product_questions_surface():
    assert should_surface_question({"question": "Should the layout be compact or spacious?"})
    assert not should_surface_question({"question": "Which internal parser library should I use?"})


def test_high_impact_plan_requires_review_but_routine_plan_auto_starts():
    routine = SimpleNamespace(plan_summary="Build the requested calculator", objective="calculator", expected_changes=(), tasks=())
    risky = SimpleNamespace(plan_summary="Install a dependency and migrate the database", objective="upgrade", expected_changes=(), tasks=())
    assert plan_review_reasons(routine) == ()
    assert plan_review_reasons(risky)

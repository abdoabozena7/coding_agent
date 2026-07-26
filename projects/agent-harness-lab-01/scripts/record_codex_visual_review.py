"""Record the supervising Codex visual verdict and re-run final acceptance."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.goal_outcome import (
    FinalAcceptanceEvidenceV1,
    GoalOutcomeState,
)
from agent.models import GoalStatus
from agent.store import StateStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--finding", action="append", default=[])
    parser.add_argument("--screenshot", action="append", default=[])
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    with StateStore(workspace) as store:
        runs = store.list_ultra_runs()
        run = (
            store.get_ultra_run(args.run_id)
            if args.run_id
            else runs[-1]
            if runs
            else None
        )
        if run is None:
            raise SystemExit("No Ultra run exists in the workspace.")
        digest = hashlib.sha256()
        screenshots: list[str] = []
        for raw in args.screenshot:
            path = Path(raw).resolve()
            digest.update(path.read_bytes())
            screenshots.append(str(path))
        store.record_final_acceptance_evidence(
            FinalAcceptanceEvidenceV1(
                ultra_run_id=run.id,
                kind="codex_visual_review",
                authority="codex-supervisor",
                passed=bool(args.accept),
                score=max(0.0, min(1.0, args.score)),
                critical_findings=0 if args.accept else max(1, len(args.finding)),
                artifact_hash=digest.hexdigest() if screenshots else "",
                details={
                    "critical": True,
                    "screenshots": screenshots,
                    "findings": list(args.finding),
                    "review_policy": "strict-direct-visual-inspection",
                },
            )
        )
        decision = store.evaluate_final_acceptance(run.goal_id, run.id)
        goal = store.get_goal(run.goal_id)
        if decision["accepted"]:
            if goal.status is GoalStatus.BLOCKED:
                store.transition_goal(
                    goal.id,
                    GoalStatus.RUNNING,
                    reason="Supervisory visual evidence satisfied the blocked final gate",
                )
                goal = store.get_goal(goal.id)
            if goal.status is GoalStatus.RUNNING:
                store.transition_goal(goal.id, GoalStatus.VERIFYING, reason="Final product evidence complete")
                goal = store.get_goal(goal.id)
            if goal.status is GoalStatus.VERIFYING:
                store.transition_goal(goal.id, GoalStatus.REVIEWING, reason="Final product review complete")
                goal = store.get_goal(goal.id)
            if goal.status is GoalStatus.REVIEWING:
                store.transition_goal(
                    goal.id,
                    GoalStatus.COMPLETED,
                    reason="GoalOutcomeContract accepted by FinalAcceptanceGate",
                )
            return 0
        store.set_goal_outcome_state(
            run.goal_id,
            GoalOutcomeState.QUALITY_BLOCKED,
            decision=decision,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

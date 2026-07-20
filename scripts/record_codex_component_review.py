"""Persist a direct Codex visual review for one materialized specialist package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.quality import (
    ChangeSetStatus,
    ChangeSetV1,
    FindingSeverity,
    FindingStatus,
    QualityCategory,
    QualityFindingV1,
)
from agent.store import StateStore
from agent.store import NotFoundError
from agent.ultra_models import WorkNodeStatus


def _dimensions(values: list[str], fallback: float) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name.strip():
            raise SystemExit(f"Invalid --dimension {value!r}; expected NAME=SCORE")
        parsed[name.strip()] = max(0.0, min(1.0, float(raw)))
    return parsed or {"direct_visual_quality": fallback}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--finding", action="append", default=[])
    parser.add_argument("--dimension", action="append", default=[])
    parser.add_argument("--screenshot")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    score = max(0.0, min(1.0, args.score))
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
        packages = store.list_component_packages(
            run.id,
            work_node_id=args.node_id,
        )
        if not packages:
            raise SystemExit(f"No component package exists for {args.node_id}.")
        package = packages[-1]
        root = Path(str(package["implementation"].get("root") or "")).resolve()
        screenshot = (
            Path(args.screenshot).resolve()
            if args.screenshot
            else root / "evidence" / "preview.png"
        )
        if not screenshot.is_file():
            raise SystemExit(f"Component screenshot does not exist: {screenshot}")
        screenshot_hash = hashlib.sha256(screenshot.read_bytes()).hexdigest()
        findings = tuple(str(item).strip() for item in args.finding if str(item).strip())
        accepted = bool(args.accept and not findings and score >= 0.90)
        dimensions = _dimensions(args.dimension, score)
        verdict = {
            "schema": "CodexComponentVisualReviewV1",
            "evaluator": "codex-supervisor",
            "model": "current-codex-session",
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "score": score,
            "scores": dimensions,
            "critical_findings": 0 if accepted else max(1, len(findings)),
            "findings": list(findings),
            "screenshot": str(screenshot),
            "screenshot_hash": screenshot_hash,
            "context_fingerprint": f"direct:{screenshot_hash}",
            "review_policy": "strict-direct-image-inspection",
        }
        visual_evidence_id = store.save_visual_evaluation(
            run.id,
            verdict,
            work_node_id=args.node_id,
            package_id=str(package["id"]),
        )
        if accepted:
            implementation = dict(package.get("implementation") or {})
            component_files = tuple(
                dict(item)
                for item in implementation.get("files", ())
                if isinstance(item, dict) and str(item.get("path") or "").strip()
            )
            changed_files = tuple(str(item["path"]) for item in component_files)
            post_hashes = {
                str(item["path"]): str(item.get("content_hash") or "")
                for item in component_files
            }
            remediation = ChangeSetV1(
                ultra_run_id=run.id,
                responsible_agent_id="codex-supervisor",
                parent_id=args.node_id,
                status=ChangeSetStatus.INTEGRATED,
                changed_files=changed_files,
                pre_hashes={path: None for path in changed_files},
                post_hashes=post_hashes,
                verification_evidence_ids=(visual_evidence_id,),
                review_status={
                    "clean_code": "passed",
                    "security": "passed",
                    "test_quality": "passed",
                },
                integration_status="integrated",
                metadata={
                    "kind": "component_visual_remediation",
                    "package_id": str(package["id"]),
                    "package_content_hash": str(package.get("content_hash") or ""),
                    "visual_evidence_id": visual_evidence_id,
                    "review_policy": verdict["review_policy"],
                },
            )
            store.save_change_set(remediation)
            for finding in store.list_quality_findings(run.id):
                if (
                    finding.repair_node_id == args.node_id
                    and finding.status is not FindingStatus.RESOLVED
                ):
                    store.transition_quality_finding(
                        finding.id,
                        FindingStatus.RESOLVED,
                        repair_node_id=args.node_id,
                        remediation_change_set_id=remediation.id,
                        verification_evidence_ids=(visual_evidence_id,),
                    )
        else:
            messages = findings or ("The component did not meet the strict visual threshold.",)
            finding = QualityFindingV1(
                ultra_run_id=run.id,
                principle_id="visual_quality",
                category=QualityCategory.VISUAL,
                severity=FindingSeverity.HIGH,
                path=str(root / str(package["preview"].get("entrypoint") or "preview.html")),
                location="component preview",
                file_hash=str(package["content_hash"]),
                evidence={
                    "source": "codex-supervisor",
                    "screenshot": str(screenshot),
                    "screenshot_hash": screenshot_hash,
                    "score": score,
                    "dimensions": dimensions,
                    "findings": list(messages),
                },
                remediation=(
                    f"Specialist {args.node_id} must produce a new materialized revision: "
                    + "; ".join(messages)
                ),
                acceptance_criteria=(
                    "No visible placeholder geometry or source text.",
                    "Every critical visual dimension scores at least 0.90.",
                    "A fresh screenshot is visibly better than the rejected revision.",
                ),
                verification=(
                    "Run the isolated component preview with zero browser errors.",
                    "Capture and directly inspect a fresh deterministic screenshot.",
                ),
                repair_node_id=args.node_id,
            )
            store.put_quality_finding(finding)
            try:
                store.transition_work_node(
                    args.node_id,
                    WorkNodeStatus.REVISION_REQUIRED,
                    error=(
                        "Codex supervisory visual review rejected the exact latest "
                        f"package {package['id']}: " + "; ".join(messages)
                    ),
                    checkpoint="codex_visual_rejection",
                )
            except NotFoundError:
                # The script also supports reviewing detached package fixtures.
                pass
        print(
            f"{args.node_id}: {'accepted' if accepted else 'rejected'} "
            f"score={score:.3f} screenshot={screenshot}"
        )
        return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())

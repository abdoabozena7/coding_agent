"""Deterministic evaluation helpers for agent architecture regressions.

These benchmarks are intentionally lightweight and local.  They let the
orchestrator measure retrieval quality, regressions, and prompt-harness changes
without relying on a strong model to judge itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping, Protocol


class RetrievalIndex(Protocol):
    def hybrid_search(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] = (),
        relation_kinds: tuple[str, ...] = (),
    ) -> tuple[Any, ...]: ...


class BenchmarkStore(Protocol):
    def record_benchmark_result(
        self,
        *,
        suite_name: str,
        scenario_name: str,
        provider: str,
        model: str,
        inputs: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        scores: Mapping[str, Any] | None = None,
        result: str = "unknown",
        artifact_refs: tuple[str, ...] = (),
        ultra_run_id: str | None = None,
        blocker: str | None = None,
    ) -> Mapping[str, Any]: ...


class BenchmarkHistoryStore(BenchmarkStore, Protocol):
    def list_benchmark_results(
        self,
        *,
        suite_name: str | None = None,
        scenario_name: str | None = None,
        limit: int = 100,
    ) -> tuple[Mapping[str, Any], ...]: ...


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkCase:
    query: str
    expected_names: tuple[str, ...]
    kinds: tuple[str, ...] = ()
    relation_kinds: tuple[str, ...] = ()
    k: int = 5

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.expected_names:
            raise ValueError("retrieval benchmark case requires a query and expected names")
        object.__setattr__(self, "expected_names", tuple(str(item) for item in self.expected_names))
        object.__setattr__(self, "k", max(1, int(self.k)))


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkResult:
    suite_name: str
    case_results: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.case_results) and all(bool(item.get("passed")) for item in self.case_results)


@dataclass(frozen=True, slots=True)
class Html3DBenchmarkResult:
    suite_name: str
    scenario_name: str
    metrics: Mapping[str, float]
    scores: Mapping[str, float]
    findings: tuple[str, ...] = ()
    artifact_hash: str = ""

    @property
    def passed(self) -> bool:
        return self.scores.get("overall", 0.0) >= 0.8 and not self.findings


@dataclass(frozen=True, slots=True)
class BenchmarkTrendResult:
    suite_name: str
    scenario_name: str
    verdict: str
    latest_id: str
    baseline_id: str
    score_deltas: Mapping[str, float] = field(default_factory=dict)
    metric_deltas: Mapping[str, float] = field(default_factory=dict)
    changed_keys: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def regressed(self) -> bool:
        return self.verdict == "regressed"

    @property
    def improved(self) -> bool:
        return self.verdict == "improved"

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "scenario_name": self.scenario_name,
            "verdict": self.verdict,
            "latest_id": self.latest_id,
            "baseline_id": self.baseline_id,
            "score_deltas": dict(self.score_deltas),
            "metric_deltas": dict(self.metric_deltas),
            "changed_keys": list(self.changed_keys),
            "notes": list(self.notes),
            "regressed": self.regressed,
            "improved": self.improved,
        }


def _numeric_values(value: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in dict(value or {}).items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def _higher_is_better(key: str, *, score: bool) -> bool:
    if score:
        return True
    lowered = key.casefold()
    lower_is_better_markers = (
        "token",
        "cost",
        "latency",
        "wall_ms",
        "duration",
        "elapsed",
        "time_ms",
        "error",
        "failure",
        "failed",
        "blocker",
    )
    return not any(marker in lowered for marker in lower_is_better_markers)


def _classify_delta(key: str, delta: float, *, score: bool, tolerance: float) -> str:
    if abs(delta) <= tolerance:
        return "stable"
    higher_better = _higher_is_better(key, score=score)
    if higher_better:
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def analyze_benchmark_trend(
    results: tuple[Mapping[str, Any], ...],
    *,
    tolerance: float = 0.01,
) -> BenchmarkTrendResult:
    """Compare latest benchmark against the previous comparable run.

    ``results`` should be newest-first, matching StateStore.list_benchmark_results.
    The function is deterministic and schema-free: it uses numeric scores and
    metrics already stored in benchmark records.
    """

    if len(results) < 2:
        suite = str(results[0].get("suite_name") if results else "")
        scenario = str(results[0].get("scenario_name") if results else "")
        return BenchmarkTrendResult(
            suite,
            scenario,
            "insufficient_history",
            str(results[0].get("id") if results else ""),
            "",
            notes=("need at least two benchmark records",),
        )
    latest, baseline = dict(results[0]), dict(results[1])
    suite_name = str(latest.get("suite_name") or baseline.get("suite_name") or "")
    scenario_name = str(latest.get("scenario_name") or baseline.get("scenario_name") or "")
    latest_result = str(latest.get("result") or "").casefold()
    baseline_result = str(baseline.get("result") or "").casefold()
    score_deltas: dict[str, float] = {}
    metric_deltas: dict[str, float] = {}
    changed: list[str] = []
    votes: list[str] = []
    latest_scores = _numeric_values(latest.get("scores") if isinstance(latest.get("scores"), Mapping) else {})
    baseline_scores = _numeric_values(baseline.get("scores") if isinstance(baseline.get("scores"), Mapping) else {})
    latest_metrics = _numeric_values(latest.get("metrics") if isinstance(latest.get("metrics"), Mapping) else {})
    baseline_metrics = _numeric_values(baseline.get("metrics") if isinstance(baseline.get("metrics"), Mapping) else {})
    for key in sorted(set(latest_scores) & set(baseline_scores)):
        delta = latest_scores[key] - baseline_scores[key]
        score_deltas[key] = delta
        verdict = _classify_delta(key, delta, score=True, tolerance=tolerance)
        if verdict != "stable":
            changed.append(f"score:{key}")
            votes.append(verdict)
    for key in sorted(set(latest_metrics) & set(baseline_metrics)):
        delta = latest_metrics[key] - baseline_metrics[key]
        metric_deltas[key] = delta
        verdict = _classify_delta(key, delta, score=False, tolerance=tolerance)
        if verdict != "stable":
            changed.append(f"metric:{key}")
            votes.append(verdict)
    notes: list[str] = []
    if latest_result == "passed" and baseline_result != "passed":
        votes.append("improved")
        notes.append("result changed to passed")
    elif latest_result != "passed" and baseline_result == "passed":
        votes.append("regressed")
        notes.append("result changed away from passed")
    if "regressed" in votes:
        verdict = "regressed"
    elif "improved" in votes:
        verdict = "improved"
    else:
        verdict = "stable"
    return BenchmarkTrendResult(
        suite_name=suite_name,
        scenario_name=scenario_name,
        verdict=verdict,
        latest_id=str(latest.get("id") or ""),
        baseline_id=str(baseline.get("id") or ""),
        score_deltas=score_deltas,
        metric_deltas=metric_deltas,
        changed_keys=tuple(changed),
        notes=tuple(notes),
    )


def record_benchmark_trend(
    store: BenchmarkHistoryStore,
    *,
    suite_name: str,
    scenario_name: str,
    provider: str,
    model: str,
    limit: int = 20,
) -> Mapping[str, Any]:
    history = store.list_benchmark_results(
        suite_name=suite_name,
        scenario_name=scenario_name,
        limit=limit,
    )
    trend = analyze_benchmark_trend(tuple(history))
    return store.record_benchmark_result(
        suite_name="benchmark-trend",
        scenario_name=f"{suite_name}/{scenario_name}",
        provider=provider,
        model=model,
        inputs={
            "source_suite_name": suite_name,
            "source_scenario_name": scenario_name,
            "latest_id": trend.latest_id,
            "baseline_id": trend.baseline_id,
            "trend": trend.to_dict(),
        },
        metrics={
            **{f"score_delta:{key}": value for key, value in trend.score_deltas.items()},
            **{f"metric_delta:{key}": value for key, value in trend.metric_deltas.items()},
        },
        scores={
            "regression": 1.0 if trend.regressed else 0.0,
            "improvement": 1.0 if trend.improved else 0.0,
        },
        result="failed" if trend.regressed else "passed",
        artifact_refs=(f"benchmark:{trend.latest_id}", f"benchmark:{trend.baseline_id}") if trend.baseline_id else (),
        blocker=("benchmark regression detected" if trend.regressed else None),
    )


def learn_from_benchmark_trend(
    store: Any,
    trend_record: Mapping[str, Any],
    *,
    ultra_run_id: str | None = None,
) -> Mapping[str, Any]:
    """Promote actionable benchmark trends into cross-run project memory.

    Benchmark rows tell the agent *what* changed.  Project memory lets later
    runs reuse that evidence before planning and generation.  This hook is
    intentionally conservative: only clear regressions/improvements are learned,
    and only when an ULTRA run exists to anchor the memory fingerprint.
    """

    inputs = trend_record.get("inputs") if isinstance(trend_record.get("inputs"), Mapping) else {}
    trend = inputs.get("trend") if isinstance(inputs.get("trend"), Mapping) else {}
    verdict = str(trend.get("verdict") or "").casefold()
    if verdict not in {"regressed", "improved"}:
        return {
            "recorded": False,
            "reason": "trend_not_actionable",
            "verdict": verdict or "unknown",
        }

    run_id = str(ultra_run_id or trend_record.get("ultra_run_id") or "").strip()
    if not run_id:
        active_run = None
        get_active = getattr(store, "get_active_ultra_run", None)
        if callable(get_active):
            active_run = get_active()
        run_id = str(getattr(active_run, "id", "") or "").strip()
    if not run_id:
        return {
            "recorded": False,
            "reason": "no_active_ultra_run",
            "verdict": verdict,
        }

    from .project_brain import ProjectBrain

    source_suite = str(inputs.get("source_suite_name") or trend.get("suite_name") or "").strip()
    source_scenario = str(inputs.get("source_scenario_name") or trend.get("scenario_name") or "").strip()
    latest_id = str(inputs.get("latest_id") or trend.get("latest_id") or "").strip()
    baseline_id = str(inputs.get("baseline_id") or trend.get("baseline_id") or "").strip()
    changed_keys = tuple(str(item) for item in trend.get("changed_keys", ()) or ())
    notes = tuple(str(item) for item in trend.get("notes", ()) or ())
    trend_ref = str(trend_record.get("id") or "").strip()
    evidence_refs = tuple(
        item
        for item in (
            f"benchmark-trend:{trend_ref}" if trend_ref else "",
            f"benchmark:{latest_id}" if latest_id else "",
            f"benchmark:{baseline_id}" if baseline_id else "",
        )
        if item
    )

    if verdict == "regressed":
        title = f"Benchmark regression: {source_suite}/{source_scenario}"
        content = (
            f"{source_suite}/{source_scenario} regressed against the previous comparable run. "
            f"Changed signals: {', '.join(changed_keys) or 'result changed'}. "
            "Future runs must inspect the latest benchmark evidence, identify the lost capability, "
            "and keep the relevant gate active until the trend is stable or improved."
        )
        confidence = 0.84
    else:
        title = f"Benchmark improvement: {source_suite}/{source_scenario}"
        content = (
            f"{source_suite}/{source_scenario} improved against the previous comparable run. "
            f"Changed signals: {', '.join(changed_keys) or 'cost/quality improved'}. "
            "Future runs should preserve the orchestration, prompt, retrieval, or evaluation changes "
            "that produced this improvement while continuing to monitor regressions."
        )
        confidence = 0.72
    if notes:
        content += " Notes: " + "; ".join(notes[:5])

    entry = ProjectBrain(store, run_id).record_knowledge(
        title,
        content,
        data={
            "source": "benchmark_trend_learning",
            "trend_record_id": trend_ref,
            "trend": dict(trend),
        },
        confidence=confidence,
        evidence_refs=evidence_refs,
    )
    return {
        "recorded": True,
        "verdict": verdict,
        "brain_entry_id": entry.id,
        "title": title,
        "evidence_refs": evidence_refs,
    }


def run_repository_retrieval_benchmark(
    index: RetrievalIndex,
    cases: tuple[RetrievalBenchmarkCase, ...],
    *,
    suite_name: str = "repository-retrieval",
) -> RetrievalBenchmarkResult:
    if not cases:
        raise ValueError("retrieval benchmark requires at least one case")
    results: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        scored_search = getattr(index, "search_with_scores", None)
        if callable(scored_search):
            scored_hits = tuple(
                scored_search(
                    case.query,
                    kinds=case.kinds,
                    relation_kinds=case.relation_kinds,
                    limit=case.k,
                )
            )
            hits = tuple(getattr(item, "entry", item) for item in scored_hits)
            hit_details = tuple(
                {
                    "name": str(getattr(getattr(item, "entry", item), "name", "")),
                    "path": str(getattr(getattr(item, "entry", item), "path", "")),
                    "kind": str(getattr(getattr(item, "entry", item), "kind", "")),
                    "score": float(getattr(item, "score", 0.0) or 0.0),
                    "channels": tuple(str(channel) for channel in getattr(item, "channels", ()) or ()),
                }
                for item in scored_hits
            )
        else:
            hits = index.hybrid_search(
                case.query,
                kinds=case.kinds,
                relation_kinds=case.relation_kinds,
            )[: case.k]
            hit_details = tuple(
                {
                    "name": str(getattr(item, "name", "")),
                    "path": str(getattr(item, "path", "")),
                    "kind": str(getattr(item, "kind", "")),
                    "score": 0.0,
                    "channels": (),
                }
                for item in hits
            )
        names = tuple(str(getattr(item, "name", "")) for item in hits)
        expected = {name.casefold() for name in case.expected_names}
        rank = next(
            (position for position, name in enumerate(names, start=1) if name.casefold() in expected),
            0,
        )
        reciprocal_ranks.append((1.0 / rank) if rank else 0.0)
        results.append(
            {
                "query": case.query,
                "expected_names": case.expected_names,
                "top_names": names,
                "top_hits": hit_details,
                "passed": bool(rank),
                "rank": rank,
                "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            }
        )
    metrics = {
        "cases": float(len(cases)),
        "accuracy_at_k": sum(1.0 for item in results if item["passed"]) / len(results),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }
    return RetrievalBenchmarkResult(suite_name=suite_name, case_results=tuple(results), metrics=metrics)


def record_repository_retrieval_benchmark(
    store: BenchmarkStore,
    index: RetrievalIndex,
    cases: tuple[RetrievalBenchmarkCase, ...],
    *,
    provider: str,
    model: str,
    ultra_run_id: str | None = None,
    scenario_name: str = "hybrid-search",
    artifact_refs: tuple[str, ...] = (),
) -> Mapping[str, Any]:
    benchmark = run_repository_retrieval_benchmark(index, cases)
    blocker = None
    if not benchmark.passed:
        failed = [item["query"] for item in benchmark.case_results if not item.get("passed")]
        blocker = "retrieval benchmark missed expected symbols: " + ", ".join(failed[:5])
    return store.record_benchmark_result(
        suite_name=benchmark.suite_name,
        scenario_name=scenario_name,
        provider=provider,
        model=model,
        ultra_run_id=ultra_run_id,
        inputs={
            "cases": [
                {
                    "query": item["query"],
                    "expected_names": item["expected_names"],
                    "top_hits": item["top_hits"],
                }
                for item in benchmark.case_results
            ],
        },
        metrics=benchmark.metrics,
        scores={
            "accuracy_at_k": float(benchmark.metrics.get("accuracy_at_k", 0.0)),
            "mean_reciprocal_rank": float(benchmark.metrics.get("mean_reciprocal_rank", 0.0)),
        },
        result="passed" if benchmark.passed else "failed",
        artifact_refs=artifact_refs,
        blocker=blocker,
    )


def _present(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.I | re.S) for pattern in patterns)


def _count(text: str, patterns: tuple[str, ...]) -> int:
    return sum(len(re.findall(pattern, text, re.I | re.S)) for pattern in patterns)


def _score(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / maximum))


def _screenshot_quality_metrics(preview: Mapping[str, Any]) -> dict[str, float]:
    """Measure rendered composition instead of trusting feature-name counts.

    These intentionally model only observable visual signals, not taste.  They
    detect blank/flat/under-rendered candidates using colorfulness, luminance
    contrast, palette diversity, edge detail, scene occupancy, and exposure.
    """

    screenshot_path = str(preview.get("screenshot_path") or "").strip()
    if not screenshot_path or not Path(screenshot_path).is_file():
        return {"screenshot_available": 0.0}
    try:
        from PIL import Image

        with Image.open(screenshot_path) as source:
            image = source.convert("RGB")
            image.thumbnail((160, 90))
            width, height = image.size
            get_pixels = getattr(image, "get_flattened_data", image.getdata)
            pixels = list(get_pixels())
    except (ImportError, OSError, ValueError):
        return {"screenshot_available": 0.0}
    if width < 2 or height < 2 or not pixels:
        return {"screenshot_available": 0.0}

    luminance = [0.2126 * red + 0.7152 * green + 0.0722 * blue for red, green, blue in pixels]
    mean_luma = sum(luminance) / len(luminance)
    contrast = math.sqrt(sum((value - mean_luma) ** 2 for value in luminance) / len(luminance))
    rg = [float(red) - float(green) for red, green, _blue in pixels]
    yb = [0.5 * (float(red) + float(green)) - float(blue) for red, green, blue in pixels]
    mean_rg = sum(rg) / len(rg)
    mean_yb = sum(yb) / len(yb)
    std_rg = math.sqrt(sum((value - mean_rg) ** 2 for value in rg) / len(rg))
    std_yb = math.sqrt(sum((value - mean_yb) ** 2 for value in yb) / len(yb))
    colorfulness = math.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * math.sqrt(mean_rg ** 2 + mean_yb ** 2)

    quantized = [((red // 32), (green // 32), (blue // 32)) for red, green, blue in pixels]
    counts: dict[tuple[int, int, int], int] = {}
    for color in quantized:
        counts[color] = counts.get(color, 0) + 1
    dominant_fraction = max(counts.values()) / len(quantized)
    unique_colors = len(counts)

    edge_count = 0
    comparisons = 0
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if x + 1 < width:
                comparisons += 1
                edge_count += abs(luminance[index] - luminance[index + 1]) >= 18.0
            if y + 1 < height:
                comparisons += 1
                edge_count += abs(luminance[index] - luminance[index + width]) >= 18.0
    edge_density = edge_count / max(1, comparisons)
    clipped_fraction = sum(value <= 5.0 or value >= 250.0 for value in luminance) / len(luminance)

    colorfulness_score = _score(colorfulness, 80.0)
    contrast_score = _score(contrast, 64.0)
    diversity_score = _score(unique_colors, 96.0)
    detail_score = _score(edge_density, 0.18)
    occupancy_score = _score(1.0 - dominant_fraction, 0.75)
    exposure_score = 1.0 - _score(clipped_fraction, 0.7)
    composition_score = (
        0.25 * colorfulness_score
        + 0.20 * contrast_score
        + 0.20 * diversity_score
        + 0.20 * detail_score
        + 0.10 * occupancy_score
        + 0.05 * exposure_score
    )
    return {
        "screenshot_available": 1.0,
        "screenshot_width": float(width),
        "screenshot_height": float(height),
        "screenshot_colorfulness": round(colorfulness, 4),
        "screenshot_luminance_contrast": round(contrast, 4),
        "screenshot_unique_colors": float(unique_colors),
        "screenshot_edge_density": round(edge_density, 6),
        "screenshot_dominant_color_fraction": round(dominant_fraction, 6),
        "screenshot_clipped_fraction": round(clipped_fraction, 6),
        "screenshot_composition_score": round(composition_score, 4),
    }


def run_single_file_3d_html_benchmark(
    html: str,
    *,
    preview: Mapping[str, Any] | None = None,
    suite_name: str = "weak-model-html",
    scenario_name: str = "threejs-single-file",
) -> Html3DBenchmarkResult:
    text = str(html or "")
    lowered = text.casefold()
    preview = dict(preview or {})
    artifact_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    findings: list[str] = []

    has_html = "<html" in lowered or "<!doctype html" in lowered
    external_assets = re.findall(r"(?i)(?:src|href)\s*=\s*['\"](?!https://cdn\.jsdelivr\.net/npm/three|https://unpkg\.com/three)(https?://|/|\.{1,2}/)[^'\"]+", text)
    single_file = has_html and not external_assets
    three_or_webgl = _present(
        text,
        (
            r"\bTHREE\.",
            r"WebGLRenderer",
            r"getContext\(\s*['\"]webgl2?['\"]",
            r"<canvas\b",
        ),
    )
    scene_camera_renderer = sum(
        1
        for patterns in (
            (r"new\s+THREE\.Scene", r"\bscene\s*="),
            (r"PerspectiveCamera", r"OrthographicCamera", r"\bcamera\s*="),
            (r"WebGLRenderer", r"getContext\(\s*['\"]webgl2?['\"]"),
        )
        if _present(text, patterns)
    )
    animation = _present(text, (r"requestAnimationFrame", r"setAnimationLoop", r"\banimate\s*\("))
    materials_lights = _count(
        text,
        (
            r"Mesh[A-Za-z]*Material",
            r"PointLight",
            r"DirectionalLight",
            r"AmbientLight",
            r"SpotLight",
            r"emissive",
            r"roughness",
            r"metalness",
            r"shadow",
        ),
    )
    geometry_density = _count(
        text,
        (
            r"new\s+THREE\.[A-Za-z]*Geometry",
            r"BoxGeometry",
            r"SphereGeometry",
            r"PlaneGeometry",
            r"CylinderGeometry",
            r"BufferGeometry",
            r"InstancedMesh",
            r"\.add\(",
        ),
    )
    gameplay = _count(
        text,
        (
            r"addEventListener\(\s*['\"](?:keydown|keyup|pointer|mouse|touch)",
            r"\bscore\b",
            r"\bhealth\b",
            r"\blevel\b",
            r"\benem(?:y|ies)\b",
            r"\bprojectile",
            r"\bcollision",
            r"\bintersect",
            r"\bhit\b",
            r"\bvelocity\b",
        ),
    )
    responsive = _present(text, (r"resize", r"innerWidth", r"innerHeight", r"setSize", r"@media", r"viewport"))
    polish = _count(
        text,
        (
            r"gradient",
            r"fog",
            r"particle",
            r"postprocess",
            r"bloom",
            r"trail",
            r"screen shake",
            r"lerp",
            r"easing",
            r"hud",
        ),
    )
    accessibility = _present(text, (r"aria-", r"\brole=", r"tabindex", r"<title>"))
    preview_status = str(preview.get("verification") or preview.get("status") or "").casefold()
    preview_errors = tuple(
        str(item)
        for key in ("console_errors", "page_errors", "network_errors")
        for item in (preview.get(key) or ())
    )
    runtime_pass = not preview or (preview_status in {"passed", "ok", "success"} and not preview_errors)
    screenshot_metrics = _screenshot_quality_metrics(preview) if preview else {"screenshot_available": 0.0}

    metrics = {
        "bytes": float(len(text.encode("utf-8", errors="replace"))),
        "single_file": 1.0 if single_file else 0.0,
        "three_or_webgl": 1.0 if three_or_webgl else 0.0,
        "scene_camera_renderer": float(scene_camera_renderer),
        "animation_loop": 1.0 if animation else 0.0,
        "materials_lights": float(materials_lights),
        "geometry_density": float(geometry_density),
        "gameplay_signals": float(gameplay),
        "responsive": 1.0 if responsive else 0.0,
        "visual_polish_signals": float(polish),
        "accessibility": 1.0 if accessibility else 0.0,
        "runtime_pass": 1.0 if runtime_pass else 0.0,
        "runtime_errors": float(len(preview_errors)),
        **screenshot_metrics,
    }
    scores = {
        "self_contained": metrics["single_file"],
        "rendering_3d": (metrics["three_or_webgl"] + _score(scene_camera_renderer, 3)) / 2,
        "animation": metrics["animation_loop"],
        "gameplay": _score(gameplay, 6),
        "visual_richness": (_score(materials_lights, 8) + _score(geometry_density, 12) + _score(polish, 6)) / 3,
        "responsive_accessible": (metrics["responsive"] + metrics["accessibility"]) / 2,
        "runtime": metrics["runtime_pass"],
    }
    scores["visual_composition"] = (
        metrics.get("screenshot_composition_score", 0.0)
        if metrics.get("screenshot_available")
        else scores["visual_richness"]
    )
    scores["overall"] = round(
        0.12 * scores["self_contained"]
        + 0.17 * scores["rendering_3d"]
        + 0.10 * scores["animation"]
        + 0.16 * scores["gameplay"]
        + 0.14 * scores["visual_richness"]
        + 0.16 * scores["visual_composition"]
        + 0.05 * scores["responsive_accessible"]
        + 0.10 * scores["runtime"],
        4,
    )
    if not single_file:
        findings.append("HTML is not self-contained enough for the single-file benchmark")
    if scores["rendering_3d"] < 0.8:
        findings.append("3D/WebGL renderer, scene, and camera signals are incomplete")
    if not animation:
        findings.append("No animation loop detected")
    if scores["gameplay"] < 0.5:
        findings.append("Gameplay/input/collision/state signals are too weak")
    if scores["visual_richness"] < 0.45:
        findings.append("Visual richness signals are too weak for a showcase benchmark")
    if not runtime_pass:
        findings.append("Browser/runtime preview evidence failed")
    if preview.get("screenshot_path") and not metrics.get("screenshot_available"):
        findings.append("Rendered screenshot evidence is unavailable")
    elif metrics.get("screenshot_available") and scores["visual_composition"] < 0.42:
        findings.append("Rendered scene is visually flat, sparse, or under-composed")
    if findings:
        # Keep the numeric headline consistent with the hard-gate verdict.
        # A candidate with a runtime or observable visual blocker must never
        # present a misleading green-looking >=0.8 aggregate score.
        scores["overall"] = min(scores["overall"], 0.79)
    return Html3DBenchmarkResult(
        suite_name=suite_name,
        scenario_name=scenario_name,
        metrics=metrics,
        scores=scores,
        findings=tuple(findings),
        artifact_hash=artifact_hash,
    )


def record_single_file_3d_html_benchmark(
    store: Any,
    html: str,
    *,
    provider: str,
    model: str,
    ultra_run_id: str | None = None,
    artifact_ref: str = "workspace:index.html",
    preview: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    result = run_single_file_3d_html_benchmark(html, preview=preview)
    return store.record_benchmark_result(
        suite_name=result.suite_name,
        scenario_name=result.scenario_name,
        provider=provider,
        model=model,
        ultra_run_id=ultra_run_id,
        inputs={"artifact_hash": result.artifact_hash},
        metrics=result.metrics,
        scores=result.scores,
        result="passed" if result.passed else "failed",
        artifact_refs=(artifact_ref,),
        blocker="; ".join(result.findings) if result.findings else None,
    )

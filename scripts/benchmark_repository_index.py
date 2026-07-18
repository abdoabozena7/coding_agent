"""Synthetic large-repository benchmark for AST, incremental, and hybrid retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.repository_index import HashingEmbeddingProvider, RepositoryIndex


def _python_file(package: int, file_index: int, lines: int) -> str:
    values = ["from __future__ import annotations", "", f"PACKAGE = {package}", ""]
    function_index = 0
    while len(values) < lines:
        values.extend(
            (
                f"def feature_{package}_{file_index}_{function_index}(value: float) -> float:",
                '    """Synthetic ML/backend feature used by retrieval benchmarks."""',
                f"    normalized = value / {function_index + 1}.0",
                "    if normalized < 0:",
                "        return 0.0",
                "    score = normalized * 0.75",
                "    return min(1.0, score)",
                "",
            )
        )
        function_index += 1
    return "\n".join(values[:lines]) + "\n"


def run(files: int, lines_per_file: int, changed_files: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="agent-index-million-") as directory:
        root = Path(directory)
        for index in range(files):
            package = index // 100
            target = root / f"package_{package:03d}" / f"module_{index:05d}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_python_file(package, index, lines_per_file), encoding="utf-8")
        index = RepositoryIndex(root, embedding_provider=HashingEmbeddingProvider(dimensions=64))
        tracemalloc.start()
        started = time.perf_counter()
        index.update_all()
        cold_seconds = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        # Allocation tracing is intentionally scoped to the memory benchmark.
        # Keeping tracemalloc enabled while timing an incremental AST refresh
        # multiplies Python allocation cost and does not represent warm-index
        # production latency.
        tracemalloc.stop()
        query_latencies = []
        for query in (
            "normalized feature score",
            "negative value guard",
            "synthetic backend feature",
        ) * 10:
            query_started = time.perf_counter()
            index.search_with_scores(query, limit=20)
            query_latencies.append(time.perf_counter() - query_started)
        for relative in sorted(index.entries)[:changed_files]:
            target = root / relative
            target.write_text(target.read_text(encoding="utf-8") + "# incremental change\n", encoding="utf-8")
        incremental_started = time.perf_counter()
        index.update_all()
        incremental_seconds = time.perf_counter() - incremental_started
        p95 = sorted(query_latencies)[max(0, int(len(query_latencies) * 0.95) - 1)]
        total_lines = files * lines_per_file
        return {
            "files": files,
            "lines": total_lines,
            "cold_index_seconds": cold_seconds,
            "incremental_changed_files": changed_files,
            "incremental_seconds": incremental_seconds,
            "query_p95_seconds": p95,
            "query_mean_seconds": statistics.fmean(query_latencies),
            "peak_bytes": peak,
            "targets": {
                "lines_at_least_1m": total_lines >= 1_000_000,
                "incremental_under_5s": incremental_seconds < 5.0,
                "query_p95_under_2s": p95 < 2.0,
                "peak_under_4gb": peak < 4 * 1024**3,
            },
            "update_stats": dict(index.last_update_stats),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=1_000)
    parser.add_argument("--lines-per-file", type=int, default=1_000)
    parser.add_argument("--changed-files", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run(args.files, args.lines_per_file, args.changed_files), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

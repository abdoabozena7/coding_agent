from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(r"D:\projects\Ai\coding_agent")
LAB = REPO / "projects" / "agent-harness-bootstrap-01"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
OUT = REPO / "output" / "hill-climb-ultra" / "normal-bootstrap.stdout.log"
ERR = REPO / "output" / "hill-climb-ultra" / "normal-bootstrap.stderr.log"
PROMPT = (
    "Repair the Ultra master-plan recovery bug in agent/ultra.py. When structured "
    "master_plan output is invalid and modules are restored from accepted architecture "
    "components, expected write paths must be derived from each component's declared "
    "write_paths instead of being hardcoded to index.html. Preserve multiple paths and "
    "deduplicate them deterministically. Add focused regression tests to the existing "
    "Ultra test file proving a component owning agent/tools/read_file.py and "
    "tests/test_tools_security.py produces those exact expected paths and never "
    "index.html. Read only relevant ranges, make minimal changes, run the focused tests, "
    "and do not claim success without executed test output."
)

environment = os.environ.copy()
environment.update(
    {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_MODEL": "gemma4:e4b",
        "OLLAMA_NUM_GPU": "999",
        "OLLAMA_CONTEXT_SIZE": "16384",
        "AGENT_REQUIRE_LOCAL_GPU": "1",
        "AGENT_REPOSITORY_INDEX_WARMUP_FILES": "0",
        "AGENT_GLOBAL_MEMORY_PATH": str(
            REPO / "output" / "hill-climb-ultra" / "isolated-global-lessons-16k.json"
        ),
    }
)
with OUT.open("w", encoding="utf-8") as stdout, ERR.open("w", encoding="utf-8") as stderr:
    raise SystemExit(
        subprocess.call(
            [
                str(PYTHON),
                "-m",
                "agent",
                "--workspace",
                str(LAB),
                "--provider",
                "ollama",
                "--model",
                "gemma4:e4b",
                "--mode",
                "normal",
                "--permissions",
                "normal",
                "--plain",
                "--no-color",
                "--auto",
                "--command",
                "/replan Use read_file on agent/ultra.py only for lines 1688-1745, where the fallback hardcodes index.html. Produce exactly two changes: that minimal fallback repair and one regression in the existing test_ultra_v9.py. The implementation must flatten each restored component's write_paths in component order, remove duplicates while preserving first occurrence, and build expected_changes from those paths. Verify with one focused pytest node or -k expression. Do not inspect README, do not run a general baseline, and do not add unrelated work.",
            ],
            cwd=LAB,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
    )

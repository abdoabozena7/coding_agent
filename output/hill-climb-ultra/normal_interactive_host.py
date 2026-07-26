from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path


REPO = Path(r"D:\projects\Ai\coding_agent")
LAB = REPO / "projects" / "agent-harness-bootstrap-01"
INPUT = REPO / "output" / "hill-climb-ultra" / "normal-interactive-input.txt"
OUT = REPO / "output" / "hill-climb-ultra" / "normal-interactive.stdout.log"
ERR = REPO / "output" / "hill-climb-ultra" / "normal-interactive.stderr.log"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"


def pump(stream, destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for line in iter(stream.readline, ""):
            handle.write(line)
            handle.flush()


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
process = subprocess.Popen(
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
        "--interactive",
    ],
    cwd=LAB,
    env=environment,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)
assert process.stdin is not None
assert process.stdout is not None
assert process.stderr is not None
stdout_thread = threading.Thread(target=pump, args=(process.stdout, OUT), daemon=True)
stderr_thread = threading.Thread(target=pump, args=(process.stderr, ERR), daemon=True)
stdout_thread.start()
stderr_thread.start()

cursor = 0
while process.poll() is None:
    lines = INPUT.read_text(encoding="utf-8").splitlines()
    for line in lines[cursor:]:
        process.stdin.write(line + "\n")
        process.stdin.flush()
    cursor = len(lines)
    time.sleep(0.25)

stdout_thread.join(timeout=2)
stderr_thread.join(timeout=2)

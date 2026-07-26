from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


REPO = Path(r"D:\projects\Ai\coding_agent")
LAB = REPO / "projects" / "agent-harness-lab-01"
GAME = REPO / "projects" / "hill-climb-gemma4-final-v3"
OUTPUT = REPO / "output" / "hill-climb-ultra"
STDOUT = OUTPUT / "hill-climb-gemma4-final-v3-resume3.stdout.log"
STDERR = OUTPUT / "hill-climb-gemma4-final-v3-resume3.stderr.log"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"


environment = os.environ.copy()
environment.update(
    {
        "PYTHONPATH": str(LAB),
        "LLM_PROVIDER": "ollama",
        "OLLAMA_MODEL": "gemma4:e4b",
        "OLLAMA_NUM_GPU": "999",
        "OLLAMA_CONTEXT_SIZE": "16384",
        "AGENT_REQUIRE_LOCAL_GPU": "1",
        "AGENT_REPOSITORY_INDEX_WARMUP_FILES": "0",
        "AGENT_GLOBAL_MEMORY_PATH": str(
            OUTPUT / "hill-climb-gemma4-isolated-lessons.json"
        ),
    }
)

process = subprocess.Popen(
    [
        str(PYTHON),
        "-m",
        "agent",
        "--workspace",
        str(GAME),
        "--provider",
        "ollama",
        "--model",
        "gemma4:e4b",
        "--mode",
        "ultra",
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

lines: queue.Queue[str] = queue.Queue()


def pump_stdout() -> None:
    with STDOUT.open("w", encoding="utf-8", newline="") as handle:
        for line in iter(process.stdout.readline, ""):
            handle.write(line)
            handle.flush()
            lines.put(line)


def pump_stderr() -> None:
    with STDERR.open("w", encoding="utf-8", newline="") as handle:
        for line in iter(process.stderr.readline, ""):
            handle.write(line)
            handle.flush()


def send(value: str) -> None:
    if process.poll() is None:
        process.stdin.write(value + "\n")
        process.stdin.flush()


stdout_thread = threading.Thread(target=pump_stdout, daemon=True)
stderr_thread = threading.Thread(target=pump_stderr, daemon=True)
stdout_thread.start()
stderr_thread.start()
send("1")
time.sleep(0.35)
send("2")
time.sleep(0.5)
send("/resume")

approved_plans: set[int] = set()
last_default_answer = 0.0
completed = False
goal_sent = True
resume_sent = True
resume_only = True
while process.poll() is None:
    try:
        line = lines.get(timeout=0.5)
    except queue.Empty:
        continue
    match = re.search(r"Plan r(\d+) is ready", line)
    if match:
        revision = int(match.group(1))
        if revision not in approved_plans:
            approved_plans.add(revision)
            send(f"/approve {revision}")
        continue
    if "Allow this action once? [y/N]" in line:
        send("y")
        continue
    if "GA3BAD [ULTRA]>" in line and not goal_sent:
        goal_sent = True
        send("Build me a simple Hill Climb Racing game.")
        continue
    if "Choose an answer to continue" in line or "Reply normally, press D" in line:
        if resume_only:
            continue
        if not resume_sent:
            resume_sent = True
            send("/resume")
            continue
        now = time.monotonic()
        if now - last_default_answer > 3:
            last_default_answer = now
            send("Use recommended defaults")
        continue
    if "STATUS COMPLETED" in line or "Goal completed" in line:
        completed = True
    if completed and "GA3BAD [ULTRA]>" in line:
        send("/quit")
        completed = False

stdout_thread.join(timeout=2)
stderr_thread.join(timeout=2)

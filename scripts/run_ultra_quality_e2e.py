"""Run the canonical weak-prompt Ultra convergence scenario.

This is intentionally a real CLI process, not a fixture.  It chooses the three
recommended intake answers, performs the single user-equivalent plan approval,
then leaves quality-only revisions to the durable Ultra convergence loop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--prompt",
        default="اعمل لي لعبة زي Crossy Road بـThree.js",
    )
    parser.add_argument("--builder", default="gemma4:e4b")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    requested_workspace = Path(args.workspace).resolve()
    env = dict(os.environ)
    env.update(
        {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": args.builder,
            # The installed Gemma model is the fixed builder. Request full GPU
            # offload and fail visibly if the local CUDA runner cannot sustain
            # it; this canonical E2E must not silently fall back to CPU.
            "OLLAMA_NUM_GPU": "999",
            # Keep the weak local model's KV cache deliberately small. Durable
            # memory and NextActionPacket retrieval carry state between calls.
            "OLLAMA_CONTEXT_SIZE": "4096",
            "AGENT_REQUIRE_LOCAL_GPU": "1",
            "AGENT_ULTRA_FIX_ATTEMPTS": "6",
            "AGENT_ULTRA_MAX_DEPTH": "8",
            "AGENT_ULTRA_MAX_NODES": "1000",
        }
    )
    def fresh_recovery_workspace(attempt: int) -> Path:
        if attempt == 0:
            return requested_workspace
        candidate = requested_workspace.with_name(
            f"{requested_workspace.name}-recovery-{attempt:02d}"
        )
        suffix = 2
        while candidate.exists():
            candidate = requested_workspace.with_name(
                f"{requested_workspace.name}-recovery-{attempt:02d}-{suffix}"
            )
            suffix += 1
        return candidate

    def command_for(workspace: Path) -> list[str]:
        if workspace.exists():
            raise RuntimeError(
                f"recovery workspace must be fresh; refusing to overwrite {workspace}"
            )
        return [
            sys.executable,
            "-m",
            "agent",
            "--workspace",
            str(workspace),
            "--create-workspace",
            "--provider",
            "ollama",
            "--model",
            args.builder,
            "--mode",
            "ultra",
            "--permissions",
            "normal",
            "--plain",
            "--command",
            f"/goal {args.prompt}",
            "--command",
            "/answer platform Desktop browser",
            "--command",
            "/answer packaging Modular staging, best final",
            "--command",
            "/answer visual_direction Polished stylized",
            "--command",
            "/approve",
            "--auto",
        ]

    recovery_attempt = 0
    while True:
        workspace = fresh_recovery_workspace(recovery_attempt)
        command = command_for(workspace)
        env["AGENT_ULTRA_RECOVERY_ATTEMPT"] = str(recovery_attempt)
        env["AGENT_ULTRA_RECOVERY_WORKSPACE"] = str(workspace)
        print(
            f"Ultra attempt {recovery_attempt} workspace: {workspace}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=repository, env=env, check=False)
        if completed.returncode == 0:
            return 0
        recovery_attempt += 1
        delay = min(30.0, 2.0 ** min(recovery_attempt, 4))
        print(
            f"Ultra process exited {completed.returncode}; recovering the durable "
            f"goal in {delay:.0f}s (attempt {recovery_attempt}).",
            flush=True,
        )
        time.sleep(delay)


if __name__ == "__main__":
    raise SystemExit(main())

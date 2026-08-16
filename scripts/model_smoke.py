"""Bounded live smoke test for one discovered GA3BAD model.

The script verifies exact catalog inventory, the provider handshake, one
required structured action, and the persisted lifecycle projection. It never
executes workspace tools and uses a temporary state directory.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from agent.model_catalog import ModelCatalog
from agent.model_status import preflight_model_selection
from agent.runtime import AgentRuntime
from agent.store import StateStore


SMOKE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_model_smoke",
        "description": "Confirm the bounded model smoke test",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok"]},
                "model": {"type": "string"},
            },
            "required": ["status", "model"],
            "additionalProperties": False,
        },
    },
}


def run(model: str, *, no_inference: bool = False) -> dict[str, object]:
    catalog = ModelCatalog(timeout=10.0)
    descriptor = next(
        (item for item in catalog.discover() if item.model == model),
        None,
    )
    if descriptor is None:
        available = [item.model for item in catalog.discover()]
        raise RuntimeError(
            f"model {model!r} is not in the exact catalog inventory; available={available}"
        )
    provider = descriptor.create_provider()
    preflight_model_selection(provider, descriptor)
    if no_inference:
        return {
            "model": descriptor.to_dict(),
            "preflight": "passed",
            "inference": "not_run",
        }

    with tempfile.TemporaryDirectory(prefix="ga3bad-model-smoke-") as directory:
        workspace = Path(directory)
        with StateStore(workspace) as store:
            runtime = AgentRuntime(
                provider,
                store,
                workspace,
                model_descriptor=descriptor,
                session_id="model-smoke",
            )
            turn = runtime._call_provider(
                [
                    {
                        "role": "user",
                        "content": (
                            "Run the bounded model smoke test. Return exactly one "
                            f"submit_model_smoke call with status='ok' and model={model!r}."
                        ),
                    }
                ],
                [SMOKE_SCHEMA],
                (
                    "You are verifying provider transport only. Do not propose or execute "
                    "workspace work. Return exactly one submit_model_smoke call."
                ),
                actor="model-smoke",
                step=1,
                stream_text=False,
                require_tool_call=True,
            )
            matching = [
                call
                for call in turn.tool_calls
                if call.name == "submit_model_smoke"
                and call.args.get("status") == "ok"
            ]
            if len(matching) != 1:
                runtime.record_model_contract_status(
                    "model_smoke",
                    verified=False,
                    error="submit_model_smoke was not returned exactly once",
                    failure_kind="tool_contract_failed",
                )
                raise RuntimeError("submit_model_smoke was not returned exactly once")
            runtime.record_model_contract_status("model_smoke", verified=True)
            return {
                "model": descriptor.to_dict(),
                "preflight": "passed",
                "inference": "passed",
                "tool": matching[0].name,
                "arguments": dict(matching[0].args),
                "transport": (
                    "constrained_json"
                    if dict(turn.native or {}).get("action_transport")
                    else "native_tool"
                ),
                "lifecycle": runtime.model_status_snapshot(),
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--no-inference", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.model, no_inference=args.no_inference), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

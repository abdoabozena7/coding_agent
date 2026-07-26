"""Collect two small, isolated local-model specs for gameplay and presentation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.local_provider import extract_first_json_object
from agent.model_catalog import ExecutionClass, ModelDescriptor


PROMPTS = {
    "gameplay": (
        "Return JSON only. Keys: lane_count integer 3-5, hop_step number 0.8-1.5, "
        "hop_duration number 0.22-0.42, traffic_speeds array of 4 numbers 3.0-7.0, "
        "score_per_crossing integer 50-150, difficulty_gain number 0.04-0.12. "
        "Choose responsive fair values for a polished arcade lane-crossing game."
    ),
    "presentation": (
        "Return JSON only. Keys: camera_height number 10-16, camera_distance number "
        "12-20, fog_near number 24-40, fog_far number 60-95, environment_density "
        "integer 12-24, hud_accent #RRGGBB, sky #RRGGBB, grass #RRGGBB. Choose a "
        "sunlit stylized farm-edge diorama with crisp silhouettes and restrained HUD."
    ),
}


def clamp(value, low, high, default):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemma4:e4b")
    args = parser.parse_args()
    os.environ.setdefault("OLLAMA_NUM_GPU", "999")
    os.environ.setdefault("OLLAMA_CONTEXT_SIZE", "4096")
    os.environ.setdefault("AGENT_REQUIRE_LOCAL_GPU", "1")
    descriptor = ModelDescriptor(
        provider="ollama", model=args.model, execution_class=ExecutionClass.LOCAL,
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"), capabilities=("tools",),
        source="game-slice-specs", metadata={"gpu_required": True},
    )
    values = {}
    for name, prompt in PROMPTS.items():
        provider = descriptor.create_provider()
        provider.reasoning_effort = "off"
        provider.force_json = True
        provider.max_output_tokens = 220
        provider.temperature = 0.3
        turn = provider.call([{"role": "user", "content": prompt}], [], f"You are the isolated {name} specialist. Make bounded decisions only; do not write implementation code.")
        values[name] = dict(extract_first_json_object(turn.text) or {})
    game = values["gameplay"]
    game["lane_count"] = int(clamp(game.get("lane_count"), 3, 5, 4))
    game["hop_step"] = clamp(game.get("hop_step"), .8, 1.5, 1.1)
    game["hop_duration"] = clamp(game.get("hop_duration"), .22, .42, .32)
    speeds = list(game.get("traffic_speeds") or [3.4, 4.2, 5.0, 5.8])[:4]
    game["traffic_speeds"] = [clamp(item, 3, 7, 4 + i * .6) for i, item in enumerate(speeds)]
    while len(game["traffic_speeds"]) < 4:
        game["traffic_speeds"].append(4 + len(game["traffic_speeds"]) * .6)
    game["score_per_crossing"] = int(clamp(game.get("score_per_crossing"), 50, 150, 100))
    game["difficulty_gain"] = clamp(game.get("difficulty_gain"), .04, .12, .07)
    present = values["presentation"]
    present["camera_height"] = clamp(present.get("camera_height"), 10, 16, 13)
    present["camera_distance"] = clamp(present.get("camera_distance"), 12, 20, 16)
    present["fog_near"] = clamp(present.get("fog_near"), 24, 40, 30)
    present["fog_far"] = clamp(present.get("fog_far"), 60, 95, 78)
    present["environment_density"] = int(clamp(present.get("environment_density"), 12, 24, 18))
    for key, default in {"hud_accent": "#ffd34e", "sky": "#9bc9e8", "grass": "#79b96b"}.items():
        value = str(present.get(key) or default)
        present[key] = value if len(value) == 7 and value.startswith("#") else default
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "specs": values}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

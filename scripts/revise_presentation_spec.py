"""Ask one isolated Gemma critic for a bounded presentation revision."""

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


def clamp(value, low, high, default):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", required=True)
    parser.add_argument("--model", default="gemma4:e4b")
    args = parser.parse_args()
    path = Path(args.specs).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    os.environ.setdefault("OLLAMA_NUM_GPU", "999")
    os.environ.setdefault("OLLAMA_CONTEXT_SIZE", "4096")
    os.environ.setdefault("AGENT_REQUIRE_LOCAL_GPU", "1")
    descriptor = ModelDescriptor(
        provider="ollama", model=args.model, execution_class=ExecutionClass.LOCAL,
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"), capabilities=("tools",),
        source="presentation-revision", metadata={"gpu_required": True},
    )
    provider = descriptor.create_provider()
    provider.reasoning_effort = "off"
    provider.force_json = True
    provider.max_output_tokens = 220
    provider.temperature = 0.25
    prompt = (
        "The current sunlit low-poly lane-crossing scene is clean but lighting is slightly flat, "
        "roadside grass repeats trees, and the chicken is a little small. Return JSON only with "
        "key_intensity (1.15-1.55), ambient_intensity (0.38-0.62), rim_intensity (0.15-0.40), "
        "fence_segments (6-10), flower_clusters (6-12), hay_bales (2-5), character_scale "
        "(0.50-0.57). Choose restrained values that add depth without clutter."
    )
    turn = provider.call([{"role": "user", "content": prompt}], [], "You are the isolated Presentation revision specialist. Change only the bounded fields and return no implementation code.")
    revision = dict(extract_first_json_object(turn.text) or {})
    normalized = {
        "key_intensity": clamp(revision.get("key_intensity"), 1.15, 1.55, 1.34),
        "ambient_intensity": clamp(revision.get("ambient_intensity"), .38, .62, .50),
        "rim_intensity": clamp(revision.get("rim_intensity"), .15, .40, .24),
        "fence_segments": int(clamp(revision.get("fence_segments"), 6, 10, 8)),
        "flower_clusters": int(clamp(revision.get("flower_clusters"), 6, 12, 8)),
        "hay_bales": int(clamp(revision.get("hay_bales"), 2, 5, 3)),
        "character_scale": clamp(revision.get("character_scale"), .50, .57, .53),
    }
    payload["presentation"].update(normalized)
    payload["presentation_revision_evidence"] = {"raw": revision, "normalized": normalized, "finding": "flat_lighting_repetitive_roadside_small_character"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "revision": normalized}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

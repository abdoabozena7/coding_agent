"""Probe the fixed local builder with full-GPU Ollama inference."""

from __future__ import annotations

import json
import os
import urllib.request


def main() -> int:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return one compact JSON object with key status='READY' and "
                    "a checks array containing 40 short GPU stability checks."
                ),
            }
        ],
        "options": {
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "999")),
            "num_ctx": int(os.getenv("OLLAMA_CONTEXT_SIZE", "4096")),
            "num_predict": int(os.getenv("OLLAMA_PROBE_TOKENS", "512")),
        },
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if str(result.get("message", {}).get("content", "")).strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())

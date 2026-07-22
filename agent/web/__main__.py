"""Run the local GA3BAD web application with one command."""

from __future__ import annotations

import argparse
import os
import secrets
from threading import Timer
import webbrowser

import uvicorn

from .app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="GA3BAD local React + FastAPI workspace")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--registry")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    token = os.getenv("GA3BAD_LAUNCH_TOKEN", "").strip() or secrets.token_urlsafe(32)
    app = create_app(
        registry_path=args.registry,
        launch_token=token,
        dev=bool(args.dev),
    )
    url = f"http://{args.host}:{args.port}/?token={token}"
    print(f"GA3BAD web app: {url}", flush=True)
    if not args.no_browser:
        Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

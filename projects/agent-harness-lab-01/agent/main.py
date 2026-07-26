"""Backward-compatible script entrypoint.

Prefer ``python -m agent``.  ``python agent/main.py`` remains supported for the
original tutorial and delegates to the import-safe CLI without executing on
module import.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

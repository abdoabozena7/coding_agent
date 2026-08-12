"""Inspect or navigate an existing Playwright browser session."""

from . import browser_session
from .browser_open import RESULT_CONTRACT as BROWSER_RESULT_CONTRACT

REQUIRES_APPROVAL = False
SCHEMA = {
    "type": "function",
    "function": {
        "name": "browser_inspect",
        "description": (
            "Read the current page URL, title, visible text, browser errors, and "
            "authoritative interactive targets. Optionally navigate the same session first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "browser_session_id": {"type": "string", "minLength": 1, "maxLength": 120},
                "url": {"type": "string", "maxLength": 4096},
                "settle_ms": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 0},
            },
            "required": ["browser_session_id"],
            "additionalProperties": False,
        },
    },
}
RESULT_CONTRACT = dict(BROWSER_RESULT_CONTRACT)


def run(browser_session_id: str, url: str = "", settle_ms: int = 0) -> str:
    try:
        if str(url).strip():
            return browser_session.navigate(browser_session_id, url, settle_ms)
        return browser_session.inspect(browser_session_id, settle_ms)
    except RuntimeError as exc:
        return f"Error: browser_inspect failed: {exc}"

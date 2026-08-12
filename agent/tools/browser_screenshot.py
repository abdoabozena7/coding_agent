"""Capture the current state of a persistent Playwright page."""

from . import browser_session

REQUIRES_APPROVAL = False
SCHEMA = {
    "type": "function",
    "function": {
        "name": "browser_screenshot",
        "description": (
            "Capture the current browser state as a PNG under output/browser in "
            "the active workspace. Use a meaningful name for each distinct state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "browser_session_id": {"type": "string", "minLength": 1, "maxLength": 120},
                "name": {"type": "string", "maxLength": 120, "default": "screenshot"},
                "full_page": {"type": "boolean", "default": True},
            },
            "required": ["browser_session_id"],
            "additionalProperties": False,
        },
    },
}
RESULT_CONTRACT = {
    "format": "json",
    "fields": [
        "browser_session_id", "status", "url", "title", "screenshot_path",
        "workspace_path", "sha256", "byte_size", "image_width", "image_height",
        "console_errors", "page_errors", "network_errors",
    ],
}


def run(browser_session_id: str, name: str = "screenshot", full_page: bool = True) -> str:
    try:
        return browser_session.screenshot(browser_session_id, name, full_page)
    except RuntimeError as exc:
        return f"Error: browser_screenshot failed: {exc}"

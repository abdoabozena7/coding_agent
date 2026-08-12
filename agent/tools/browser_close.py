"""Close a managed Playwright browser session."""

from . import browser_session

REQUIRES_APPROVAL = False
SCHEMA = {
    "type": "function",
    "function": {
        "name": "browser_close",
        "description": "Close one managed browser session after all required captures are complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "browser_session_id": {"type": "string", "minLength": 1, "maxLength": 120},
            },
            "required": ["browser_session_id"],
            "additionalProperties": False,
        },
    },
}
RESULT_CONTRACT = {"format": "json", "fields": ["browser_session_id", "status"]}


def run(browser_session_id: str) -> str:
    return browser_session.close(browser_session_id)

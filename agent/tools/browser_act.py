"""Perform bounded user-like actions in a persistent browser session."""

from . import browser_session

REQUIRES_APPROVAL = True
_TARGET_PROPERTIES = {
    "selector": {"type": "string", "maxLength": 500},
    "role": {"type": "string", "maxLength": 80},
    "name": {"type": "string", "maxLength": 240},
    "exact": {"type": "boolean", "default": True},
}
SCHEMA = {
    "type": "function",
    "function": {
        "name": "browser_act",
        "description": (
            "Click, fill, press, select, check, uncheck, or hover on the current "
            "Playwright page. Targets come from browser_open/browser_inspect receipts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "browser_session_id": {"type": "string", "minLength": 1, "maxLength": 120},
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 40,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["click", "fill", "press", "select", "check", "uncheck", "hover", "upload"]},
                            **_TARGET_PROPERTIES,
                            "value": {"type": "string", "maxLength": 4000},
                            "key": {"type": "string", "maxLength": 100},
                            "files": {
                                "type": "array", "minItems": 1, "maxItems": 20,
                                "items": {"type": "string", "minLength": 1, "maxLength": 4096},
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
                "settle_ms": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 300},
            },
            "required": ["browser_session_id", "actions"],
            "additionalProperties": False,
        },
    },
}
RESULT_CONTRACT = {
    "format": "json",
    "fields": [
        "browser_session_id", "status", "url", "title", "actions", "text",
        "interaction_targets", "console_errors", "page_errors", "network_errors",
    ],
}


def run(browser_session_id: str, actions: list, settle_ms: int = 300) -> str:
    try:
        return browser_session.act(browser_session_id, actions, settle_ms)
    except RuntimeError as exc:
        return f"Error: browser_act failed: {exc}"

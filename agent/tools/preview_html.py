from . import web_preview
REQUIRES_APPROVAL = True
SCHEMA = {"type":"function","function":{"name":"preview_html","description":"Serve an HTML file securely on loopback, verify it in a real browser, and optionally run approval-bound interaction scenarios.","parameters":{"type":"object","properties":{"path":{"type":"string","minLength":1,"maxLength":4096},"open_browser":{"type":"boolean","default":True},"verify":{"type":"boolean","default":True},"settle_ms":{"type":"integer","minimum":0,"maximum":10000,"default":1500},"interactions":{"type":"array","maxItems":20,"items":{"type":"object","properties":{"name":{"type":"string","minLength":1,"maxLength":160},"reload":{"type":"boolean","default":True},"steps":{"type":"array","maxItems":40,"items":{"type":"object","properties":{"action":{"type":"string","enum":["click","press","fill"]},"role":{"type":"string","maxLength":80},"name":{"type":"string","maxLength":240},"selector":{"type":"string","maxLength":500},"key":{"type":"string","maxLength":80},"value":{"type":"string","maxLength":2000},"exact":{"type":"boolean","default":True}},"required":["action"],"additionalProperties":False}},"assertions":{"type":"array","minItems":1,"maxItems":20,"items":{"type":"object","properties":{"role":{"type":"string","maxLength":80},"name":{"type":"string","maxLength":240},"selector":{"type":"string","maxLength":500},"property":{"type":"string","enum":["value","text","textContent","id","visible","checked","count","visibleCount","dataObjectCount","dataVisualState"]},"equals":{"type":"string","maxLength":2000},"contains":{"type":"string","maxLength":2000},"exact":{"type":"boolean","default":True}},"required":["property"],"additionalProperties":False}}},"required":["name","steps","assertions"],"additionalProperties":False}}},"required":["path"],"additionalProperties":False}}}
RESULT_CONTRACT = {
    "format": "json",
    "fields": [
        "status", "preview_id", "url", "http_status", "browser_opened",
        "verification", "failure_kind", "interaction_targets", "console_errors", "page_errors", "network_errors",
        "screenshot_path", "interaction_results",
    ],
    "passed_evidence": (
        "verification=passed, HTTP 200, empty console/page/network error arrays, "
        "and an existing screenshot_path"
    ),
}
def run(path: str, open_browser: bool = True, verify: bool = True, settle_ms: int = 1500, interactions: list | None = None) -> str: return web_preview.create(path, open_browser, verify, settle_ms, interactions or [])

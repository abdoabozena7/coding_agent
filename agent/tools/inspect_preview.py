from . import web_preview
REQUIRES_APPROVAL = False
SCHEMA = {"type":"function","function":{"name":"inspect_preview","description":"Re-run browser verification for an active HTML preview.","parameters":{"type":"object","properties":{"preview_id":{"type":"string","minLength":1,"maxLength":128},"settle_ms":{"type":"integer","minimum":0,"maximum":10000,"default":500}},"required":["preview_id"],"additionalProperties":False}}}
def run(preview_id: str, settle_ms: int = 500) -> str: return web_preview.inspect(preview_id, settle_ms)

from . import web_preview
REQUIRES_APPROVAL = True
SCHEMA = {"type":"function","function":{"name":"preview_html","description":"Serve an HTML file securely on loopback, verify it in a real browser, and optionally open a visible isolated browser window.","parameters":{"type":"object","properties":{"path":{"type":"string","minLength":1,"maxLength":4096},"open_browser":{"type":"boolean","default":True},"verify":{"type":"boolean","default":True},"settle_ms":{"type":"integer","minimum":0,"maximum":10000,"default":1500}},"required":["path"],"additionalProperties":False}}}
def run(path: str, open_browser: bool = True, verify: bool = True, settle_ms: int = 1500) -> str: return web_preview.create(path, open_browser, verify, settle_ms)

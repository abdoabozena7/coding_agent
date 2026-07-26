from . import web_preview
REQUIRES_APPROVAL = True
SCHEMA = {"type":"function","function":{"name":"stop_preview","description":"Stop a preview server and browser created by preview_html.","parameters":{"type":"object","properties":{"preview_id":{"type":"string","minLength":1,"maxLength":128}},"required":["preview_id"],"additionalProperties":False}}}
def run(preview_id: str) -> str: return web_preview.stop(preview_id)

from . import process_manager
REQUIRES_APPROVAL = False
SCHEMA = {"type":"function","function":{"name":"poll_process","description":"Return status and exit code for a managed process.","parameters":{"type":"object","properties":{"process_id":{"type":"string","minLength":1,"maxLength":128}},"required":["process_id"],"additionalProperties":False}}}
def run(process_id: str) -> str: return process_manager.poll(process_id)

from . import process_manager
REQUIRES_APPROVAL = True
SCHEMA = {"type":"function","function":{"name":"stop_process","description":"Stop one process previously started by start_process.","parameters":{"type":"object","properties":{"process_id":{"type":"string","minLength":1,"maxLength":128}},"required":["process_id"],"additionalProperties":False}}}
def run(process_id: str) -> str: return process_manager.stop(process_id)

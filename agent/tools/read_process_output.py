from . import process_manager
REQUIRES_APPROVAL = False
SCHEMA = {"type":"function","function":{"name":"read_process_output","description":"Read the bounded tail of a managed process log.","parameters":{"type":"object","properties":{"process_id":{"type":"string","minLength":1,"maxLength":128},"lines":{"type":"integer","minimum":1,"maximum":2000,"default":100}},"required":["process_id"],"additionalProperties":False}}}
def run(process_id: str, lines: int = 100) -> str: return process_manager.output(process_id, lines)

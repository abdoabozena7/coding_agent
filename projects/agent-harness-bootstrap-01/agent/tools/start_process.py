"""Start a session-owned long-running process."""

from . import process_manager

REQUIRES_APPROVAL = True
SCHEMA = {"type":"function","function":{"name":"start_process","description":"Start a server or application that must remain running; returns a managed process_id.","parameters":{"type":"object","properties":{"command":{"type":"string","minLength":1,"maxLength":32768},"cwd":{"type":"string","default":"."},"readiness_type":{"type":"string","enum":["none","port","url","log"],"default":"none"},"readiness_value":{"type":"string","default":""},"timeout_seconds":{"type":"integer","minimum":0,"maximum":300,"default":30}},"required":["command"],"additionalProperties":False}}}

def run(command: str, cwd: str = ".", readiness_type: str = "none", readiness_value: str = "", timeout_seconds: int = 30) -> str:
    return process_manager.start(command, cwd, readiness_type, readiness_value, timeout_seconds)

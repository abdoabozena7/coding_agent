"""Start a session-owned long-running process."""

from . import process_manager

REQUIRES_APPROVAL = True
SCHEMA = {"type":"function","function":{"name":"start_process","description":"Start a server or application that must remain running. A real readiness check is mandatory. For browser work use port or loopback URL readiness; the verified readiness_url is the only URL that may be passed to browser_open.","parameters":{"type":"object","properties":{"command":{"type":"string","minLength":1,"maxLength":32768},"cwd":{"type":"string","default":"."},"readiness_type":{"type":"string","enum":["port","url","log"]},"readiness_value":{"type":"string","minLength":1},"timeout_seconds":{"type":"integer","minimum":0,"maximum":300,"default":30}},"required":["command","readiness_type","readiness_value"],"additionalProperties":False}}}
RESULT_CONTRACT = {
    "format": "json",
    "fields": ["process_id", "pid", "status", "ready", "readiness", "requested_readiness", "readiness_source", "readiness_url", "output_tail"],
    "lifecycle": "Use poll_process/read_process_output and stop_process with process_id.",
}

def run(command: str, cwd: str = ".", readiness_type: str = "none", readiness_value: str = "", timeout_seconds: int = 30) -> str:
    return process_manager.start(command, cwd, readiness_type, readiness_value, timeout_seconds)

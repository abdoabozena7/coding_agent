"""
read_file — the agent's first tool (read-only, no side effects).

A tool is two things bundled together:
  • SCHEMA — what the MODEL sees, so it knows the tool exists and how to call it.
  • run()  — what actually executes on our machine when the model asks for it.
"""

# What the model sees. This is OpenAI's "function tool" format: the model reads
# the description + parameters to decide WHEN to call it and WITH WHAT arguments.
REQUIRES_APPROVAL = False  # read-only — safe to run automatically

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read and return the full contents of a text file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the current directory.",
                },
            },
            "required": ["path"],
        },
    },
}


def run(path: str) -> str:
    """Read the file and return its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

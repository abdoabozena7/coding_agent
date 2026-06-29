import subprocess
import requests
# Change this to any model shown by: ollama list
MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# This list is only a description for the model.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text from a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a terminal command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]

def run_tool(name, args):
    """This is where tool calls become real actions on your computer."""
    try:
        if name == "write_file":
            with open(args["path"], "w", encoding="utf-8") as file:
                file.write(args["content"])
            return f"wrote file: {args['path']}"

        if name == "read_file":
            with open(args["path"], "r", encoding="utf-8") as file:
                return file.read()

        if name == "run_command":
            result = subprocess.run(
                args["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.stdout + result.stderr

        return f"unknown tool: {name}"
    except Exception as error:
        return f"tool error: {error}"


def ask_ollama(messages):
    """Send the conversation to Ollama and return the assistant message."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]


messages = [
    {
        "role": "system",
        "content": (
            "You are a coding agent. "
            "Use tools when you need to read files, write files, or run commands. "
            "When the task is finished, reply with a short final answer."
        ),
    }
]


print(f"Simple coding agent - model={MODEL}")
print("Type 'exit' to quit.\n")

while True:
    user_text = input("You> ").strip()

    if user_text.lower() in {"exit", "quit"}:
        break

    if not user_text:
        continue

    messages.append({"role": "user", "content": user_text})

    # One user request may need multiple tool calls.
    for step in range(1, 11):
        assistant_message = ask_ollama(messages)
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            print("Agent>", assistant_message.get("content", ""))
            print()
            break

        for tool_call in tool_calls:
            function = tool_call["function"]
            tool_name = function["name"]
            tool_args = function["arguments"]

            print(f"Tool {step}> {tool_name}({tool_args})")

            tool_result = run_tool(tool_name, tool_args)

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                }
            )
    else:
        print("Agent> stopped after too many tool steps")
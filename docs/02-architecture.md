# 02 — Architecture of *Our* Agent

The previous doc was the general theory. This one is concrete: the components we'll build, how data flows between them, and the design decisions behind them.

## 1. Components at a glance

Five responsibilities, each kept separate so you can understand them in isolation:

```
┌──────────────────────────────────────────────────────────────┐
│                         main.py (REPL)                         │
│   • reads your input    • runs the AGENT LOOP    • prints out   │
└───────────────┬───────────────────────────────┬──────────────┘
                │                                 │
                ▼                                 ▼
        ┌───────────────┐                 ┌───────────────────┐
        │    llm.py     │                 │   prompts.py      │
        │ talk to the   │                 │ the system prompt │
        │  OpenAI API   │                 └───────────────────┘
        └───────┬───────┘
                │ passes the tool catalog ▲ and runs tools ▼
                ▼
        ┌──────────────────────────────────────────────────┐
        │                    tools/                          │
        │  registry  → read_file, list_files, edit_file,     │
        │              run_bash, ...   (one file each)       │
        └──────────────────────────────────────────────────┘
```

| Component | Responsibility | Roughly |
|---|---|---|
| `main.py` | The REPL and the **agent loop**. The orchestrator. | ~60 lines |
| `llm.py` | A thin wrapper: build the request, call the API, return the response. | ~30 lines |
| `tools/` | Each tool = a function + its schema. A registry maps names → functions. | ~30 lines/tool |
| `prompts.py` | The system prompt string. | ~20 lines |
| `requirements.txt` / `.env` | Dependencies & API key. | — |

### A note on API choice

We use OpenAI's **Chat Completions API** directly — *not* the OpenAI **Agents SDK** or the higher-level **Responses API** agentic loop. Those are great for production, but they *run the agent loop for you*, which would hide the exact mechanic we're here to learn. Chat Completions is also perfectly **stateless** (you manage the message list yourself), which makes the lesson from [doc 01 §3](01-how-coding-agents-work.md) crystal clear. Once you've built the loop by hand, graduating to the Responses API / Agents SDK is a small step ([see references](04-references.md)).

## 2. The central data structure: `conversation`

Everything revolves around one growing list of messages (see [doc 01 §3](01-how-coding-agents-work.md)). It holds four kinds of messages:

| Message | Who makes it | Example |
|---|---|---|
| `role: "user"` | the human | `"add error handling to parse()"` |
| `role: "assistant"` (text) | the model | `"I'll start by reading the file."` |
| `role: "assistant"` with `tool_calls` | the model | `read_file(path="parse.py")` |
| `role: "tool"` | our harness | the file contents (or an error) |

The agent loop's entire job is appending messages to this list and deciding when to stop.

## 3. The agent loop in pseudocode

This is the heart of `main.py`. Read it carefully — it's the whole project in ~20 lines:

```python
import json

conversation = []

while True:
    user_input = input("you> ")
    conversation.append({"role": "user", "content": user_input})

    # Inner loop: let the model use as many tools as it needs
    # before it gives a final answer.
    while True:
        response = llm.call(conversation, tools=TOOL_SCHEMAS, system=SYSTEM_PROMPT)
        message = response.choices[0].message
        conversation.append(message)          # assistant message (may carry tool_calls)

        if not message.tool_calls:
            print(message.content)            # final answer → break to user
            break

        # The model asked for one or more tools. Run them all.
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)        # arguments arrive as a JSON string
            output = run_tool(call.function.name, args)       # ← the registry
            conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            })
        # loop again — the model now sees the results
```

Two nested loops:

- **Outer loop** = turns of conversation with *you*.
- **Inner loop** = the agent working autonomously (tool → result → tool → result…) until it has a final answer. A response whose `message.tool_calls` is non-empty (equivalently `finish_reason == "tool_calls"`) is the signal "I'm not done — run my tools and come back."

> **Key API detail:** when the model asks for tools, append its `assistant` message (containing `tool_calls`) to history *and then* append **one `role: "tool"` message per call**, each linked by the matching `tool_call_id`. The API rejects the next call if any `tool_call` is left unanswered. This pairing is the #1 thing beginners get wrong. (Also: `call.function.arguments` is a JSON *string* — always `json.loads` it.)

## 4. The tool registry pattern

Each tool is two things bundled together: a **schema** (what the model sees) and an **implementation** (what actually runs). We keep them next to each other and register them by name.

```python
# tools/read_file.py
SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the full contents of a file at a given path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file"}},
            "required": ["path"],
        },
    },
}

def run(path: str) -> str:
    with open(path) as f:
        return f.read()
```

```python
# tools/__init__.py  — the registry
from . import read_file, list_files, edit_file, run_bash

TOOLS = [read_file, list_files, edit_file, run_bash]

TOOL_SCHEMAS = [t.SCHEMA for t in TOOLS]                       # sent to the model
_BY_NAME = {t.SCHEMA["function"]["name"]: t for t in TOOLS}    # name → module

def run_tool(name: str, args: dict) -> str:
    try:
        return _BY_NAME[name].run(**args)
    except Exception as e:
        return f"Error: {e}"     # errors are fed back to the model, not crashed on
```

Why this shape:

- **Adding a tool = adding one file + one import.** Nothing else changes. This makes the incremental roadmap clean.
- **Errors become tool results, not crashes.** If `read_file` hits a missing path, the model *sees* the error and can recover (e.g. `list_files` first). An agent that crashes on a bad tool call is useless.

## 5. `llm.py` — the thin wrapper

Keeps all OpenAI-specific code in one place. Note that OpenAI passes the system prompt as the **first message** (`role: "system"`), not as a separate parameter — so the wrapper prepends it:

```python
from openai import OpenAI

client = OpenAI()    # reads OPENAI_API_KEY from the env

def call(conversation, tools, system, model="gpt-5.5"):
    messages = [{"role": "system", "content": system}] + conversation
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
```

That's the entire integration. Later we extend this one function for **streaming** (`stream=True`, show text as it's generated) and tune **reasoning effort** — without touching the loop or the tools.

> **Model choice:** examples use `gpt-5.5`, the model configured in `llm.py`. If you want to try a different model later, change the model string in that file.

## 6. Design decisions & why

| Decision | Why |
|---|---|
| One file per tool | Each phase adds exactly one file; easy to reason about and test in isolation. |
| Tool errors → tool results | Keeps the agent resilient; lets the model self-correct instead of the program dying. |
| Chat Completions, no framework | The mechanics *are* the lesson; the Agents SDK / Responses loop would hide them. |
| Read-only tools before write/exec | Safest capabilities first; you can trust the agent before giving it dangerous powers. |
| Permission gate only on risky tools | Auto-run safe stuff (reads); pause for irreversible stuff (bash, deletes). Reversibility is the criterion. |
| Single `conversation` list | Mirrors the real API shape exactly; no hidden state to get confused by. |

## 7. How it grows (preview of the roadmap)

The architecture above is the *destination*. We get there in small steps:

- Start: everything in **one file**, no tools — just a chat loop.
- Then: extract `llm.py`, add the **tool infrastructure** + first read-only tool.
- Then: more tools, the system prompt, streaming, permissions.
- Finally: context management and a couple of stretch features.

Each step is runnable on its own. The full sequence is in **[03 — Roadmap](03-roadmap.md)**.

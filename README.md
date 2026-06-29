# Building a Coding Agent from Scratch

A learning project: build a simple but real **coding agent** (think a tiny Claude Code / Cursor / Codex), step by step, from nothing — to understand how complex AI agents are actually structured under the hood.

> **The one-sentence mental model:** *An agent is just an LLM, a loop, and some tools.* Everything else is refinement. ([source](https://ampcode.com/notes/how-to-build-an-agent))

## Project description

This repository is an educational, from-scratch Python implementation of a terminal coding agent. It shows the core mechanics behind modern AI coding tools: a model-driven agent loop, tool calling, file exploration, targeted code edits, shell command execution, permission checks, streaming responses, usage reporting, context compaction, and pluggable OpenAI/Gemini providers.

It is meant to be read, modified, and learned from. It is not a production coding agent, but a small reference implementation for understanding how production coding agents are structured under the hood.

## What we're building

By the end, a command-line agent that can:

- Hold a conversation with you in your terminal
- **Read** files in a project, **list/search** directories
- **Edit** and **create** files
- **Run** shell commands (run tests, install deps, grep, etc.)
- Decide *on its own* which of those to do, in what order, to accomplish a task you describe in plain English

The whole thing is ~300–500 lines of Python. The point isn't the line count — it's that once you've built it, the "magic" of coding agents disappears and you understand every moving part.

## Why this is worth doing

Modern coding agents (Claude Code, Cursor, Codex, Copilot agents) look intimidating, but they share one surprisingly small core. As Simon Willison puts it, *"a simple tool loop can be achieved with a few dozen lines of code on top of an existing LLM API."* Building it yourself teaches you:

- The **agent loop** — the heartbeat of every agent
- **Tool use / function calling** — how an LLM "does things" in the real world
- **Context management** — how a stateless model holds a long conversation
- **System prompts** — how you steer behavior
- **Safety & permissions** — why agents ask before running `rm -rf`

These are the exact same concepts that scale up to production agents.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| Language | **Python** | Most readable; minimal boilerplate; keeps focus on agent concepts |
| Model | **OpenAI or Gemini** | Pluggable providers; OpenAI (GPT-5.5) by default, switch to Gemini with one env var |
| SDK | `openai` + `google-genai` (official) | Handle the API, tool schemas, streaming |
| Interface | Terminal (REPL) | Simplest possible UI; no web/UI distractions |

## How to use this documentation

Read the docs in order. Each builds on the last.

1. **[docs/01-how-coding-agents-work.md](docs/01-how-coding-agents-work.md)** — The big picture. The agent loop, tools, and the core mental model. *Read this first.*
2. **[docs/02-architecture.md](docs/02-architecture.md)** — How *our* agent is structured: the components, the data flow, and how each file fits together.
3. **[docs/03-roadmap.md](docs/03-roadmap.md)** — The incremental plan. Numbered phases, each one a small, runnable milestone. **This is the build plan we'll follow.**
4. **[docs/04-references.md](docs/04-references.md)** — Sources and further reading from the research that informed this plan.

## Repository layout

```
coding-agent-from-scratch/
├── README.md                 # you are here
├── docs/                     # the plan & learning material
│   ├── 01-how-coding-agents-work.md
│   ├── 02-architecture.md
│   ├── 03-roadmap.md
│   └── 04-references.md
├── agent/                    # the agent itself
│   ├── __init__.py
│   ├── main.py               # entry point: the REPL + agent loop + permission gate
│   ├── llm.py                # provider-agnostic LLM wrapper
│   ├── context.py            # conversation history + compaction
│   ├── prompts.py            # the system prompt
│   ├── ui.py                 # terminal input/output helpers
│   ├── providers/            # pluggable LLM backends
│   │   ├── __init__.py
│   │   ├── base.py           # the provider interface
│   │   ├── openai_provider.py
│   │   └── gemini_provider.py
│   └── tools/                # one file per tool
│       ├── __init__.py       # tool registry
│       ├── read_file.py
│       ├── list_files.py
│       ├── grep.py
│       ├── edit_file.py
│       ├── write_file.py
│       └── run_bash.py
├── requirements.txt
├── LICENSE                   # MIT
└── .env.example              # API keys for the provider you use
```

> In early phases the whole agent lives in a single file — it's split out into the structure above as it grows (see the roadmap).

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/osama96gh/coding-agent-from-scratch.git
cd coding-agent-from-scratch

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env             # then edit .env and paste your key

# 4. Run the agent
python3 agent/main.py             # OpenAI (default)
LLM_PROVIDER=gemini python3 agent/main.py   # or Gemini
```

Then just talk to it in your terminal — ask it to read a file, run your tests, make an edit. Type `exit` (or Ctrl-C) to quit.

## Status

✅ **Built.** The agent is complete through Phase 8 — a working terminal agent with a tool loop, file read/write/edit, search, bash execution, a permission gate for risky actions, multi-provider support (OpenAI + Gemini), streaming, usage reporting, and conversation compaction. Follow the build yourself via the [roadmap](docs/03-roadmap.md), which breaks it into numbered, runnable phases.

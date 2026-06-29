# 01 — How Coding Agents Work (The Big Picture)

Before we write any code, let's build the mental model. Everything we do later is an instance of the ideas on this page.

## 1. The core insight

A coding agent is **three things**:

```
Agent = LLM  +  Loop  +  Tools
```

- **LLM** — the "brain." It reads text and writes text. That's *all* it can do natively. It cannot read your files, run your code, or change anything. It just predicts text.
- **Tools** — the "hands." Functions *you* write (read a file, run a command) that the LLM is *allowed to ask you to run on its behalf*.
- **Loop** — the "heartbeat." A simple `while` loop that keeps sending the conversation to the LLM, running any tools it asks for, feeding the results back, and repeating until the task is done.

That's the whole thing. The cleverness lives in the model (which is trained to use tools well) — not in our harness, which stays small.

## 2. The agent loop, step by step

This is the single most important diagram in this project:

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│   1. User types a request                                     │
│            │                                                  │
│            ▼                                                  │
│   2. Add it to the conversation history                       │
│            │                                                  │
│            ▼                                                  │
│   3. Send the WHOLE conversation + tool list to the LLM ──┐   │
│            │                                              │   │
│            ▼                                              │   │
│   4. LLM responds. Two possibilities:                    │   │
│            │                                              │   │
│       ┌────┴─────────────────┐                           │   │
│       ▼                      ▼                            │   │
│  (a) Plain text         (b) "Please run                  │   │
│      answer.                 tool X with                  │   │
│      → show user,           these args"                  │   │
│        STOP. ✅              │                            │   │
│                             ▼                            │   │
│                   5. Harness RUNS the tool                │   │
│                      (read file, run cmd…)                │   │
│                             │                             │   │
│                             ▼                             │   │
│                   6. Add tool result to the               │   │
│                      conversation ──────────────────────►─┘   │
│                      (loop back to step 3)                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

Walk through a real example — you ask: *"What does `main.py` do?"*

1. The LLM doesn't know — it can't see your disk. So it responds: *"call `read_file('main.py')`"*.
2. Your harness runs `read_file`, gets the contents, and sends them back as a "tool result."
3. The LLM now has the file in context, so it responds with a plain-text explanation. Loop ends.

Now a harder one — *"add a test for the `parse()` function"*:

1. LLM → `list_files()` (to see the project)
2. LLM → `read_file('parser.py')` (to understand `parse`)
3. LLM → `read_file('tests/test_parser.py')` (to match the existing test style)
4. LLM → `edit_file(...)` (to add the test)
5. LLM → `run_bash('pytest')` (to check it passes)
6. LLM → plain text: *"Done. I added a test and it passes."* ✅

**Nobody told the model to do those steps in that order.** It chained them itself, reacting to each tool result. That autonomy *is* the agent.

## 3. The LLM is stateless — so we replay everything

This trips up everyone at first. The LLM API (OpenAI's Chat Completions) has **no memory**. Each API call is independent. The model only "remembers" the conversation because **we send the entire history every single time.**

So our `conversation` is just a growing list of messages:

```
[
  {role: "user",      content: "What does main.py do?"},
  {role: "assistant", tool_calls: [ read_file(path="main.py") ]},
  {role: "tool",      tool_call_id: "...", content: "<contents of main.py>"},
  {role: "assistant", content: "It's the entry point that..."},
]
```

Notice tool calls and tool results are *just more messages* in that list. The loop's job is to keep appending to it.

> **Cost implication:** because we resend everything, long conversations cost more tokens each turn. Later we'll use **prompt caching** to make the repeated prefix cheap. For now, just know *why* we resend.

## 4. What a "tool" actually is

A tool is a normal function plus a **description the LLM can read**. You give the model a little catalog:

```jsonc
{
  "name": "read_file",
  "description": "Read the full contents of a file at the given path.",
  "input_schema": {
    "type": "object",
    "properties": { "path": { "type": "string", "description": "File path" } },
    "required": ["path"]
  }
}
```

The model uses the `description` and schema to decide *when* and *how* to call it. Two big lessons from the research:

- **You don't have to beg.** You don't need "IF YOU NEED A FILE, USE read_file!!!" — modern models (like GPT-5.5) are trained to recognize when a tool helps and to call it with the right arguments. Clear descriptions are enough. (Over-aggressive instructions can actually *backfire* on current models.)
- **Tool results are ground truth.** Every tool result is real feedback from the actual environment (the real file, the real test output). This is what keeps the agent grounded instead of hallucinating — it's reacting to reality at each step.

## 5. Typical coding-agent toolset

Almost every coding agent converges on roughly this set. We'll build them in this order (read-only first, write/execute later — safest things first):

| Tool | What it does | Risk |
|---|---|---|
| `list_files` | List files in a directory | none (read-only) |
| `read_file` | Read a file's contents | none (read-only) |
| `grep` / search | Find text across files | none (read-only) |
| `edit_file` | Replace text in a file | ⚠️ changes your code |
| `write_file` | Create / overwrite a file | ⚠️ changes your code |
| `run_bash` | Run a shell command | 🚨 can do *anything* |

> **Insight:** a `run_bash` tool is almost *too* powerful — with it the agent can do nearly everything (including the other tools). Dedicated tools like `edit_file` exist not because bash can't edit files, but because a dedicated tool gives *your harness* a typed, inspectable hook it can validate, display nicely, or gate behind a permission prompt. (More on this in the architecture doc.)

## 6. The system prompt: steering the behavior

The **system prompt** is a block of instructions the user never sees, sent at the top of every conversation. It's where you define the agent's "personality" and rules:

- *"You are a coding assistant working in the user's project directory."*
- *"Prefer reading a file before editing it."*
- *"After changing code, run the tests."*
- *"Keep responses concise."*

In production agents this can be hundreds of lines. We'll start with a short one and grow it.

## 7. Safety & permissions (why agents pause)

An agent with `run_bash` can delete files, push to git, or call APIs. So real agents add a **permission layer**: before running anything irreversible, the harness pauses and asks the human *"OK to run `git push`? [y/n]"*.

The useful rule of thumb: **gate actions that are hard to reverse.** Reading a file is safe and automatic; deleting one should ask first. We'll add a simple version of this.

## 8. Workflows vs. agents (a useful distinction)

Anthropic's *Building Effective Agents* draws a line worth remembering:

- **Workflow** — you, the developer, hard-code the sequence of LLM/tool steps. Predictable, but rigid.
- **Agent** — the *LLM* decides the sequence at runtime, reacting to feedback in a loop. Flexible, but less predictable.

What we're building is a true **agent** — the model drives. Their other key advice, which we follow throughout: *start simple, use the API directly, avoid heavy frameworks* — they hide the very mechanics we're trying to learn.

---

### Recap

- Agent = **LLM + loop + tools**.
- The **loop**: send history → model asks for a tool → run it → feed result back → repeat → stop on a plain-text answer.
- The model is **stateless**; we **replay the whole conversation** every turn.
- A **tool** = a function + a description the model reads to decide when to call it.
- **System prompt** steers behavior; **permissions** keep dangerous actions safe.

Next: **[02 — Architecture](02-architecture.md)** — how we turn this mental model into actual Python files.

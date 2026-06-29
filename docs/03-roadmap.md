# 03 — The Incremental Roadmap (Build Plan)

This is the plan we'll follow. Each **phase** is a small, self-contained milestone that *runs on its own* and teaches one concept. We never write a big pile of code at once — we grow the agent one capability at a time, exactly the way the best tutorials (Thorsten Ball, Simon Willison) recommend.

**Legend:** 🎯 goal · 🧠 concept you learn · ✅ done-when (how you'll know it works)

---

## Phase 0 — Project setup

🎯 A working Python project that can talk to the OpenAI API.

- Create a virtualenv, `requirements.txt` with `openai` (and `python-dotenv`).
- Add `.env.example` and `.env` with `OPENAI_API_KEY`.
- Write a 5-line script that sends *"Hello"* to the API and prints the reply.

🧠 How the OpenAI SDK is initialized; how the API key is read from the environment; the shape of a basic `chat.completions.create` request and response.

✅ **Done when:** running the script prints a real reply from GPT-5.5.

---

## Phase 1 — The bare chat loop (no tools yet)

🎯 A terminal REPL that holds a multi-turn conversation.

- A `while True` loop: read input → append to `conversation` → call the API → print → repeat.
- Maintain the `conversation` list across turns.

🧠 **The most important lesson of the whole project:** the API is *stateless*. The model only "remembers" because we resend the full `conversation` every turn. Prove it to yourself: tell it your name, then ask for it back.

✅ **Done when:** you can have a coherent back-and-forth and it remembers earlier turns.

---

## Phase 2 — Tool infrastructure + your first tool (`read_file`)

🎯 Give the agent its first "hand." This is where a chatbot becomes an *agent*.

- Define the **tool schema** format (name, description, `input_schema`).
- Build the **inner loop**: detect tool calls (`message.tool_calls` is non-empty / `finish_reason == "tool_calls"`), run the tool, append a `role:"tool"` result, loop back.
- Implement `read_file` (read-only — the safest possible first tool).
- Set up the `tools/` package + registry (`run_tool`, `TOOL_SCHEMAS`).

🧠 Tool use / function calling end-to-end. The assistant `tool_calls` → `role:"tool"` result pairing (linked by `tool_call_id`); remembering to `json.loads` the arguments string. The inner loop. Why we run a loop, not a single call.

✅ **Done when:** you ask *"what's in `README.md`?"* and the agent reads the file itself and answers.

---

## Phase 3 — Read-only exploration tools (`list_files`, `grep`)

🎯 Let the agent *explore* a codebase before acting.

- `list_files(path=".")` — list a directory (recursive, with a sensible ignore list for `.git`, `node_modules`, etc.).
- `grep(pattern, path=".")` — search file contents for text.

🧠 **Tool chaining.** Watch the agent autonomously `list_files` → `grep` → `read_file` to answer a question like *"where is the `parse` function defined?"* — with no instructions on the order. This is the autonomy that defines an agent.

✅ **Done when:** the agent can answer questions about an unfamiliar project by exploring it on its own.

---

## Phase 4 — Write tools (`edit_file`, `write_file`)

🎯 The agent can now *change* your code. (We cross from read-only into mutation here — deliberately after exploration works.)

- `write_file(path, content)` — create or overwrite a file.
- `edit_file(path, old_str, new_str)` — targeted string replacement (the workhorse of code editing; create-if-missing when `old_str` is empty).

🧠 Why `edit_file` (targeted replace) usually beats `write_file` (full overwrite) for editing — smaller diffs, less to get wrong. A first taste of *why we'll want permissions* (the agent can now break things).

✅ **Done when:** you say *"create a `fizzbuzz.py` and then change it to stop at 50"* and it writes, then edits, the file correctly.

---

## Phase 5 — The shell tool (`run_bash`)

🎯 The single most powerful tool: run arbitrary shell commands.

- `run_bash(command)` — run a command, capture stdout/stderr/exit code, feed it all back.

🧠 How one general tool unlocks huge capability (run tests, install packages, use `git`, even substitute for the other tools). And the flip side: this is *dangerous* — which motivates Phase 7. Also: feeding **exit codes and stderr** back is what lets the agent *debug* (run tests → see failure → fix → re-run).

✅ **Done when:** you say *"run the tests and fix any failures"* and it runs `pytest`, reads the output, edits code, and re-runs until green.

---

## Phase 6 — System prompt + UX polish

🎯 Make it feel like a real tool, and steer its behavior.

- Move the system prompt into `prompts.py` (sent as the first `role:"system"` message); give it real guidance (working directory, "read before editing," "run tests after changes," "be concise").
- Add **streaming** (`stream=True`) so text appears token-by-token instead of after a long pause.
- Nicer terminal output: clearly show which tool is being called with what arguments.

🧠 How the system prompt shapes behavior (and that *over*-instructing current models can backfire — keep it crisp). How streaming works (`stream=True` on `chat.completions.create`) and why it matters for UX on long responses.

✅ **Done when:** the agent behaves noticeably more "on-task," and you see responses stream in live.

---

## Phase 7 — Safety & permissions

🎯 Stop the agent from doing irreversible things without your OK.

- A permission gate: before `run_bash` (and destructive edits), print the action and ask `[y/n]`.
- Mark read-only tools as auto-approved; gate the risky ones.
- ✅ An allow-list of always-safe commands (`ls`, `cat`, `git status`, `pytest`, …) to reduce prompt fatigue. The gate is now **argument-aware**: `run_bash.requires_approval(args)` inspects the command and auto-approves read-only ones, while anything with shell chaining/redirection/substitution (`|`, `>`, `;`, `$(…)`) or a non-allow-listed program still asks. Dual-use commands like `git branch`/`git remote` deliberately stay gated.

🧠 The reversibility criterion for gating actions. Human-in-the-loop checkpoints. Why a typed `edit_file` tool is easier to gate than an opaque `bash` string — the architectural reason dedicated tools exist alongside bash.

✅ **Done when:** the agent pauses for confirmation before running shell commands, and you can deny one.

---

## Phase 8 — Context management (handling long sessions)

🎯 Keep the agent fast and affordable over long conversations.

- Lean on OpenAI's **automatic prompt caching** — repeated prompt prefixes are cached and billed at a discount with no code change. The job is to *keep the prefix stable*: put the unchanging parts (system prompt, tool list) first so the cache actually hits.
- Verify cache hits via `usage.prompt_tokens_details.cached_tokens`.
- (Optional) **manual compaction** — when the conversation gets very long, summarize old turns ourselves (OpenAI has no server-side compaction) so you don't blow the context window.

🧠 Why resending the whole conversation gets expensive (ties back to statelessness from Phase 1), and the two standard fixes: let the cached prefix do the heavy lifting, and summarize the old stuff yourself. This is exactly how production agents stay viable in long sessions.

✅ **Done when:** in a long session, repeated context shows up as `cached_tokens` in the usage numbers instead of being re-billed in full every turn.

---

## Phase 9 — Stretch goals (pick what interests you)

🎯 Optional extensions that mirror features of real production agents.

- **Sub-agents** — spawn a fresh agent for a focused sub-task (e.g. "explore the codebase") to keep the main context clean.
- **More tools** — `apply_patch` (unified diffs), web search, a TODO/task tracker tool.
- **MCP (Model Context Protocol)** — connect external tool servers instead of hand-writing every tool (supported natively by OpenAI's Responses API).
- **A different interface** — a simple web UI or a `--print` non-interactive mode for scripting.
- **Reasoning effort** — let the model think harder on tough tasks via `reasoning_effort="low" | "medium" | "high"`.
- **Graduate to the Responses API / OpenAI Agents SDK** — the production agentic layer that runs the loop for you, with built-in tools (web search, code interpreter, file search). Now that you've built the loop by hand, you'll understand exactly what it's doing for you.

🧠 How the small core scales into the big systems: the same loop, just more tools, smarter context handling, and orchestration.

✅ **Done when:** you've extended the agent in at least one direction and understand how it generalizes.

---

## Suggested pace

| Sitting | Phases | Outcome |
|---|---|---|
| 1 | 0–2 | A chat that can read your files — your first real agent |
| 2 | 3–5 | It can explore, edit, and run code — a genuinely useful coding agent |
| 3 | 6–7 | Polished and safe to actually use |
| 4 | 8–9 | Production-flavored: cheap long sessions + an extension of your choice |

## How we'll work through each phase

For every phase I'll: **(1)** explain what we're adding and how it fits the big picture → **(2)** write the code with you, in small pieces → **(3)** run it and watch it work → **(4)** recap the concept before moving on. You'll always have a runnable agent at the end of each phase.

---

➡️ **Next action:** start **Phase 0**. Just say the word and we'll set up the project and make the first API call.

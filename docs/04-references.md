# 04 — References & Further Reading

The research that informed this plan. Roughly ordered from "read this first" to "deeper dives."

## Core tutorials (build-it-yourself)

> **Note:** the build-it-yourself tutorials below use Claude/Anthropic (and some use Go). That's fine — **the agent loop is identical with OpenAI**; only the SDK calls differ (`tool_calls` + `role:"tool"` messages instead of Anthropic's `tool_use`/`tool_result` blocks). Read them for the concepts and structure, not the exact API syntax — our OpenAI syntax is in [doc 02](02-architecture.md).

- **[How to Build an Agent — Thorsten Ball (ampcode.com)](https://ampcode.com/notes/how-to-build-an-agent)**
  The famous post that kicked off the "it's just a loop" realization. Builds a code-editing agent in ~300 lines of Go using exactly three tools: `read_file`, `list_files`, `edit_file`. Our Phase 2–4 structure is modeled on its incremental order. *Start here.*

- **[How Coding Agents Work — Simon Willison](https://simonwillison.net/guides/agentic-engineering-patterns/how-coding-agents-work/)**
  Concise, authoritative explainer of the agent loop, tools as harness-provided functions, the role of the system prompt, and why context (statelessness + caching) matters. Backs most of doc 01.

- **[Build a Coding Agent from Scratch: The Complete Python Tutorial — Sid Bharath](https://sidbharath.com/blog/build-a-coding-agent-python-tutorial/)**
  A Python walkthrough (our language). Frames an agent as brain + tools + instructions + memory.

- **[How to Build an Agent (Python port) — Janitha Rathnayake](https://medium.com/@jbrathnayake98/how-to-build-an-agent-by-thorsten-ball-python-version-ebbabb8665f6)**
  A direct Python translation of Thorsten Ball's Go original — handy side-by-side reference for Phase 2+.

## Concepts & patterns

- **[Building Effective AI Agents — Anthropic](https://www.anthropic.com/research/building-effective-agents)**
  The foundational guidance: workflows vs. agents, the five workflow patterns (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer), and the strong advice to *start simple and use the API directly* rather than reaching for frameworks. Backs doc 01 §8.

- **[Coding Agents Demystified — KDnuggets](https://ai-report.kdnuggets.com/p/coding-agents-demystified)**
  Higher-level tour of how production coding agents are structured.

## Ports in other languages (if you ever want to compare)

- **[How to Build an Agent in JavaScript — Kevin Yank](https://kevinyank.com/posts/how-to-build-an-agent-in-javascript/)**
- **[Build Your Own Agent (TypeScript) — Damian Demasi](https://www.damiandemasi.com/projects/build-your-own-agent)** — covers system prompts, parallel tool execution, and safety guardrails.

## Official API documentation (our toolbox)

We're using the **OpenAI API** directly. The relevant building blocks:

- **[Chat Completions](https://platform.openai.com/docs/api-reference/chat)** — `client.chat.completions.create(...)`: the endpoint everything goes through.
- **[Function calling / tools](https://platform.openai.com/docs/guides/function-calling)** — defining tools, the `tool_calls` / `role:"tool"` cycle, `finish_reason`. *This is the engine of the agent.*
- **Streaming** — `stream=True` for live token output (Phase 6).
- **[Prompt caching](https://platform.openai.com/docs/guides/prompt-caching)** — automatic; the cached prefix shows up as `usage.prompt_tokens_details.cached_tokens` (Phase 8). No code needed beyond keeping the prefix stable.
- **Reasoning effort** — `reasoning_effort="low"|"medium"|"high"` for harder tasks (Phase 9, optional).
- **[GPT-5.5 model card](https://developers.openai.com/api/docs/models/gpt-5.5)** — capabilities and the model id (`gpt-5.5`).

### The production layer we deliberately skip (for now)

We hand-write the loop to learn it. Once that clicks, these are the graduation path — both *run the loop for you*:

- **[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)** — a lightweight, production-ready agent runtime (auto schema generation, the loop, tool execution). The successor to Swarm.
- **[Responses API](https://platform.openai.com/docs/guides/migrate-to-responses)** — an agentic endpoint that can call multiple tools (including built-in web search, code interpreter, file search, and remote MCP servers) within one request.

> We'll cite the exact snippets from the OpenAI docs as we implement each phase.

## The throughline

Every source above repeats the same conclusion, and it's the thesis of this project:

> **There is no moat.** A capable coding agent is a small amount of code — an LLM, a loop, and a handful of tools. The sophistication lives in the model and in careful refinement (good tools, good prompts, good context management), not in hidden complexity.

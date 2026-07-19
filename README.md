# GA3BAD Coding Agent

A plan-first terminal coding agent designed to make smaller tool-calling models
far more reliable on long, complicated projects. The model supplies judgment and
code; the harness supplies durable state, mandatory plan approval, structured
subtasks, safety policy, recovery, and evidence-gated completion.

This began as a small “agent = model + loop + tools” learning project. It now
keeps that readable core while adding a deterministic control plane around it.

## What makes it different

- **The goal survives turns, compaction, Ctrl-C, and restarts.** It remains active
  until an explicit completion gate passes or the user explicitly cancels it.
- **Plans require user approval.** A fresh-context critic checks the draft first;
  approval is bound to the exact plan revision and fingerprint. That fingerprint
  includes inspected workspace facts, an executable strategy, expected edits, and
  the complete task DAG—not just explanatory prose.
- **The checklist is live and user-editable.** Add, edit, remove, reopen, block,
  skip, or complete tasks. Structural changes create a new revision that must be
  approved before work continues.
- **Roles are generated from the task.** There is no fixed “researcher/coder/tester”
  roster. The coordinator writes a narrow mission, constraints, deliverables, and
  tool policy for each delegated worker; workers can recursively propose focused
  children within configurable safety limits.
- **Prose is never “done.”** The model must update task evidence and call
  `finish_goal`. The harness then checks every task and uncertain action before a
  separate reviewer can pass or create repair tasks.
- **Small-model guardrails are structural.** Typed control calls, narrow worker
  contexts, retry/backoff, repeated-action circuit breakers, automatic reprompts,
  independent plan/final critique, and deterministic verification gates do not
  depend on the model remembering instructions.
- **Goal retries are unbounded.** `/auto` keeps making durable attempts until the
  completion gate passes or real user input/approval is required. Each failed
  attempt injects a self-reflection prompt, changes the required approach, and
  uses bounded exponential backoff; Ctrl-C still checkpoints immediately.
- **Workspace security is enforced in code.** Canonical path containment,
  sensitive-file denial, atomic edits, strict schemas, bounded output, explicit
  shell approval, a scrubbed child environment, and crash-window journaling are
  harness invariants—not prompt suggestions.

No harness can literally give a small model knowledge or reasoning it does not
have. This design raises its effective reliability by decomposing work, narrowing
context, preserving facts, demanding evidence, and making unsafe/invalid state
transitions impossible.

## Readiness architecture

The harness now has executable answers for the large-repository and weak-model
failure modes that prompts alone cannot solve:

- `RepositoryIndex` uses Python AST plus HTML DOM extraction, overlapping chunks,
  dependency/resolved-call/ownership graphs, and bounded semantic maps. Retrieval
  combines lexical, graph, sparse-semantic, and optional Ollama embeddings
  (`AGENT_EMBEDDING_MODEL=nomic-embed-text:latest`) with an offline hashing fallback.
- Critical ULTRA phases require auditable decision, counterargument, evidence,
  rejected-alternative, verification, and reasoning-graph artifacts. The harness
  scores these external summaries; it never attempts to expose or persist hidden
  model chain-of-thought.
- Swarm work uses versioned messages, routing, leases, proposal/vote/quorum
  consensus, decision publication, and leader metadata. Frames support canonical
  JSON plus bounded DSL and binary encodings.
- Project lessons persist across runs with evidence references, reuse counts,
  confidence history, and asymmetric outcome updates. Failed evaluation writes
  remediation knowledge for later runs.
- Retrieval, reasoning, swarm, learning, runtime, and interactive output have
  deterministic benchmarks. Screenshot statistics are anomaly checks only;
  visual acceptance requires an independent vision model, two clean verdicts,
  and a blind pairwise preference. The builder cannot judge its own output.

Run the structural and behavioral audit, persist its metrics, and compare the next
run against it:

```powershell
.venv\Scripts\python -m agent --workspace D:\path\to\project --command "/doctor --record"
```

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env for OpenAI/Gemini, or leave Ollama selected.

.venv\Scripts\python -m agent --workspace D:\path\to\your-project
```

On macOS/Linux, use `.venv/bin/python` and `cp .env.example .env`.

You can also launch the workspace chooser:

```bash
python -m agent
```

The interactive terminal opens on a full-screen `GA3BAD` welcome. Press Enter,
then use the arrow keys and Enter to choose one workspace, model, access level,
and interaction mode at a time. The focused row always includes its description;
Esc moves back without changing the current selection. Model discovery and
Docker checks use a semantic inline-square activity indicator rather than blocking on
an unexplained blank screen.

At the prompt, `/` opens a contextual, nested command palette instead of dumping
every command. F2, F3, and F4 reopen Mode, Model, and Access; Ctrl+Q exits safely.
Direct commands and legacy `:command` syntax remain available for scripts and
power users. Use `--plain` for the line-oriented/SSH UI or `--reduced-motion` for
static, accessible motion. Ollama models are discovered first through `/api/tags`
and `/api/show`; non-tool models are omitted. Local models use one sequential
worker, while Ollama cloud/OpenAI/Gemini models run independent, write-disjoint
nodes concurrently.

## ULTRA mode

`/mode ultra` turns a compact request into `GoalSpecV1`, architecture, and an
approval-bound master plan. After one `/approve`, the background scheduler runs
the same pipeline for local and cloud models:

```text
context → mini-plan → decompose → research → implement
        → independent review → tests → bounded fixes
        → integration → Project Brain write-back
```

Dynamic child nodes inherit their parent's forbidden changes and write scope.
New interfaces, dependencies, or out-of-scope paths stop at a new master-plan
approval. SQLite schema v9 stores prompt completeness, intake briefs/questions,
AST and graph metadata, the hierarchical specialist task graph, staged component
files, materialized package revisions, interface contracts, package-consumption
evidence, independent visual verdicts/pairwise comparisons, typed messages,
versioned decisions and lessons, evaluation runs, redacted prompt traces, memory
access, and fenced path/leader leases. Component specialists stage one real file
at a time and publish a manifest; only FinalAssembler owns final output paths.
`/tree`, `/agents`, `/memory`,
`/trace`, `/insights`, and `/metrics` keep the default scrollback uncluttered.

`/permissions full` is fail-closed: it works only after `/setup` builds the
versioned non-root Docker image. The workspace is the only writable bind mount;
the host home, Docker socket, and credentials are never mounted or injected.

The original command remains supported:

```bash
python agent/main.py --workspace /path/to/project
```

## Normal workflow

1. Every message enters `Intake/Planning` through Intent Architect. It inspects
   discoverable repository context, creates a canonical execution brief, and asks
   only consequential missing product decisions. Each question has exactly three
   suggestions (the first Recommended) plus a free-form fourth answer.
2. Normal is the default durable goal workflow. Ultra is selected automatically
   at complexity `>= 0.65` or for hard triggers such as multi-component systems,
   high-risk changes, and high-quality visual/interactive work. The read-only planner then inspects
   the repository and submits a
   structured plan containing factual applicability evidence, an execution
   strategy, expected workspace changes, and task-bound verification.
3. Deterministic validation checks every plan; a separate critic is used only
   for complex or high-risk work.
4. Review the sparse status and full `/plan`. Use `/edit`, `/add`, or `/remove`,
   then `/approve`.
5. Use `/mode normal` for the cohesive durable workflow or `/mode ultra` for
   recursive specialists, architecture debate, component isolation, and consensus.
   Legacy `chat|plan|goal` values map to Normal for compatibility. Ctrl-C checkpoints
   automatic work safely.
6. Add guidance or edit the checklist at any checkpoint. The durable objective is
   always re-injected, even after context compaction or restart.
7. Completion requires all accepted tasks, direct evidence, no uncertain action,
   and a passing independent review.

Example control surface:

```text
Understanding goal
Architecture ready · 8 modules
[Physics · coder] editing motor.py
[reviewer] found 2 issues
Fix loop 2/3
[Physics · tester] updated
GA3BAD [ULTRA]>
```

The logo appears once. Normal output is append-only scrollback; detailed trees,
agents, memory, traces, and metrics appear only when requested.

Live activity is intentionally summarized: a tool call and its result resolve as
one operation, repeated read-only inspections are coalesced, usage counters and
recoverable schema details stay folded, and a plan is announced only after the
independent critic accepts it. Provider thoughts drive a compact single-line
square loader whose gray-to-white motion changes by activity state and whose
label reflects the current thought or tool, then collapse at the end of each model step; `/thinking`
opens the redacted, session-only blocks again. Use `/trace`, `/history`, or
`/metrics` when durable technical detail is needed.

## Commands

| Command | Effect |
|---|---|
| `/` | Open the interactive slash-command palette |
| `/mode normal`, `/mode ultra` | Select the durable Normal workflow or recursive specialist Ultra; legacy `chat|plan|goal` aliases map to Normal |
| `/settings [NAME [VALUE]]` | Inspect safe session settings or change color/runtime limits; secrets are never displayed |
| `/model [NAME]` | Reopen the picker or switch models at a safe checkpoint |
| `/permissions normal\|full`, `/setup` | Select approvals or initialize the fail-closed Docker Full sandbox |
| `/skills` | Show the real local tool registry, availability, risk, and approval policy |
| `/processes`, `/stop-process ID` | Inspect or stop agent-owned processes and HTML previews |
| `/sleep on\|off\|status` | Control the session-scoped Sleep profile; requires Ultra and ready Full Docker access |
| `/tree [NODE]`, `/agents [--all]` | Inspect hierarchical work and isolated agent runs |
| `/memory [SECTION]`, `/trace [latest\|RUN_ID]` | Inspect Project Brain and redacted prompts/context/summaries |
| `/thinking` | Expand redacted provider thoughts captured during this session |
| `/insights [NODE]`, `/metrics` | Inspect durable findings and execution metrics |
| `/questions`, `/answer ID VALUE` | Advanced fallback for intake/planning questions; plain text, `1/2/3`, and `4 custom text` work directly |
| plain text / `/goal TEXT` | Enter the same Intent Architect gate when idle; otherwise add durable user guidance |
| `/plan`, `/status` | Render the complete plan or a compact scrollback status |
| `/approve [REV]` | Approve the exact latest plan revision |
| `/reject FEEDBACK`, `/replan FEEDBACK` | Reject and regenerate with feedback |
| `/add TEXT :: CRITERIA` | Add a checklist item as a new plan revision |
| `/edit ID [FIELD] VALUE`, `/remove ID` | Edit the whole task or `title`, `description`, `accept`, `verify`, `depends`, `risk`; dependants are invalidated. Separate multiple criteria with `||` |
| `/done ID NOTE` | Complete a task with user-supplied evidence |
| `/todo ID`, `/block ID NOTE`, `/skip ID NOTE` | Reopen or change task state |
| `/run [STEPS]` | Run one bounded work slice; the goal itself has no slice deadline |
| `/auto` | Retry and self-prompt without an attempt limit until verified completion or real input/approval |
| `/pause`, `/resume` | Cooperatively checkpoint and continue |
| `/history` | Show durable events and generated worker roles/results |
| `/resolve ENTITY_ID applied\|not-run NOTE` | Reconcile an uncertain crash-window action or worker after inspecting real workspace state |
| `/cancel CANCEL` | Explicitly abandon an unfinished goal |
| `/quit` / `exit` | Exit without losing the goal |

All slash commands also accept the legacy `:` prefix.

For scripting/non-interactive inspection:

```bash
python -m agent --workspace ./project --command "/status"
python -m agent --workspace ./project --provider ollama --model gemma4:e4b --mode normal --command "/approve 2"
```

## Lifecycle and completion authority

```text
 NEW -> DISCOVERING -> AWAITING_PLAN_APPROVAL -> RUNNING
                         ^                       |   |
                         |    plan revision -----+   +-> PAUSED -> RUNNING
                         |                               |
                         +---- failed review <- REVIEWING <- VERIFYING
                                                   |
                                                   +-> COMPLETED

 crash during work -> RECOVERING -> PAUSED (uncertain actions are never replayed)
 explicit user only ------------------------------------------------> CANCELLED
```

The provider never writes a goal status directly. It requests transitions through
typed control tools; `AgentRuntime`, the task DAG, and `StateStore` validate them.
Ordinary Chat uses the same workspace binding, permission adapter, action journal,
and typed tool results; it cannot bypass Full Docker routing or count a failed
write as a mutation.

## Adaptive subtasking

`delegate_task` contains:

- a role/mission synthesized for the exact work;
- explicit success criteria and narrow context;
- an allowlist of worker tools;
- a fresh conversation so unrelated history cannot distract it;
- a structured `return_work` result with evidence, changed paths, risks, and
  proposed children.

Delegation depth, steps, and per-slice count are configurable safety bounds, not
a fixed role count. The root coordinator remains the only component that can
update the accepted checklist or request root completion.

## Persistence and recovery

Each workspace gets `.coding-agent/state.db`. In a normal Git repository the
harness adds `/.coding-agent/` to the untracked local `.git/info/exclude` (never
the tracked `.gitignore`); linked worktrees may need the same local exclude added
manually. SQLite WAL and
transactions store goals, plan revisions/fingerprints, applicability evidence,
expected edits, execution strategies, tasks/DAGs, approvals, evidence,
delegations, retry attempts, action intents/results, and an append-only event journal.
Planner inspections receive stable harness references such as `inspection:I001`;
the harness records and reuses them instead of requiring provider-native call IDs.

Before a tool runs, its intent is recorded. If the process stops after a side
effect but before its result is journaled, the next launch marks the action and
in-flight task `uncertain`, pauses the goal, and asks for inspection. It never
blindly retries an uncertain write or shell command.

Conversation text is deliberately not the source of truth. Long histories can be
compacted while the objective, accepted plan, evidence, memories, and approvals
remain exact in SQLite. Every coordinator call gets a bounded authoritative view;
`inspect_task` provides exact paginated task/evidence retrieval when the full
history would be wasteful.

## Security model

- All file paths are canonicalized and must remain under the configured workspace,
  including symlink targets and Windows-specific aliases/devices.
- `.coding-agent`, `.env` variants, credentials, private keys, cloud auth paths,
  and detected private-key content are hidden from model-readable tools.
  `.env.example` remains readable.
- Tool arguments are strictly validated before approval or execution; unknown or
  mistyped fields never reach implementations.
- Reads, traversal, grep, regex complexity, writes, commands, and captured output
  have deterministic caps.
- Writes use temp files, `fsync`, identity checks, and atomic replacement; a failed
  edit preserves the original.
- Every shell command requires a separate user approval. Shell children use the
  workspace as explicit `cwd` and inherit only an operational environment
  allowlist—never API keys or arbitrary secrets.
- Plan approval does not imply action approval. Tool output is redacted before it
  is sent back to a provider or written to durable events.

This is a strong local harness boundary, not an OS sandbox. An approved shell
command still has the operating-system permissions of the user who launched it.

## Providers

Set `LLM_PROVIDER=openai`, `gemini`, or `ollama` in `.env`, or pass `--provider`.
Use `--model` for a one-run override. Adapters normalize streaming, tool calls,
usage, IDs, malformed arguments, and provider-native replay metadata. Gemini
thought signatures/function IDs and Ollama thinking/tool names are retained.

## Runtime tuning

All limits apply to one recoverable slice, not the lifetime of the goal:

| Variable | Default | Purpose |
|---|---:|---|
| `AGENT_PLANNING_STEPS` | 16 | Planner tool/model steps |
| `AGENT_WORK_QUANTUM` | 24 | Coordinator steps before a user checkpoint |
| `AGENT_REVIEW_STEPS` | 12 | Plan/final reviewer steps |
| `AGENT_SUBAGENT_STEPS` | 16 | Steps per focused worker |
| `AGENT_MAX_DELEGATION_DEPTH` | 4 | Recursive delegation safety bound |
| `AGENT_MAX_DELEGATIONS_PER_SLICE` | 12 | Worker fan-out bound |
| `AGENT_PROVIDER_RETRIES` | 3 | Transient provider retries |
| `AGENT_REPEAT_LIMIT` | 2 | Identical-action no-progress circuit breaker |
| `AGENT_NO_ACTION_LIMIT` | 3 | Prose-only reprompt limit |
| `AGENT_STALLED_SLICE_LIMIT` | 3 | No-progress attempts between stronger decomposition/escalation prompts |
| `AGENT_CONTEXT_CHARS` | 120000 | Conversation compaction threshold |
| `AGENT_GOAL_RETRY_BASE_MS` | 1000 | Initial backoff between unbounded goal attempts |
| `AGENT_GOAL_RETRY_MAX_MS` | 30000 | Maximum per-attempt backoff; retry count remains unlimited |

## Architecture

See [docs/02-architecture.md](docs/02-architecture.md) for the component and data
flow, and [docs/03-roadmap.md](docs/03-roadmap.md) for implemented phases and
remaining production extensions.

Key modules:

- `runtime.py` — deterministic lifecycle, planning, execution, delegation, review
- `models.py` / `store.py` — typed domain state and transactional SQLite journal
- `control.py` / `prompts.py` — validated control protocol and stable cached prompts
- `tools/` — central registry for contained file/search/edit/patch/shell,
  dependency, managed-process, and secure browser-preview capabilities
- `providers/` — OpenAI, Gemini, and Ollama adapters
- `events.py` / `ui.py` / `commands.py` / `cli.py` — event-driven ASCII interface
- `testing.py` — deterministic offline provider for lifecycle tests

Weak-model specialization is implemented in `weak_model.py`, `run_context.py`,
`convergence.py`, `diagnostics.py`, `repository_index.py`, and
`local_provider.py`. The implementation report and live evidence are in
[docs/07-local-model-quality-convergence-evidence.md](docs/07-local-model-quality-convergence-evidence.md).

## Verification

The suite is network-free and uses the standard library runner:

```bash
python -m unittest discover -s tests -v
```

It covers inspected/applicable plan approval and fingerprint staleness, editable
revisions, unbounded self-retry recovery, false completion, dynamic worker
isolation, failed reviews and repair plans, crash recovery, provider replay
formats, malformed tool calls, path/symlink/secret attacks, atomic-write failure,
shell environment/cwd/output bounds, conversation pairing, and ASCII snapshots.

The suite also covers active Goal Contract projections, policy persistence,
quality-gated completion, fresh artifact hashes, delta refinement, mixed HTML
indexing, provider capability fallback, and truthful visual-review boundaries.

## License

MIT — see [LICENSE](LICENSE).

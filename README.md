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
- **Goal work persists without hiding broken providers.** `/auto` keeps making
  durable, approach-changing attempts until completion or a real input/approval
  boundary. Repeated provider-access failures stop after a configurable limit and
  show an actionable model/network checkpoint instead of retrying forever.
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

Open `/settings`, choose **Diagnostics**, then run the structural and behavioral
audit. Recorded results remain available to Advanced Tracing for comparison.

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

The interactive terminal opens on the original animated full-screen `GA3BAD` welcome. Press
Enter, then review the five explicit setup decisions: workspace, project protection,
model, permissions, and workflow mode. Back returns to the previous decision. Ollama
models are probed during discovery; configured cloud credentials are clearly labeled
as unverified until their first request.

After setup, there is one calm, persistent workspace rather than a sequence of
screens. It always shows the conversation, a fixed project-progress area, one
responsive telemetry row, and a composer you can type in. Progress is derived
from durable tasks: phase, current task/operation, completed and remaining counts,
retries or blockers, elapsed time, and an explicitly approximate ETA range. ETA
starts as `learning`, pauses while input is required, and is never shown when the
total is unknown. CPU and RAM are sampled asynchronously; GPU is shown only when
a bounded NVIDIA/ROCm probe succeeds. Context remaining is numeric only when the
provider exposes a capacity, otherwise it remains `?`. `F2` switches between
**Simple** (the default) and **Advanced**.
Simple hides tool calls, JSON, coordinator steps, and stack traces. Advanced keeps
the coalesced diagnostic transcript without interrupting the work. Errors,
warnings, blockers, file-change receipts, test results, and required decisions are
kept visible in both modes.

Questions, plan reviews, errors, and permissions appear in one small area directly
above the composer. Use arrows, number keys, shortcut letters, the mouse, or type a
custom answer. One option may be marked **Recommended**, while **Enter default**
identifies the separate safe default. Escape goes back or chooses the declared safe
cancel option; invalid input leaves the decision open with inline guidance.
Ordinary project edits and isolated checks continue automatically;
dependency installs, network/host actions, deletes, secrets, and out-of-project
work stay visibly paused until an explicit choice. Closing a prompt never becomes a
silent denial.

While work runs, type guidance and press Enter; it is saved for the next safe point.
After 60 quiet seconds the compact activity strip identifies whether a model call
or worker is still open and offers **Keep waiting** or **Stop safely** without
blocking completion. Typing `/` or pressing `Ctrl+K` opens the same filterable
command palette; Up/Down selects, Tab completes, Enter runs, and Escape closes it.
With an empty composer, `?` opens context-sensitive keyboard help; inside a
message or plan document it remains ordinary text. The footer shows only the
four or five controls actionable in the current view. Below 80x24 the workspace
shows a resize gate while background work and telemetry continue safely.
`/live` remains available for calm progress while `/advanced-tracing` opens the
complete developer trace only when requested. `F3` opens model
selection and `F4` opens permissions. `Ctrl+Q` exits only at a saved checkpoint;
while work is active it requests a checkpoint and keeps the session open. Use
`--plain` to skip the animated intro and use the line-oriented SSH/screen-reader
UI, or `--reduced-motion` for lower-motion setup and progress surfaces.

`F6` toggles **Safe Auto** at any time. The full Sleep policy is configured under
Runtime in `/settings`; explicit Full Auto can accept tool approvals in the
selected workspace and every automatic approval is audited. In eligible
Execution/Full-Docker sessions, enabling Sleep also attempts to arm the deeper
recursive Sleep profile; failure of that stricter gate does not disable Safe Auto.

`F7` opens Change Review for the current immutable checkpoint and `F8` opens the exact
selected project workspace in the operating-system file explorer. The fixed
footer changes by execution class: local models prioritize CPU/GPU/RAM, while
cloud models prioritize provider/model activity, input/output/cached tokens,
known context remaining, and only provider limits actually reported by the
provider. Unknown quotas and limits are shown as unavailable, never estimated.

## Ultra Plan and durable conversation

The two explicit interaction choices are **Execution** and **Ultra Plan**.
Execution routes each message automatically: conversation becomes evidence-aware
Chat, while an implementation request enters the recursive execution engine.
Ultra Plan adds a user-selected planning boundary before mutation. Selecting it
or entering `/ultra-plan` opens the planning workspace. The browser asks as many
material questions as needed, presents one editable plan document, then uses two
explicit confirmations: **Prepare agents** shows the exact first-layer team, and
**Approve & start** hands that revision back to the terminal. Execution never
depends on the browser remaining open.

The web surface has only two pages. Plan owns the pre-execution conversation and
approval. Live is a read-only projection of current work, the agent tree, durable
timeline, and recorded diffs. The terminal/runtime remains the control plane and
source of truth; the browser exposes only small blocker actions such as retry,
change model, allow/deny, and stop safely.

The persistent terminal keeps the durable workspace conversation in its main
surface. Failed partial model streams and raw reasoning are never promoted into
durable chat.

### Run, inspect, and deliver visual output

“Run this project” is treated as an operational request: the agent inspects the
README, manifests, lockfiles, and declared scripts; installs only declared
project dependencies; starts long-running processes under lifecycle management;
checks readiness/logs; and repairs an in-scope startup blocker before retrying.

Image delivery is platform-neutral. Browser scenarios save distinct screenshots
under the project `output/browser/` tree. Exact SHA-256 plus a perceptual hash
prevent identical and near-identical page states from satisfying a request for
distinct images, including screenshots restored from an older session.
`inspect_images` sends the exact current bytes to the selected model, first
requiring it to pass a pixel-only Vision probe. A conclusive probe failure is
cached and pauses the task at the visual-evidence gate after one attempt; it is
never converted into a successful Output. A capable model then scores, ranks,
and cites visible facts for each image. `publish_output` accepts only current
Vision-backed images and builds the generic Output page with independently
copyable sections and attachments. No social-platform mode, template, or
destination preference is inferred or persisted.

`/pause` records a cooperative pause request and stops new scheduling. The UI
shows `Pause pending` while the current model/tool/agent drains, and changes the
goal to PAUSED only after a saved checkpoint. `/resume` appears
only after that durable state exists. An API call is not presented as a
server-side job and does not claim to keep running after the computer shuts down.

## Execution

Execution turns a compact request into `GoalSpecV1`, architecture, and an
approval-bound master plan. After the master plan is applied in Ultra Plan, the background scheduler runs
the same pipeline for local and cloud models:

```text
context → mini-plan → decompose → research → implement
        → independent review → tests → bounded fixes
        → integration → Project Brain write-back
```

Dynamic child nodes inherit their parent's forbidden changes and write scope.
New interfaces, dependencies, or out-of-scope paths stop at a new master-plan
approval. SQLite schema v13 stores prompt completeness, intake briefs/questions,
AST and graph metadata, the hierarchical specialist task graph, staged component
files, materialized package revisions, interface contracts, package-consumption
evidence, independent visual verdicts/pairwise comparisons, typed messages,
versioned decisions and lessons, evaluation runs, redacted prompt traces, memory
access, and fenced path/leader leases. Component specialists stage one real file
at a time and publish a manifest; only FinalAssembler owns final output paths.

Before execution, the recursive engine derives a typed concern-coverage matrix from the task
family and repository evidence, not from prompt length. Critical concerns receive
named owners and executable acceptance checks: spatial semantics, progression,
world continuity, and frame performance for games; security, data integrity,
concurrency, recovery, and operability for backends; async states, accessibility,
client security, and rendering performance for frontends; and leakage,
reproducibility, evaluation validity, and serving reliability for ML. Existing
master modules receive those contracts directly; a new specialist swarm is
created only when the approved plan has not already divided the responsibility.

`/live` and `/advanced-tracing` keep the default scrollback uncluttered while
making concise progress or complete developer evidence available on demand.

During a long Execution run, the main body keeps a two-line activity strip with the
human-readable phase, active operation, completed/remaining nodes, elapsed time,
and an approximate ETA only after enough work has been observed. Blockers add one
temporary third line. CPU, GPU, RAM, context, model activity, and Sleep state stay
in the fixed footer rather than interrupting the transcript.

`/live` opens the read-only Execution workspace without starting a nested
terminal application.
Every materialized specialist receives a short path-derived name and a read-only
workspace showing its status, role/phase, capabilities, owned concerns and
interfaces, assignment, deliverable, and latest redacted prompt. Up/Down switches
specialists without affecting execution; Tab or Left/Right switches between the
agent workspace and a status-aware hierarchy map. The view refreshes from durable
SQLite state while the local model continues running. Advanced Tracing exposes
redacted reasoning summaries, prompt records, context decisions, and diagnostics;
raw chain-of-thought is never stored or shown.
Plain/redirected terminals retain
the compact text fallback.

Full access is selectable through `/access full` or Settings. It is a
session-wide unattended approval policy for the already selected workspace and
accepted task, so browser, process, and file tools keep one consistent host
environment. The optional versioned non-root Docker sandbox remains available
for explicitly isolated command work; it is not a prerequisite for Full access.

The original command remains supported:

```bash
python agent/main.py --workspace /path/to/project
```

## Automatic Chat and Execution workflow

1. Every message enters `Intake/Planning` through Intent Architect. It inspects
   discoverable repository context, creates a canonical execution brief, and asks
   only consequential missing product decisions. Each question has exactly three
   suggestions (the first Recommended) plus a free-form fourth answer.
2. Execution is the default. Chat is not a selectable mode: the semantic router
   chooses it per message for conversation or project-aware explanation. A real
   implementation request enters the durable Goal workflow and the harness selects
   staged or recursive depth from task demand and model capability. The read-only planner then inspects
   the repository and submits a
   structured plan containing factual applicability evidence, an execution
   strategy, expected workspace changes, and task-bound verification.
3. Deterministic validation checks every plan; a separate critic is used only
   for complex or high-risk work.
4. Open `/ultra-plan`, edit or revise the plan, then use the explicit Approve & Start action.
5. Routing and recursive depth are selected from task complexity and model
   capability. Use `/ultra-plan` only when you explicitly want the editable planning
   boundary before mutation. Changes requested during work are applied only after
   the nearest saved checkpoint; Ctrl-C requests that checkpoint cooperatively.
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
GA3BAD [EXECUTION]>
```

The logo appears once. Execution output is append-only scrollback; detailed trees,
agents, memory, traces, and metrics appear only when requested.

Live activity is intentionally summarized: a tool call and its result resolve as
one operation, repeated read-only inspections are coalesced, usage counters and
recoverable schema details stay folded, and a plan is announced only after the
independent critic accepts it. Provider thoughts drive a compact single-line
square loader whose gray-to-white motion changes by activity state and whose
label reflects the current thought or tool, then collapses at the end of each
model step. Use `/advanced-tracing` when durable technical detail is needed.

## Commands

| Command | Effect |
|---|---|
| `/ultra-plan` | Open Ultra Plan and its editable document / first-layer approval flow |
| `/live` | Open the simple read-only progress workspace |
| `/todo` | Inspect the goal, tasks, criteria, and verification evidence |
| `/show-diff` | Open the standalone read-only workflow diff with live and recorded changes |
| `/advanced-tracing` | Open the standalone developer trace for the current or a prior run |
| `/settings` | Open Runtime, Providers, Project, Terminal, and Diagnostics settings |
| `/access [normal\|full]` | Inspect or change the separate permission profile; Full runs accepted in-scope actions without repeated questions |
| `/pause` | Request a cooperative saved checkpoint |
| `/resume` | Continue from the durable checkpoint |
| `/stop` | Stop now while keeping the saved stage resumable |
| `/undo [STEPS]` | With explicit approval, safely revert accepted checkpoints while preserving the undo in Git history; blocked during active work or with dirty files |
| `/help` | Show the public command surface and the active key bindings |
| `/quit` | Checkpoint and leave the session |

There are no legacy aliases or `:` command prefix. Approval, retry, questions,
model recovery, Explorer, and managed-process actions appear as typed TUI
controls or inside Settings.

For scripting/non-interactive inspection:

```bash
python -m agent --workspace ./project --command "/help"
python -m agent --workspace ./project --provider ollama --model gemma4:e4b --command "Build and verify the requested artifact"
```

To start a genuinely empty workflow thread while reusing the selected project
and its protection settings, add `--new-session`.  This is different from
`--session`, which deliberately restores the saved prompt, goal, model, and
timeline of an existing thread:

```powershell
python -m agent --workspace ./project --new-session --provider ollama --model gemma4:e4b --command "Build the requested artifact and verify it" --auto
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
- Read/write work inside the selected project is automatic. In Normal access,
  dependency installs, network, destructive, preview, and host actions require
  one visible session-wide decision; Full access applies the user's explicit
  unattended grant without further questions. Shell children use the workspace as explicit `cwd`
  and inherit only an operational environment
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
The TUI can add or replace a masked key under Providers in `/settings`; choose
Runtime / Model there to select or reconnect at a safe checkpoint.

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
| `AGENT_PROVIDER_RETRIES` | 3 | Transport retries inside one provider call |
| `AGENT_PROVIDER_FAILURE_LIMIT` | 3 | Consecutive failed provider cycles before an actionable pause |
| `AGENT_REPEAT_LIMIT` | 2 | Identical-action no-progress circuit breaker |
| `AGENT_NO_ACTION_LIMIT` | 3 | Prose-only reprompt limit |
| `AGENT_STALLED_SLICE_LIMIT` | 3 | No-progress attempts between stronger decomposition/escalation prompts |
| `AGENT_CONTEXT_CHARS` | 120000 | Conversation compaction threshold |
| `AGENT_GOAL_RETRY_BASE_MS` | 1000 | Initial backoff between durable attempts |
| `AGENT_GOAL_RETRY_MAX_MS` | 30000 | Maximum per-attempt backoff |

## Architecture

See [docs/02-architecture.md](docs/02-architecture.md) for the component and data
flow, and [docs/03-roadmap.md](docs/03-roadmap.md) for implemented phases and
remaining production extensions. The workflow-focused defect matrix, estimated
impact, remaining hardening work, and release gates live in
[docs/09-production-ux-audit.md](docs/09-production-ux-audit.md).

Key modules:

- `runtime.py` — deterministic lifecycle, planning, execution, delegation, review
- `ui_state.py` — one normalized question session across Intake, Normal, and ULTRA
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
revisions, durable self-retry recovery, provider-failure pause boundaries, false completion, dynamic worker
isolation, failed reviews and repair plans, crash recovery, provider replay
formats, malformed tool calls, path/symlink/secret attacks, atomic-write failure,
shell environment/cwd/output bounds, conversation pairing, and ASCII snapshots.

The suite also covers active Goal Contract projections, policy persistence,
quality-gated completion, fresh artifact hashes, delta refinement, mixed HTML
indexing, provider capability fallback, and truthful visual-review boundaries.

## License

MIT — see [LICENSE](LICENSE).

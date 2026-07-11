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

The interactive terminal opens with a green `GA3BAD` ASCII banner. Type `/` to
open the live command palette; arrow keys and Tab select completions as you type.
Legacy `:command` syntax remains accepted for existing scripts.

The original command remains supported:

```bash
python agent/main.py --workspace /path/to/project
```

## Normal workflow

1. Enter a goal in plain language.
2. The read-only planner must successfully inspect the repository and submit a
   structured plan containing factual applicability evidence, an execution
   strategy, expected workspace changes, and task-bound verification.
3. A separate plan critic checks coverage and verifiability.
4. Review the dashboard. Use `/edit`, `/add`, or `/remove`, then `/approve`.
5. Choose `/mode plan` to wait for a manual `/run` after approval, or `/mode goal`
   to continue automatically after that explicit approval. `/auto` is also
   available directly. Ctrl-C checkpoints automatic work safely.
6. Add guidance or edit the checklist at any checkpoint. The durable objective is
   always re-injected, even after context compaction or restart.
7. Completion requires all accepted tasks, direct evidence, no uncertain action,
   and a passing independent review.

Example control surface:

```text
+ GA3BAD CODING AGENT -------------------------------------------------------+
| MODE GOAL | STATUS RUNNING | PLAN r3 / r3 | [########------] 5/9           |
| GOAL Implement the service, migration, tests, security review, and docs     |
+-----------------------------------------------------------------------------+
| CHECKLIST                              | ACTIVE WORKERS / DYNAMIC ROLES      |
| [x] T001 Map current architecture      | 8ac19f RUN T005 - schema migrator   |
| [>] T004 Add durable job state         |                                     |
| [ ] T005 Migrate existing records      |                                     |
| [!] T006 Resolve upstream API choice   |                                     |
+-----------------------------------------------------------------------------+
| RECENT ACTIVITY                                                            |
| task.status_changed: T004 -> in_progress                                    |
| delegation.created: crash-safe SQLite migration specialist                  |
+-----------------------------------------------------------------------------+
| / /mode /approve /run [steps] /plan /settings /status /help /quit          |
+-----------------------------------------------------------------------------+
```

All framework labels and borders are ASCII-only for clean Windows/SSH output.

## Commands

| Command | Effect |
|---|---|
| `/` | Open the interactive slash-command palette |
| `/mode plan`, `/mode goal` | Switch between manual post-approval execution and automatic post-approval execution; neither mode bypasses plan approval |
| `/settings [NAME [VALUE]]` | Inspect safe session settings or change color/runtime limits; secrets are never displayed |
| `/model [NAME]` | Show or switch the provider model for this session |
| plain text / `/goal TEXT` | Start a goal when idle; otherwise add durable user guidance |
| `/plan`, `/status` | Render current objective, revision, checklist, workers, and progress |
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
python -m agent --workspace ./project --mode goal --command "/approve 2"
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
- `tools/` — contained file/search/edit/shell capabilities
- `providers/` — OpenAI, Gemini, and Ollama adapters
- `events.py` / `ui.py` / `commands.py` / `cli.py` — event-driven ASCII interface
- `testing.py` — deterministic offline provider for lifecycle tests

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

## License

MIT — see [LICENSE](LICENSE).

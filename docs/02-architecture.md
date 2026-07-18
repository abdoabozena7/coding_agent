# Architecture: deterministic harness around probabilistic workers

The system intentionally separates **judgment** from **authority**. A model can
suggest the next action; only the harness can approve a plan revision, mutate
durable task state, run a policy-checked tool, or complete the root goal.

## Component map

```text
                         +-------------------------+
 user commands --------> | cli.py / commands.py    |
                         +------------+------------+
                                      |
                                      v
 +------------+ events       +--------+---------+      transactions +-------------+
 | ui.py ASCII| <----------- | runtime.py       | <---------------> | store.py    |
 +------------+              | lifecycle owner  |                   | SQLite WAL  |
                             +---+----------+----+                   +------+
                                 |          |                               |
                    model calls  |          | validated actions             | typed
                                 v          v                               v
                         +-------+--+   +---+----------------+       +------+------+
                         | providers |   | control.py/tools/ |       | models.py   |
                         +----------+   +--------------------+       +-------------+
```

No provider SDK, interactive prompt implementation, terminal printing, or
process-global `chdir` exists in
the runtime core. The provider, store, approval callback, event bus, sleep policy,
and runtime limits are injected, which makes the full lifecycle testable offline.

The CLI exposes only `Normal` and `Ultra`. Both enter the same durable
`Intake/Planning` gate and both require one explicit approval of the exact plan
fingerprint before the first mutation. Normal then runs the persistent
plan/execute/review/repair loop; Ultra adds recursive specialists, architecture
debate, component packages, and consensus. Legacy `chat|plan|goal` values are
migration aliases for Normal, not separate quality levels. This session policy
never auto-approves a plan and never bypasses shell/tool approval or
crash-recovery boundaries. `prompt-toolkit` adds the live `/` completion palette;
the plain-input fallback still accepts every slash command and opens the palette
when `/` is submitted.

## Durable domain

`models.py` defines goals, plans, tasks, role profiles, evidence, delegations, and
events. Transition graphs reject impossible jumps. Plans contain a validated DAG:
task IDs are unique, dependencies and parents must exist, and cycles are rejected
before a transaction commits.

`store.py` materializes current state and keeps an append-only event journal:

```text
goals -> plans(revision,fingerprint,status) -> tasks(DAG,status,criteria,role)
      -> approvals(exact revision + fingerprint)
      -> evidence(task/review/memory/guidance)
      -> delegations(parent,role,brief,result)
      -> actions(intent,args hash,risk,mutating,status,result)
      -> events(sequence,type,entity,payload)
```

There can be at most one unfinished goal per workspace. Terminal goals remain in
history. A new structural checklist change creates a new plan revision; previous
accepted work can be copied with evidence, while changed tasks and dependants are
reset. The new revision pauses execution until approval.

## Request flow

### 1. Planning

The planner receives the objective and read-only tools. It must successfully use
at least one workspace inspection tool before `propose_plan`. The structured
proposal includes typed tasks, criteria, verification, dependencies, risk,
task-bound factual applicability evidence, an executable strategy, and expected
workspace changes. All of those fields participate in the approval fingerprint.
A fresh plan critic calls `submit_plan_review`. Only a critic-passed plan is
persisted as pending approval. The user may edit it; `approve_plan` rejects stale
revisions or changed fingerprints.

### 2. Execution

The coordinator receives a stable system prefix plus a late, dynamic state
envelope. The stable prefix preserves provider prompt-cache value. The envelope
is rebuilt from SQLite every step, so context compaction cannot erase the goal.

Each work slice is bounded, but the durable retry count is not. Reaching a limit
creates a checkpoint and leaves the goal `RUNNING`. `/auto` applies bounded
exponential backoff and keeps retrying until verified completion or a genuine
input/approval boundary. Every failed attempt injects a self-reflection prompt;
repeated stalls escalate to a different decomposition/role/plan, while repeated
identical actions still trip a circuit breaker.

Before every workspace tool:

1. Validate schema and workspace/tool policy.
2. Request separate approval when required.
3. Journal action intent.
4. Execute with bounded output and explicit workspace context.
5. Redact result and journal terminal state.
6. Return a paired neutral tool-result message to the provider.

### 3. Adaptive delegation

The coordinator supplies a free-form task-specific role, assignment, criteria,
context, and tool allowlist. The worker receives a fresh conversation and must
return typed `return_work` evidence. It cannot edit the root checklist or finish
the goal. Recursive children use the same mechanism with depth/fan-out/step
bounds. Roles are persisted for inspection and recovery.

### 4. Completion

`finish_goal` is only a request. The deterministic gate requires:

- latest plan is the accepted revision;
- every non-obsolete task is completed;
- each completed task has evidence for that revision;
- no action remains uncertain after a crash.

The goal then enters verifying/reviewing. A fresh reviewer sees the original
objective and every exact task criterion/verification in bounded chunks, plus
recent per-task evidence; `inspect_task` pages any additional durable evidence.
It can inspect files but cannot execute shell commands or mutate the workspace.
`submit_review(pass)`
with no issues completes the goal. Failure becomes targeted repair tasks in a new
revision and returns to user approval.

## Provider-neutral history

The neutral shape remains user / assistant / tool. Tool calls and results stay
paired by harness IDs. Adapters retain optional `native` replay metadata for wire
formats that require it (notably Gemini thought signatures and call IDs). Partial
or malformed arguments become normal validation errors rather than process
crashes.

## Recovery semantics

SQLite commits state transitions and action records. On startup, an action left
`running`, a task left `in_progress`, or a worker left `in_progress` is marked
`uncertain`. The goal goes through recovering to paused, and the user inspects
actual state. The harness never assumes a side effect did not happen and never
automatically replays an uncertain write or command.

## Security boundary

The tool layer uses an explicit `ToolContext`; there is no global working-directory
mutation. Canonical resolution, symlink checks, sensitive-path pruning, private-key
detection, atomic writes, strict arguments, traversal/size/regex limits, shell
approval, a scrubbed child environment, and redaction are enforced in Python.
The system prompt adds a prompt-injection boundary, but correctness does not rely
on the prompt.

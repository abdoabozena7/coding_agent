# Implementation roadmap and status

This project grew from an educational tool loop into a persistent orchestration
harness. Every checked phase has offline regression coverage.

## Implemented

- [x] Provider-neutral streaming tool loop (OpenAI, Gemini, Ollama)
- [x] Contained read/list/grep and atomic edit/write tools
- [x] Fully approval-gated shell with explicit workspace and scrubbed environment
- [x] Typed central tool registry and shared Chat workspace/permission execution boundary
- [x] Durable generated-code artifacts and action-oriented prose-refusal recovery
- [x] Project-local dependency install, managed processes, and secure loopback HTML preview
- [x] Playwright browser verification with console/page/network evidence and screenshots
- [x] Sensitive-file and private-key exfiltration policy
- [x] Typed goal/plan/task/evidence/delegation domain model
- [x] Transactional workspace-local SQLite state and event/action journal
- [x] Crash recovery with uncertain-side-effect semantics (never blind replay)
- [x] Read-only structured planning and fresh-context plan critique
- [x] Workspace-inspection gate and fingerprinted applicability/expected-change evidence
- [x] Exact-revision/fingerprint user plan approval
- [x] User-editable, dependency-aware checklist revisions
- [x] Bounded execution slices over an unbounded durable goal
- [x] Automatic prose-only reprompts and repeated-action circuit breaker
- [x] Unlimited durable goal retries with self-reflection and bounded backoff
- [x] Dynamic task-derived worker roles and isolated structured worker results
- [x] Recursive delegation with configurable depth/fan-out/step safety limits
- [x] Durable memories/evidence outside conversation context
- [x] Safe conversation compaction that preserves tool-call/result pairing
- [x] Deterministic completion precheck and independent final review
- [x] Review failures converted to approval-gated repair revisions
- [x] Responsive ASCII dashboard, history, commands, pause/resume, auto mode
- [x] Full-screen keyboard-first setup, contextual command palette, and semantic activity motion
- [x] Import-safe package/CLI and deterministic `ScriptedProvider` tests

## High-value future extensions

These are optional production integrations, not missing invariants in the current
single-process harness:

- True concurrent read-only workers with a single-writer coordinator/event queue.
- File-claim conflict detection or per-worker git worktrees for concurrent writes.
- OS/container sandbox profiles and network policy for approved shell commands.
- Provider-native structured-output/forced-tool modes where uniformly available.
- Token/cost budgets and interactive rate-limit dashboards.
- Searchable artifact indexes for multi-million-line repositories.
- Persistent full-screen activity/composer renderer; startup and nested pickers
  already use the alternate-screen `prompt_toolkit` UI.
- MCP/server tools, web research, and external connectors behind the same policy API.
- Benchmarks comparing task success, cost, and recovery across small/large models.

The core rule for extensions: models may propose; the harness continues to own
durable state, plan/action approval, security policy, and completion authority.

# GA3BAD workflow architecture (schema v8)

GA3BAD opens in **Normal**. Every message first enters the shared Intent
Architect, which discovers repository facts, creates a canonical execution brief,
asks only consequential missing decisions, and routes to Normal or Ultra.
Planning is an internal stage, not a third public mode. Selecting or automatically
routing a mode is never evidence of approval or completion.

Normal action requests use a deterministic postcondition gate. Create/edit/install/run
intent authorizes only the matching semantic tool, while arbitrary shell work keeps
the existing Normal/Full policy. Tool-less refusals are corrected up to the configured
no-action limit. Large generated code blocks are stored by content hash and replaced
in provider history with a `CHAT_ARTIFACT` handle, so a later “save it” writes exact
bytes without regeneration. HTML “run it” uses a tokenized loopback server, browser
verification, and a managed visible browser rather than a timed foreground shell.

## State machines

The harness owns public `SessionMode` values (`normal`, `ultra`), `PlanState`,
`RunState`, `AgentState`, `UltraProfile`, and `SleepState`. Legacy session modes
migrate to Normal. Invalid Plan transitions are rejected by
`ensure_plan_state_transition`. Durable workflow checkpoints live in
`workflow_sessions`; schema v8 also persists intake sessions, questions/answers,
repository intelligence, specialist profiles/messages/packages, evaluation runs,
lessons, artifacts, session actions, and managed-resource metadata. Sleep
authorization is deliberately reset to off on process startup.

Legacy `GoalStatus`, `PlanStatus`, and task rows remain the persistence authority for existing projects. The v4 states make orchestration stages explicit without destructively rewriting those records.

## Plan pipeline

Plan runs are `inspect -> draft -> normalize -> validate -> risk-based critic -> final validation -> persist -> present -> await approval`.

Inspection is cached by normalized tool call. An empty result is a valid inventory meaning that the project will be created from scratch. The model proposes content, earlier-task numbers, criteria, and verification; `normalize_plan_draft` creates `T001` IDs and cross references. Scalar verification, numeric dependencies, whitespace, duplicates, and empty optional values are repaired mechanically. Semantic defects produce JSON-pointer errors and one targeted repair. Only complex or high-risk plans use an independent critic.

Approval is stored on the Plan and remains separate from the selected mode. A
single pending plan recognizes narrow approval utterances such as “do it” and
“go ahead”. The presentation layer renders the persisted plan immediately.
Normal and Ultra reject mutations before approval.

## Goal pipeline

The harness computes the first dependency-ready task and supplies it to execution. Dependencies are not advanced by a coordinator checklist ceremony. Worker contracts bound the objective, task, path scope, expected files, criteria, tools, verification, evidence, and exclusions. Durable tool actions and executable evidence—not prose claims—control task and goal completion. Recovery marks uncertain side effects and never replays them.

## Typed returns

All typed outputs follow `parse -> deterministic normalization -> schema construction -> semantic validation -> optional targeted repair -> persist -> agent completed`. `TypedReturnProcessor` is the shared implementation. Ultra also validates phase payloads inside `_invoke`; a malformed `GoalSpecV1` is recorded as a failed agent run and receives at most one targeted repair before a field-specific foundation error.

## Ultra and quality

Ultra performs GoalSpec, architecture, an approval-bound Master Plan, module execution, independent clean-code/security/test-quality review, executable tests, remediation, integration, global review, and final evidence. The Master Plan fingerprint includes the quality checklist. A goal-scoped baseline inventory and `QualityPolicyV1` snapshot are persisted before approval.

Every mutating Ultra tool action enters a coherent `ChangeSetV1` and mutation ledger with paths, pre/post hashes, diff or hash delta, shell-created files, command metadata, and its responsible real agent. Coder completion closes the Change Set. Independent reviewers update their own review category. A Change Set cannot integrate until all required reviews pass. Each repair agent creates a new Change Set and therefore requires fresh review.

`QualityFindingV1` fingerprints category, principle, path/location, file hash, and evidence so duplicates collapse. Critical, High, and Medium findings block completion. Performance findings require measurement. Scope/interface/dependency changes remain approval-bound. Final completion rejects open blocking findings, unreviewed Change Sets, uncertain mutations, or failed global evidence.

## Sleep profile

Sleep is `UltraProfile.SLEEP`, never a fifth interaction mode. Enabling it requires Ultra, ready Full Docker access, a safe checkpoint, and no uncertain mutation. It repeats whole-project quality cycles without a global cycle cap. Four equivalent failures are rejected: after three attempts the next cycle must use a materially different approach fingerprint. `sleep off` takes effect at a safe checkpoint and does not delete findings. Findings and cycles survive restart; Sleep authorization does not.

## Project Brain, retries, and evidence

Project Brain v4 adds quality policy, findings, Change Sets, cycles, verified decisions, and active blockers. FTS5 is used when available; the existing deterministic SQL fallback remains active otherwise. Agents receive selective goal/node/decision/file/evidence/blocker context, not full history.

Retry accounting is split by transport, typed parsing, plan format, plan semantics, critic revision, worker return, review verdict, no-progress, verification, and Sleep approach. Every record contains stage, reason, attempt, input/output fingerprints, progress, and next action.

Traces store messages, lifecycle summaries, redacted prompts, typed-validation events, tool calls/results, evidence, retries, findings, and completion decisions. They do not store or expose hidden chain-of-thought.

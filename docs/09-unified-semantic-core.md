# Unified semantic core (schema v14)

New Chat, Normal, Plan, and Ultra requests enter one durable state machine:

`exact request → repository inspection → SemanticGoalV2 → independent critic → plan → execute → verify → review → refine`

Modes are execution policies only. Chat prefers a direct low-risk contract,
Normal adapts to observed scope, Plan never executes, and Ultra permits deeper
reasoning and independent-task concurrency. No mode chooses a product domain,
role tree, artifact path, or task topology from request keywords.

## Accepted planning contract

A plan is valid only when:

- `original_request` matches the submitted request exactly;
- exclusions and negations remain exclusions;
- repository inspections are cited by stable `inspection:I###` references;
- every new path cites either an inspected repository convention or
  `user:request`;
- every existing target cites inspected evidence;
- every acceptance criterion has a verification method and evidence authority;
- a fresh critic passes the semantics and plan against the original request.

Structured output receives three exact-field repair opportunities. Failure
creates a resumable `model_capability_exhausted` checkpoint. The harness does
not create fallback tasks or paths.

## Mutation safety

Accepted `ResourceClaimV1` records become temporary write leases immediately
before a mutating action and are released afterward. Direct writes and patches
outside the lease are denied. Shell mutations outside the lease become
uncertain and pause the goal for inspection.

The v14 mutation journal stores hashes and original bytes only for leased,
non-sensitive paths. It excludes environment files, credentials, VCS data,
caches, dependency directories, and generated/build directories. Rollback
applies only while the current hash still matches the agent-written hash, so a
later user edit is never overwritten.

## Migration

Schema v14 preserves legacy goal, plan, event, and evidence history but marks
unfinished pre-v14 semantics invalid and requires fresh inspection/planning.
Legacy full-workspace recovery baselines are audited by run ID and byte size,
then removed from `.coding-agent/recovery`; new runs use mutation journals.

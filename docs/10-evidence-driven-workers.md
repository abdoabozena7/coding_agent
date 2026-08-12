# Evidence-driven workers for weak models

## Status

The worker system is implemented behind `AdaptiveOrchestrationPolicyV1`.
`shadow_mode=true` is the safe default: routing and worker impact are measured,
but the new policy does not change production decisions. Activation with
`shadow_mode=false` is rejected unless the caller supplies a passed, matched
`BenchmarkReportV1` for the exact active model fingerprint. Returning to shadow
mode is an immediate rollback and does not require a benchmark.

No production observation is labelled causal. Causality is accepted only for
rows produced by `FourArmBenchmarkRunner` with `matched_benchmark=true`.

## Runtime design

Execution uses these provider-neutral contracts regardless of whether the
message routed to staged or recursive depth:

- `AdaptiveOrchestrationPolicyV1`: the four-call ceiling, risk budgets, shadow
  switch, learning threshold, suppression threshold, and 10% deterministic
  re-probe.
- `WorkerMissionV2`: role, bounded objective, stable criterion contracts,
  context visibility, mutation policy, tools, falsification targets, seed, and
  approach fingerprint.
- `EvidenceClaimV1`: criterion id, current artifact hash, durable evidence
  references, separating/falsification check, authority, verifier, and verdict.
- `WorkerImpactV1`: evidence novelty, verified and false findings, accepted
  fixes, score delta, tokens, latency, and `useful|neutral|harmful` outcome.

The deterministic router uses declared task risk, expected changed-file count,
interface/security/test signals, failed approaches, missing evidence, design
ambiguity, and deterministic-gate availability. It does not use model
confidence.

The adaptive stage is followed by a separate fixed post-build review stage; it
does not replace or broaden adaptive routing:

1. the coordinator dispatches the builder and any risk-selected predictor,
   challenger, selector, falsifier, or repairer;
2. deterministic/runtime verification produces executable evidence;
3. six read-only specialists run in this exact order: `security`,
   `clean_code`, `testing`, `architecture`, `fidelity`, and `regression`;
4. `FixedSpecialistReviewGateV1` mechanically aggregates those six verdicts
   against one artifact hash and mutation sequence;
5. a failed gate creates repair tasks tagged `required_worker_role=repairer`
   and supplies only the verified specialist findings to that single writer;
6. every mutation invalidates the old specialist results, so completion needs
   six fresh reviews and a fresh gate for the repaired artifact revision.

`SpecialistReviewResultV1` is the durable per-role contract. A missing role,
unknown verdict, old artifact hash, or old mutation sequence yields
`NEEDS_EVIDENCE`; one explicit failed role yields `FAILED`; only six current
passes yield `PASSED`. Runtime/test failures remain hard veto evidence and are
not counted as a seventh specialist vote.

| Observed tier | Maximum model calls including builder | Extra roles |
|---|---:|---|
| Low | 1 | none; deterministic verification only |
| Medium | 2 | blind falsifier |
| High | 4 | predictor, blind falsifier, repairer |
| Architecturally ambiguous | 4 | staged challenger, anonymous selector, repairer |

Predictor, falsifier, selector, and reviewer are read-only. A challenger can
return a complete candidate only through `return_work.staged_candidate`; the
harness materializes it under an isolated agent-owned staging directory and
does not treat it as verified. A repairer is not scheduled without a current,
non-infrastructure failing action or verified finding. Final workspace mutation
still uses the existing task leases and a single writer.

Every accepted plan criterion receives a stable id such as `T001:AC1`. In
active mode, verifier/repair roles cannot return `success` unless every current
criterion has a claim referencing fresh Harness/User evidence. Evidence from a
direct write does not prove behavior. Test/runtime criteria require executable
or browser evidence; exact file/content criteria may use a current file hash or
read result. A later mutation invalidates evidence from an older mutation
sequence.

## Recursive execution consensus

The deterministic triage projection is not a voter. Harness failures and
trusted failed test/browser evidence are vetoes. Model votes without a trusted
evidence reference become abstentions when the policy is active. An open or
tied round becomes `NEEDS_EVIDENCE`; it is not repaired by adding another
correlated vote.

All six post-build review roles now execute for every completed artifact.
Accepted change signals are retained as applicability evidence inside each
mission, allowing an unaffected dimension to pass without inventing a defect;
they no longer remove a role from the fixed stage. Shadow mode still controls
the earlier adaptive worker decisions and does not bypass the fixed review or
evidence gate.

## Durable measurement and diagnostics

Schema v17 adds:

- `orchestration_experiments` for staged and recursive Execution arms, fingerprints, matched
  status, scores, false completion, calls, tokens, time, GPU time, evidence, and
  metrics.
- `worker_contributions` for role, approach fingerprint, evidence novelty,
  verified/false findings, accepted fixes, score delta, cost, claims, and impact
  outcome.

Advanced Tracing exposes the experiment, why a role ran, its evidence, novelty,
cost, measured impact, shadow state, and whether a result is observational or a
matched causal benchmark. Diagnostics verifies the router, evidence contracts,
veto behavior, balanced fixture set, benchmark surfaces, storage, and learning
hooks.

After at least 20 homogeneous observations for one `(model fingerprint, task
class, role)`, a role with useful rate below 15% and non-positive mean score
delta is suppressed for 90% of future units. The remaining deterministic 10%
is a re-probe; goal/revision/task identity participates in the sample so common
ids such as `T001` do not freeze exploration.

## Matched benchmark

`agent.worker_benchmark` defines 24 isolated fixtures, four in each class:
exact edit, bug repair, multi-file integration, browser/tool use,
security/edge case, and requirement fidelity. The default seeds are 11, 29,
and 47. One run therefore contains 72 trials per arm and 288 trials total.

The arms are:

1. single agent (`1x`);
2. independent compute-matched sampling with no evidence roles;
3. current orchestration;
4. adaptive evidence orchestration.

The runner forbids mixed model fingerprints, gives every arm the fixture's same
tool and context limits, caps calls at four, and runs the compute-matched arm
with exactly the adaptive arm's observed call count for that fixture and seed.
The model executor never receives hidden checks. A provider-specific executor
must run the model and a separate deterministic evaluator must populate
`TrialObservationV1`.

Activation requires all of the following:

- exactly 24 fixtures x 3 seeds in all four arms;
- relative success uplift of at least 20% over single-agent;
- paired bootstrap 95% success-delta interval with lower bound above zero;
- at least 5% uplift over compute-matched, or no lower success with at least
  25% fewer tokens;
- at least 40% lower false completion;
- at least 50% verified worker contribution;
- median calls no more than `3x` and p95 no more than `4x`;
- zero deterministic runtime/security-gate regressions.

Wilson intervals are reported for arm success rates. The paired bootstrap uses
the exact `(fixture, seed)` matches, so task difficulty is not allowed to drift
between arms.

## Improvement hypotheses

These are non-additive acceptance hypotheses for weak models, not measured
results. They remain labelled `hypothesis` until a complete matched benchmark
replaces them with an estimate and confidence interval.

| Change | Weak-model hypothesis |
|---|---:|
| Causal benchmark and ablation | 0% direct quality; 80-100% fewer unsupported improvement claims |
| Criterion-to-independent-evidence binding | 10-18% higher success; 40-60% lower false completion |
| Blind falsifier | 8-15% higher medium/high-risk success; 20-35% fewer missed defects |
| Predictor or independent challenger | 5-12% higher ambiguous/high-risk success |
| Evidence consensus | 4-9% higher acceptance accuracy; 25-40% lower false acceptance |
| Verified-finding-only repairer | 8-14% higher repair success; 35-55% fewer no-progress retries |
| Adaptive routing and early stop | 25-45% fewer calls than fixed reviewers; quality unchanged to 5% higher |
| Per-role utility learning | 5-10% higher success after enough data; 20-40% fewer ineffective workers |

The project target remains 20-35% relative task-success uplift over
single-agent, at least 5% over compute-matched (or equal quality with 25% fewer
tokens), and 40-60% lower false completion. These targets are not asserted as
achieved before the live `gemma4:e4b` benchmark passes. Confirmation on
`qwen2.5-coder:7b` and `deepseek-coder:6.7b` must run separately and must not
mix models within a task.

## Current evidence and remaining boundary

The implementation has deterministic and integration coverage for routing,
role permissions, staged candidates, duplicate fingerprints, schema migration,
fresh evidence after mutation, false-completion rejection, deterministic veto,
adaptive review applicability, the fixed six-role specialist gate, Repairer
routing, contribution learning, four-arm matching,
activation statistics, multimodal provider transport, pixel-only Vision probing,
and hash-bound media delivery. The final local verification completed with **1,164
passed, 2 skipped, and 88 subtests passed**. The one warning is an existing
Starlette deprecation notice for the FastAPI test client; it is not a failed
runtime, security, or orchestration gate.

The live 288-trial `gemma4:e4b` benchmark has not been run as part of this code
change. Therefore no percentage uplift is claimed, adaptive decisions remain in
shadow mode, and the activation gate is intentionally closed.

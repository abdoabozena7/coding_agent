# Weak-Model-First Quality Convergence Evidence

## Integrated architecture

- Every new Goal persists one versioned `WeakModelPolicy` and one fingerprinted `GoalContractV1` in the durable Goal journal. Restored runtimes rehydrate the persisted policy rather than silently taking newer defaults.
- Every participating provider call receives a compact, actor-specific Goal Contract projection. Trace events retain the contract fingerprint, policy version, applied rules, provider capability choice, and request adapter.
- Chat, Plan, Goal, and Ultra mode changes update execution policy while retaining the same run ID, contract fingerprint, objective, plan, artifact references, mutation sequence, and convergence state.
- Accepted plan criteria and verification requirements become a persisted Quality Target. HTML targets receive visual/interaction dimensions; castle-siege goals receive actor, motion, depth, stability, self-containment, and responsive dimensions.
- Goal mutations increment a durable mutation sequence, invalidate prior evaluation, update the mixed-artifact index, and create a hash-bound Goal Change Set containing pre/post hashes, diff, responsible actor, parent task, refinement links, and review state.
- Completion rejects unfinished tasks, unverified model-authored prose, below-target/refining quality state, stale converged evaluations, uncertain actions, and uncertain workers. A passing independent review creates a fresh per-dimension evaluation bound to current artifact hashes and mutation sequence.
- Visual artifacts without a trustworthy visual evaluator stop at `USER_REVIEW_REQUIRED`; structural checks or a reviewer assertion cannot fabricate subjective visual convergence.

## Runtime policy and convergence behavior

- One primary bounded proposal is requested per weak-model call. When native tools are absent, the model emits one minimal JSON action proposal; the harness extracts the first balanced object, normalizes safe field aliases, generates the action ID, validates the tool name/arguments, and alone executes the action.
- Short feedback is attached to the current Goal Contract and Quality Target, classified into affected dimensions, matched against deterministically indexed components, and converted into a delta refinement action without replacing the run or plan.
- Equivalent failed action approaches are persisted. After three failures, the fourth equivalent action is rejected and a forced-approach-change trace is emitted.
- Failures create normalized signatures containing domain, operation, command/exit data where available, paths, current file hashes, and a stable fingerprint. The next contract projection carries the failed hypothesis, bounded repair constraints, and retrieved component scope.
- Ordinary Chat mutations are labeled `BELOW_TARGET` because they have not passed Goal convergence. The continuation proposal recommends Goal mode and preserves the Chat run ID, objective, artifacts, and messages when the user answers “yes”, “continue”, or equivalent.

## Provider compatibility

- Ollama configuration separates base URL, protocol, endpoint, selected model, capability profile, request compiler, and response parser.
- Safe capability probing uses `GET /api/version` and `GET /api/tags`, records daemon/model evidence, and caches the result before meaningful generation.
- The request compiler omits tools, structured output, thinking, and known-unsupported fields unless supported. A reachable unsupported-field HTTP rejection is classified accurately and retried once with only that safe field removed.
- Connection refusal, DNS/socket failure, timeout, HTTP 4xx/5xx, missing endpoint/model, model-load failure, invalid payload, unsupported tools/format/parameter, context overflow, and a wholly malformed NDJSON stream have distinct diagnostics. Provider bodies are retained after secret redaction.

## Live local-model validation (2026-07-12)

- Daemon: Ollama `0.31.2`, `http://127.0.0.1:11434`.
- Installed local models observed: `gemma4:e4b`, `qwen2.5-coder:7b`, `deepseek-coder:6.7b`, plus an embedding model and cloud entry.
- `qwen2.5-coder:7b` profile: native chat `/api/chat`, 32,768 context, completion/tools advertised, structured output/thinking/vision not advertised, digest `dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364`.
- The first live qwen request exposed a real HTTP 400 because the old compiler sent `think` to a model that does not support thinking. The daemon was correctly classified reachable; capability-aware omission fixed the request. A subsequent qwen generation remained queued for 184 seconds because `gemma4:e4b` was resident and the daemon did not unload it, so no qwen generation success is claimed.
- `gemma4:e4b` profile: native chat `/api/chat`, completion/thinking/tools advertised, structured output/vision not advertised, digest `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb`.
- Real planning call: succeeded in `50.968s` (38 prompt tokens, 310 output tokens).
- Real bounded implementation call: succeeded in `12.546s`, returning one native `write_file` action.
- Live convergence exercise: implementation call `5.710s`; refinement call `80.095s`. The first artifact was valid but wrong (`Hello from weak local model with no punctuation`, hash `d70e35...b5c4`). Deterministic evaluation marked it `BELOW_TARGET`. The model refinement repeated the identical wrong content and produced no hash change, so it was not counted as progress.
- Forced different approach: the harness used a deterministic exact transformation. The final content was `Hello from weak local model`, hash `9cddc1...f19f`; the hash changed and the fresh exact-content gate passed.

## Automated verification

- Full suite after executable-Chat/browser tooling: **251 passed, 2 skipped** in 23.047 seconds.
- Real browser smoke: tokenized loopback preview returned HTTP 200, opened an isolated visible Chrome profile, passed Playwright console/page/network verification, captured a screenshot, and stopped cleanly.
- Coverage includes active policy/contract projection, mode continuity, Chat-to-Goal continuation, below-target completion rejection, visual evaluator absence, stale evaluation contracts, mutation Change Sets, mixed HTML indexing, short-feedback delta refinement, fourth-attempt blocking, provider capability fallback/adaptation, HTTP 400 truthfulness, malformed stream handling, crash uncertainty, migration compatibility, and castle-siege candidate gating.
- The castle-siege Goal end-to-end fixture now exercises a weak static first artifact, independent deficiency detection, a harness-approved in-scope repair revision, an intentionally failing runtime command, normalized error-signature persistence, a later repair Change Set, fresh successful narrow verification, final hash-bound evaluation, and truthful `USER_REVIEW_REQUIRED` for the remaining subjective visual judgment.

## Honest limitations

- The live convergence exercise used an isolated exact-text artifact, not a full live castle-siege generation, because local model latency and resident-model contention made a multi-call visual run impractical in this validation window.
- Subjective visual excellence still requires a trustworthy vision evaluator or explicit user review. The harness intentionally stops short of claiming it from DOM/CSS labels, screenshots alone, or text review.
- Goal quality structures are persisted transactionally inside the existing Goal metadata journal for backward compatibility rather than in new normalized SQL tables. Legacy completed Goals remain readable and, because they lack a Quality Target/evaluation fingerprint, are not represented as newly convergence-verified.

# Weak-Model-First Quality Convergence Evidence

## Integrated architecture

- The public surface has two modes only: `Normal` and `Ultra`. Every ordinary
  message enters the same durable `Intent Architect` intake gate; planning and
  clarification are internal stages rather than weaker modes.
- Ambiguous consequential decisions use `ClarificationQuestionV1`: exactly three
  mutually exclusive suggestions, only the first marked Recommended, plus the
  permanent free-form fourth answer. Clear requests route without unnecessary
  questions.
- Every task persists one canonical `ExecutionBriefV1`, one versioned
  `WeakModelPolicy`, and one fingerprinted `GoalContractV1`. Restored runtimes
  rehydrate the persisted policy, intake answers, plan, specialist tree, and
  checkpoints rather than silently starting again.
- Every participating provider call receives a compact, actor-specific Goal Contract projection. Trace events retain the contract fingerprint, policy version, applied rules, provider capability choice, and request adapter.
- `Normal` is always goal-based plan/execute/review/repair. `Ultra` adds
  architecture candidates, independent critics, a clean-context judge, recursive
  specialist packages, typed messages, consensus, and parent integration. Legacy
  `chat|plan|goal` values restore as `Normal` without losing state.
- Accepted plan criteria and verification requirements become a persisted Quality Target. HTML targets receive visual/interaction dimensions; castle-siege goals receive actor, motion, depth, stability, self-containment, and responsive dimensions.
- Goal mutations increment a durable mutation sequence, invalidate prior evaluation, update the mixed-artifact index, and create a hash-bound Goal Change Set containing pre/post hashes, diff, responsible actor, parent task, refinement links, and review state.
- Completion rejects unfinished tasks, unverified model-authored prose, below-target/refining quality state, stale converged evaluations, uncertain actions, and uncertain workers. A passing independent review creates a fresh per-dimension evaluation bound to current artifact hashes and mutation sequence.
- Visual artifacts without a trustworthy visual evaluator stop at `USER_REVIEW_REQUIRED`; structural checks or a reviewer assertion cannot fabricate subjective visual convergence.

## V8 code intelligence, swarm, and learning

- Repository understanding is layered: Python AST, Tree-sitter adapters for
  JavaScript/TypeScript/TSX/CSS, HTML DOM extraction, explicit low-confidence
  fallback, imports/exports/inheritance/routes, dependency and confidence-rated
  call graphs, `CODEOWNERS` plus Git ownership, and requirement-to-code semantic
  maps.
- Incremental indexing is hash-based and hierarchical
  (`repository -> package -> module -> symbol -> code slice`). Hybrid retrieval
  fuses FTS5/BM25, float32 embeddings with optional HNSW, and graph traversal
  using reciprocal-rank fusion while retaining provenance, confidence, and file
  hashes.
- `WorkforceDesigner` creates typed specialist contracts and recursively
  decomposes work to a bounded depth/node budget. Leaves publish
  `ComponentPackageV1`; only parent assemblers integrate packages and only the
  `FinalAssembler` owns final output paths.
- The reasoning harness evaluates inspectable hypotheses, architecture
  candidates, decision records, critic verdicts, and quality findings. It never
  stores or claims access to hidden chain-of-thought.
- Typed swarm frames carry run/node/sender/recipient/schema/correlation/deadline,
  evidence, lease, and fencing data. SQLite provides durability while the event
  bus provides immediate delivery. Two-thirds consensus, deterministic leader
  recovery, and an independent tie judge are executable and tested.
- `LearnedLessonV1` persists project/global lessons with applicability, evidence,
  success/failure counts, confidence history, recency, version, and
  supersession. Evaluation outcomes reinforce or decay lessons before later runs
  retrieve them.

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

## End-to-end validation (2026-07-13)

- Full suite: **347 passed, 2 skipped, 46 subtests passed** in 47.21 seconds.
- `/doctor --record`: structural and behavioral readiness passed, including code
  retrieval MRR `1.00`, shallow reasoning rejection (`0.2`), strong reasoning-graph
  acceptance (`1.0`), two-voter swarm consensus, and evaluation-driven remediation.
  A second process on the same workspace recorded both structural and behavioral
  trends as `stable`.
- Real semantic retrieval used `nomic-embed-text:latest` without fallback, returned
  768-dimensional vectors, and ranked the paraphrased authentication target first.
- Cross-process learning persisted one Project Memory after closing and reopening
  SQLite. Verified reuse/outcome evidence raised stored confidence from `0.82` to
  `0.855` and effective confidence to `0.88`.
- Legacy Chat/Plan/Goal aliases and ULTRA all booted against the same durable
  state; v8 now exposes those legacy values only as Normal compatibility aliases.
- A real single-file Three.js road-crossing game passed loopback browser verification
  with no console/page/network failures. Screenshot-aware quality scored `0.9144`;
  Playwright exercised Start, keyboard motion, collision, in-page Game Over, Restart,
  and touch controls.
- The E2E run exposed a weak-model transport defect where literal `\\n` separators
  could corrupt a full document. The runtime now repairs source-layout escapes for
  native and textual tool calls while preserving intended escapes inside strings;
  a regression test covers the exact failure.

## V8 acceptance evidence (2026-07-19)

- Full suite: **365 tests passed, 2 skipped** in `87.853s`. Coverage includes
  intake UX/routing, schema-v8 restart, local/cloud pipeline parity, recursive
  single-file specialists, candidate composition across revisions, contract
  refinement from critic findings, consensus/tie/leader recovery, cross-run
  learning, browser fail-closed gates, and mixed ML/backend/frontend repositories.
- Synthetic large-repository benchmark: **1,000,000 lines across 500 files**.
  Cold indexing took `190.540s`; updating 50 changed files took `4.104s`;
  query p95 was `0.128s`; measured peak allocation was `164,125,072` bytes.
  All acceptance thresholds passed: update `<5s`, query p95 `<2s`, memory `<4GB`.
- Live `/doctor --live --record` against local `gemma4:e4b` passed structural,
  behavioral, and live orchestration-delta gates. The raw bounded request failed
  to produce JSON after using its output budget for thinking; the controlled
  harness request returned valid structured JSON. Retrieval MRR was `1.00`,
  shallow reasoning scored `0.2` and was rejected, the evidence graph scored
  `1.0`, two-voter consensus closed, and failed evaluation created durable
  remediation memory.
- The single-file Three.js road-crossing artifact was reopened through a loopback
  server with Playwright. Start, keyboard input, collision, Game Over, Restart,
  touch controls, and a real WebGL context were observed with **zero console
  errors and zero warnings**.
- Screenshot-aware evaluation of the active gameplay frame passed at
  **overall `0.9549`**: self-containment, 3D rendering, animation, gameplay,
  visual richness, responsive/accessibility, and runtime each scored `1.0`;
  visual composition scored `0.7184`. A separate blurred Game Over frame scored
  `0.79` and was correctly rejected as a beauty screenshot rather than being
  allowed to inflate the final visual claim.

## Automated verification

- Earlier executable-Chat/browser milestone: **251 passed, 2 skipped** in 23.047 seconds.
- Real browser smoke: tokenized loopback preview returned HTTP 200, opened an isolated visible Chrome profile, passed Playwright console/page/network verification, captured a screenshot, and stopped cleanly.
- Coverage includes active policy/contract projection, mode continuity, Chat-to-Goal continuation, below-target completion rejection, visual evaluator absence, stale evaluation contracts, mutation Change Sets, mixed HTML indexing, short-feedback delta refinement, fourth-attempt blocking, provider capability fallback/adaptation, HTTP 400 truthfulness, malformed stream handling, crash uncertainty, migration compatibility, and castle-siege candidate gating.
- The castle-siege Goal end-to-end fixture now exercises a weak static first artifact, independent deficiency detection, a harness-approved in-scope repair revision, an intentionally failing runtime command, normalized error-signature persistence, a later repair Change Set, fresh successful narrow verification, final hash-bound evaluation, and truthful `USER_REVIEW_REQUIRED` for the remaining subjective visual judgment.

## Honest limitations

- A local `qwen2.5-coder:7b` completed bounded calls but did not reliably finish the
  full 16 KB visual refinement within the practical per-call window. The harness
  rejected/no-op timed-out attempts instead of reporting success. The cloud model
  later hit its account session quota (HTTP 429), so provider capacity remains an
  external limit rather than something orchestration can manufacture away.

- The live convergence exercise used an isolated exact-text artifact, not a full live castle-siege generation, because local model latency and resident-model contention made a multi-call visual run impractical in this validation window.
- Subjective visual excellence still requires a trustworthy vision evaluator or explicit user review. The harness intentionally stops short of claiming it from DOM/CSS labels, screenshots alone, or text review.
- These results demonstrate a measurable architecture-driven uplift and reliable
  acceptance behavior for the tested tasks. They do not prove that one local
  model equals every larger model on every repository or domain; unknown
  capabilities remain blockers rather than being converted into fabricated
  success.
- Goal quality structures are persisted transactionally inside the existing Goal metadata journal for backward compatibility rather than in new normalized SQL tables. Legacy completed Goals remain readable and, because they lack a Quality Target/evaluation fingerprint, are not represented as newly convergence-verified.

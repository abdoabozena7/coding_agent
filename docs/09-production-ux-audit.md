# Production workflow UX audit

Date: 2026-08-05

This audit follows one project from first prompt through planning, approval,
execution, model changes, recovery, review, and completion. The percentages are
independent estimates of how much each change reduces workflow interruption or
operator confusion for the affected path. They are not telemetry measurements
and must not be added together.

## Release-blocking workflow findings

| Area | Observed failure | Resolution | Status | Estimated impact |
|---|---|---|---|---:|
| Startup recovery | Reopening a project with a saved `dispatching` stage started a new Web server, attempted a mode change, then exited with `Mode is locked`. | Startup now derives recovery from the durable workflow lock, preserves the saved mode, and opens the live workspace without replaying the turn. | Fixed and regression-tested | 35% |
| Multi-launch ownership | Starting the same workspace more than once created competing loopback servers and eventually surfaced a misleading mode-lock fatal. | Each session now has an OS-level owner lease with heartbeat. A second launcher attaches to the authenticated owner Web endpoint; the OS releases the lease after a crash so the next launch can take over safely. | Fixed and regression-tested | 30% |
| First-prompt handoff | The terminal accepted the prompt while Web could open late or show an empty intake, encouraging the user to submit the same request twice. | Web opens only after the durable semantic turn/Goal exists, shows that exact request immediately, and rejects duplicate active or queued prompts idempotently. | Fixed and regression-tested | 30% |
| Pre-Goal provider boundary | A provider failure during semantic intake left a durable prompt but no Goal row, so Web showed `Internet or provider is unavailable` beside a fresh request composer and disabled `/retry`, `/resume`, and local continuation. Older live sessions can also lack the provisional Goal projection. | A non-terminal semantic turn is projected as the one provisional workflow identity. The Web client also recognizes a saved retrying/routing checkpoint from its runtime identity when a legacy server omits that provisional row, blocks duplicate submission, and exposes the matching recovery commands. Pre-Goal local continuation updates the same semantic turn's capability envelope and resumes it without creating a duplicate Goal. | Fixed and browser-tested | 34% |
| First-prompt request failure | The first Web submit could durably save the exact request and then return a provider error. The composer remained populated or the Plan surface offered no recovery buttons, so the user could unknowingly submit the same request twice. | The client refreshes the durable workspace after a failed first submit, clears the composer, renders one saved-request checkpoint with Retry, strongest-local continuation, and Stop safely controls (including on Plan), and states that the original prompt must not be submitted again. | Fixed and browser-tested | 32% |
| Legacy recovery surface | An older Web response could still render the generic `Plan your next request` attention panel above the saved local-provider boundary, making one request look like both an active recovery and a new prompt. | When the legacy response has no Goal identity, the client suppresses the generic attention panel and renders one saved-request recovery surface with Retry, strongest-local continuation, and Stop safely actions. | Fixed and live-verified | 16% |
| State authority | Terminal, Plan, and Execution could project different phases, revisions, or model modes from the same session. | `StateStore` is the durable authority; runtime snapshots now project the current phase, waiting owner, plan revision, execution class, and interaction mode to both surfaces. | Fixed and regression-tested | 40% |
| Project reopening | Opening an existing project replayed the first-run protection/model/capacity/permission choices even though the project already had a durable setup. GitHub/local history and the saved model could be treated as unrelated pieces. | Startup hydrates the saved model, access, mode, capacity, and version-control profile from the project’s existing journal/config. The unified `/settings` hub exposes Project setup and protection at a safe checkpoint; a missing GitHub remote no longer forces a new first-run questionnaire. | Fixed and browser-tested | 34% |
| Owner/UI version skew | A browser tab connected to an older owner could load the newer static shell, then show a raw `404` when a new settings surface was opened. | Project settings now names the stale owner, states that the durable workflow is unchanged, and gives the one safe remediation (restart the owner once) instead of exposing a transport error or inviting a duplicate setup. | Fixed and live-verified | 12% |
| Awaiting-plan state | Web could say both `Working on your request` and `No workflow is active`, show `plan r—`, or label a recursive run as `ultra`. | Awaiting approval is now one coherent runtime state with `waiting_on=user`, the pending revision, the authoritative interaction mode, and a plan-review message. | Fixed and regression-tested | 25% |
| Approval ownership | The terminal said approval was required but exposed no usable choice while Web did not reliably show the request. | Pending approval is durable and fingerprinted. Web is primary while connected; terminal keys are an emergency fallback when its SSE connection drops. | Fixed and regression-tested | 45% |
| Approval status projection | The approval banner could say `Waiting for your approval` while the lower Execution card still said `running` and offered `Pause safely`. | Execution now projects `waiting` from the same durable runtime snapshot and labels the boundary explicitly; the pause control is not shown while approval is pending. | Fixed and browser-tested | 12% |
| Approval choices | Approval only behaved like a binary allow/deny decision. | The card and terminal contract support `Allow once`, `Always allow this session`, `Deny`, and `Stop safely`; session grants are policy-group scoped and audited. | Fixed and regression-tested | 20% |
| Accepted approval residue | An accepted card stayed visible and a second click returned `already accepted`. | The matching card is removed immediately and stale copies reconcile from accepted-action events without replaying the action. | Fixed and browser-tested | 18% |
| Sleep semantics | Enabling Sleep did not resolve every pending approval and the user could not tell which policy was active. | One persistent control has Off, Safe Auto, and typed-confirmation Full Auto. Full Auto accepts a critic-reviewed plan, deterministic intake/plan/ULTRA questions (recommended, explicit default, then first option), and every tool boundary in the selected workspace; conflicting or malformed questions remain a truthful named boundary. | Fixed and regression-tested | 35% |
| Access projection consistency | Web could show Full access while the terminal footer still said Normal, especially after reconnect or when access changed before the terminal renderer mounted. | Access changes publish one durable compatibility event consumed by both surfaces, and terminal startup hydrates the UI store from the saved session before rendering. | Fixed and regression-tested | 18% |
| Restarted tool boundary | Enabling Full Auto after a process restart could persist an approval without re-entering the saved paused goal. | Full Auto now consumes the exact durable approval marker, resumes the paused goal once, and lets the controller start the next saved slice without replaying an already-running worker. | Fixed and regression-tested | 24% |
| Timeline stability | Polling/heartbeats replaced timeline rows, causing visible flicker. | Rows are keyed and reconciled in place; unchanged polling produces zero DOM mutations and only genuinely new events animate. | Fixed and browser-tested | 15% |
| Review status projection | A review page could show `Review the recorded changes` while the live header/sidebar still said `Working` or `running`. | Reviewable change sets now project a user-owned `reviewing`/`waiting` boundary through the shared runtime snapshot and all Web surfaces. | Fixed and browser-tested | 12% |
| Controls discoverability | Important tree, agent, tool, status, retry, and view controls were scattered or terminal-only. | The fixed composer exposes one state-aware slash palette for model, local continuation, tree, agents, tools, activity, status, pause/resume/retry/stop, Sleep, and all workspace views. | Fixed and browser-tested | 22% |
| Model switching | Changing a model mid-work could imply an unsafe immediate swap or lose plan context. | Normal model changes occur only at a saved checkpoint and retain the accepted plan and evidence. Invalid choices are disabled with a reason. | Fixed and regression-tested | 30% |
| Local-model continuation | Recovery offered `switch to local` even when already local, and a weaker local model could fail large structured turns repeatedly. | Recovery no longer offers the misleading action when local. `/continue with local model` selects the strongest tool-capable local model by its advertised parameter/capability envelope, narrows only remaining task packets, and keeps the accepted plan and all quality gates unchanged. | Fixed and regression-tested | 38% |
| Full Auto cloud exhaustion | Full Auto still stopped at a cloud provider failure instead of continuing unattended on local capacity. | Full Auto now discovers a local tool-capable model, switches at the saved checkpoint, records `provider.local_fallback`, preserves accepted quality gates, and resumes automatically. It pauses truthfully only when local fallback is unavailable or also fails. | Fixed and regression-tested | 42% |
| Full Auto activation after a visible boundary | Turning Full Auto on while the terminal was already blocked could leave the owner asleep because no new approval event arrived to wake it. | Enabling Full Auto now wakes the owner to inspect the durable boundary immediately; it resolves only the bounded policy decision and never replays a stale action. | Fixed and regression-tested | 31% |
| Cloud failure during initial planning | Planning caught provider exhaustion internally and left a paused goal, so the Full Auto worker never reached the execution fallback branch. | The controller now recognizes the durable planning recovery marker, clears the old provider backoff, selects the strongest local tool-capable model, and resumes the saved planning stage once. | Fixed and regression-tested | 30% |
| Local continuation at the pre-plan checkpoint | Choosing local continuation after a cloud failure during semantic intake changed the provider but left the saved turn paused, so the browser appeared to accept the action without progress. | The continuation action now recognizes the provider recovery marker even before a Goal/Plan exists, clears the old provider backoff, and resumes the same semantic turn exactly once. | Fixed and regression-tested | 38% |
| Local provider interruption in Full Auto | A local model load/GPU/provider interruption opened a recovery attention even though Full Auto was enabled, leaving an unattended session stopped indefinitely. | Full Auto now saves the exact semantic or Goal checkpoint and retries it automatically with a persistent exponential backoff (5 seconds up to 5 minutes). Web names the scheduled retry and does not present a hidden user-owned modal. | Fixed and regression-tested | 33% |
| Repeated local timeout | A local runner that stayed unavailable could be retried forever even when a second compatible tool-capable local model was installed. | Full Auto gives the current local runner two bounded attempts, records its aliases, then selects the next-strongest compatible local model once. The existing local continuation policy narrows only remaining packets and keeps the accepted plan, quality target, executable evidence, and final review gates unchanged. If no alternative exists, the named local boundary remains visible. | Fixed and regression-tested | 24% |
| Local planning repair leakage | Typed-output repair diagnostics were appended to every public task description, creating repeated noisy plan text. | Technical repair diagnostics are stored in metadata; legacy suffixes are stripped from the public projection. User-authored task text is preserved. | Fixed and regression-tested | 20% |
| Provider/network boundary | A disconnected provider or exhausted cloud model could look like generic `paused`, `working`, or a runtime error; a local Ollama failure was mislabeled as an Internet outage. | Network loss, local model-runner loss, runtime loss, rate limit, and exhausted quota have distinct states and recovery actions. Saved stage continuity is stated explicitly, with `waiting_on=model` for local runners and `waiting_on=network` for cloud transport. | Fixed and browser-tested | 32% |
| Duplicate work | Repeated prompts, actions, findings, or retries could create repeated edits or repeated UI blocks. | Active/queued prompts and workspace actions are idempotent; plan/action/finding fingerprints collapse duplicates; retry ledgers require a changed approach after equivalent failures. | Fixed and regression-tested | 28% |
| Local-model recovery copy | A local session could still offer “switch to local model,” even though it was already local; the action label made the first planning failure look like a model-selection mistake. | Recovery hides the redundant action during an ordinary local checkpoint and says “Try strongest available local model” only when a local provider boundary justifies a ranked replacement. | Fixed and browser-tested | 14% |
| Missing session envelope race | A Web/controller read in the small startup or reconnect window could turn a missing presentation row into an internal-error card even though the durable goal still existed. | Runtime reads now recreate the secret-free session envelope from the latest durable goal and record `workflow.session_recreated`; the upsert is idempotent and leaves goals, plans, and tasks untouched. | Fixed and regression-tested | 27% |
| Provider failure transport | A first Web prompt could raise a provider exception through ASGI and log a raw server traceback while the client was trying to recover the saved request. | Provider-unavailable failures now return a named retryable `503` response with `saved_stage=true`; the client can refresh the exact checkpoint without exposing a transport error. | Fixed and regression-tested | 19% |
| Drawer interaction | The model drawer opened over an active slash menu, with background controls still focusable. | Opening a drawer closes the palette, makes the app shell inert, traps keyboard focus, supports Escape, and restores the invoking control on close. | Fixed and browser-tested | 10% |
| No-op attention action | `Review plan` could appear while the user was already on the current Plan view and do nothing. | Navigation-only attention actions are hidden when they target the active view. | Fixed and browser-tested | 8% |

## Production hardening still required

These items do not invalidate the repaired single-machine workflow, but they
should remain visible release work rather than being described as complete.

| Priority | Gap | Required production work | Estimated impact |
|---|---|---|---:|
| P0 | Real provider limit telemetry | Ingest provider-reported remaining quota/reset metadata where available. Continue to show `unavailable` when a provider does not report it; never invent a percentage. | 25% |
| P0 | Live local acceptance | Run the isolated Ollama workflow with the actual local runner: one prompt, saved Plan, real files, checks/preview, and Web/terminal/DB identity matching. This acceptance is currently pending because `127.0.0.1:11434` is not reachable. | 35% |
| P1 | Cross-terminal manual matrix | Run mouse, resize, Unicode, screen-reader/contrast, and reconnect checks in Windows Terminal, classic console, VS Code terminal, tmux/zellij, and SSH. Automated TUI tests cannot prove emulator-specific behavior. | 18% |
| P1 | Long-run disconnect soak | Soak-test browser close/reopen, SSE reconnect, provider offline/online, machine sleep, and process crash during every approval and mutation boundary. | 25% |
| P1 | Observable budgets | Add optional per-goal token, time, and cost budgets plus provider rate-limit counters, while preserving quality gates and safe checkpoints. | 15% |
| P1 | Measured UX telemetry | Record privacy-safe funnel metrics for duplicate submissions, approval latency, recovery success, model-switch success, and abandoned sessions so the estimates above can be replaced with measured values. | 20% |
| P2 | Concurrent workers | Add a single-writer coordinator plus read-worker isolation/file claims before enabling true parallel mutation. | 12% |
| P2 | Large-repository indexing | Add a searchable artifact/index layer and benchmark it on million-line workspaces. | 10% |

## Release gates

A production candidate should not ship until all of these remain green:

1. One prompt produces one durable Goal and one visible Web workspace.
2. Terminal and Web show the same revision, phase, model, waiting owner, and
   recovery action from the same snapshot.
3. Every mutation has one fingerprinted decision and cannot be replayed by a
   double click, refresh, reconnect, or restored process.
4. Full Auto is explicit, workspace-scoped, session-bounded, and fully audited.
5. Model changes preserve accepted scope; compensated local continuation changes
   packet size, never the quality target or completion gates.
6. Network, runtime, rate-limit, quota, validation, and stale-state failures are
   named distinctly and always retain a safe recovery or stop path.
7. Unchanged polling does not move focus, scroll, composer text, or timeline DOM.
8. Completion still requires executable evidence and independent final review.

## Verification evidence

- Focused workflow and browser suites pass, including the local-runner,
  sleep-sync, missing-session, and provider-transport regressions.
- Browser regressions exercise the real rendered workspace, approval lifecycle,
  drawer focus, timeline stability, quota/network boundaries, and responsive
  layouts.
- TUI regressions exercise keyboard and mouse-up activation for the same durable
  approval request.
- Targeted workflow, provider, storage, Web, browser, CLI, TUI, and ULTRA
  suites pass after the state/local-policy changes. A full repository run was
  started but exceeded the bounded verification window before producing a
  final count; it is not reported as a pass.
- Live project-128 recovery was inspected through the Web surface: the local
  boundary is named `Local model runner is unavailable`, Full Auto is shown as
  active, the composer says not to repeat the saved request, and `/retry`,
  `/resume`, and `/continue with local model` become available even when a
  legacy server response omits the provisional Goal row. The same live check
  now shows one `Recover this request` surface (the generic new-request
  attention panel is hidden), identifies `gemma4:e4b` as a local model in the
  model control, and leaves the recovery DOM unchanged across a polling tick.
- Local fallback ranking is covered by a regression that verifies a failed
  `gemma4:e4b` alias selects `qwen2.5-coder:7b` next rather than retrying the
  same descriptor indefinitely; the full-auto retry tests still prove that the
  first transient failure remains on the same model.
- The attached project-128 owner process was started at 16:03 and predates the
  repaired Python runtime. Its live Web process therefore cannot prove the
  post-restart backend path until that one durable owner is safely restarted;
  no active process was killed or replayed during this audit.
- A live provider smoke check now passes against the installed
  `qwen2.5-coder:7b`: Ollama exposes the local model, structured JSON is valid,
  and the controlled orchestration probe completes on the detected RTX 3060.
  The full project acceptance (one prompt through a durable Plan, real file
  changes, checks, and preview) remains a separate release gate and was not
  claimed from this smoke check alone.

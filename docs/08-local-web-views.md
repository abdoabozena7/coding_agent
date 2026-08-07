# Local Web Workspace

The local Web workspace is GA3BAD's primary visual control surface. A user can
start with one prompt, review and approve the plan, follow recursive execution,
change models at a safe checkpoint, review recorded changes, inspect the result,
and browse durable history without returning to the terminal. The terminal is a
compact fallback when the Web workspace is disconnected.
The Execution workspace opens once, immediately after the first project prompt
is accepted; it does not wait for planning to finish, so a local-model planning
failure still leaves every recovery and inspection control visible in the Web UI.
Opening is gated on the durable semantic turn or Goal becoming visible, so the
browser never presents a second blank intake while the terminal prompt is already
being accepted. Before the first Plan revision exists, Plan projects that same
Goal as a preparing state with the original request and explicitly marks the
composer as controls/follow-up only. Exact duplicates of the active request or
an existing queued follow-up are rejected idempotently by the server.

## Product flow

```text
Prompt -> reviewed Plan -> Execution tree -> Change Review -> Result
   |            |                |                |             |
 model      edit/revise       pause/resume      file/hunk     preview,
 picker     then approve      retry/stop         feedback      artifacts
```

The fixed composer is the only primary request input. While work is active, new
text is queued without interrupting the current stage. Typing `/` opens commands
that are enabled only when valid for the current durable state. Simple mode keeps
the operational workflow concise; Advanced exposes IDs, contracts, logs, memory,
raw diffs, and provider diagnostics. Direct commands include `/tree`, `/agents`,
`/tools`, `/activity`, `/status`, `/refresh`, `/continue with local model`, and
the main workflow controls.

## State authority

`StateStore` remains the durable SQLite authority for sessions, goals, immutable
plan revisions, approvals, Ultra runs, recursive work nodes, Change Sets,
artifacts, and audit events. `AgentRuntime` owns execution and safe checkpoint
rules. `CoreWebAdapter` projects those objects into secret-free browser payloads;
the browser never mutates repository files directly.

```text
Browser workspace <-> CoreWebAdapter <-> AgentRuntime / StateStore
       ^                                        |
       +--------------- SSE events -------------+
```

Polling is a recovery path. Server-sent events update compact live activity
without replacing an in-progress edit, moving focus, or changing scroll position.
The compact timeline keeps keyed rows in place and animates only a genuinely new
event, so heartbeat and clock refreshes cannot make unchanged rows flicker.
Out-of-order view responses are discarded so a slow request cannot overwrite the
view the user selected later.

One OS-level owner lease is held for each workspace session. It carries a
heartbeat and the authenticated Web endpoint, so launching the same project a
second time attaches to the existing workspace instead of starting a competing
provider loop or loopback server. If the owner process exits unexpectedly, the
OS releases the lock and the next launcher can take over; the durable workflow
checkpoint remains the source of truth.

## Workspace surfaces

- `/plan` — prompt intake, immutable plan revisions, model-fit explanation,
  explicit approval, and queued follow-ups.
- `/execution` — pause/resume/retry/stop controls, recursive node hierarchy,
  agent details, changed files, verified artifacts, and local preview receipts.
- `/review` — fixed Change Set, file/hunk decisions, inline feedback, and fixer
  activation when changes are requested.
- `/history` — durable milestones. Routine provider/heartbeat signals are
  collapsed by default and remain available through the Everything filter.
- `/agents`, `/tree`, and `/diff` remain compatibility aliases for Execution or
  Review and are canonicalized before rendering.

## Models and recovery

The model catalog is discovered only when the picker opens, so an offline local
provider cannot slow the main workspace poll. Switching calls the runtime's
authoritative replacement path and is allowed only at a saved safe checkpoint.
The approved plan and completed evidence are retained. A weaker model is not
presented as a false hard blocker: recursive execution is shown as the explicit
compensation, including the expected extra hand-offs.

`/continue with local model` is a deliberately narrower recovery path than a
normal model switch. It selects the strongest discovered tool-capable local
model. If its documented envelope is weaker than the model bound at approval,
the accepted plan, scope, acceptance criteria, verification, Quality Target,
and completion gates remain immutable while only the remaining work is projected
as smaller capability-bounded packets. Fresh executable evidence and independent
final evaluation remain mandatory; the runtime records the compensation policy
and exposes it through `/status`.

Errors carry stable codes. The workspace distinguishes offline browser state,
unreachable local runtime, expired session, stale state, invalid input, rate
limit, exhausted quota, and server failure. Recovery offers the action that can
actually help: retry, refresh, change model, inspect history, resume a saved
stage, or stop safely. Accepted workspace actions carry fingerprints so a retry
cannot apply the same decision twice.
When the active provider is already local, recovery never offers the misleading
"Switch to compatible local" action; Change model remains available and the
local-only picker is used only when recovering from a non-local provider.

Sleep is visible as one persistent control with three states. Off asks for every
required approval. Safe Auto continues reversible project checks, tests, and
previews. Full Auto is an explicit, typed-confirmation mode that accepts a
critic-reviewed plan, deterministic intake/plan/ULTRA questions (recommended,
explicit default, then first option), and every tool approval in the selected
workspace, writing a durable audit event for each decision. Conflicting or
malformed questions still pause rather than silently inventing a requirement.
A pending approval presents Allow once, Always allow this session, Deny, and
Stop safely. Session access is scoped to the displayed policy group, is audited
when reused, and resets when that workflow session ends.
The request is projected from durable `pending_tool_approval` state rather than
a bounded event window, so heartbeat noise cannot make the decision disappear.
Once a matching decision is accepted, the approval card is removed immediately;
accepted action events also reconcile stale cards left by an older browser state.

When Full Auto is enabled and a cloud provider reaches a quota, network, or
provider failure at a saved checkpoint, the controller automatically discovers
the strongest available local tool-capable model, switches safely, records a
`provider.local_fallback` event, and resumes the same stage. After plan approval
it uses the narrower local continuation contract: only remaining packets are
decomposed, while the accepted plan, quality target, executable evidence, and
independent final evaluation remain unchanged. If no compatible local model is
available, or the local provider also fails, the workflow pauses with an
explicit recovery action instead of retrying forever.

Terminal fallback follows a real browser SSE connection, not merely the lifetime
of the loopback server. If the Web tab disconnects, terminal approval keys become
available again; reconnecting the browser returns ownership to the Web surface.

## Result and preview safety

The Result surface exposes only recorded artifacts and Change Set paths. Preview
links are emitted only for loopback HTTP(S) hosts; credentials are rejected and
query strings/fragments are removed. External artifact URIs, provider secrets,
tool arguments, prompts, and hidden model reasoning never enter the payload.

## Security

- Loopback bind and Host validation.
- One high-entropy, process-lifetime handshake token tied to one session.
- SameSite Strict HTTP-only session cookie after the token redirect.
- Double-submit CSRF and local Origin checks for every mutation.
- Exact runtime-session checks for every route.
- `no-store`, frame denial, MIME-sniff protection, and a local-only CSP.
- Uvicorn access logging disabled; tokens are never written to access logs.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_local_web_views.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_local_web_views_browser.py -q
```

The suites cover immutable revision conflicts, approval authority, model
switching, error contracts, recursive execution, artifact sanitization, Change
Review/Fixer behavior, slash navigation, history noise filtering, responsive
layouts, SSE focus/scroll preservation, and the absence of fake progress.

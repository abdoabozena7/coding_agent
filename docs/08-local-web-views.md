# Local Web Views

GA3BAD remains terminal-first. The terminal owns conversation, goals, compact
events, errors, Stop/Resume, simple permissions, and view launchers. The local
web server renders only artifacts that need a larger visual editing surface.
There is no web chat or command line.

## Current architecture

`StateStore` is the durable SQLite authority for session state, goals, immutable
plan revisions, tasks, approvals, evidence, Ultra work nodes and agents, Change
Sets, memory, and audit events. `AgentRuntime` coordinates execution against that
store. `EventBus` fans compact runtime events to the terminal without letting the
terminal read browser state.

The local layer is an adapter over those objects:

```text
Terminal launchers and compact events
                  |
                  v
             EventBus
                  |
       +----------+----------+
       |                     |
 AgentRuntime / StateStore   LocalWebServer (127.0.0.1, random port)
       ^                     |
       +------ CoreWebAdapter+
                             |
                 Plan / Review / Agent Tree
```

The browser never edits repository files. Plan Apply calls the same plan and
approval authority used by the runtime. Review decisions update the real
immutable Change Set snapshot and activate its existing work node as a Fixer
when changes are requested. Agent Tree reads actual Ultra nodes, agent runs,
events, contracts, and memory references.

## Lifecycle and launchers

The CLI starts one FastAPI/Uvicorn server after creating `AgentRuntime`, binds an
already-reserved random socket to `127.0.0.1`, and stops it from
`AgentRuntime.close()`. Monitoring views do not block execution.

```text
/plan    -> Plan Studio
/review  -> Change Review
/agents  -> Agent Tree / Execution Map
```

Plan-review and checkpoint-review events may open their mandatory view
automatically. If the operating system browser launcher fails, GA3BAD prints the
one-time authenticated URL.

## Security

- Loopback bind and Host validation.
- One high-entropy, process-lifetime handshake token tied to one session.
- The token is exchanged for a SameSite Strict, HTTP-only session cookie and is
  removed from the address bar with a redirect.
- Mutations require a separate double-submit CSRF token and valid local Origin.
- Every route checks the exact runtime session ID; cross-session reads return
  not found.
- Responses are `no-store`, frame-denied, MIME-sniff protected, and restricted
  by a local-only Content Security Policy.
- Access tokens are never written to access logs; Uvicorn access logging is
  disabled.

## Plan Studio consistency

Opening Plan Studio captures an exact plan revision. Apply sends that base
revision to the backend. `StateStore.create_plan(...,
expected_parent_revision=N)` performs the comparison inside the same
`BEGIN IMMEDIATE` transaction that creates the next revision. A stale snapshot
therefore cannot win a race or overwrite a newer revision.

Backend validation covers task IDs, required acceptance criteria and tests,
roles, bounded fields, missing dependencies, duplicate IDs, self-dependencies,
and cycles. Existing applicability evidence and expected changes are retained;
new tasks receive explicit revision evidence and expected-output bindings. After
validation, Apply creates and activates the new immutable revision and emits a
compact `plan.revision.applied` event.

Drafts live in the existing workflow session state and never become the active
plan. Closing the tab performs no action.

## Change Review consistency

Change Review parses a fixed `ChangeSetV1.diff` snapshot. Decisions are accepted
only for files and hunk IDs present in that checkpoint. Rejections and change
requests require reasons. Comments remain associated with file, hunk, and line.
Closed or integrated checkpoints reject new submissions.

Submission appends an auditable review revision to the Change Set metadata,
updates file/hunk decision state, emits `review.submitted`, and moves the
responsible work node to `FIXING` with the user feedback when needed. Accepted
decisions remain in the immutable review record.

## Agent Tree updates

Agent Tree is read-only in the MVP and polls every two seconds. It shows the
core, work nodes, dependencies, assignments, statuses, retries, blockers, file
scope, recent logs, memory references, tools, and latest output. Request
Explanation writes an event for the selected real agent; it does not add a web
chat.

## Running and verification

Install the declared dependencies and start GA3BAD normally:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m agent --interactive
```

Focused backend and browser tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_local_web_views.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_local_web_views_browser.py -q
```

The browser tests cover adding and reordering tasks, Simple/Advanced modes,
Apply, stale-revision conflict UI, file and hunk decisions, inline comments,
review submission/Fixer activation, and the Agent inspector.

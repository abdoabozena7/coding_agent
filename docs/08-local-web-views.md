# Ultra Plan, Live, and Advanced Tracing

The local browser workspace is deliberately small. It helps users who do not
want to operate the whole workflow from the terminal, without becoming a second
control plane or duplicating runtime state.

There are three intentionally simple pages and one standalone developer page:

- `/plan` — planning conversation, editable plan document, first-layer preview,
  and explicit execution handoff.
- `/live` — read-only Overview, Agents tree, Timeline, and recorded diffs.
- `/show-diff` — read-only Workflow Diff with a changed-file rail, exact unified
  patches, live working changes, and every recorded workflow change set.
- `/advanced-tracing` — a read-only Index → Timeline → Inspector workspace
  with run selection, filtering, redacted JSON export, and frozen terminal snapshots.

Workflow Diff and Advanced Tracing are not tabs in Plan or Live, never open automatically, and
do not attach as browser control planes. They observe the existing SSE stream.

## Interaction modes

Ultra is terminal-first. Ultra Plan opens `/plan` automatically when selected.
Both modes always use recursive Ultra execution. Planning changes only the
pre-mutation boundary; it never downgrades execution to a staged coordinator.

```text
Ultra Plan selected
        |
        v
request -> material Q&A (repeat as needed) -> editable plan document
        -> Prepare agents -> exact first-layer preview
        -> Approve & start -> terminal handoff -> recursive Ultra execution
```

The second confirmation is intentionally separate. Preparing agents creates an
immutable plan revision and binds a fingerprinted first-layer preview to it. If
the document, revision, or team changes, approval is rejected until the team is
prepared again. Once accepted, the tab attempts to close; if the browser refuses,
a calm handoff screen explains that execution is already running in the terminal.

## Live projection

Live reads durable runtime state and updates through server-sent events with
polling as recovery:

- Overview shows the current operation, completed and remaining nodes, elapsed
  time, ETA only when authoritative, recent activity, changes, and model.
- Agents shows the recursive tree with simple status colors and a selected-node
  inspector for mission, attempts, files, elapsed time, and blockers.
- Timeline shows durable events and a Problems filter for failures, retries,
  approvals, and blockers.
- Recorded diffs open in a read-only drawer. Review and repair authority remains
  in the runtime.

Normal operation exposes no broad controls. At a real blocker, the page may show
only the action that can safely cross that boundary: Retry, Change model,
Allow/Deny, or Stop safely. The model catalog is loaded only when requested.

## Advanced Tracing projection

The left index selects Overview, Timeline, Files, Problems, Agents & Models,
Plans & Prompts, Context, or Changes. The center keeps a numbered chronological
trace; the right inspector opens selected evidence, diffs, file lifecycles,
agent attempts, or context decisions without leaving the page.

The projection fuses durable events, actions, plans, work nodes, all scheduled
and attempted agents, prompt traces, repository-context candidates, memory
access, artifacts, change sets, mutation records, quality findings, and scheduled
actions. “Inspect next” is derived only from open findings, failed reads,
unverified changed files, and reviewable diffs. Prompts and context remain
concealed until an explicit Reveal, and reveal never includes secrets or hidden
chain-of-thought.

Every Advanced Tracing section can be copied as redacted JSON. Individual
timeline records, IDs, paths, and diffs also have explicit copy actions so a
developer can hand the exact evidence to another analysis tool. Changes remain
time ordered: original writes and later repairs are separate records.

While a goal is active the page is `LIVE`. On completed, failed, blocked, or
cancelled termination the server stores a compressed, versioned snapshot with a
cutoff sequence and SHA-256 digest and the page becomes `FROZEN`. Resuming creates
a new revision without replacing earlier snapshots.

## Authority and synchronization

```text
Browser projection <- CoreWebAdapter <- AgentRuntime / StateStore <- terminal
                           ^                    |
                           +------ SSE ---------+
```

`StateStore` is the durable SQLite authority. `AgentRuntime` owns scheduling,
approvals, model switching, tool permissions, and safe checkpoints. The browser
never mutates repository files and never prevents terminal actions merely because
a tab is connected. SSE refreshes do not detach or overwrite an active plan edit.

## Security

- Loopback-only bind and Host validation.
- One process-lifetime handshake token tied to the runtime session.
- SameSite Strict HTTP-only session cookie after the token redirect.
- Double-submit CSRF and local Origin validation for every mutation.
- Exact runtime-session checks, `no-store`, frame denial, MIME-sniff protection,
  and a local-only CSP.
- Secrets, raw model reasoning, and credential-bearing artifact URLs never enter
  browser payloads.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_local_web_views.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_local_web_views_browser.py -q
```

The suites cover standalone three-page routing, terminal authority, plan/team
fingerprint conflicts, two-step approval, real-browser theme and responsive
layout, blocker controls, agent inspection, timeline filtering, read-only diffs,
SSE draft preservation, CSP, and console errors.

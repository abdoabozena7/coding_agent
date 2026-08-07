"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");
const lines = (value) => Array.isArray(value) ? value.join("\n") : "";
const parseLines = (value) => [...new Set(String(value || "").split("\n").map((x) => x.trim()).filter(Boolean))];

const pathParts = location.pathname.split("/").filter(Boolean);
const sessionId = pathParts[1] || "";
const canonicalView = (view) => ({ result: "execution", agents: "execution", tree: "execution", diff: "review" }[view] || view);
const initialView = ["thread", "plan", "result", "review", "agents", "execution", "history", "tree", "diff"].includes(pathParts[2])
  ? canonicalView(pathParts[2]) : "plan";
const csrf = decodeURIComponent(
  (document.cookie.split("; ").find((item) => item.startsWith("ga3bad_csrf=")) || "=").split("=").slice(1).join("="),
);

const state = {
  currentView: initialView,
  detail: localStorage.getItem("ga3bad-detail") === "advanced" ? "advanced" : "simple",
  context: null,
  plan: null,
  planModel: null,
  planDirty: false,
  editingTask: null,
  editingSummary: false,
  planConfirm: false,
  review: null,
  reviewExpanded: new Set(),
  reviewDecisions: new Map(),
  reviewComments: [],
  reviewConfirm: false,
  agents: null,
  history: null,
  thread: null,
  inspector: null,
  inspectorSection: "environment",
  inspectorOpen: localStorage.getItem("ga3bad-inspector") === "open",
  projectSessions: null,
  projectSessionsFingerprint: "",
  selectedNode: null,
  busy: false,
  draggedPromptId: null,
  contextTimer: null,
  eventSource: null,
  liveEvents: [],
  lastActivitySequence: 0,
  liveConnected: false,
  liveClock: null,
  viewFingerprints: {},
  pendingUpdates: {},
  planConflict: null,
  scrollingUntil: 0,
  viewLoadId: 0,
  modelCatalog: null,
  commandSelection: 0,
  resolvedApprovalFingerprint: "",
  drawerReturnFocus: null,
};

function statusLabel(status) {
  const labels = {
    pending: "Not started",
    ready: "Ready",
    in_progress: "Working",
    verifying: "Checking",
    reviewing: "Checking changes",
    testing: "Checking",
    fixing: "Fixing",
    integrating: "Integrating",
    blocked: "Needs attention",
    failed: "Needs attention",
    uncertain: "Needs attention",
    completed: "Done",
    accepted: "Approved",
    pending_approval: "Awaiting approval",
    awaiting_approval: "Awaiting approval",
    routing: "Routing",
    planning: "Planning",
    working: "Working",
    paused: "Paused",
    retrying: "Retrying",
    generating: "Generating",
    processing_response: "Generating",
    network_unavailable: "Runner unavailable",
    runner_unreachable: "Runner unavailable",
    contract_invalid: "Contract needs repair",
    quota_exhausted: "Limit reached",
    network_offline: "Offline",
    reviewing: "Checking changes",
    waiting: "Waiting",
    waiting_for_approval: "Waiting",
    open: "Ready for review",
  };
  return labels[status] || String(status || "Unknown").replaceAll("_", " ");
}

function statusClass(status) {
  if (["completed", "accepted", "approved", "integrated"].includes(status)) return "done";
  if (["blocked", "failed", "uncertain", "changes_requested"].includes(status)) return "attention";
  if (["pending", "pending_approval", "open", "waiting", "waiting_for_approval"].includes(status)) return "waiting";
  if (["ready", "in_progress", "verifying", "reviewing", "testing", "fixing", "integrating", "running"].includes(status)) return "working";
  return "";
}

function savedWorkflowId(context = state.context) {
  return context?.goal?.id || context?.workflow_identity?.goal_id || null;
}

function hasSavedWorkflow(context = state.context) {
  if (!context) return false;
  if (context.goal || context.workflow_identity?.goal_id || context.provider_recovery) return true;
  const runtime = context.runtime || {};
  const phase = String(runtime.phase || context.phase || "").toLowerCase();
  return Boolean(
    runtime.objective
    && [
      "routing", "planning", "starting", "dispatching", "working", "retrying",
      "paused", "recovering", "reviewing", "verifying", "waiting", "waiting_for_approval",
    ].includes(phase),
  );
}

function badge(status) {
  return `<span class="status-badge status-${statusClass(status)}">${escapeHtml(statusLabel(status))}</span>`;
}

function compactList(items, emptyText = "No additional requirements were recorded.") {
  const values = (items || []).filter((item) => String(item || "").trim());
  if (!values.length) return `<p class="subtle">${escapeHtml(emptyText)}</p>`;
  return `<ul class="plain-list">${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function requirementAnchors(items) {
  const anchors = Array.isArray(items) ? items : [];
  if (!anchors.length) return "";
  return `
    <div class="requirement-anchors">
      <h4>What your words mean in the finished result</h4>
      <ul class="plain-list">
        ${anchors.map((anchor) => `
          <li>
            <strong>“${escapeHtml(anchor.verbatim_span || anchor.id)}”</strong>
            — ${escapeHtml(anchor.interpreted_requirement || "Required as written")}
            ${compactList(anchor.observable_implications, "")}
          </li>`).join("")}
      </ul>
    </div>`;
}

function executionPresentation(plan) {
  const recursive = plan.execution_strategy === "recursive";
  const decision = plan.strategy_decision || {};
  const modelFit = plan.model_fit || {};
  const envelope = plan.capability_envelope || {};
  const nodes = plan.execution_nodes || [];
  const topLevel = nodes.filter((node) => !node.parent_id);
  const plannedCount = recursive ? (topLevel.length || plan.tasks.length) : plan.tasks.length;
  const label = recursive ? "Recursive specialists" : "Staged coordinator";
  const explanation = recursive
    ? `${plannedCount} top-level node${plannedCount === 1 ? "" : "s"} are planned. Each node can split into narrower specialists; integration and independent review remain explicit gates.`
    : `One coordinator will run ${plannedCount} dependency-ordered step${plannedCount === 1 ? "" : "s"}. Specialist agents are not created for this plan.`;
  const nodeRows = recursive && nodes.length
    ? `<ol class="blueprint-list">${nodes.map((node) => `
        <li class="blueprint-node${node.parent_id ? " child" : ""}">
          <div><strong>${escapeHtml(node.title)}</strong><span>${escapeHtml(node.assigned_role || "Focused specialist")}</span></div>
          <p>${escapeHtml(node.objective || "Bounded specialist outcome")}</p>
        </li>`).join("")}</ol>`
    : "";
  return `
    <section class="section execution-plan">
      <div class="section-head">
        <div>
          <h3>How the system will execute this plan</h3>
          <p>Selected automatically from this request and the chosen model. This is read-only.</p>
        </div>
        <span class="strategy-label">${escapeHtml(label)}</span>
      </div>
      <p class="reading compact">${escapeHtml(explanation)}</p>
      <div class="model-fit model-fit-${escapeHtml(modelFit.status || "unknown")}">
        <span class="status-badge ${modelFit.status === "fit" ? "status-done" : "status-working"}">${modelFit.status === "fit" ? "Model fit" : "Fit compensated"}</span>
        <p>${escapeHtml(modelFit.summary || "The harness will adapt execution depth to the selected model.")}</p>
        <button class="quiet-button" data-open-model-picker type="button">Compare models</button>
      </div>
      ${compactList(decision.reasons, "The capability policy did not record an additional reason.")}
      ${nodeRows}
      <div class="meta-line advanced-only">
        <span>Model: ${escapeHtml(envelope.model || "unknown")}</span>
        <span>Capability: ${escapeHtml(envelope.capability_band || "minimal")}</span>
        <span>Concurrency: ${escapeHtml(decision.max_concurrency || envelope.max_concurrency || 1)}</span>
        <span>${plan.strategy_locked ? "Locked at approval" : "Provisional until approval"}</span>
      </div>
      ${plan.capabilities.can_increase_depth ? `<div class="section-actions advanced-only"><button class="secondary-button" data-action="increase-depth" type="button">Increase execution depth</button></div>` : ""}
    </section>`;
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    headers["X-GA3BAD-CSRF"] = csrf;
  }
  let response;
  try {
    response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}${path}`, {
      credentials: "same-origin",
      ...options,
      headers,
    });
  } catch (cause) {
    const error = new Error("The local workspace is offline.");
    error.code = navigator.onLine ? "runtime_unreachable" : "network_offline";
    error.cause = cause;
    throw error;
  }
  let payload = {};
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    error.code = payload.code || (
      response.status === 401 ? "session_expired"
        : response.status === 403 ? "permission_denied"
          : response.status === 404 ? "not_found"
            : response.status === 409 ? "stale_state"
              : response.status === 422 ? "invalid_request"
                : response.status === 429 ? (payload.quota_exhausted ? "quota_exhausted" : "rate_limited")
                  : response.status >= 500 ? "runtime_error" : "request_failed"
    );
    error.retryAfter = response.headers.get("Retry-After") || payload.retry_after || "";
    throw error;
  }
  return payload;
}

function announce(message, isError = false) {
  $("#liveRegion").textContent = message;
  $$(".toast", $("#toastRegion")).forEach((item) => item.remove());
  const toast = document.createElement("div");
  toast.className = `toast${isError ? " error" : ""}`;
  toast.textContent = message;
  $("#toastRegion").append(toast);
  setTimeout(() => toast.remove(), 4400);
}

function setBusy(busy) {
  state.busy = busy;
  $("#app").setAttribute("aria-busy", String(busy));
  $$("button").forEach((button) => {
    if (button.dataset.keepEnabled !== "true") button.disabled = busy;
  });
}

function reconcileResolvedApproval(context) {
  if (!context || !state.resolvedApprovalFingerprint) return context;
  const visibleFingerprint = String(context.tool_approval?.action_fingerprint || "");
  if (!visibleFingerprint) {
    state.resolvedApprovalFingerprint = "";
    return context;
  }
  if (visibleFingerprint !== state.resolvedApprovalFingerprint) {
    state.resolvedApprovalFingerprint = "";
    return context;
  }
  return {
    ...context,
    tool_approval: null,
    required_action: null,
    waiting_on: "tool",
    attention: {
      state: "working",
      eyebrow: "Approval recorded",
      title: "Resuming the workflow",
      body: "The accepted action is no longer waiting for input.",
      action: null,
    },
  };
}

function markApprovalResolved(actionFingerprint) {
  const fingerprint = String(actionFingerprint || "");
  if (!fingerprint || state.context?.tool_approval?.action_fingerprint !== fingerprint) return;
  state.resolvedApprovalFingerprint = fingerprint;
  state.context = reconcileResolvedApproval(state.context);
  state.pendingUpdates = {};
  renderChrome();
}

function autoGrow(target) {
  const items = target ? [target] : $$("textarea");
  items.forEach((area) => {
    // Chromium uses CSS field-sizing: content. Touching the value forces a
    // layout pass without introducing CSP-blocked inline style attributes.
    area.rows = Math.max(1, String(area.value || "").split("\n").length);
  });
}

function showError(error) {
  $("#loading").classList.add("hidden");
  $("#viewRoot").classList.add("hidden");
  const root = $("#errorState");
  const code = String(error.code || "runtime_error");
  const presentation = {
    network_offline: ["You are offline", "Reconnect to the internet. Your saved workflow has not changed.", "Offline"],
    local_runner_unreachable: ["The local model runner is unavailable", "Start Ollama or reconnect the configured local runner. The exact stage is saved and no workspace change was replayed.", "Local runner unavailable"],
    runtime_unreachable: ["The local runtime is not responding", "Keep this page open. Retry after the GA3BAD process is available.", "Runtime unavailable"],
    session_expired: ["This workspace session expired", "Open the workspace again from GA3BAD to create a fresh secure session.", "Session expired"],
    stale_state: ["The workflow changed before this action finished", "Refresh the latest durable state, then try the action again.", "State conflict"],
    rate_limited: ["The model is temporarily rate limited", error.retryAfter ? `Try again after ${error.retryAfter}. The saved stage is unchanged.` : "Wait briefly or switch to another available model.", "Rate limited"],
    quota_exhausted: ["This model has reached its usage limit", "Switch to another model or wait for the provider limit to reset.", "Limit reached"],
    invalid_request: ["That action could not be applied", "Check the highlighted input and try again. No workspace change was accepted.", "Invalid action"],
  }[code] || ["The workspace could not load", "Your durable work was not changed. Retry or inspect another workspace view.", "Runtime error"];
  root.classList.remove("hidden");
  root.innerHTML = `
    <div>
      <p class="eyebrow">${escapeHtml(presentation[2])}</p>
      <h2>${escapeHtml(presentation[0])}</h2>
      <p>${escapeHtml(presentation[1])}</p>
      <p class="error-code advanced-only">${escapeHtml(code)}${state.detail === "advanced" && error.message ? ` · ${escapeHtml(error.message)}` : ""}</p>
      <div class="error-actions">
        <button class="primary-button" id="retryLoad" type="button">Retry</button>
        ${["rate_limited", "quota_exhausted"].includes(code) ? '<button class="secondary-button" id="errorChangeModel" type="button">Change model</button>' : ""}
        <button class="secondary-button" id="errorOpenHistory" type="button">Open history</button>
      </div>
    </div>`;
  $("#retryLoad").addEventListener("click", () => loadView(state.currentView, true));
  $("#errorChangeModel")?.addEventListener("click", openModelPicker);
  $("#errorOpenHistory")?.addEventListener("click", () => navigate("history"));
  setConnection(false, presentation[2]);
}

function setConnection(connected, label = "") {
  const node = $("#connection");
  node.classList.toggle("connected", connected);
  node.classList.toggle("offline", !connected);
  $("span", node).textContent = label || (connected ? "Live" : "Reconnecting");
}

function compactBytes(value) {
  const size = Math.max(0, Number(value || 0));
  if (size < 1024) return `${Math.round(size)} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function activityTime(value) {
  const date = new Date(value || "");
  return Number.isNaN(date.getTime()) ? "--:--:--" : date.toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

function appendLiveEvent(event) {
  if (!event || typeof event !== "object") return;
  const sequence = Number(event.sequence || 0);
  if (sequence && state.liveEvents.some((item) => Number(item.sequence) === sequence)) return;
  state.liveEvents.push(event);
  state.liveEvents = state.liveEvents.slice(-80);
  state.lastActivitySequence = Math.max(state.lastActivitySequence, sequence);
}

function liveEventKey(event, index) {
  const sequence = Number(event?.sequence || 0);
  if (sequence) return `sequence-${sequence}`;
  return `event-${event?.timestamp || ""}-${event?.source || ""}-${event?.message || event?.operation || ""}-${index}`;
}

function renderLiveTimeline() {
  const events = state.liveEvents.slice(-3);
  const root = $("#liveTimeline");
  const visibleEvents = events.length ? events : [{ source: "HARNESS", message: "Waiting for the first runtime event", placeholder: true }];
  const existing = new Map($$(".live-event", root).map((node) => [node.dataset.eventKey, node]));
  visibleEvents.forEach((event, index) => {
    const source = String(event.source || "HARNESS").toUpperCase();
    const key = liveEventKey(event, index);
    let row = existing.get(key);
    if (!row) {
      row = document.createElement("li");
      row.dataset.eventKey = key;
      row.innerHTML = '<time></time><span class="event-source"></span><span class="event-message"></span>';
      row.classList.add("is-new");
      row.addEventListener("animationend", () => row.classList.remove("is-new"), { once: true });
    }
    const nextClass = `live-event source-${source.toLowerCase()}${row.classList.contains("is-new") ? " is-new" : ""}`;
    if (row.className !== nextClass) row.className = nextClass;
    const time = $("time", row);
    const timestamp = String(event.timestamp || "");
    const displayTime = event.placeholder ? "" : activityTime(event.timestamp);
    const sourceNode = $(".event-source", row);
    const messageNode = $(".event-message", row);
    const rawMessage = event.message || event.operation || "";
    const message = rawMessage
      ? publicThreadRecoveryReason({ error: rawMessage })
      : "Activity received";
    if (time.dateTime !== timestamp) time.dateTime = timestamp;
    if (time.textContent !== displayTime) time.textContent = displayTime;
    if (sourceNode.textContent !== source) sourceNode.textContent = source;
    if (messageNode.textContent !== message) messageNode.textContent = message;
    const rowAtPosition = root.children[index] || null;
    if (rowAtPosition !== row) root.insertBefore(row, rowAtPosition);
    existing.delete(key);
  });
  existing.forEach((row) => row.remove());
}

function renderAdvancedLiveDetails() {
  const runtime = state.context?.runtime || {};
  const tasks = Array.isArray(runtime.task_items) ? runtime.task_items : [];
  const taskRoot = $("#advancedTaskList");
  if (!taskRoot) return;
  const currentId = String(runtime.current_task_id || "");
  const done = new Set(["done", "completed", "complete", "skipped"]);
  taskRoot.innerHTML = tasks.length ? tasks.map((task) => {
    const status = String(task.status || "pending").toLowerCase();
    const current = currentId && String(task.id || "") === currentId;
    const cls = [
      "task-check-item",
      current ? "is-current" : "",
      done.has(status) ? "is-done" : "",
      ["blocked", "uncertain", "failed"].includes(status) ? "is-blocked" : "",
    ].filter(Boolean).join(" ");
    const mark = done.has(status) ? "[x]" : status === "in_progress" || status === "running" ? "[>]" : status === "blocked" || status === "uncertain" ? "[!]" : "[ ]";
    const evidence = task.evidence ? `<small>${escapeHtml(task.evidence)}</small>` : "";
    return `<li class="${cls}"><span class="task-status">${mark} ${escapeHtml(status)}</span><p>${escapeHtml(task.title || task.id || "Task")}${evidence}</p></li>`;
  }).join("") : `<li class="empty-state">No authoritative task checklist is available yet.</li>`;

  const streamState = String(runtime.stream_state || runtime.liveness || "idle");
  const streamKind = String(runtime.stream_kind || "none");
  $("#advancedStreamState").textContent = streamState.replaceAll("_", " ");
  const output = $("#advancedModelOutput");
  if (runtime.safe_stream_preview) {
    output.textContent = runtime.safe_stream_preview;
  } else if (["structured", "tool"].includes(streamKind) || ["active", "validating"].includes(streamState)) {
    output.textContent = "Structured response · text hidden until validation";
  } else {
    output.textContent = "No safe text stream available.";
  }
  $("#advancedTransport").textContent = `Transport · ${compactBytes(Number(runtime.received_bytes || 0))} · ${Number(runtime.received_chunks || 0)} chunks · ${Number(runtime.received_tokens || 0)} tokens`;
  const activityRoot = $("#advancedActivityList");
  if (!activityRoot) return;
  const events = state.liveEvents.slice(-24).reverse();
  if (!events.length) {
    if (!activityRoot.querySelector(".empty-state")) {
      activityRoot.replaceChildren(
        Object.assign(document.createElement("li"), {
          className: "empty-state",
          textContent: "No runtime events have been received yet.",
        }),
      );
    }
    return;
  }
  const existing = new Map(
    $$(".advanced-activity-item", activityRoot).map((node) => [node.dataset.eventKey, node]),
  );
  events.forEach((event, index) => {
    const key = liveEventKey(event, index);
    let row = existing.get(key);
    if (!row) {
      row = document.createElement("li");
      row.className = "advanced-activity-item";
      row.dataset.eventKey = key;
      row.innerHTML = "<time></time><p><strong></strong><span></span></p>";
    }
    const time = $("time", row);
    const source = $("strong", row);
    const message = $("span", row);
    const timestamp = String(event.timestamp || "");
    const sourceText = String(event.source || "HARNESS");
    const rawMessage = event.message || event.operation || "";
    const messageText = ` · ${rawMessage ? publicThreadRecoveryReason({ error: rawMessage }) : "Activity received"}`;
    if (time.dateTime !== timestamp) time.dateTime = timestamp;
    const displayTime = activityTime(timestamp);
    if (time.textContent !== displayTime) time.textContent = displayTime;
    if (source.textContent !== sourceText) source.textContent = sourceText;
    if (message.textContent !== messageText) message.textContent = messageText;
    const rowAtPosition = activityRoot.children[index] || null;
    if (rowAtPosition !== row) activityRoot.insertBefore(row, rowAtPosition);
    existing.delete(key);
  });
  existing.forEach((row) => row.remove());
}

function renderLiveWorkflow() {
  const runtime = state.context?.runtime || {};
  const root = $("#liveWorkflow");
  const needsApproval = Boolean(state.context?.tool_approval?.action_fingerprint);
  const savedWorkflow = hasSavedWorkflow(state.context);
  const liveness = String(runtime.liveness || (runtime.phase === "ready" ? "ready" : "client_active"));
  root.classList.toggle("is-active", !needsApproval && ["client_active", "request_sent", "request_created", "provider_connected", "server_processing", "receiving", "processing_response"].includes(liveness));
  root.classList.toggle("is-receiving", liveness === "receiving");
  root.classList.toggle("is-stalled", ["stalled", "network_unavailable", "disconnected"].includes(liveness));
  const phase = String(runtime.phase || state.context?.phase || "ready").replaceAll("_", " ");
  const goalStatus = String(state.context?.goal?.status || "");
  const localRunner = String(runtime.execution_class || "").toLowerCase() === "local";
  const failureKind = String(
    runtime.failure_kind
      || state.context?.workflow_identity?.failure_kind
      || "",
  ).toLowerCase();
  const diagnosticText = [runtime.reason, runtime.last_error, runtime.provider_error, runtime.active_operation].filter(Boolean).join(" ").toLowerCase();
  const providerIssue = failureKind === "quota" || /quota|usage limit|limit exhausted/.test(diagnosticText) ? "quota_exhausted"
    : failureKind === "rate_limit" || /rate limit|too many requests|429/.test(diagnosticText) ? "rate_limited"
      : failureKind === "transport" || /offline|network|unreachable|connection refused|timed out/.test(diagnosticText) ? "network_unavailable" : "";
  const providerRecovery = state.context?.provider_recovery || {};
  const recoveryState = String(providerRecovery.state || "").toLowerCase();
  const recoveryNeedsAttention = Object.keys(providerRecovery).length > 0 && (
    ["paused", "waiting", "retrying", "recovering", "failed"].includes(recoveryState)
    || ["paused", "retrying", "recovering", "blocked"].includes(String(phase).toLowerCase())
    || /submit_semantic_route|submit_semantic_turn|exactly once|could not be validated/.test(diagnosticText)
  );
  const retryNotBefore = Number(
    providerRecovery.retry_not_before
      || runtime.retry_not_before
      || 0,
  );
  const fullAutoRecovery = String(state.context?.sleep_policy || runtime.sleep_policy || "").toLowerCase() === "full"
    && (
      Boolean(providerRecovery.full_auto_retry)
      || ["retrying", "waiting"].includes(String(phase).toLowerCase())
    );
  const title = needsApproval ? "Waiting for your approval"
      : goalStatus === "awaiting_plan_approval" ? "Plan ready to start"
      : phase === "ready" && !savedWorkflow ? "What do you want to build?"
      : state.context?.required_view === "review" && state.context?.attention?.state === "waiting" ? "Changes need your attention"
    : providerIssue === "quota_exhausted" ? "This model has reached its usage limit"
      : providerIssue === "rate_limited" ? "This model is temporarily rate limited"
        : liveness === "network_unavailable" ? (localRunner ? "Local model runner is unavailable" : "Internet or provider is unavailable")
        : recoveryNeedsAttention ? "Saved request needs recovery"
          : liveness === "paused" ? "Paused at a safe checkpoint"
          : liveness === "waiting" ? "Waiting for you"
            : liveness === "completed" ? "Result ready"
              : liveness === "receiving" ? "The model is responding"
                : liveness === "processing_response" ? "Checking the model response"
                  : liveness === "server_processing" ? "The model is working"
                    : runtime.current_task || "Working on your request";
  $("#liveNowTitle").textContent = title;
  const bytes = Number(runtime.received_bytes || 0);
  const chunks = Number(runtime.received_chunks || 0);
  const tokens = Number(runtime.received_tokens || 0);
  const signalAge = runtime.last_signal_at ? Math.max(0, Math.floor(Date.now() / 1000 - Number(runtime.last_signal_at))) : null;
  let evidence = savedWorkflow
    ? (runtime.reason || "The durable workflow is ready at its current checkpoint.")
    : "No workflow is active.";
  if (needsApproval) evidence = "Choose Allow once, Always allow this session, Deny, or Stop safely below";
  else if (providerIssue === "quota_exhausted") evidence = "Your checkpoint is saved · change model or wait for the provider limit to reset";
  else if (providerIssue === "rate_limited") evidence = "Your checkpoint is saved · retry later or change model";
  else if (liveness === "server_processing") evidence = "The provider is actively generating · your saved stage is unchanged";
  else if (liveness === "network_unavailable") evidence = localRunner
    ? "Local model runner unavailable · saved stage unchanged"
    : "Internet/provider unavailable · saved stage unchanged";
  else if (liveness === "request_created") evidence = "Request created · no response bytes yet";
  else if (liveness === "request_sent") evidence = "Request open · no response bytes yet";
  else if (liveness === "receiving") evidence = `Receiving model output · ${compactBytes(bytes)} · ${chunks} chunks${tokens ? ` · ${tokens} tokens` : ""}`;
  else if (liveness === "processing_response") evidence = `Response received · ${compactBytes(bytes)} · validating structured output`;
  else if (liveness === "client_active") evidence = "Worker heartbeat active · waiting for the next verified signal";
  else if (liveness === "stalled") evidence = signalAge !== null && signalAge <= 5
    ? "A boundary was recorded · the harness is checking the saved workflow"
    : "No fresh runtime signal · inspect the saved workflow before waiting longer";
  else if (liveness === "disconnected") evidence = "Live connection lost · saved state is unchanged · reconnecting";
  else if (liveness === "paused") evidence = "Paused at a durable checkpoint";
  else if (liveness === "waiting") evidence = runtime.reason || "Your input is required";
  else if (liveness === "completed") evidence = "Workflow completed with recorded evidence";
  if (!needsApproval && recoveryNeedsAttention && !fullAutoRecovery && !providerIssue) evidence = "The original request is preserved · recovery resumes the saved checkpoint";
  if (!needsApproval && signalAge !== null && !["ready", "completed"].includes(liveness)) evidence += ` · last signal ${signalAge}s ago`;
  $("#liveEvidence").textContent = evidence;
  const goal = runtime.objective || state.context?.goal?.objective || "";
  $("#liveGoal").textContent = goal ? `Goal · ${goal}${runtime.current_task ? ` · Task · ${runtime.current_task}` : ""}` : "";
  $("#liveGoal").classList.toggle("hidden", !goal);
  const autoRecovery = $("#liveAutoRecovery");
  if (autoRecovery) {
    const retrySeconds = retryNotBefore > (Date.now() / 1000)
      ? Math.max(1, Math.ceil(retryNotBefore - (Date.now() / 1000)))
      : 0;
    autoRecovery.textContent = fullAutoRecovery
      ? `Full Auto is active · no action is required${retrySeconds ? ` · next retry in about ${retrySeconds}s` : ""}. Quality gates and the exact saved request stay unchanged.`
      : "";
    autoRecovery.classList.toggle("hidden", !fullAutoRecovery);
  }
  $("#liveProvenance").textContent = liveness === "receiving" ? "Model" : "Harness";
  renderLiveTimeline();
}

function renderChrome() {
  document.body.classList.toggle("advanced", state.detail === "advanced");
  $("#simpleMode").classList.toggle("selected", state.detail === "simple");
  $("#advancedMode").classList.toggle("selected", state.detail === "advanced");
  $$(".nav-button").forEach((button) => {
    const buttonView = canonicalView(button.dataset.view);
    button.classList.toggle("selected", buttonView === state.currentView);
    button.setAttribute("aria-current", buttonView === state.currentView ? "page" : "false");
  });
  if (!state.context) {
    renderInspector();
    updateComposer();
    return;
  }
  const embeddedProjectSessions = state.context.project_sessions;
  if (embeddedProjectSessions && projectSessionsFingerprint(embeddedProjectSessions) !== state.projectSessionsFingerprint) {
    state.projectSessionsFingerprint = projectSessionsFingerprint(embeddedProjectSessions);
    renderProjectSessions(embeddedProjectSessions);
  } else if (!state.projectSessions) {
    const goal = state.context.goal;
    const fallbackSessions = {
      session_id: sessionId,
      projects: [{ id: sessionId, name: "GA3BAD", active: true }],
      tasks: goal ? [{ id: goal.id, title: goal.objective, status: goal.status, active: true }] : [],
      pinned: [],
      archived: [],
    };
    state.projectSessionsFingerprint = projectSessionsFingerprint(fallbackSessions);
    renderProjectSessions(fallbackSessions);
  }
  const runtime = state.context.runtime || {};
  const mode = String(runtime.session_mode || state.context.mode || "ready").replaceAll("_", " ");
  const route = String(runtime.route || state.context.route || "pending");
  const strategy = String(runtime.execution_strategy || state.context.execution_strategy || "pending");
  const phase = String(runtime.phase || state.context.phase || "ready").replaceAll("_", " ");
  const model = runtime.model ? ` · ${runtime.provider || "model"}/${runtime.model}` : "";
  $("#sessionMeta").textContent = `Session ${mode} · Route ${route} · Strategy ${strategy} · Status ${phase}${model}`;
  $("#modelButtonLabel").textContent = runtime.model || "Choose model";
  $("#modelComposer").textContent = runtime.model ? `Model · ${runtime.model}` : "Model";
  const executionClass = String(runtime.execution_class || "").toLowerCase();
  const modelButton = $("#modelButton");
  if (runtime.model) {
    const classLabel = executionClass === "local" ? "local model" : executionClass === "cloud" ? "cloud model" : "model";
    modelButton.setAttribute("aria-label", `Model ${runtime.model} (${classLabel})`);
    modelButton.title = `${runtime.provider || "Provider"}/${runtime.model} · ${classLabel}`;
  } else {
    modelButton.setAttribute("aria-label", "Choose model");
    modelButton.removeAttribute("title");
  }
  $("#workflowIdentity").textContent = hasSavedWorkflow(state.context)
    ? `${statusLabel(String(runtime.phase || state.context.goal?.status || "active").toLowerCase())} · plan r${state.context.goal?.plan_revision || "—"}`
    : "No active goal";
  const sleepButton = $("#sleepToggle");
  const sleepPolicy = String(state.context.sleep_policy || (state.context.sleep_enabled ? "safe" : "off"));
  const sleepLabel = sleepPolicy === "full" ? "Sleep: Full Auto"
    : sleepPolicy === "safe" ? "Sleep: Safe Auto" : "Sleep: Off";
  sleepButton.innerHTML = `<span class="sleep-indicator" aria-hidden="true">${sleepPolicy === "full" ? "FULL" : sleepPolicy === "safe" ? "SAFE" : "AUTO"}</span><span>${sleepLabel}</span>`;
  sleepButton.setAttribute("aria-pressed", String(Boolean(state.context.sleep_enabled)));
  sleepButton.dataset.policy = sleepPolicy;
  const sleepTopbar = $("#sleepTopbar");
  sleepTopbar.querySelector(".sleep-indicator").textContent = sleepPolicy === "full" ? "FULL" : sleepPolicy === "safe" ? "SAFE" : "AUTO";
  sleepTopbar.setAttribute("aria-label", `Sleep mode: ${sleepPolicy === "full" ? "Full Auto" : sleepPolicy === "safe" ? "Safe Auto" : "Off"}`);
  sleepTopbar.dataset.policy = sleepPolicy;
  $("#sleepComposer").textContent = sleepPolicy === "full" ? "Sleep · Full" : sleepPolicy === "safe" ? "Sleep · Safe" : "Sleep";
  $("#accessButton").textContent = `Access · ${String(state.context.runtime?.access_level || "normal")}`;
  $("#modeButton").textContent = `Mode · ${String(state.context.mode || "working") === "plan" ? "Plan" : "Working"}`;
  Object.entries(state.context.navigation || {}).forEach(([view, item]) => {
    const node = $(`[data-badge="${canonicalView(view)}"]`);
    if (!node) return;
    const count = Number(item.badge || 0);
    node.textContent = String(count);
    node.classList.toggle("hidden", count === 0);
  });
  const attention = state.context.attention;
  const legacyPreGoalCheckpoint = hasSavedWorkflow(state.context)
    && !state.context.goal
    && !state.context.workflow_identity?.goal_id;
  const root = $("#attention");
  root.classList.remove("hidden");
  $("#attentionEyebrow").textContent = attention.eyebrow;
  $("#attentionTitle").textContent = attention.title;
  $("#attentionBody").textContent = attention.body;
  const action = $("#attentionAction");
  const actionTargetsCurrentView = Boolean(
    attention.action
    && !attention.action.kind
    && canonicalView(attention.action.view) === state.currentView
  );
  action.classList.toggle("hidden", !attention.action || actionTargetsCurrentView);
  if (attention.action) {
    action.textContent = attention.action.label;
    action.dataset.view = attention.action.view;
    action.dataset.kind = attention.action.kind || "";
  } else {
    action.dataset.view = "";
    action.dataset.kind = "";
  }
  const toolApproval = state.context.tool_approval;
  const toolActions = $("#toolApprovalActions");
  const allowButton = $("#toolAllow");
  const allowSessionButton = $("#toolAllowSession");
  const denyButton = $("#toolDeny");
  const hasToolApproval = Boolean(toolApproval && toolApproval.action_fingerprint);
  root.classList.toggle("approval-attention", hasToolApproval);
  toolActions.classList.toggle("hidden", !hasToolApproval);
  action.classList.toggle(
    "hidden",
    !attention.action || hasToolApproval || actionTargetsCurrentView,
  );
  if (hasToolApproval) {
    const args = toolApproval.arguments || {};
    const command = args.command || args.path || "the requested project action";
    $("#attentionBody").textContent = `${attention.body} Tool: ${toolApproval.tool}. Risk: ${toolApproval.risk}. Target: ${String(command).slice(0, 800)}`;
    allowButton.disabled = Boolean(state.busy);
    allowSessionButton.disabled = Boolean(state.busy);
    denyButton.disabled = Boolean(state.busy);
    allowButton.dataset.fingerprint = toolApproval.action_fingerprint;
    allowSessionButton.dataset.fingerprint = toolApproval.action_fingerprint;
    denyButton.dataset.fingerprint = toolApproval.action_fingerprint;
    const group = String(toolApproval.policy_group || "matching actions").replaceAll("_", " ");
    $("#toolSessionScope").textContent = `Session access applies to: ${group}. It resets when this session ends.`;
  }
  const required = state.context.required_action;
  const requiredRoot = $("#requiredAction");
  const hasPrimaryAttention = Boolean(attention?.action || hasToolApproval);
  requiredRoot.classList.toggle("hidden", !required || hasPrimaryAttention);
  if (required && !hasPrimaryAttention) {
    $("#requiredActionLabel").textContent = required.label || "Action required";
    $("#requiredActionDescription").textContent = `${required.description || ""} ${required.consequence || ""}`.trim();
    const button = $("#requiredActionButton");
    button.textContent = required.label || "Continue";
    button.dataset.actionKind = required.kind || "";
    button.dataset.fingerprint = required.fingerprint || "";
    const alternatives = $("#requiredActionAlternatives");
    const alternativeItems = Array.isArray(required.alternatives) ? required.alternatives : [];
    alternatives.innerHTML = alternativeItems.map((item) => `<button class="secondary-button" data-required-alternative="${escapeHtml(item.kind || "")}" type="button">${escapeHtml(item.label || item.kind || "Option")}</button>`).join("");
    alternatives.classList.toggle("hidden", !alternativeItems.length);
    $$('[data-required-alternative]', alternatives).forEach((alternative) => alternative.addEventListener("click", () => {
      if (alternative.dataset.requiredAlternative === "switch_model") openModelPicker();
      else runWorkspaceAction(alternative.dataset.requiredAlternative);
    }));
    const input = $("#requiredActionInput");
    if (required.kind === "answer" && required.question) {
      const questionId = String(required.question.id || "");
      if (document.activeElement?.id === "requiredAnswer" && input.dataset.questionId === questionId) {
        // Keep the user's in-progress answer and caret while context polling
        // updates the surrounding runtime snapshot.
        input.classList.remove("hidden");
      } else {
      const existingAnswer = $("#requiredAnswer")?.value || "";
      const options = (required.question.options || []).map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label || option.value)}</option>`).join("");
      input.innerHTML = `
        <label for="requiredAnswer">Your answer</label>
        ${options ? `<select id="requiredAnswerSelect"><option value="">Choose an answer</option>${options}</select>` : ""}
        <textarea id="requiredAnswer" rows="1" placeholder="Answer the question in your own words">${escapeHtml(existingAnswer)}</textarea>`;
      input.dataset.questionId = questionId;
      input.classList.remove("hidden");
      autoGrow($("#requiredAnswer"));
      $("#requiredAnswer")?.addEventListener("input", (event) => autoGrow(event.currentTarget));
      $("#requiredAnswerSelect")?.addEventListener("change", (event) => {
        const answer = $("#requiredAnswer");
        if (answer && event.target.value) {
          answer.value = event.target.value;
          autoGrow(answer);
        }
      });
      }
    } else {
      input.innerHTML = "";
      input.dataset.questionId = "";
      input.classList.add("hidden");
    }
  } else {
    $("#requiredActionInput").innerHTML = "";
    $("#requiredActionInput").classList.add("hidden");
    $("#requiredActionAlternatives").innerHTML = "";
    $("#requiredActionAlternatives").classList.add("hidden");
  }
  // Older runtimes can expose a saved semantic turn without a provisional
  // Goal row. Their generic "Plan your next request" attention copy would
  // compete with the recovery surface rendered for the same checkpoint.
  if (legacyPreGoalCheckpoint) {
    root.classList.add("hidden");
    requiredRoot.classList.add("hidden");
  }
  (runtime.timeline_preview || []).forEach(appendLiveEvent);
  renderLiveWorkflow();
  renderAdvancedLiveDetails();
  renderInspector();
  updateComposer();
}

function stableFingerprint(value) {
  const volatile = new Set([
    "updated_at",
    "heartbeat_at",
    "heartbeat_age",
    "last_event_id",
  ]);
  const scrub = (item) => {
    if (Array.isArray(item)) return item.map(scrub);
    if (!item || typeof item !== "object") return item;
    return Object.fromEntries(
      Object.entries(item)
        .filter(([key]) => !volatile.has(key))
        .map(([key, nested]) => [key, scrub(nested)]),
    );
  };
  return JSON.stringify(scrub(value || {}));
}

// A view refresh must be driven by durable content, not by the live telemetry
// envelope.  Provider heartbeats and timeline counters are intentionally
// ignored for Plan/Review so a slow request cannot make the page repaint on
// every poll.
function viewFingerprint(value, view) {
  const telemetry = new Set([
    "activity_sequence",
    "last_event_id",
    "last_signal_at",
    "heartbeat_at",
    "heartbeat_age",
    "received_bytes",
    "received_chunks",
    "received_tokens",
    "provider_request_state",
    "liveness",
    "session_revision",
    "attempt_id",
    "attempt_state",
    "attempt_model",
    "retry_at",
    "failure_kind",
    "safe_stream_preview",
    "stream_state",
    "timeline_preview",
  ]);
  const scrub = (item) => {
    if (Array.isArray(item)) return item.map(scrub);
    if (!item || typeof item !== "object") return item;
    return Object.fromEntries(
      Object.entries(item)
        .filter(([key]) => key !== "updated_at" && key !== "connection")
        .filter(([key]) => !(view !== "execution" && key === "runtime"))
        .filter(([key]) => !(view !== "execution" && telemetry.has(key)))
        .map(([key, nested]) => [key, scrub(nested)]),
    );
  };
  return JSON.stringify(scrub(value || {}));
}

function captureScrollAnchor() {
  const anchors = $$('[data-scroll-anchor]', $("#viewRoot"));
  const visible = anchors.find((node) => node.getBoundingClientRect().bottom > 96);
  if (visible) {
    return {
      id: visible.dataset.scrollAnchor,
      top: visible.getBoundingClientRect().top,
      scrollY: window.scrollY,
    };
  }
  return { id: "", top: 0, scrollY: window.scrollY };
}

function restoreScrollAnchor(anchor) {
  if (!anchor) return;
  if (anchor.id) {
    const target = $$('[data-scroll-anchor]', $("#viewRoot"))
      .find((node) => node.dataset.scrollAnchor === anchor.id);
    if (target) {
      window.scrollBy(0, target.getBoundingClientRect().top - anchor.top);
      return;
    }
  }
  window.scrollTo({ top: anchor.scrollY, behavior: "auto" });
}

function readerIsBusy() {
  const focused = document.activeElement;
  return Boolean(
    focused
      && $("#viewRoot").contains(focused)
      && focused.matches("input,textarea,select,[contenteditable='true']"),
  );
}

function reviewIsDirty() {
  return state.reviewDecisions.size > 0
    || state.reviewComments.length > 0
    || state.reviewConfirm;
}

function rememberPlanConflict(snapshot, fingerprint) {
  const acknowledged = state.planConflict?.acknowledgedFingerprint;
  if (acknowledged && acknowledged === fingerprint) return false;
  const draft = state.planModel
    ? JSON.parse(JSON.stringify(state.planModel))
    : null;
  state.planConflict = {
    fingerprint,
    snapshot,
    draft,
    reviewing: false,
  };
  state.pendingUpdates.plan = true;
  return true;
}

function applyLiveActivity(event) {
  const anchor = readerIsBusy() ? captureScrollAnchor() : null;
  appendLiveEvent(event);
  if (!state.context) return;
  const runtime = { ...(state.context.runtime || {}) };
  if (event.phase) runtime.phase = event.phase;
  if (event.actor) runtime.active_actor = event.actor;
  if (event.task) runtime.current_task = event.task;
  if (event.operation) runtime.active_operation = publicThreadRecoveryReason({ error: event.operation });
  if (event.waiting_on) runtime.waiting_on = event.waiting_on;
  runtime.activity_sequence = Math.max(Number(runtime.activity_sequence || 0), Number(event.sequence || 0));
  runtime.received_bytes = Math.max(Number(runtime.received_bytes || 0), Number(event.received_bytes || 0));
  runtime.received_chunks = Math.max(Number(runtime.received_chunks || 0), Number(event.received_chunks || 0));
  runtime.received_tokens = Math.max(Number(runtime.received_tokens || 0), Number(event.received_tokens || 0));
  if (event.stream_kind) runtime.stream_kind = event.stream_kind;
  if (event.state) runtime.stream_state = event.state === "receiving" ? "receiving" : runtime.stream_state;
  if (event.safe_text_fragment) runtime.safe_stream_preview = `${runtime.safe_stream_preview || ""}${event.safe_text_fragment}`.slice(-4000);
  runtime.last_signal_at = Date.now() / 1000;
  if (event.source === "MODEL") {
    runtime.provider_request_state = event.state;
    const providerState = event.provider_state || event.state;
    runtime.liveness = providerState === "receiving" ? "receiving"
      : providerState === "completed" ? "processing_response"
      : providerState === "failed" ? "stalled"
      : providerState === "server_processing" ? "server_processing"
      : providerState === "network_unavailable" ? "network_unavailable"
      : providerState === "request_created" ? "request_created"
      : "request_sent";
  } else if (event.state === "waiting") {
    runtime.liveness = "waiting";
  } else if (event.state === "failed") {
    runtime.liveness = "stalled";
  }
  state.context = { ...state.context, runtime };
  renderLiveWorkflow();
  renderAdvancedLiveDetails();
  restoreScrollAnchor(anchor);
}

function startLiveEvents() {
  state.eventSource?.close();
  const source = new EventSource(`/api/sessions/${encodeURIComponent(sessionId)}/events`);
  state.eventSource = source;
  source.addEventListener("open", () => {
    // TCP/HTTP open is not yet a verified live workflow channel. Wait for a
    // valid snapshot or activity event before claiming the UI is live.
    state.liveConnected = false;
    setConnection(false, "Reconnecting");
  });
  source.addEventListener("snapshot", (message) => {
    try {
      const anchor = readerIsBusy() ? captureScrollAnchor() : null;
      const context = reconcileResolvedApproval(JSON.parse(message.data));
      state.context = context;
      (context.runtime?.timeline_preview || []).forEach(appendLiveEvent);
      state.liveConnected = true;
      renderChrome();
      restoreScrollAnchor(anchor);
      setConnection(true, "Live");
    } catch {
      setConnection(false, "Reconnecting");
    }
  });
  source.addEventListener("activity", (message) => {
    try {
      applyLiveActivity(JSON.parse(message.data));
      state.liveConnected = true;
      setConnection(true, "Live");
    } catch {
      // A malformed presentation event is ignored; the durable snapshot and
      // reconciliation polling remain authoritative.
    }
  });
  source.addEventListener("error", () => {
    state.liveConnected = false;
    setConnection(false, "Reconnecting");
    if (state.context) {
      state.context = {
        ...state.context,
        runtime: { ...(state.context.runtime || {}), liveness: "disconnected" },
      };
      renderLiveWorkflow();
    }
  });
}

function threadItemSummary(item) {
  const payload = item?.payload || {};
  return String(
    payload.summary || payload.content || payload.reason || payload.status || "Activity recorded",
  );
}

function publicThreadRecoveryReason(payload = {}) {
  const raw = String(payload.error || payload.reason || payload.summary || "").trim();
  const lower = raw.toLowerCase();
  if (
    lower.includes("submit_semantic_route")
    || lower.includes("submit_semantic_turn")
    || lower.includes("must be called exactly once")
    || lower.includes("only allowed call is")
    || lower.includes("semantic routing is saved but could not be validated")
  ) {
    return "The model's routing response could not be validated. The saved request is ready for a targeted retry.";
  }
  if (["contract", "schema", "semantic"].includes(String(payload.failure_kind || "").toLowerCase())) {
    return "The model response needs a small repair. The saved request is ready for a targeted retry.";
  }
  return raw || "The saved request is ready for recovery.";
}

function publicThreadStatusReason(payload = {}) {
  const recovery = state.context?.provider_recovery || {};
  if (Object.keys(recovery).length) {
    const stateName = String(recovery.state || "").toLowerCase();
    return ["retrying", "waiting", "paused", "recovering"].includes(stateName)
      ? "Retrying the saved checkpoint; your original request is preserved."
      : "The saved checkpoint is preserved and ready for recovery.";
  }
  return payload.reason
    ? publicThreadRecoveryReason(payload)
    : `Working with ${payload.provider || "the selected provider"}/${payload.model || "model"}.`;
}

function threadItemMarkup(item) {
  const payload = item?.payload || {};
  const type = String(item?.type || item?.kind || "workflow_status");
  const itemId = escapeHtml(item?.item_id || "thread-item");
  const sequence = Number(item?.sequence || 0);
  const title = {
    user_message: "You",
    assistant_message: "GA3BAD",
    workflow_status: "Workflow",
    plan: "Plan",
    tool_run: "Tool run",
    approval: "Approval",
    change_set: "Changes",
    review: "Changes",
    recovery: "Recovery",
    completion: "Completion",
  }[type] || "Workflow";
  if (type === "plan") {
    const tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    return `<article class="thread-item thread-plan" data-thread-item="${itemId}" data-sequence="${sequence}">
      <div class="thread-item-meta"><span>${title}</span><span>r${escapeHtml(payload.revision ?? "—")}</span></div>
      <h3>${escapeHtml(payload.summary || "Plan in progress")}</h3>
      <p class="thread-reading">${escapeHtml(payload.goal_status || payload.status || "Open the saved plan before work starts.")}</p>
      ${tasks.length ? `<ol class="thread-task-list">${tasks.slice(0, 8).map((task, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(task.title || task.id || "Task")}</strong><em>${escapeHtml(statusLabel(task.status || "pending"))}</em></li>`).join("")}</ol>` : ""}
      <div class="thread-item-actions"><button class="secondary-button" data-thread-view="plan" type="button">Open plan</button>${payload.status === "pending_approval" || payload.goal_status === "awaiting_plan_approval" ? '<button class="primary-button" data-thread-view="plan" type="button">Approve & work</button>' : ""}</div>
    </article>`;
  }
  if (type === "approval") {
    const fingerprint = escapeHtml(payload.action_fingerprint || "");
    const pending = Boolean(payload.pending);
    return `<article class="thread-item thread-approval${pending ? " is-pending" : ""}" data-thread-item="${itemId}" data-sequence="${sequence}">
      <div class="thread-item-meta"><span>${title}</span><span>${pending ? "Needs your decision" : "Recorded"}</span></div>
      <h3>${escapeHtml(payload.tool ? `Allow ${String(payload.tool).replaceAll("_", " ")}?` : payload.summary || "Approval recorded")}</h3>
      <p>${escapeHtml(payload.summary || (pending ? "This action is paused until you choose how it may run." : `Decision: ${payload.decision || "recorded"}`))}</p>
      ${pending && fingerprint ? `<div class="thread-approval-actions"><button class="primary-button" data-thread-approval="allow_once" data-fingerprint="${fingerprint}" type="button">Allow once</button><button class="secondary-button" data-thread-approval="allow_session" data-fingerprint="${fingerprint}" type="button">Always allow this session</button><button class="quiet-button" data-thread-approval="deny" data-fingerprint="${fingerprint}" type="button">Deny</button></div>` : ""}
    </article>`;
  }
  if (type === "tool_run") {
    return `<article class="thread-item thread-tool" data-thread-item="${itemId}" data-sequence="${sequence}">
      <div class="thread-item-meta"><span>${title}</span><span>${escapeHtml(payload.status || "recorded")}</span></div>
      <p><strong>${escapeHtml(payload.tool || payload.event_type || "Tool activity")}</strong> · ${escapeHtml(payload.summary || "Activity recorded")}</p>
    </article>`;
  }
  if (type === "workflow_status") {
    return `<article class="thread-item thread-status" data-thread-item="${itemId}" data-sequence="${sequence}">
      <div class="thread-item-meta"><span>${title}</span><span>${escapeHtml(statusLabel(payload.phase || payload.status || "ready"))}</span></div>
      <p>${escapeHtml(publicThreadStatusReason(payload))}</p>
    </article>`;
  }
  if (type === "recovery") {
    return `<article class="thread-item thread-recovery" data-thread-item="${itemId}" data-sequence="${sequence}">
      <div class="thread-item-meta"><span>${title}</span><span>Saved checkpoint</span></div>
      <h3>The exact request is preserved</h3><p>${escapeHtml(publicThreadRecoveryReason(payload))}</p>
      <div class="thread-item-actions"><button class="secondary-button" data-thread-view="execution" type="button">Open result</button></div>
    </article>`;
  }
  if (type === "change_set" || type === "review") {
    return `<article class="thread-item thread-review" data-thread-item="${itemId}" data-sequence="${sequence}">
      <div class="thread-item-meta"><span>${title}</span><span>${escapeHtml(payload.status || "Ready for review")}</span></div>
      <h3>${escapeHtml(payload.file_count ? `${payload.file_count} changed file${payload.file_count === 1 ? "" : "s"}` : payload.checkpoint_id || "Recorded changes")}</h3>
      <p>Changes and decisions stay attached to this thread.</p>
      <div class="thread-item-actions"><button class="secondary-button" data-thread-view="review" type="button">Open changes</button></div>
    </article>`;
  }
  const content = String(payload.content || threadItemSummary(item));
  return `<article class="thread-item thread-${escapeHtml(type)}" data-thread-item="${itemId}" data-sequence="${sequence}" dir="auto">
    <div class="thread-item-meta"><span>${title}</span><span>${escapeHtml(item?.created_at ? activityTime(item.created_at) : "")}</span></div>
    <p class="thread-reading">${escapeHtml(content)}</p>
  </article>`;
}

function renderThread(snapshot) {
  const items = Array.isArray(snapshot?.items) ? snapshot.items : [];
  const hasRecovery = items.some((item) => String(item?.type || item?.kind || "") === "recovery");
  const visibleItems = items.filter((item) => {
    if (!hasRecovery || String(item?.type || item?.kind || "") !== "plan") return true;
    const payload = item?.payload || {};
    const hasTasks = Array.isArray(payload.tasks) && payload.tasks.length > 0;
    const revision = payload.revision;
    return hasTasks || (revision !== null && revision !== undefined && revision !== "" && revision !== "pending");
  });
  const root = $("#viewRoot");
  root.innerHTML = `<div class="thread-view" aria-label="Unified workflow thread">
    <header class="thread-head"><div><p class="eyebrow">GA3BAD workspace</p><h1 id="threadTitle">Unified workflow thread</h1><p class="thread-objective">${escapeHtml(state.context?.goal?.objective || state.context?.runtime?.objective || "Describe what you want to build")}</p><p>Plans, approvals, changes, and the final result stay in one durable conversation.</p></div><span class="status-badge status-${statusClass(state.context?.runtime?.phase || "ready")}">${escapeHtml(statusLabel(state.context?.runtime?.phase || "ready"))}</span></header>
    <div class="thread-items">${visibleItems.length ? visibleItems.map(threadItemMarkup).join("") : '<p class="empty-state">No thread items have been recorded yet. Send a request to begin.</p>'}</div>
  </div>`;
  $$('[data-thread-view]', root).forEach((button) => button.addEventListener("click", () => navigate(button.dataset.threadView || "plan")));
  $$('[data-thread-approval]', root).forEach((button) => button.addEventListener("click", () => resolveToolApproval(button.dataset.threadApproval, button.dataset.fingerprint)));
}

function inspectorSectionMarkup(section, data) {
  const payload = data || {};
  if (section === "environment") {
    const environment = payload.environment || {};
    const model = payload.model || {};
    const git = payload.git || {};
    return `<div class="inspector-summary"><p class="eyebrow">Environment</p><h3>${escapeHtml(environment.name || "Workspace")}</h3><p class="inspector-path">${escapeHtml(environment.workspace || "Local workspace")}</p>
      <dl class="inspector-facts"><div><dt>Model</dt><dd>${escapeHtml(`${model.provider || "model"}/${model.model || "saved"}`)}</dd></div><div><dt>Git</dt><dd>${escapeHtml(git.branch || git.tier || "snapshot")}</dd></div><div><dt>Access</dt><dd>${escapeHtml(payload.access?.level || "normal")}</dd></div><div><dt>Sleep</dt><dd>${escapeHtml(payload.sleep?.policy || "off")}</dd></div></dl><button class="secondary-button inspector-settings" data-inspector-settings type="button">Project settings</button></div>`;
  }
  if (section === "changes") {
    const changes = payload.changes || [];
    return `<div class="inspector-summary"><p class="eyebrow">Changes</p><h3>${changes.length} recorded file${changes.length === 1 ? "" : "s"}</h3>${changes.length ? `<ul class="inspector-list">${changes.map((item) => `<li><span>${escapeHtml(item.path || "file")}</span><em>${escapeHtml(item.status || "modified")}</em></li>`).join("")}</ul>` : '<p class="muted">No file changes are recorded yet.</p>'}</div>`;
  }
  if (section === "processes") {
    const processes = payload.processes || [];
    return `<div class="inspector-summary"><p class="eyebrow">Processes</p><h3>Live work</h3>${processes.length ? `<ul class="inspector-list">${processes.map((item) => `<li><span><strong>${escapeHtml(item.name || item.id)}</strong><small>${escapeHtml(item.task || "")}</small></span><em>${escapeHtml(statusLabel(item.status || "running"))}</em></li>`).join("")}</ul>` : '<p class="muted">No active processes.</p>'}</div>`;
  }
  if (section === "tree") {
    const tree = payload.tree || [];
    const flatten = (nodes, depth = 0) => nodes.map((node) => `<li class="tree-depth-${Math.min(depth, 6)}"><span>${escapeHtml(node.title || node.id || "node")}</span>${badge(node.status || "pending")}${node.children?.length ? `<ul>${flatten(node.children, depth + 1)}</ul>` : ""}</li>`).join("");
    return `<div class="inspector-summary"><p class="eyebrow">Agents & tree</p><h3>Work hierarchy</h3>${tree.length ? `<ul class="inspector-tree">${flatten(tree)}</ul>` : '<p class="muted">The work map will appear after a plan starts.</p>'}</div>`;
  }
  const sources = payload.sources || {};
  return `<div class="inspector-summary"><p class="eyebrow">Sources</p><h3>${Number(sources.timeline_entries || 0)} timeline entries</h3>${(sources.artifacts || []).length ? `<ul class="inspector-list">${sources.artifacts.map((item) => `<li><span>${escapeHtml(item.name || item.kind || "artifact")}</span><em>${escapeHtml(item.kind || "artifact")}</em></li>`).join("")}</ul>` : '<p class="muted">No attached sources or artifacts.</p>'}</div>`;
}

function renderInspector(snapshot = state.inspector) {
  state.inspector = snapshot || state.inspector;
  document.body.classList.toggle("inspector-visible", state.inspectorOpen);
  const inspector = $("#inspector");
  if (!inspector || !state.inspector) return;
  inspector.classList.toggle("open", state.inspectorOpen);
  inspector.setAttribute("aria-hidden", String(!state.inspectorOpen));
  $("#inspectorToggle")?.setAttribute("aria-expanded", String(state.inspectorOpen));
  $("#inspectorBody").innerHTML = inspectorSectionMarkup(state.inspectorSection, state.inspector);
  $$(".inspector-tab").forEach((tab) => {
    const selected = tab.dataset.inspectorSection === state.inspectorSection;
    tab.classList.toggle("selected", selected);
    tab.setAttribute("aria-selected", String(selected));
  });
  $("[data-inspector-settings]")?.addEventListener("click", openProjectSettings);
}

function renderProjectSessions(snapshot = state.projectSessions) {
  state.projectSessions = snapshot || state.projectSessions;
  if (!state.projectSessions) return;
  $("#railProjectName").textContent = String(state.projectSessions.projects?.[0]?.name || "GA3BAD");
  const query = String($("#railSearch")?.value || "").trim().toLowerCase();
  const tasks = (state.projectSessions.tasks || []).filter((task) => !query || `${task.title} ${task.status}`.toLowerCase().includes(query));
  $("#railTaskList").innerHTML = tasks.length ? tasks.map((task) => `<li><button class="rail-task${task.active ? " active" : ""}" data-rail-task="${escapeHtml(task.id)}" type="button"><span>${escapeHtml(task.title)}</span><small>${escapeHtml(statusLabel(task.status))}</small></button></li>`).join("") : '<li class="rail-empty">No saved tasks</li>';
  $$('[data-rail-task]').forEach((button) => button.addEventListener("click", () => {
    state.historyFilter = { goal: button.dataset.railTask };
    navigate("thread");
  }));
}

function projectSessionsFingerprint(snapshot) {
  return JSON.stringify({
    projects: (snapshot?.projects || []).map((item) => ({
      id: item.id,
      name: item.name,
      path: item.path,
      active: Boolean(item.active),
    })),
    tasks: (snapshot?.tasks || []).map((item) => ({
      id: item.id,
      title: item.title,
      status: item.status,
      active: Boolean(item.active),
      plan_revision: item.plan_revision,
    })),
  });
}

async function loadInspector(section = state.inspectorSection) {
  try {
    const query = state.inspector ? (section ? `?section=${encodeURIComponent(section)}` : "") : "";
    const snapshot = await api(`/inspector${query}`);
    // Section responses contain only a slice. Keep the full snapshot from the
    // initial request when possible and reconcile the selected slice in place.
    if (snapshot.section && state.inspector) {
      state.inspector = { ...state.inspector, [section]: snapshot.section };
    } else {
      state.inspector = snapshot;
    }
    renderInspector();
  } catch {
    // Inspector is supplementary; the central workflow remains usable when a
    // legacy owner does not expose this additive endpoint yet.
  }
}

async function loadProjectSessions() {
  try {
    const response = await fetch("/api/sessions", { credentials: "same-origin", headers: { Accept: "application/json" } });
    if (!response.ok) return;
    renderProjectSessions(await response.json());
  } catch {
    // The session index is additive and must never block the workspace.
  }
}

function openActivityTimeline() {
  const events = state.liveEvents.slice().reverse();
  openDrawer({
    eyebrow: "Verified runtime events",
    title: "Live workflow activity",
    body: events.length ? `<ol class="activity-detail-list">${events.map((event) => `
      <li>
        <div class="activity-detail-meta">${escapeHtml(activityTime(event.timestamp))} · ${escapeHtml(event.source || "HARNESS")} · ${escapeHtml(event.state || "active")} · #${escapeHtml(event.sequence || "-")}</div>
        <p>${escapeHtml(event.message || event.operation ? publicThreadRecoveryReason({ error: event.message || event.operation }) : "Activity received")}</p>
        ${state.detail === "advanced" && event.operation ? `<div class="activity-detail-meta">${escapeHtml(event.phase || "")} · ${escapeHtml(event.actor || "")} · ${escapeHtml(publicThreadRecoveryReason({ error: event.operation }))}</div>` : ""}
      </li>`).join("")}</ol>` : `<p class="empty-state">No runtime events have been received yet.</p>`,
  });
}

async function refreshContext({ enforceGate = true } = {}) {
  try {
    const context = reconcileResolvedApproval(await api("/workspace"));
    const liveUnavailable = Boolean(state.eventSource && !state.liveConnected);
    state.context = liveUnavailable
      ? {
          ...context,
          runtime: { ...(context.runtime || {}), liveness: "disconnected" },
        }
      : context;
    setConnection(!liveUnavailable, liveUnavailable ? "Polling" : "Connected");
    const requiredView = context.required_view ? canonicalView(context.required_view) : "";
    if (enforceGate && requiredView && requiredView !== state.currentView && state.currentView !== "thread") {
      await navigate(requiredView, { gate: true });
      return;
    }
    renderChrome();
  } catch (error) {
    setConnection(false);
    if (!state.context) throw error;
  }
}

async function navigate(view, { replace = false, gate = false } = {}) {
  view = canonicalView(view);
  if (!["thread", "plan", "review", "execution", "history"].includes(view)) return;
  if (state.planDirty && state.currentView === "plan" && view !== "plan" && !gate) {
    announce("Save or discard the current plan draft before leaving.", true);
    return;
  }
  state.currentView = view;
  const url = `/sessions/${encodeURIComponent(sessionId)}/${view}`;
  history[replace ? "replaceState" : "pushState"]({ view }, "", url);
  window.scrollTo({ top: 0, behavior: "auto" });
  renderChrome();
  await loadView(view, true);
}

function planPayload() {
  const model = state.planModel;
  return {
    base_revision: model.base_revision,
    summary: model.summary.trim(),
    tasks: model.tasks.map((task) => ({
      id: task.id,
      title: task.title.trim(),
      description: task.description.trim(),
      parent_id: task.parent_id || null,
      dependencies: [...task.dependencies],
      agent_role: task.agent_role || "coder",
      inputs: [...task.inputs],
      outputs: [...task.outputs],
      expected_files: [...task.expected_files],
      acceptance_criteria: [...task.acceptance_criteria],
      requirement_refs: [...(task.requirement_refs || [])],
      tests: [...task.tests],
      risk_level: task.risk_level || "medium",
      required_tools: [...task.required_tools],
      memory_dependencies: [...task.memory_dependencies],
      retry_policy: { ...task.retry_policy },
      approval_gate: Boolean(task.approval_gate),
      constraints: [...task.constraints],
      parallel: Boolean(task.parallel),
      comments: [...task.comments],
    })),
    global_constraints: [...model.global_constraints],
    protected_paths: [...model.protected_paths],
    change_note: model.change_note || "Edited in Plan Studio",
  };
}

function makePlanModel(snapshot) {
  const source = snapshot.draft || snapshot;
  const byId = new Map(snapshot.tasks.map((task) => [task.id, task]));
  return {
    base_revision: snapshot.revision,
    summary: source.summary ?? snapshot.summary,
    tasks: (source.tasks || snapshot.tasks).map((task) => ({
      ...task,
      status: byId.get(task.id)?.status || "pending",
      parent_id: task.parent_id || null,
      dependencies: [...(task.dependencies || [])],
      inputs: [...(task.inputs || [])],
      outputs: [...(task.outputs || [])],
      expected_files: [...(task.expected_files || [])],
      acceptance_criteria: [...(task.acceptance_criteria || ["Requested behavior is implemented."])],
      requirement_refs: [...(task.requirement_refs || [])],
      tests: [...(task.tests || ["Run the relevant verification."])],
      required_tools: [...(task.required_tools || [])],
      memory_dependencies: [...(task.memory_dependencies || [])],
      constraints: [...(task.constraints || [])],
      comments: [...(task.comments || [])],
      retry_policy: { max_retries: 2, backoff_seconds: 0, ...(task.retry_policy || {}) },
    })),
    global_constraints: [...(source.global_constraints || snapshot.global_constraints || [])],
    protected_paths: [...(source.protected_paths || snapshot.protected_paths || [])],
    change_note: source.change_note || "Edited in Plan Studio",
  };
}

function planPhase(plan) {
  if (plan.goal_status === "awaiting_plan_approval") return "Awaiting approval";
  if (plan.goal_status === "completed") return "Completed";
  return "Running";
}

function taskEditor(task, index) {
  return `
    <div class="task-editor" data-editor="${index}">
      <div class="form-grid">
        <div class="form-field">
          <label for="task-${index}-title">Title</label>
          <input id="task-${index}-title" data-task="${index}" data-field="title"
                 value="${escapeHtml(task.title)}" maxlength="180">
        </div>
        <div class="form-field advanced-only">
          <label for="task-${index}-id">Task ID</label>
          <input id="task-${index}-id" data-task="${index}" data-field="id"
                 value="${escapeHtml(task.id)}" maxlength="24">
        </div>
        <div class="form-field full">
          <label for="task-${index}-description">What this task must do</label>
          <textarea id="task-${index}-description" data-task="${index}" data-field="description">${escapeHtml(task.description)}</textarea>
        </div>
        <div class="form-field">
          <label for="task-${index}-criteria">Acceptance criteria <small>(one per line)</small></label>
          <textarea id="task-${index}-criteria" data-task="${index}" data-field="acceptance_criteria" data-list="true">${escapeHtml(lines(task.acceptance_criteria))}</textarea>
        </div>
        <div class="form-field">
          <label for="task-${index}-tests">Verification <small>(one per line)</small></label>
          <textarea id="task-${index}-tests" data-task="${index}" data-field="tests" data-list="true">${escapeHtml(lines(task.tests))}</textarea>
        </div>
      </div>
      <details class="advanced-section advanced-only">
        <summary>Scope and dependencies</summary>
        <div class="form-grid">
          <div class="form-field">
            <label>Depends on <small>(task IDs, one per line)</small></label>
            <textarea data-task="${index}" data-field="dependencies" data-list="true">${escapeHtml(lines(task.dependencies))}</textarea>
          </div>
          <div class="form-field">
            <label>Expected files <small>(one per line)</small></label>
            <textarea data-task="${index}" data-field="expected_files" data-list="true">${escapeHtml(lines(task.expected_files))}</textarea>
          </div>
          <div class="form-field">
            <label>Inputs <small>(one per line)</small></label>
            <textarea data-task="${index}" data-field="inputs" data-list="true">${escapeHtml(lines(task.inputs))}</textarea>
          </div>
          <div class="form-field">
            <label>Outputs <small>(one per line)</small></label>
            <textarea data-task="${index}" data-field="outputs" data-list="true">${escapeHtml(lines(task.outputs))}</textarea>
          </div>
        </div>
      </details>
      <details class="advanced-section advanced-only">
        <summary>Agent and retry policy</summary>
        <div class="form-grid">
          <div class="form-field">
            <label>Agent role</label>
            <input data-task="${index}" data-field="agent_role" value="${escapeHtml(task.agent_role)}">
          </div>
          <div class="form-field">
            <label>Risk level</label>
            <select data-task="${index}" data-field="risk_level">
              ${["low", "medium", "high", "critical"].map((value) =>
                `<option value="${value}"${task.risk_level === value ? " selected" : ""}>${value}</option>`).join("")}
            </select>
          </div>
          <div class="form-field">
            <label>Max retries</label>
            <input type="number" min="0" max="20" data-task="${index}"
                   data-field="retry_policy.max_retries" value="${Number(task.retry_policy.max_retries)}">
          </div>
          <div class="form-field">
            <label>Backoff seconds</label>
            <input type="number" min="0" max="3600" data-task="${index}"
                   data-field="retry_policy.backoff_seconds" value="${Number(task.retry_policy.backoff_seconds)}">
          </div>
        </div>
      </details>
      <div class="editor-actions">
        <button class="quiet-button" data-action="cancel-task-edit" type="button">Close editor</button>
        <button class="danger-button" data-action="delete-task" data-index="${index}" type="button">Remove task</button>
      </div>
    </div>`;
}

function renderNewPlanRequest() {
  const envelope = state.plan.capability_envelope || {};
  $("#viewRoot").innerHTML = `
    <header class="view-head">
      <div>
        <p class="eyebrow">Plan before work</p>
        <h2>What do you want to build or change?</h2>
        <p>Describe the complete outcome. The system will inspect the workspace and prepare a plan; no files change yet.</p>
      </div>
      <span class="status-badge status-waiting">Plan</span>
    </header>
    <section class="section">
      <div class="form-field full">
        <label for="newPlanRequest">Request</label>
        <textarea id="newPlanRequest" autofocus
          placeholder="Describe the outcome, constraints, and how you will know it works"></textarea>
        <p class="subtle">Ctrl/Cmd + Enter submits. You will review the generated plan before work starts.</p>
      </div>
      <div class="section-actions">
        <button class="primary-button" data-action="submit-plan-request" type="button">Prepare plan</button>
      </div>
    </section>
    <section class="section advanced-only">
      <div class="section-head"><div><h3>Selected model capability</h3><p>Metadata only; unknown fields use the conservative minimal envelope.</p></div></div>
      <div class="meta-line">
        <span>Band: ${escapeHtml(envelope.capability_band || "minimal")}</span>
        <span>Parameters: ${escapeHtml(envelope.parameter_count_billions ?? "unknown")}B</span>
        <span>Context: ${escapeHtml(envelope.context_window_tokens ?? "unknown")}</span>
        <span>Concurrency: ${escapeHtml(envelope.max_concurrency ?? 1)}</span>
      </div>
    </section>`;
  $$('[data-action]', $("#viewRoot")).forEach((button) => button.addEventListener("click", handlePlanAction));
  const input = $("#newPlanRequest");
  input.addEventListener("input", () => autoGrow(input));
  input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      submitPlanRequest();
    }
  });
  autoGrow(input);
}

function renderPlan() {
  const plan = state.plan;
  if (plan.state === "new_request") {
    renderNewPlanRequest();
    return;
  }
  if (plan.state === "preparing_plan") {
    const runtime = plan.runtime || {};
    const localContinuation = localContinuationPresentation({
      ...(state.context || {}),
      runtime: { ...(state.context?.runtime || {}), ...runtime },
      provider_recovery: plan.provider_recovery || state.context?.provider_recovery,
    }, plan);
    const blocked = ["paused", "failed", "cancelled"].includes(plan.goal_status);
    $("#viewRoot").innerHTML = `
      <header class="view-head">
        <div>
          <p class="eyebrow">${blocked ? "SAVED CHECKPOINT" : "PLANNING IN PROGRESS"}</p>
          <h2>${blocked ? "Your request is saved" : "Your request is already running"}</h2>
          <p>${blocked ? "Planning stopped at a recoverable checkpoint." : "GA3BAD is preparing the first reviewable plan revision."}</p>
        </div>
        ${badge(plan.status)}
      </header>
      <section class="section current-request" data-testid="current-request">
        <div class="section-head"><div><p class="eyebrow">ORIGINAL REQUEST · SOURCE OF TRUTH</p><h3>Already submitted</h3></div></div>
        <p class="reading">${escapeHtml(plan.current_request || plan.objective)}</p>
        <div class="meta-line advanced-only">
          <span>Goal: ${escapeHtml(plan.goal_id)}</span>
          <span>Phase: ${escapeHtml(runtime.phase || "planning")}</span>
          <span>Status: ${escapeHtml(plan.goal_status)}</span>
        </div>
      </section>
      <section class="section notice-section">
        <h3>Do not submit this request again</h3>
        <p>${blocked
          ? "Choose a recovery action below. Retrying resumes this same saved request; it does not replay the prompt."
          : "This page updates automatically when the plan is ready. The composer below is for controls or an explicitly supported follow-up only."}</p>
        ${blocked ? `
          <div class="section-actions recovery-actions">
            <button class="primary-button" data-recovery-action="retry" type="button">Retry saved stage</button>
            ${localContinuation.show ? `<button class="secondary-button" data-recovery-action="continue_local_model" type="button">${localContinuation.label}</button>` : ""}
            <button class="quiet-button danger-action" data-recovery-action="stop" type="button">Stop safely</button>
          </div>` : ""}
      </section>`;
    if (blocked) bindRecoveryActions($("#viewRoot"));
    return;
  }
  const model = state.planModel;
  const canEdit = Boolean(plan.capabilities.can_edit);
  const phase = planPhase(plan);
  const active = plan.tasks.find((task) => ["in_progress", "verifying", "blocked"].includes(task.status));
  const planConflict = state.planConflict?.snapshot ? state.planConflict : null;
  const conflictNotice = planConflict ? `
    <section class="section draft-conflict" role="alert" aria-live="polite">
      <div>
        <p class="eyebrow">Saved plan changed</p>
        <h3>New saved plan revision</h3>
        <p>${planConflict.reviewing
          ? "You are reviewing the latest saved revision. Your local draft is still available."
          : "A newer saved revision arrived while you were editing. Nothing was overwritten."}</p>
      </div>
      <div class="section-actions">
        ${planConflict.reviewing
          ? ""
          : `<button class="secondary-button" data-action="review-plan-conflict" type="button">Inspect changes</button>`}
        <button class="quiet-button" data-action="keep-plan-draft" type="button">Keep my draft</button>
      </div>
    </section>` : "";
  const taskRows = model.tasks.map((task, index) => `
    <li class="task-row" data-scroll-anchor="task-${escapeHtml(task.id)}">
      <div class="task-main">
        <span class="task-order">${String(index + 1).padStart(2, "0")}</span>
        <div class="task-copy">
          <h4>${escapeHtml(task.title)}</h4>
          <p>${escapeHtml(task.description || "No description yet.")}</p>
          <div class="meta-line">
            <span>${escapeHtml(task.id)}</span>
            ${task.dependencies.length ? `<span>After ${escapeHtml(task.dependencies.join(", "))}</span>` : "<span>No dependencies</span>"}
          </div>
        </div>
        <div class="row-actions">
          ${badge(task.status)}
          ${canEdit ? `
            <button class="quiet-button" data-action="move-task-up" data-index="${index}"
                    data-help="Moves priority earlier. Dependencies still control when it can run."
                    type="button"${index === 0 ? " disabled" : ""}>Up</button>
            <button class="quiet-button" data-action="move-task-down" data-index="${index}"
                    data-help="Moves priority later. Dependencies remain authoritative."
                    type="button"${index === model.tasks.length - 1 ? " disabled" : ""}>Down</button>
            <button class="quiet-button" data-action="edit-task" data-index="${index}" type="button">
              ${state.editingTask === index ? "Close" : "Edit"}
            </button>` : ""}
        </div>
      </div>
      ${state.editingTask === index ? taskEditor(task, index) : ""}
      <div class="advanced-only">
        <details class="advanced-section">
          <summary>Task contract</summary>
          <div class="meta-line">
            <span>Role: ${escapeHtml(task.agent_role)}</span>
            <span>Risk: ${escapeHtml(task.risk_level)}</span>
            <span>Retries: ${Number(task.retry_policy.max_retries)}</span>
            <span>Expected: ${escapeHtml(task.expected_files.join(", ") || "resolved during execution")}</span>
          </div>
        </details>
      </div>
    </li>`).join("");

  $("#viewRoot").innerHTML = `
    <header class="view-head">
      <div>
        <p class="eyebrow">${escapeHtml(phase)}</p>
        <h2>Plan</h2>
        <p class="subtle runtime-context">Mode: ${escapeHtml((plan.runtime?.session_mode || plan.interaction_mode || "working"))} · Route: ${escapeHtml(plan.runtime?.route || "goal")} · Strategy: ${escapeHtml(plan.runtime?.execution_strategy || plan.execution_strategy || "pending")} · Model: ${escapeHtml(plan.runtime?.model || plan.capability_envelope?.model || "unknown")}</p>
        <p>Revision ${plan.revision} · ${model.tasks.length} tasks</p>
      </div>
      ${badge(plan.status)}
    </header>

    ${conflictNotice}

    ${executionPresentation(plan)}

    <section class="section">
      <div class="section-head">
        <div>
          <h3>Outcome</h3>
          <p>The exact project objective and the execution outline.</p>
        </div>
        ${canEdit && !state.editingSummary
          ? `<button class="quiet-button" data-action="edit-summary" type="button">Edit</button>`
          : ""}
      </div>
      ${state.editingSummary ? `
        <div class="inline-edit">
          <textarea id="planSummary" aria-label="Plan summary">${escapeHtml(model.summary)}</textarea>
          <div class="editor-actions">
            <button class="quiet-button" data-action="cancel-summary" type="button">Cancel</button>
            <button class="secondary-button" data-action="commit-summary" type="button">Keep in draft</button>
          </div>
        </div>` : `
        <div class="editable-copy">
          <div>
            <p class="reading">${escapeHtml(model.summary)}</p>
            <p class="subtle">${escapeHtml(plan.objective)}</p>
          </div>
        </div>`}
    </section>

    <section class="section plan-contract">
      <div class="section-head"><div><h3>What this plan must deliver</h3><p>Accepted outcomes and success checks carried into implementation and review.</p></div></div>
      <div class="contract-columns">
        <div><h4>Required outcomes</h4>${compactList(plan.semantic_goal?.required_outcomes, "The task contracts below define the required outcome.")}</div>
        <div><h4>Success checks</h4>${compactList(plan.semantic_goal?.acceptance_criteria, "The task criteria below define completion.")}</div>
      </div>
      ${requirementAnchors(plan.semantic_goal?.requirement_anchors)}
      ${(plan.semantic_goal?.constraints || []).length ? `<details class="advanced-section"><summary>Constraints and exclusions</summary><div class="contract-columns"><div><h4>Constraints</h4>${compactList(plan.semantic_goal.constraints)}</div><div><h4>Exclusions</h4>${compactList(plan.semantic_goal.exclusions)}</div></div></details>` : ""}
    </section>

    <section class="section advanced-only plan-revision-history">
      <div class="section-head"><div><h3>Plan revision history</h3><p>Every revision keeps the same goal and request fingerprint. Approval locks the selected revision.</p></div></div>
      <ol class="plain-list">${(plan.revisions || []).map((revision) => `<li>r${escapeHtml(revision.revision)} · ${escapeHtml(statusLabel(revision.status))} · ${escapeHtml(revision.summary)}</li>`).join("") || "<li>No previous revisions.</li>"}</ol>
    </section>

    ${!canEdit && active ? `
      <section class="section">
        <div class="section-head"><div><h3>Current work</h3><p>This task is controlled by the harness and cannot be interrupted here.</p></div></div>
        <div class="current-work">
          <h4>${escapeHtml(active.title)}</h4>
          <p>${escapeHtml(active.description)} · ${escapeHtml(statusLabel(active.status))}</p>
        </div>
      </section>` : ""}

    <section class="section">
      <div class="section-head">
        <div>
          <h3>${canEdit ? "Work order" : "Plan tasks"}</h3>
          <p>${canEdit
            ? "Reordering changes priority only. Dependencies still decide when a task becomes ready."
            : "Statuses are read-only and updated only by the workflow harness."}</p>
        </div>
        ${canEdit ? `<button class="secondary-button" data-action="add-task" type="button">Add task</button>` : ""}
      </div>
      <ol class="task-list">${taskRows}</ol>
    </section>

    ${canEdit ? `
      <section class="section advanced-only">
        <div class="section-head"><div><h3>Plan boundaries</h3><p>These constraints are included with every task in this revision.</p></div></div>
        <div class="form-grid">
          <div class="form-field">
            <label for="globalConstraints">Global constraints <small>(one per line)</small></label>
            <textarea id="globalConstraints" data-plan-field="global_constraints">${escapeHtml(lines(model.global_constraints))}</textarea>
          </div>
          <div class="form-field">
            <label for="protectedPaths">Protected files and directories <small>(one per line)</small></label>
            <textarea id="protectedPaths" data-plan-field="protected_paths">${escapeHtml(lines(model.protected_paths))}</textarea>
          </div>
        </div>
      </section>` : ""}

    ${plan.capabilities.can_manage_queue ? renderQueue(plan.queue) : ""}

    ${!canEdit && ["running", "verifying", "reviewing"].includes(plan.goal_status) ? `
      <section class="section work-started" role="status">
        <div>
          <h3>Work is running</h3>
          <p>This plan is now read-only. Follow the work map, send guidance, pause, or inspect the recorded output here.</p>
        </div>
        <button class="primary-button" data-action="open-execution" type="button">Open result</button>
      </section>` : ""}

    ${(canEdit || plan.capabilities.can_approve) ? `
      <section class="section">
        <div class="section-head">
          <div>
            <h3>Save or start</h3>
            <p>Saving a revision never starts work. Approve & start is the single approval that locks this plan and starts execution.</p>
          </div>
          <div class="section-actions">
            ${canEdit ? `<button class="quiet-button" data-action="save-draft"
                    data-help="Creates a draft only. Nothing will run yet." type="button">Save draft</button>
            <button class="secondary-button" data-action="save-revision"
                    data-help="Creates a new pending revision. It is not approved." type="button">Save revision</button>` : ""}
            ${plan.capabilities.can_approve ? `
              <button class="primary-button" data-action="confirm-approval" type="button">
                ${state.planDirty ? "Save revision & start" : "Approve & start"}
              </button>` : ""}
          </div>
        </div>
      </section>` : ""}
  `;
  bindPlan();
  autoGrow();
}

function renderQueue(queue) {
  const items = queue?.items || [];
  const pendingIds = items.filter((item) => item.status === "pending").map((item) => item.id);
  return `
    <section class="section" id="promptQueue">
      <div class="section-head">
        <div>
          <h3>Up next</h3>
          <p>Follow-ups sent from the composer wait here without interrupting current work.</p>
        </div>
        <span class="subtle">${items.length} of ${queue?.capacity || 10}</span>
      </div>
      ${items.length ? `<ol class="queue-list">
        ${items.map((item) => {
          const pendingIndex = pendingIds.indexOf(item.id);
          return `
          <li class="queue-row" data-prompt-id="${escapeHtml(item.id)}"
              draggable="${item.status === "pending"}">
            <div class="queue-main">
              <span class="drag-handle" aria-hidden="true">${item.status === "pending" ? "⠿" : "•"}</span>
              <div class="queue-copy">
                <h4>${escapeHtml(item.text)}</h4>
                <p>${escapeHtml(item.mode)} mode · ${escapeHtml(statusLabel(item.status))}</p>
              </div>
              <div class="row-actions">
                ${badge(item.status)}
                ${item.status === "pending" ? `
                  <button class="quiet-button" data-action="queue-up" data-id="${escapeHtml(item.id)}"
                          data-help="Moves this waiting request earlier. Current work will not be interrupted."
                          type="button"${pendingIndex === 0 ? " disabled" : ""}>Up</button>
                  <button class="quiet-button" data-action="queue-down" data-id="${escapeHtml(item.id)}"
                          data-help="Moves this waiting request later. Current work will not be interrupted."
                          type="button"${pendingIndex === pendingIds.length - 1 ? " disabled" : ""}>Down</button>` : ""}
              </div>
            </div>
          </li>`;
        }).join("")}
      </ol>` : `<p class="empty-state">No waiting requests. The current project can finish without another instruction.</p>`}
      <p class="subtle">Drag waiting requests to reorder them. The running request never moves.</p>
    </section>`;
}

function markPlanDirty() {
  state.planDirty = true;
  state.planConfirm = false;
}

function bindPlan() {
  $$("[data-action]", $("#viewRoot")).forEach((button) => button.addEventListener("click", handlePlanAction));
  $$("[data-open-model-picker]", $("#viewRoot")).forEach((button) => button.addEventListener("click", openModelPicker));
  $$("[data-task]", $("#viewRoot")).forEach((input) => {
    input.addEventListener("input", () => {
      const task = state.planModel.tasks[Number(input.dataset.task)];
      const field = input.dataset.field;
      let value = input.dataset.list === "true" ? parseLines(input.value) : input.value;
      if (field.startsWith("retry_policy.")) {
        task.retry_policy[field.split(".")[1]] = Number(value);
      } else {
        task[field] = value;
      }
      markPlanDirty();
      if (input.tagName === "TEXTAREA") autoGrow(input);
    });
    input.addEventListener("keydown", (event) => {
      if (input.tagName === "INPUT" && event.key === "Enter") {
        event.preventDefault();
        input.blur();
        announce("Kept in the local draft. Running work was not changed.");
      }
      if (event.key === "Escape") {
        event.preventDefault();
        state.editingTask = null;
        state.planModel = makePlanModel(state.plan);
        state.planDirty = false;
        renderPlan();
      }
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        saveDraft();
      }
    });
  });
  $$("[data-plan-field]", $("#viewRoot")).forEach((area) => {
    area.addEventListener("input", () => {
      state.planModel[area.dataset.planField] = parseLines(area.value);
      markPlanDirty();
      autoGrow(area);
    });
  });
  const summary = $("#planSummary");
  if (summary) {
    summary.addEventListener("input", () => autoGrow(summary));
    summary.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        state.editingSummary = false;
        renderPlan();
      } else if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        state.planModel.summary = summary.value;
        markPlanDirty();
        saveDraft();
      }
    });
  }
  bindQueue();
}

async function handlePlanAction(event) {
  const action = event.currentTarget.dataset.action;
  const index = Number(event.currentTarget.dataset.index);
  if (action === "submit-plan-request") return submitPlanRequest();
  if (action === "review-plan-conflict") return reviewPlanConflict();
  if (action === "keep-plan-draft") return keepPlanDraft();
  if (action === "increase-depth") return increaseExecutionDepth();
  if (action === "enqueue") return enqueuePrompt();
  if (action === "queue-up") return handleQueueMove(event.currentTarget.dataset.id, -1);
  if (action === "queue-down") return handleQueueMove(event.currentTarget.dataset.id, 1);
  if (action === "edit-summary") state.editingSummary = true;
  if (action === "cancel-summary") state.editingSummary = false;
  if (action === "commit-summary") {
    state.planModel.summary = $("#planSummary").value;
    state.editingSummary = false;
    markPlanDirty();
    announce("Kept in the draft. Running work was not changed.");
  }
  if (action === "edit-task") state.editingTask = state.editingTask === index ? null : index;
  if (action === "cancel-task-edit") state.editingTask = null;
  if (action === "move-task-up" && index > 0) {
    [state.planModel.tasks[index - 1], state.planModel.tasks[index]] =
      [state.planModel.tasks[index], state.planModel.tasks[index - 1]];
    markPlanDirty();
  }
  if (action === "move-task-down" && index < state.planModel.tasks.length - 1) {
    [state.planModel.tasks[index + 1], state.planModel.tasks[index]] =
      [state.planModel.tasks[index], state.planModel.tasks[index + 1]];
    markPlanDirty();
  }
  if (action === "add-task") {
    const used = new Set(state.planModel.tasks.map((task) => task.id));
    let number = state.planModel.tasks.length + 1;
    while (used.has(`T${String(number).padStart(3, "0")}`)) number += 1;
    state.planModel.tasks.push({
      id: `T${String(number).padStart(3, "0")}`,
      title: "New task",
      description: "Describe the required outcome.",
      status: "pending",
      parent_id: null,
      dependencies: [],
      agent_role: "coder",
      inputs: [],
      outputs: [],
      expected_files: [],
      acceptance_criteria: ["The requested outcome is implemented."],
      requirement_refs: [],
      tests: ["Run the relevant verification."],
      risk_level: "medium",
      required_tools: [],
      memory_dependencies: [],
      retry_policy: { max_retries: 2, backoff_seconds: 0 },
      approval_gate: false,
      constraints: [],
      parallel: false,
      comments: [],
    });
    state.editingTask = state.planModel.tasks.length - 1;
    markPlanDirty();
  }
  if (action === "delete-task" && state.planModel.tasks.length > 1) {
    state.planModel.tasks.splice(index, 1);
    state.editingTask = null;
    markPlanDirty();
  }
  if (action === "save-draft") return saveDraft();
  if (action === "save-revision") return saveRevision();
  if (action === "confirm-approval") return confirmApproval();
  if (action === "open-execution") {
    navigate("execution");
    return;
  }
  renderPlan();
}

async function submitPlanRequest() {
  const input = $("#newPlanRequest");
  const request = String(input?.value || "").trim();
  if (!request) {
    announce("Describe the outcome before preparing the plan.", true);
    input?.focus();
    return;
  }
  try {
    setBusy(true);
    announce("Inspecting the workspace and preparing the plan…");
    await api("/plan/request", {
      method: "POST",
      body: JSON.stringify({ request }),
    });
    await refreshContext({ enforceGate: false });
    await loadView("plan", true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function increaseExecutionDepth() {
  try {
    setBusy(true);
    const result = await api("/plan/depth", { method: "POST", body: JSON.stringify({}) });
    announce(`Strategy depth increased. Plan revision ${result.revision || "is being prepared"}.`);
    await loadView("plan", true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

function reviewPlanConflict() {
  const conflict = state.planConflict;
  if (!conflict?.snapshot) return;
  conflict.reviewing = true;
  state.plan = conflict.snapshot;
  state.planModel = makePlanModel(conflict.snapshot);
  state.planDirty = false;
  state.editingTask = null;
  state.editingSummary = false;
  state.pendingUpdates.plan = false;
  state.viewFingerprints.plan = conflict.fingerprint;
  renderPlan();
  announce("Showing the newer saved plan revision. Your draft is still available.");
}

function keepPlanDraft() {
  const conflict = state.planConflict;
  if (!conflict) return;
  if (conflict.draft) {
    state.planModel = JSON.parse(JSON.stringify(conflict.draft));
    state.planDirty = true;
  }
  // Remember the acknowledged saved fingerprint so polling does not reopen
  // the same conflict on every heartbeat. A genuinely newer revision will
  // still create a new notice.
  state.planConflict = {
    acknowledgedFingerprint: String(conflict.fingerprint || ""),
  };
  state.pendingUpdates.plan = false;
  renderPlan();
  announce("Kept your draft. Inspect the saved revision before saving or starting work.");
}

async function saveDraft() {
  try {
    setBusy(true);
    await api("/plan/draft", { method: "POST", body: JSON.stringify(planPayload()) });
    state.planDirty = false;
    state.planConflict = null;
    state.pendingUpdates.plan = false;
    announce("Saved to draft — running work was not changed.");
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function saveRevision() {
  try {
    setBusy(true);
    const result = await api("/plan/revision", { method: "POST", body: JSON.stringify(planPayload()) });
    state.planDirty = false;
    state.planConflict = null;
    state.pendingUpdates.plan = false;
    state.editingTask = null;
    state.editingSummary = false;
    announce(`Revision ${result.revision} saved. Nothing is running yet.`);
    await refreshContext({ enforceGate: false });
    await loadView("plan", true);
    return result;
  } catch (error) {
    announce(error.message, true);
    throw error;
  } finally {
    setBusy(false);
  }
}

async function confirmApproval() {
  try {
    setBusy(true);
    let revision = state.plan.revision;
    if (state.planDirty) {
      const result = await api("/plan/revision", { method: "POST", body: JSON.stringify(planPayload()) });
      revision = result.revision;
      await refreshContext({ enforceGate: false });
    }
    const result = await api("/actions", {
      method: "POST",
      body: JSON.stringify({
        action: "approve_plan",
        target_id: String(revision),
        value: String(revision),
        action_fingerprint: String(state.plan?.fingerprint || `plan-r${revision}`),
        expected_sequence: state.context?.history_cursor || 0,
        source: "web",
      }),
    });
    state.planDirty = false;
    state.planConfirm = false;
    state.planConflict = null;
    state.pendingUpdates.plan = false;
    announce(result.message || (
      result.execution_requested || result.next_phase === "working" || result.next_phase === "starting"
        ? `Plan r${revision} approved. Work is starting.`
        : `Plan r${revision} approved. The workflow is ready to continue.`
    ));
    await refreshContext({ enforceGate: false });
    await loadView("plan", true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function resolveToolApproval(decision) {
  const approval = state.context?.tool_approval;
  if (!approval?.action_fingerprint) return;
  try {
    setBusy(true);
    const result = await api("/actions", {
      method: "POST",
      body: JSON.stringify({
        action: decision === "deny"
          ? "deny_tool"
          : decision === "allow_session" ? "allow_tool_session" : "allow_tool",
        target_id: approval.tool || "tool",
        action_fingerprint: approval.action_fingerprint,
        expected_sequence: state.context?.history_cursor || 0,
        source: "web",
      }),
    });
    markApprovalResolved(approval.action_fingerprint);
    if (!result.duplicate) {
      announce(result.message || (decision === "deny" ? "Denied. The action remains stopped." : "Approved. Resuming the saved action."));
    }
    await refreshContext({ enforceGate: false });
    await loadView(state.currentView, true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

function pendingQueueIds() {
  return (state.plan?.queue?.items || []).filter((item) => item.status === "pending").map((item) => item.id);
}

async function persistQueue(ids) {
  try {
    setBusy(true);
    const result = await api("/queue/order", {
      method: "PATCH",
      body: JSON.stringify({ ordered_ids: ids }),
    });
    state.plan.queue = result.queue;
    announce("Waiting requests reordered. Current work was not interrupted.");
    renderPlan();
  } catch (error) {
    announce(error.message, true);
    await loadView("plan", true);
  } finally {
    setBusy(false);
  }
}

function bindQueue() {
  const prompt = $("#queuePrompt");
  if (prompt) {
    prompt.addEventListener("input", () => autoGrow(prompt));
    prompt.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        enqueuePrompt();
      }
    });
  }
  $$(".queue-row[draggable='true']").forEach((row) => {
    row.addEventListener("dragstart", () => {
      state.draggedPromptId = row.dataset.promptId;
      row.classList.add("dragging");
    });
    row.addEventListener("dragend", () => {
      row.classList.remove("dragging");
      $$(".queue-row").forEach((item) => item.classList.remove("drag-over"));
    });
    row.addEventListener("dragover", (event) => {
      event.preventDefault();
      row.classList.add("drag-over");
    });
    row.addEventListener("dragleave", () => row.classList.remove("drag-over"));
    row.addEventListener("drop", (event) => {
      event.preventDefault();
      const target = row.dataset.promptId;
      const ids = pendingQueueIds();
      const from = ids.indexOf(state.draggedPromptId);
      const to = ids.indexOf(target);
      if (from >= 0 && to >= 0 && from !== to) {
        const [moved] = ids.splice(from, 1);
        ids.splice(to, 0, moved);
        persistQueue(ids);
      }
    });
  });
}

async function enqueuePrompt() {
  const prompt = $("#queuePrompt");
  const text = prompt?.value.trim();
  if (!text) return announce("Write a request before adding it.", true);
  try {
    setBusy(true);
    const result = await api("/queue", { method: "POST", body: JSON.stringify({ text }) });
    state.plan.queue = result.queue;
    announce(result.duplicate
      ? (result.duplicate_of === "active_request" ? "That is already the active request. Nothing was queued." : "That follow-up is already queued.")
      : "Request added to Up next.");
    renderPlan();
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function handleQueueMove(id, direction) {
  const ids = pendingQueueIds();
  const index = ids.indexOf(id);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= ids.length) return;
  [ids[index], ids[target]] = [ids[target], ids[index]];
  await persistQueue(ids);
}

function decisionForFile(file) {
  const direct = state.reviewDecisions.get(`file:${file.path}`);
  if (direct) return direct;
  const hunks = file.hunks || [];
  if (hunks.length && hunks.every((hunk) => state.reviewDecisions.has(`hunk:${file.path}:${hunk.id}`))) {
    return { decision: "resolved_by_hunks" };
  }
  return null;
}

function reviewedCount() {
  return (state.review?.files || []).filter((file) => decisionForFile(file)).length;
}

function renderReadableDiff(file) {
  const hunks = file.hunks || [];
  const chunks = hunks.length
    ? hunks.map((hunk) => ({
        label: hunk.header,
        content: String(hunk.content || "").split(/\r?\n/).slice(1).join("\n"),
      }))
    : file.diff
      ? [{ label: "Recorded patch", content: file.diff }]
      : [];
  if (!chunks.length) {
    return `
      <div class="diff-unavailable" role="status">
        <strong>Diff unavailable</strong>
        <span>The workflow recorded this file, but no patch content was saved for it.</span>
      </div>`;
  }
  return `
    <section class="readable-diff" aria-label="Changes in ${escapeHtml(file.path)}">
      <div class="diff-heading">
        <h5>Changes</h5>
        <span>Green lines were added. Red lines were removed.</span>
      </div>
      ${chunks.map((chunk) => `
        <div class="diff-hunk">
          <div class="diff-hunk-label">${escapeHtml(chunk.label)}</div>
          <div class="diff-lines" role="region" aria-label="${escapeHtml(chunk.label)}">
            ${String(chunk.content || "").split(/\r?\n/).map((line) => {
              const kind = line.startsWith("+") && !line.startsWith("+++")
                ? "added"
                : line.startsWith("-") && !line.startsWith("---")
                  ? "deleted"
                  : line.startsWith("@@")
                    ? "metadata"
                    : "context";
              return `<code class="diff-line ${kind}">${escapeHtml(line || " ")}</code>`;
            }).join("")}
          </div>
        </div>`).join("")}
    </section>`;
}

function renderReview() {
  const review = state.review;
  const reviewed = reviewedCount();
  const total = review.files.length;
  const files = review.files.map((file, fileIndex) => {
    const fileDecision = state.reviewDecisions.get(`file:${file.path}`);
    const expanded = state.reviewExpanded.has(file.path);
    const detailsId = `review-file-${fileIndex}-changes`;
    return `
      <li class="file-row">
        <div class="file-summary">
          <div class="file-copy">
            <h4>${escapeHtml(file.path)}</h4>
            <p>${escapeHtml(file.reason)} · +${file.additions} −${file.deletions}</p>
          </div>
          <div class="decision-group">
            <button class="decision-button accept ${fileDecision?.decision === "accepted" ? "selected" : ""}"
                    data-review="accept-file" data-file="${escapeHtml(file.path)}" type="button">Looks good</button>
            <button class="decision-button change ${fileDecision?.decision === "changes_requested" ? "selected" : ""}"
                    data-review="change-file" data-file="${escapeHtml(file.path)}" type="button">Request changes</button>
            <button class="quiet-button" data-review="toggle-file" data-file="${escapeHtml(file.path)}"
                    type="button" aria-expanded="${expanded}" aria-controls="${detailsId}">${expanded ? "Hide changes" : "View changes"}</button>
          </div>
        </div>
        ${expanded ? `
          <div class="file-details" id="${detailsId}">
            ${renderReadableDiff(file)}
            <dl class="file-facts">
              <div class="fact"><dt>Why it changed</dt><dd>${escapeHtml(file.reason)}</dd></div>
              <div class="fact"><dt>Verification</dt><dd>${escapeHtml(typeof file.tests === "string" ? file.tests : JSON.stringify(file.tests || "Not recorded"))}</dd></div>
              <div class="fact"><dt>Responsible work</dt><dd>${escapeHtml(file.task || "Workflow")}</dd></div>
            </dl>
            <button class="quiet-button" data-review="comment-file" data-file="${escapeHtml(file.path)}" type="button">Add comment</button>
            <div class="advanced-only">
              ${(file.hunks || []).map((hunk) => {
                const key = `hunk:${file.path}:${hunk.id}`;
                const choice = state.reviewDecisions.get(key);
                return `
                  <details class="advanced-section">
                    <summary>${escapeHtml(hunk.header)}</summary>
                    <div class="decision-group">
                      <button class="decision-button accept ${choice?.decision === "accepted" ? "selected" : ""}"
                              data-review="accept-hunk" data-file="${escapeHtml(file.path)}"
                              data-hunk="${escapeHtml(hunk.id)}" type="button">Accept hunk</button>
                      <button class="decision-button change ${choice?.decision === "changes_requested" ? "selected" : ""}"
                              data-review="change-hunk" data-file="${escapeHtml(file.path)}"
                              data-hunk="${escapeHtml(hunk.id)}" type="button">Request hunk change</button>
                    </div>
                    <pre>${escapeHtml(hunk.content)}</pre>
                  </details>`;
              }).join("")}
              ${file.diff ? `<details class="advanced-section"><summary>Raw diff</summary><pre>${escapeHtml(file.diff)}</pre></details>` : ""}
            </div>
          </div>` : ""}
      </li>`;
  }).join("");
  const changes = [...state.reviewDecisions.values()].filter((item) =>
    ["changes_requested", "rejected"].includes(item.decision)).length;

  $("#viewRoot").innerHTML = `
    <header class="view-head">
      <div>
        <p class="eyebrow">Workspace › Changes · Decision surface</p>
        <h2>Recorded changes</h2>
        <p>Checkpoint ${escapeHtml(review.checkpoint_id)} · Plan r${review.plan_revision}</p>
      </div>
      ${badge(review.checkpoint_status)}
    </header>
    <section class="section">
      <div class="section-head">
        <div>
          <h3>Decide every file</h3>
          <p>Looks good continues execution. Request changes sends precise feedback to a fixer.</p>
        </div>
        <span class="review-progress">${reviewed} of ${total} files reviewed</span>
      </div>
      <ul class="file-list">${files}</ul>
    </section>
    <section class="section">
      <div class="section-head">
        <div>
          <h3>What happens after submit</h3>
          <p>${changes
            ? "A fixer will start for the requested changes. Accepted work is preserved unless a fix makes a related edit unavoidable."
            : "Once every file is approved, the harness continues to the next execution or verification step."}</p>
        </div>
        <button class="primary-button" data-review="prepare-submit" type="button"
                ${reviewed !== total || total === 0 ? "disabled" : ""}>Submit review</button>
      </div>
      ${state.reviewConfirm ? `
        <div class="confirm-panel" role="alert">
          <p>${changes
            ? `${changes} change request(s) will start a fixer.`
            : "All files are resolved. Work will continue after submission."}</p>
          <div class="section-actions">
            <button class="quiet-button" data-review="cancel-submit" type="button">Go back</button>
            <button class="primary-button" data-review="confirm-submit" type="button">Confirm submit</button>
          </div>
        </div>` : ""}
    </section>`;
  $$("[data-review]").forEach((button) => button.addEventListener("click", handleReviewAction));
}

function removeHunkDecisions(filePath) {
  [...state.reviewDecisions.keys()].filter((key) => key.startsWith(`hunk:${filePath}:`))
    .forEach((key) => state.reviewDecisions.delete(key));
}

function openDecisionDrawer({ filePath, hunkId = null }) {
  openDrawer({
    eyebrow: "Request changes",
    title: hunkId ? "What should change in this hunk?" : `What should change in ${filePath}?`,
    body: `
      <div class="form-field">
        <label for="changeReason">Required feedback</label>
        <textarea id="changeReason" placeholder="Describe the expected correction and why it matters."></textarea>
      </div>
      <div class="editor-actions">
        <button class="primary-button" id="saveChangeRequest" type="button">Save change request</button>
      </div>`,
    onReady: () => {
      const area = $("#changeReason");
      autoGrow(area);
      area.addEventListener("input", () => autoGrow(area));
      $("#saveChangeRequest").addEventListener("click", () => {
        const reason = area.value.trim();
        if (!reason) return announce("A change request needs a reason.", true);
        if (hunkId) {
          state.reviewDecisions.delete(`file:${filePath}`);
          state.reviewDecisions.set(`hunk:${filePath}:${hunkId}`, {
            target_type: "hunk", file_path: filePath, hunk_id: hunkId,
            decision: "changes_requested", reason,
          });
        } else {
          removeHunkDecisions(filePath);
          state.reviewDecisions.set(`file:${filePath}`, {
            target_type: "file", file_path: filePath, hunk_id: null,
            decision: "changes_requested", reason,
          });
        }
        closeDrawer();
        state.reviewConfirm = false;
        renderReview();
      });
      area.focus();
    },
  });
}

function openCommentDrawer(filePath) {
  openDrawer({
    eyebrow: "Change note",
    title: `Add a comment to ${filePath}`,
    body: `
      <div class="form-field">
        <label for="reviewComment">Comment</label>
        <textarea id="reviewComment" placeholder="Add context for the next agent or reviewer."></textarea>
      </div>
      <div class="editor-actions">
        <button class="primary-button" id="saveReviewComment" type="button">Add comment</button>
      </div>`,
    onReady: () => {
      const area = $("#reviewComment");
      autoGrow(area);
      area.addEventListener("input", () => autoGrow(area));
      $("#saveReviewComment").addEventListener("click", () => {
        const body = area.value.trim();
        if (!body) return announce("Write a comment first.", true);
        state.reviewComments.push({ file_path: filePath, hunk_id: null, line: null, body });
        closeDrawer();
        announce("Comment added to this review.");
      });
      area.focus();
    },
  });
}

async function handleReviewAction(event) {
  const action = event.currentTarget.dataset.review;
  const filePath = event.currentTarget.dataset.file;
  const hunkId = event.currentTarget.dataset.hunk || null;
  if (action === "toggle-file") {
    state.reviewExpanded.has(filePath) ? state.reviewExpanded.delete(filePath) : state.reviewExpanded.add(filePath);
  }
  if (action === "accept-file") {
    removeHunkDecisions(filePath);
    state.reviewDecisions.set(`file:${filePath}`, {
      target_type: "file", file_path: filePath, hunk_id: null, decision: "accepted", reason: "",
    });
    state.reviewConfirm = false;
  }
  if (action === "change-file") return openDecisionDrawer({ filePath });
  if (action === "comment-file") return openCommentDrawer(filePath);
  if (action === "accept-hunk") {
    state.reviewDecisions.delete(`file:${filePath}`);
    state.reviewDecisions.set(`hunk:${filePath}:${hunkId}`, {
      target_type: "hunk", file_path: filePath, hunk_id: hunkId, decision: "accepted", reason: "",
    });
    state.reviewConfirm = false;
  }
  if (action === "change-hunk") return openDecisionDrawer({ filePath, hunkId });
  if (action === "prepare-submit") state.reviewConfirm = true;
  if (action === "cancel-submit") state.reviewConfirm = false;
  if (action === "confirm-submit") return submitReview();
  renderReview();
}

async function submitReview() {
  try {
    setBusy(true);
    const result = await api("/review/submit", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_id: state.review.checkpoint_id,
        decisions: [...state.reviewDecisions.values()],
        comments: state.reviewComments,
        summary: "Changes submitted in the local workspace.",
      }),
    });
    announce(result.fixer_started ? "Changes submitted. A fixer has started." : "Changes submitted. Work can continue.");
    state.reviewConfirm = false;
    await refreshContext({ enforceGate: false });
    await loadView("review", true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

function executionTreeMarkup(nodes, depth = 0) {
  if (!nodes.length) return "";
  return `<ul class="node-list node-tree" data-tree-depth="${depth}">${nodes.map((node) => `
    <li class="node-branch">
      <div class="node-row" data-node-depth="${Math.min(depth, 6)}">
        <div class="node-main">
          <div class="node-copy">
            <h4><button class="link-button" data-node-id="${escapeHtml(node.id)}" type="button">${escapeHtml(node.title)}</button></h4>
            <p>${escapeHtml(node.assigned_role || "Harness assigned")} · ${node.dependencies.length ? `After ${escapeHtml(node.dependencies.join(", "))}` : depth ? "Child specialist" : "Root work"}</p>
          </div>
          ${badge(node.status)}
        </div>
      </div>
      ${executionTreeMarkup(node.children || [], depth + 1)}
    </li>`).join("")}</ul>`;
}

function executionControls() {
  const status = String(state.context?.goal?.status || "").toLowerCase();
  const runtimePhase = String(
    state.context?.runtime?.phase || state.context?.phase || "",
  ).toLowerCase();
  const waitingForApproval = runtimePhase === "waiting_for_approval";
  const savedWorkflow = hasSavedWorkflow(state.context);
  const paused = (["paused", "blocked"].includes(status) || ["paused", "retrying", "waiting"].includes(runtimePhase)) && !waitingForApproval;
  const finished = ["completed", "failed", "cancelled"].includes(status) || runtimePhase === "completed";
  if (!savedWorkflow || finished) {
    return `<div class="execution-controls"><button class="secondary-button" data-execution-action="model" type="button">Change model</button></div>`;
  }
  return `<div class="execution-controls" aria-label="Work controls">
    ${waitingForApproval
      ? '<span class="control-note">Approval required above</span>'
      : paused
      ? '<button class="primary-button" data-execution-action="resume" type="button">Resume</button><button class="secondary-button" data-execution-action="retry" type="button">Retry saved stage</button>'
      : '<button class="secondary-button" data-execution-action="pause" type="button">Pause safely</button>'}
    <button class="secondary-button" data-execution-action="model" type="button">Change model</button>
    <button class="quiet-button danger-action" data-execution-action="stop" type="button">Stop</button>
  </div>`;
}

function renderAgents() {
  const data = state.agents;
  const activeAgent = data.agents.find((agent) => ["running", "queued"].includes(agent.status));
  const nextNode = data.nodes.find((node) => ["ready", "pending"].includes(node.status));
  const blockers = [
    ...data.nodes.filter((node) => node.blocked).map((node) => node.title),
    ...data.agents.flatMap((agent) => agent.blockers || []),
  ];
  const phase = activeAgent?.phase || data.core.status;
  const result = data.result || { artifacts: [], changed_files: [] };
  const preview = (result.artifacts || []).find((artifact) => artifact.preview_url);
  $("#viewRoot").innerHTML = `
    <header class="view-head">
      <div>
        <p class="eyebrow">Workspace › Result · Read-only</p>
        <h2>Result</h2>
        <p>Read-only operational state · Plan r${data.plan_revision || "—"}</p>
      </div>
      ${badge(data.core.status)}
    </header>
    ${executionControls()}
    <dl class="execution-summary">
      <div class="fact"><dt>Current phase</dt><dd>${escapeHtml(statusLabel(phase))}</dd></div>
      <div class="fact"><dt>Active work</dt><dd>${escapeHtml(activeAgent?.task || "No active task")}</dd></div>
      <div class="fact"><dt>Next work</dt><dd>${escapeHtml(nextNode?.title || "Determined by the harness")}</dd></div>
      <div class="fact"><dt>Your attention</dt><dd>${state.context?.tool_approval ? "Approval required" : blockers.length ? "Needed for a blocker" : "Not required"}</dd></div>
    </dl>
    <section class="section result-surface">
      <div class="section-head">
        <div><p class="eyebrow">Result</p><h3>${result.status === "completed" ? "Ready to use" : "Recorded output"}</h3><p>${escapeHtml(result.summary || "Outputs appear here as they are verified.")}</p></div>
        ${preview ? `<a class="primary-button result-link" href="${escapeHtml(preview.preview_url)}" target="_blank" rel="noreferrer">Open preview</a>` : ""}
      </div>
      ${(result.artifacts || []).length ? `<ul class="result-artifacts">${result.artifacts.map((artifact) => `<li><span><strong>${escapeHtml(artifact.path || artifact.kind || "Artifact")}</strong><small>${escapeHtml(artifact.kind || "artifact")}${artifact.content_hash ? ` · ${escapeHtml(String(artifact.content_hash).slice(0, 10))}` : ""}</small></span>${artifact.verified ? '<span class="status-badge status-done">Verified</span>' : '<span class="status-badge status-working">Recorded</span>'}</li>`).join("")}</ul>` : `<p class="empty-state compact">No file or preview receipt has been recorded yet.</p>`}
      ${(result.changed_files || []).length ? `<p class="result-files"><strong>Changed files</strong> · ${escapeHtml(result.changed_files.join(", "))}</p>` : ""}
    </section>
    <section class="section">
      <div class="section-head"><div><h3>Work map</h3><p>Read-only hierarchy owned by the harness. Select a node to inspect its history.</p></div></div>
      ${data.nodes.length ? executionTreeMarkup(data.tree?.length ? data.tree : data.nodes.filter((node) => !node.parent_id)) : `<p class="empty-state">No specialist work nodes are active. Current work remains visible in the Plan view.</p>`}
    </section>
    <section class="section">
      <div class="section-head"><div><h3>Active agents</h3><p>Open Advanced for IDs, logs, memory, tool use, and raw results.</p></div></div>
      ${data.agents.length ? `<ul class="agent-list">${data.agents.map((agent) => `
        <li class="agent-row">
          <div class="agent-main">
            <i class="activity-dot ${agent.status === "running" ? "live" : ""}" aria-hidden="true"></i>
            <div class="agent-copy">
              <h4>${escapeHtml(agent.name)}</h4>
              <p>${escapeHtml(agent.task)} · ${escapeHtml(statusLabel(agent.phase || agent.status))}
                ${agent.progress === null ? " · activity reported without an authoritative percentage" : ` · ${agent.progress}% authoritative`}</p>
            </div>
            <div class="row-actions">
              ${badge(agent.status)}
              <button class="quiet-button" data-agent-explain="${escapeHtml(agent.id)}" type="button">Ask for explanation</button>
            </div>
          </div>
          <div class="technical-grid advanced-only">
            <dl class="fact"><dt>Agent run ID</dt><dd>${escapeHtml(agent.id)}</dd></dl>
            <dl class="fact"><dt>Current file</dt><dd>${escapeHtml(agent.current_file || "None")}</dd></dl>
            <dl class="fact"><dt>Retries</dt><dd>${agent.retries}</dd></dl>
            <dl class="fact"><dt>Tools</dt><dd>${escapeHtml((agent.tools || []).join(", ") || "None recorded")}</dd></dl>
            <dl class="fact"><dt>Recent logs</dt><dd>${escapeHtml((agent.logs || []).map((log) => log.summary).join(" · ") || "No logs")}</dd></dl>
            <dl class="fact"><dt>Memory entries</dt><dd>${escapeHtml((agent.memory || []).map((item) => item.title).join(", ") || "None")}</dd></dl>
          </div>
        </li>`).join("")}</ul>` : `<p class="empty-state">No delegated agents are running.</p>`}
    </section>
    ${state.selectedNode ? `<section class="section node-inspector"><div class="section-head"><div><p class="eyebrow">Workspace › Result › Node</p><h3>${escapeHtml(state.selectedNode.title || "Selected node")}</h3><p>${escapeHtml(state.selectedNode.objective || "")}</p></div>${badge(state.selectedNode.status)}</div><dl class="technical-grid"><div class="fact"><dt>Dependencies</dt><dd>${escapeHtml((state.selectedNode.dependencies || []).join(", ") || "None")}</dd></div><div class="fact"><dt>Assigned role</dt><dd>${escapeHtml(state.selectedNode.assigned_role || "Harness")}</dd></div><div class="fact"><dt>Attempts</dt><dd>${Number(state.selectedNode.attempts || 0)}</dd></div></dl><button class="secondary-button" data-node-history="${escapeHtml(state.selectedNode.id)}" type="button">Open node history</button></section>` : ""}
    ${blockers.length ? `
      <section class="section"><div class="section-head"><div><h3>Blockers</h3><p>${escapeHtml(blockers.join(" · "))}</p></div></div></section>` : ""}`;
  $$("[data-agent-explain]").forEach((button) => button.addEventListener("click", () =>
    openExplanationDrawer(button.dataset.agentExplain)));
  $$('[data-node-id]').forEach((button) => button.addEventListener("click", () => {
    state.selectedNode = data.nodes.find((node) => node.id === button.dataset.nodeId) || null;
    renderAgents();
  }));
  $$('[data-node-history]').forEach((button) => button.addEventListener("click", () => {
    state.historyFilter = { entity: button.dataset.nodeHistory };
    navigate("history");
  }));
  $$('[data-execution-action]').forEach((button) => button.addEventListener("click", () => {
    const action = button.dataset.executionAction;
    if (action === "model") return openModelPicker();
    return runWorkspaceAction(action);
  }));
}

function localContinuationPresentation(context = state.context, plan = state.plan) {
  const runtime = {
    ...(context?.runtime || {}),
    ...(plan?.runtime || {}),
  };
  const executionClass = String(runtime.execution_class || "").toLowerCase();
  const recovery = context?.provider_recovery || plan?.provider_recovery || {};
  const diagnostic = [
    runtime.reason,
    runtime.last_error,
    runtime.provider_error,
    runtime.active_operation,
    recovery.error,
  ].filter(Boolean).join(" ");
  const localBoundary = executionClass === "local" && (
    Boolean(recovery.error)
    || /provider|runner|unavailable|failed|quota|network|timeout/i.test(diagnostic)
  );
  if (executionClass === "local" && !localBoundary) {
    return { show: false, label: "" };
  }
  return {
    show: true,
    label: executionClass === "local"
      ? "Try strongest available local model"
      : "Continue with strongest local model",
  };
}

function renderSavedBoundary(view, error) {
  const context = state.context || {};
  const runtime = context.runtime || {};
  const localContinuation = localContinuationPresentation(context);
  const goal = context.goal || {};
  const objective = runtime.objective || goal.objective || "The saved request";
  const phase = String(runtime.phase || context.phase || "checkpoint").replaceAll("_", " ");
  const localRunner = String(runtime.execution_class || "").toLowerCase() === "local";
  const sleepPolicy = String(context.sleep_policy || runtime.sleep_policy || "").toLowerCase();
  const providerRecovery = context.provider_recovery || {};
  const retryNotBefore = Number(
    providerRecovery.retry_not_before
      || runtime.retry_not_before
      || 0
  );
  const autoRecovery = sleepPolicy === "full"
    && (
      Boolean(providerRecovery.full_auto_retry)
      || ["retrying", "waiting"].includes(String(phase).toLowerCase())
    );
  const retrySeconds = retryNotBefore > (Date.now() / 1000)
    ? Math.max(1, Math.ceil(retryNotBefore - (Date.now() / 1000)))
    : 0;
  const liveness = String(runtime.liveness || "").toLowerCase();
  const diagnostic = [runtime.reason, runtime.last_error, runtime.provider_error, runtime.active_operation]
    .filter(Boolean).join(" ");
  const quota = /quota|usage limit|limit exhausted/i.test(diagnostic);
  const rateLimited = /rate limit|too many requests|429/i.test(diagnostic);
  const title = quota
    ? "This model has reached its usage limit"
    : rateLimited
      ? "This model is temporarily rate limited"
      : localRunner && liveness === "network_unavailable"
        ? "Local model runner is unavailable"
        : "Your request is saved";
  const body = autoRecovery
    ? "Full Auto is active. The saved stage will retry automatically; no action is required."
    : quota
    ? "The exact request is preserved. Change the model or wait for the provider limit to reset."
    : rateLimited
      ? "The exact request is preserved. Retry later or change the model."
      : localRunner && liveness === "network_unavailable"
        ? "The exact request is preserved. The local runner must answer before planning can continue."
        : "The exact request is preserved at a recoverable checkpoint.";
  const resumeCopy = autoRecovery
    ? `Automatic recovery is active${retrySeconds ? ` · next retry in about ${retrySeconds}s` : ""}. Use the controls only if you want to intervene.`
    : quota || rateLimited || (localRunner && liveness === "network_unavailable")
    ? "Choose one recovery action below. Do not submit the original prompt again."
    : "Choose a recovery action below. The saved request will be resumed, not replayed.";
  const preferLocal = quota || rateLimited || (localRunner && liveness === "network_unavailable");
  const retryClass = preferLocal ? "secondary-button" : "primary-button";
  const localClass = preferLocal ? "primary-button" : "secondary-button";
  const errorDetail = error?.message && !/does not have a plan/i.test(error.message)
    ? `<p class="subtle advanced-only">${escapeHtml(error.message)}</p>` : "";
  $("#viewRoot").innerHTML = `
    <header class="view-head">
      <div>
        <p class="eyebrow">SAVED CHECKPOINT · ${escapeHtml(phase)}</p>
        <h2>${escapeHtml(title)}</h2>
        <p>${escapeHtml(body)}</p>
      </div>
      <span class="status-badge status-waiting">Recovery</span>
    </header>
    <section class="section current-request" data-testid="current-request">
      <div class="section-head"><div><p class="eyebrow">ORIGINAL REQUEST · SOURCE OF TRUTH</p><h3>Already submitted</h3></div></div>
      <p class="reading">${escapeHtml(objective)}</p>
      <div class="meta-line advanced-only">
        <span>Phase: ${escapeHtml(phase)}</span>
        <span>Model: ${escapeHtml(runtime.model ? `${runtime.provider || "model"}/${runtime.model}` : "saved model")}</span>
      </div>
    </section>
      <section class="section notice-section" role="status">
        <h3>Recover this request</h3>
        <p>${escapeHtml(resumeCopy)}</p>
        ${autoRecovery ? '<p class="subtle auto-recovery-note" data-testid="auto-recovery-note">Full Auto will keep the exact request and quality gates unchanged while it recovers.</p>' : ""}
        <div class="section-actions recovery-actions">
        <button class="${retryClass}" data-recovery-action="retry" type="button">Retry saved stage</button>
        ${localContinuation.show ? `<button class="${localClass}" data-recovery-action="continue_local_model" type="button">${localContinuation.label}</button>` : ""}
        <button class="quiet-button danger-action" data-recovery-action="stop" type="button">Stop safely</button>
      </div>
      ${errorDetail}
    </section>`;
  bindRecoveryActions($("#viewRoot"));
}

function bindRecoveryActions(root) {
  $$('[data-recovery-action]', root).forEach((button) => button.addEventListener("click", async () => {
    const action = button.dataset.recoveryAction;
    if (action === "continue_local_model") return continueWithLocalModel();
    return runWorkspaceAction(action);
  }));
}

function isRoutineHistory(item) {
  return /(^|\.)(heartbeat|poll|snapshot|stream|usage|token|progress)(\.|$)/i.test(item.event_type || "")
    || /^(provider\.|model_)/i.test(item.event_type || "");
}

function renderHistory() {
  const data = state.history || { items: [] };
  const items = data.items || [];
  const selectedGoal = state.historyFilter?.goal || data.goal_id || "";
  const routineCount = items.filter(isRoutineHistory).length;
  $("#viewRoot").innerHTML = `
    <header class="view-head"><div><p class="eyebrow">Durable workflow record</p><h2>History</h2><p>Milestones explain what happened, why, who acted, and what the harness did next.</p></div><span class="status-badge status-waiting">${items.length} milestones</span></header>
    <section class="section history-toolbar"><label>Show <select id="historyFilter"><option value="important">Important milestones</option><option value="all">Everything</option><option value="failure">Failures only</option></select></label>${routineCount ? `<p>${routineCount} routine signal${routineCount === 1 ? "" : "s"} available</p>` : ""}</section>
    ${(data.goals || []).length ? `<section class="section"><div class="section-head"><div><h3>Goals in this session</h3><p>Opening a previous goal is read-only and never changes the active workflow.</p></div></div><ul class="plain-list history-goals">${data.goals.map((goal) => `<li><button class="link-button${selectedGoal === goal.id ? " selected" : ""}" data-history-goal="${escapeHtml(goal.id)}" type="button"><strong>${escapeHtml(goal.objective)}</strong> · ${escapeHtml(statusLabel(goal.status))} · r${escapeHtml(goal.plan_revision || "—")}</button></li>`).join("")}</ul></section>` : ""}
    <section class="section"><ol class="history-list">${items.length ? items.map((item) => `
      <li class="history-item${isRoutineHistory(item) ? " routine-history hidden" : ""}" data-history-type="${escapeHtml(item.event_type)}" data-history-routine="${isRoutineHistory(item) ? "true" : "false"}">
        <button class="history-row" data-history-sequence="${Number(item.sequence)}" type="button">
          <span class="history-sequence">#${Number(item.sequence)}</span><time>${escapeHtml(activityTime(item.created_at))}</time><span class="history-actor">${escapeHtml(item.actor)}</span><strong>${escapeHtml(item.summary)}</strong>
        </button>
        <div class="history-detail hidden" id="history-detail-${Number(item.sequence)}"><p>${escapeHtml(item.why || "No additional reason was recorded.")}</p><dl class="technical-grid"><div class="fact"><dt>Phase</dt><dd>${escapeHtml(item.phase || "—")}</dd></div><div class="fact"><dt>Mutation</dt><dd>${item.workspace_mutated ? "Workspace changed" : "No workspace mutation"}</dd></div><div class="fact"><dt>Next state</dt><dd>${escapeHtml(item.next_state || "—")}</dd></div></dl>${item.evidence?.length ? `<p class="subtle">Evidence: ${escapeHtml(item.evidence.map((e) => typeof e === "string" ? e : e.summary || e.id || "recorded").join(" · "))}</p>` : ""}</div>
      </li>`).join("") : `<li class="empty-state">No durable milestones are recorded for this workflow yet.</li>`}</ol>${data.has_more ? `<button class="secondary-button history-more" id="historyMore" type="button">Load older milestones</button>` : ""}</section>`;
  $$('[data-history-sequence]').forEach((row) => row.addEventListener("click", () => {
    $(`#history-detail-${row.dataset.historySequence}`)?.classList.toggle("hidden");
  }));
  $$('[data-history-goal]').forEach((button) => button.addEventListener("click", () => {
    state.historyFilter = { goal: button.dataset.historyGoal };
    navigate("history");
  }));
  $("#historyFilter")?.addEventListener("change", (event) => {
    $$(".history-item").forEach((item) => {
      const failure = /(fail|error|blocked|paused|uncertain)/i.test(item.dataset.historyType);
      const hidden = event.target.value === "important"
        ? item.dataset.historyRoutine === "true"
        : event.target.value === "failure" && !failure;
      item.classList.toggle("hidden", hidden);
    });
  });
  $("#historyMore")?.addEventListener("click", async () => {
    const cursor = Number(data.next_cursor || 0);
    if (!cursor) return;
    try {
      setBusy(true);
      const params = new URLSearchParams({ after: String(cursor) });
      const goal = state.historyFilter?.goal || data.goal_id;
      if (goal) params.set("goal_id", goal);
      const next = await api(`/history?${params.toString()}`);
      const seen = new Set(items.map((item) => Number(item.sequence)));
      state.history = { ...data, ...next, items: [...items, ...(next.items || []).filter((item) => !seen.has(Number(item.sequence)))] };
      renderHistory();
    } catch (error) {
      announce(error.message, true);
    } finally {
      setBusy(false);
    }
  });
}

function openExplanationDrawer(agentId) {
  openDrawer({
    eyebrow: "Ask the running agent",
    title: "Request an explanation",
    body: `
      <p class="muted">This question is posted to the selected agent's durable inbox. Its response appears when the workflow processes that message.</p>
      <div class="form-field">
        <label for="agentQuestion">Question</label>
        <textarea id="agentQuestion">Explain your current work and any blockers.</textarea>
      </div>
      <div class="editor-actions">
        <button class="primary-button" id="sendAgentQuestion" type="button">Send question</button>
      </div>`,
    onReady: () => {
      const area = $("#agentQuestion");
      autoGrow(area);
      area.addEventListener("input", () => autoGrow(area));
      $("#sendAgentQuestion").addEventListener("click", async () => {
        const question = area.value.trim();
        if (!question) return announce("Write a question first.", true);
        try {
          setBusy(true);
          await api("/agents/explain", {
            method: "POST",
            body: JSON.stringify({ agent_id: agentId, question }),
          });
          closeDrawer();
          announce("Question sent to the agent's durable inbox.");
        } catch (error) {
          announce(error.message, true);
        } finally {
          setBusy(false);
        }
      });
      area.focus();
    },
  });
}

function openDrawer({ eyebrow, title, body, onReady }) {
  state.drawerReturnFocus = document.activeElement;
  $("#drawerEyebrow").textContent = eyebrow;
  $("#drawerTitle").textContent = title;
  $("#drawerBody").innerHTML = body;
  $("#commandMenu").classList.add("hidden");
  $("#drawer").classList.remove("hidden");
  $("#drawerBackdrop").classList.remove("hidden");
  $(".app-shell").inert = true;
  $(".app-shell").setAttribute("aria-hidden", "true");
  document.body.classList.add("drawer-open");
  if (onReady) onReady();
  else $("#drawerClose").focus();
}

function closeDrawer() {
  $("#drawer").classList.add("hidden");
  $("#drawerBackdrop").classList.add("hidden");
  $(".app-shell").inert = false;
  $(".app-shell").removeAttribute("aria-hidden");
  document.body.classList.remove("drawer-open");
  const returnTarget = state.drawerReturnFocus;
  state.drawerReturnFocus = null;
  if (returnTarget?.isConnected) returnTarget.focus({ preventScroll: true });
}

async function loadView(view, force = false) {
  view = canonicalView(view);
  if (state.busy && !force) return;
  if (!force && view === "plan" && (
    state.planDirty || state.editingTask !== null || state.editingSummary
  )) return;
  const silent = !force;
  const loadId = ++state.viewLoadId;
  const activeId = document.activeElement?.id || "";
  const scrollAnchor = captureScrollAnchor();
  $("#errorState").classList.add("hidden");
  if (!silent) {
    $("#loading").classList.remove("hidden");
    $("#viewRoot").classList.add("hidden");
  }
  try {
    if (view === "thread") {
      const snapshot = await api("/thread?limit=200");
      if (loadId !== state.viewLoadId || view !== state.currentView) return;
      const fingerprint = viewFingerprint(snapshot, "thread");
      if (silent && state.viewFingerprints.thread === fingerprint) return;
      state.viewFingerprints.thread = fingerprint;
      state.thread = snapshot;
      renderThread(snapshot);
      restoreScrollAnchor(scrollAnchor);
    } else if (view === "plan") {
      const snapshot = await api("/plan");
      if (loadId !== state.viewLoadId || view !== state.currentView) return;
      const fingerprint = viewFingerprint(snapshot, "plan");
      if (silent && state.viewFingerprints.plan === fingerprint) {
        setConnection(!state.eventSource || state.liveConnected, state.eventSource ? (state.liveConnected ? "Live" : "Reconnecting") : "Connected");
        return;
      }
      if (silent && (state.planDirty || state.editingTask !== null || state.editingSummary)) {
        if (rememberPlanConflict(snapshot, fingerprint)) renderPlan();
        return;
      }
      state.planConflict = null;
      state.pendingUpdates.plan = false;
      state.viewFingerprints.plan = fingerprint;
      state.plan = snapshot;
      if (!state.planDirty || force) {
        state.planModel = makePlanModel(snapshot);
        state.planDirty = Boolean(snapshot.draft);
      }
      renderPlan();
      restoreScrollAnchor(scrollAnchor);
    } else if (view === "review") {
      const snapshot = await api("/review");
      if (loadId !== state.viewLoadId || view !== state.currentView) return;
      const fingerprint = viewFingerprint(snapshot, "review");
      if (silent && state.viewFingerprints.review === fingerprint) return;
      if (silent && reviewIsDirty()) {
        state.pendingUpdates.review = true;
        return;
      }
      state.viewFingerprints.review = fingerprint;
      state.review = snapshot;
      state.reviewDecisions = new Map();
      state.reviewComments = [];
      state.reviewConfirm = false;
      renderReview();
      restoreScrollAnchor(scrollAnchor);
    } else if (view === "history") {
      const params = new URLSearchParams();
      const historyGoal = state.historyFilter?.goal || (state.historyFilter?.entity ? state.context?.goal?.id : "");
      if (historyGoal) params.set("goal_id", historyGoal);
      if (state.historyFilter?.entity) params.set("entity_id", state.historyFilter.entity);
      const query = params.toString();
      const snapshot = await api(`/history${query ? `?${query}` : ""}`);
      if (loadId !== state.viewLoadId || view !== state.currentView) return;
      const fingerprint = viewFingerprint(snapshot, "history");
      if (silent && state.viewFingerprints.history === fingerprint) return;
      state.viewFingerprints.history = fingerprint;
      state.history = snapshot;
      renderHistory();
      restoreScrollAnchor(scrollAnchor);
    } else {
      const snapshot = await api("/execution");
      if (loadId !== state.viewLoadId || view !== state.currentView) return;
      const fingerprint = viewFingerprint(snapshot, "execution");
      if (silent && state.viewFingerprints.execution === fingerprint) return;
      state.viewFingerprints.execution = fingerprint;
      state.agents = snapshot;
      renderAgents();
      restoreScrollAnchor(scrollAnchor);
    }
    $("#loading").classList.add("hidden");
    $("#viewRoot").classList.remove("hidden");
    if (activeId) document.getElementById(activeId)?.focus({ preventScroll: true });
    if (!silent) {
      $("#workspace").focus({ preventScroll: true });
    }
    setConnection(!state.eventSource || state.liveConnected, state.eventSource ? (state.liveConnected ? "Live" : "Reconnecting") : "Connected");
  } catch (error) {
    if (loadId !== state.viewLoadId || view !== state.currentView) return;
    const legacyPreGoalCheckpoint = view === "execution"
      && hasSavedWorkflow(state.context)
      && !state.context?.goal
      && !state.context?.workflow_identity?.goal_id;
    if (error.status === 404 && legacyPreGoalCheckpoint) {
      const recoveryFingerprint = `legacy:${stableFingerprint({
        context: state.context,
        message: error.message,
      })}`;
      if (state.viewFingerprints.execution !== recoveryFingerprint) {
        state.viewFingerprints.execution = recoveryFingerprint;
        renderSavedBoundary(view, error);
      }
      $("#loading").classList.add("hidden");
      $("#viewRoot").classList.remove("hidden");
      if (!silent) $("#workspace").focus({ preventScroll: true });
      setConnection(!state.eventSource || state.liveConnected, state.eventSource ? (state.liveConnected ? "Live" : "Reconnecting") : "Connected");
    } else if (error.status === 404 && view !== "plan" && view !== "history") {
      $("#loading").classList.add("hidden");
      $("#viewRoot").classList.remove("hidden");
      $("#viewRoot").innerHTML = `
        <header class="view-head"><div><p class="eyebrow">${view === "review" ? "No changes gate" : "No result map"}</p>
        <h2 aria-label="${view === "review" ? "Changes" : "Result"}">${view === "review" ? "Changes" : "Result"}</h2></div></header>
        <p class="empty-state">${escapeHtml(error.message)}. Nothing is required from you on this page.</p>`;
    } else {
      showError(error);
    }
  } finally {
    if (loadId === state.viewLoadId) $("#app").setAttribute("aria-busy", "false");
  }
}

function updateComposer() {
  const destination = $("#promptDestination");
  const input = $("#globalPrompt");
  if (!destination || !input) return;
  if (!state.context) {
    destination.textContent = "Connecting to the workspace";
    input.placeholder = "Connecting…";
    input.dataset.acceptsText = "false";
    return;
  }
  const savedWorkflow = hasSavedWorkflow(state.context);
  if (!savedWorkflow) {
    destination.textContent = "Starts a reviewed plan";
    input.placeholder = "Ask GA3BAD to build, fix, or explain…";
    input.dataset.acceptsText = "true";
  } else if (state.context.goal && state.context.capabilities?.can_manage_queue) {
    destination.textContent = `Follow-up only · Current request: ${state.context.goal.objective || "active"}`;
    input.placeholder = "Add a follow-up without interrupting current work…";
    input.dataset.acceptsText = "true";
  } else {
    destination.textContent = "Current request already submitted · use / for controls";
    input.placeholder = "Do not repeat the prompt — type / for controls or wait…";
    input.dataset.acceptsText = "false";
  }
}

function openWorkflowStatus() {
  const context = state.context || {};
  const runtime = context.runtime || {};
  const goal = context.goal || {};
  const localBoundary = String(runtime.execution_class || "").toLowerCase() === "local"
    && String(runtime.liveness || "").toLowerCase() === "network_unavailable";
  const rows = [
    ["Workflow", goal.status || (hasSavedWorkflow(context) ? statusLabel(runtime.phase || context.phase || "active") : "No active goal")],
    ["Phase", runtime.phase || context.phase || "ready"],
    ["Current task", runtime.current_task || "No active task"],
    ["Waiting on", localBoundary ? "model" : runtime.waiting_on || context.waiting_on || "nothing"],
    ["Model", runtime.model ? `${runtime.provider || "model"}/${runtime.model}` : "Not selected"],
    ["Sleep", context.sleep_policy || "off"],
    ...(context.provider_recovery?.automatic_fallback
      ? [["Provider recovery", `Full Auto switched to ${context.provider_recovery.provider || "local"}/${context.provider_recovery.model || "model"}`]]
      : []),
    ["Local continuation", context.local_continuation?.active
      ? `${context.local_continuation.provider}/${context.local_continuation.model} · ${context.local_continuation.abstraction_level} packets · quality gates unchanged`
      : "Off"],
    ["Connection", state.liveConnected ? "Live" : "Polling / reconnecting"],
  ];
  openDrawer({
    eyebrow: "Current checkpoint",
    title: "Workflow status",
    body: `<dl class="technical-grid">${rows.map(([label, value]) => `<div class="fact"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value).replaceAll("_", " "))}</dd></div>`).join("")}</dl>`,
  });
}

async function refreshWorkspaceNow() {
  state.pendingUpdates = {};
  await refreshContext({ enforceGate: true });
  await loadView(state.currentView, true);
  announce("Workspace refreshed to the latest saved state.");
}

async function openToolActivity() {
  if (state.detail !== "advanced") $("#advancedMode").click();
  await navigate("execution");
  $("#advancedLiveDetails")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function continueWithLocalModel() {
  try {
    setBusy(true);
    const result = await api("/actions", {
      method: "POST",
      body: JSON.stringify({
        action: "continue_local_model",
        target_id: savedWorkflowId() || null,
        action_fingerprint: `continue-local:${state.context?.history_cursor || 0}`,
        expected_sequence: null,
        source: "web",
      }),
    });
    announce(result.message || "Continuing with the strongest available local model.");
    await refreshContext({ enforceGate: false });
    await navigate(result.next_view || "execution", { replace: true });
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

function commandCatalog() {
  const status = String(state.context?.goal?.status || "").toLowerCase();
  const hasWorkflow = hasSavedWorkflow(state.context);
  const runtimePhase = String(state.context?.runtime?.phase || state.context?.phase || "").toLowerCase();
  const paused = ["paused", "blocked"].includes(status) || ["paused", "retrying", "waiting"].includes(runtimePhase);
  const finished = ["completed", "failed", "cancelled"].includes(status) || runtimePhase === "completed";
  return [
    { name: "/thread", description: "Open the unified Codex-style workflow thread", allowed: true, run: () => navigate("thread") },
    { name: "/model", description: "Change the model at a safe checkpoint", allowed: true, run: openModelPicker },
    { name: "/project settings", description: "Reuse or reconfigure the saved project setup", allowed: true, run: openProjectSettings },
    { name: "/continue with local model", description: "Use the strongest local model with smaller remaining-task packets and unchanged quality gates", allowed: hasWorkflow && !finished, run: continueWithLocalModel },
    { name: "/tree", description: "Open the work map and node hierarchy", allowed: hasWorkflow, run: () => navigate("execution") },
    { name: "/agents", description: "Open active agents, ownership, and progress", allowed: hasWorkflow, run: () => navigate("execution") },
    { name: "/tools", description: "Show live tool activity and model diagnostics", allowed: hasWorkflow, run: openToolActivity },
    { name: "/activity", description: "Open the verified live activity timeline", allowed: true, run: openActivityTimeline },
    { name: "/status", description: "Show the current checkpoint, task, model, and connection", allowed: true, run: openWorkflowStatus },
    { name: "/pause", description: "Pause after the current safe boundary", allowed: hasWorkflow && !paused && !finished, run: () => runWorkspaceAction("pause") },
    { name: "/resume", description: "Resume the saved workflow", allowed: hasWorkflow && paused, run: () => runWorkspaceAction("resume") },
    { name: "/retry", description: "Retry the saved failed stage once", allowed: hasWorkflow && paused, run: () => runWorkspaceAction("retry") },
    { name: "/stop", description: "Stop the active workflow safely", allowed: hasWorkflow && !finished, run: () => runWorkspaceAction("stop") },
    { name: "/sleep", description: "Choose Off, Safe Auto, or Full Auto", allowed: true, run: openSleepControls },
    { name: "/refresh", description: "Discard stale presentation state and load the latest saved state", allowed: true, run: refreshWorkspaceNow },
    { name: "/plan", description: "Open the plan and request surface", allowed: true, run: () => navigate("plan") },
    { name: "/result", description: "Open the recorded result and work map", allowed: hasWorkflow, run: () => navigate("execution") },
    { name: "/changes", description: "Open recorded file changes and review decisions", allowed: hasWorkflow, run: () => navigate("review") },
    { name: "/history", description: "Open the durable workflow history", allowed: true, run: () => navigate("history") },
    { name: "/simple", description: "Show the essential workflow only", allowed: true, run: () => $("#simpleMode").click() },
    { name: "/advanced", description: "Show diagnostics and model output", allowed: true, run: () => $("#advancedMode").click() },
  ];
}

function matchingCommands() {
  const value = String($("#globalPrompt")?.value || "").trimStart();
  if (!value.startsWith("/") || value.includes("\n")) return [];
  const query = value.slice(1).toLowerCase();
  return commandCatalog().filter((item) => item.name.slice(1).startsWith(query));
}

function renderCommandMenu() {
  const root = $("#commandMenu");
  if (!root) return;
  const commands = matchingCommands();
  if (!commands.length) {
    root.classList.add("hidden");
    root.innerHTML = "";
    state.commandSelection = 0;
    return;
  }
  state.commandSelection = Math.min(state.commandSelection, commands.length - 1);
  root.innerHTML = commands.map((item, index) => `
    <button class="command-option${index === state.commandSelection ? " selected" : ""}" data-command="${item.name}" role="option" aria-selected="${index === state.commandSelection}" type="button"${item.allowed ? "" : " disabled"}>
      <strong>${item.name}</strong><span>${escapeHtml(item.description)}</span><small>${item.allowed ? "Available" : "Not at this stage"}</small>
    </button>`).join("");
  root.classList.remove("hidden");
  $$('[data-command]', root).forEach((button) => button.addEventListener("click", () => executeCommand(button.dataset.command)));
}

async function executeCommand(name) {
  const command = commandCatalog().find((item) => item.name === name);
  if (!command || !command.allowed) {
    announce("That command is not available in the current workflow state.", true);
    return;
  }
  $("#commandMenu").classList.add("hidden");
  $("#globalPrompt").value = "";
  autoGrow($("#globalPrompt"));
  await command.run();
}

async function runWorkspaceAction(action, value = null) {
  try {
    setBusy(true);
    const result = await api("/actions", {
      method: "POST",
      body: JSON.stringify({
        action,
        target_id: savedWorkflowId() || null,
        value,
        expected_sequence: ["retry", "resume"].includes(action)
          ? (state.context?.history_cursor ?? null)
          : null,
        source: "web",
      }),
    });
    announce(result.message || "Action accepted.");
    await refreshContext({ enforceGate: false });
    if (result.next_view) await navigate(result.next_view, { replace: true });
    else await loadView(state.currentView, true);
    return true;
  } catch (error) {
    if (["stale_state", "rate_limited", "quota_exhausted", "network_offline", "runtime_unreachable", "runtime_error"].includes(error.code)) {
      // A failed action must never leave a modal covering the recovery surface.
      // Close the drawer before showing the named boundary so Retry/History and
      // the reconnect status remain reachable even when the runtime disappears.
      closeDrawer();
      showError(error);
    }
    else announce(error.message, true);
    return false;
  } finally {
    setBusy(false);
  }
}

function openSleepControls() {
  const policy = String(state.context?.sleep_policy || (state.context?.sleep_enabled ? "safe" : "off"));
  openDrawer({
    eyebrow: "Unattended execution",
    title: "Sleep mode",
    body: `
      <p class="checkpoint-note">Current mode: <strong>${policy === "full" ? "Full Auto" : policy === "safe" ? "Safe Auto" : "Off"}</strong></p>
      <div class="sleep-mode-list" role="list" aria-label="Sleep modes">
        <button class="sleep-mode-choice${policy === "off" ? " selected" : ""}" data-sleep-mode="off" type="button">
          <span><strong>Off</strong><small>Ask before every action that needs approval.</small></span><em>${policy === "off" ? "Current" : "Select"}</em>
        </button>
        <button class="sleep-mode-choice${policy === "safe" ? " selected" : ""}" data-sleep-mode="safe" type="button">
          <span><strong>Safe Auto</strong><small>Continue reversible checks, tests, and previews automatically.</small></span><em>${policy === "safe" ? "Current" : "Select"}</em>
        </button>
        <section class="full-auto-choice${policy === "full" ? " selected" : ""}">
          <div><strong>Full Auto</strong><p>Automatically approve critic-reviewed plans, deterministic planning questions, and every tool action in this workspace, including shell commands and installs. Every decision is recorded in History.</p></div>
          <label for="fullAutoConfirmation">Type <code>FULL AUTO</code> to confirm</label>
          <div class="full-auto-confirm">
            <input id="fullAutoConfirmation" autocomplete="off" spellcheck="false" placeholder="FULL AUTO"${policy === "full" ? " disabled" : ""}>
            <button id="enableFullAuto" class="danger-button" type="button" disabled>${policy === "full" ? "Full Auto is on" : "Enable Full Auto"}</button>
          </div>
        </section>
      </div>`,
    onReady: () => {
      $$('[data-sleep-mode]', $("#drawerBody")).forEach((button) => button.addEventListener("click", async () => {
        const next = button.dataset.sleepMode;
        if (next === policy) return closeDrawer();
        if (await runWorkspaceAction(next === "safe" ? "sleep_on" : "sleep_off")) closeDrawer();
      }));
      const confirmation = $("#fullAutoConfirmation");
      const enable = $("#enableFullAuto");
      confirmation?.addEventListener("input", () => {
        enable.disabled = confirmation.value !== "FULL AUTO";
      });
      enable?.addEventListener("click", async () => {
        if (confirmation.value !== "FULL AUTO") return;
        if (await runWorkspaceAction("sleep_full_on", "FULL AUTO")) closeDrawer();
      });
    },
  });
}

async function recoverSubmittedRequestAfterFailure(input) {
  try {
    const context = reconcileResolvedApproval(await api("/workspace"));
    if (!hasSavedWorkflow(context)) return false;
    state.context = context;
    input.value = "";
    autoGrow(input);
    // This branch only runs after the first prompt failed.  Even when the
    // backend still reports `required_view=plan`, Result is the one surface
    // that contains Retry, local continuation, and Stop safely controls.
    await navigate("execution", { replace: true, gate: true });
    announce("Your request is saved at a recoverable checkpoint. Do not submit it again.");
    return true;
  } catch {
    return false;
  }
}

async function submitGlobalPrompt() {
  const input = $("#globalPrompt");
  const text = String(input.value || "").trim();
  if (!text || state.busy) return;
  const normalizedCommand = text.toLowerCase().replace(/\s+/g, " ");
  const exactCommand = commandCatalog().find((item) => item.name === normalizedCommand);
  if (text.startsWith("/") && exactCommand) {
    await executeCommand(exactCommand.name);
    return;
  }
  try {
    setBusy(true);
    if (!hasSavedWorkflow(state.context)) {
      await api("/plan/request", { method: "POST", body: JSON.stringify({ request: text }) });
      input.value = "";
      announce("Request saved. GA3BAD is preparing a plan for your review.");
      await refreshContext({ enforceGate: false });
      await navigate("plan", { replace: true });
    } else if (state.context.goal && state.context.capabilities?.can_manage_queue) {
      const result = await api("/queue", { method: "POST", body: JSON.stringify({ text }) });
      input.value = "";
      announce(result.duplicate
        ? (result.duplicate_of === "active_request" ? "That is already the active request. Nothing was queued." : "That follow-up is already queued.")
        : "Follow-up queued. Current work was not interrupted.");
      await refreshContext({ enforceGate: false });
      await loadView(state.currentView, true);
    } else {
      announce("This checkpoint cannot accept a follow-up yet. Use /pause, /model, or wait for the next safe boundary.", true);
    }
    autoGrow(input);
  } catch (error) {
    const recovered = await recoverSubmittedRequestAfterFailure(input);
    if (!recovered) {
      if (error.code) showError(error);
      else announce(error.message, true);
    }
  } finally {
    setBusy(false);
    updateComposer();
  }
}

async function openModelPicker() {
  openDrawer({
    eyebrow: "Session model",
    title: "Choose the right model",
    body: '<p class="empty-state">Checking local and configured cloud models…</p>',
  });
  try {
    const catalog = await api("/models");
    state.modelCatalog = catalog;
    const models = catalog.models || [];
    $("#drawerBody").innerHTML = `
      <p class="checkpoint-note${catalog.safe_checkpoint ? "" : " is-blocked"}">${catalog.safe_checkpoint
        ? "You are at a safe checkpoint. Switching keeps the saved plan and completed work."
        : "Pause and wait for active agents to reach a safe checkpoint before switching."}</p>
      <div class="model-list">${models.length ? models.map((item) => `
        <button class="model-choice${item.selected ? " selected" : ""}" data-model-id="${escapeHtml(item.id)}" type="button"${catalog.safe_checkpoint && !item.selected ? "" : " disabled"}>
          <span><strong>${escapeHtml(item.display_name || item.model)}</strong><small>${escapeHtml(item.provider)} · ${escapeHtml(item.execution_class)} · ${(item.capabilities || []).map(escapeHtml).join(", ") || "capabilities unknown"}</small></span>
          <small>${item.selected ? "Current" : "Select"}</small>
        </button>`).join("") : '<p class="empty-state">No tool-capable model is available.</p>'}</div>
      ${(catalog.diagnostics || []).map((item) => `<p class="model-diagnostic">${escapeHtml(item.source)} · ${escapeHtml(item.message)}</p>`).join("")}`;
    $$('[data-model-id]', $("#drawerBody")).forEach((button) => button.addEventListener("click", () => switchModel(button.dataset.modelId)));
  } catch (error) {
    $("#drawerBody").innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p><div class="error-actions"><button id="retryModels" class="secondary-button" type="button">Retry discovery</button></div>`;
    $("#retryModels").addEventListener("click", openModelPicker);
  }
}

async function reconfigureProject(action, value) {
  try {
    setBusy(true);
    const result = await api("/actions", {
      method: "POST",
      body: JSON.stringify({
        action,
        target_id: String(value || ""),
        value: String(value || ""),
        action_fingerprint: `project-settings:${action}:${value}:${state.context?.history_cursor || 0}`,
        expected_sequence: null,
        source: "web",
      }),
    });
    closeDrawer();
    announce(result.message || "Project settings updated.");
    await refreshContext({ enforceGate: false });
    await loadView(state.currentView, true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function openProjectSettings() {
  openDrawer({
    eyebrow: "Durable project profile",
    title: "Project settings",
    body: '<p class="empty-state">Loading the saved setup…</p>',
  });
  try {
    const settings = await api("/project-settings");
    const model = settings.model || {};
    const protection = settings.protection || {};
    const safe = Boolean(settings.safe_checkpoint);
    const modeSafe = safe && settings.mode_reconfigurable !== false;
    const currentProtection = String(protection.configured_provider || protection.tier || "snapshot");
    const currentMode = String(settings.mode || "working");
    const currentAccess = String(settings.access_level || "normal");
    const currentConcurrency = Math.max(1, Math.min(8, Number(settings.concurrency || 1)));
    const protectionOptions = [
      ["github", "GitHub + local checkpoints", "Use GitHub when connected and keep local recovery points."],
      ["local_git", "Local Git only", "Keep multi-step undo without pushing to a remote."],
      ["snapshot", "Snapshots only", "Use current-run recovery without version history."],
    ];
    $("#drawerBody").innerHTML = `
      <p class="checkpoint-note${safe ? "" : " is-blocked"}">${escapeHtml(
        safe ? settings.reopen_behavior : "Reconfiguration is paused while active work owns this checkpoint. You can inspect the saved values now."
      )}</p>
      <dl class="settings-summary">
        <div><dt>Model</dt><dd>${escapeHtml(`${model.provider || "model"}/${model.model || "saved"}`)} · ${escapeHtml(model.execution_class || "local")}</dd></div>
        <div><dt>Workflow default</dt><dd>${escapeHtml(currentMode === "plan" ? "Plan" : "Working")}</dd></div>
        <div><dt>Permissions</dt><dd>${escapeHtml(currentAccess)}</dd></div>
        <div><dt>Agent capacity</dt><dd>${currentConcurrency}</dd></div>
        <div><dt>Protection</dt><dd>${escapeHtml(currentProtection)} · ${escapeHtml(protection.detail || "")}</dd></div>
      </dl>
      <section class="settings-group">
        <div class="section-head"><div><h3>Model</h3><p>Changing it keeps the saved workflow and applies only at a safe checkpoint.</p></div></div>
        <button id="projectSettingsModel" class="secondary-button" type="button">Change model</button>
      </section>
      <section class="settings-group">
        <div class="section-head"><div><h3>Project protection</h3><p>Reopening this project will reuse the selected protection tier.</p></div></div>
        <div class="settings-choice-list">${protectionOptions.map(([value, label, description]) => `
          <button class="settings-choice${currentProtection === value ? " selected" : ""}" data-project-protection="${value}" type="button"${safe && currentProtection !== value ? "" : " disabled"}>
            <span><strong>${label}</strong><small>${description}</small></span><em>${currentProtection === value ? "Current" : "Use"}</em>
          </button>`).join("")}</div>
      </section>
      <section class="settings-group">
        <div class="section-head"><div><h3>Workflow defaults</h3><p>${escapeHtml(settings.mode_lock_reason || "These defaults are used for the next request; an active workflow keeps its own durable mode.")}</p></div></div>
        <div class="settings-inline-actions">
          <button class="quiet-button" data-project-mode="normal" type="button"${modeSafe && currentMode !== "working" ? "" : " disabled"}>Working</button>
          <button class="quiet-button" data-project-mode="plan" type="button"${modeSafe && currentMode !== "plan" ? "" : " disabled"}>Plan</button>
          <label class="settings-select">Capacity
            <select id="projectConcurrency"${safe ? "" : " disabled"}>${Array.from({length: 8}, (_, index) => index + 1).map((value) => `<option value="${value}"${value === currentConcurrency ? " selected" : ""}>${value}</option>`).join("")}</select>
          </label>
        </div>
      </section>
      <section class="settings-group">
        <div class="section-head"><div><h3>Permissions</h3><p>Full access still requires the configured Docker sandbox and remains auditable.</p></div></div>
        <div class="settings-inline-actions">
          <button class="quiet-button" data-project-permission="normal" type="button"${safe && currentAccess !== "normal" ? "" : " disabled"}>Normal</button>
          <button class="quiet-button" data-project-permission="full" type="button"${safe && currentAccess !== "full" ? "" : " disabled"}>Full</button>
        </div>
      </section>`;
    $("#projectSettingsModel").addEventListener("click", () => {
      closeDrawer();
      openModelPicker();
    });
    $$('[data-project-protection]', $("#drawerBody")).forEach((button) => button.addEventListener("click", () =>
      reconfigureProject("reconfigure_protection", button.dataset.projectProtection)));
    $$('[data-project-mode]', $("#drawerBody")).forEach((button) => button.addEventListener("click", () =>
      reconfigureProject("reconfigure_mode", button.dataset.projectMode)));
    $$('[data-project-permission]', $("#drawerBody")).forEach((button) => button.addEventListener("click", () =>
      reconfigureProject("reconfigure_permissions", button.dataset.projectPermission)));
    $("#projectConcurrency").addEventListener("change", (event) =>
      reconfigureProject("reconfigure_concurrency", event.target.value));
  } catch (error) {
    const message = String(error?.message || "");
    if (error?.status === 404 || message.includes("404")) {
      $("#drawerBody").innerHTML = `
        <p class="checkpoint-note is-blocked">This workspace is connected to an older GA3BAD owner.</p>
        <div class="compatibility-notice">
          <h3>Project settings are unavailable on this owner</h3>
          <p>The saved workflow is unchanged. Restart the GA3BAD owner once, then reopen this workspace to load the durable project profile and its reconfiguration controls.</p>
          <p class="muted">The current Web page cannot safely change settings until the owner and Web versions match.</p>
        </div>`;
    } else {
      $("#drawerBody").innerHTML = `<p class="empty-state">${escapeHtml(message || "Project settings could not be loaded.")}</p>`;
    }
  }
}

async function switchModel(modelId) {
  try {
    setBusy(true);
    const result = await api("/actions", {
      method: "POST",
      body: JSON.stringify({
        action: "switch_model",
        target_id: modelId,
        value: modelId,
        action_fingerprint: `model:${modelId}:${state.context?.history_cursor || 0}`,
        expected_sequence: null,
        source: "web",
      }),
    });
    closeDrawer();
    announce(result.message || "Model changed.");
    await refreshContext({ enforceGate: false });
    await loadView(state.currentView, true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

function wireGlobalEvents() {
  $("#simpleMode").addEventListener("click", () => {
    state.detail = "simple";
    localStorage.setItem("ga3bad-detail", state.detail);
    renderChrome();
  });
  $("#advancedMode").addEventListener("click", () => {
    state.detail = "advanced";
    localStorage.setItem("ga3bad-detail", state.detail);
    renderChrome();
  });
  $$(".nav-button").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
  $("#workspaceHome").addEventListener("click", () => navigate("plan"));
  $("#inspectorToggle")?.addEventListener("click", () => {
    state.inspectorOpen = !state.inspectorOpen;
    localStorage.setItem("ga3bad-inspector", state.inspectorOpen ? "open" : "closed");
    $("#inspectorToggle").setAttribute("aria-expanded", String(state.inspectorOpen));
    renderInspector();
    if (state.inspectorOpen) loadInspector();
  });
  $("#inspectorClose")?.addEventListener("click", () => {
    state.inspectorOpen = false;
    localStorage.setItem("ga3bad-inspector", "closed");
    $("#inspectorToggle").setAttribute("aria-expanded", "false");
    renderInspector();
  });
  $$(".inspector-tab").forEach((tab) => tab.addEventListener("click", () => {
    state.inspectorSection = tab.dataset.inspectorSection || "environment";
    renderInspector();
    loadInspector(state.inspectorSection);
  }));
  $("#railSearch")?.addEventListener("input", () => renderProjectSessions());
  $("#newTaskButton")?.addEventListener("click", () => {
    state.currentView = "plan";
    history.pushState({ view: "plan" }, "", `/sessions/${encodeURIComponent(sessionId)}/plan`);
    $("#globalPrompt").focus();
    announce("New task composer is ready for this project.");
    updateComposer();
  });
  $("#attachButton")?.addEventListener("click", () => openDrawer({
    eyebrow: "Thread sources",
    title: "Attach context",
    body: '<p class="muted">Sources are read from the project workspace and remain visible in Inspector → Sources.</p><button class="secondary-button" data-inspector-source type="button">Open sources</button>',
    onReady: () => $("[data-inspector-source]")?.addEventListener("click", () => { closeDrawer(); state.inspectorOpen = true; state.inspectorSection = "sources"; renderInspector(); loadInspector("sources"); }),
  }));
  $("#accessButton")?.addEventListener("click", openProjectSettings);
  $("#modeButton")?.addEventListener("click", openProjectSettings);
  $("#sleepComposer")?.addEventListener("click", openSleepControls);
  $("#modelComposer")?.addEventListener("click", openModelPicker);
  $("#modelButton").addEventListener("click", openModelPicker);
  $("#projectSettingsButton").addEventListener("click", openProjectSettings);
  $("#attentionAction").addEventListener("click", async () => {
    const button = $("#attentionAction");
    if (button.dataset.kind) {
      try {
        setBusy(true);
        const result = await api("/actions", { method: "POST", body: JSON.stringify({
          action: button.dataset.kind, target_id: savedWorkflowId() || null,
          expected_sequence: state.context?.history_cursor || 0, source: "web",
        }) });
        announce(result.message || "The saved workflow is resuming.");
        await refreshContext({ enforceGate: false });
        await loadView(state.currentView, true);
      } catch (error) { announce(error.message, true); }
      finally { setBusy(false); }
      return;
    }
    navigate(button.dataset.view);
  });
  $("#requiredActionButton").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const kind = button.dataset.actionKind;
    if (kind === "approve_plan") return navigate("plan");
    if (kind === "allow_tool") return resolveToolApproval("allow_once");
    if (!kind) return;
    try {
      setBusy(true);
      const answer = kind === "answer" ? String($("#requiredAnswer")?.value || "").trim() : "";
      if (kind === "answer" && !answer) {
        announce("Answer the current question before submitting.", true);
        return;
      }
      const result = await api("/actions", { method: "POST", body: JSON.stringify({
        action: kind, target_id: kind === "answer" ? (state.context?.required_action?.question?.id || null) : (savedWorkflowId() || null),
        value: answer || null,
        action_fingerprint: button.dataset.fingerprint || "", expected_sequence: state.context?.history_cursor || 0, source: "web",
      }) });
      announce(result.message || "Action accepted. The saved workflow is resuming.");
      await refreshContext({ enforceGate: false });
      await loadView(state.currentView, true);
    } catch (error) { announce(error.message, true); }
    finally { setBusy(false); }
  });
  $("#toolAllow").addEventListener("click", () => resolveToolApproval("allow_once"));
  $("#toolAllowSession").addEventListener("click", () => resolveToolApproval("allow_session"));
  $("#toolDeny").addEventListener("click", () => resolveToolApproval("deny"));
  $("#toolStop").addEventListener("click", () => runWorkspaceAction("stop"));
  $("#refreshButton").addEventListener("click", async () => {
    await refreshContext({ enforceGate: true });
    await loadView(state.currentView, true);
    announce("Workspace refreshed.");
  });
  $("#sleepToggle").addEventListener("click", openSleepControls);
  $("#sleepTopbar").addEventListener("click", openSleepControls);
  window.addEventListener("scroll", () => {
    state.scrollingUntil = Date.now() + 800;
  }, { passive: true });
  window.addEventListener("resize", () => {
    // A desktop Inspector becomes a fixed mobile drawer at narrow widths.
    // Close it on an automatic viewport transition so it never covers an
    // approval or composer control; the user can reopen it explicitly.
    if (window.innerWidth <= 720 && state.inspectorOpen) {
      state.inspectorOpen = false;
      localStorage.setItem("ga3bad-inspector", "closed");
      renderInspector();
    }
  }, { passive: true });
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  $("#openActivity").addEventListener("click", openActivityTimeline);
  const globalPrompt = $("#globalPrompt");
  globalPrompt.addEventListener("input", () => {
    autoGrow(globalPrompt);
    state.commandSelection = 0;
    renderCommandMenu();
  });
  globalPrompt.addEventListener("keydown", (event) => {
    const commands = matchingCommands();
    if (!$("#commandMenu").classList.contains("hidden") && commands.length) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const delta = event.key === "ArrowDown" ? 1 : -1;
        state.commandSelection = (state.commandSelection + delta + commands.length) % commands.length;
        renderCommandMenu();
        return;
      }
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        executeCommand(commands[state.commandSelection].name);
        return;
      }
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitGlobalPrompt();
    }
  });
  $("#promptSubmit").addEventListener("click", submitGlobalPrompt);
  document.addEventListener("keydown", (event) => {
    const drawer = $("#drawer");
    if (!drawer.classList.contains("hidden")) {
      if (event.key === "Escape") {
        event.preventDefault();
        closeDrawer();
        return;
      }
      if (event.key === "Tab") {
        const focusable = $$('button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [href], [tabindex]:not([tabindex="-1"])', drawer)
          .filter((node) => node.getClientRects().length > 0);
        if (focusable.length) {
          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
          }
        }
      }
    }
    if (event.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "")) {
      event.preventDefault();
      globalPrompt.focus();
      if (!globalPrompt.value) {
        globalPrompt.value = "/";
        renderCommandMenu();
      }
    }
  });
  window.addEventListener("offline", () => {
    setConnection(false, "Offline");
    if (state.context) {
      state.context = { ...state.context, runtime: { ...(state.context.runtime || {}), liveness: "network_unavailable" } };
      renderLiveWorkflow();
    }
  });
  window.addEventListener("online", () => {
    setConnection(false, "Reconnecting");
    refreshContext({ enforceGate: false }).catch(() => {});
  });
  window.addEventListener("popstate", () => {
    const view = canonicalView(location.pathname.split("/").filter(Boolean)[2] || "plan");
    state.currentView = view;
    renderChrome();
    loadView(view, true);
  });
}

async function boot() {
  wireGlobalEvents();
  renderChrome();
  try {
    await refreshContext({ enforceGate: false });
    if (state.context.required_view && state.currentView !== "thread") {
      state.currentView = canonicalView(state.context.required_view);
      history.replaceState({ view: state.currentView }, "", `/sessions/${sessionId}/${state.currentView}`);
    }
    renderChrome();
    startLiveEvents();
    if (state.inspectorOpen) loadInspector();
    state.liveClock = setInterval(() => renderLiveWorkflow(), 1000);
    await loadView(state.currentView, true);
    state.contextTimer = setInterval(async () => {
      await refreshContext({ enforceGate: true });
      const canRefreshView = state.currentView === "execution" || state.currentView === "thread"
        || (state.currentView === "plan" && !state.planDirty
          && state.editingTask === null && !state.editingSummary);
      if (!state.busy && canRefreshView && document.visibilityState === "visible") {
        await loadView(state.currentView, false);
      }
      if (state.inspectorOpen && !state.busy) loadInspector(state.inspectorSection);
    }, 10000);
  } catch (error) {
    showError(error);
  }
}

boot();

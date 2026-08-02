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
const initialView = ["plan", "review", "agents"].includes(pathParts[2]) ? pathParts[2] : "plan";
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
  busy: false,
  draggedPromptId: null,
  contextTimer: null,
  viewFingerprints: {},
  pendingUpdates: {},
  scrollingUntil: 0,
};

function statusLabel(status) {
  const labels = {
    pending: "Not started",
    ready: "Ready",
    in_progress: "Working",
    verifying: "Checking",
    reviewing: "Reviewing",
    testing: "Checking",
    fixing: "Fixing",
    integrating: "Integrating",
    blocked: "Needs attention",
    failed: "Needs attention",
    uncertain: "Needs attention",
    completed: "Done",
    accepted: "Approved",
    pending_approval: "Awaiting approval",
    open: "Ready for review",
  };
  return labels[status] || String(status || "Unknown").replaceAll("_", " ");
}

function statusClass(status) {
  if (["completed", "accepted", "approved", "integrated"].includes(status)) return "done";
  if (["blocked", "failed", "uncertain", "changes_requested"].includes(status)) return "attention";
  if (["pending", "pending_approval", "open"].includes(status)) return "waiting";
  if (["ready", "in_progress", "verifying", "reviewing", "testing", "fixing", "integrating", "running"].includes(status)) return "working";
  return "";
}

function badge(status) {
  return `<span class="status-badge status-${statusClass(status)}">${escapeHtml(statusLabel(status))}</span>`;
}

function compactList(items, emptyText = "No additional requirements were recorded.") {
  const values = (items || []).filter((item) => String(item || "").trim());
  if (!values.length) return `<p class="subtle">${escapeHtml(emptyText)}</p>`;
  return `<ul class="plain-list">${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function executionPresentation(plan) {
  const recursive = plan.execution_strategy === "recursive";
  const decision = plan.strategy_decision || {};
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
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}${path}`, {
    credentials: "same-origin",
    ...options,
    headers,
  });
  let payload = {};
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function announce(message, isError = false) {
  $("#liveRegion").textContent = message;
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
  root.classList.remove("hidden");
  root.innerHTML = `
    <div>
      <p class="eyebrow">The workspace could not load</p>
      <h2>${escapeHtml(error.message || error)}</h2>
      <p>Your durable work was not changed. Refresh after the local runtime is available.</p>
      <button class="secondary-button" id="retryLoad" type="button">Try again</button>
    </div>`;
  $("#retryLoad").addEventListener("click", () => loadView(state.currentView, true));
  setConnection(false);
}

function setConnection(connected) {
  const node = $("#connection");
  node.classList.toggle("connected", connected);
  node.classList.toggle("offline", !connected);
  $("span", node).textContent = connected ? "Connected" : "Offline";
}

function renderChrome() {
  document.body.classList.toggle("advanced", state.detail === "advanced");
  $("#simpleMode").classList.toggle("selected", state.detail === "simple");
  $("#advancedMode").classList.toggle("selected", state.detail === "advanced");
  $$(".nav-button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.view === state.currentView);
    button.setAttribute("aria-current", button.dataset.view === state.currentView ? "page" : "false");
  });
  if (!state.context) return;
  const runtime = state.context.runtime || {};
  const mode = String(runtime.session_mode || state.context.mode || "ready").replaceAll("_", " ");
  const route = String(runtime.route || state.context.route || "pending");
  const strategy = String(runtime.execution_strategy || state.context.execution_strategy || "pending");
  const phase = String(runtime.phase || state.context.phase || "ready").replaceAll("_", " ");
  const model = runtime.model ? ` · ${runtime.provider || "model"}/${runtime.model}` : "";
  $("#sessionMeta").textContent = `Session ${mode} · Route ${route} · Execution ${strategy} · Status ${phase}${model}`;
  Object.entries(state.context.navigation || {}).forEach(([view, item]) => {
    const node = $(`[data-badge="${view}"]`);
    const count = Number(item.badge || 0);
    node.textContent = String(count);
    node.classList.toggle("hidden", count === 0);
  });
  const attention = state.context.attention;
  const root = $("#attention");
  root.classList.remove("hidden");
  $("#attentionEyebrow").textContent = attention.eyebrow;
  $("#attentionTitle").textContent = attention.title;
  $("#attentionBody").textContent = attention.body;
  const action = $("#attentionAction");
  action.classList.toggle("hidden", !attention.action);
  if (attention.action) {
    action.textContent = attention.action.label;
    action.dataset.view = attention.action.view;
  }
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
  return window.scrollY > 24
    || Date.now() < state.scrollingUntil
    || Boolean(focused && $("#viewRoot").contains(focused));
}

function showActivityBanner(show = true) {
  $("#activityBanner").classList.toggle("hidden", !show);
}

async function refreshContext({ enforceGate = true } = {}) {
  try {
    const context = await api("/workspace");
    state.context = context;
    setConnection(true);
    if (enforceGate && context.required_view && context.required_view !== state.currentView) {
      await navigate(context.required_view, { gate: true });
      return;
    }
    renderChrome();
  } catch (error) {
    setConnection(false);
    if (!state.context) throw error;
  }
}

async function navigate(view, { replace = false, gate = false } = {}) {
  if (!["plan", "review", "agents"].includes(view)) return;
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
  const model = state.planModel;
  const canEdit = Boolean(plan.capabilities.can_edit);
  const phase = planPhase(plan);
  const active = plan.tasks.find((task) => ["in_progress", "verifying", "blocked"].includes(task.status));
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
        <p class="subtle runtime-context">Mode: ${escapeHtml((plan.runtime?.session_mode || plan.interaction_mode || "working"))} · Route: ${escapeHtml(plan.runtime?.route || "goal")} · Execution: ${escapeHtml(plan.runtime?.execution_strategy || plan.execution_strategy || "pending")} · Model: ${escapeHtml(plan.runtime?.model || plan.capability_envelope?.model || "unknown")}</p>
        <p>Revision ${plan.revision} · ${model.tasks.length} tasks · ${escapeHtml(plan.interaction_mode || "working")}</p>
      </div>
      ${badge(plan.status)}
    </header>

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
      ${(plan.semantic_goal?.constraints || []).length ? `<details class="advanced-section"><summary>Constraints and exclusions</summary><div class="contract-columns"><div><h4>Constraints</h4>${compactList(plan.semantic_goal.constraints)}</div><div><h4>Exclusions</h4>${compactList(plan.semantic_goal.exclusions)}</div></div></details>` : ""}
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
          <h3>${canEdit ? "Execution order" : "Plan tasks"}</h3>
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
          <h3>Work is running in the terminal</h3>
          <p>Plan Studio is now read-only. Return to the terminal to follow live steps, send guidance, pause, or inspect output.</p>
        </div>
        <button class="primary-button" data-action="return-to-terminal" type="button">Return to terminal</button>
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
          <p>Add requests without interrupting current work. Only waiting requests can move.</p>
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
      <div class="queue-compose">
        <textarea id="queuePrompt" placeholder="Add a request for after the current work"
                  aria-label="New queued request"></textarea>
        <button class="secondary-button" data-action="enqueue" type="button">Add request</button>
      </div>
      <p class="subtle">Ctrl/Cmd + Enter adds the request. Execution depth is selected automatically for the configured model.</p>
    </section>`;
}

function markPlanDirty() {
  state.planDirty = true;
  state.planConfirm = false;
}

function bindPlan() {
  $$("[data-action]", $("#viewRoot")).forEach((button) => button.addEventListener("click", handlePlanAction));
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
  if (action === "return-to-terminal") {
    window.close();
    setTimeout(() => announce("Switch to the terminal window to follow the running workflow."), 120);
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
    announce(`Execution depth increased. Plan revision ${result.revision || "is being prepared"}.`);
    await loadView("plan", true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function saveDraft() {
  try {
    setBusy(true);
    await api("/plan/draft", { method: "POST", body: JSON.stringify(planPayload()) });
    state.planDirty = false;
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
    state.editingTask = null;
    state.editingSummary = false;
    announce(`Revision ${result.revision} saved. Nothing is running yet.`);
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
    }
    const result = await api("/plan/approve", {
      method: "POST",
      body: JSON.stringify({ revision }),
    });
    state.planDirty = false;
    state.planConfirm = false;
    announce(
      result.execution_requested
        ? `Plan r${revision} approved. Execution is starting in the terminal.`
        : `Plan r${revision} approved. Return to the terminal to continue execution.`,
    );
    await refreshContext({ enforceGate: false });
    await loadView("plan", true);
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
    announce("Request added to Up next.");
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
        <p class="eyebrow">What you are reviewing</p>
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
            : "All files are resolved. Execution will continue after submission."}</p>
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
    eyebrow: "Review note",
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
        summary: "Reviewed in the local workspace.",
      }),
    });
    announce(result.fixer_started ? "Review submitted. A fixer has started." : "Review submitted. Execution can continue.");
    state.reviewConfirm = false;
    await refreshContext({ enforceGate: false });
    await loadView("review", true);
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false);
  }
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
  $("#viewRoot").innerHTML = `
    <header class="view-head">
      <div>
        <p class="eyebrow">Optional live view</p>
        <h2>Execution</h2>
        <p>Read-only operational state · Plan r${data.plan_revision || "—"}</p>
      </div>
      ${badge(data.core.status)}
    </header>
    <dl class="execution-summary">
      <div class="fact"><dt>Current phase</dt><dd>${escapeHtml(statusLabel(phase))}</dd></div>
      <div class="fact"><dt>Active work</dt><dd>${escapeHtml(activeAgent?.task || "No active task")}</dd></div>
      <div class="fact"><dt>Next work</dt><dd>${escapeHtml(nextNode?.title || "Determined by the harness")}</dd></div>
      <div class="fact"><dt>Your attention</dt><dd>${blockers.length ? "Needed for a blocker" : "Not required"}</dd></div>
    </dl>
    <section class="section">
      <div class="section-head"><div><h3>Work map</h3><p>Tasks and authoritative phase changes. Percentages appear only when reported by the runtime.</p></div></div>
      ${data.nodes.length ? `<ul class="node-list">${data.nodes.map((node) => `
        <li class="node-row">
          <div class="node-main">
            <div class="node-copy">
              <h4>${escapeHtml(node.title)}</h4>
              <p>${escapeHtml(node.assigned_role || "Harness assigned")} · ${node.dependencies.length ? `After ${escapeHtml(node.dependencies.join(", "))}` : "No dependencies"}</p>
            </div>
            ${badge(node.status)}
          </div>
        </li>`).join("")}</ul>` : `<p class="empty-state">No specialist work nodes are active. Current work remains visible in the Plan view.</p>`}
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
    ${blockers.length ? `
      <section class="section"><div class="section-head"><div><h3>Blockers</h3><p>${escapeHtml(blockers.join(" · "))}</p></div></div></section>` : ""}`;
  $$("[data-agent-explain]").forEach((button) => button.addEventListener("click", () =>
    openExplanationDrawer(button.dataset.agentExplain)));
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
  $("#drawerEyebrow").textContent = eyebrow;
  $("#drawerTitle").textContent = title;
  $("#drawerBody").innerHTML = body;
  $("#drawer").classList.remove("hidden");
  $("#drawerBackdrop").classList.remove("hidden");
  document.body.classList.add("drawer-open");
  onReady?.();
}

function closeDrawer() {
  $("#drawer").classList.add("hidden");
  $("#drawerBackdrop").classList.add("hidden");
  document.body.classList.remove("drawer-open");
}

async function loadView(view, force = false) {
  if (state.busy && !force) return;
  if (!force && view === "plan" && (
    state.planDirty || state.editingTask !== null || state.editingSummary
  )) return;
  const silent = !force;
  const activeId = document.activeElement?.id || "";
  const scrollAnchor = captureScrollAnchor();
  $("#errorState").classList.add("hidden");
  if (!silent) {
    $("#loading").classList.remove("hidden");
    $("#viewRoot").classList.add("hidden");
  }
  try {
    if (view === "plan") {
      const snapshot = await api("/plan");
      const fingerprint = stableFingerprint(snapshot);
      if (silent && state.viewFingerprints.plan === fingerprint) {
        setConnection(true);
        return;
      }
      if (silent && (state.planDirty || state.editingTask !== null || state.editingSummary || readerIsBusy())) {
        state.pendingUpdates.plan = true;
        showActivityBanner(true);
        return;
      }
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
      const fingerprint = stableFingerprint(snapshot);
      if (silent && state.viewFingerprints.review === fingerprint) return;
      if (silent && readerIsBusy()) {
        state.pendingUpdates.review = true;
        showActivityBanner(true);
        return;
      }
      state.viewFingerprints.review = fingerprint;
      state.review = snapshot;
      state.reviewDecisions = new Map();
      state.reviewComments = [];
      state.reviewConfirm = false;
      renderReview();
      restoreScrollAnchor(scrollAnchor);
    } else {
      const snapshot = await api("/agents");
      const fingerprint = stableFingerprint(snapshot);
      if (silent && state.viewFingerprints.agents === fingerprint) return;
      state.viewFingerprints.agents = fingerprint;
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
    setConnection(true);
  } catch (error) {
    if (error.status === 404 && view !== "plan") {
      $("#loading").classList.add("hidden");
      $("#viewRoot").classList.remove("hidden");
      $("#viewRoot").innerHTML = `
        <header class="view-head"><div><p class="eyebrow">${view === "review" ? "No review gate" : "No execution map"}</p>
        <h2>${view === "review" ? "Review" : "Execution"}</h2></div></header>
        <p class="empty-state">${escapeHtml(error.message)}. Nothing is required from you on this page.</p>`;
    } else {
      showError(error);
    }
  } finally {
    $("#app").setAttribute("aria-busy", "false");
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
  $("#attentionAction").addEventListener("click", () => navigate($("#attentionAction").dataset.view));
  $("#refreshButton").addEventListener("click", async () => {
    await refreshContext({ enforceGate: true });
    await loadView(state.currentView, true);
    announce("Workspace refreshed.");
  });
  $("#applyActivity").addEventListener("click", async () => {
    state.pendingUpdates = {};
    showActivityBanner(false);
    await loadView(state.currentView, true);
  });
  window.addEventListener("scroll", () => {
    state.scrollingUntil = Date.now() + 800;
  }, { passive: true });
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#drawer").classList.contains("hidden")) closeDrawer();
  });
  window.addEventListener("popstate", () => {
    const view = location.pathname.split("/").filter(Boolean)[2] || "plan";
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
    if (state.context.required_view) {
      state.currentView = state.context.required_view;
      history.replaceState({ view: state.currentView }, "", `/sessions/${sessionId}/${state.currentView}`);
    }
    renderChrome();
    await loadView(state.currentView, true);
    state.contextTimer = setInterval(async () => {
      await refreshContext({ enforceGate: true });
      const canRefreshView = state.currentView === "agents"
        || (state.currentView === "plan" && !state.planDirty
          && state.editingTask === null && !state.editingSummary);
      if (!state.busy && canRefreshView && document.visibilityState === "visible") {
        await loadView(state.currentView, false);
      }
    }, 4000);
  } catch (error) {
    showError(error);
  }
}

boot();

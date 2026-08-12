"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const pathParts = location.pathname.split("/").filter(Boolean);
const sessionId = pathParts[1] || "";
const requestedShell = pathParts[2] || "plan";
const legacyTab = ["agents", "tree", "execution"].includes(requestedShell)
  ? "agents" : ["history"].includes(requestedShell) ? "timeline" : "overview";
const initialPage = requestedShell === "plan" ? "plan" : "live";
const csrf = decodeURIComponent((document.cookie.match(/(?:^|; )ga3bad_csrf=([^;]*)/) || [])[1] || "");
$("#outputNav").href = `/sessions/${encodeURIComponent(sessionId)}/output`;

function draftStorageKey(kind, suffix = "") {
  return `ga3bad:${sessionId}:${kind}${suffix ? `:${suffix}` : ""}`;
}

function readDraft(kind, suffix = "") {
  try { return sessionStorage.getItem(draftStorageKey(kind, suffix)) || ""; }
  catch (_) { return ""; }
}

function writeDraft(kind, value, suffix = "") {
  try {
    const key = draftStorageKey(kind, suffix);
    value ? sessionStorage.setItem(key, value) : sessionStorage.removeItem(key);
  } catch (_) { /* The in-memory draft still remains authoritative. */ }
}

const state = {
  page: initialPage,
  liveTab: legacyTab,
  timelineFilter: "all",
  context: null,
  plan: null,
  agents: null,
  history: null,
  review: null,
  planDocument: "",
  documentDirty: false,
  requestDraft: readDraft("plan-request"),
  requestDirty: Boolean(readDraft("plan-request")),
  questionDraft: "",
  questionDirty: false,
  activeQuestionId: "",
  selectedNode: "core",
  loading: true,
  refreshing: false,
  refreshQueued: false,
  eventSource: null,
  pollTimer: null,
  toastTimer: null,
};

function statusLabel(value) {
  const key = String(value || "").toLowerCase();
  return ({
    awaiting_plan_approval: "Ready",
    discovering: "Planning",
    revising: "Revising",
    running: "Running",
    verifying: "Verifying",
    reviewing: "Reviewing",
    paused: "Waiting",
    blocked: "Blocked",
    recovering: "Recovering",
    completed: "Complete",
    cancelled: "Stopped",
    in_progress: "Working",
    testing: "Testing",
    fixing: "Fixing",
    integrating: "Integrating",
    queued: "Waiting",
    waiting: "Waiting",
    rate_limited: "Retrying",
    failed: "Failed",
    uncertain: "Needs attention",
    ready: "Ready",
    pending: "Waiting",
  })[key] || (key ? key.replaceAll("_", " ") : "Idle");
}

function statusTone(value) {
  const key = String(value || "").toLowerCase();
  if (["running", "discovering", "revising", "verifying", "reviewing", "in_progress"].includes(key)) return "running";
  if (["paused", "pending", "ready", "queued", "awaiting_plan_approval", "waiting"].includes(key)) return "waiting";
  if (["completed", "done", "integrated"].includes(key)) return "done";
  if (["blocked", "failed", "uncertain", "cancelled"].includes(key)) return "blocked";
  return "";
}

async function api(path, options = {}) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}${path}`, {
    cache: "no-store",
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.method && options.method !== "GET" ? { "X-GA3BAD-CSRF": csrf } : {}),
      ...(options.headers || {}),
    },
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : { error: await response.text() };
  if (!response.ok) {
    const error = new Error(payload.error || payload.detail || `Request failed (${response.status})`);
    error.code = payload.code || "request_failed";
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.classList.remove("hidden");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => node.classList.add("hidden"), 3600);
  $("#announce").textContent = message;
}

function setConnection(kind, label) {
  const node = $("#connection");
  node.className = `connection ${kind}`;
  $("span", node).textContent = label;
}

function setLoading(value) {
  state.loading = value;
  $("#loading").classList.toggle("hidden", !value);
  $("#app").setAttribute("aria-busy", String(value));
}

function showFatal(error) {
  setLoading(false);
  const node = $("#fatalError");
  node.classList.remove("hidden");
  node.innerHTML = `<h2>The saved workspace could not load</h2><p>${escapeHtml(error.message || error)}</p><button class="secondary-button" id="retryLoad" type="button">Retry</button>`;
  $("#retryLoad").addEventListener("click", () => refresh({ force: true }));
}

function runtimeStatus() {
  return state.context?.runtime?.phase || state.context?.goal?.status || state.plan?.goal_status || "idle";
}

function updateChrome() {
  const context = state.context || {};
  const runtime = context.runtime || {};
  const status = runtimeStatus();
  $$("[data-page]").forEach((button) => {
    const selected = button.dataset.page === state.page;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-current", selected ? "page" : "false");
  });
  $("#planPage").classList.toggle("hidden", state.page !== "plan");
  $("#livePage").classList.toggle("hidden", state.page !== "live");
  $("#statusKicker").textContent = state.page === "plan" ? "ULTRA PLAN" : "EXECUTION";
  $("#statusTitle").textContent = runtime.active_operation || runtime.current_task || context.attention?.title || (state.page === "plan" ? "Plan the next Execution run" : "No active work");
  $("#statusDetail").textContent = runtime.reason || context.attention?.body || "The terminal runtime is the source of truth.";
  const badge = $("#statusBadge");
  badge.textContent = statusLabel(status);
  badge.className = `status-badge ${statusTone(status)}`;
  const model = runtime.model || state.plan?.runtime?.model || "";
  $("#modelName").textContent = model || "Not selected";
  $("#modelButton").hidden = !model;
  $("#liveGoal").textContent = context.goal?.objective || "The terminal runtime is the source of truth.";
  $("#elapsedValue").textContent = formatDuration(runtime.elapsed_seconds);
  renderBlocker();
}

function formatDuration(seconds) {
  const value = Number(seconds || 0);
  if (!value) return "—";
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = Math.floor(value % 60);
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${secs}s` : `${secs}s`;
}

function formatTime(value) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function navigate(page, tab = null, { replace = false } = {}) {
  state.page = page === "plan" ? "plan" : "live";
  if (tab) state.liveTab = tab;
  const suffix = state.page === "plan" ? "plan" : "live";
  const url = `/sessions/${encodeURIComponent(sessionId)}/${suffix}`;
  history[replace ? "replaceState" : "pushState"]({ page: state.page, tab: state.liveTab }, "", url);
  updateChrome();
  state.page === "plan" ? renderPlan() : renderLive();
  $("#workspace").focus({ preventScroll: true });
}

function renderBlocker() {
  const context = state.context || {};
  const required = context.required_action || null;
  const node = $("#blocker");
  if (!required || ["answer", "approve_plan"].includes(required.kind)) {
    node.classList.add("hidden");
    return;
  }
  node.classList.remove("hidden");
  $("#blockerTitle").textContent = context.attention?.title || required.description || "The workflow needs attention";
  $("#blockerBody").textContent = required.description || context.attention?.body || "The saved checkpoint is waiting for a decision.";
  const actions = $("#blockerActions");
  actions.innerHTML = "";
  const add = (label, action, className = "secondary-button") => {
    const button = document.createElement("button");
    button.className = className;
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => runBlockerAction(action, required));
    actions.append(button);
  };
  if (required.kind === "allow_tool" || context.tool_approval) {
    add("Allow once", "allow_tool", "primary-button");
    add("Deny", "deny_tool");
    add("Stop safely", "stop", "danger-button");
  } else {
    const choices = [
      { kind: required.kind, label: required.label || "Continue" },
      ...(required.alternatives || []),
    ].filter((choice, index, all) => choice?.kind && all.findIndex((item) => item?.kind === choice.kind) === index);
    choices.forEach((choice, index) => add(
      choice.label || choice.kind,
      choice.kind,
      choice.kind === "stop" ? "danger-button" : index === 0 ? "primary-button" : "secondary-button",
    ));
  }
}

async function runBlockerAction(action, required) {
  if (action === "switch_model") return openModelPicker(true);
  if (action === "inspect") {
    const runtime = state.context?.runtime || {};
    const reason = required.description || state.context?.attention?.body || runtime.reason || "The workflow is paused at a saved checkpoint.";
    return openDrawer(
      "READ-ONLY CHECKPOINT",
      "Why work stopped",
      `<p>${escapeHtml(reason)}</p><dl class="detail-list"><div><dt>State</dt><dd>${escapeHtml(runtime.phase || "paused")}</dd></div><div><dt>Waiting on</dt><dd>${escapeHtml(runtime.waiting_on || state.context?.waiting_on || "recovery")}</dd></div><div><dt>Next action</dt><dd>${escapeHtml(required.retry_exhausted ? "Change model for one fresh attempt" : state.context?.resume_action || "Retry saved stage")}</dd></div></dl><p><small>Inspect is read-only. It does not start a worker or consume the saved checkpoint.</small></p>`,
    );
  }
  try {
    const fingerprint = required.fingerprint || state.context?.tool_approval?.action_fingerprint || "";
    const result = await api("/actions", {
      method: "POST",
      body: JSON.stringify({
        action,
        target_id: required.target_id || null,
        action_fingerprint: fingerprint,
        source: "web",
      }),
    });
    toast(result.message || "The saved workflow is continuing.");
    await refresh({ force: true });
  } catch (error) { toast(error.message, true); }
}

let renderPlan;
let submitPlanRequest;
let answerQuestion;
let prepareAgents;
let approveAndStart;
function renderHandoff(completed) {
  $("#planRoot").innerHTML = `<section class="handoff"><div class="handoff-inner"><div class="handoff-mark">${completed ? "✓" : "→"}</div><span class="eyebrow">${completed ? "EXECUTION COMPLETE" : "TERMINAL HANDOFF"}</span><h1>${completed ? "The work is complete." : "Execution is running."}</h1><p>${completed ? "Open Live to inspect the final tree, timeline and recorded changes." : "The exact plan and first-layer team were accepted. You can close this tab and return to the terminal; execution does not depend on the tab closing."}</p><div class="button-row centered-actions"><button id="openLiveAfterHandoff" class="secondary-button" type="button">Open Live</button></div></div></section>`;
  $("#openLiveAfterHandoff").addEventListener("click", () => navigate("live", "overview"));
}

function renderLive() {
  if (state.page !== "live" || !state.context) return;
  $$('[data-live-tab]').forEach((button) => {
    const selected = button.dataset.liveTab === state.liveTab;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  if (state.liveTab === "agents") renderAgents();
  else if (state.liveTab === "timeline") renderTimeline();
  else renderOverview();
}

function activityRows(limit = 5) {
  const rows = state.context?.runtime?.timeline_preview || [];
  return rows.slice(-limit).reverse();
}

function todoDetailMarkup(item) {
  const checklist = item.checklist || [];
  const dependencies = item.dependencies || [];
  const evidence = item.verified_evidence?.length ? item.verified_evidence : (item.evidence || []);
  const changed = item.changed_files || [];
  const issues = item.issues || [];
  const paths = [...new Set([...(item.read_paths || []), ...(item.write_paths || [])])];
  const checks = checklist.length
    ? `<ol class="todo-detail-checklist">${checklist.map((check) => `<li class="todo-${escapeHtml(check.state || check.status || "pending")}"><span class="todo-mark" aria-hidden="true"></span><div><strong>${escapeHtml(check.title || "Check")}</strong><small>${escapeHtml(statusLabel(check.state || check.status || "pending"))}</small>${(check.evidence || []).length ? `<ul class="evidence-list">${check.evidence.map((receipt) => `<li>${escapeHtml(receipt)}</li>`).join("")}</ul>` : ""}</div></li>`).join("")}</ol>`
    : `<p class="muted-note">No smaller checklist was recorded for this task.</p>`;
  return `<section class="todo-detail"><p>${escapeHtml(item.description || item.objective || "No task description was recorded.")}</p><div class="todo-progress"><strong>${Number(item.verification_percent || 0)}%</strong><span>evidence verified</span></div><h3>Checklist</h3>${checks}<dl class="facts"><div><dt>Status</dt><dd>${escapeHtml(statusLabel(item.state || item.status))}</dd></div><div><dt>Role</dt><dd>${escapeHtml(item.assigned_role || "Harness selected")}</dd></div><div><dt>Attempts</dt><dd>${Number(item.attempts || 0)}</dd></div><div><dt>Dependencies</dt><dd>${escapeHtml(dependencies.join(", ") || "None")}</dd></div><div><dt>Files read</dt><dd>${escapeHtml((item.read_paths || []).join(", ") || "None")}</dd></div><div><dt>Files changing</dt><dd>${escapeHtml((item.write_paths || []).join(", ") || "None")}</dd></div><div><dt>Files in scope</dt><dd>${escapeHtml(paths.join(", ") || "Not recorded")}</dd></div><div><dt>Changed files</dt><dd>${escapeHtml(changed.join(", ") || "None")}</dd></div></dl>${evidence.length ? `<h3>Verification evidence</h3><ul class="evidence-list">${evidence.map((receipt) => `<li>${escapeHtml(receipt)}</li>`).join("")}</ul>` : `<p class="muted-note">No authoritative verification receipt has been recorded yet.</p>`}${issues.length ? `<h3>Open findings</h3><ul class="issue-list">${issues.map((issue) => `<li>${escapeHtml(issue)}</li>`).join("")}</ul>` : ""}</section>`;
}

function openTodo(item) {
  openDrawer("TASK DETAILS", item.title || item.id || "Task", todoDetailMarkup(item));
}

function renderOverview() {
  const root = $("#liveRoot");
  const context = state.context || {};
  const runtime = context.runtime || {};
  const data = state.agents || { nodes: [], agents: [], result: {} };
  const nodes = data.nodes || [];
  const todo = context.todo || { items: [] };
  const plan = context.current_plan || {};
  const todoItems = todo.items || [];
  const done = Number(todo.done ?? nodes.filter((node) => ["completed", "done", "integrated"].includes(String(node.status))).length);
  const active = (data.agents || []).find((agent) => ["running", "in_progress"].includes(String(agent.status))) || null;
  const paused = ["paused", "retrying", "waiting_for_approval"].includes(String(runtime.phase || ""));
  const current = paused
    ? runtime.current_task || runtime.active_operation || context.attention?.title || "Saved checkpoint"
    : active?.last_action || active?.task || runtime.active_operation || runtime.current_task || context.attention?.title || "No active work";
  const remaining = Math.max(0, Number(todo.total ?? nodes.length) - done);
  const eta = runtime.eta_seconds ? formatDuration(runtime.eta_seconds) : "—";
  const events = activityRows();
  const files = data.result?.changed_files || [];
  const planTasks = (plan.tasks || []).slice(0, 12);
  const planContinuity = todo.pending_plan_revision && todo.plan_revision
    ? `<p class="todo-continuity"><strong>Progress kept from accepted plan r${escapeHtml(todo.plan_revision)}.</strong> Repair plan r${escapeHtml(todo.pending_plan_revision)} is waiting for approval and is not counted yet.</p>`
    : "";
  const planMarkup = plan.summary
    ? `<details class="readonly-plan" open><summary><span>Plan r${escapeHtml(plan.revision || "—")}${plan.approved_revision && plan.approved_revision !== plan.revision ? ` / accepted r${escapeHtml(plan.approved_revision)}` : ""}</span><em>${escapeHtml(statusLabel(plan.status || "preparing"))}</em></summary><p>${escapeHtml(plan.summary)}</p>${planTasks.length ? `<ol>${planTasks.map((task) => `<li><strong>${escapeHtml(task.title)}</strong><small>${escapeHtml(task.description || statusLabel(task.status))}</small></li>`).join("")}</ol>` : ""}</details>`
    : `<p>The current plan appears here as soon as it is saved.</p>`;
  const todoMarkup = todoItems.length
    ? `${planContinuity}<div class="todo-percentages"><span><strong>${Number(todo.completion_percent || 0)}%</strong> complete</span><span><strong>${Number(todo.verification_percent || 0)}%</strong> evidence verified</span></div><ol class="todo-list">${todoItems.slice(0, 30).map((item) => `<li class="todo-${escapeHtml(item.state || "pending")}"><span class="todo-mark" aria-hidden="true"></span><button data-todo-node="${escapeHtml(item.id)}" type="button"><strong>${escapeHtml(item.title || "Task")}</strong><small>${escapeHtml(item.state === "working" ? "Working now" : statusLabel(item.status || item.state))} / ${Number(item.verification_percent || 0)}% verified</small></button></li>`).join("")}</ol>${todoItems.length > 30 ? `<p class="muted-note">${todoItems.length - 30} more tasks remain visible in Agents.</p>` : ""}`
    : `<p>The to-do list appears when the plan is materialized.</p>`;
  root.innerHTML = `<section class="overview"><div class="now-row"><div class="now-main"><span class="eyebrow">NOW</span><h2>${escapeHtml(current)}</h2><p>${escapeHtml(runtime.reason || context.attention?.body || "Live state is projected from the terminal runtime.")}</p></div><div class="metric"><strong>${done}</strong><span>Completed</span></div><div class="metric"><strong>${remaining}</strong><span>Remaining</span></div><div class="metric"><strong>${eta}</strong><span>ETA</span></div></div><div class="plan-todo-grid"><section class="overview-section"><h3>Current plan</h3>${planMarkup}</section><section class="overview-section"><h3>To-do</h3><p class="todo-summary">${Number(todo.working || 0)} working · ${Number(todo.done || 0)} done · ${Number(todo.pending || 0)} waiting${Number(todo.blocked || 0) ? ` · ${Number(todo.blocked)} blocked` : ""}</p>${todoMarkup}</section></div><section class="overview-section"><h3>Recent activity</h3>${events.length ? `<ol class="activity-list">${events.map((event) => `<li><time>${escapeHtml(formatTime(event.timestamp))}</time><p>${escapeHtml(event.message || event.operation || "Activity recorded")}</p></li>`).join("")}</ol>` : `<p>No activity has been recorded yet.</p>`}</section><section class="overview-section"><h3>Recorded changes</h3><div>${files.length ? `<ul class="file-list">${files.map((file) => `<li><button class="file-button" data-file="${escapeHtml(file)}" type="button">${escapeHtml(file)}</button></li>`).join("")}</ul>` : `<p>No file changes have been recorded.</p>`}</div></section><section class="overview-section"><h3>Execution model</h3><p>${escapeHtml(`${runtime.provider || "provider"}/${runtime.model || "model"} · ${context.execution_strategy || "recursive"} · ${runtime.execution_class || "runtime"}`)}</p></section></section>`;
  $$('[data-file]').forEach((button) => button.addEventListener("click", () => openDiff(button.dataset.file)));
  $$('[data-todo-node]').forEach((button) => button.addEventListener("click", () => {
    const item = todoItems.find((candidate) => candidate.id === button.dataset.todoNode);
    if (item) openTodo(item);
  }));
}

function treeMarkup(nodes, parentId = null) {
  const children = nodes.filter((node) => (node.parent_id || null) === parentId);
  if (!children.length) return "";
  const agents = state.agents?.agents || [];
  return `<ul>${children.map((node) => {
    const assigned = agents.filter((agent) => agent.task_id === node.id);
    return `<li>${nodeButton(node)}${agentBranchMarkup(assigned)}${treeMarkup(nodes, node.id)}</li>`;
  }).join("")}</ul>`;
}

function nodeButton(node) {
  const status = node.status || "pending";
  return `<button class="agent-node task-node status-${escapeHtml(status)}${state.selectedNode === node.id ? " selected" : ""}" data-node="${escapeHtml(node.id)}" type="button"><span class="dot" aria-hidden="true"></span><span><strong>${escapeHtml(node.title || "Task")}</strong><small>${escapeHtml(node.objective || statusLabel(status))}</small></span><em>Task / ${escapeHtml(statusLabel(status))}</em></button>`;
}

function agentSelectionId(agent) {
  return `agent:${agent.id}`;
}

function agentButton(agent) {
  const selectionId = agentSelectionId(agent);
  const status = agent.status || "queued";
  const attempt = Number(agent.attempt || 1);
  const phase = agent.phase ? statusLabel(agent.phase) : "Specialist";
  const workDetail = agent.last_action || `${phase}${attempt > 1 ? ` / attempt ${attempt}` : ""}`;
  const detail = `${agent.short_id ? `ID ${agent.short_id} · ` : ""}${workDetail}`;
  return `<button class="agent-node agent-run status-${escapeHtml(status)}${state.selectedNode === selectionId ? " selected" : ""}" data-node="${escapeHtml(selectionId)}" type="button"><span class="dot" aria-hidden="true"></span><span><strong>${escapeHtml(agent.display_name || agent.name || agent.role || "Agent")}</strong><small>${escapeHtml(detail)}</small></span><em>${escapeHtml(statusLabel(status))}</em></button>`;
}

function agentBranchMarkup(agents) {
  if (!agents.length) return "";
  return `<ul class="agent-run-list">${agents.map((agent) => `<li>${agentButton(agent)}</li>`).join("")}</ul>`;
}

function agentResultMarkup(agent) {
  const result = agent.latest_output || {};
  const summary = result.summary || (agent.status === "queued" ? "Waiting for its dependency or execution slot." : "No result has been recorded yet.");
  const changed = result.changed_files || [];
  const artifacts = result.artifacts || [];
  const issues = result.issues || [];
  const tests = result.tests || [];
  return `<section class="agent-result"><h3>Latest result</h3><p>${escapeHtml(summary)}</p>${changed.length ? `<h4>Changed files</h4><ul>${changed.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}${tests.length ? `<h4>Checks</h4><ul>${tests.slice(-5).map((item) => `<li>${escapeHtml(item.summary || item.name || item.command || JSON.stringify(item))}</li>`).join("")}</ul>` : ""}${artifacts.length ? `<h4>Artifacts</h4><ul>${artifacts.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}${issues.length ? `<h4>Issues</h4><ul>${issues.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}</section>`;
}

function agentProblemMarkup(agent) {
  if (!agent.problem) return "";
  const stateLabel = agent.recovered
    ? "Recovered automatically"
    : agent.retrying
    ? "A replacement attempt is running"
    : agent.superseded
    ? "Superseded by a later attempt"
    : agent.attention_required
    ? "Action is available"
    : "Recorded attempt issue";
  const actions = agent.attention_required
    ? `<div class="agent-action-row"><button data-agent-remedy="retry" type="button">Retry saved branch</button><button data-agent-remedy="model" type="button">Change model</button><button data-agent-remedy="advanced" type="button">Technical details</button></div>`
    : `<div class="agent-action-row"><button data-agent-remedy="advanced" type="button">Technical details</button></div>`;
  return `<section class="agent-problem resolution-${escapeHtml(agent.resolution_state || "recorded")}"><span class="eyebrow">${escapeHtml(stateLabel)}</span><h3>${escapeHtml(agent.problem.summary || "This attempt needs attention")}</h3><p>${escapeHtml(agent.problem.detail || "The saved checkpoint remains available.")}</p>${actions}</section>`;
}

function selectedAgentMarkup() {
  const data = state.agents || { nodes: [], agents: [], core: {} };
  if (state.selectedNode === "core") {
    return `<span class="eyebrow">ROOT COORDINATOR</span><h2>${escapeHtml(data.core?.name || "GA3BAD Core")}</h2><p>Owns the active plan, integration and final evidence.</p><dl class="facts"><div><dt>Status</dt><dd>${escapeHtml(statusLabel(data.core?.status))}</dd></div><div><dt>Plan</dt><dd>r${escapeHtml(data.plan_revision || "-")}</dd></div><div><dt>Agent runs</dt><dd>${(data.agents || []).length}</dd></div><div><dt>Tasks</dt><dd>${(data.nodes || []).length}</dd></div></dl>`;
  }
  if (state.selectedNode.startsWith("agent:")) {
    const agentId = state.selectedNode.slice(6);
    const agent = (data.agents || []).find((item) => item.id === agentId) || {};
    const node = (data.nodes || []).find((item) => item.id === agent.task_id) || {};
    return `<span class="eyebrow">${escapeHtml(agent.role || "SPECIALIST")}</span><h2>${escapeHtml(agent.display_name || agent.name || "Agent")}</h2><p class="agent-id">ID ${escapeHtml(agent.short_id || agent.id || "—")}</p><p>${escapeHtml(agent.goal || node.objective || "No mission recorded.")}</p>${agentProblemMarkup(agent)}<dl class="facts"><div><dt>Status</dt><dd>${escapeHtml(statusLabel(agent.status))}</dd></div><div><dt>Current work</dt><dd>${escapeHtml(agent.last_action || agent.phase || "Waiting")}</dd></div><div><dt>Model</dt><dd>${escapeHtml([agent.provider, agent.model].filter(Boolean).join("/") || "Not recorded")}</dd></div><div><dt>Elapsed</dt><dd>${escapeHtml(formatDuration(agent.elapsed_seconds))}</dd></div><div><dt>Attempt</dt><dd>${escapeHtml(agent.attempt || 1)}</dd></div><div><dt>Files read</dt><dd>${escapeHtml((agent.files_inspected || []).join(", ") || "None")}</dd></div><div><dt>Files changing</dt><dd>${escapeHtml((agent.files_modifying || []).join(", ") || "None")}</dd></div><div><dt>Blockers</dt><dd>${escapeHtml((agent.blockers || []).join(" / ") || "None")}</dd></div></dl>${agentResultMarkup(agent)}`;
  }
  const node = (data.nodes || []).find((item) => item.id === state.selectedNode) || {};
  const assigned = (data.agents || []).filter((item) => item.task_id === state.selectedNode);
  return `<span class="eyebrow">TASK</span><h2>${escapeHtml(node.title || "Task")}</h2>${todoDetailMarkup(node)}<dl class="facts"><div><dt>Agent runs</dt><dd>${assigned.length}</dd></div><div><dt>Blockers</dt><dd>${node.blocked ? "Blocked" : "None"}</dd></div></dl>`;
}

function renderAgents() {
  const root = $("#liveRoot");
  const data = state.agents || { nodes: [], agents: [], core: {} };
  const nodes = data.nodes || [];
  const agents = data.agents || [];
  const validSelections = new Set(["core", ...nodes.map((node) => node.id), ...agents.map(agentSelectionId)]);
  if (!validSelections.has(state.selectedNode)) state.selectedNode = "core";
  const nodeIds = new Set(nodes.map((node) => node.id));
  const unassigned = agents.filter((agent) => !agent.task_id || !nodeIds.has(agent.task_id));
  root.innerHTML = `<section class="agent-workspace"><div class="tree-canvas"><ul class="tree-list"><li><button class="agent-node status-${escapeHtml(data.core?.status || "pending")}${state.selectedNode === "core" ? " selected" : ""}" data-node="core" type="button"><span class="dot"></span><span><strong>${escapeHtml(data.core?.name || "GA3BAD Core")}</strong><small>Active plan and integration owner / ${agents.length} agent run${agents.length === 1 ? "" : "s"}</small></span><em>${escapeHtml(statusLabel(data.core?.status))}</em></button>${treeMarkup(nodes)}${agentBranchMarkup(unassigned)}</li></ul>${!nodes.length && !agents.length ? `<p class="empty-state">The first-layer tree appears after the plan starts.</p>` : ""}</div><aside class="agent-inspector">${selectedAgentMarkup()}</aside></section>`;
  $$('[data-node]').forEach((button) => button.addEventListener("click", () => {
    state.selectedNode = button.dataset.node;
    renderAgents();
  }));
  $$('[data-agent-remedy]').forEach((button) => button.addEventListener("click", () => {
    const remedy = button.dataset.agentRemedy;
    if (remedy === "advanced") {
      location.href = `/sessions/${encodeURIComponent(sessionId)}/advanced-tracing`;
      return;
    }
    if (remedy === "model") return openModelPicker(true);
    if (remedy === "retry") return runBlockerAction("retry", state.context?.required_action || {});
  }));
}

function isProblem(event) {
  const value = `${event.event_type || ""} ${event.phase || ""} ${event.summary || ""}`.toLowerCase();
  return /fail|block|error|denied|uncertain|retry|approval|required/.test(value);
}

function renderTimeline() {
  const root = $("#liveRoot");
  let items = state.history?.items || [];
  if (state.timelineFilter === "problems") items = items.filter(isProblem);
  root.innerHTML = `<div class="timeline-toolbar"><p>${items.length} durable events</p><div class="timeline-filter"><button data-filter="all" class="${state.timelineFilter === "all" ? "selected" : ""}" type="button">All</button><button data-filter="problems" class="${state.timelineFilter === "problems" ? "selected" : ""}" type="button">Problems</button></div></div><ol class="timeline-list">${items.slice().reverse().map((event) => `<li class="timeline-row${isProblem(event) ? " problem" : ""}"><details><summary><time>${escapeHtml(formatTime(event.created_at))}</time><span class="actor">${escapeHtml(event.actor || "harness")}</span><strong>${escapeHtml(event.summary || event.event_type)}</strong><span class="event-type">${escapeHtml(event.phase || event.event_type || "event")}</span></summary><div class="timeline-detail"><p>${escapeHtml(event.why || "Recorded by the durable runtime.")}</p>${event.evidence?.length ? `<pre>${escapeHtml(JSON.stringify(event.evidence, null, 2))}</pre>` : ""}</div></details></li>`).join("") || `<li class="empty-state">No matching events.</li>`}</ol>`;
  $$('[data-filter]').forEach((button) => button.addEventListener("click", () => { state.timelineFilter = button.dataset.filter; renderTimeline(); }));
}

function openDrawer(eyebrow, title, body) {
  $("#drawerEyebrow").textContent = eyebrow;
  $("#drawerTitle").textContent = title;
  $("#drawerBody").innerHTML = body;
  $("#drawerBackdrop").classList.remove("hidden");
  $("#drawer").classList.remove("hidden");
  $("#app").inert = true;
  $("#drawerClose").focus();
}

function closeDrawer() {
  $("#drawerBackdrop").classList.add("hidden");
  $("#drawer").classList.add("hidden");
  $("#app").inert = false;
}

function openDiff(filePath) {
  const file = (state.review?.files || []).find((item) => item.path === filePath);
  if (!file) return openDrawer("RECORDED CHANGE", filePath, `<p>No line-level diff was recorded for this file. The terminal and durable evidence remain authoritative.</p>`);
  const lines = String(file.diff || "").split("\n");
  openDrawer("READ-ONLY DIFF", filePath, `<p>This is an audit view. Review and repair decisions stay with the runtime.</p><div class="diff">${lines.map((line) => `<div class="diff-line ${line.startsWith("+") && !line.startsWith("+++") ? "added" : line.startsWith("-") && !line.startsWith("---") ? "deleted" : ""}">${escapeHtml(line || " ")}</div>`).join("")}</div>`);
}

async function openModelPicker(fromBlocker = false) {
  try {
    const catalog = await api("/models");
    const models = catalog.models || [];
    openDrawer("SAFE CHECKPOINT", "Change model", `<p>Choose a discovered model. The saved plan and completed evidence remain unchanged.</p><div class="model-list">${models.map((model) => `<button class="model-option" data-model="${escapeHtml(model.id)}" type="button"><span><strong>${escapeHtml(model.display_name || model.model)}</strong><br><small>${escapeHtml(model.provider)} · ${escapeHtml(model.execution_class)}</small></span><span>${model.selected ? "Current" : "Choose"}</span></button>`).join("") || `<p>No compatible models are available.</p>`}</div>`);
    $$('[data-model]', $("#drawerBody")).forEach((button) => button.addEventListener("click", async () => {
      try {
        const result = await api("/actions", { method: "POST", body: JSON.stringify({ action: "switch_model", target_id: button.dataset.model, action_fingerprint: `model:${button.dataset.model}`, source: "web" }) });
        closeDrawer();
        toast(result.message || "Model changed.");
        if (fromBlocker) {
          try { await api("/actions", { method: "POST", body: JSON.stringify({ action: "retry", source: "web" }) }); } catch (_) { /* saved checkpoint may resume during switch */ }
        }
        await refresh({ force: true });
      } catch (error) { toast(error.message, true); }
    }));
  } catch (error) { toast(error.message, true); }
}

async function refresh({ force = false } = {}) {
  if (state.refreshing) { state.refreshQueued = true; return; }
  state.refreshing = true;
  try {
    const [context, plan, agents, history] = await Promise.all([
      api("/workspace"),
      api("/plan").catch((error) => error.status === 404 ? null : Promise.reject(error)),
      api("/agents").catch((error) => error.status === 404 ? null : Promise.reject(error)),
      api("/history?limit=200").catch((error) => error.status === 404 ? null : Promise.reject(error)),
    ]);
    state.context = context;
    state.plan = plan;
    state.agents = agents;
    state.history = history;
    if (!state.documentDirty && plan?.document) state.planDocument = plan.document;
    if (agents?.result?.changed_files?.length) {
      try { state.review = await api("/review"); } catch (error) { if (error.status !== 404 && error.status !== 422) throw error; state.review = null; }
    } else {
      state.review = null;
    }
    $("#fatalError").classList.add("hidden");
    setLoading(false);
    updateChrome();
    if (state.page === "plan") {
      // Live refreshes never detach a prompt, answer, or plan while the user
      // is editing it. The first render is still allowed so a restored draft
      // has somewhere to appear.
      const hasActiveDraft = state.documentDirty || state.requestDirty || state.questionDirty;
      if (!hasActiveDraft || !$("#planRoot").hasChildNodes()) renderPlan();
    } else {
      renderLive();
    }
    setConnection("connected", "Live");
  } catch (error) {
    setConnection("offline", "Reconnecting");
    if (!state.context || force) showFatal(error);
  } finally {
    state.refreshing = false;
    if (state.refreshQueued) { state.refreshQueued = false; setTimeout(() => refresh(), 0); }
  }
}

function startEvents() {
  state.eventSource?.close();
  const source = new EventSource(`/api/sessions/${encodeURIComponent(sessionId)}/events`);
  state.eventSource = source;
  source.addEventListener("open", () => setConnection("connected", "Live"));
  const schedule = () => {
    clearTimeout(state.sseTimer);
    state.sseTimer = setTimeout(() => refresh(), 120);
  };
  source.addEventListener("snapshot", schedule);
  source.addEventListener("activity", schedule);
  source.onerror = () => setConnection("offline", "Polling");
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(() => refresh(), 5000);
}

function bindShell() {
  $$("[data-page]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.page)));
  $$('[data-live-tab]').forEach((button) => button.addEventListener("click", () => { state.liveTab = button.dataset.liveTab; renderLive(); }));
  $("#brand").addEventListener("click", () => navigate("plan"));
  $("#modelButton").addEventListener("click", () => {
    if (state.context?.required_action && !["answer", "approve_plan"].includes(state.context.required_action.kind)) openModelPicker(true);
  });
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !$("#drawer").classList.contains("hidden")) closeDrawer(); });
  window.addEventListener("popstate", (event) => {
    state.page = event.state?.page || (location.pathname.endsWith("/plan") ? "plan" : "live");
    state.liveTab = event.state?.tab || state.liveTab;
    updateChrome();
    state.page === "plan" ? renderPlan() : renderLive();
  });
}

async function init() {
  bindShell();
  history.replaceState({ page: state.page, tab: state.liveTab }, "", location.pathname);
  updateChrome();
  await refresh({ force: true });
  startEvents();
}

(() => {
  "use strict";

  const pathParts = location.pathname.split("/").filter(Boolean);
  const sessionId = pathParts[1] || "";
  const view = pathParts[2] || "plan";
  const state = {
    data: null,
    mode: "simple",
    selectedFile: 0,
    selectedAgent: null,
    decisions: new Map(),
    comments: [],
    dirty: false,
    timer: null,
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const root = $("#viewRoot");
  const csrf = () => document.cookie.split("; ").find(v => v.startsWith("ga3bad_csrf="))?.split("=")[1] || "";
  const escapeHtml = value => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const lines = value => String(value ?? "").split("\n").map(v => v.trim()).filter(Boolean);
  const comma = value => String(value ?? "").split(",").map(v => v.trim()).filter(Boolean);

  function toast(message, kind = "") {
    const item = document.createElement("div");
    item.className = `toast ${kind}`;
    item.textContent = message;
    $("#toastRegion").append(item);
    setTimeout(() => item.remove(), 4200);
  }

  async function api(url, options = {}) {
    const headers = { "Accept": "application/json", ...(options.headers || {}) };
    if (options.body) headers["Content-Type"] = "application/json";
    if (!["GET", "HEAD"].includes(options.method || "GET")) headers["X-GA3BAD-CSRF"] = csrf();
    const response = await fetch(url, { ...options, headers, credentials: "same-origin" });
    let body = {};
    try { body = await response.json(); } catch { body = {}; }
    if (!response.ok) {
      const error = new Error(body.error || body.detail || `Request failed (${response.status})`);
      error.status = response.status;
      error.body = body;
      throw error;
    }
    return body;
  }

  function setChrome(name, artifact, session) {
    $("#viewName").textContent = name;
    $("#artifactMeta").textContent = artifact || "Live state";
    $("#sessionMeta").textContent = `Session ${String(session || sessionId).slice(0, 8)}`;
    document.title = `${name} · GA3BAD`;
  }

  function markConnected(connected = true) {
    const node = $("#connection");
    node.classList.toggle("offline", !connected);
    node.lastChild.textContent = connected ? " Connected" : " Offline";
  }

  function showError(error) {
    $("#loading").classList.add("hidden");
    root.classList.add("hidden");
    const node = $("#errorState");
    node.classList.remove("hidden");
    node.innerHTML = `<div><strong>Unable to open this workspace</strong><p>${escapeHtml(error.message || error)}</p></div>`;
    markConnected(false);
  }

  function showRoot() {
    $("#loading").classList.add("hidden");
    $("#errorState").classList.add("hidden");
    root.classList.remove("hidden");
    $("#app").setAttribute("aria-busy", "false");
    markConnected(true);
  }

  function planTaskTemplate(task, index) {
    const retry = task.retry_policy || { max_retries: 2, backoff_seconds: 0 };
    return `
      <article class="task-row ${state.mode === "advanced" ? "expanded" : ""}" data-task-index="${index}" draggable="true">
        <div class="task-index">${String(index + 1).padStart(2, "0")}</div>
        <div class="task-main">
          <div class="task-title-line">
            <input class="title-input" aria-label="Task ${index + 1} title" data-field="title" value="${escapeHtml(task.title)}">
            <select aria-label="Task ${index + 1} status" data-field="status">
              ${["pending", "ready", "in_progress", "blocked", "completed"].map(value =>
                `<option value="${value}" ${task.status === value ? "selected" : ""}>${value.replaceAll("_", " ")}</option>`).join("")}
            </select>
          </div>
          <textarea class="description-input" aria-label="Task ${index + 1} description" data-field="description">${escapeHtml(task.description)}</textarea>
        </div>
        <div class="task-controls" aria-label="Task actions">
          <button class="icon-button move-up" type="button" title="Move task up" aria-label="Move task up">↑</button>
          <button class="icon-button move-down" type="button" title="Move task down" aria-label="Move task down">↓</button>
          <button class="icon-button split-task" type="button" title="Split task" aria-label="Split task">⑂</button>
          <button class="icon-button expand-task" type="button" title="Advanced fields" aria-label="Toggle advanced fields">•••</button>
          <button class="icon-button danger delete-task" type="button" title="Delete task" aria-label="Delete task">×</button>
        </div>
        <div class="advanced-fields">
          <div class="field-group">
            <label class="field-label">Task ID</label>
            <input data-field="id" value="${escapeHtml(task.id)}" aria-label="Task ID">
          </div>
          <div class="field-group">
            <label class="field-label">Agent role</label>
            <input data-field="agent_role" value="${escapeHtml(task.agent_role)}" aria-label="Agent role">
          </div>
          <div class="field-group">
            <label class="field-label">Dependencies</label>
            <input data-field="dependencies" value="${escapeHtml((task.dependencies || []).join(", "))}" aria-label="Dependencies">
            <p class="field-help">Comma-separated task IDs</p>
          </div>
          <div class="field-group">
            <label class="field-label">Risk level</label>
            <select data-field="risk_level" aria-label="Risk level">
              ${["low", "medium", "high", "critical"].map(value =>
                `<option ${task.risk_level === value ? "selected" : ""}>${value}</option>`).join("")}
            </select>
          </div>
          <div class="field-group">
            <label class="field-label">Inputs</label>
            <textarea data-field="inputs" aria-label="Inputs">${escapeHtml((task.inputs || []).join("\n"))}</textarea>
          </div>
          <div class="field-group">
            <label class="field-label">Outputs</label>
            <textarea data-field="outputs" aria-label="Outputs">${escapeHtml((task.outputs || []).join("\n"))}</textarea>
          </div>
          <div class="field-group wide">
            <label class="field-label">Expected files</label>
            <textarea data-field="expected_files" aria-label="Expected files">${escapeHtml((task.expected_files || []).join("\n"))}</textarea>
          </div>
          <div class="field-group">
            <label class="field-label">Acceptance criteria</label>
            <textarea data-field="acceptance_criteria" aria-label="Acceptance criteria">${escapeHtml((task.acceptance_criteria || []).join("\n"))}</textarea>
          </div>
          <div class="field-group">
            <label class="field-label">Tests</label>
            <textarea data-field="tests" aria-label="Tests">${escapeHtml((task.tests || []).join("\n"))}</textarea>
          </div>
          <div class="field-group">
            <label class="field-label">Required tools</label>
            <input data-field="required_tools" value="${escapeHtml((task.required_tools || []).join(", "))}" aria-label="Required tools">
          </div>
          <div class="field-group">
            <label class="field-label">Memory dependencies</label>
            <input data-field="memory_dependencies" value="${escapeHtml((task.memory_dependencies || []).join(", "))}" aria-label="Memory dependencies">
          </div>
          <div class="field-group">
            <label class="field-label">Max retries</label>
            <input type="number" min="0" max="20" data-field="max_retries" value="${retry.max_retries}" aria-label="Maximum retries">
          </div>
          <div class="field-group">
            <label class="field-label">Backoff seconds</label>
            <input type="number" min="0" max="3600" step="0.5" data-field="backoff_seconds" value="${retry.backoff_seconds}" aria-label="Retry backoff seconds">
          </div>
          <div class="field-group wide">
            <label class="field-label">Constraints</label>
            <textarea data-field="constraints" aria-label="Task constraints">${escapeHtml((task.constraints || []).join("\n"))}</textarea>
          </div>
          <div class="field-group wide">
            <label class="field-label">Comments</label>
            <textarea data-field="comments" aria-label="Task comments">${escapeHtml((task.comments || []).join("\n"))}</textarea>
          </div>
          <div class="field-group wide check-grid">
            ${[
              ["parallel", "May run in parallel"],
              ["approval_gate", "Approval required before start"],
              ["paused", "Pause this task"],
              ["disabled", "Disable this task"],
            ].map(([field, label]) => `<label class="check-label"><input type="checkbox" data-field="${field}" ${task[field] ? "checked" : ""}> ${label}</label>`).join("")}
          </div>
        </div>
      </article>`;
  }

  function renderPlan() {
    const data = state.data;
    setChrome("Plan Studio", `Plan r${data.revision}`, data.session_short);
    root.innerHTML = `
      <header class="view-heading">
        <div><span class="eyebrow">Editable artifact · revision protected</span><h1>Plan Studio</h1></div>
        <div class="heading-tools">
          <div class="mode-switch" role="group" aria-label="Plan detail mode">
            <button type="button" class="mode-button ${state.mode === "simple" ? "active" : ""}" data-mode="simple">Simple</button>
            <button type="button" class="mode-button ${state.mode === "advanced" ? "active" : ""}" data-mode="advanced">Advanced</button>
          </div>
          <button id="addTask" class="button" type="button">+ Add task</button>
        </div>
      </header>
      <div id="conflictSlot"></div>
      <div class="plan-layout">
        <section class="primary-pane" aria-label="Plan tasks">
          <div class="plan-summary">
            <label class="field-label" for="planSummary">Plan summary</label>
            <textarea id="planSummary">${escapeHtml(data.summary)}</textarea>
          </div>
          <div class="pane-header"><strong>${data.tasks.length} tasks</strong><span>Drag or use arrows to reorder</span></div>
          <div id="taskList" class="task-list">${data.tasks.map(planTaskTemplate).join("")}</div>
          <div class="action-bar">
            <button id="discardPlan" class="button danger" type="button">Discard changes</button>
            <button id="saveDraft" class="button" type="button">Save draft</button>
            <button id="applyPlan" class="button primary" type="button">Apply to GA3BAD</button>
          </div>
        </section>
        <aside class="secondary-pane" aria-label="Plan constraints">
          <div class="pane-header"><strong>Plan boundaries</strong><span>Global</span></div>
          <div class="secondary-content">
            <section class="side-section">
              <label class="field-label" for="globalConstraints">Global constraints</label>
              <textarea id="globalConstraints" rows="8">${escapeHtml((data.global_constraints || []).join("\n"))}</textarea>
              <p>One constraint per line. Agents receive these with the active revision.</p>
            </section>
            <section class="side-section">
              <label class="field-label" for="protectedPaths">Protected files and directories</label>
              <textarea id="protectedPaths" rows="7">${escapeHtml((data.protected_paths || []).join("\n"))}</textarea>
              <p>Workspace-relative paths only. The web layer never modifies files directly.</p>
            </section>
            <section class="side-section">
              <label class="field-label" for="changeNote">Revision note</label>
              <textarea id="changeNote" rows="4">Edited in Plan Studio</textarea>
            </section>
            <section class="side-section">
              <h2>Snapshot</h2>
              <div class="tag-list">
                <span class="tag">OPENED r${data.revision}</span>
                <span class="tag">${escapeHtml(data.status)}</span>
                <span class="tag">${new Date(data.updated_at).toLocaleTimeString()}</span>
              </div>
            </section>
          </div>
        </aside>
      </div>`;
    bindPlan();
    state.dirty = false;
    showRoot();
  }

  function readTask(row) {
    const value = field => $(`[data-field="${field}"]`, row);
    return {
      id: value("id").value.trim(),
      title: value("title").value.trim(),
      description: value("description").value.trim(),
      status: value("status").value,
      parent_id: null,
      dependencies: comma(value("dependencies").value),
      agent_role: value("agent_role").value.trim() || "coder",
      inputs: lines(value("inputs").value),
      outputs: lines(value("outputs").value),
      expected_files: lines(value("expected_files").value),
      acceptance_criteria: lines(value("acceptance_criteria").value),
      tests: lines(value("tests").value),
      risk_level: value("risk_level").value,
      required_tools: comma(value("required_tools").value),
      memory_dependencies: comma(value("memory_dependencies").value),
      retry_policy: {
        max_retries: Number(value("max_retries").value || 0),
        backoff_seconds: Number(value("backoff_seconds").value || 0),
      },
      approval_gate: value("approval_gate").checked,
      constraints: lines(value("constraints").value),
      parallel: value("parallel").checked,
      paused: value("paused").checked,
      disabled: value("disabled").checked,
      comments: lines(value("comments").value),
    };
  }

  function planPayload() {
    return {
      base_revision: state.data.revision,
      summary: $("#planSummary").value.trim(),
      tasks: $$(".task-row").map(readTask),
      global_constraints: lines($("#globalConstraints").value),
      protected_paths: lines($("#protectedPaths").value),
      change_note: $("#changeNote").value.trim(),
    };
  }

  function newTask() {
    const ids = new Set($$(".task-row").map(row => $("[data-field=id]", row).value.toUpperCase()));
    let n = 1;
    while (ids.has(`T${String(n).padStart(3, "0")}`)) n++;
    const id = `T${String(n).padStart(3, "0")}`;
    return {
      id, title: "Untitled task", description: "", status: "pending", dependencies: [],
      agent_role: "coder", inputs: [], outputs: [], expected_files: [],
      acceptance_criteria: ["Define observable completion criteria"],
      tests: ["Define the direct verification method"], risk_level: "medium",
      required_tools: [], memory_dependencies: [], retry_policy: { max_retries: 2, backoff_seconds: 0 },
      approval_gate: false, constraints: [], parallel: false, paused: false, disabled: false, comments: [],
    };
  }

  function rerenderPlanTasks(tasks) {
    $("#taskList").innerHTML = tasks.map(planTaskTemplate).join("");
    bindTaskRows();
    state.dirty = true;
  }

  function bindTaskRows() {
    $$(".task-row").forEach((row, index) => {
      $(".move-up", row).onclick = () => {
        const tasks = $$(".task-row").map(readTask);
        if (index > 0) [tasks[index - 1], tasks[index]] = [tasks[index], tasks[index - 1]];
        rerenderPlanTasks(tasks);
      };
      $(".move-down", row).onclick = () => {
        const tasks = $$(".task-row").map(readTask);
        if (index < tasks.length - 1) [tasks[index + 1], tasks[index]] = [tasks[index], tasks[index + 1]];
        rerenderPlanTasks(tasks);
      };
      $(".delete-task", row).onclick = () => {
        if ($$(".task-row").length === 1) return toast("A plan must keep at least one task.", "error");
        row.remove();
        rerenderPlanTasks($$(".task-row").map(readTask));
      };
      $(".expand-task", row).onclick = () => row.classList.toggle("expanded");
      $(".split-task", row).onclick = () => {
        const tasks = $$(".task-row").map(readTask);
        const original = tasks[index];
        const child = newTask();
        child.title = `${original.title} — follow-up`;
        child.description = "Complete the second bounded part of the original task.";
        child.dependencies = [original.id];
        tasks.splice(index + 1, 0, child);
        rerenderPlanTasks(tasks);
      };
      row.addEventListener("input", () => state.dirty = true);
      row.addEventListener("dragstart", event => event.dataTransfer.setData("text/plain", String(index)));
      row.addEventListener("dragover", event => event.preventDefault());
      row.addEventListener("drop", event => {
        event.preventDefault();
        const from = Number(event.dataTransfer.getData("text/plain"));
        const tasks = $$(".task-row").map(readTask);
        const [moved] = tasks.splice(from, 1);
        tasks.splice(index, 0, moved);
        rerenderPlanTasks(tasks);
      });
    });
  }

  function showConflict(error) {
    const current = error.body?.current_revision ?? "?";
    $("#conflictSlot").innerHTML = `
      <div class="conflict-banner" role="alert">
        <div><strong>This page was opened with Plan r${state.data.revision}. The current plan is Plan r${current}.</strong>
          <p>Your edits were not applied. Keep them as a draft or reload the latest revision.</p></div>
        <div class="conflict-actions">
          <button id="keepDraftConflict" class="button small" type="button">Keep as draft</button>
          <button id="reloadConflict" class="button small primary" type="button">Reload latest</button>
        </div>
      </div>`;
    $("#keepDraftConflict").onclick = () => submitPlan("draft");
    $("#reloadConflict").onclick = () => load(true);
  }

  async function submitPlan(kind) {
    const payload = planPayload();
    try {
      if (kind === "draft") {
        await api(`/api/sessions/${sessionId}/plan/draft`, { method: "POST", body: JSON.stringify(payload) });
        toast(`Draft saved against Plan r${payload.base_revision}.`);
        state.dirty = false;
      } else {
        const result = await api(`/api/sessions/${sessionId}/plan/apply`, { method: "POST", body: JSON.stringify(payload) });
        toast(`Plan r${result.previous_revision} → r${result.revision} applied.`);
        await load(true);
      }
    } catch (error) {
      if (error.status === 409) showConflict(error);
      else toast(error.message, "error");
    }
  }

  function bindPlan() {
    bindTaskRows();
    $$("[data-mode]").forEach(button => button.onclick = () => {
      state.mode = button.dataset.mode;
      $$(".mode-button").forEach(item => item.classList.toggle("active", item === button));
      $$(".task-row").forEach(row => row.classList.toggle("expanded", state.mode === "advanced"));
    });
    $("#addTask").onclick = () => rerenderPlanTasks([...$$(".task-row").map(readTask), newTask()]);
    $("#saveDraft").onclick = () => submitPlan("draft");
    $("#applyPlan").onclick = () => submitPlan("apply");
    $("#discardPlan").onclick = async () => {
      await api(`/api/sessions/${sessionId}/plan/draft`, { method: "DELETE" });
      await load(true);
      toast("Local edits discarded.");
    };
    ["#planSummary", "#globalConstraints", "#protectedPaths", "#changeNote"].forEach(selector =>
      $(selector).addEventListener("input", () => state.dirty = true));
  }

  function decisionKey(type, path, hunk = "") { return `${type}|${path}|${hunk}`; }

  function setDecision(type, path, hunk, decision) {
    let reason = "";
    if (decision !== "accepted") {
      reason = window.prompt(decision === "rejected" ? "Why should this change be rejected?" : "What needs to change?") || "";
      if (!reason.trim()) return;
    }
    state.decisions.set(decisionKey(type, path, hunk), {
      target_type: type, file_path: path, hunk_id: hunk || null, decision, reason: reason.trim(),
    });
    renderReviewWorkspace();
  }

  function addLineComment(file, hunk, line) {
    const body = window.prompt(`Comment on ${file}:${line}`);
    if (!body?.trim()) return;
    state.comments.push({ file_path: file, hunk_id: hunk || null, line, body: body.trim() });
    renderReviewWorkspace();
    toast("Inline comment added.");
  }

  function reviewInspector(file) {
    const comments = state.comments.filter(item => item.file_path === file.path);
    return `
      <div class="inspector-header"><strong>Change context</strong><span>${escapeHtml(file.status)}</span></div>
      <div class="inspector-content">
        <section class="side-section"><h2>Why this changed</h2><p>${escapeHtml(file.reason)}</p></section>
        <section class="side-section">
          <dl class="definition-list">
            <dt>Task</dt><dd>${escapeHtml(file.task)}</dd>
            <dt>Agent</dt><dd>${escapeHtml(file.agent)}</dd>
            <dt>Checkpoint</dt><dd>${escapeHtml(state.data.checkpoint_id)}</dd>
            <dt>Plan</dt><dd>r${escapeHtml(state.data.plan_revision)}</dd>
          </dl>
        </section>
        <section class="side-section"><h2>Test results</h2>
          <pre>${escapeHtml(JSON.stringify(file.tests || {}, null, 2))}</pre>
        </section>
        <section class="side-section"><h2>Evidence</h2>
          ${(file.evidence || []).length ? `<div class="tag-list">${file.evidence.map(item => `<span class="tag">${escapeHtml(item.summary)}</span>`).join("")}</div>` : "<p>No linked evidence.</p>"}
        </section>
        <section class="side-section"><h2>Review comments · ${comments.length}</h2>
          <div class="comment-list">${comments.map(item => `<div class="comment"><strong>LINE ${item.line || "FILE"}</strong><p>${escapeHtml(item.body)}</p></div>`).join("") || "<p>No comments yet. Use + beside a diff line.</p>"}</div>
        </section>
      </div>`;
  }

  function renderReviewWorkspace() {
    const file = state.data.files[state.selectedFile];
    const fileDecision = state.decisions.get(decisionKey("file", file.path));
    $("#diffWorkspace").innerHTML = `
      <div class="diff-toolbar">
        <div><div class="diff-path">${escapeHtml(file.path)}</div>
          ${fileDecision ? `<span class="decision-badge ${fileDecision.decision}">${fileDecision.decision.replaceAll("_", " ")}</span>` : ""}
        </div>
        <div class="decision-controls" aria-label="File decision">
          <button class="button small accept-file" type="button">Accept file</button>
          <button class="button small request-file" type="button">Request changes</button>
          <button class="button small danger reject-file" type="button">Reject file</button>
        </div>
      </div>
      <div class="diff-scroll">
        ${file.hunks.length ? file.hunks.map(hunk => {
          const decision = state.decisions.get(decisionKey("hunk", file.path, hunk.id));
          return `<section class="hunk">
            <div class="hunk-header"><span>${escapeHtml(hunk.header)}</span>
              <span class="hunk-actions">
                ${decision ? `<span class="decision-badge ${decision.decision}">${decision.decision.replaceAll("_", " ")}</span>` : ""}
                <button class="button small accept-hunk" data-hunk="${escapeHtml(hunk.id)}" type="button">Accept</button>
                <button class="button small request-hunk" data-hunk="${escapeHtml(hunk.id)}" type="button">Change</button>
                <button class="button small danger reject-hunk" data-hunk="${escapeHtml(hunk.id)}" type="button">Reject</button>
              </span></div>
            ${hunk.lines.map(line => `<div class="diff-line ${line.kind}">
              <span class="line-no">${line.number}</span>
              <button class="comment-trigger" type="button" data-comment-hunk="${escapeHtml(hunk.id)}" data-line="${line.number}" aria-label="Comment on line ${line.number}">+</button>
              <code>${escapeHtml(line.text)}</code>
            </div>`).join("")}
          </section>`;
        }).join("") : `<div class="empty-state"><div><strong>Diff content unavailable</strong><p>The checkpoint records this file but has no unified diff body.</p></div></div>`}
      </div>`;
    $("#reviewInspector").innerHTML = reviewInspector(file);
    $(".accept-file").onclick = () => setDecision("file", file.path, null, "accepted");
    $(".request-file").onclick = () => setDecision("file", file.path, null, "changes_requested");
    $(".reject-file").onclick = () => setDecision("file", file.path, null, "rejected");
    $$(".accept-hunk").forEach(button => button.onclick = () => setDecision("hunk", file.path, button.dataset.hunk, "accepted"));
    $$(".request-hunk").forEach(button => button.onclick = () => setDecision("hunk", file.path, button.dataset.hunk, "changes_requested"));
    $$(".reject-hunk").forEach(button => button.onclick = () => setDecision("hunk", file.path, button.dataset.hunk, "rejected"));
    $$("[data-comment-hunk]").forEach(button => button.onclick = () =>
      addLineComment(file.path, button.dataset.commentHunk, Number(button.dataset.line)));
  }

  function renderReview() {
    const data = state.data;
    setChrome("Change Review", `Checkpoint ${data.checkpoint_id.slice(0, 12)}`, data.session_short);
    root.innerHTML = `
      <header class="view-heading">
        <div><span class="eyebrow">Immutable checkpoint · review workspace</span><h1>Change Review</h1></div>
        <p>This diff remains fixed while you review it. Decisions affect the checkpoint; this page never writes repository files.</p>
      </header>
      <div class="review-layout">
        <nav class="file-list" aria-label="Changed files">
          <div class="list-header"><strong>${data.files.length} changed files</strong><span>r${data.plan_revision}</span></div>
          ${data.files.map((file, index) => `<button type="button" class="file-item ${index === state.selectedFile ? "active" : ""}" data-file-index="${index}">
            <span class="file-path">${escapeHtml(file.path)}</span>
            <span class="file-stats"><b class="plus">+${file.additions}</b> <b class="minus">−${file.deletions}</b></span>
          </button>`).join("")}
        </nav>
        <section id="diffWorkspace" class="primary-pane" aria-label="Code diff"></section>
        <aside id="reviewInspector" class="inspector-pane" aria-label="Change details"></aside>
      </div>
      <div class="action-bar">
        <span>${state.decisions.size} decisions · ${state.comments.length} comments</span>
        <button id="submitReview" class="button primary" type="button">Submit review</button>
      </div>`;
    $$("[data-file-index]").forEach(button => button.onclick = () => {
      state.selectedFile = Number(button.dataset.fileIndex);
      $$("[data-file-index]").forEach(item => item.classList.toggle("active", item === button));
      renderReviewWorkspace();
    });
    $("#submitReview").onclick = submitReview;
    renderReviewWorkspace();
    showRoot();
  }

  async function submitReview() {
    if (!state.decisions.size) return toast("Record at least one file or hunk decision.", "warning");
    const payload = {
      checkpoint_id: state.data.checkpoint_id,
      decisions: [...state.decisions.values()],
      comments: state.comments,
      summary: "Review submitted from Change Review.",
    };
    try {
      const result = await api(`/api/sessions/${sessionId}/review/submit`, { method: "POST", body: JSON.stringify(payload) });
      toast(`${result.counts.approved} files approved · ${result.counts.changes_requested} require changes.`);
      if (result.fixer_started) toast("Fixer started with your feedback.");
      await load(true);
    } catch (error) { toast(error.message, "error"); }
  }

  function agentInspector(agent) {
    if (!agent) return `<div class="empty-state"><div><strong>Select an agent</strong><p>Inspect its actual task, context, files, logs, memory, and retry history.</p></div></div>`;
    return `
      <div class="inspector-header"><strong>${escapeHtml(agent.role)}</strong><span class="status-badge ${escapeHtml(agent.status)}">${escapeHtml(agent.status)}</span></div>
      <div class="inspector-content">
        <section class="side-section"><span class="eyebrow">Current goal</span><h2>${escapeHtml(agent.goal)}</h2><p>${escapeHtml(agent.task)}</p></section>
        <section class="side-section"><dl class="definition-list">
          <dt>Agent ID</dt><dd>${escapeHtml(agent.id)}</dd>
          <dt>Plan</dt><dd>r${escapeHtml(agent.plan_revision)}</dd>
          <dt>Current file</dt><dd>${escapeHtml(agent.current_file || "—")}</dd>
          <dt>Elapsed</dt><dd>${Math.floor(agent.elapsed_seconds / 60)}m ${agent.elapsed_seconds % 60}s</dd>
          <dt>Retries</dt><dd>${agent.retries}</dd>
          <dt>Parent</dt><dd>${escapeHtml(agent.parent_node || "GA3BAD Core")}</dd>
          <dt>Children</dt><dd>${agent.child_agents}</dd>
        </dl></section>
        <section class="side-section"><h2>Files inspected</h2><div class="tag-list">${agent.files_inspected.map(item => `<span class="tag">${escapeHtml(item)}</span>`).join("") || "<span class='tag'>NONE</span>"}</div></section>
        <section class="side-section"><h2>Files being modified</h2><div class="tag-list">${agent.files_modifying.map(item => `<span class="tag">${escapeHtml(item)}</span>`).join("") || "<span class='tag'>NONE</span>"}</div></section>
        <section class="side-section"><h2>Memory references</h2><div class="tag-list">${agent.memory.map(item => `<span class="tag">${escapeHtml(item.title)}</span>`).join("") || "<span class='tag'>NONE</span>"}</div></section>
        <section class="side-section"><h2>Recent logs</h2><div class="log-list">${agent.logs.map(item => `<div class="log-row"><i></i><span>${escapeHtml(item.summary)}</span></div>`).join("") || "<p>No recent logs.</p>"}</div></section>
        <section class="side-section"><h2>Latest output</h2><pre>${escapeHtml(JSON.stringify(agent.latest_output || {}, null, 2))}</pre></section>
        <button id="requestExplanation" class="button" type="button">Request explanation</button>
      </div>`;
  }

  function renderAgents() {
    const data = state.data;
    setChrome("Agent Tree", data.plan_revision ? `Plan r${data.plan_revision}` : "No active plan", data.session_short);
    const byNode = new Map(data.nodes.map(node => [node.id, []]));
    data.agents.forEach(agent => {
      if (!byNode.has(agent.task_id)) byNode.set(agent.task_id, []);
      byNode.get(agent.task_id).push(agent);
    });
    const selected = data.agents.find(agent => agent.id === state.selectedAgent) || null;
    root.innerHTML = `
      <header class="view-heading">
        <div><span class="eyebrow">Live state · refreshes every 2 seconds</span><h1>Execution Map</h1></div>
        <p>Read-only operational view of GA3BAD Core, work nodes, dependencies, assignments, retries, and blockers.</p>
      </header>
      <div class="agent-layout">
        <section class="primary-pane">
          <div class="pane-header"><strong>${data.agents.length} agents · ${data.nodes.length} nodes</strong><span>Updated ${new Date(data.updated_at).toLocaleTimeString()}</span></div>
          <div class="execution-canvas">
            <div class="core-node"><span class="eyebrow">Root coordinator</span><strong>${escapeHtml(data.core.name)}</strong><span>${escapeHtml(data.core.status)}</span></div>
            ${data.nodes.length ? `<div class="node-grid">${data.nodes.map(node => {
              const agents = byNode.get(node.id) || [];
              return `<article class="execution-node ${node.blocked ? "blocked" : ""} ${agents.some(item => item.status === "running") ? "active" : ""}">
                <div class="node-header"><div><strong>${escapeHtml(node.title)}</strong><small>${escapeHtml(node.id)}</small></div><span class="status-badge ${escapeHtml(node.status)}">${escapeHtml(node.status)}</span></div>
                ${agents.map(agent => `<button type="button" class="agent-item ${agent.id === state.selectedAgent ? "active" : ""}" data-agent-id="${escapeHtml(agent.id)}">
                  <span class="agent-avatar">${escapeHtml(agent.role.slice(0, 2).toUpperCase())}</span>
                  <span class="agent-copy"><strong>${escapeHtml(agent.role)}</strong><small>${escapeHtml(agent.last_action)}</small><span class="progress-track"><span class="progress-fill" style="width:${Math.max(0, Math.min(100, agent.progress))}%"></span></span></span>
                  <span class="status-badge ${escapeHtml(agent.status)}">${agent.blocked ? "blocked" : escapeHtml(agent.status)}</span>
                </button>`).join("") || `<div class="empty-state"><p>Waiting for an agent assignment.</p></div>`}
              </article>`;
            }).join("")}</div>` : `<div class="empty-state"><div><strong>No active agent graph</strong><p>GA3BAD Core is available, but this session has not started an Ultra execution graph.</p></div></div>`}
          </div>
        </section>
        <aside id="agentInspector" class="inspector-pane" aria-label="Agent details">${agentInspector(selected)}</aside>
      </div>`;
    $$("[data-agent-id]").forEach(button => button.onclick = () => {
      state.selectedAgent = button.dataset.agentId;
      renderAgents();
    });
    if (selected && $("#requestExplanation")) $("#requestExplanation").onclick = async () => {
      const question = window.prompt("What should this agent explain?", "Explain your current work and any blockers.");
      if (!question?.trim()) return;
      try {
        await api(`/api/sessions/${sessionId}/agents/explain`, {
          method: "POST", body: JSON.stringify({ agent_id: selected.id, question: question.trim() }),
        });
        toast("Explanation request sent to the agent event queue.");
      } catch (error) { toast(error.message, "error"); }
    };
    showRoot();
  }

  async function load(force = false) {
    if (view === "plan" && state.dirty && !force) return toast("Save, apply, or discard your plan edits before refreshing.", "warning");
    try {
      state.data = await api(`/api/sessions/${sessionId}/${view}`);
      if (view === "plan") renderPlan();
      else if (view === "review") renderReview();
      else renderAgents();
    } catch (error) { showError(error); }
  }

  $("#refreshButton").onclick = () => load(false);
  window.addEventListener("beforeunload", event => {
    if (view === "plan" && state.dirty) {
      event.preventDefault();
      event.returnValue = "";
    }
  });

  load(true);
  if (view === "agents") state.timer = setInterval(() => load(true), 2000);
})();

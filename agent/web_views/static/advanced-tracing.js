(() => {
  "use strict";

  const pathParts = location.pathname.split("/").filter(Boolean);
  const sessionId = pathParts[1] || "";
  const apiRoot = `/api/sessions/${encodeURIComponent(sessionId)}/advanced-tracing`;
  const state = {
    section: (location.hash || "#overview").slice(1),
    overview: null,
    goalId: "",
    runId: "",
    query: "",
    category: "",
    refreshTimer: null,
    selected: null,
    eventSource: null,
    latestSequence: 0,
    copyPayload: null,
  };

  const $ = (id) => document.getElementById(id);
  const content = $("traceContent");
  const inspector = $("traceInspector");
  const inspectorEmpty = $("inspectorEmpty");
  const inspectorContent = $("inspectorContent");
  const loading = $("loadingState");
  const errorState = $("errorState");

  const sectionCopy = {
    overview: ["RUN EVIDENCE", "Overview", "What happened, what needs inspection, and where the evidence lives."],
    timeline: ["ORDERED JOURNAL", "Timeline", "Every durable stage and decision in sequence."],
    files: ["FILE LIFECYCLE", "Files", "Requested, considered, opened, changed, verified, and excluded paths."],
    problems: ["FAILURE LINEAGE", "Problems", "Findings, failed attempts, remediations, and proof."],
    agents: ["RECURSIVE HARNESS", "Agents & models", "All scheduled, waiting, running, and completed workers."],
    prompts: ["SAVED INPUTS", "Plans & prompts", "Plan revisions and redacted prompt receipts by agent."],
    context: ["CONTEXT PROVENANCE", "Context", "Retrieval choices, memory access, omission, and rotation."],
    changes: ["MUTATION EVIDENCE", "Changes", "Diffs, responsible agents, reviews, and integration state."],
  };

  function element(tag, options = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(options).forEach(([key, value]) => {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = String(value ?? "");
      else if (key.startsWith("data-")) node.setAttribute(key, String(value));
      else if (key === "onClick") node.addEventListener("click", value);
      else if (key === "ariaLabel") node.setAttribute("aria-label", value);
      else if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    });
    const values = Array.isArray(children) ? children : [children];
    values.forEach((child) => {
      if (child === null || child === undefined) return;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return node;
  }

  function cookie(name) {
    return document.cookie.split(";").map((item) => item.trim()).find((item) => item.startsWith(`${name}=`))?.split("=").slice(1).join("=") || "";
  }

  function selectionParams(extra = {}) {
    const params = new URLSearchParams();
    if (state.goalId) params.set("goal_id", state.goalId);
    if (state.runId) params.set("run_id", state.runId);
    Object.entries(extra).forEach(([key, value]) => {
      if (value !== "" && value !== null && value !== undefined) params.set(key, String(value));
    });
    const encoded = params.toString();
    return encoded ? `?${encoded}` : "";
  }

  async function request(path, options = {}) {
    const response = await fetch(`${apiRoot}${path}`, {
      credentials: "same-origin",
      cache: "no-store",
      ...options,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || data.detail || `Request failed (${response.status})`);
    return data;
  }

  function setBusy(active) {
    loading.hidden = !active;
    content.hidden = active;
    if (active) errorState.hidden = true;
  }

  function showError(error) {
    loading.hidden = true;
    content.hidden = true;
    errorState.hidden = false;
    errorState.textContent = `Advanced Tracing could not project this run.\n${error.message || error}`;
  }

  let toastTimer = null;
  function toast(message) {
    const target = $("toast");
    target.textContent = message;
    target.dataset.show = "true";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { target.dataset.show = "false"; }, 2600);
  }

  function tone(value) {
    const text = String(value || "").toLowerCase();
    if (["completed", "complete", "passed", "verified", "selected", "resolved", "ready"].some((item) => text.includes(item))) return "success";
    if (["failed", "error", "uncertain", "blocked", "denied"].some((item) => text.includes(item))) return "error";
    if (["pending", "waiting", "retry", "warning", "excluded", "paused"].some((item) => text.includes(item))) return "warning";
    return "info";
  }

  function chip(label) {
    return element("span", { class: "status-chip", "data-tone": tone(label), text: label || "recorded" });
  }

  function shortTime(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value).slice(0, 16) : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function traceRow({ type, data, title, detail, time, kind, chips = [], id = "" }) {
    const button = element("button", { class: "trace-row", type: "button" }, [
      element("span", { class: "row-time", text: time || kind || "record" }),
      element("span", { class: "row-main" }, [
        element("span", { class: "row-title", text: title || "Recorded item" }),
        element("span", { class: "row-detail", text: detail || "" }),
      ]),
      element("span", { class: "row-meta" }, chips.map(chip)),
    ]);
    button.dataset.entityId = id || data?.id || data?.path || "";
    button.addEventListener("click", () => {
      document.querySelectorAll(".trace-row[aria-selected=true]").forEach((row) => row.removeAttribute("aria-selected"));
      button.setAttribute("aria-selected", "true");
      showInspector(type, data, title);
    });
    return button;
  }

  function filtered(items, textFor) {
    const needle = state.query.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((item) => String(textFor(item)).toLowerCase().includes(needle));
  }

  function listOrEmpty(rows, message) {
    return rows.length ? element("div", { class: "trace-list" }, rows) : element("p", { class: "empty-list", text: message });
  }

  function metric(value, label) {
    return element("div", { class: "metric" }, [element("b", { text: value ?? 0 }), element("span", { text: label })]);
  }

  function sectionBlock(title, body, description = "") {
    return element("section", { class: "section-block" }, [
      element("h2", { text: title }),
      description ? element("p", { text: description }) : null,
      body,
    ]);
  }

  function updateHeading(section) {
    const copy = sectionCopy[section] || sectionCopy.overview;
    $("sectionEyebrow").textContent = copy[0];
    $("sectionTitle").textContent = copy[1];
    $("sectionDescription").textContent = copy[2];
    document.querySelectorAll("#sectionNav button").forEach((button) => {
      if (button.dataset.section === section) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
  }

  function updateOverviewChrome(overview) {
    state.overview = overview;
    state.latestSequence = Number(overview.cutoff_sequence || 0);
    const badge = $("stateBadge");
    badge.textContent = overview.state || "EMPTY";
    badge.dataset.state = String(overview.state || "empty").toLowerCase();
    const modelStatus = overview.model_status || {};
    const lifecycle = `probe ${modelStatus.probe_status || "unknown"} · response ${modelStatus.response_status || "not run"} · contract ${modelStatus.contract_status || "not run"}`;
    $("runLabel").textContent = overview.goal_id
      ? `${overview.provider || "provider"}/${overview.model || "model"} · ${overview.status || "recorded"} · ${lifecycle} · ${String(overview.goal_id).slice(-8)}`
      : overview.state === "LIVE"
        ? `${overview.provider || "provider"}/${overview.model || "model"} · ${lifecycle}`
        : "No durable activity is available yet";
    const counts = overview.counts || {};
    $("countOverview").textContent = "1";
    $("countTimeline").textContent = counts.events ?? 0;
    $("countFiles").textContent = counts.files ?? 0;
    $("countProblems").textContent = counts.problems ?? 0;
    $("countAgents").textContent = counts.agents ?? 0;
    const picker = $("runPicker");
    const selectedKey = `${overview.goal_id || ""}|${overview.run_id || ""}`;
    picker.replaceChildren();
    (overview.runs || []).forEach((run) => {
      const option = element("option", {
        value: `${run.goal_id || ""}|${run.run_id || ""}`,
        text: `${String(run.objective || "Untitled run").slice(0, 52)} · ${run.model || run.status}`,
      });
      if (option.value === selectedKey) option.selected = true;
      picker.append(option);
    });
  }

  async function renderOverview() {
    const overview = state.overview;
    const counts = overview.counts || {};
    const recent = await request(`/timeline${selectionParams({ limit: 14 })}`);
    const attention = (overview.inspect_next || []).map((item) => element("div", { class: "attention-item" }, [
      element("strong", { class: "mono", text: item.path }),
      element("span", { text: item.reason }),
    ]));
    const eventRows = (recent.items || []).map((item) => traceRow({
      type: "event", data: item, title: item.summary, detail: `${item.actor}${item.stage ? ` · ${item.stage}` : ""}`,
      time: shortTime(item.timestamp), chips: [item.category, item.status], id: item.id,
    }));
    content.replaceChildren(
      element("div", { class: "metric-strip" }, [
        metric(counts.events, "events"), metric(counts.files, "files"), metric(counts.problems, "problems"),
        metric(counts.agents, "agent calls"), metric(counts.scheduled, "scheduled"),
      ]),
      sectionBlock("Objective", element("p", { class: "row-detail", text: overview.objective || "No active objective." })),
      sectionBlock("Inspect next", attention.length ? element("div", { class: "attention-list" }, attention) : element("p", { class: "empty-list", text: "No deterministic inspection warning is open." })),
      sectionBlock("Recent evidence", listOrEmpty(eventRows, "No durable events have been recorded yet.")),
    );
    return { overview, recent };
  }

  async function renderTimeline() {
    const tools = $("sectionTools");
    const category = element("select", { ariaLabel: "Timeline category" });
    [["", "All categories"], ["workflow", "Workflow"], ["plans", "Plans"], ["agents", "Agents"], ["files", "Files"], ["problems", "Problems"], ["context", "Context"]]
      .forEach(([value, label]) => category.append(element("option", { value, text: label })));
    category.value = state.category;
    category.addEventListener("change", () => { state.category = category.value; renderSection("timeline"); });
    tools.replaceChildren(category);
    const data = await request(`/timeline${selectionParams({ limit: 1000, category: state.category, query: state.query })}`);
    const rows = (data.items || []).map((item) => traceRow({
      type: "event", data: item, title: item.summary, detail: `${item.kind} · ${item.actor}${item.stage ? ` · ${item.stage}` : ""}`,
      time: `#${item.sequence}\n${shortTime(item.timestamp)}`, chips: [item.category, item.status], id: item.id,
    }));
    content.replaceChildren(listOrEmpty(rows, "No events match the current filters."));
    return data;
  }

  async function renderFiles() {
    const data = await request(`/sections/files${selectionParams()}`);
    $("countFiles").textContent = (data.items || []).length;
    const items = filtered(data.items || [], (item) => `${item.path} ${(item.states || []).join(" ")}`);
    const rows = items.map((item) => traceRow({
      type: "file", data: item, title: item.path, detail: `${(item.records || []).length} recorded decisions${item.diff ? " · diff available" : ""}`,
      kind: "file", chips: item.states || [], id: item.path,
    }));
    const attention = (data.inspect_next || []).map((item) => element("div", { class: "attention-item" }, [
      element("strong", { class: "mono", text: item.path }), element("span", { text: item.reason }),
    ]));
    const blocks = [];
    if (attention.length) {
      blocks.push(sectionBlock("Inspect next", element("div", { class: "attention-list" }, attention)));
    }
    blocks.push(sectionBlock("File lifecycle", listOrEmpty(rows, "No file lifecycle evidence matches this filter.")));
    content.replaceChildren(...blocks);
    return data;
  }

  async function renderProblems() {
    const data = await request(`/sections/problems${selectionParams()}`);
    $("countProblems").textContent = (data.items || []).length;
    const items = filtered(data.items || [], (item) => `${item.problem} ${item.solution} ${item.path} ${item.category}`);
    const rows = items.map((item) => traceRow({
      type: "problem", data: item, title: item.problem || item.category, detail: `${item.path || item.category || "harness"}${item.solution ? ` · ${item.solution}` : ""}`,
      time: shortTime(item.timestamp), chips: [item.severity, item.status], id: item.id,
    }));
    content.replaceChildren(listOrEmpty(rows, "No recorded problem matches this filter."));
    return data;
  }

  async function renderAgents() {
    const data = await request(`/sections/agents${selectionParams()}`);
    $("countAgents").textContent = (data.agents || []).length;
    const models = (data.models || []).map((item) => traceRow({
      type: "model", data: item, title: item.provider ? `${item.provider}/${item.model}` : item.model, detail: `${item.completed} completed · ${item.failed} failed · ${item.tokens} recorded tokens`,
      kind: "model", chips: [`${item.calls} calls`, `${item.attempts} attempts`], id: item.provider ? `${item.provider}/${item.model}` : item.model,
    }));
    const nodes = filtered(data.nodes || [], (item) => `${item.title} ${item.objective} ${item.assigned_role} ${item.status}`).map((item) => traceRow({
      type: "node", data: item, title: item.title, detail: `${item.assigned_role} · ${item.objective || "No objective receipt"}`,
      kind: `depth ${item.depth}`, chips: [item.status, `${item.attempts || 0} attempts`], id: item.id,
    }));
    const agents = filtered(data.agents || [], (item) => `${item.name} ${item.role} ${item.model} ${item.phase} ${item.status}`).map((item) => traceRow({
      type: "agent", data: item, title: `${item.name || item.role || "agent"} · ${item.model || "model pending"}`, detail: `${item.role || item.phase || "specialist"} · ${item.phase || "scheduled"} · node ${String(item.work_node_id || "—").slice(-10)}`,
      time: shortTime(item.started_at || item.last_event_at || item.scheduled_at), chips: [item.status, `attempt ${item.attempt || 0}`], id: item.id,
    }));
    const scheduled = filtered(data.scheduled || [], (item) => `${item.name} ${item.role} ${item.phase} ${item.status} ${item.packet_preview}`).map((item) => traceRow({
      type: "scheduled", data: item, title: `${item.name || item.role || "agent"} · ${item.phase || "scheduled"}`, detail: item.packet_preview || "Waiting for its durable turn.",
      kind: `queue ${item.sequence || "—"}`, chips: [item.status], id: item.id,
    }));
    const workerImpact = filtered(data.worker_contributions || [], (item) => `${item.role} ${item.outcome} ${item.reason} ${item.task_class}`).map((item) => traceRow({
      type: "worker-impact", data: item, title: `${item.role} · ${item.outcome}`, detail: `${item.reason || "No verified contribution reason."} · Δ ${Number(item.score_delta || 0).toFixed(3)}`,
      time: shortTime(item.created_at), chips: [`${item.total_tokens || 0} tokens`, `${item.latency_ms || 0} ms`, `novelty ${Number(item.evidence_novelty || 0).toFixed(2)}`], id: item.id,
    }));
    const experiments = filtered(data.orchestration_experiments || [], (item) => `${item.arm} ${item.task_class} ${item.metrics?.risk_tier || ""}`).map((item) => traceRow({
      type: "orchestration-experiment", data: item, title: `${item.arm} · ${item.task_class}`, detail: `${item.causal ? "matched causal benchmark" : "observational only"} · score ${Number(item.candidate_score || 0).toFixed(3)} · ${item.model_calls || 0} calls`,
      time: shortTime(item.created_at), chips: [item.success ? "success" : "not passed", item.false_completion ? "false completion" : "evidence gated"], id: item.id,
    }));
    content.replaceChildren(
      sectionBlock("Model evidence", listOrEmpty(models, "No model calls are recorded for this run.")),
      sectionBlock("Worker impact", listOrEmpty(workerImpact, "No verified worker contributions are recorded.")),
      sectionBlock("Orchestration experiments", listOrEmpty(experiments, "No orchestration experiments are recorded.")),
      sectionBlock("Recursive nodes", listOrEmpty(nodes, "No recursive work nodes are recorded.")),
      sectionBlock("Agent calls", listOrEmpty(agents, "No agent calls are recorded.")),
      sectionBlock("Scheduled and waiting", listOrEmpty(scheduled, "No scheduled agents are waiting.")),
    );
    return data;
  }

  async function renderPrompts() {
    const data = await request(`/sections/prompts${selectionParams()}`);
    $("countPrompts").textContent = (data.traces || []).length;
    const plans = filtered(data.plans || [], (item) => `${item.summary} ${item.status} ${item.revision}`).map((item) => traceRow({
      type: "plan", data: item, title: `Plan r${item.revision} · ${item.summary}`, detail: `${(item.tasks || []).length} tasks · ${String(item.fingerprint || "").slice(0, 16)}`,
      time: shortTime(item.updated_at), chips: [item.status], id: item.id,
    }));
    const traces = filtered(data.traces || [], (item) => `${item.role} ${item.system_preview} ${item.context_preview} ${item.self_preview}`).map((item) => traceRow({
      type: "prompt", data: item, title: `${item.role} prompt`, detail: item.self_preview || item.system_preview,
      time: shortTime(item.created_at), chips: [item.redacted ? "redacted" : "stored", item.truncated ? "truncated" : "complete"], id: item.id,
    }));
    content.replaceChildren(
      sectionBlock("Plan revisions", listOrEmpty(plans, "No durable plan revision is recorded.")),
      sectionBlock("Prompt receipts", listOrEmpty(traces, "No prompt traces are recorded for this run."), "Select a prompt, then explicitly reveal the already-redacted stored text."),
    );
    return data;
  }

  async function renderContext() {
    const data = await request(`/sections/context${selectionParams()}`);
    const total = (data.retrievals || []).length + (data.rotations || []).length + (data.memory_access || []).length;
    $("countContext").textContent = total;
    const retrievals = filtered(data.retrievals || [], (item) => `${item.query} ${item.stage} ${JSON.stringify(item.candidates)}`).map((item) => traceRow({
      type: "retrieval", data: item, title: item.query || "Repository retrieval", detail: `${item.selected_count} selected · ${item.excluded_count} excluded`,
      time: shortTime(item.timestamp), chips: [item.stage || "retrieval"], id: item.id,
    }));
    const rotations = (data.rotations || []).map((item) => traceRow({
      type: "rotation", data: item, title: `${item.actor} context rotated`, detail: `${item.before_chars} → ${item.after_chars} chars · ${item.suspended_messages} messages suspended`,
      time: shortTime(item.timestamp), chips: [item.model || "model", "compacted"], id: item.checkpoint_fingerprint,
    }));
    const memory = (data.memory_access || []).map((item) => traceRow({
      type: "memory", data: item, title: `${item.direction || "memory"} · ${item.brain_entry_id || item.id || "entry"}`, detail: item.query || JSON.stringify(item.metadata || {}),
      time: shortTime(item.created_at), chips: [item.direction || "memory"], id: item.id,
    }));
    content.replaceChildren(
      sectionBlock("Repository decisions", listOrEmpty(retrievals, "No repository retrieval manifest has been recorded yet.")),
      sectionBlock("Context rotations", listOrEmpty(rotations, "No context rotation was required.")),
      sectionBlock("Memory access", listOrEmpty(memory, "No Project Brain memory access is recorded.")),
    );
    return data;
  }

  async function renderChanges() {
    const data = await request(`/sections/changes${selectionParams()}`);
    $("countChanges").textContent = (data.items || []).length;
    const rows = filtered(data.items || [], (item) => `${item.agent_id} ${item.status} ${(item.changed_files || []).join(" ")} ${item.parent_id}`).map((item, index) => traceRow({
      type: "change", data: item, title: `Change ${data.items.length - index} · ${(item.changed_files || []).length} file${(item.changed_files || []).length === 1 ? "" : "s"}`, detail: `${item.agent_id || "coordinator"} · ${item.parent_id || "workflow"} · ${(item.changed_files || []).join(", ")}`,
      time: shortTime(item.created_at), chips: [item.status, item.integration_status || "not integrated", `${(item.verification_evidence_ids || []).length} proofs`], id: item.id,
    }));
    content.replaceChildren(sectionBlock(
      "Change history",
      listOrEmpty(rows, "No recorded change set matches this filter."),
      "Every mutation remains separate and time ordered, including later repairs in the same session.",
    ));
    return data;
  }

  async function copyValue(value, message = "Copied to clipboard") {
    try {
      await navigator.clipboard.writeText(typeof value === "string" ? value : JSON.stringify(value, null, 2));
      toast(message);
    } catch (_) { toast("Clipboard access is unavailable"); }
  }

  function installCopyTool(payload) {
    state.copyPayload = payload;
    $("sectionTools").append(element("button", {
      type: "button",
      text: state.section === "timeline" ? "Copy timeline" : "Copy section",
      onClick: () => copyValue(state.copyPayload, `${sectionCopy[state.section]?.[1] || "Section"} copied`),
    }));
  }

  async function renderSection(section, { quiet = false } = {}) {
    if (!sectionCopy[section]) section = "overview";
    state.section = section;
    updateHeading(section);
    if (!quiet) setBusy(true);
    $("sectionTools").replaceChildren();
    try {
      let copyPayload = null;
      if (section === "overview") copyPayload = await renderOverview();
      else if (section === "timeline") copyPayload = await renderTimeline();
      else if (section === "files") copyPayload = await renderFiles();
      else if (section === "problems") copyPayload = await renderProblems();
      else if (section === "agents") copyPayload = await renderAgents();
      else if (section === "prompts") copyPayload = await renderPrompts();
      else if (section === "context") copyPayload = await renderContext();
      else if (section === "changes") copyPayload = await renderChanges();
      installCopyTool(copyPayload);
      loading.hidden = true;
      errorState.hidden = true;
      content.hidden = false;
      if (quiet && content.firstElementChild) content.firstElementChild.classList.add("new-record");
    } catch (error) {
      showError(error);
    }
  }

  function primitiveEntries(data) {
    return Object.entries(data || {}).filter(([key, value]) => key !== "diff" && key !== "payload" && key !== "context_package" && (value === null || ["string", "number", "boolean"].includes(typeof value)));
  }

  function showInspector(type, data, title) {
    state.selected = { type, data };
    inspectorEmpty.hidden = true;
    inspectorContent.hidden = false;
    inspector.dataset.open = "true";
    const head = element("div", { class: "inspector-head" }, [
      element("p", { class: "eyebrow", text: String(type || "record").toUpperCase() }),
      element("h2", { text: title || data?.path || data?.id || "Trace record" }),
      element("div", { class: "inspector-actions" }, [
        element("button", { type: "button", text: "Copy ID / path", onClick: async () => {
          const value = data?.path || data?.id || data?.checkpoint_fingerprint || "";
          await copyValue(String(value));
        }}),
        element("button", { type: "button", text: "Copy record", onClick: () => copyValue(data, "Record copied") }),
        data?.diff ? element("button", { type: "button", text: "Copy diff", onClick: () => copyValue(data.diff, "Diff copied") }) : null,
        element("button", { type: "button", text: "Close", onClick: () => {
          inspector.dataset.open = "false"; inspectorContent.hidden = true; inspectorEmpty.hidden = false;
        }}),
      ]),
    ]);
    const dl = element("dl");
    primitiveEntries(data).forEach(([key, value]) => dl.append(element("div", { class: "detail-pair" }, [
      element("dt", { text: key.replaceAll("_", " ") }), element("dd", { text: value ?? "—" }),
    ])));
    const jsonCopy = { ...data };
    delete jsonCopy.diff;
    const children = [head, dl];
    if (data?.diff) children.push(element("pre", { class: "diff-view", text: data.diff }));
    children.push(element("pre", { class: "detail-json", text: JSON.stringify(jsonCopy, null, 2) }));
    if (type === "prompt") {
      children.push(element("button", { class: "reveal-button", type: "button", text: "Reveal stored redacted prompt", onClick: () => revealPrompt(data.id) }));
    }
    inspectorContent.replaceChildren(...children);
  }

  async function revealPrompt(traceId) {
    if (!window.confirm("Reveal the complete locally stored prompt trace? Secrets and chain-of-thought remain unavailable.")) return;
    try {
      const data = await request("/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-GA3BAD-CSRF": decodeURIComponent(cookie("ga3bad_csrf")) },
        body: JSON.stringify({ trace_id: traceId, goal_id: state.goalId || null, run_id: state.runId || null }),
      });
      showInspector("revealed prompt", data, `${data.id} · stored redacted text`);
      toast("Stored prompt revealed locally");
    } catch (error) { toast(error.message || "Prompt reveal failed"); }
  }

  async function exportTrace() {
    const includeStored = window.confirm("Export previews only by default. Press OK to include complete stored redacted prompt/context text; Cancel keeps previews only.");
    try {
      const response = await fetch(`${apiRoot}/export${selectionParams({ include_stored_text: includeStored })}`, { credentials: "same-origin", cache: "no-store" });
      if (!response.ok) throw new Error(`Export failed (${response.status})`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = element("a", { href: url, download: `ga3bad-trace-${sessionId.slice(0, 8)}.json` });
      document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(url);
      toast(includeStored ? "Exported with stored redacted text" : "Exported redacted previews");
    } catch (error) { toast(error.message || "Export failed"); }
  }

  async function loadOverview({ preserveSelection = false } = {}) {
    const data = await request(`/overview${selectionParams()}`);
    if (!preserveSelection) {
      state.goalId = data.goal_id || "";
      state.runId = data.run_id || "";
    }
    updateOverviewChrome(data);
    return data;
  }

  function scheduleLiveRefresh() {
    if (state.overview?.state !== "LIVE") return;
    clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(async () => {
      try {
        await loadOverview({ preserveSelection: true });
        await renderSection(state.section, { quiet: true });
      } catch (error) { /* the next durable SSE signal retries */ }
    }, 350);
  }

  function connectEvents() {
    state.eventSource?.close();
    state.eventSource = new EventSource(`/api/sessions/${encodeURIComponent(sessionId)}/events?after=${state.latestSequence}&observer=1`);
    state.eventSource.addEventListener("activity", scheduleLiveRefresh);
    state.eventSource.addEventListener("snapshot", scheduleLiveRefresh);
    state.eventSource.onerror = () => {
      if (state.overview?.state === "FROZEN") state.eventSource?.close();
    };
  }

  async function init() {
    if (!sessionId) return showError(new Error("Session id is missing from the URL."));
    setBusy(true);
    try {
      await loadOverview();
      await renderSection(state.section);
      connectEvents();
    } catch (error) { showError(error); }
  }

  document.querySelectorAll("#sectionNav button").forEach((button) => button.addEventListener("click", () => {
    const section = button.dataset.section;
    history.replaceState(null, "", `#${section}`);
    renderSection(section);
    document.querySelector(".trace-workspace")?.focus({ preventScroll: true });
  }));
  $("traceSearch").addEventListener("input", (event) => {
    state.query = event.target.value;
    clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(() => renderSection(state.section, { quiet: true }), 180);
  });
  $("runPicker").addEventListener("change", async (event) => {
    const [goalId, runId] = String(event.target.value).split("|");
    state.goalId = goalId || ""; state.runId = runId || ""; state.selected = null;
    inspector.dataset.open = "false"; inspectorContent.hidden = true; inspectorEmpty.hidden = false;
    setBusy(true);
    try { await loadOverview({ preserveSelection: true }); await renderSection(state.section); connectEvents(); }
    catch (error) { showError(error); }
  });
  $("exportButton").addEventListener("click", exportTrace);
  window.addEventListener("hashchange", () => renderSection((location.hash || "#overview").slice(1)));
  window.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== $("traceSearch")) {
      event.preventDefault(); $("traceSearch").focus();
    }
    if (event.key === "Escape" && inspector.dataset.open === "true") {
      inspector.dataset.open = "false"; inspectorContent.hidden = true; inspectorEmpty.hidden = false;
    }
  });
  window.addEventListener("beforeunload", () => state.eventSource?.close());
  init();
})();

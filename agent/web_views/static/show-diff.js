(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const sessionId = location.pathname.match(/^\/sessions\/([^/]+)\/show-diff\/?$/)?.[1] || "";
  const apiRoot = `/api/sessions/${encodeURIComponent(sessionId)}/show-diff`;
  const state = { snapshot: null, detail: null, changeId: "", filePath: "", query: "", source: null, refreshTimer: null, signature: "" };

  function element(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (key === "text") node.textContent = value;
      else if (key === "onClick") node.addEventListener("click", value);
      else if (value !== null && value !== undefined) node.setAttribute(key, value);
    });
    (Array.isArray(children) ? children : [children]).filter(Boolean).forEach((child) => node.append(child));
    return node;
  }

  async function request(path = "") {
    const response = await fetch(`${apiRoot}${path}`, { credentials: "same-origin", cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || data.detail || `Request failed (${response.status})`);
    return data;
  }

  let toastTimer = null;
  function toast(message) {
    $("toast").textContent = message;
    $("toast").dataset.show = "true";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("toast").dataset.show = "false"; }, 2300);
  }

  async function copyText(value, label) {
    try { await navigator.clipboard.writeText(String(value || "")); toast(label); }
    catch (_) { toast("Clipboard access is unavailable"); }
  }

  function setView(name) {
    $("loadingState").hidden = name !== "loading";
    $("errorState").hidden = name !== "error";
    $("emptyState").hidden = name !== "empty";
    $("diffContent").hidden = name !== "diff";
  }

  function timeLabel(value) {
    const date = new Date(value || "");
    return Number.isNaN(date.getTime()) ? "just now" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function statusLetter(status) {
    return status === "added" ? "A" : status === "deleted" ? "D" : "M";
  }

  function renderHeartbeat(snapshot) {
    const badge = $("stateBadge");
    badge.textContent = snapshot.state || "IDLE";
    badge.dataset.state = String(snapshot.state || "idle").toLowerCase();
    $("updatedLabel").textContent = `${snapshot.goal_status || "idle"} · updated ${timeLabel(snapshot.updated_at)}`;
    $("advancedLink").href = `/sessions/${encodeURIComponent(sessionId)}/advanced-tracing#changes`;
  }

  function renderChrome(snapshot) {
    renderHeartbeat(snapshot);
    const picker = $("changePicker");
    picker.replaceChildren();
    (snapshot.changes || []).forEach((change) => {
      const option = element("option", {
        value: change.id,
        text: `${change.kind === "working" ? "Current workspace" : timeLabel(change.created_at)} · ${change.file_count} files · ${change.agent || "workflow"}`,
      });
      if (change.id === state.changeId) option.selected = true;
      picker.append(option);
    });
    picker.disabled = !(snapshot.changes || []).length;
    $("copyAllButton").disabled = !state.detail?.diff;
  }

  function renderFiles() {
    const detail = state.detail;
    const files = (detail?.files || []).filter((file) => file.path.toLowerCase().includes(state.query.toLowerCase()));
    $("indexSummary").replaceChildren(
      element("strong", { text: `${detail?.files?.length || 0} file${detail?.files?.length === 1 ? "" : "s"}` }),
      element("span", { text: `+${detail?.additions || 0} −${detail?.deletions || 0}` }),
    );
    const list = $("fileList");
    list.replaceChildren();
    if (!files.length) {
      list.append(element("p", { class: "no-files", text: state.query ? "No changed file matches this search." : "No line-level file patch was recorded." }));
      return;
    }
    files.forEach((file) => {
      const button = element("button", { class: "file-row", type: "button" }, [
        element("span", { class: "file-status", "data-status": file.status, text: statusLetter(file.status) }),
        element("span", { class: "file-name", text: file.path }),
        element("span", { class: "file-counts", text: `+${file.additions || 0} −${file.deletions || 0}` }),
      ]);
      if (file.path === state.filePath) button.setAttribute("aria-current", "true");
      button.addEventListener("click", () => { state.filePath = file.path; renderFiles(); renderDiff(); });
      list.append(button);
    });
  }

  function diffRows(raw) {
    let oldLine = 0;
    let newLine = 0;
    return String(raw || "").split("\n").map((text) => {
      let kind = "context";
      let oldNumber = "";
      let newNumber = "";
      let sign = text[0] || " ";
      const hunk = text.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      if (hunk) {
        oldLine = Number(hunk[1]); newLine = Number(hunk[2]); kind = "hunk"; sign = "";
      } else if (text === "--- /dev/null") {
        newLine = 1; kind = "meta"; sign = "";
      } else if (text === "+++ /dev/null") {
        oldLine = 1; kind = "meta"; sign = "";
      } else if (text.startsWith("+") && !text.startsWith("+++")) {
        kind = "added"; newNumber = String(newLine++);
      } else if (text.startsWith("-") && !text.startsWith("---")) {
        kind = "deleted"; oldNumber = String(oldLine++);
      } else if (text.startsWith(" ")) {
        oldNumber = String(oldLine++); newNumber = String(newLine++); sign = " ";
      } else if (/^(diff --git|index |--- |\+\+\+ |new file|deleted file|similarity|rename)/.test(text)) {
        kind = "meta"; sign = "";
      } else {
        kind = "meta"; sign = "";
      }
      return { text, kind, oldNumber, newNumber, sign };
    });
  }

  function renderDiff() {
    const files = state.detail?.files || [];
    let file = files.find((item) => item.path === state.filePath);
    if (!file) file = files[0];
    if (!file) { setView("empty"); return; }
    state.filePath = file.path;
    $("changeLabel").textContent = `${state.detail.kind === "working" ? "CURRENT WORKSPACE" : "RECORDED WORKFLOW CHANGE"} · ${String(state.detail.status || "recorded").toUpperCase()}`;
    $("fileTitle").textContent = file.path;
    $("fileMeta").textContent = `${file.status} · +${file.additions || 0} −${file.deletions || 0}${file.agent ? ` · ${file.agent}` : ""}${file.task ? ` · ${file.task}` : ""}`;
    const table = $("diffTable");
    table.replaceChildren();
    if (!file.diff) {
      table.append(element("p", { class: "no-files", text: "This file was recorded as changed, but no line-level patch is available yet." }));
    } else {
      diffRows(file.diff).forEach((line) => table.append(element("div", { class: `diff-line ${line.kind}`, role: "row" }, [
        element("span", { class: "line-no", text: line.oldNumber }),
        element("span", { class: "line-no", text: line.newNumber }),
        element("span", { class: "line-sign", text: line.sign }),
        element("span", { class: "line-code", text: line.text ? line.text.slice(line.sign ? 1 : 0) : " " }),
      ])));
    }
    $("copyFileButton").onclick = () => copyText(file.diff, "File diff copied");
    $("downloadButton").onclick = () => downloadPatch(file.diff, file.path);
    $("copyAllButton").disabled = !state.detail.diff;
    setView("diff");
  }

  function downloadPatch(value, path) {
    const blob = new Blob([String(value || "")], { type: "text/x-diff;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = element("a", { href: url, download: `${String(path || "workflow").replace(/[\\/:*?\"<>|]+/g, "-")}.patch` });
    document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(url);
    toast("Patch downloaded");
  }

  async function loadDetail(id, snapshot) {
    if (!id) { state.detail = null; state.filePath = ""; setView("empty"); return; }
    state.detail = snapshot.selected?.id === id ? snapshot.selected : await request(`/${encodeURIComponent(id)}`);
    state.changeId = id;
    if (!(state.detail.files || []).some((file) => file.path === state.filePath)) state.filePath = state.detail.files?.[0]?.path || "";
    renderChrome(snapshot); renderFiles(); renderDiff();
  }

  async function refresh({ initial = false } = {}) {
    if (initial) setView("loading");
    try {
      const snapshot = await request();
      const signature = JSON.stringify({
        state: snapshot.state,
        goal_status: snapshot.goal_status,
        selected_id: snapshot.selected_id,
        changes: (snapshot.changes || []).map((item) => [
          item.id, item.status, item.diff_hash, item.file_count,
          item.additions, item.deletions, item.integration_status,
        ]),
      });
      if (!initial && signature === state.signature) {
        state.snapshot = snapshot;
        renderHeartbeat(snapshot);
        return;
      }
      state.signature = signature;
      state.snapshot = snapshot;
      const ids = new Set((snapshot.changes || []).map((item) => item.id));
      const selected = ids.has(state.changeId) ? state.changeId : snapshot.selected_id;
      renderChrome(snapshot);
      if (!selected) {
        state.detail = null; state.changeId = ""; state.filePath = "";
        $("fileList").replaceChildren();
        $("indexSummary").replaceChildren(element("strong", { text: "0 files" }), element("span", { text: "+0 −0" }));
        setView("empty");
        return;
      }
      await loadDetail(selected, snapshot);
    } catch (error) {
      $("errorState").textContent = `Workflow Diff could not be loaded.\n${error.message || error}`;
      setView("error");
    }
  }

  function scheduleRefresh() {
    clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(() => refresh(), 300);
  }

  function connectEvents() {
    state.source?.close();
    state.source = new EventSource(`/api/sessions/${encodeURIComponent(sessionId)}/events?observer=1`);
    state.source.addEventListener("activity", scheduleRefresh);
    state.source.addEventListener("snapshot", scheduleRefresh);
  }

  $("changePicker").addEventListener("change", async (event) => {
    try { setView("loading"); await loadDetail(event.target.value, state.snapshot); }
    catch (error) { $("errorState").textContent = error.message || error; setView("error"); }
  });
  $("fileSearch").addEventListener("input", (event) => { state.query = event.target.value; renderFiles(); });
  $("copyAllButton").addEventListener("click", () => copyText(state.detail?.diff, "Full diff copied"));
  window.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== $("fileSearch")) { event.preventDefault(); $("fileSearch").focus(); }
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "c") { event.preventDefault(); copyText(state.detail?.diff, "Full diff copied"); }
  });
  window.addEventListener("beforeunload", () => state.source?.close());

  if (!sessionId) {
    $("errorState").textContent = "Session id is missing from the URL.";
    setView("error");
    return;
  }
  refresh({ initial: true }).then(connectEvents);
  setInterval(() => { if (!document.hidden) refresh(); }, 3000);
})();

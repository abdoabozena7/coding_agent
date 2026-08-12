(() => {
  "use strict";

  const parts = location.pathname.split("/").filter(Boolean);
  const sessionId = parts[1] || "";
  const api = `/api/sessions/${encodeURIComponent(sessionId)}/output`;
  const $ = (id) => document.getElementById(id);
  const liveUrl = `/sessions/${encodeURIComponent(sessionId)}/live`;
  let toastTimer = 0;

  ["liveLink", "liveNav", "emptyLiveLink"].forEach((id) => { $(id).href = liveUrl; });

  function toast(message) {
    $("toast").textContent = message;
    $("toast").classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $("toast").classList.remove("show"), 1600);
  }

  async function copyText(value, button) {
    const text = String(value || "");
    try {
      await navigator.clipboard.writeText(text);
    } catch (_error) {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    const original = button.textContent;
    button.textContent = "Copied";
    toast("Copied to clipboard");
    setTimeout(() => { button.textContent = original; }, 1200);
  }

  function renderCopySections(items) {
    $("copyItems").replaceChildren();
    $("copySection").classList.toggle("hidden", !items.length);
    items.forEach((item) => {
      const row = document.createElement("article");
      row.className = "copy-item";
      const controls = document.createElement("div");
      const label = document.createElement("h3");
      label.textContent = item.label || "Copy ready";
      const button = document.createElement("button");
      button.className = "copy-button";
      button.type = "button";
      button.textContent = "Copy section";
      button.addEventListener("click", () => copyText(item.text, button));
      controls.append(label, button);
      const text = document.createElement("pre");
      text.textContent = item.text || "";
      row.append(controls, text);
      $("copyItems").append(row);
    });
  }

  function splitMessage(value) {
    const text = String(value || "").trim();
    const match = /\n(?:Evidence|Verification details|Verification):\s*\n/i.exec(text);
    if (!match) return { response: text, verification: "" };
    return {
      response: text.slice(0, match.index).trim(),
      verification: text.slice(match.index + 1).trim(),
    };
  }

  function displayLabel(value, fallback) {
    const cleaned = String(value || "")
      .replace(/-[0-9a-f]{8}$/i, "")
      .replace(/^\d+[_-]+/, "")
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    return cleaned || fallback;
  }

  function renderAssets(items) {
    $("assets").replaceChildren();
    $("assetsSection").classList.toggle("hidden", !items.length);
    items.forEach((item) => {
      if (item.kind === "image") {
        const figure = document.createElement("figure");
        figure.className = "asset";
        const link = document.createElement("a");
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noopener";
        const image = document.createElement("img");
        image.src = item.url;
        const readableLabel = displayLabel(item.label, "Result image");
        image.alt = readableLabel;
        image.loading = "lazy";
        link.append(image);
        const caption = document.createElement("figcaption");
        const detail = document.createElement("div");
        const label = document.createElement("strong");
        label.textContent = readableLabel;
        const path = document.createElement("code");
        path.textContent = item.path || "";
        detail.append(label, path);
        const download = document.createElement("a");
        download.href = item.download_url;
        download.textContent = "Download";
        caption.append(detail, download);
        figure.append(link, caption);
        $("assets").append(figure);
      } else {
        const row = document.createElement("div");
        row.className = "file-asset";
        const label = document.createElement("span");
        label.textContent = displayLabel(item.label || item.path, "File");
        const download = document.createElement("a");
        download.href = item.download_url;
        download.textContent = "Download";
        row.append(label, download);
        $("assets").append(row);
      }
    });
  }

  async function load() {
    const response = await fetch(api, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Output request failed (${response.status})`);
    const data = await response.json();
    $("result").setAttribute("aria-busy", "false");
    if (data.status === "empty") {
      $("outputTitle").textContent = "No output yet";
      $("messageSection").classList.add("hidden");
      $("emptyState").classList.remove("hidden");
      $("outputState").querySelector("span").textContent = "Waiting";
      return;
    }
    document.title = "Result · GA3BAD";
    $("outputTitle").textContent = "Result";
    const complete = data.status === "ready";
    $("outputMeta").textContent = complete
      ? (data.created_at ? `Completed ${new Date(data.created_at).toLocaleString()}` : "Completed task response")
      : (data.status === "needs_attention" ? "The saved result still needs attention" : "The task is still updating this result");
    const content = splitMessage(data.message);
    $("message").textContent = content.response;
    $("verificationDetails").classList.toggle("hidden", !content.verification);
    $("verificationText").textContent = content.verification;
    $("outputState").classList.toggle("ready", complete);
    $("outputState").querySelector("span").textContent = complete
      ? "Complete"
      : (data.status === "needs_attention" ? "Needs attention" : "Working");
    $("copyAll").disabled = false;
    $("copyAll").addEventListener("click", () => copyText(data.message, $("copyAll")), { once: false });
    renderCopySections(Array.isArray(data.copy_sections) ? data.copy_sections : []);
    renderAssets(Array.isArray(data.assets) ? data.assets : []);
  }

  load().catch((error) => {
    $("result").setAttribute("aria-busy", "false");
    $("outputTitle").textContent = "Output could not load";
    $("outputMeta").textContent = error.message;
    $("outputState").querySelector("span").textContent = "Unavailable";
  });
})();

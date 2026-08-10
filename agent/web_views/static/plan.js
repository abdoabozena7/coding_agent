"use strict";

function planRequestText(plan) {
  return plan.current_request || plan.semantic_goal?.original_request || plan.objective || "";
}

function planUserMessage(plan) {
  const request = planRequestText(plan);
  return request
    ? `<article class="chat-message user-message"><span class="message-role">You</span><p>${escapeHtml(request)}</p></article>`
    : "";
}

function autoGrowPlanInput(editor, maxHeight = 180) {
  if (!editor) return;
  editor.style.height = "auto";
  editor.style.height = `${Math.min(editor.scrollHeight, maxHeight)}px`;
}

function planRequirementCoverage(plan) {
  const anchors = plan.semantic_goal?.requirement_anchors || [];
  if (!anchors.length) return "";
  return `<details class="requirement-meaning" open>
    <summary>${anchors.length} request detail${anchors.length === 1 ? "" : "s"} carried into this plan</summary>
    <ul>${anchors.map((anchor) => `<li><strong>${escapeHtml(anchor.verbatim_span || anchor.id || "Requirement")}</strong><span>${escapeHtml(anchor.interpreted_requirement || (anchor.observable_implications || []).join(" · "))}</span></li>`).join("")}</ul>
  </details>`;
}

function planRequestComposer() {
  return `<form class="plan-composer" id="requestComposer">
    <label class="sr-only" for="planRequest">What should Ultra plan?</label>
    <textarea id="planRequest" rows="1" placeholder="Describe what you want to build…">${escapeHtml(state.requestDraft)}</textarea>
    <div class="composer-footer"><span>Draft saved here · Ctrl + Enter to send</span><button id="submitPlanRequest" class="primary-button" type="submit">Send to LLM</button></div>
  </form>`;
}

function bindRequestComposer() {
  const editor = $("#planRequest");
  autoGrowPlanInput(editor);
  $("#requestComposer").addEventListener("submit", (event) => {
    event.preventDefault();
    submitPlanRequest();
  });
  editor.addEventListener("input", () => {
    state.requestDraft = editor.value;
    state.requestDirty = true;
    writeDraft("plan-request", editor.value);
    autoGrowPlanInput(editor);
  });
  editor.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") submitPlanRequest();
  });
}

function planQuestionMarkup(question) {
  const choices = (question.options || []).map((option, index) => `
    <button class="choice-button" data-answer="${escapeHtml(option.value)}" type="button"><span>${index + 1}</span><span>${escapeHtml(option.label || option.value)}</span></button>`).join("");
  const composer = question.allow_freeform === false ? "" : `<form class="plan-composer" id="answerComposer">
    <label class="sr-only" for="freeAnswer">Your answer</label>
    <textarea id="freeAnswer" rows="1" placeholder="Write a precise answer…">${escapeHtml(state.questionDraft)}</textarea>
    <div class="composer-footer"><span>Draft saved here · Ctrl + Enter to send</span><button class="primary-button" type="submit">Send answer</button></div>
  </form>`;
  return `<article class="chat-message assistant-message question-stage"><span class="message-role">Ultra · one question</span><h2>${escapeHtml(question.question)}</h2><p>I will keep every earlier detail and continue the plan after this answer.</p>${choices ? `<div class="choices">${choices}</div>` : ""}</article>${composer}`;
}

function bindPlanQuestion(question) {
  $$('[data-answer]').forEach((button) => button.addEventListener("click", () => answerQuestion(question, button.dataset.answer)));
  const editor = $("#freeAnswer");
  if (!editor) return;
  autoGrowPlanInput(editor);
  editor.addEventListener("input", () => {
    state.questionDraft = editor.value;
    state.questionDirty = true;
    writeDraft("plan-answer", editor.value, state.activeQuestionId);
    autoGrowPlanInput(editor);
  });
  editor.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") answerQuestion(question, editor.value);
  });
  $("#answerComposer").addEventListener("submit", (event) => {
    event.preventDefault();
    answerQuestion(question, editor.value);
  });
}

function planTeamMarkup(preview) {
  return `<article id="teamStage" class="chat-message assistant-message team-stage">
    <div class="message-heading"><div><span class="message-role">Ultra · first layer</span><h2>Run preview</h2></div><p>${preview.agents.length} dependency-aware agent${preview.agents.length === 1 ? "" : "s"}. The recursive harness grows child nodes only when the work needs real boundaries.</p></div>
    <div class="team-list">${preview.agents.map((agent, index) => `<article class="team-member"><header><h3>${escapeHtml(agent.name)}</h3><span class="agent-index">A${String(index + 1).padStart(2, "0")}</span></header><p>${escapeHtml(agent.mission)}</p><small>${escapeHtml(agent.role)}${agent.depends_on?.length ? ` · after ${escapeHtml(agent.depends_on.join(", "))}` : " · ready first"}</small></article>`).join("")}</div>
    <div class="document-actions"><span class="handoff-note"><i></i>The terminal will receive this exact plan and first layer.</span><button id="approveStart" class="primary-button" type="button">Start Ultra</button></div>
  </article>`;
}

renderPlan = function renderPlanConversation() {
  if (state.page !== "plan" || !state.plan || !state.context) return;
  const root = $("#planRoot");
  const plan = state.plan;
  const question = state.context.pending_question || state.context.required_action?.question;

  if (question) {
    const questionId = String(question.id || "question");
    if (state.activeQuestionId !== questionId) {
      state.activeQuestionId = questionId;
      state.questionDraft = readDraft("plan-answer", questionId);
      state.questionDirty = Boolean(state.questionDraft);
    }
    root.innerHTML = `<section class="plan-chat"><div class="plan-thread">${planUserMessage(plan)}${planQuestionMarkup(question)}</div></section>`;
    bindPlanQuestion(question);
    return;
  }

  if (plan.state === "new_request") {
    root.innerHTML = `<section class="plan-chat empty-chat"><div class="plan-thread"><div class="chat-welcome"><span class="eyebrow">ULTRA PLAN</span><h1>What should we plan?</h1><p>Send a short request or a complete brief. Every detail stays attached while Ultra inspects the workspace and writes the full plan.</p><small>No code runs until you review the plan and start Ultra.</small></div></div>${planRequestComposer()}</section>`;
    bindRequestComposer();
    return;
  }

  if (plan.state === "preparing_plan" || !plan.revision) {
    const stopped = ["paused", "blocked", "cancelled"].includes(String(plan.goal_status));
    root.innerHTML = `<section class="plan-chat"><div class="plan-thread">${planUserMessage(plan)}<article class="chat-message assistant-message planning-message"><span class="message-role">Ultra</span><div class="planning-copy"><div><h2>${escapeHtml(stopped ? "The saved plan needs attention." : "Writing the complete plan…")}</h2><p>${escapeHtml(plan.runtime?.active_operation || plan.summary || "Inspecting the workspace, preserving every request detail, and defining verifiable work.")}</p></div><span class="status-badge ${stopped ? "blocked" : "running"}">${stopped ? "Waiting" : "Planning"}</span></div><div class="planning-lines" aria-hidden="true"><span></span><span></span><span></span></div></article></div></section>`;
    return;
  }

  if (["running", "verifying", "reviewing", "completed"].includes(String(plan.goal_status))) {
    renderHandoff(plan.goal_status === "completed");
    return;
  }

  if (!state.documentDirty) state.planDocument = plan.document || state.planDocument;
  const preview = plan.team_preview;
  root.innerHTML = `<section class="plan-chat"><div class="plan-thread">${planUserMessage(plan)}
    <article class="chat-message assistant-message document-stage">
      <div class="message-heading"><div><span class="message-role">Ultra · plan r${escapeHtml(plan.revision)}</span><h2>Here is the complete plan.</h2></div><p>Edit it directly before preparing the run.</p></div>
      <label class="sr-only" for="planDocument">Editable plan document</label>
      <textarea id="planDocument" class="plan-document" spellcheck="true">${escapeHtml(state.planDocument || plan.document || "")}</textarea>
      ${planRequirementCoverage(plan)}
      <div class="document-actions"><p id="documentState">${preview ? "This plan matches the team shown below." : "Nothing changes in the workspace yet."}</p><button id="prepareAgents" class="primary-button" type="button">${preview ? "Update run preview" : "Preview the run"}</button></div>
    </article>${preview ? planTeamMarkup(preview) : ""}</div></section>`;

  const editor = $("#planDocument");
  editor.addEventListener("input", () => {
    state.planDocument = editor.value;
    state.documentDirty = true;
    $("#teamStage")?.classList.add("hidden");
    $("#prepareAgents").textContent = "Update run preview";
    $("#documentState").textContent = "Your edit is safe in this tab. Update the preview when ready.";
  });
  $("#prepareAgents").addEventListener("click", prepareAgents);
  $("#approveStart")?.addEventListener("click", approveAndStart);
};

submitPlanRequest = async function submitPlanRequestToLlm() {
  const input = $("#planRequest");
  const request = input.value.trim();
  if (!request) return toast("Describe the outcome first.", true);
  const button = $("#submitPlanRequest");
  button.disabled = true;
  button.textContent = "Sending…";
  try {
    await api("/plan/request", { method: "POST", body: JSON.stringify({ request }) });
    writeDraft("plan-request", "");
    state.requestDraft = "";
    state.requestDirty = false;
    toast("Request saved. Ultra is writing the plan.");
    await refresh({ force: true });
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
    button.textContent = "Send to LLM";
  }
};

answerQuestion = async function answerPlanQuestion(question, value) {
  const answer = String(value || "").trim();
  if (!answer) return toast("Write an answer first.", true);
  try {
    await api("/actions", { method: "POST", body: JSON.stringify({ action: "answer", target_id: question.id, action_fingerprint: question.id, value: answer, source: "web" }) });
    writeDraft("plan-answer", "", state.activeQuestionId);
    state.questionDraft = "";
    state.questionDirty = false;
    toast("Answer saved. Ultra is continuing the plan.");
    await refresh({ force: true });
  } catch (error) { toast(error.message, true); }
};

prepareAgents = async function preparePlanRunPreview() {
  const documentValue = $("#planDocument").value.trim();
  if (!documentValue) return toast("The plan document cannot be empty.", true);
  const button = $("#prepareAgents");
  button.disabled = true;
  button.textContent = "Preparing preview…";
  try {
    await api("/plan/team-preview", { method: "POST", body: JSON.stringify({ base_revision: state.plan.revision, document: documentValue }) });
    state.documentDirty = false;
    state.planDocument = documentValue;
    toast("The run preview is ready.");
    await refresh({ force: true });
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
    button.textContent = "Preview the run";
  }
};

approveAndStart = async function startUltraFromPlan() {
  const preview = state.plan?.team_preview;
  if (!preview) return toast("Update the run preview before starting.", true);
  const button = $("#approveStart");
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    await api("/plan/approve", { method: "POST", body: JSON.stringify({ revision: preview.plan_revision, plan_fingerprint: preview.plan_fingerprint, team_fingerprint: preview.team_fingerprint }) });
    state.eventSource?.close();
    clearInterval(state.pollTimer);
    renderHandoff(false);
    setConnection("", "Terminal handoff");
    toast("Plan sent. The terminal is starting Ultra.");
    setTimeout(() => window.close(), 120);
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
    button.textContent = "Start Ultra";
  }
};

init();

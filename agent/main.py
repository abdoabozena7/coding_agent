"""
Phase 8 — Context management, now multi-provider.

Cumulative agent: chat loop, tools, streaming, system prompt, permission gate,
usage reporting, and compaction. The LLM backend is now pluggable (see llm.py +
providers/): switch with the LLM_PROVIDER env var — no other file changes.

Run:  python agent/main.py                      (OpenAI, the default)
      LLM_PROVIDER=gemini python agent/main.py   (Gemini, with visible thinking)
      (type 'exit', or Ctrl-C, to quit)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import llm
import tools
import context
import ui
from prompts import SYSTEM_PROMPT

load_dotenv(override=True)


APP_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_ROOT = Path(os.getenv("AGENT_PROJECTS_DIR", APP_ROOT / "projects")).resolve()


def _next_project_name() -> str:
    """Return the first available project-NNN directory name."""
    i = 1
    while True:
        name = f"project-{i:03d}"
        if not (PROJECTS_ROOT / name).exists():
            return name
        i += 1


def choose_workspace() -> Path:
    """Pick an existing workspace under projects/, or create one on skip."""
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    projects = sorted([p for p in PROJECTS_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    print(f"Projects root: {PROJECTS_ROOT}")
    if projects:
        print("Choose a workspace:")
        for idx, project in enumerate(projects, start=1):
            print(f"  {idx}. {project.name}")
    else:
        print("No workspaces yet.")
    choice = input("Workspace number/name, or Enter/skip to create a new one: ").strip()

    selected = None
    if choice and choice.lower() != "skip":
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(projects):
                selected = projects[idx - 1]
        else:
            candidate = PROJECTS_ROOT / choice
            if candidate.is_dir() and candidate.resolve().parent == PROJECTS_ROOT:
                selected = candidate

        if selected is None:
            print(f"Workspace '{choice}' was not found; creating a new one instead.")

    if selected is None:
        selected = PROJECTS_ROOT / _next_project_name()
        selected.mkdir(parents=True, exist_ok=False)
        print(f"Created workspace: {selected.name}")

    os.chdir(selected)
    print(f"Active workspace: {selected}\n")
    return selected

# Each provider needs its own key — check the one for the active provider.
_PROVIDER = llm.provider_name()
if _PROVIDER == "gemini":
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env for LLM_PROVIDER=gemini.")
elif _PROVIDER == "ollama":
    pass
elif not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY in your .env for LLM_PROVIDER=openai.")

WORKSPACE = choose_workspace()
ACTIVE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + f"\n\nThe active workspace is {WORKSPACE}. Treat relative file paths and shell commands "
      "as operating inside this workspace unless the user explicitly asks for another path."
)

def confirm(name, args) -> bool:
    """Ask the human to approve a risky action before it runs.

    Defaults to NO: anything other than an explicit 'y'/'yes' is treated as a
    denial, so a stray Enter never runs a shell command by accident.
    """
    return ui.confirm_prompt(name) in {"y", "yes"}


# Our entire memory: the running list of messages, resent on every call.
conversation = []

ui.banner(_PROVIDER, llm.model_name())

while True:
    # ----- Outer loop: one turn of conversation with the human. -----
    try:
        user_input = input(ui.user_prompt()).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in {"exit", "quit"}:
        break
    if not user_input:
        continue

    conversation.append({"role": "user", "content": user_input})

    # ----- Inner loop: let the model use tools until it has a final answer. -----
    step = 0
    empty_retries = 0
    while True:
        step += 1
        ui.step_header(step)

        # Stream the reply live. The Streamer prints section headers lazily so a
        # tool-only step (no text) never shows an empty 'answer' block, and a
        # provider with no visible reasoning (OpenAI) shows no 'thinking' block.
        streamer = ui.Streamer()

        # Provider-agnostic: returns a neutral AssistantTurn (text + tool_calls + usage).
        turn = llm.call(
            conversation,
            tools=tools.TOOL_SCHEMAS,
            system=ACTIVE_SYSTEM_PROMPT,
            on_text=streamer.on_text,
            on_thought=streamer.on_thought,
        )
        streamer.close()

        if not turn.text and not turn.tool_calls:
            if empty_retries < 1:
                empty_retries += 1
                conversation.append({
                    "role": "user",
                    "content": (
                        "You returned no answer and no tool calls. Continue now: "
                        "either call the needed tools to complete the user's request, "
                        "or give a concise final answer."
                    ),
                })
                continue
            print("No answer or tool call was returned by the model.")
            ui.turn_end()
            break

        # Append the assistant turn to history in the neutral shape.
        conversation.append(turn.to_message())

        # Token usage. Watch `cached` climb across steps: the repeated prefix is
        # served from cache (~10% the cost). cached=0 on tiny prompts is normal.
        if turn.usage:
            ui.usage(turn.usage)

        # No tool calls -> the model is done. End the turn.
        if not turn.tool_calls:
            ui.turn_end()
            break

        # Run each requested tool and feed the result back.
        for call in turn.tool_calls:
            ui.tool_call(call.name, call.args)   # call.args is already a dict

            # Permission gate, three cases:
            #   - safe (read-only file tools, allow-listed bash) -> just run it
            #   - risky but approved -> run it
            #   - risky and denied -> feed back a denial so the model can adapt
            # The decision can depend on the ARGS, not just the tool name: a
            # `git status` runs unprompted while a `git push` still asks.
            if not tools.requires_approval(call.name, call.args):
                result = tools.run_tool(call.name, call.args)
            elif confirm(call.name, call.args):
                result = tools.run_tool(call.name, call.args)
            else:
                result = "Permission denied by the user."

            ui.tool_result(result)   # show the model what we'll feed back

            # Neutral tool-result message — the adapter maps id/name to its wire format.
            conversation.append({
                "role": "tool",
                "id": call.id,
                "name": call.name,
                "content": result,
            })
        # Loop again — now the model can see the tool results and continue.

    # ----- Turn finished: compact the history if it has grown too large. -----
    # Prompt caching (the [usage] line) keeps the repeated prefix CHEAP; this
    # keeps it from growing UNBOUNDED by summarizing older turns. See context.py.
    conversation = context.maybe_compact(conversation)

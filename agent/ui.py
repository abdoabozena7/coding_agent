"""
ui.py — terminal presentation for the agent loop.

main.py owns the control flow; this module owns how each step LOOKS. Keeping the
two apart leaves the loop readable and puts all the ANSI/formatting fuss in one
place. Each step of a turn is rendered as a clearly separated, labeled block:

  ▸ step N        — which reasoning/tool step we're on within the turn
  💭 thinking     — the model's reasoning summary (Gemini; OpenAI omits it)
  💬 answer       — the model's text to the user
  🔧 tool         — the tool name + its pretty-printed arguments
  📤 result       — what the tool returned (truncated if very long)
  📊 tokens       — token accounting for the step

Colors degrade to plain text automatically when stdout isn't a TTY (pipes/logs).
"""

import json
import sys

_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    """A color code when attached to a real terminal, otherwise nothing."""
    return code if _TTY else ""


RESET = _c("\033[0m")
DIM = _c("\033[2m")
BOLD = _c("\033[1m")
CYAN = _c("\033[36m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
MAGENTA = _c("\033[35m")
BLUE = _c("\033[34m")
GREY = _c("\033[90m")

_RULE = "─" * 56


def banner(provider: str) -> None:
    """One-time header printed at startup."""
    print(f"{BOLD}Coding agent{RESET} {DIM}· provider={provider} · type 'exit' to quit{RESET}\n")


def user_prompt() -> str:
    """The 'you' input prompt, styled to match the other labeled lines."""
    return f"{BOLD}{CYAN}🧑 you{RESET} "


def step_header(n: int) -> None:
    """Open a new reasoning/tool step within the current turn."""
    print(f"\n{DIM}{_RULE}{RESET}")
    print(f"{BOLD}{BLUE}▸ step {n}{RESET}")


class Streamer:
    """Renders streamed thinking/answer fragments under section headers.

    Headers print lazily — only once content for that section actually arrives —
    so a tool-only step (no text) never leaves an orphaned 'answer' label, and a
    provider with no visible reasoning never prints an empty 'thinking' block.
    """

    def __init__(self):
        self._thought_open = False
        self._text_open = False

    def on_thought(self, fragment: str) -> None:
        if not self._thought_open:
            print(f"{MAGENTA}💭 thinking{RESET}")
            print(DIM, end="", flush=True)  # dim the reasoning body
            self._thought_open = True
        print(fragment, end="", flush=True)

    def on_text(self, fragment: str) -> None:
        # Close the thinking block (reset dim + blank line) before the answer.
        if self._thought_open and not self._text_open:
            print(RESET)
            print()
        if not self._text_open:
            print(f"{BOLD}{CYAN}💬 answer{RESET}")
            self._text_open = True
        print(fragment, end="", flush=True)

    def close(self) -> None:
        """End whichever block was streaming, resetting any lingering style."""
        if self._text_open or self._thought_open:
            print(RESET)


def tool_call(name: str, args: dict) -> None:
    """Show the tool the model wants to run and its arguments."""
    print(f"\n{BOLD}{YELLOW}🔧 tool · {name}{RESET}")
    if args:
        pretty = json.dumps(args, indent=2, ensure_ascii=False)
        for line in pretty.splitlines():
            print(f"   {DIM}{line}{RESET}")
    else:
        print(f"   {DIM}(no arguments){RESET}")


def tool_result(result, limit: int = 100) -> None:
    """Show what the tool returned, condensed to a single line.

    We summarize rather than dump: the first line (truncated) plus a count of
    whatever else was hidden, so the transcript stays scannable. The model still
    receives the FULL result — this only trims what the human sees.
    """
    text = str(result).strip()
    lines = text.splitlines() or [""]
    first = lines[0]

    hidden = []
    if len(first) > limit:
        hidden.append(f"+{len(first) - limit} chars")
        first = first[:limit] + "…"
    if len(lines) > 1:
        hidden.append(f"+{len(lines) - 1} lines")
    suffix = f" {DIM}({', '.join(hidden)}){RESET}" if hidden else ""

    print()  # blank line above, matching the spacing of the other blocks
    print(f"{BOLD}{GREEN}📤 result{RESET} {first}{suffix}")


def usage(u) -> None:
    """One dim line of token accounting. Watch `cached` climb across steps."""
    print(
        f"{GREY}📊 tokens · in={u.input_tokens} "
        f"cached={u.cached_tokens} out={u.output_tokens}{RESET}"
    )


def confirm_prompt(name: str) -> str:
    """The styled approval question for a risky tool. Returns the raw answer."""
    return input(f"   {YELLOW}⚠ run {name}? [y/N]{RESET} ").strip().lower()


def turn_end() -> None:
    """Blank separation after the model finishes a turn (no more tools)."""
    print()

"""
Terminal presentation for the agent loop.

main.py owns the control flow; this module owns how each step looks. The labels
are intentionally ASCII-only so the app runs cleanly in Windows code pages.
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

_RULE = "-" * 56


def banner(provider: str, model: str | None = None) -> None:
    """One-time header printed at startup."""
    model_part = f" - model={model}" if model else ""
    print(f"{BOLD}Coding agent{RESET} {DIM}- provider={provider}{model_part} - type 'exit' to quit{RESET}\n")


def user_prompt() -> str:
    """The user input prompt, styled to match the other labeled lines."""
    return f"{BOLD}{CYAN}you>{RESET} "


def step_header(n: int) -> None:
    """Open a new reasoning/tool step within the current turn."""
    print(f"\n{DIM}{_RULE}{RESET}")
    print(f"{BOLD}{BLUE}> step {n}{RESET}")


class Streamer:
    """Renders streamed thinking/answer fragments under section headers."""

    def __init__(self):
        self._thought_open = False
        self._text_open = False

    def on_thought(self, fragment: str) -> None:
        if not self._thought_open:
            print(f"{MAGENTA}thinking{RESET}")
            print(DIM, end="", flush=True)
            self._thought_open = True
        print(fragment, end="", flush=True)

    def on_text(self, fragment: str) -> None:
        if self._thought_open and not self._text_open:
            print(RESET)
            print()
        if not self._text_open:
            print(f"{BOLD}{CYAN}answer{RESET}")
            self._text_open = True
        print(fragment, end="", flush=True)

    def close(self) -> None:
        """End whichever block was streaming, resetting any lingering style."""
        if self._text_open or self._thought_open:
            print(RESET)


def tool_call(name: str, args: dict) -> None:
    """Show the tool the model wants to run and its arguments."""
    print(f"\n{BOLD}{YELLOW}tool - {name}{RESET}")
    if args:
        pretty = json.dumps(args, indent=2, ensure_ascii=False)
        for line in pretty.splitlines():
            print(f"   {DIM}{line}{RESET}")
    else:
        print(f"   {DIM}(no arguments){RESET}")


def tool_result(result, limit: int = 100) -> None:
    """Show what the tool returned, condensed to a single line."""
    text = str(result).strip()
    lines = text.splitlines() or [""]
    first = lines[0]

    hidden = []
    if len(first) > limit:
        hidden.append(f"+{len(first) - limit} chars")
        first = first[:limit] + "..."
    if len(lines) > 1:
        hidden.append(f"+{len(lines) - 1} lines")
    suffix = f" {DIM}({', '.join(hidden)}){RESET}" if hidden else ""

    print()
    print(f"{BOLD}{GREEN}result{RESET} {first}{suffix}")


def usage(u) -> None:
    """One dim line of token accounting."""
    print(f"{GREY}tokens - in={u.input_tokens} cached={u.cached_tokens} out={u.output_tokens}{RESET}")


def confirm_prompt(name: str) -> str:
    """The styled approval question for a risky tool. Returns the raw answer."""
    return input(f"   {YELLOW}run {name}? [y/N]{RESET} ").strip().lower()


def turn_end() -> None:
    """Blank separation after the model finishes a turn."""
    print()

"""Reusable full-screen terminal UX primitives.

The module deliberately keeps navigation and rendering logic independent from
``prompt_toolkit``.  Callers can therefore snapshot the UI and exercise every
choice transition without owning a real terminal.  ``prompt_toolkit`` is an
optional enhancement: importing this module remains safe in minimal or
redirected environments.
"""

from __future__ import annotations

import os
import asyncio
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO

from .ui_state import (
    ActivityStage,
    AttentionKind,
    ExperienceMode,
    WorkspaceSnapshot,
    WorkspaceUIStore,
)
from .tui_commands import CommandSpec, matching_commands
from .plan_document import PlanDocumentError, parse_plan_document

try:  # Optional at import time; line-mode callers must keep working without it.
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import DynamicContainer, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.output.defaults import create_output
    from prompt_toolkit.styles import Style

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - covered through the public fallback API.
    Application = get_app = Buffer = FormattedText = KeyBindings = None  # type: ignore[assignment]
    DynamicContainer = HSplit = Layout = VSplit = Window = None  # type: ignore[assignment]
    BufferControl = FormattedTextControl = Dimension = MouseEventType = Style = create_output = None  # type: ignore[assignment]
    PROMPT_TOOLKIT_AVAILABLE = False


_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}
_EXIT_SELECTION = object()
_UNUSABLE_OUTPUT = object()
NINE_DOT_STATES = (
    "discover",
    "search",
    "sync",
    "plan",
    "run",
    "review",
    "success",
    "warning",
    "error",
    "idle",
)
_NINE_DOT_TOPOLOGY = (0, 1, 2, 7, 8, 3, 6, 5, 4)


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in _FALSE_ENV_VALUES


def _isatty(stream: Any) -> bool:
    """Best-effort TTY detection, including common stream wrappers."""

    current = stream
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        method = getattr(current, "isatty", None)
        if callable(method):
            try:
                return bool(method())
            except (AttributeError, OSError, ValueError):
                return False
        current = getattr(current, "wrapped", None) or getattr(current, "stream", None)
    return False


def rich_terminal_available(
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> bool:
    """Return whether a full-screen interactive application is appropriate.

    Explicit plain-UI settings and redirected streams always win.  Requiring
    the built-in ``input`` avoids unexpectedly taking over tests or embedders
    which supplied their own input callback.
    """

    if not PROMPT_TOOLKIT_AVAILABLE or input_func is not input:
        return False
    if _env_enabled("GA3BAD_PLAIN_UI") or os.environ.get("TERM", "").lower() == "dumb":
        return False
    if not (_isatty(sys.stdin) and _isatty(output)):
        return False
    if output is not sys.stdout and _prompt_output(output, None) is _UNUSABLE_OUTPUT:
        return False
    return True


class UserExitRequested(Exception):
    """Raised when Ctrl+Q requests leaving the whole interactive session."""


def terminal_supports_unicode(output: TextIO = sys.stdout) -> bool:
    """Conservatively determine whether fixed-width Unicode UI marks are safe."""

    encoding = str(getattr(output, "encoding", "") or "").lower()
    if not encoding:
        return os.name != "nt"
    try:
        # Test the complete fixed-width glyph vocabulary, not merely whether the
        # codec has a non-ASCII name.  Windows code pages such as cp1256 report
        # a valid encoding but cannot represent the selector or nine-dot marks.
        "›↑↓●∙·─✓×◇".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class ChoiceItem:
    """One keyboard-selectable item.

    ``value`` is optional integration data.  Selectors return the full item so
    callers can use either its stable ``key`` or ``value`` without another
    lookup.
    """

    key: str
    label: str
    description: str = ""
    meta: str = ""
    value: Any = None
    disabled: bool = False
    disabled_reason: str = ""

    @property
    def resolved_value(self) -> Any:
        return self.key if self.value is None else self.value

    @property
    def search_text(self) -> str:
        return " ".join(
            str(value) for value in (self.key, self.label, self.description, self.meta) if value
        ).casefold()


@dataclass(slots=True)
class ChoiceListState:
    """Pure state machine shared by the renderer and full-screen selector."""

    items: tuple[ChoiceItem, ...]
    selected_index: int = -1
    query: str = ""
    page_size: int = 7
    filterable: bool = True
    feedback: str = ""

    def __post_init__(self) -> None:
        self.items = tuple(self.items)
        self.page_size = max(1, int(self.page_size))
        if not (0 <= self.selected_index < len(self.items)):
            self.selected_index = self._default_index(self.matching_indices)
        self._ensure_selected()

    @classmethod
    def create(
        cls,
        items: Iterable[ChoiceItem],
        *,
        initial_key: str | None = None,
        page_size: int = 7,
        filterable: bool = True,
    ) -> "ChoiceListState":
        values = tuple(items)
        selected = next(
            (index for index, item in enumerate(values) if item.key == initial_key),
            -1,
        )
        return cls(values, selected, page_size=page_size, filterable=filterable)

    @property
    def matching_indices(self) -> tuple[int, ...]:
        needle = self.query.strip().casefold()
        if not needle:
            return tuple(range(len(self.items)))
        return tuple(index for index, item in enumerate(self.items) if needle in item.search_text)

    @property
    def current(self) -> ChoiceItem | None:
        matches = self.matching_indices
        if self.selected_index not in matches:
            return None
        return self.items[self.selected_index]

    def _default_index(self, indices: Sequence[int]) -> int:
        return next((index for index in indices if not self.items[index].disabled), indices[0] if indices else -1)

    def _ensure_selected(self) -> None:
        matches = self.matching_indices
        if self.selected_index not in matches:
            self.selected_index = self._default_index(matches)

    def select_key(self, key: str) -> bool:
        for index in self.matching_indices:
            if self.items[index].key == key:
                self.selected_index = index
                self.feedback = ""
                return True
        return False

    def move(self, amount: int) -> None:
        matches = self.matching_indices
        if not matches:
            self.selected_index = -1
            return
        try:
            position = matches.index(self.selected_index)
        except ValueError:
            position = 0
        position = max(0, min(len(matches) - 1, position + int(amount)))
        self.selected_index = matches[position]
        self.feedback = ""

    def home(self) -> None:
        matches = self.matching_indices
        self.selected_index = matches[0] if matches else -1
        self.feedback = ""

    def end(self) -> None:
        matches = self.matching_indices
        self.selected_index = matches[-1] if matches else -1
        self.feedback = ""

    def page(self, direction: int) -> None:
        self.move((1 if direction >= 0 else -1) * self.page_size)

    def set_query(self, value: str) -> None:
        if not self.filterable:
            return
        self.query = " ".join(str(value).splitlines())
        self.feedback = ""
        self._ensure_selected()

    def append_query(self, value: str) -> None:
        if self.filterable and value and value.isprintable():
            self.set_query(self.query + value)

    def backspace(self) -> None:
        if self.filterable and self.query:
            self.set_query(self.query[:-1])

    def clear_query(self) -> None:
        self.set_query("")

    def activate(self) -> ChoiceItem | None:
        item = self.current
        if item is None:
            self.feedback = "No matching choice."
            return None
        if item.disabled:
            self.feedback = item.disabled_reason or f"{item.label} is unavailable."
            return None
        self.feedback = ""
        return item

    def viewport(self, size: int | None = None) -> tuple[tuple[int, ...], bool, bool]:
        """Return visible indices and whether rows exist above/below them."""

        matches = self.matching_indices
        limit = max(1, int(size or self.page_size))
        if len(matches) <= limit:
            return matches, False, False
        try:
            position = matches.index(self.selected_index)
        except ValueError:
            position = 0
        start = max(0, min(len(matches) - limit, position - limit // 2))
        end = start + limit
        return matches[start:end], start > 0, end < len(matches)


def _levels(*, high: Iterable[int] = (), medium: Iterable[int] = (), low: Iterable[int] = ()) -> tuple[int, ...]:
    cells = [0] * 9
    for index in low:
        cells[index] = max(cells[index], 1)
    for index in medium:
        cells[index] = max(cells[index], 2)
    for index in high:
        cells[index] = 3
    return tuple(cells)


def _trail_frames(path: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    frames: list[tuple[int, ...]] = []
    for position, head in enumerate(path):
        medium = (path[(position - 1) % len(path)],) if len(path) > 1 else ()
        low = (path[(position - 2) % len(path)],) if len(path) > 2 else ()
        frames.append(_levels(high=(head,), medium=medium, low=low))
    return tuple(frames)


_STATIC_NINE_DOT: dict[str, tuple[int, ...]] = {
    "discover": _levels(high=(0, 2, 4, 6, 8), low=(1, 3, 5, 7)),
    "search": _levels(high=(7, 8, 3), low=(0, 1, 2, 4, 5, 6)),
    "sync": _levels(medium=range(8), low=(8,)),
    "plan": _levels(high=(1, 3, 5, 7, 8), low=(0, 2, 4, 6)),
    "run": _levels(high=(0, 8, 4), low=(2, 6)),
    "review": _levels(high=(0, 2, 4, 6, 8), low=(1, 3, 5, 7)),
    "success": _levels(high=range(9)),
    "warning": _levels(high=(1, 8, 5), low=(0, 2, 4, 6)),
    "error": _levels(high=(0, 2, 4, 6, 8)),
    "idle": _levels(medium=(8,), low=range(8)),
}

_ANIMATED_NINE_DOT: dict[str, tuple[tuple[int, ...], ...]] = {
    "discover": _trail_frames((8, 0, 2, 4, 6, 1, 3, 5, 7)),
    "search": tuple(
        _levels(high=current, medium=previous)
        for current, previous in (
            ((0, 7, 6), ()),
            ((1, 8, 5), (0, 7, 6)),
            ((2, 3, 4), (1, 8, 5)),
            ((1, 8, 5), (2, 3, 4)),
        )
    ),
    "sync": _trail_frames(tuple(range(8))),
    "plan": (
        _levels(high=(8,)),
        _levels(high=(1, 3, 5, 7, 8), low=(0, 2, 4, 6)),
        _levels(high=range(9)),
        _levels(high=(1, 3, 5, 7, 8), low=(0, 2, 4, 6)),
    ),
    "run": (
        _levels(high=(0,)),
        _levels(high=(0, 8)),
        _levels(high=(0, 8, 4)),
        _levels(high=(2, 8, 6)),
    ),
    "review": (
        _levels(high=(0, 2, 4, 6, 8), low=(1, 3, 5, 7)),
        _levels(high=(1, 3, 5, 7, 8), low=(0, 2, 4, 6)),
    ),
    "success": (
        _levels(high=(8,)),
        _levels(high=(1, 3, 5, 7, 8)),
        _levels(high=range(9)),
    ),
    "warning": _trail_frames((0, 1, 2, 3, 2, 1)),
    "error": (
        _levels(high=(0, 2, 4, 6, 8)),
        _levels(low=(0, 2, 4, 6, 8)),
        _levels(high=(0, 2, 4, 6, 8)),
    ),
    "idle": (_STATIC_NINE_DOT["idle"],),
}

_STATE_ALIASES = {
    "discovering": "discover",
    "searching": "search",
    "syncing": "sync",
    "reconnecting": "sync",
    "planning": "plan",
    "thinking": "plan",
    "running": "run",
    "executing": "run",
    "reviewing": "review",
    "testing": "review",
    "verified": "success",
    "complete": "success",
    "completed": "success",
    "retry": "warning",
    "retrying": "warning",
    "failed": "error",
    "paused": "idle",
    "waiting": "idle",
}
_STATE_COLORS = {
    "discover": "cyan",
    "search": "cyan",
    "sync": "cyan",
    "plan": "cyan",
    "run": "blue",
    "review": "magenta",
    "success": "green",
    "warning": "amber",
    "error": "red",
    "idle": "neutral",
}


@dataclass(frozen=True, slots=True)
class NineDotFrame:
    state: str
    cells: tuple[int, ...]
    color: str
    animated: bool

    def __post_init__(self) -> None:
        if len(self.cells) != 9 or any(level not in {0, 1, 2, 3} for level in self.cells):
            raise ValueError("a nine-dot frame must contain nine intensity levels from 0 to 3")


def _normalize_nine_dot_state(state: str) -> str:
    normalized = str(state).strip().lower().replace("-", "_")
    normalized = _STATE_ALIASES.get(normalized, normalized)
    if normalized not in NINE_DOT_STATES:
        choices = ", ".join(NINE_DOT_STATES)
        raise ValueError(f"unknown nine-dot state {state!r}; expected one of: {choices}")
    return normalized


def nine_dot_frame(
    state: str,
    tick: int = 0,
    *,
    reduced_motion: bool = False,
    no_color: bool = False,
) -> NineDotFrame:
    """Return a deterministic semantic animation frame.

    Success and error are finite animations which settle on their final static
    state.  Other active states loop.  Reduced-motion and no-color modes use a
    single state-specific static pattern and schedule no redraws.
    """

    normalized = _normalize_nine_dot_state(state)
    static = reduced_motion or no_color or normalized == "idle"
    if static:
        return NineDotFrame(normalized, _STATIC_NINE_DOT[normalized], "neutral" if no_color else _STATE_COLORS[normalized], False)

    frames = _ANIMATED_NINE_DOT[normalized]
    position = max(0, int(tick))
    finite = normalized in {"success", "error"}
    index = min(position, len(frames) - 1) if finite else position % len(frames)
    cells = frames[index]
    if finite and index == len(frames) - 1:
        cells = _STATIC_NINE_DOT[normalized]
    return NineDotFrame(normalized, cells, _STATE_COLORS[normalized], not finite or index < len(frames) - 1)


def inline_square_levels(
    state: str,
    tick: int = 0,
    *,
    reduced_motion: bool = False,
    no_color: bool = False,
) -> tuple[int, ...]:
    """Flatten the nine loader cells into one fixed-size horizontal row.

    The animation topology remains 3x3 internally, but layout is deliberately
    independent: only the loader occupies one row; surrounding activity text is
    free to wrap onto normal terminal rows.
    """

    frame = nine_dot_frame(
        state,
        tick,
        reduced_motion=reduced_motion,
        no_color=no_color,
    )
    return tuple(frame.cells[index] for index in _NINE_DOT_TOPOLOGY)


def loading_grid_levels(
    state: str,
    tick: int = 0,
    *,
    reduced_motion: bool = False,
    no_color: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Return the semantic loader as a visible three-by-three square grid."""

    levels = inline_square_levels(
        state,
        tick,
        reduced_motion=reduced_motion,
        no_color=no_color,
    )
    return (
        (levels[0], levels[1], levels[2]),
        (levels[3], levels[4], levels[5]),
        (levels[6], levels[7], levels[8]),
    )


def render_nine_dot(
    state: str,
    tick: int = 0,
    *,
    reduced_motion: bool = False,
    no_color: bool = False,
    unicode: bool | None = None,
) -> str:
    """Render a semantic frame as three stable, ANSI-free text rows."""

    frame = nine_dot_frame(state, tick, reduced_motion=reduced_motion, no_color=no_color)
    if unicode is None:
        unicode = not no_color and terminal_supports_unicode(sys.stdout)
    glyphs = ("·", "∙", "•", "●") if unicode else (".", "o", "O", "@")
    ordered = [glyphs[frame.cells[index]] for index in _NINE_DOT_TOPOLOGY]
    return "\n".join(" ".join(ordered[offset : offset + 3]) for offset in range(0, 9, 3))


def _clean_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _fit(value: Any, width: int) -> str:
    text = _clean_line(value)
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _wrapped(value: Any, width: int, max_lines: int = 4) -> list[str]:
    lines = textwrap.wrap(_clean_line(value), max(1, width), break_long_words=True) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit(lines[-1], width).rstrip()
        if width > 3:
            lines[-1] = lines[-1][: width - 3] + "..."
    return lines


def _choice_rows(state: ChoiceListState, width: int, *, unicode: bool = True) -> list[str]:
    visible, above, below = state.viewport()
    lines: list[str] = []
    if above:
        lines.append(_fit("  ↑ more" if unicode else "  [more above]", width))
    if not visible:
        lines.append(_fit("  No matching choices", width))
    for index in visible:
        item = state.items[index]
        marker = "›" if unicode and index == state.selected_index else ">" if index == state.selected_index else " "
        status = "Unavailable" if item.disabled else item.meta
        suffix = f"  {status}" if status else ""
        lines.append(_fit(f"{marker} {item.label}{suffix}", width))
    if below:
        lines.append(_fit("  ↓ more" if unicode else "  [more below]", width))
    return lines


def _choice_details(state: ChoiceListState, width: int) -> list[str]:
    item = state.current
    if item is None:
        return ["No choice selected", "", "Type to change the filter."]
    lines = [_fit(item.label, width).rstrip()]
    if item.meta:
        lines.append(_fit(item.meta, width).rstrip())
    lines.append("")
    lines.extend(_wrapped(item.description or "No additional details.", width, 6))
    if item.disabled:
        lines.extend(("", _fit("Unavailable", width).rstrip()))
        lines.extend(_wrapped(item.disabled_reason or "This choice cannot be selected.", width, 4))
    return lines


def _choice_footer(width: int, action_label: str, filterable: bool, unicode: bool) -> str:
    separator = " · "
    if width < 48:
        move = "↑↓" if unicode else "Keys"
        return separator.join((move, "Enter", "Esc", "Ctrl+Q"))
    if width < 72:
        move = "↑↓" if unicode else "Arrows"
        return separator.join((move, f"Enter {action_label}", "Esc", "Ctrl+Q Exit"))
    if width < 96:
        parts = ["↑↓ Move" if unicode else "Up/Down Move", "Home/End"]
        if filterable:
            parts.append("Type Filter")
        parts.extend((f"Enter {action_label}", "Esc Back", "Ctrl+Q Exit"))
        return separator.join(parts)
    parts = ["↑↓ Move" if unicode else "Up/Down Move", "Home/End", "PgUp/PgDn"]
    if filterable:
        parts.append("Type Filter")
    parts.extend((f"Enter {action_label}", "Esc Back", "Ctrl+Q Exit"))
    return separator.join(parts)


def render_choices(
    state: ChoiceListState,
    *,
    title: str,
    subtitle: str = "",
    step_label: str = "",
    action_label: str = "Choose",
    width: int = 100,
    height: int = 24,
    unicode: bool = True,
) -> str:
    """Pure, color-free selector snapshot used by tests and plain previews."""

    width = max(36, int(width))
    height = max(10, int(height))
    header = [line for line in (step_label, title, subtitle) if line]
    filter_line = f"Filter: {state.query}" if state.query else ""
    footer = _choice_footer(width, action_label, state.filterable, unicode)
    if width >= 88:
        left_width = max(30, int(width * 0.45))
        gap = 3
        right_width = width - left_width - gap
        left = _choice_rows(state, left_width, unicode=unicode)
        right = _choice_details(state, right_width)
        count = max(len(left), len(right))
        body = [
            _fit(left[index] if index < len(left) else "", left_width)
            + " " * gap
            + _fit(right[index] if index < len(right) else "", right_width).rstrip()
            for index in range(count)
        ]
    else:
        body = _choice_rows(state, width, unicode=unicode)
        body.extend(("", *_choice_details(state, width)))
    lines = [*header, ""]
    if filter_line:
        lines.extend((filter_line, ""))
    lines.extend(body)
    if state.feedback:
        lines.extend(("", f"! {state.feedback}"))
    while len(lines) < height - 2:
        lines.append("")
    lines.extend(("", _fit(footer, width).rstrip()))
    return "\n".join(lines[:height])


_WELCOME_GLYPHS: dict[str, tuple[str, ...]] = {
    # A compact 9x11 source grid leaves enough horizontal room to render every
    # pixel two or three columns wide on normal terminals.  Thick stems and a
    # deliberate inter-glyph gutter keep the six characters visually distinct.
    "G": (
        "111111111",
        "111111111",
        "111000000",
        "110000000",
        "110000000",
        "110001111",
        "110001111",
        "110000111",
        "111000111",
        "111111111",
        "111111111",
    ),
    "A": (
        "011111110",
        "011111110",
        "111000111",
        "111000111",
        "111000111",
        "111111111",
        "111111111",
        "111000111",
        "111000111",
        "111000111",
        "111000111",
    ),
    "3": (
        "111111111",
        "111111111",
        "000000111",
        "000000111",
        "111111111",
        "111111111",
        "000000111",
        "000000111",
        "000000111",
        "111111111",
        "111111111",
    ),
    "B": (
        "111111110",
        "111111111",
        "111000111",
        "111000111",
        "111111111",
        "111111110",
        "111000111",
        "111000111",
        "111000111",
        "111111111",
        "111111110",
    ),
    "D": (
        "111111110",
        "111111111",
        "111000011",
        "111000011",
        "111000011",
        "111000011",
        "111000011",
        "111000011",
        "111000011",
        "111111111",
        "111111110",
    ),
}


def _responsive_welcome_brand(
    brand: str,
    width: int,
    height: int,
    *,
    unicode: bool,
) -> str:
    """Render GA3BAD as a detailed numeric-pixel wordmark on roomy screens."""

    clean = _clean_line(brand)
    if clean.upper() != "GA3BAD" or width < 72 or height < 20:
        return brand
    glyph_width = len(next(iter(_WELCOME_GLYPHS.values()))[0])
    glyph_height = len(next(iter(_WELCOME_GLYPHS.values())))
    x_scale = 1
    for candidate in (3, 2):
        candidate_width = (
            len(clean) * glyph_width * candidate
            + (len(clean) - 1) * max(1, candidate - 1)
        )
        if candidate_width <= width:
            x_scale = candidate
            break
    y_scale = max(1, min(2, (height - 10) // glyph_height))
    base_width = len(clean) * glyph_width * x_scale
    available_gap = max(1, (width - base_width) // max(1, len(clean) - 1))
    gap = " " * min(6, available_gap)
    rows: list[str] = []
    for row_index in range(glyph_height):
        segments = []
        for character_index, character in enumerate(clean.upper()):
            pattern = _WELCOME_GLYPHS[character][row_index]
            pixels: list[str] = []
            for cell_index, cell in enumerate(pattern):
                for subpixel in range(x_scale):
                    digit = str((character_index * 7 + row_index * 3 + cell_index + subpixel) % 10)
                    pixels.append(digit if cell == "1" else " ")
            segments.append("".join(pixels))
        row = gap.join(segments).rstrip()
        rows.extend(row for _ in range(y_scale))
    return "\n".join(rows)


def _welcome_layout_lines(
    brand: str,
    subtitle: str,
    action_label: str,
    width: int,
    height: int,
) -> tuple[list[str], int, int]:
    brand_lines = brand.splitlines() or [""]
    content = [*brand_lines]
    if subtitle:
        content.extend(("", _clean_line(subtitle)))
    content.extend(("", "", _clean_line(action_label)))
    top = max(0, (max(1, height - 1) - len(content)) // 2)
    lines = [""] * top + content
    while len(lines) < height - 1:
        lines.append("")
    return lines[: max(0, height - 1)], top, len(brand_lines)


def render_welcome(
    *,
    brand: str = "GA3BAD",
    subtitle: str = "coding agent",
    action_label: str = "Press Enter to begin",
    width: int = 80,
    height: int = 24,
) -> str:
    """Return a centered, plain-text snapshot of the welcome screen."""

    width, height = max(24, int(width)), max(8, int(height))
    rendered_brand = _responsive_welcome_brand(brand, width, height, unicode=False)
    content, _top, _brand_rows = _welcome_layout_lines(
        rendered_brand, subtitle, action_label, width, height
    )
    lines = [line.center(width) for line in content]
    lines.append("Enter Begin · Esc / Ctrl+Q Exit".center(width))
    return "\n".join(lines[:height])


_COLOR_STYLE = {
    "welcome.brand": "bold #35d06f",
    "welcome.subtitle": "#35d06f",
    "welcome.action": "#a0a0a0",
    "welcome.shimmer": "bold #ffffff",
    "header.step": "#6c6c6c",
    "header.title": "bold #f0f0f0",
    "header.subtitle": "#999999",
    "choice": "#c8c8c8",
    "choice.meta": "#a0a0a0",
    "choice.selected": "bold #ffffff bg:#164e63",
    "choice.disabled": "#8a8a8a",
    "details.title": "bold #ffffff",
    "details.body": "#a8a8a8",
    "warning": "bold #f0b429",
    "filter": "#35d06f",
    "composer.prompt": "bold #35d06f",
    "composer.input": "#f0f0f0",
    "footer": "#a0a0a0",
    "loading.dots": "bold #35d06f",
    "loading.square.0": "#444444",
    "loading.square.1": "#707070",
    "loading.square.2": "#aaaaaa",
    "loading.square.3": "#f0f0f0",
    "loading.title": "bold #f0f0f0",
    "loading.detail": "#8a8a8a",
    "workspace.header": "bold #f0f0f0",
    "workspace.mode": "bold #35d06f",
    "workspace.user": "bold #35d06f",
    "workspace.assistant": "#e8e8e8",
    "workspace.muted": "#a0a0a0",
    "workspace.stage.done": "#35d06f",
    "workspace.stage.active": "bold #f0f0f0",
    "workspace.stage.pending": "#555555",
    "workspace.attention": "bold #f0b429",
    "workspace.option": "#b8b8b8",
    "workspace.option.selected": "bold #0b0b0b bg:#35d06f",
    "workspace.error": "bold #ff6b6b",
    "workspace.command.title": "bold #BE9765",
    "workspace.command": "#d0d0d0",
    "workspace.command.selected": "bold #ffffff bg:#71533D",
    "workspace.project": "bold #BE9765",
    "workspace.phase": "bold #937152",
    "workspace.resource": "#808080",
    "workspace.agent": "#c084fc",
    "workspace.actor.architect": "bold #E0A63A",
    "workspace.actor.reviewer": "bold #c084fc",
    "workspace.actor.implementer": "bold #2dd4bf",
    "workspace.actor.tool": "bold #BE9765",
    "workspace.actor.test": "bold #67e8f9",
    "workspace.ultra": "bold #D6AD3A",
    "workspace.success": "#35d06f",
    "workspace.warning": "#f0b429",
}
_NO_COLOR_STYLE = {
    key: ("reverse bold" if key == "choice.selected" else "bold" if key in {"welcome.brand", "header.title", "details.title", "warning"} else "")
    for key in _COLOR_STYLE
}


def _make_style(no_color: bool) -> Any:
    return Style.from_dict(_NO_COLOR_STYLE if no_color else _COLOR_STYLE) if Style is not None else None


def _prompt_output(output: TextIO, app_output: Any | None) -> Any | None:
    if app_output is not None:
        return app_output
    if output is sys.stdout or create_output is None:
        return None
    try:
        return create_output(stdout=output)
    except Exception:
        # Windows prompt_toolkit can raise NoConsoleScreenBufferError for a
        # TTY-like wrapper that is not the real console screen buffer.
        return _UNUSABLE_OUTPUT


def _workspace_copy(locale: str, key: str) -> str:
    values = {
        "en": {
            "understanding": "Understanding",
            "planning": "Planning",
            "building": "Building",
            "checking": "Checking",
            "done": "Done",
            "ready": "Ready",
            "simple": "Simple",
            "advanced": "Advanced",
            "write": "Write a message",
            "guide": "Send guidance while work continues",
            "details": "Details",
            "you": "You",
            "queued": "queued",
            "stop_hint": "Ctrl+C Stop safely",
            "clear_hint": "Ctrl+C Clear draft",
            "model": "Model",
            "permissions": "Permissions",
            "actions": "Actions",
            "custom": "Type a custom answer, or choose above",
        },
        "ar": {
            "understanding": "فهم الطلب",
            "planning": "التخطيط",
            "building": "التنفيذ",
            "checking": "التحقق",
            "done": "تم",
            "ready": "جاهز",
            "simple": "بسيط",
            "advanced": "متقدم",
            "write": "اكتب رسالة",
            "guide": "أرسل توجيهًا أثناء استمرار العمل",
            "details": "التفاصيل",
            "you": "أنت",
            "queued": "في الانتظار",
            "stop_hint": "Ctrl+C إيقاف آمن",
            "clear_hint": "Ctrl+C مسح المسودة",
            "model": "النموذج",
            "permissions": "الصلاحيات",
            "actions": "الإجراءات",
            "custom": "اكتب إجابة مخصصة أو اختر من الأعلى",
        },
    }
    return values.get(locale, values["en"]).get(key, values["en"].get(key, key))


def _workspace_stage_progress(
    stage: ActivityStage,
    *,
    completed: int = 0,
    total: int = 0,
    locale: str = "en",
) -> str:
    if stage in {ActivityStage.IDLE, ActivityStage.PAUSED, ActivityStage.PROBLEM} or total <= 0:
        return ""
    label = "خطوات الخطة" if locale == "ar" else "Plan steps"
    return f"{label}: {min(completed, total)}/{total} complete (not a time estimate)"


def _compact_duration(seconds: int | float | None) -> str:
    value = max(0, int(seconds or 0))
    if value < 60:
        return f"{value}s"
    minutes, remaining = divmod(value, 60)
    if minutes < 60:
        return f"{minutes}m {remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _progress_lines(snapshot: WorkspaceSnapshot, width: int) -> list[str]:
    progress = snapshot.progress
    activity = snapshot.activity
    elapsed = progress.elapsed_seconds or activity.elapsed_seconds()
    phase_value = progress.phase
    if not phase_value or (phase_value == "idle" and activity.stage is not ActivityStage.IDLE):
        phase_value = activity.stage.value
    phase = phase_value.replace("_", " ").upper()
    current = progress.current_task or activity.summary or "Waiting for the next step"
    if activity.stage is ActivityStage.PAUSED:
        estimate = "ETA paused"
    elif progress.eta_low_seconds is None or progress.eta_high_seconds is None:
        estimate = "ETA learning"
    else:
        estimate = f"ETA approx {_compact_duration(progress.eta_low_seconds)}–{_compact_duration(progress.eta_high_seconds)}"
    if width < 80:
        count = (
            f"{progress.completed}/{progress.total} done · {progress.remaining} left"
            if progress.total > 0
            else "progress total unknown"
        )
        operation = progress.active_operation or current
        issue = progress.blocker or progress.retry_reason
        final = f"elapsed {_compact_duration(elapsed)} · {estimate}"
        if progress.total_low_seconds is not None and progress.total_high_seconds is not None:
            final += (
                f" · total approx {_compact_duration(progress.total_low_seconds)}–"
                f"{_compact_duration(progress.total_high_seconds)}"
            )
        if issue:
            final += f" · {'blocked' if progress.blocker else 'retry'} {issue}"
        return [
            _fit(f"{phase} · {current}", width).rstrip(),
            _fit(f"{count} · now {operation}", width).rstrip(),
            _fit(final, width).rstrip(),
        ]

    lines = [_fit(f"{phase}  {current}  · elapsed {_compact_duration(elapsed)}", width).rstrip()]
    if progress.total > 0:
        lines.append(_fit(f"Project {progress.completed}/{progress.total} complete · {progress.remaining} remaining", width).rstrip())
    if progress.active_operation:
        lines.append(_fit(f"Now  {progress.active_operation}", width).rstrip())
    if progress.blocker:
        lines.append(_fit(f"Blocked  {progress.blocker}", width).rstrip())
    elif progress.retry_count or progress.retry_reason:
        issue = f"Retry {progress.retry_count or 1}"
        if progress.retry_reason:
            issue += f" · {progress.retry_reason}"
        lines.append(_fit(issue, width).rstrip())
    if estimate == "ETA learning":
        estimate = "ETA learning from completed work"
    elif estimate.startswith("ETA approx"):
        estimate += " remaining"
        if progress.total_low_seconds is not None and progress.total_high_seconds is not None:
            estimate += f" · total approx {_compact_duration(progress.total_low_seconds)}–{_compact_duration(progress.total_high_seconds)}"
    lines.append(_fit(estimate, width).rstrip())
    if activity.last_success:
        lines.append(_fit(f"✓ {activity.last_success}", width).rstrip())
    return lines


def _compact_progress_lines(snapshot: WorkspaceSnapshot, width: int) -> list[str]:
    """Return the calm default activity strip; detailed facts live in /status."""

    progress = snapshot.progress
    activity = snapshot.activity
    elapsed = progress.elapsed_seconds or activity.elapsed_seconds()
    raw_phase = progress.phase or activity.stage.value
    normalized = raw_phase.casefold().replace("_", " ").strip()
    phase = {
        "architecture": "Planning architecture",
        "discovering": "Understanding the project",
        "reviewing": "Reviewing the result",
        "verifying": "Verifying the result",
        "running": "Building",
    }.get(normalized, normalized.title() or "Working")
    operation = progress.active_operation or progress.current_task or activity.summary
    lines = [_fit(f"{phase}  ·  {operation or 'Waiting for the next step'}", width).rstrip()]
    facts: list[str] = []
    if progress.total > 0:
        percent = min(100, max(0, int(progress.completed * 100 / progress.total)))
        facts.append(
            f"Step {progress.completed}/{progress.total} · {percent}% · {progress.remaining} tasks remaining"
        )
    elif snapshot.running:
        facts.append("Planning · total not known yet")
    changes = snapshot.changes
    if changes.files:
        facts.append(
            f"{changes.files} file{'s' if changes.files != 1 else ''} · "
            f"+{changes.additions} -{changes.deletions}"
        )
    facts.append(f"elapsed {_compact_duration(elapsed)}")
    if (
        activity.stage is not ActivityStage.PAUSED
        and progress.eta_low_seconds is not None
        and progress.eta_high_seconds is not None
    ):
        facts.append(
            f"ETA ~{_compact_duration(progress.eta_low_seconds)}–"
            f"{_compact_duration(progress.eta_high_seconds)}"
        )
    lines.append(_fit(" · ".join(facts), width).rstrip())
    if progress.blocker:
        lines.append(_fit(f"Blocked · {progress.blocker}", width).rstrip())
    elif progress.retry_reason:
        lines.append(
            _fit(f"Retry {progress.retry_count or 1} · {progress.retry_reason}", width).rstrip()
        )
    return lines[:3]


def _telemetry_text(snapshot: WorkspaceSnapshot, width: int) -> str:
    resource = snapshot.resources
    if resource.execution_class == "cloud":
        used = resource.context_used_tokens
        remaining = resource.context_remaining_tokens
        context = (
            f"context {used / 1000:.1f}k · remaining {(remaining or 0) / 1000:.1f}k"
            if resource.context_window_tokens
            else f"context {used / 1000:.1f}k · remaining unavailable"
        )
        limits = resource.provider_limits or "provider limits unavailable"
        cloud_parts = [
            f"cloud {snapshot.model}",
            resource.model_activity,
            context,
            f"out {resource.output_tokens / 1000:.1f}k",
        ]
        if resource.cached_tokens:
            cloud_parts.append(f"cached {resource.cached_tokens / 1000:.1f}k")
        cloud_parts.extend((limits, "Sleep ON" if snapshot.sleep_enabled else "Sleep off"))
        return _fit(" · ".join(cloud_parts), width).rstrip()
    if width < 80:
        parts: list[str] = []
        if resource.cpu_percent is not None:
            parts.append(f"CPU {resource.cpu_percent:.0f}%")
        if resource.memory_percent is not None:
            parts.append(f"RAM {resource.memory_percent:.0f}%")
        if resource.gpu_available:
            parts.append("GPU ?" if resource.gpu_percent is None else f"GPU {resource.gpu_percent:.0f}%")
        remaining = resource.context_remaining_tokens
        if resource.context_window_tokens:
            parts.append(f"ctx {resource.context_used_tokens / 1000:.1f}k left {(remaining or 0) / 1000:.1f}k")
        else:
            parts.append(f"ctx {resource.context_used_tokens / 1000:.1f}k left ?")
        activity = {
            "processing result": "result",
            "calling model": "call",
            "using tool": "tool",
        }.get(resource.model_activity, resource.model_activity)
        parts.append(f"mdl {activity[:8]}")
        parts.append("Sleep ON" if snapshot.sleep_enabled else "Sleep off")
        return _fit("  ".join(parts), width).rstrip()

    parts: list[str] = []
    if resource.cpu_percent is not None:
        parts.append(f"CPU {resource.cpu_percent:.0f}%")
    if resource.memory_percent is not None:
        if width >= 100 and resource.memory_used_gib is not None and resource.memory_total_gib is not None:
            parts.append(f"RAM {resource.memory_used_gib:.1f}/{resource.memory_total_gib:.1f}G ({resource.memory_percent:.0f}%)")
        else:
            parts.append(f"RAM {resource.memory_percent:.0f}%")
    if resource.process_memory_mib is not None and width >= 120:
        parts.append(f"proc {resource.process_memory_mib:.0f}MB")
    if resource.gpu_available:
        gpu = "GPU ?" if resource.gpu_percent is None else f"GPU {resource.gpu_percent:.0f}%"
        if width >= 110 and resource.gpu_memory_used_mib is not None and resource.gpu_memory_total_mib:
            gpu += f" {resource.gpu_memory_used_mib / 1024:.1f}/{resource.gpu_memory_total_mib / 1024:.1f}G"
        parts.append(gpu)
    used = resource.context_used_tokens
    remaining = resource.context_remaining_tokens
    if resource.context_window_tokens:
        parts.append(f"ctx {used / 1000:.1f}k · left {(remaining or 0) / 1000:.1f}k")
    else:
        parts.append(f"ctx {used / 1000:.1f}k · left ?")
    parts.append(f"model {resource.model_activity}")
    parts.append("Sleep ON" if snapshot.sleep_enabled else "Sleep off")
    return _fit(" · ".join(parts), width).rstrip()


def render_persistent_workspace(
    snapshot: WorkspaceSnapshot,
    *,
    width: int = 100,
    height: int = 30,
    now: float | None = None,
) -> str:
    """Render a deterministic plain-text snapshot of the persistent workspace."""

    width, height = max(44, int(width)), max(16, int(height))
    mode = _workspace_copy(snapshot.locale, snapshot.mode.value)
    header_left = f"GA3BAD  {mode.upper()}"
    header_right = " · ".join(item for item in (snapshot.model, snapshot.status, snapshot.workspace) if item)
    header = _fit(
        header_left + " " * max(1, width - len(header_left) - len(header_right)) + header_right,
        width,
    ).rstrip()
    lines = [header, "─" * width if width > 50 else "-" * width]
    entries = snapshot.transcript[-8:]
    if not entries:
        lines.extend(("", _workspace_copy(snapshot.locale, "ready"), ""))
    for entry in entries:
        label = "You" if entry.role == "user" else "GA3BAD"
        lines.append(f"{label}  {textwrap.shorten(entry.text, width=max(12, width - len(label) - 2), placeholder='…')}")
        lines.append("")
    activity = snapshot.activity
    if snapshot.running or activity.stage is not ActivityStage.IDLE:
        lines.extend(_compact_progress_lines(snapshot, width))
        if snapshot.sleep_log:
            lines.append(_fit(f"Sleep  {snapshot.sleep_log[-1]}", width).rstrip())
    elif snapshot.sleep_log:
        lines.append(_fit(f"Sleep  {snapshot.sleep_log[-1]}", width).rstrip())
    if snapshot.attention is not None:
        lines.extend(("", f"! {snapshot.attention.title}"))
        if snapshot.attention.message:
            lines.extend(textwrap.wrap(snapshot.attention.message, width=width)[:3])
        options = []
        for index, option in enumerate(snapshot.attention.options):
            marker = "[" if index == snapshot.attention_index else " "
            closer = "]" if index == snapshot.attention_index else " "
            description = f" — {option.description}" if option.description else ""
            flags = []
            if option.recommended:
                flags.append("Recommended")
            if option.key == snapshot.attention.default_key:
                flags.append("Enter default")
            suffix = f" [{' · '.join(flags)}]" if flags else ""
            options.append(f"{marker}{option.label}{closer}{suffix}{description}")
        if options:
            lines.extend(options)
        if snapshot.attention_feedback:
            lines.append(f"! {snapshot.attention_feedback}")
    footer_space = 4
    lines = lines[: max(1, height - footer_space)]
    while len(lines) < height - footer_space:
        lines.append("")
    placeholder = (
        _workspace_copy(snapshot.locale, "guide")
        if snapshot.running
        else _workspace_copy(snapshot.locale, "write")
    )
    lines.extend(("› " + placeholder, "─" * width if width > 50 else "-" * width))
    lines.append(_fit(_telemetry_text(snapshot, width), width).rstrip())
    queue = f" · queued {snapshot.queued_count}" if snapshot.queued_count else ""
    interrupt = _workspace_copy(snapshot.locale, "stop_hint" if snapshot.running else "clear_hint")
    lines.append(
        _fit(
            f"F2 {_workspace_copy(snapshot.locale, 'advanced')} · "
            f"F3 {_workspace_copy(snapshot.locale, 'model')} · "
            f"F4 {_workspace_copy(snapshot.locale, 'permissions')} · F6 Sleep · "
            f"Ctrl+K Commands · {interrupt}{queue}",
            width,
        ).rstrip()
    )
    return "\n".join(lines[:height])


@dataclass(frozen=True, slots=True)
class WorkspaceInput:
    text: str = ""
    queued: bool = False
    kind: str = "message"


class PersistentWorkspaceApp:
    """One long-lived prompt_toolkit application for the complete work session."""

    def __init__(
        self,
        store: WorkspaceUIStore,
        *,
        on_input: Callable[[WorkspaceInput], None],
        on_interrupt: Callable[[], None],
        on_exit: Callable[[], bool | None],
        output: TextIO = sys.stdout,
        no_color: bool = False,
        app_input: Any | None = None,
        app_output: Any | None = None,
    ) -> None:
        if not PROMPT_TOOLKIT_AVAILABLE:
            raise RuntimeError("prompt_toolkit is required for the persistent workspace")
        self.store = store
        self.on_input = on_input
        self.on_interrupt = on_interrupt
        self.on_exit = on_exit
        self.output = output
        self.no_color = no_color
        self._animate = not no_color and not _env_enabled("GA3BAD_REDUCED_MOTION")
        self._application: Any | None = None
        self._last_mode = store.snapshot().mode
        self._palette_open = False
        self._palette_index = 0
        self._palette_matches: tuple[CommandSpec, ...] = ()
        self._overlay_kind = ""
        self._overlay_title = ""
        self._overlay_text = ""
        self._editor_original = ""
        self._editor_warning = ""
        self._swarm_snapshot: Mapping[str, Any] = {}
        self._swarm_state = SwarmInspectorState()
        self._buffer = Buffer(multiline=False, on_text_changed=self._on_buffer_changed)
        self._editor_buffer = Buffer(multiline=True)
        self._bindings = KeyBindings()
        self._install_bindings()

        transcript = Window(
            content=FormattedTextControl(self._transcript_fragments),
            wrap_lines=True,
            always_hide_cursor=True,
            height=Dimension(weight=1),
        )
        self._transcript_window = transcript
        self._follow_transcript = True
        activity = Window(
            content=FormattedTextControl(self._activity_fragments),
            wrap_lines=True,
            always_hide_cursor=True,
            height=Dimension(min=1, max=3, preferred=2),
        )
        telemetry = Window(
            content=FormattedTextControl(self._telemetry_fragments),
            wrap_lines=False,
            always_hide_cursor=True,
            height=1,
        )
        attention = Window(
            content=FormattedTextControl(self._attention_fragments),
            wrap_lines=True,
            always_hide_cursor=True,
            height=Dimension(min=1, max=12, preferred=6),
        )
        composer = Window(
            content=BufferControl(buffer=self._buffer, focusable=True),
            height=Dimension(min=1, max=6, preferred=2),
            wrap_lines=True,
        )
        self._editor_control = BufferControl(buffer=self._editor_buffer, focusable=True)
        self._editor_root = HSplit(
            [
                Window(content=FormattedTextControl(self._editor_header_fragments), height=2),
                Window(content=self._editor_control, wrap_lines=True),
                Window(content=FormattedTextControl(self._editor_footer_fragments), height=2),
            ]
        )
        self._main_root = HSplit(
            [
                Window(content=FormattedTextControl(self._header_fragments), height=1),
                Window(height=1, char="─" if terminal_supports_unicode(output) else "-"),
                transcript,
                activity,
                attention,
                DynamicContainer(self._palette_container),
                Window(content=FormattedTextControl(self._composer_prompt_fragments), height=1),
                composer,
                Window(height=1, char="─" if terminal_supports_unicode(output) else "-"),
                Window(content=FormattedTextControl(self._telemetry_fragments), height=1),
                Window(content=FormattedTextControl(self._footer_fragments), height=1),
            ]
        )
        self._overlay_window = Window(
            content=FormattedTextControl(self._overlay_fragments),
            wrap_lines=True,
            always_hide_cursor=True,
        )
        root = DynamicContainer(self._root_container)
        kwargs: dict[str, Any] = {}
        prompt_output = _prompt_output(output, app_output)
        if prompt_output is _UNUSABLE_OUTPUT:
            raise RuntimeError("the terminal output cannot host the persistent workspace")
        if app_input is not None:
            kwargs["input"] = app_input
        if prompt_output is not None:
            kwargs["output"] = prompt_output
        self._application = Application(
            layout=Layout(root, focused_element=composer),
            key_bindings=self._bindings,
            style=_make_style(no_color),
            full_screen=True,
            mouse_support=True,
            refresh_interval=1.0,
            **kwargs,
        )
        store.subscribe(self.request_redraw)

    @property
    def application(self) -> Any:
        return self._application

    @property
    def overlay_kind(self) -> str:
        return self._overlay_kind

    def _root_container(self) -> Any:
        if self._overlay_kind == "plan_edit":
            return self._editor_root
        return self._overlay_window if self._overlay_kind else self._main_root

    def _editor_header_fragments(self) -> Any:
        warning = f"\n{self._editor_warning}" if self._editor_warning else ""
        return FormattedText(
            [("class:details.title", f" Edit plan · structured Markdown{warning}")]
        )

    def _editor_footer_fragments(self) -> Any:
        return FormattedText(
            [("class:footer", " Ctrl+S save new revision · Esc keep/discard · Ctrl+Q safe exit")]
        )

    def _palette_container(self) -> Any:
        if not self._palette_open or self.store.active_attention() is not None:
            return Window(height=0)
        return Window(
            content=FormattedTextControl(self._palette_fragments),
            height=Dimension(
                min=2,
                max=11,
                preferred=min(10, len(self._palette_matches) + 1),
            ),
            wrap_lines=False,
            always_hide_cursor=True,
        )

    def _on_buffer_changed(self, _buffer: Any) -> None:
        value = self._buffer.text
        if value.startswith("/") and "\n" not in value:
            query = value.split(maxsplit=1)[0]
            self._palette_matches = matching_commands(query, snapshot=self.store.snapshot())
            self._palette_open = bool(self._palette_matches)
            self._palette_index = max(
                0,
                min(self._palette_index, max(0, len(self._palette_matches) - 1)),
            )
        else:
            self._palette_open = False
            self._palette_matches = ()
            self._palette_index = 0
        self.request_redraw()

    def _open_palette(self) -> None:
        if not self._buffer.text.startswith("/"):
            self._buffer.text = "/"
            self._buffer.cursor_position = 1
        self._palette_matches = matching_commands(
            self._buffer.text.split(maxsplit=1)[0], snapshot=self.store.snapshot()
        )
        self._palette_open = True
        self._palette_index = 0

    def _complete_palette(self, *, execute: bool) -> bool:
        if not self._palette_open or not self._palette_matches:
            return False
        spec = self._palette_matches[self._palette_index]
        current = self._buffer.text.strip()
        parts = current.split(maxsplit=1)
        has_arguments = len(parts) > 1 and bool(parts[1].strip())
        requires_arguments = bool(spec.arguments and not spec.arguments.startswith("["))
        if execute and (not requires_arguments or has_arguments):
            value = current if has_arguments else spec.name
            self._buffer.reset()
            self._palette_open = False
            self.on_input(WorkspaceInput(text=value))
            return True
        self._buffer.text = spec.name + (" " if spec.arguments else "")
        self._buffer.cursor_position = len(self._buffer.text)
        self._palette_open = bool(spec.arguments)
        return not execute or not spec.arguments

    def open_swarm(
        self,
        snapshot: Mapping[str, Any],
        *,
        tab: str = "agents",
        target: str | None = None,
    ) -> None:
        self._swarm_snapshot = dict(snapshot)
        self._swarm_state = SwarmInspectorState(
            tab=tab if tab in {"agents", "tree"} else "agents"
        )
        if target:
            normalized = str(target).strip().casefold()
            nodes = list(self._swarm_snapshot.get("nodes") or ())
            for index, node in enumerate(nodes):
                node_id = str(_swarm_field(node, "id", "")).casefold()
                title = str(_swarm_field(node, "title", "")).casefold()
                if (
                    normalized in {str(index + 1), node_id, title}
                    or node_id.startswith(normalized)
                ):
                    self._swarm_state.selected_index = index
                    break
        self._overlay_kind = "swarm"
        self._overlay_title = "Specialists"
        self.request_redraw()

    def update_swarm(self, snapshot: Mapping[str, Any]) -> None:
        self._swarm_snapshot = dict(snapshot)
        self._swarm_state.clamp(self._swarm_snapshot)
        self.request_redraw()

    def open_details(self, title: str, value: str, *, kind: str = "details") -> None:
        self._overlay_kind = kind
        self._overlay_title = str(title)
        self._overlay_text = str(value or "(no details recorded)")
        self.request_redraw()

    def open_plan_editor(self, title: str, value: str) -> None:
        self._overlay_kind = "plan_edit"
        self._overlay_title = str(title)
        self._editor_original = str(value)
        self._editor_warning = ""
        self._editor_buffer.text = str(value)
        self._editor_buffer.cursor_position = 0
        try:
            if self._application is not None:
                self._application.layout.focus(self._editor_control)
        except (AttributeError, ValueError):
            pass
        self.request_redraw()

    def close_overlay(self) -> None:
        self._overlay_kind = ""
        self._overlay_title = ""
        self._overlay_text = ""
        self._editor_warning = ""
        try:
            if self._application is not None:
                self._application.layout.focus(self._buffer)
        except (AttributeError, ValueError):
            pass
        self.request_redraw()

    def _install_bindings(self) -> None:
        @self._bindings.add("c-e", eager=True)
        def _edit_plan(event: Any) -> None:
            if self._overlay_kind == "plan":
                self.open_plan_editor(self._overlay_title or "Plan", self._overlay_text)
                event.app.invalidate()

        @self._bindings.add("c-s", eager=True)
        def _save_plan(event: Any) -> None:
            if self._overlay_kind != "plan_edit":
                return
            try:
                parse_plan_document(self._editor_buffer.text)
            except PlanDocumentError as exc:
                self._editor_warning = str(exc)
                event.app.invalidate()
                return
            value = self._editor_buffer.text
            self._editor_original = value
            self._editor_warning = "Saving as a new revision…"
            self.on_input(WorkspaceInput(text=value, kind="plan_save"))
            event.app.invalidate()

        @self._bindings.add("f2", eager=True)
        def _toggle_mode(event: Any) -> None:
            self.store.toggle_mode()
            event.app.invalidate()

        @self._bindings.add("f3", eager=True)
        def _choose_model(event: Any) -> None:
            self.on_input(WorkspaceInput(kind="model"))

        @self._bindings.add("f4", eager=True)
        def _choose_permissions(event: Any) -> None:
            self.on_input(WorkspaceInput(kind="permissions"))

        @self._bindings.add("f6", eager=True)
        def _toggle_sleep(event: Any) -> None:
            self.store.toggle_sleep_mode()
            event.app.invalidate()

        @self._bindings.add("f7", eager=True)
        def _open_diff(event: Any) -> None:
            self.on_input(WorkspaceInput(text="/diff"))

        @self._bindings.add("f8", eager=True)
        def _open_folder(event: Any) -> None:
            self.on_input(WorkspaceInput(text="/explorer"))

        @self._bindings.add("c-k", eager=True)
        def _open_actions(event: Any) -> None:
            if self.store.active_attention() is None and not self._overlay_kind:
                self._open_palette()
                event.app.invalidate()

        @self._bindings.add("pageup", eager=True)
        def _transcript_up(event: Any) -> None:
            if self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_up(count=8)
                event.app.invalidate()
                return
            if self._overlay_kind == "swarm":
                self._swarm_state.move(self._swarm_snapshot, -8)
                event.app.invalidate()
                return
            if self._overlay_kind:
                self._overlay_window.vertical_scroll = max(
                    0, int(self._overlay_window.vertical_scroll) - 8
                )
                event.app.invalidate()
                return
            self._follow_transcript = False
            self._transcript_window.vertical_scroll = max(
                0, int(self._transcript_window.vertical_scroll) - 8
            )
            event.app.invalidate()

        @self._bindings.add("pagedown", eager=True)
        def _transcript_down(event: Any) -> None:
            if self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_down(count=8)
                event.app.invalidate()
                return
            if self._overlay_kind == "swarm":
                self._swarm_state.move(self._swarm_snapshot, 8)
                event.app.invalidate()
                return
            if self._overlay_kind:
                self._overlay_window.vertical_scroll += 8
                event.app.invalidate()
                return
            self._transcript_window.vertical_scroll += 8
            snapshot = self.store.snapshot()
            width = self._current_width()
            estimated_lines = sum(
                max(2, (len(entry.text) // max(20, width - 4)) + entry.text.count("\n") + 2)
                for entry in snapshot.transcript
            )
            try:
                visible_lines = max(3, get_app().output.get_size().rows - 17)
            except (AttributeError, RuntimeError, ValueError):
                visible_lines = 8
            if self._transcript_window.vertical_scroll + visible_lines >= estimated_lines:
                self._follow_transcript = True
                self._transcript_window.vertical_scroll = 10**9
            event.app.invalidate()

        @self._bindings.add("end", eager=True)
        def _transcript_end(event: Any) -> None:
            if self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_position = len(self._editor_buffer.text)
                event.app.invalidate()
                return
            if self.store.active_attention() is not None:
                snapshot = self.store.snapshot()
                if snapshot.attention is not None and snapshot.attention.options:
                    self.store.select_attention_index(len(snapshot.attention.options) - 1)
            else:
                self._follow_transcript = True
                self._transcript_window.vertical_scroll = 10**9
            event.app.invalidate()

        @self._bindings.add("home", eager=True)
        def _attention_home(event: Any) -> None:
            if self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_position = 0
                event.app.invalidate()
                return
            snapshot = self.store.snapshot()
            if snapshot.attention is not None and snapshot.attention.options:
                self.store.select_attention_index(0)
            else:
                self._buffer.cursor_position = 0
            event.app.invalidate()

        @self._bindings.add("up", eager=True)
        def _attention_up(event: Any) -> None:
            if self.store.active_attention() is not None:
                self.store.move_attention(-1)
                event.app.invalidate()
            elif self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_up(count=1)
                event.app.invalidate()
            elif self._overlay_kind == "swarm":
                self._swarm_state.move(self._swarm_snapshot, -1)
                event.app.invalidate()
            elif self._palette_open and self._palette_matches:
                self._palette_index = (self._palette_index - 1) % len(self._palette_matches)
                event.app.invalidate()

        @self._bindings.add("down", eager=True)
        def _attention_down(event: Any) -> None:
            if self.store.active_attention() is not None:
                self.store.move_attention(1)
                event.app.invalidate()
            elif self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_down(count=1)
                event.app.invalidate()
            elif self._overlay_kind == "swarm":
                self._swarm_state.move(self._swarm_snapshot, 1)
                event.app.invalidate()
            elif self._palette_open and self._palette_matches:
                self._palette_index = (self._palette_index + 1) % len(self._palette_matches)
                event.app.invalidate()

        @self._bindings.add("left", eager=True)
        def _attention_left(event: Any) -> None:
            if self.store.active_attention() is not None:
                self.store.move_attention(-1)
                event.app.invalidate()
            elif self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_left(count=1)
            elif self._overlay_kind == "swarm":
                self._swarm_state.select_tab("agents")
                event.app.invalidate()
            else:
                self._buffer.cursor_left(count=1)

        @self._bindings.add("right", eager=True)
        def _attention_right(event: Any) -> None:
            if self.store.active_attention() is not None:
                self.store.move_attention(1)
                event.app.invalidate()
            elif self._overlay_kind == "plan_edit":
                self._editor_buffer.cursor_right(count=1)
            elif self._overlay_kind == "swarm":
                self._swarm_state.select_tab("tree")
                event.app.invalidate()
            else:
                self._buffer.cursor_right(count=1)

        @self._bindings.add("enter", eager=True)
        def _submit(event: Any) -> None:
            if self._overlay_kind == "plan_edit":
                self._editor_buffer.insert_text("\n")
                return
            value = self._buffer.text.strip()
            attention = self.store.active_attention()
            if attention is not None:
                if value and attention.allow_custom:
                    self._buffer.reset()
                    self.store.resolve_attention("custom", text=value)
                elif not value:
                    self.store.resolve_selected_attention()
                else:
                    self.store.set_attention_feedback(
                        "This decision does not accept free text. Use a shown shortcut or clear the input."
                    )
                return
            if self._overlay_kind == "swarm":
                self._swarm_state.prompt_expanded = not self._swarm_state.prompt_expanded
                event.app.invalidate()
                return
            if self._overlay_kind == "chat":
                if value:
                    self.close_overlay()
                    self._buffer.reset()
                    self.on_input(WorkspaceInput(text=value))
                return
            if self._overlay_kind:
                return
            if self._palette_open and self._complete_palette(execute=True):
                event.app.invalidate()
                return
            if value:
                self._buffer.reset()
                self.on_input(WorkspaceInput(text=value))

        @self._bindings.add("tab", eager=True)
        def _queue(event: Any) -> None:
            if self._overlay_kind == "plan_edit":
                self._editor_buffer.insert_text("    ")
                return
            if self._overlay_kind == "swarm":
                self._swarm_state.select_tab(
                    "tree" if self._swarm_state.tab == "agents" else "agents"
                )
                event.app.invalidate()
                return
            if self._palette_open:
                self._complete_palette(execute=False)
                event.app.invalidate()
                return
            snapshot = self.store.snapshot()
            value = self._buffer.text.strip()
            if snapshot.mode is ExperienceMode.ADVANCED and snapshot.running and value:
                self._buffer.reset()
                self.on_input(WorkspaceInput(text=value, queued=True))
                return
            if value:
                self._buffer.insert_text("\t")

        @self._bindings.add("c-c", eager=True)
        def _interrupt(event: Any) -> None:
            if self.store.active_attention() is not None:
                self.store.cancel_attention()
            elif self._overlay_kind:
                self.close_overlay()
            elif self._palette_open:
                self._palette_open = False
                event.app.invalidate()
            elif self.store.snapshot().running:
                self.on_interrupt()
            elif self._buffer.text:
                self._buffer.reset()

        @self._bindings.add("escape", eager=True)
        def _back_or_interrupt(event: Any) -> None:
            if self.store.active_attention() is not None:
                self._buffer.reset()
                self.store.cancel_attention()
            elif self._overlay_kind == "plan_edit":
                if self._editor_buffer.text != self._editor_original and not self._editor_warning.startswith("Unsaved"):
                    self._editor_warning = "Unsaved edits · press Esc again to discard, or Ctrl+S to save"
                    event.app.invalidate()
                else:
                    self.close_overlay()
            elif self._overlay_kind:
                self.close_overlay()
            elif self._palette_open:
                self._palette_open = False
                event.app.invalidate()
            elif self.store.snapshot().running:
                self.on_interrupt()
            elif self._buffer.text:
                self._buffer.reset()

        @self._bindings.add("c-q", eager=True)
        def _exit(event: Any) -> None:
            if self.on_exit() is not False:
                event.app.exit(result=None)

        for key in tuple("123456789") + tuple("abcdefghijklmnopqrstuvwxyz"):
            def _shortcut(event: Any, pressed: str = key) -> None:
                request = self.store.active_attention()
                if request is None:
                    if self._overlay_kind == "plan_edit":
                        self._editor_buffer.insert_text(pressed)
                        return
                    if self._overlay_kind == "chat":
                        self.close_overlay()
                        self._buffer.insert_text(pressed)
                        return
                    if self._overlay_kind == "swarm":
                        if pressed == "r":
                            self.on_input(WorkspaceInput(kind="overlay_refresh"))
                        return
                    if self._overlay_kind:
                        return
                    self._buffer.insert_text(pressed)
                    return
                if self._buffer.text or (request.allow_custom and not pressed.isdigit()):
                    self._buffer.insert_text(pressed)
                    return
                match = next(
                    (
                        option
                        for index, option in enumerate(request.options, 1)
                        if option.shortcut.casefold() == pressed.casefold()
                        or str(index) == pressed
                    ),
                    None,
                )
                if match is not None:
                    self.store.resolve_attention(match.key)
                elif not request.allow_custom:
                    self.store.set_attention_feedback(
                        f"{pressed!r} is not a valid choice. Use a shown shortcut or arrow keys."
                    )

            self._bindings.add(key, eager=True)(_shortcut)

    def _palette_fragments(self) -> Any:
        fragments: list[tuple[str, str]] = [
            ("class:workspace.command.title", " Commands  ↑↓ select · Tab complete · Enter run · Esc close\n")
        ]
        for index, spec in enumerate(self._palette_matches):
            selected = index == self._palette_index
            style = (
                "class:workspace.command.selected"
                if selected
                else "class:workspace.command"
            )
            suffix = f" {spec.arguments}" if spec.arguments else ""
            marker = "›" if selected and terminal_supports_unicode(self.output) else ">" if selected else " "
            fragments.append((style, f" {marker} {spec.name}{suffix}"))
            fragments.append(("class:workspace.muted", f"  {spec.description}\n"))
        if not self._palette_matches:
            fragments.append(("class:workspace.muted", "  No matching command\n"))
        return FormattedText(fragments)

    def _overlay_fragments(self) -> Any:
        width = self._current_width()
        try:
            height = max(16, int(get_app().output.get_size().rows) - 1)
        except (AttributeError, RuntimeError, ValueError):
            height = 32
        if self._overlay_kind == "swarm":
            body = render_swarm_inspector(
                self._swarm_snapshot,
                self._swarm_state,
                width=max(60, width - 1),
                height=height,
                unicode=terminal_supports_unicode(self.output),
            )
            return FormattedText([("class:details.body", body)])
        if self._overlay_kind == "diff":
            fragments: list[tuple[str, str]] = [
                ("class:details.title", (self._overlay_title or "Project changes") + "\n"),
                ("class:workspace.muted", ("─" if terminal_supports_unicode(self.output) else "-") * max(8, width - 1) + "\n"),
            ]
            for line in self._overlay_text.splitlines() or [""]:
                style = (
                    "class:workspace.success"
                    if line.startswith("+") and not line.startswith("+++")
                    else "class:workspace.error"
                    if line.startswith("-") and not line.startswith("---")
                    else "class:workspace.actor.tool"
                    if line.startswith(("diff --git", "@@", "+++", "---"))
                    else "class:details.body"
                )
                fragments.append((style, line + "\n"))
            fragments.append(("class:footer", "Esc close · F7 refresh · Ctrl+Q exit safely"))
            return FormattedText(fragments)
        title = self._overlay_title or "Details"
        rule = "─" if terminal_supports_unicode(self.output) else "-"
        lines = [title, rule * max(8, width - 1)]
        for paragraph in self._overlay_text.splitlines() or [""]:
            lines.extend(textwrap.wrap(paragraph, width=max(20, width - 3)) or [""])
        controls = (
            "Ctrl+E edit · Esc close · Ctrl+Q exit safely"
            if self._overlay_kind == "plan"
            else "Esc close · Ctrl+Q exit safely"
        )
        lines.extend((rule * max(8, width - 1), controls))
        return FormattedText(
            [
                ("class:details.title", lines[0] + "\n"),
                ("class:details.body", "\n".join(lines[1:])),
            ]
        )

    def _header_fragments(self) -> Any:
        snapshot = self.store.snapshot()
        mode = _workspace_copy(snapshot.locale, snapshot.mode.value).upper()
        project = os.path.basename(snapshot.workspace.rstrip("/\\")) or "GA3BAD"
        phase = (
            snapshot.progress.phase.replace("_", " ").title()
            if snapshot.progress.phase
            else snapshot.status
        )
        right = " · ".join(item for item in (phase, snapshot.model) if item)
        try:
            width = max(24, get_app().output.get_size().columns)
        except (AttributeError, RuntimeError, ValueError):
            width = 100
        left = f" {project}  {mode}"
        remaining = max(0, width - len(left) - 2)
        right = textwrap.shorten(right, width=max(1, remaining), placeholder="…") if right else ""
        gap = " " * max(1, width - len(left) - len(right))
        return FormattedText([
            ("class:workspace.project", f" {project}  "),
            ("class:workspace.mode", mode),
            ("class:workspace.phase", gap + right),
        ])

    def _transcript_fragments(self) -> Any:
        snapshot = self.store.snapshot()
        fragments: list[tuple[Any, ...]] = []
        if not snapshot.transcript:
            fragments.append(("class:workspace.muted", "\n Ready — describe what you want to build.\n"))
        for entry in snapshot.transcript[-80:]:
            rendered = entry.text
            if snapshot.mode is ExperienceMode.SIMPLE and (
                len(rendered) > 2_000 or rendered.count("\n") >= 12
            ):
                first_lines = [line for line in rendered.splitlines() if line.strip()][:2]
                preview = "\n ".join(first_lines)[:500]
                rendered = (
                    f"{preview}\n … {len(entry.text):,} chars collapsed "
                    f"· /details {entry.id}"
                )
            if entry.role == "user":
                fragments.extend(
                    (("class:workspace.user", f"\n {_workspace_copy(snapshot.locale, 'you')}\n"), ("class:workspace.assistant", f" {rendered}\n"))
                )
            else:
                fragments.extend(
                    (("class:workspace.mode", "\n GA3BAD\n"), ("class:workspace.assistant", f" {rendered}\n"))
                )
        if snapshot.mode is ExperienceMode.ADVANCED and snapshot.advanced_log:
            fragments.append(("class:workspace.muted", f"\n {_workspace_copy(snapshot.locale, 'details')}\n"))
            fragments.extend(
                ("class:workspace.muted", f" · {line}\n")
                for line in snapshot.advanced_log[-10:]
                if not line.startswith("sleep.auto_choice:")
            )
        if snapshot.mode is ExperienceMode.ADVANCED and snapshot.sleep_log:
            fragments.append(("class:workspace.muted", "\n Sleep choices\n"))
            fragments.extend(
                ("class:workspace.muted", f" · {line}\n") for line in snapshot.sleep_log
            )
        return FormattedText(fragments)

    def _activity_fragments(self) -> Any:
        snapshot = self.store.snapshot()
        activity = snapshot.activity
        if not snapshot.running and activity.stage is ActivityStage.IDLE:
            lines = " Ready\n"
            return FormattedText([("class:workspace.muted", lines)])
        quiet = 0 if activity.last_signal_at is None else max(0, int(time.monotonic() - activity.last_signal_at))
        style = "class:workspace.error" if activity.stage is ActivityStage.PROBLEM else "class:workspace.stage.active"
        lines = _compact_progress_lines(snapshot, self._current_width())
        if snapshot.running:
            spinner = "·" if not self._animate else "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() * 8) % 10]
            lines[0] = f"{spinner} {lines[0]}"
        swarm = snapshot.swarm
        if swarm.running or swarm.reviewing or swarm.blocked:
            agent_line = (
                f"Agents {swarm.total} · {swarm.running} running · "
                f"{swarm.reviewing} reviewing"
            )
            if swarm.blocked:
                agent_line += f" · {swarm.blocked} blocked"
            lines.insert(min(2, len(lines)), agent_line)
        if snapshot.running and quiet >= 60:
            model_state = snapshot.resources.model_activity
            waiting_on = (
                "model call open"
                if model_state in {"calling model", "thinking", "streaming"}
                else f"worker active ({model_state})"
            )
            signal = (
                f"No runtime event for {_compact_duration(quiet)} · {waiting_on} · Esc/Ctrl+C pauses safely"
                if quiet >= 60
                else f"Still working · last runtime event {_compact_duration(quiet)} ago"
            )
            lines.insert(min(2, len(lines)), signal)
        fragments: list[tuple[str, str]] = []
        for index, line in enumerate(lines[:3]):
            line_style = style if index == 0 else (
                "class:workspace.error" if line.startswith("Blocked") else
                "class:workspace.agent" if line.startswith("Agents") else
                "class:workspace.muted"
            )
            actor_match = None
            if index == 0:
                actor_match = re.match(
                    r"^(?P<prefix>[^·:\[]+|\[[^\]]+\])(?P<sep>\s*[·:]\s*)(?P<body>.*)$",
                    line,
                )
            if actor_match:
                actor = actor_match.group("prefix")
                normalized_actor = actor.casefold()
                actor_style = (
                    "class:workspace.actor.architect"
                    if any(token in normalized_actor for token in ("architect", "planner", "planning"))
                    else "class:workspace.actor.reviewer"
                    if any(token in normalized_actor for token in ("review", "critic"))
                    else "class:workspace.actor.test"
                    if any(token in normalized_actor for token in ("test", "verify"))
                    else "class:workspace.actor.implementer"
                )
                fragments.extend(
                    (
                        (actor_style, f" {actor}"),
                        (line_style, actor_match.group("sep") + actor_match.group("body") + "\n"),
                    )
                )
            else:
                fragments.append((line_style, f" {line}\n"))
        return FormattedText(fragments)

    def _current_width(self) -> int:
        try:
            return max(24, get_app().output.get_size().columns)
        except (AttributeError, RuntimeError, ValueError):
            return 100

    def _telemetry_fragments(self) -> Any:
        snapshot = self.store.snapshot()
        resource = snapshot.resources
        hot = any(
            value is not None and value >= 90
            for value in (
                resource.cpu_percent,
                resource.memory_percent,
                resource.gpu_percent,
            )
        )
        style = "class:workspace.warning" if hot else "class:workspace.resource"
        return FormattedText(
            [(style, f" {_telemetry_text(snapshot, self._current_width())}\n")]
        )

    def _attention_fragments(self) -> Any:
        snapshot = self.store.snapshot()
        request = snapshot.attention
        if request is None:
            return FormattedText([("", "")])
        fragments: list[tuple[Any, ...]] = [
            ("class:workspace.attention", f" ! {request.title}\n"),
        ]
        if request.message:
            fragments.append(("class:workspace.assistant", f"   {request.message}\n\n"))

        def handler_for(key: str) -> Callable[[Any], None]:
            def handler(mouse_event: Any) -> None:
                if MouseEventType is not None and mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self.store.resolve_attention(key)
            return handler

        for index, option in enumerate(request.options):
            selected = index == snapshot.attention_index
            style = "class:workspace.option.selected" if selected else "class:workspace.option"
            shortcut = option.shortcut.upper() if option.shortcut else str(index + 1)
            flags: list[str] = []
            if option.recommended:
                flags.append("Recommended")
            if option.key == request.default_key:
                flags.append("Enter default")
            flag = f"  [{' · '.join(flags)}]" if flags else ""
            fragments.append((style, f"  [{shortcut}] {option.label}{flag}", handler_for(option.key)))
            if option.description:
                fragments.append(("class:workspace.muted", f" — {option.description}"))
            fragments.append(("", "\n"))
        if request.allow_custom:
            fragments.append(("class:workspace.muted", "\n   Or type your answer below."))
        if snapshot.attention_feedback:
            fragments.append(("class:workspace.error", f"\n   {snapshot.attention_feedback}"))
        elif request.cancel_key:
            fragments.append(("class:workspace.muted", "\n   Esc goes back safely."))
        if snapshot.mode is ExperienceMode.ADVANCED and request.details:
            detail = textwrap.shorten(
                " ".join(request.details.split()), width=180, placeholder="…"
            )
            fragments.append(("class:workspace.muted", f"\n   Details: {detail}"))
        return FormattedText(fragments)

    def _composer_prompt_fragments(self) -> Any:
        snapshot = self.store.snapshot()
        label = _workspace_copy(snapshot.locale, "guide" if snapshot.running else "write")
        if snapshot.attention is not None and snapshot.attention.allow_custom:
            label = _workspace_copy(snapshot.locale, "custom")
        return FormattedText(
            [("class:composer.prompt", " › "), ("class:workspace.muted", label)]
        )

    def _footer_fragments(self) -> Any:
        snapshot = self.store.snapshot()
        queue = f" · {_workspace_copy(snapshot.locale, 'queued')} {snapshot.queued_count}" if snapshot.queued_count else ""
        target_mode = ExperienceMode.SIMPLE if snapshot.mode is ExperienceMode.ADVANCED else ExperienceMode.ADVANCED
        mode_hint = f"F2 {_workspace_copy(snapshot.locale, target_mode.value)}"
        interrupt = (
            _workspace_copy(snapshot.locale, "stop_hint")
            if snapshot.running
            else _workspace_copy(snapshot.locale, "clear_hint")
        )
        return FormattedText(
            [("class:footer", f" {mode_hint} · F3 {_workspace_copy(snapshot.locale, 'model')} · F4 {_workspace_copy(snapshot.locale, 'permissions')} · F6 Sleep · F7 Diff · F8 Folder · Ctrl+K Commands · {interrupt}{queue}")]
        )

    def request_redraw(self) -> None:
        application = self._application
        if application is None:
            return
        loop = getattr(application, "loop", None)
        try:
            if self._follow_transcript:
                self._transcript_window.vertical_scroll = 10**9
            if loop is not None and getattr(application, "is_running", False):
                loop.call_soon_threadsafe(application.invalidate)
            else:
                application.invalidate()
        except (RuntimeError, AttributeError):
            return

    def stop(self) -> None:
        application = self._application
        if application is None:
            return
        loop = getattr(application, "loop", None)

        def close() -> None:
            if getattr(application, "is_running", False):
                application.exit(result=None)

        try:
            if loop is not None:
                loop.call_soon_threadsafe(close)
            else:
                close()
        except (RuntimeError, AttributeError):
            return

    def run(self) -> None:
        self._application.run()


def _welcome_fragments(
    brand: str,
    subtitle: str,
    action_label: str,
    tick: int,
    animate: bool,
    width: int,
    height: int,
) -> Any:
    lines, brand_top, brand_rows = _welcome_layout_lines(
        brand, subtitle, action_label, width, height
    )
    # A single, fast light sweep crosses the wordmark and its subtitle.
    brand_band = tick % (width + 22) - 11
    action_band = tick % (len(action_label) + 8) - 4
    action_row = max(0, len(lines) - 1)
    for index in range(len(lines) - 1, -1, -1):
        if lines[index] == action_label:
            action_row = index
            break
    fragments: list[tuple[str, str]] = []
    for row_index, line in enumerate(lines):
        centered = line.center(width)
        if brand_top <= row_index < brand_top + brand_rows:
            for column, character in enumerate(centered):
                shimmer = animate and character != " " and abs(column - brand_band) <= 3
                animated_character = (
                    str((int(character) + tick + column) % 10)
                    if animate and character.isdigit()
                    else character
                )
                fragments.append(("class:welcome.shimmer" if shimmer else "class:welcome.brand", animated_character))
        elif line == subtitle and subtitle:
            for column, character in enumerate(centered):
                shimmer = animate and character != " " and abs(column - brand_band) <= 3
                fragments.append(("class:welcome.shimmer" if shimmer else "class:welcome.brand", character))
        elif row_index == action_row and animate and action_label:
            left = max(0, (width - len(line)) // 2)
            for column, character in enumerate(centered):
                local = column - left
                shimmer = 0 <= local < len(action_label) and action_band <= local < action_band + 3
                fragments.append(("class:welcome.shimmer" if shimmer else "class:welcome.action", character))
        else:
            style = "class:welcome.subtitle" if line else "class:welcome.action"
            fragments.append((style, centered))
        if row_index < len(lines) - 1:
            fragments.append(("", "\n"))
    return FormattedText(fragments)


def run_welcome_screen(
    *,
    brand: str = "GA3BAD",
    subtitle: str = "coding agent",
    action_label: str = "Press Enter to begin",
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    no_color: bool = False,
    reduced_motion: bool = False,
    force: bool = False,
    app_input: Any | None = None,
    app_output: Any | None = None,
) -> bool:
    """Show an alternate-screen welcome; Enter accepts, Esc/Ctrl+Q cancel."""

    if not PROMPT_TOOLKIT_AVAILABLE:
        return False
    if not force and not rich_terminal_available(input_func, output):
        return False
    animate = not reduced_motion and not no_color and not _env_enabled("GA3BAD_REDUCED_MOTION")
    started = time.monotonic()
    bindings = KeyBindings()

    @bindings.add("enter")
    def _accept(event: Any) -> None:
        event.app.exit(result=True)

    @bindings.add("escape")
    @bindings.add("c-q")
    @bindings.add("c-c")
    def _cancel(event: Any) -> None:
        event.app.exit(result=False)

    def content() -> Any:
        tick = int((time.monotonic() - started) / 0.065) if animate else 0
        try:
            size = get_app().output.get_size()
            width, height = max(24, size.columns), max(8, size.rows)
        except (AttributeError, RuntimeError, ValueError):
            width, height = 100, 30
        responsive_brand = _responsive_welcome_brand(
            brand,
            width,
            height,
            unicode=terminal_supports_unicode(output),
        )
        return _welcome_fragments(
            responsive_brand,
            subtitle,
            action_label,
            tick,
            animate,
            width,
            height,
        )

    main = Window(
        content=FormattedTextControl(content),
        align="LEFT",
        wrap_lines=False,
        always_hide_cursor=True,
        height=Dimension(weight=1),
    )
    root = HSplit(
        [
            main,
            Window(
                content=FormattedTextControl(
                    FormattedText([("class:footer", "Enter Begin · Esc / Ctrl+Q Exit")])
                ),
                align="CENTER",
                height=1,
                always_hide_cursor=True,
            ),
        ]
    )
    kwargs: dict[str, Any] = {}
    prompt_output = _prompt_output(output, app_output)
    if prompt_output is _UNUSABLE_OUTPUT:
        return False
    if app_input is not None:
        kwargs["input"] = app_input
    if prompt_output is not None:
        kwargs["output"] = prompt_output
    application = Application(
        layout=Layout(root),
        key_bindings=bindings,
        style=_make_style(no_color),
        full_screen=True,
        mouse_support=False,
        refresh_interval=0.11 if animate else None,
        **kwargs,
    )
    try:
        return bool(application.run())
    except (EOFError, KeyboardInterrupt):
        return False


def run_loading_task(
    task: Callable[[], Any],
    *,
    title: str,
    detail: str = "",
    state: str = "discover",
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    no_color: bool = False,
    reduced_motion: bool = False,
    force: bool = False,
    app_input: Any | None = None,
    app_output: Any | None = None,
) -> Any | None:
    """Run a bounded startup probe behind one full-screen semantic indicator.

    This is intentionally for read-only discovery/status work. Esc returns
    ``None``; Ctrl+Q raises :class:`UserExitRequested`. The daemon worker may
    finish its bounded probe, but its result is ignored after cancellation.
    """

    if not callable(task):
        raise TypeError("loading task must be callable")
    if not PROMPT_TOOLKIT_AVAILABLE or (
        not force and not rich_terminal_available(input_func, output)
    ):
        try:
            return task()
        except (EOFError, KeyboardInterrupt):
            return None

    animate = not reduced_motion and not no_color and not _env_enabled(
        "GA3BAD_REDUCED_MOTION"
    )
    started = time.monotonic()
    cancelled = Event()
    holder: dict[str, Any] = {}
    bindings = KeyBindings()

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event: Any) -> None:
        cancelled.set()
        event.app.exit(result=False)

    @bindings.add("c-q")
    def _exit_session(event: Any) -> None:
        cancelled.set()
        event.app.exit(result=_EXIT_SELECTION)

    use_unicode = terminal_supports_unicode(output)

    def content() -> Any:
        elapsed = max(0, int(time.monotonic() - started))
        tick = int((time.monotonic() - started) / 0.12) if animate else 0
        grid = loading_grid_levels(
            state,
            tick,
            reduced_motion=reduced_motion,
            no_color=no_color,
        )
        # Keep discovery compact and top-aligned: copy first, then a small,
        # tightly packed 3x3 activity mark underneath it.
        square = "·" if use_unicode else "."
        fragments: list[tuple[str, str]] = [
            ("class:loading.title", f"  {_clean_line(title)}\n"),
        ]
        if detail:
            fragments.append(
                ("class:loading.detail", f"  {_clean_line(detail)} · {elapsed}s\n")
            )
        fragments.append(("", "\n"))
        for row_index, row in enumerate(grid):
            fragments.append(("", "  "))
            for level in row:
                fragments.append((f"class:loading.square.{level}", square))
            if row_index < len(grid) - 1:
                fragments.append(("", "\n"))
        return FormattedText(fragments)

    main = Window(
        content=FormattedTextControl(content),
        align="LEFT",
        always_hide_cursor=True,
        height=Dimension(preferred=7),
    )
    root = HSplit(
        [
            Window(height=2),
            main,
            Window(height=Dimension(weight=1)),
            Window(
                content=FormattedTextControl(
                    FormattedText([("class:footer", "Esc Back · Ctrl+Q Exit")])
                ),
                align="CENTER",
                height=1,
                always_hide_cursor=True,
            ),
        ]
    )
    kwargs: dict[str, Any] = {}
    prompt_output = _prompt_output(output, app_output)
    if prompt_output is _UNUSABLE_OUTPUT:
        return task()
    if app_input is not None:
        kwargs["input"] = app_input
    if prompt_output is not None:
        kwargs["output"] = prompt_output
    application = Application(
        layout=Layout(root),
        key_bindings=bindings,
        style=_make_style(no_color),
        full_screen=True,
        mouse_support=False,
        refresh_interval=0.12 if animate else 1.0,
        **kwargs,
    )

    def pre_run() -> None:
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                holder["value"] = task()
            except BaseException as exc:  # re-raised on the application thread
                holder["error"] = exc
            if not cancelled.is_set():
                loop.call_soon_threadsafe(lambda: application.exit(result=True))

        def start_worker() -> None:
            Thread(target=worker, name="ga3bad-startup-probe", daemon=True).start()

        # ``pre_run`` fires just before the Application marks itself running.
        # Defer one event-loop tick so even an instant probe can exit cleanly.
        loop.call_soon(start_worker)

    try:
        result = application.run(pre_run=pre_run)
    except (EOFError, KeyboardInterrupt):
        cancelled.set()
        return None
    if result is _EXIT_SELECTION:
        raise UserExitRequested()
    completed = bool(result)
    if not completed:
        return None
    if "error" in holder:
        raise holder["error"]
    return holder.get("value")


def _app_columns(default: int = 100) -> int:
    try:
        return int(get_app().output.get_size().columns)
    except (AttributeError, RuntimeError, ValueError):
        return default


def _swarm_field(item: Any, name: str, default: Any = "") -> Any:
    if item is None:
        return default
    value = item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)
    return getattr(value, "value", value)


def _swarm_node_metadata(item: Any) -> Mapping[str, Any]:
    contract = _swarm_field(item, "contract", None)
    value = _swarm_field(contract, "metadata", {}) if contract is not None else {}
    return value if isinstance(value, Mapping) else {}


def swarm_agent_name(node: Any, nodes: Iterable[Any] = ()) -> str:
    """Return a short stable persona name derived from the specialist path."""

    metadata = _swarm_node_metadata(node)
    domain = str(metadata.get("specialist_domain") or "").strip(".")
    if domain:
        parts = [part.replace("_", " ").title() for part in domain.split(".")]
        return " · ".join(parts[-2:])
    title = str(_swarm_field(node, "title", _swarm_field(node, "id", "Agent")))
    title = title.replace(" specialist", "").replace(" Specialist", "").strip()
    parent_id = str(_swarm_field(node, "parent_id", "") or "")
    if parent_id:
        parent = next(
            (item for item in nodes if str(_swarm_field(item, "id")) == parent_id),
            None,
        )
        if parent is not None:
            parent_title = str(_swarm_field(parent, "title", "")).replace(" specialist", "").strip()
            if parent_title and parent_title.casefold() not in title.casefold():
                return f"{parent_title} · {title}"
    return title or "Agent"


@dataclass
class SwarmInspectorState:
    """Pure navigation state for the read-only live specialist inspector."""

    selected_index: int = 0
    tab: str = "agents"
    prompt_expanded: bool = False

    def clamp(self, snapshot: Mapping[str, Any]) -> None:
        nodes = list(snapshot.get("nodes") or ())
        self.selected_index = max(0, min(self.selected_index, max(0, len(nodes) - 1)))

    def move(self, snapshot: Mapping[str, Any], delta: int) -> None:
        self.clamp(snapshot)
        nodes = list(snapshot.get("nodes") or ())
        if nodes:
            self.selected_index = max(0, min(len(nodes) - 1, self.selected_index + int(delta)))

    def select_tab(self, value: str) -> None:
        if value in {"agents", "tree"}:
            self.tab = value


def _swarm_status_mark(status: str, unicode: bool) -> str:
    normalized = str(status).casefold()
    if normalized in {"running", "in_progress", "planning", "reviewing", "testing"}:
        return "◉" if unicode else ">"
    if normalized in {"completed", "done"}:
        return "●" if unicode else "x"
    if normalized in {"failed", "blocked", "revision_required", "uncertain"}:
        return "!"
    if normalized in {"cancelled", "skipped"}:
        return "–" if unicode else "-"
    return "○" if unicode else "o"


def _swarm_latest_by_node(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for agent in snapshot.get("agents") or ():
        node_id = str(_swarm_field(agent, "work_node_id", "") or "")
        if node_id:
            latest[node_id] = agent
    return latest


def _swarm_node_status(node: Any, latest: Mapping[str, Any]) -> str:
    node_id = str(_swarm_field(node, "id", ""))
    agent = latest.get(node_id)
    return str(_swarm_field(agent, "status", _swarm_field(node, "status", "pending")))


def _swarm_ancestry(node: Any, nodes: Sequence[Any]) -> str:
    by_id = {str(_swarm_field(item, "id")): item for item in nodes}
    path = [swarm_agent_name(node, nodes)]
    parent_id = str(_swarm_field(node, "parent_id", "") or "")
    seen: set[str] = set()
    while parent_id and parent_id not in seen and parent_id in by_id:
        seen.add(parent_id)
        parent = by_id[parent_id]
        path.append(swarm_agent_name(parent, nodes))
        parent_id = str(_swarm_field(parent, "parent_id", "") or "")
    return " › ".join(reversed(path))


def _swarm_tree_lines(
    nodes: Sequence[Any],
    *,
    selected_id: str,
    unicode: bool,
) -> list[str]:
    by_parent: dict[str | None, list[Any]] = {}
    for node in nodes:
        parent = _swarm_field(node, "parent_id", None)
        parent_key = str(parent) if parent not in {None, ""} else None
        by_parent.setdefault(parent_key, []).append(node)
    for children in by_parent.values():
        children.sort(key=lambda item: (int(_swarm_field(item, "position", 0) or 0), str(_swarm_field(item, "id"))))
    latest: dict[str, Any] = {}
    lines: list[str] = []

    def visit(parent_id: str | None, prefix: str = "") -> None:
        children = by_parent.get(parent_id, [])
        for index, node in enumerate(children):
            last = index == len(children) - 1
            node_id = str(_swarm_field(node, "id"))
            branch = ("└─" if last else "├─") if unicode else ("`-" if last else "|-")
            continuation = "  " if last else ("│ " if unicode else "| ")
            selected = "›" if unicode and node_id == selected_id else ">" if node_id == selected_id else " "
            status = _swarm_node_status(node, latest)
            phase = str(_swarm_field(node, "checkpoint", _swarm_field(node, "phase", "")) or "")
            suffix = f" · {phase.replace('_', ' ')}" if phase else ""
            lines.append(
                f"{selected} {prefix}{branch} {_swarm_status_mark(status, unicode)} "
                f"{swarm_agent_name(node, nodes)} · {status.replace('_', ' ')}{suffix}"
            )
            visit(node_id, prefix + continuation)

    visit(None)
    return lines


def render_swarm_inspector(
    snapshot: Mapping[str, Any],
    state: SwarmInspectorState,
    *,
    width: int = 120,
    height: int = 34,
    unicode: bool = True,
) -> str:
    """Render one responsive frame of the live read-only swarm inspector."""

    def finish(value: str) -> str:
        if unicode:
            return value
        return value.translate(
            str.maketrans(
                {
                    "·": "|",
                    "›": ">",
                    "│": "|",
                    "─": "-",
                    "↑": "^",
                    "↓": "v",
                    "←": "<",
                    "→": ">",
                    "…": "...",
                    "–": "-",
                    "◉": ">",
                    "●": "x",
                    "○": "o",
                }
            )
        )

    nodes = list(snapshot.get("nodes") or ())
    state.clamp(snapshot)
    latest = _swarm_latest_by_node(snapshot)
    selected = nodes[state.selected_index] if nodes else None
    selected_id = str(_swarm_field(selected, "id", ""))
    statuses = [_swarm_node_status(node, latest).casefold() for node in nodes]
    running = sum(item in {"running", "in_progress", "planning", "reviewing", "testing"} for item in statuses)
    completed = sum(item in {"completed", "done"} for item in statuses)
    failed = sum(item in {"failed", "blocked", "revision_required", "uncertain"} for item in statuses)
    tabs = "[AGENTS]  TREE" if state.tab == "agents" else " AGENTS  [TREE]"
    header = (
        f"SWARM INSPECTOR  ·  {tabs}  ·  {len(nodes)} agents  ·  "
        f"{running} working  ·  {completed} done"
        + (f"  ·  {failed} attention" if failed else "")
    )
    rule = "─" * max(8, width) if unicode else "-" * max(8, width)
    if not nodes:
        return finish("\n".join((header[:width], rule[:width], "No specialists have been materialized yet.", "Esc close · R refresh")))

    if state.tab == "tree":
        tree = _swarm_tree_lines(nodes, selected_id=selected_id, unicode=unicode)
        body = tree[: max(4, height - 5)]
        hidden = len(tree) - len(body)
        if hidden > 0:
            body.append(f"… {hidden} more nodes" if unicode else f"... {hidden} more nodes")
        return finish("\n".join(
            [header[:width], rule[:width], *(_fit(line, width).rstrip() for line in body), rule[:width], "↑↓ select · ←→/Tab view · Enter prompt · R refresh · Esc close"]
        ))

    left_width = max(28, min(42, int(width * 0.36)))
    right_width = max(28, width - left_width - 3)
    list_height = max(7, height - 6)
    half = list_height // 2
    start = max(0, min(max(0, len(nodes) - list_height), state.selected_index - half))
    visible = nodes[start : start + list_height]
    left: list[str] = []
    for offset, node in enumerate(visible, start=start):
        node_id = str(_swarm_field(node, "id"))
        status = _swarm_node_status(node, latest)
        cursor = "›" if unicode and offset == state.selected_index else ">" if offset == state.selected_index else " "
        left.append(
            _fit(
                f"{cursor} {offset + 1:02d} {_swarm_status_mark(status, unicode)} {swarm_agent_name(node, nodes)}",
                left_width,
            ).rstrip()
        )

    profiles = snapshot.get("profiles") or {}
    traces = snapshot.get("traces") or {}
    profile = profiles.get(selected_id, {}) if isinstance(profiles, Mapping) else {}
    trace = traces.get(selected_id) if isinstance(traces, Mapping) else None
    agent = latest.get(selected_id)
    status = _swarm_node_status(selected, latest)
    phase = str(_swarm_field(agent, "phase", _swarm_field(selected, "checkpoint", "pending")))
    role = str(_swarm_field(agent, "role", _swarm_field(selected, "assigned_role", "specialist")))
    mission = str(profile.get("mission") or _swarm_field(selected, "objective", "")).strip()
    deliverable = str(profile.get("deliverable") or "Materialized, independently verified component package.").strip()
    expertise = list(profile.get("expertise") or ())
    interfaces = list(profile.get("owned_interfaces") or ())
    concerns = list(_swarm_node_metadata(selected).get("concern_ids") or ())
    capability_values = [*expertise[:3], *concerns[:3], *interfaces[:2]]
    capabilities = ", ".join(dict.fromkeys(str(item) for item in capability_values if str(item).strip())) or "Bounded implementation and verification"
    prompt = ""
    if trace is not None:
        prompt = str(_swarm_field(trace, "self_prompt", "") or _swarm_field(trace, "system_prompt", "")).strip()
    right: list[str] = [
        swarm_agent_name(selected, nodes).upper(),
        f"{_swarm_status_mark(status, unicode)} {status.replace('_', ' ')}  ·  {role}/{phase}",
        f"Path  {_swarm_ancestry(selected, nodes)}",
        "",
        "CAN DO",
        *textwrap.wrap(capabilities, width=right_width)[:3],
        "",
        "ASSIGNMENT",
        *textwrap.wrap(mission or "Waiting for its typed assignment.", width=right_width)[:4],
        "",
        "DELIVERS",
        *textwrap.wrap(deliverable, width=right_width)[:2],
        "",
        "CURRENT PROMPT · REDACTED",
    ]
    prompt_rows = max(3, height - len(right) - 5) if state.prompt_expanded else 4
    right.extend(textwrap.wrap(" ".join(prompt.split()), width=right_width)[:prompt_rows] or ["(model call has not started yet)"])
    rows = max(len(left), len(right))
    body = [
        _fit(left[index] if index < len(left) else "", left_width)
        + " │ "
        + _fit(right[index] if index < len(right) else "", right_width).rstrip()
        for index in range(rows)
    ]
    return finish("\n".join(
        [header[:width], rule[:width], *body[: max(8, height - 4)], rule[:width], "↑↓ switch agent · ←→/Tab tree · Enter expand prompt · R refresh · Esc close"]
    ))


def run_swarm_inspector(
    snapshot_provider: Callable[[], Mapping[str, Any]],
    *,
    initial_tab: str = "agents",
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    no_color: bool = False,
    reduced_motion: bool = False,
    force: bool = False,
    app_input: Any | None = None,
    app_output: Any | None = None,
) -> None:
    """Open a live, keyboard-navigable, read-only swarm workspace."""

    if not PROMPT_TOOLKIT_AVAILABLE:
        return
    if not force and not rich_terminal_available(input_func, output):
        return
    state = SwarmInspectorState(tab=initial_tab if initial_tab in {"agents", "tree"} else "agents")
    holder: dict[str, Mapping[str, Any]] = {"snapshot": snapshot_provider()}
    unicode = terminal_supports_unicode(output)
    bindings = KeyBindings()

    def redraw(event: Any) -> None:
        state.clamp(holder["snapshot"])
        event.app.invalidate()

    @bindings.add("up")
    def _up(event: Any) -> None:
        state.move(holder["snapshot"], -1)
        redraw(event)

    @bindings.add("down")
    def _down(event: Any) -> None:
        state.move(holder["snapshot"], 1)
        redraw(event)

    @bindings.add("pageup")
    def _page_up(event: Any) -> None:
        state.move(holder["snapshot"], -8)
        redraw(event)

    @bindings.add("pagedown")
    def _page_down(event: Any) -> None:
        state.move(holder["snapshot"], 8)
        redraw(event)

    @bindings.add("left")
    def _left(event: Any) -> None:
        state.select_tab("agents")
        redraw(event)

    @bindings.add("right")
    def _right(event: Any) -> None:
        state.select_tab("tree")
        redraw(event)

    @bindings.add("tab")
    def _tab(event: Any) -> None:
        state.select_tab("tree" if state.tab == "agents" else "agents")
        redraw(event)

    @bindings.add("enter")
    @bindings.add("p")
    def _prompt(event: Any) -> None:
        state.prompt_expanded = not state.prompt_expanded
        redraw(event)

    @bindings.add("r")
    def _refresh(event: Any) -> None:
        holder["snapshot"] = snapshot_provider()
        redraw(event)

    @bindings.add("escape")
    @bindings.add("c-c")
    def _close(event: Any) -> None:
        event.app.exit(result=None)

    def frame() -> Any:
        try:
            size = get_app().output.get_size()
            width, height = int(size.columns), int(size.rows)
        except (AttributeError, RuntimeError, ValueError):
            width, height = 120, 34
        rendered = render_swarm_inspector(
            holder["snapshot"],
            state,
            width=max(60, width - 1),
            height=max(18, height - 1),
            unicode=unicode,
        )
        return FormattedText([("class:details.body", rendered)])

    root = Window(
        content=FormattedTextControl(frame),
        wrap_lines=False,
        always_hide_cursor=True,
    )
    kwargs: dict[str, Any] = {}
    prompt_output = _prompt_output(output, app_output)
    if prompt_output is _UNUSABLE_OUTPUT:
        return
    if app_input is not None:
        kwargs["input"] = app_input
    if prompt_output is not None:
        kwargs["output"] = prompt_output
    application = Application(
        layout=Layout(root),
        key_bindings=bindings,
        style=_make_style(no_color),
        full_screen=True,
        mouse_support=False,
        **kwargs,
    )

    async def refresher() -> None:
        interval = 1.5 if reduced_motion else 0.75
        while True:
            await asyncio.sleep(interval)
            holder["snapshot"] = snapshot_provider()
            state.clamp(holder["snapshot"])
            get_app().invalidate()

    def pre_run() -> None:
        get_app().create_background_task(refresher())

    try:
        application.run(pre_run=pre_run)
    except (EOFError, KeyboardInterrupt):
        return


def _list_fragments(state: ChoiceListState, unicode: bool) -> Any:
    width = max(28, int(_app_columns() * (0.45 if _app_columns() >= 88 else 0.9)))
    fragments: list[tuple[str, str]] = []
    visible, above, below = state.viewport()
    if above:
        fragments.append(("class:choice.meta", _fit("  ↑ more" if unicode else "  [more above]", width) + "\n"))
    if not visible:
        fragments.append(("class:choice.disabled", _fit("  No matching choices", width)))
    for row, index in enumerate(visible):
        item = state.items[index]
        selected = index == state.selected_index
        marker = "›" if unicode and selected else ">" if selected else " "
        status = "Unavailable" if item.disabled else item.meta
        text = f"{marker} {item.label}" + (f"  {status}" if status else "")
        style = "class:choice.selected" if selected else "class:choice.disabled" if item.disabled else "class:choice"
        fragments.append((style, _fit(text, width)))
        if row < len(visible) - 1 or below:
            fragments.append((style, "\n"))
    if below:
        fragments.append(("class:choice.meta", _fit("  ↓ more" if unicode else "  [more below]", width)))
    return FormattedText(fragments)


def _detail_fragments(state: ChoiceListState) -> Any:
    item = state.current
    if item is None:
        return FormattedText([("class:details.title", "No choice selected\n\n"), ("class:details.body", "Type to change the filter.")])
    fragments: list[tuple[str, str]] = [("class:details.title", item.label + "\n")]
    if item.meta:
        fragments.append(("class:choice.meta", item.meta + "\n"))
    fragments.append(("class:details.body", "\n" + (item.description or "No additional details.")))
    if item.disabled:
        fragments.extend(
            (
                ("class:warning", "\n\nUnavailable\n"),
                ("class:details.body", item.disabled_reason or "This choice cannot be selected."),
            )
        )
    return FormattedText(fragments)


def prompt_text(
    *,
    title: str,
    subtitle: str = "",
    step_label: str = "",
    initial: str = "",
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    no_color: bool = False,
    force: bool = False,
    app_input: Any | None = None,
    app_output: Any | None = None,
) -> str | None:
    """Collect one free-form answer in the same full-screen visual language."""

    def plain_fallback() -> str | None:
        try:
            value = input_func(f"{title}: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return value or None

    if not PROMPT_TOOLKIT_AVAILABLE:
        return plain_fallback()
    if not force and not rich_terminal_available(input_func, output):
        return plain_fallback()

    buffer = Buffer(multiline=False)
    buffer.text = str(initial)
    feedback = {"text": ""}
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event: Any) -> None:
        value = buffer.text.strip()
        if not value:
            feedback["text"] = "Write an answer before continuing."
            event.app.invalidate()
            return
        event.app.exit(result=value)

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    @bindings.add("c-q")
    def _exit_session(event: Any) -> None:
        event.app.exit(result=_EXIT_SELECTION)

    header_parts: list[tuple[str, str]] = []
    if step_label:
        header_parts.append(("class:header.step", step_label + "\n"))
    header_parts.append(("class:header.title", title + "\n"))
    if subtitle:
        header_parts.append(("class:header.subtitle", subtitle))
    header = Window(
        content=FormattedTextControl(FormattedText(header_parts)),
        height=1 + bool(step_label) + bool(subtitle),
        always_hide_cursor=True,
    )
    input_window = Window(
        content=BufferControl(buffer=buffer, focusable=True),
        style="class:composer.input",
        height=1,
    )
    composer = VSplit(
        [
            Window(
                content=FormattedTextControl(
                    FormattedText([("class:composer.prompt", "> ")])
                ),
                width=2,
                always_hide_cursor=True,
            ),
            input_window,
        ]
    )

    def feedback_fragments() -> Any:
        return FormattedText([("class:warning", feedback["text"])])

    root = HSplit(
        [
            header,
            Window(height=2),
            composer,
            Window(height=1),
            Window(
                content=FormattedTextControl(feedback_fragments),
                height=1,
                always_hide_cursor=True,
            ),
            Window(height=Dimension(weight=1)),
            Window(
                content=FormattedTextControl(
                    FormattedText([("class:footer", "Enter Save · Esc Back · Ctrl+Q Exit")])
                ),
                height=1,
                always_hide_cursor=True,
            ),
        ]
    )
    kwargs: dict[str, Any] = {}
    prompt_output = _prompt_output(output, app_output)
    if prompt_output is _UNUSABLE_OUTPUT:
        return plain_fallback()
    if app_input is not None:
        kwargs["input"] = app_input
    if prompt_output is not None:
        kwargs["output"] = prompt_output
    application = Application(
        layout=Layout(root, focused_element=input_window),
        key_bindings=bindings,
        style=_make_style(no_color),
        full_screen=True,
        mouse_support=False,
        **kwargs,
    )
    try:
        result = application.run()
    except (EOFError, KeyboardInterrupt):
        return None
    if result is _EXIT_SELECTION:
        raise UserExitRequested()
    return str(result) if isinstance(result, str) and result.strip() else None


def _horizontal_action_fragments(state: ChoiceListState) -> Any:
    fragments: list[tuple[str, str]] = []
    for index, item in enumerate(state.items):
        selected = index == state.selected_index
        style = (
            "class:choice.selected"
            if selected
            else "class:choice.disabled"
            if item.disabled
            else "class:choice"
        )
        label = f"  {item.label}  "
        fragments.append((style, label))
        if index < len(state.items) - 1:
            fragments.append(("", "   "))
    return FormattedText(fragments)


def select_horizontal_action(
    items: Iterable[ChoiceItem],
    *,
    title: str,
    body: str = "",
    subtitle: str = "",
    step_label: str = "",
    initial_key: str | None = None,
    shortcuts: Mapping[str, str] | None = None,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    no_color: bool = False,
    force: bool = False,
    app_input: Any | None = None,
    app_output: Any | None = None,
    require_explicit_selection: bool = False,
    cancelable: bool = True,
) -> ChoiceItem | None:
    """Choose from a fixed horizontal action bar with Left/Right and Enter."""

    values = tuple(items)
    if not values:
        raise ValueError("horizontal action selector requires at least one item")
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None
    if not force and not rich_terminal_available(input_func, output):
        return None

    state = ChoiceListState.create(
        values,
        initial_key=initial_key,
        page_size=len(values),
        filterable=False,
    )
    selection_armed = not require_explicit_selection
    bindings = KeyBindings()

    def redraw(event: Any) -> None:
        event.app.invalidate()

    def move(amount: int) -> None:
        nonlocal selection_armed
        indices = state.matching_indices
        if not indices:
            return
        try:
            position = indices.index(state.selected_index)
        except ValueError:
            position = 0
        for offset in range(1, len(indices) + 1):
            candidate = indices[(position + amount * offset) % len(indices)]
            if not state.items[candidate].disabled:
                state.selected_index = candidate
                state.feedback = ""
                selection_armed = True
                return

    @bindings.add("left")
    @bindings.add("s-tab")
    def _left(event: Any) -> None:
        move(-1)
        redraw(event)

    @bindings.add("right")
    @bindings.add("tab")
    def _right(event: Any) -> None:
        move(1)
        redraw(event)

    @bindings.add("enter")
    def _enter(event: Any) -> None:
        if not selection_armed:
            state.feedback = "Choose with Left/Right first, or press the highlighted letter."
            redraw(event)
            return
        selected = state.activate()
        if selected is not None:
            event.app.exit(result=selected)
        else:
            redraw(event)

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event: Any) -> None:
        if cancelable:
            event.app.exit(result=None)
            return
        state.feedback = "Choose Yes or No explicitly; this request is still waiting."
        redraw(event)

    @bindings.add("c-q")
    def _exit_session(event: Any) -> None:
        event.app.exit(result=_EXIT_SELECTION)

    for shortcut, target_key in dict(shortcuts or {}).items():
        normalized = str(shortcut).strip().lower()
        if not normalized:
            continue

        def _shortcut(event: Any, key: str = str(target_key)) -> None:
            if not state.select_key(key):
                state.feedback = "That action is unavailable."
                redraw(event)
                return
            selected = state.activate()
            if selected is not None:
                event.app.exit(result=selected)
            else:
                redraw(event)

        bindings.add(normalized)(_shortcut)

    header_parts: list[tuple[str, str]] = []
    if step_label:
        header_parts.append(("class:header.step", step_label + "\n"))
    header_parts.append(("class:header.title", title + "\n"))
    if subtitle:
        header_parts.append(("class:header.subtitle", subtitle))
    header = Window(
        content=FormattedTextControl(FormattedText(header_parts)),
        height=1 + bool(step_label) + bool(subtitle),
        always_hide_cursor=True,
    )
    body_window = Window(
        content=FormattedTextControl(
            FormattedText([("class:details.body", str(body).strip())])
        ),
        wrap_lines=True,
        always_hide_cursor=True,
        height=Dimension(weight=1),
    )
    feedback = Window(
        content=FormattedTextControl(
            lambda: FormattedText([("class:warning", state.feedback)])
        ),
        height=1,
        always_hide_cursor=True,
    )
    actions = Window(
        content=FormattedTextControl(lambda: _horizontal_action_fragments(state)),
        align="CENTER",
        wrap_lines=True,
        height=2,
        always_hide_cursor=True,
    )
    shortcut_hint = ""
    if shortcuts:
        shortcut_hint = " · " + "/".join(str(key).upper() for key in shortcuts) + " Quick select"
    footer = Window(
        content=FormattedTextControl(
            FormattedText(
                [("class:footer", "←/→ Switch · Enter Select · Esc Back" + shortcut_hint)]
            )
        ),
        align="CENTER",
        height=1,
        always_hide_cursor=True,
    )
    root = HSplit(
        [
            header,
            Window(height=1),
            body_window,
            feedback,
            Window(height=1, char="─" if terminal_supports_unicode(output) else "-"),
            actions,
            footer,
        ]
    )
    kwargs: dict[str, Any] = {}
    prompt_output = _prompt_output(output, app_output)
    if prompt_output is _UNUSABLE_OUTPUT:
        return None
    if app_input is not None:
        kwargs["input"] = app_input
    if prompt_output is not None:
        kwargs["output"] = prompt_output
    application = Application(
        layout=Layout(root),
        key_bindings=bindings,
        style=_make_style(no_color),
        full_screen=True,
        mouse_support=False,
        **kwargs,
    )
    try:
        result = application.run()
    except (EOFError, KeyboardInterrupt):
        return None
    if result is _EXIT_SELECTION:
        raise UserExitRequested()
    return result if isinstance(result, ChoiceItem) else None


def select_choice(
    items: Iterable[ChoiceItem],
    *,
    title: str,
    subtitle: str = "",
    step_label: str = "",
    action_label: str = "Choose",
    initial_key: str | None = None,
    filterable: bool = True,
    page_size: int = 7,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    no_color: bool = False,
    reduced_motion: bool = False,
    force: bool = False,
    app_input: Any | None = None,
    app_output: Any | None = None,
    shortcuts: Mapping[str, str] | None = None,
) -> ChoiceItem | None:
    """Open a full-screen keyboard selector and return its item, or ``None``.

    Disabled choices remain focusable so their explanation is visible, but
    Enter cannot activate them.  Typing filters across labels, keys, metadata,
    and descriptions. Esc returns ``None``; Ctrl+Q raises
    :class:`UserExitRequested` to leave the whole session.
    """

    del reduced_motion  # The selector itself has no continuously animated region.
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None
    if not force and not rich_terminal_available(input_func, output):
        return None
    unicode = terminal_supports_unicode(output)
    state = ChoiceListState.create(
        items,
        initial_key=initial_key,
        page_size=page_size,
        filterable=filterable,
    )
    bindings = KeyBindings()

    def redraw(event: Any) -> None:
        event.app.invalidate()

    @bindings.add("up")
    def _up(event: Any) -> None:
        state.move(-1)
        redraw(event)

    @bindings.add("down")
    def _down(event: Any) -> None:
        state.move(1)
        redraw(event)

    @bindings.add("home")
    def _home(event: Any) -> None:
        state.home()
        redraw(event)

    @bindings.add("end")
    def _end(event: Any) -> None:
        state.end()
        redraw(event)

    @bindings.add("pageup")
    def _page_up(event: Any) -> None:
        state.page(-1)
        redraw(event)

    @bindings.add("pagedown")
    def _page_down(event: Any) -> None:
        state.page(1)
        redraw(event)

    @bindings.add("backspace")
    def _backspace(event: Any) -> None:
        state.backspace()
        redraw(event)

    @bindings.add("c-u")
    def _clear(event: Any) -> None:
        state.clear_query()
        redraw(event)

    @bindings.add("enter")
    def _enter(event: Any) -> None:
        selected = state.activate()
        if selected is not None:
            event.app.exit(result=selected)
        else:
            redraw(event)

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    @bindings.add("c-q")
    def _exit_session(event: Any) -> None:
        event.app.exit(result=_EXIT_SELECTION)

    for shortcut, target_key in dict(shortcuts or {}).items():
        normalized_shortcut = str(shortcut).strip().lower()
        if not normalized_shortcut or normalized_shortcut in {
            "up", "down", "home", "end", "pageup", "pagedown", "enter",
            "escape", "c-c", "c-q", "backspace", "c-u",
        }:
            continue

        def _shortcut(event: Any, key: str = str(target_key)) -> None:
            if not state.select_key(key):
                state.feedback = "That shortcut is unavailable."
                redraw(event)
                return
            selected = state.activate()
            if selected is not None:
                event.app.exit(result=selected)
            else:
                redraw(event)

        bindings.add(normalized_shortcut)(_shortcut)

    @bindings.add("<any>")
    def _filter(event: Any) -> None:
        if filterable and event.data and event.data.isprintable():
            state.append_query(event.data)
            redraw(event)

    header_parts: list[tuple[str, str]] = []
    if step_label:
        header_parts.append(("class:header.step", step_label + "\n"))
    header_parts.append(("class:header.title", title + "\n"))
    if subtitle:
        header_parts.append(("class:header.subtitle", subtitle))
    header_height = 1 + bool(step_label) + bool(subtitle)
    header = Window(
        content=FormattedTextControl(FormattedText(header_parts)),
        height=header_height,
        always_hide_cursor=True,
    )

    def filter_fragments() -> Any:
        if not filterable:
            return FormattedText([])
        label = f"Filter: {state.query}" if state.query else "Type to filter"
        return FormattedText([("class:filter", label)])

    filter_window = Window(
        content=FormattedTextControl(filter_fragments),
        height=1 if filterable else 0,
        always_hide_cursor=True,
    )

    def make_list_window() -> Any:
        return Window(
            content=FormattedTextControl(lambda: _list_fragments(state, unicode)),
            wrap_lines=False,
            always_hide_cursor=True,
        )

    def make_detail_window() -> Any:
        return Window(
            content=FormattedTextControl(lambda: _detail_fragments(state)),
            wrap_lines=True,
            always_hide_cursor=True,
        )

    wide = VSplit(
        [make_list_window(), Window(width=2, char=" "), make_detail_window()],
        padding=1,
    )
    narrow = HSplit(
        [
            make_list_window(),
            Window(height=1, char="─" if unicode else "-"),
            make_detail_window(),
        ]
    )
    body = DynamicContainer(lambda: wide if _app_columns() >= 88 else narrow)

    def feedback_fragments() -> Any:
        return FormattedText([("class:warning", state.feedback)]) if state.feedback else FormattedText([])

    feedback = Window(
        content=FormattedTextControl(feedback_fragments),
        height=1,
        always_hide_cursor=True,
    )
    def footer_fragments() -> Any:
        shortcut_hint = ""
        if shortcuts:
            labels = ", ".join(str(key).upper() for key in shortcuts)
            shortcut_hint = f" · {labels} Quick action"
        return FormattedText(
            [
                (
                    "class:footer",
                    _fit(
                        _choice_footer(_app_columns(), action_label, filterable, unicode)
                        + shortcut_hint,
                        _app_columns(),
                    ).rstrip(),
                )
            ]
        )

    footer = Window(
        content=FormattedTextControl(footer_fragments),
        height=1,
        always_hide_cursor=True,
    )
    root = HSplit([header, Window(height=1), filter_window, Window(height=1), body, feedback, footer])
    kwargs: dict[str, Any] = {}
    prompt_output = _prompt_output(output, app_output)
    if prompt_output is _UNUSABLE_OUTPUT:
        return None
    if app_input is not None:
        kwargs["input"] = app_input
    if prompt_output is not None:
        kwargs["output"] = prompt_output
    application = Application(
        layout=Layout(root),
        key_bindings=bindings,
        style=_make_style(no_color),
        full_screen=True,
        mouse_support=False,
        **kwargs,
    )
    try:
        result = application.run()
    except (EOFError, KeyboardInterrupt):
        return None
    if result is _EXIT_SELECTION:
        raise UserExitRequested()
    return result if isinstance(result, ChoiceItem) else None


__all__ = [
    "ChoiceItem",
    "ChoiceListState",
    "SwarmInspectorState",
    "NINE_DOT_STATES",
    "NineDotFrame",
    "PROMPT_TOOLKIT_AVAILABLE",
    "PersistentWorkspaceApp",
    "UserExitRequested",
    "WorkspaceInput",
    "nine_dot_frame",
    "inline_square_levels",
    "loading_grid_levels",
    "render_choices",
    "render_nine_dot",
    "render_welcome",
    "render_persistent_workspace",
    "prompt_text",
    "select_horizontal_action",
    "rich_terminal_available",
    "run_loading_task",
    "run_swarm_inspector",
    "run_welcome_screen",
    "select_choice",
    "render_swarm_inspector",
    "swarm_agent_name",
    "terminal_supports_unicode",
]

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
import sys
import textwrap
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Callable, Iterable, Sequence, TextIO

try:  # Optional at import time; line-mode callers must keep working without it.
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import DynamicContainer, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.output.defaults import create_output
    from prompt_toolkit.styles import Style

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - covered through the public fallback API.
    Application = get_app = FormattedText = KeyBindings = None  # type: ignore[assignment]
    DynamicContainer = HSplit = Layout = VSplit = Window = None  # type: ignore[assignment]
    FormattedTextControl = Dimension = Style = create_output = None  # type: ignore[assignment]
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
    "choice.meta": "#777777",
    "choice.selected": "bold #ffffff bg:#164e63",
    "choice.disabled": "#666666",
    "details.title": "bold #ffffff",
    "details.body": "#a8a8a8",
    "warning": "bold #f0b429",
    "filter": "#35d06f",
    "footer": "#777777",
    "loading.dots": "bold #35d06f",
    "loading.square.0": "#444444",
    "loading.square.1": "#707070",
    "loading.square.2": "#aaaaaa",
    "loading.square.3": "#f0f0f0",
    "loading.title": "bold #f0f0f0",
    "loading.detail": "#8a8a8a",
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
        return task()

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
        tick = int((time.monotonic() - started) / 0.12) if animate else 0
        levels = inline_square_levels(
            state,
            tick,
            reduced_motion=reduced_motion,
            no_color=no_color,
        )
        square = "▪" if use_unicode else "#"
        fragments: list[tuple[str, str]] = [
            (f"class:loading.square.{level}", square) for level in levels
        ]
        fragments.extend((("", "  "), ("class:loading.title", _clean_line(title))))
        if detail:
            fragments.append(("class:loading.detail", "\n" + _clean_line(detail)))
        return FormattedText(fragments)

    main = Window(
        content=FormattedTextControl(content),
        align="CENTER",
        always_hide_cursor=True,
        height=Dimension(preferred=2 if detail else 1),
    )
    root = HSplit(
        [
            Window(height=Dimension(weight=1)),
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
        refresh_interval=0.12 if animate else None,
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
        return FormattedText(
            [("class:footer", _choice_footer(_app_columns(), action_label, filterable, unicode))]
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
    "NINE_DOT_STATES",
    "NineDotFrame",
    "PROMPT_TOOLKIT_AVAILABLE",
    "UserExitRequested",
    "nine_dot_frame",
    "inline_square_levels",
    "render_choices",
    "render_nine_dot",
    "render_welcome",
    "rich_terminal_available",
    "run_loading_task",
    "run_welcome_screen",
    "select_choice",
    "terminal_supports_unicode",
]

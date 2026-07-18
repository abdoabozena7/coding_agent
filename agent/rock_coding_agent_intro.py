#!/usr/bin/env python3
"""
GA3BAD Birth Intro
================================

A dependency-free, cross-platform terminal animation inspired by a small,
angry rock character. Every visible part of the character is drawn with
rapidly changing digits.

Run:
    python rock_coding_agent_intro.py

Integration:
    from rock_coding_agent_intro import play_intro

    play_intro()
    # Start your real coding agent here.
"""

from __future__ import annotations

import atexit
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# EASY SETTINGS
# ---------------------------------------------------------------------------

TITLE = "GA3BAD"
PROMPT = "ENTER TO CONTINUE"

FPS = 24
LOOP_ANIMATION = True
USE_ALTERNATE_SCREEN = True

# Main brown / stone palette inspired by the supplied character.
STONE_LIGHT = (177, 142, 103)
STONE_MID_1 = (147, 113, 82)
STONE_MID_2 = (113, 83, 61)
STONE_DARK = (72, 52, 40)
STONE_EDGE = (45, 33, 27)
CRACK = (24, 18, 15)

EYE_RING = (226, 207, 170)
IRIS = (83, 59, 40)
PUPIL = (8, 7, 6)
HIGHLIGHT = (255, 251, 235)

MOUTH = (12, 8, 7)
MOUTH_INNER = (60, 29, 22)
TITLE_COLOR = (190, 151, 101)
PROMPT_COLOR = (126, 101, 76)
FLOOR_COLOR = (65, 47, 35)

# Birth intro timing:
# 1) a large brown "GA3BAD" word appears,
# 2) the digits collapse and gather,
# 3) they form the rock character,
# 4) then the original loop keeps repeating forever.
BIRTH_HOLD_FRAMES = 96
BIRTH_MORPH_FRAMES = 46
BIRTH_TOTAL_FRAMES = BIRTH_HOLD_FRAMES + BIRTH_MORPH_FRAMES

TITLE_FONT_5X7 = {
    # Bold 7x9 terminal font. The doubled strokes make GA3BAD much clearer
    # than the earlier thin 5x7 version.
    "G": (
        "0111110",
        "1100011",
        "1100000",
        "1100000",
        "1101111",
        "1100011",
        "1100011",
        "1100011",
        "0111110",
    ),
    "A": (
        "0011100",
        "0110110",
        "1100011",
        "1100011",
        "1111111",
        "1100011",
        "1100011",
        "1100011",
        "1100011",
    ),
    "3": (
        "1111110",
        "0000011",
        "0000011",
        "0000110",
        "0011100",
        "0000110",
        "0000011",
        "1100011",
        "0111110",
    ),
    "B": (
        "1111100",
        "1100110",
        "1100011",
        "1100110",
        "1111100",
        "1100110",
        "1100011",
        "1100110",
        "1111100",
    ),
    "D": (
        "1111100",
        "1100110",
        "1100011",
        "1100011",
        "1100011",
        "1100011",
        "1100011",
        "1100110",
        "1111100",
    ),
}

RESET = "\x1b[0m"
HOME = "\x1b[H"
CLEAR = "\x1b[2J"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
ALT_ON = "\x1b[?1049h"
ALT_OFF = "\x1b[?1049l"


@dataclass
class Pose:
    body_x: float = 0.0
    body_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    arm_state: float = 0.0   # 0 normal, 1 flex, 2 overhead, 3 ground slam
    mouth_open: float = 0.0
    shake: float = 0.0
    impact: float = 0.0
    debris: float = 0.0
    energy: float = 0.0


# ---------------------------------------------------------------------------
# TERMINAL CONTROL
# ---------------------------------------------------------------------------

def _enable_windows_ansi() -> None:
    """Enable ANSI escape sequences in classic Windows terminals."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        # Windows Terminal, PowerShell, and modern cmd usually work anyway.
        pass


class KeyReader:
    """Non-blocking Enter detection on Windows, Linux, and macOS."""

    def __init__(self) -> None:
        self._old_settings = None

    def __enter__(self) -> "KeyReader":
        if os.name != "nt" and sys.stdin.isatty():
            try:
                import termios
                import tty

                self._old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self._old_settings = None
        return self

    def pressed_enter(self) -> bool:
        if not sys.stdin.isatty():
            return False

        if os.name == "nt":
            try:
                import msvcrt

                while msvcrt.kbhit():
                    key = msvcrt.getwch()
                    if key in ("\r", "\n"):
                        return True
                return False
            except Exception:
                return False

        try:
            import select

            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                key = sys.stdin.read(1)
                return key in ("\r", "\n")
        except Exception:
            pass
        return False

    def __exit__(self, exc_type, exc, tb) -> None:
        if os.name != "nt" and self._old_settings is not None:
            try:
                import termios

                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    self._old_settings,
                )
            except Exception:
                pass


_terminal_is_active = False


def _restore_terminal() -> None:
    global _terminal_is_active
    if not _terminal_is_active:
        return
    try:
        suffix = SHOW_CURSOR + RESET
        if USE_ALTERNATE_SCREEN:
            suffix += ALT_OFF
        sys.stdout.write(suffix)
        sys.stdout.flush()
    except Exception:
        pass
    _terminal_is_active = False


atexit.register(_restore_terminal)


# ---------------------------------------------------------------------------
# SMALL MATH HELPERS
# ---------------------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t)


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_out_back(t: float) -> float:
    t = clamp(t)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def rgb(color: Tuple[int, int, int]) -> str:
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m"


def stable_noise(x: float, y: float, salt: int = 0) -> float:
    """Fast deterministic pseudo-noise in the range 0..1."""
    value = math.sin(x * 12.9898 + y * 78.233 + salt * 37.719) * 43758.5453
    return value - math.floor(value)


def digit_for(x: float, y: float, frame: int, salt: int = 0) -> str:
    """
    Produce a digit that changes every frame but remains spatially textured.
    """
    value = (
        int(abs(x) * 19)
        + int(abs(y) * 31)
        + frame * (3 + salt % 5)
        + int(stable_noise(x, y, salt) * 97)
    )
    return str(value % 10)


def ellipse_metric(
    x: float,
    y: float,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle: float = 0.0,
) -> float:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x - cx
    dy = y - cy
    px = dx * cos_a + dy * sin_a
    py = -dx * sin_a + dy * cos_a
    return (px / rx) ** 2 + (py / ry) ** 2


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * abx + (py - ay) * aby) / length_sq
    t = clamp(t)
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def polyline_distance(
    x: float,
    y: float,
    points: Tuple[Tuple[float, float], ...],
) -> float:
    result = 999.0
    for start, end in zip(points, points[1:]):
        result = min(
            result,
            point_segment_distance(
                x, y, start[0], start[1], end[0], end[1]
            ),
        )
    return result


# ---------------------------------------------------------------------------
# ANIMATION TIMELINE
# ---------------------------------------------------------------------------

SEQUENCE_FRAMES = 192


def pose_for_frame(frame: int) -> Pose:
    f = max(0, frame - BIRTH_TOTAL_FRAMES) % SEQUENCE_FRAMES
    pose = Pose()

    # Always-alive micro motion.
    idle_wave = math.sin(frame * 0.16)
    pose.body_x = idle_wave * 0.55
    pose.body_y = math.sin(frame * 0.31) * 0.12
    pose.energy = 0.15 + 0.10 * abs(math.sin(frame * 0.22))

    # 0..28: suspicious side-to-side idle.
    if f < 28:
        pose.body_x += math.sin(f * 0.28) * 0.65
        pose.arm_state = 0.08 * (0.5 + 0.5 * math.sin(f * 0.35))
        return pose

    # 28..50: squash, squeeze, and spring back.
    if f < 50:
        t = (f - 28) / 22.0
        pulse = math.sin(math.pi * t)
        pose.scale_x = 1.0 + 0.16 * pulse
        pose.scale_y = 1.0 - 0.24 * pulse
        pose.body_y += 1.9 * pulse
        pose.arm_state = 0.25 * pulse
        pose.energy = pulse
        return pose

    # 50..84: flex both arms.
    if f < 84:
        t = (f - 50) / 34.0
        rise = ease_out_back(min(1.0, t * 1.5))
        pose.arm_state = clamp(rise)
        muscle = max(0.0, math.sin((t - 0.25) * math.pi * 3.0))
        pose.scale_x = 1.0 + 0.035 * muscle
        pose.scale_y = 1.0 - 0.020 * muscle
        pose.body_y -= 0.35 * rise
        pose.energy = 0.65 + 0.35 * muscle
        return pose

    # 84..108: raise fists over the head.
    if f < 108:
        t = smoothstep((f - 84) / 24.0)
        pose.arm_state = 1.0 + t
        pose.body_y -= 0.8 * t
        pose.scale_y = 1.0 + 0.035 * math.sin(t * math.pi)
        pose.energy = 1.0
        return pose

    # 108..124: violent ground slam.
    if f < 124:
        t = (f - 108) / 16.0
        if t < 0.42:
            windup = smoothstep(t / 0.42)
            pose.arm_state = 2.0
            pose.body_y = -0.8 - 0.7 * windup
            pose.scale_y = 1.0 + 0.08 * windup
        else:
            hit = smoothstep((t - 0.42) / 0.58)
            pose.arm_state = lerp(2.0, 3.0, hit)
            pose.body_y = lerp(-1.5, 2.6, hit)
            pose.scale_x = 1.0 + 0.20 * hit
            pose.scale_y = 1.0 - 0.28 * hit
            pose.impact = hit
            pose.debris = hit
            pose.shake = (1.0 - hit * 0.65) * 1.25
        pose.energy = 1.0
        return pose

    # 124..160: roar with an open mouth and screen vibration.
    if f < 160:
        t = (f - 124) / 36.0
        recover = smoothstep(t)
        pose.arm_state = lerp(3.0, 0.65, recover)
        pose.scale_x = lerp(1.20, 1.02, recover)
        pose.scale_y = lerp(0.72, 1.03, recover)
        pose.body_y = lerp(2.6, -0.20, recover)
        pose.mouth_open = math.sin(math.pi * clamp(t * 1.15)) ** 0.55
        pose.shake = (1.0 - t) * 0.45 + pose.mouth_open * 0.18
        pose.debris = max(0.0, 1.0 - t * 1.25)
        pose.energy = 1.0
        return pose

    # 160..192: settle back into the angry idle.
    t = smoothstep((f - 160) / 32.0)
    pose.arm_state = lerp(0.65, 0.0, t)
    pose.body_y -= 0.18 * (1.0 - t)
    pose.scale_x = lerp(1.02, 1.0, t)
    pose.scale_y = lerp(1.03, 1.0, t)
    pose.mouth_open = 0.0
    pose.energy = 0.35 * (1.0 - t)
    return pose


# ---------------------------------------------------------------------------
# CHARACTER GEOMETRY
# ---------------------------------------------------------------------------

CRACKS = (
    ((-15.0, -7.6), (-11.5, -4.0), (-13.0, -0.7), (-9.5, 2.2)),
    ((3.5, -9.6), (2.1, -6.4), (5.4, -3.2), (3.9, 0.2)),
    ((14.3, -7.0), (11.8, -3.8), (14.7, -1.0)),
    ((-4.0, 4.7), (-1.4, 2.3), (1.3, 4.0), (4.4, 3.0)),
    ((-17.8, 4.8), (-14.2, 3.2), (-12.2, 6.0)),
    ((10.8, 5.1), (8.7, 2.9), (11.8, 1.4), (13.4, -0.2)),
    ((-3.8, -9.8), (-1.5, -7.8), (-3.2, -5.9)),
)


def interpolate_arm_state(state: float) -> Tuple[float, float, float, float, float]:
    """
    Returns positive-side arm geometry:
    upper_x, upper_y, fist_x, fist_y, angle
    """
    normal = (25.2, 2.9, 28.9, 4.4, -0.10)
    flexed = (23.8, -1.6, 28.0, -5.2, -0.60)
    raised = (18.4, -7.1, 11.6, -12.0, -0.92)
    impact = (24.7, 5.4, 27.9, 9.1, 0.34)

    if state <= 1.0:
        a, b, t = normal, flexed, state
    elif state <= 2.0:
        a, b, t = flexed, raised, state - 1.0
    else:
        a, b, t = raised, impact, state - 2.0

    t = smoothstep(t)
    return tuple(lerp(a[i], b[i], t) for i in range(5))  # type: ignore[return-value]


def stone_color(x: float, y: float, edge: float, frame: int) -> Tuple[int, int, int]:
    light = (
        0.52
        - x * 0.010
        - y * 0.022
        + (stable_noise(x * 0.8, y * 0.8, frame // 7) - 0.5) * 0.22
    )

    if edge > 0.91:
        return STONE_EDGE
    if light > 0.64:
        return STONE_LIGHT
    if light > 0.49:
        return STONE_MID_1
    if light > 0.33:
        return STONE_MID_2
    return STONE_DARK


def sample_character(
    x: float,
    y: float,
    pose: Pose,
    frame: int,
) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    # Screen vibration is deterministic so the whole character stays coherent.
    shake_x = math.sin(frame * 2.90) * pose.shake
    shake_y = math.sin(frame * 3.71 + 0.9) * pose.shake * 0.35

    world_x = x - shake_x
    world_y = y - shake_y

    # Body-local coordinates, including squash and stretch.
    bx = (world_x - pose.body_x) / max(0.25, pose.scale_x)
    by = (world_y - pose.body_y) / max(0.25, pose.scale_y)

    result: Optional[Tuple[str, Tuple[int, int, int]]] = None

    # Feet sit behind the main body.
    left_foot = ellipse_metric(world_x, world_y, -13.2 + pose.body_x, 10.4 + pose.body_y, 6.4, 2.6, -0.08)
    right_foot = ellipse_metric(world_x, world_y, 13.2 + pose.body_x, 10.4 + pose.body_y, 6.4, 2.6, 0.08)
    if min(left_foot, right_foot) <= 1.0:
        edge = min(left_foot, right_foot)
        result = (
            digit_for(world_x, world_y, frame, 13),
            stone_color(world_x, world_y, edge, frame),
        )

    # Arms and fists.
    upper_x, upper_y, fist_x, fist_y, arm_angle = interpolate_arm_state(pose.arm_state)
    for side in (-1.0, 1.0):
        ux = side * upper_x + pose.body_x
        fx = side * fist_x + pose.body_x
        angle = arm_angle * side

        upper = ellipse_metric(
            world_x,
            world_y,
            ux,
            upper_y + pose.body_y,
            3.25,
            5.0,
            angle,
        )
        fist = ellipse_metric(
            world_x,
            world_y,
            fx,
            fist_y + pose.body_y,
            3.75,
            3.35,
            angle * 0.35,
        )
        limb = min(upper, fist)
        if limb <= 1.0:
            result = (
                digit_for(world_x, world_y, frame, 19 + int(side)),
                stone_color(world_x, world_y, limb, frame),
            )

    # Rough main boulder silhouette.
    theta = math.atan2(by / 10.2, bx / 24.8)
    roughness = (
        0.045 * math.sin(theta * 7.0 + 0.5)
        + 0.026 * math.sin(theta * 13.0 - 1.3)
        + 0.015 * math.sin(theta * 23.0 + 2.0)
    )
    body_metric = (bx / 24.8) ** 2 + ((by + 0.25) / 10.45) ** 2
    body_limit = (1.0 + roughness) ** 2

    if body_metric <= body_limit:
        normalized_edge = body_metric / max(0.01, body_limit)
        result = (
            digit_for(bx, by, frame, 2),
            stone_color(bx, by, normalized_edge, frame),
        )

        # Stone cracks.
        for crack_index, crack in enumerate(CRACKS):
            threshold = 0.20 + 0.06 * (crack_index % 3)
            if polyline_distance(bx, by, crack) < threshold:
                result = (
                    digit_for(bx, by, frame, 41 + crack_index),
                    CRACK,
                )
                break

    # Eye geometry.
    for side in (-1.0, 1.0):
        eye_x = side * 9.1
        eye_y = -2.2

        eye = ellipse_metric(bx, by, eye_x, eye_y, 7.25, 4.55)
        if eye <= 1.0:
            result = (
                digit_for(bx, by, frame, 61 + int(side)),
                EYE_RING if eye < 0.83 else STONE_EDGE,
            )

        # Huge dark pupil/iris like the reference.
        look_offset = math.sin(frame * 0.065) * 0.45
        iris = ellipse_metric(
            bx,
            by,
            eye_x + side * 0.35 + look_offset,
            eye_y + 0.25,
            4.75,
            3.60,
        )
        if iris <= 1.0:
            result = (
                digit_for(bx, by, frame, 67 + int(side)),
                IRIS if iris > 0.68 else PUPIL,
            )

        # Two glossy highlights per eye, still digits.
        highlight_1 = ellipse_metric(
            bx,
            by,
            eye_x - side * 1.95 + look_offset,
            eye_y - 1.55,
            0.95,
            0.72,
        )
        highlight_2 = ellipse_metric(
            bx,
            by,
            eye_x + side * 1.65 + look_offset,
            eye_y + 1.25,
            0.52,
            0.40,
        )
        if highlight_1 <= 1.0 or highlight_2 <= 1.0:
            result = (
                digit_for(bx, by, frame, 73),
                HIGHLIGHT,
            )

    # Heavy angry brows.
    left_brow = point_segment_distance(bx, by, -16.0, -7.8, -3.0, -4.6)
    right_brow = point_segment_distance(bx, by, 16.0, -7.8, 3.0, -4.6)
    brow_gate = by < -3.5 and abs(bx) < 17.5
    if brow_gate and min(left_brow, right_brow) < 1.05:
        result = (
            digit_for(bx, by, frame, 83),
            STONE_EDGE,
        )

    # Mouth: frown when closed, roaring oval when open.
    if pose.mouth_open < 0.10:
        if abs(bx) <= 5.2:
            mouth_curve_y = 5.6 - 0.047 * bx * bx
            if abs(by - mouth_curve_y) < 0.34:
                result = (
                    digit_for(bx, by, frame, 91),
                    MOUTH,
                )
    else:
        mouth_ry = 0.75 + 3.2 * pose.mouth_open
        mouth = ellipse_metric(bx, by, 0.0, 5.5, 5.1, mouth_ry)
        if mouth <= 1.0:
            color = MOUTH if mouth > 0.72 else MOUTH_INNER
            result = (
                digit_for(bx, by, frame, 97),
                color,
            )

            # A small stone-colored lower lip helps the roar read clearly.
            lip = ellipse_metric(bx, by, 0.0, 6.5 + mouth_ry * 0.35, 3.3, 0.55)
            if lip <= 1.0:
                result = (
                    digit_for(bx, by, frame, 101),
                    STONE_DARK,
                )

    return result


def sample_debris(
    x: float,
    y: float,
    pose: Pose,
    frame: int,
) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    if pose.debris <= 0.01:
        return None

    # Small digit fragments burst outward from the impact area.
    for i in range(26):
        direction = -1.0 if i % 2 == 0 else 1.0
        speed = 4.0 + (i % 7) * 0.70
        start_x = direction * (6.0 + (i % 5) * 2.8)
        target_x = start_x + direction * speed * pose.debris
        target_y = 11.1 - math.sin(pose.debris * math.pi) * (2.0 + (i % 4))
        target_y += (i % 3) * 0.45

        if abs(x - target_x) < 0.45 and abs(y - target_y) < 0.40:
            color = STONE_MID_2 if i % 3 else STONE_DARK
            return digit_for(x, y, frame, 120 + i), color

    return None



# ---------------------------------------------------------------------------
# BIRTH INTRO HELPERS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def title_source_points() -> Tuple[Tuple[float, float], ...]:
    """
    Build a wide, bold GA3BAD word from numeric particles.

    The font uses a 7x9 matrix with thick strokes. Each active matrix cell
    creates several nearby particles, so the letters look solid and readable
    while still being made entirely from changing digits.
    """
    scale_x = 1.28
    scale_y = 1.45
    gap = 1.90

    text_value = TITLE
    widths = []
    for char in text_value:
        pattern = TITLE_FONT_5X7.get(char, TITLE_FONT_5X7["A"])
        widths.append(len(pattern[0]))

    total_units = sum(widths) + gap * (len(text_value) - 1)
    start_x = -(total_units - 1.0) * scale_x / 2.0
    start_y = -5.8

    points = []
    cursor = start_x

    for char_index, char in enumerate(text_value):
        pattern = TITLE_FONT_5X7.get(char, TITLE_FONT_5X7["A"])

        for row_index, row in enumerate(pattern):
            for col_index, cell in enumerate(row):
                if cell != "1":
                    continue

                px = cursor + col_index * scale_x
                py = start_y + row_index * scale_y

                # Main point.
                points.append((px, py))

                # Horizontal thickness. This is deliberately wider than the
                # old font so each stroke occupies a clearly visible band.
                points.append((px + 0.52, py))

                # Slight lower fill removes holes caused by terminal-cell
                # aspect ratios and gives the word a heavier logo-like feel.
                if (row_index + col_index + char_index) % 2 == 0:
                    points.append((px, py + 0.42))
                    points.append((px + 0.52, py + 0.42))

        cursor += (len(pattern[0]) + gap) * scale_x

    return tuple(points)


@lru_cache(maxsize=1)
def target_character_points() -> Tuple[Tuple[float, float, Tuple[int, int, int]], ...]:
    """
    Collect a cloud of points from the neutral rock character silhouette.
    Those points are the destinations for the word particles.
    """
    neutral_pose = Pose()
    points = []

    y = -13.5
    while y <= 13.5:
        x = -28.0
        while x <= 28.0:
            sampled = sample_character(x, y, neutral_pose, 0)
            if sampled is not None:
                _, color = sampled
                points.append((x, y, color))
            x += 1.05
        y += 0.90

    return tuple(points)


def world_to_cell(
    x: float,
    y: float,
    art_width: int,
    art_height: int,
    scale_x: float,
    scale_y: float,
) -> Optional[Tuple[int, int]]:
    col = int(round(x * scale_x + (art_width - 1) / 2.0))
    row = int(round(y * scale_y + (art_height - 1) / 2.0))
    if 0 <= col < art_width and 0 <= row < art_height:
        return row, col
    return None


def render_birth_art(frame: int, art_width: int, art_height: int) -> list[str]:
    scale_x = art_width / 72.0
    scale_y = art_height / 28.0

    # Buffer stores (priority, char, color)
    buffer: dict[Tuple[int, int], Tuple[int, str, Tuple[int, int, int]]] = {}

    def plot(
        x: float,
        y: float,
        char: str,
        color: Tuple[int, int, int],
        priority: int = 0,
    ) -> None:
        cell = world_to_cell(x, y, art_width, art_height, scale_x, scale_y)
        if cell is None:
            return
        old = buffer.get(cell)
        if old is None or priority >= old[0]:
            buffer[cell] = (priority, char, color)

    sources = title_source_points()
    targets = target_character_points()

    if frame < BIRTH_HOLD_FRAMES:
        # Phase 1: large brown GA3BAD word.
        pulse = 0.92 + 0.08 * math.sin(frame * 0.24)
        title_color = tuple(int(channel * pulse) for channel in TITLE_COLOR)
        for i, (sx, sy) in enumerate(sources):
            char = str((i + frame * 3) % 10)
            plot(sx, sy, char, title_color, priority=1)

    else:
        # Phase 2: the word collapses and becomes the character.
        raw_t = (frame - BIRTH_HOLD_FRAMES) / max(1, BIRTH_MORPH_FRAMES)
        morph_t = smoothstep(raw_t)
        particle_count = max(len(sources), len(targets))

        for i in range(particle_count):
            sx, sy = sources[i % len(sources)]
            tx, ty, target_color = targets[i % len(targets)]

            # A small stagger makes the gathering feel more cinematic.
            delay = (i % 17) * 0.012
            local_t = clamp((morph_t - delay) / (1.0 - delay + 1e-9))
            local_t = smoothstep(local_t)

            spiral = (1.0 - local_t) * (1.6 + (i % 5) * 0.22)
            angle = frame * 0.18 + i * 0.61
            px = lerp(sx, tx, local_t) + math.cos(angle) * spiral
            py = lerp(sy, ty, local_t) + math.sin(angle * 1.17) * spiral * 0.55

            color = TITLE_COLOR if local_t < 0.72 else target_color
            char = str((i * 7 + frame * 5) % 10)
            plot(px, py, char, color, priority=1)

        # During the final reveal, fade in the actual character behind
        # the particles so the facial details become crisp.
        if morph_t > 0.55:
            reveal = clamp((morph_t - 0.55) / 0.45)
            reveal_threshold = 1.0 - reveal

            neutral_pose = Pose()
            for row in range(art_height):
                y = (row - (art_height - 1) / 2.0) / max(0.01, scale_y)
                for col in range(art_width):
                    x = (col - (art_width - 1) / 2.0) / max(0.01, scale_x)
                    sampled = sample_character(x, y, neutral_pose, frame)
                    if sampled is None:
                        continue
                    if stable_noise(x, y, frame // 3 + 77) < reveal_threshold:
                        continue
                    char, color = sampled
                    plot(x, y, char, color, priority=2)

    # Convert the buffer to terminal lines.
    lines = []
    for row in range(art_height):
        pieces = []
        current_color: Optional[Tuple[int, int, int]] = None
        for col in range(art_width):
            info = buffer.get((row, col))
            if info is None:
                if current_color is not None:
                    pieces.append(RESET)
                    current_color = None
                pieces.append(" ")
                continue

            _, char, color = info
            if color != current_color:
                pieces.append(rgb(color))
                current_color = color
            pieces.append(char)

        if current_color is not None:
            pieces.append(RESET)
        lines.append("".join(pieces))

    return lines


# ---------------------------------------------------------------------------
# FRAME RENDERING
# ---------------------------------------------------------------------------

def center_visible(text: str, visible_length: int, terminal_width: int) -> str:
    left = max(0, (terminal_width - visible_length) // 2)
    return " " * left + text


def render_frame(frame: int, pose: Pose) -> str:
    terminal = shutil.get_terminal_size((100, 38))
    cols = max(44, terminal.columns)
    rows = max(20, terminal.lines)

    # Keep enough room for title and prompt.
    available_width = max(42, cols - 2)
    available_height = max(14, rows - 7)

    # The design coordinate system is approximately 72 x 28.
    scale = min(available_width / 72.0, available_height / 28.0, 1.0)
    art_width = max(42, int(72 * scale))
    art_height = max(14, int(28 * scale))
    scale_x = art_width / 72.0
    scale_y = art_height / 28.0

    total_height = art_height + 5
    top_padding = max(0, (rows - total_height) // 2)

    lines = [""] * top_padding

    # For the one-time birth animation, the big brown GA3BAD word is already
    # on-screen, so we hide the small header until the character is born.
    show_small_title = frame >= BIRTH_TOTAL_FRAMES
    if show_small_title:
        title_text = rgb(TITLE_COLOR) + TITLE + RESET
        lines.append(center_visible(title_text, len(TITLE), cols))
        lines.append("")
    else:
        lines.append("")
        lines.append("")

    left_padding = max(0, (cols - art_width) // 2)

    if frame < BIRTH_TOTAL_FRAMES:
        birth_lines = render_birth_art(frame, art_width, art_height)
        for line in birth_lines:
            lines.append(" " * left_padding + line)
    else:
        for row in range(art_height):
            y = (row - (art_height - 1) / 2.0) / max(0.01, scale_y)
            pieces = [" " * left_padding]
            current_color: Optional[Tuple[int, int, int]] = None

            for col in range(art_width):
                x = (col - (art_width - 1) / 2.0) / max(0.01, scale_x)

                sampled = sample_character(x, y, pose, frame)
                if sampled is None:
                    sampled = sample_debris(x, y, pose, frame)

                # Sparse ground digits appear during the slam.
                if sampled is None and pose.impact > 0.08 and abs(y - 12.45) < 0.33:
                    if stable_noise(x, frame * 0.1, 212) > 0.58:
                        sampled = (
                            digit_for(x, y, frame, 213),
                            FLOOR_COLOR,
                        )

                if sampled is None:
                    if current_color is not None:
                        pieces.append(RESET)
                        current_color = None
                    pieces.append(" ")
                    continue

                char, color = sampled
                if color != current_color:
                    pieces.append(rgb(color))
                    current_color = color
                pieces.append(char)

            if current_color is not None:
                pieces.append(RESET)
            lines.append("".join(pieces))

    lines.append("")
    pulse = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(frame * 0.16))
    prompt_color = tuple(int(channel * pulse) for channel in PROMPT_COLOR)
    prompt_text = rgb(prompt_color) + PROMPT + RESET
    lines.append(center_visible(prompt_text, len(PROMPT), cols))

    # Pad the remaining terminal rows so old frames never leak through.
    while len(lines) < rows:
        lines.append("")

    return HOME + "\n".join(lines[:rows])


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def play_intro(*, loop: bool = LOOP_ANIMATION, fps: int = FPS) -> None:
    """
    Play the intro until Enter is pressed.

    After the user presses Enter, the terminal is restored and this function
    returns, so the real coding-agent interface can begin.
    """
    global _terminal_is_active

    _enable_windows_ansi()
    frame_duration = 1.0 / max(1, fps)

    prefix = ""
    if USE_ALTERNATE_SCREEN:
        prefix += ALT_ON
    prefix += HIDE_CURSOR + CLEAR + HOME
    sys.stdout.write(prefix)
    sys.stdout.flush()
    _terminal_is_active = True

    frame = 0
    started = time.perf_counter()

    try:
        with KeyReader() as keys:
            while True:
                target_time = started + frame * frame_duration
                now = time.perf_counter()
                if target_time > now:
                    time.sleep(target_time - now)

                pose = pose_for_frame(frame)
                sys.stdout.write(render_frame(frame, pose))
                sys.stdout.flush()

                if keys.pressed_enter():
                    break

                frame += 1
                if not loop and frame >= BIRTH_TOTAL_FRAMES + SEQUENCE_FRAMES:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        _restore_terminal()


def main() -> None:
    play_intro()

    # Standalone demo destination.
    # Replace these lines with your real agent UI after integrating the file.
    sys.stdout.write(CLEAR + HOME)
    sys.stdout.write(rgb(TITLE_COLOR) + TITLE + RESET + "\n\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

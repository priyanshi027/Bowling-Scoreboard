#!/usr/bin/env python3
"""Render a synthetic bowling scoreboard clip with known ground truth.

Used to exercise the whole pipeline without needing the real capture, and to
give the project a self-test that does not depend on a binary asset.

    python tools/make_synthetic.py --out data/synthetic.mp4

The clip deliberately reproduces the awkward parts of the real footage:

* the active player's row is dark-on-yellow while the rest are white-on-blue,
* a stretch of frames shows only a pin animation, with no table at all,
* another stretch has a pin-animation window pasted over part of the table,
* a throw appears partway through, so the board is live rather than static.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402

W, H = 1280, 720
MARGIN = 20

DEEP_BLUE = (150, 40, 12)
MID_BLUE = (190, 95, 30)
LIGHT_BLUE = (225, 165, 90)
YELLOW = (140, 225, 240)
PALE = (225, 230, 232)
RED = (48, 40, 205)
WHITE = (255, 255, 255)
GOLD = (90, 205, 245)
INK = (110, 45, 15)

FONT = cv2.FONT_HERSHEY_DUPLEX

LANE = "6"
BOWLER = "TARUN"

# Ground truth. Player T's fourth frame appears partway through the clip.
PLAYERS = [
    ("J", ["X", "5-", "-7", "4-"], [15, 20, 27, 31]),
    ("V", ["8-", "3-", "71", "8o1"], [8, 11, 19, 28]),
    ("P", ["X", "4/", "9-", "6-"], [20, 39, 48, 54]),
    ("T", ["61", "1/", "8-"], [7, 25, 33]),
]
T_LATE_THROW = "3"
T_LATE_TOTAL = 36
ACTIVE_ROW = 3


# ------------------------------------------------------------------- geometry


def geometry() -> dict[str, float]:
    table_w = W - 2 * MARGIN
    table_h = table_w / (cfg.CANON_W / cfg.CANON_H)
    title_h = table_h * cfg.TITLE_HEIGHT_FRAC
    y0 = MARGIN + title_h
    header_h = table_h * cfg.HEADER_HEIGHT_FRAC
    sub_h = (table_h - header_h) / (cfg.N_PLAYER_ROWS * 2)
    init_w = table_w * cfg.INITIAL_COL_FRAC
    ttl_w = table_w * cfg.TTL_COL_FRAC
    frame_w = (table_w - init_w - ttl_w) / cfg.N_FRAMES
    return {
        "x0": float(MARGIN), "x1": float(W - MARGIN), "table_w": table_w,
        "y0": y0, "y1": y0 + table_h, "table_h": table_h,
        "title_top": float(MARGIN), "title_h": title_h,
        "header_h": header_h, "sub_h": sub_h,
        "init_w": init_w, "ttl_w": ttl_w, "frame_w": frame_w,
    }


def centred(img: np.ndarray, text: str, box: tuple[float, float, float, float],
            scale: float, colour: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw ``text`` centred in ``box`` = (x, y, w, h), handling circled digits."""
    x, y, w, h = box
    if text.endswith("o") and len(text) == 2:
        text = text[0]
        circle = True
    else:
        circle = False

    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    tx = int(x + (w - tw) / 2)
    ty = int(y + (h + th) / 2)
    cv2.putText(img, text, (tx, ty), FONT, scale, colour, thickness, cv2.LINE_AA)
    if circle:
        cv2.circle(img, (int(tx + tw / 2), int(ty - th / 2)),
                   int(max(tw, th) * 0.78), colour, 2, cv2.LINE_AA)


def draw_throws(img: np.ndarray, text: str, box: tuple[float, float, float, float],
                scale: float, colour: tuple[int, int, int]) -> None:
    """Lay out a throws cell glyph by glyph so marks stay separable."""
    x, y, w, h = box
    tokens: list[str] = []
    for ch in text:
        if ch == "o" and tokens:
            tokens[-1] += "o"
        else:
            tokens.append(ch)
    if not tokens:
        return

    slot = w / max(len(tokens), 2)
    start = x + (w - slot * len(tokens)) / 2
    for i, token in enumerate(tokens):
        centred(img, token, (start + i * slot, y, slot, h), scale, colour)


# --------------------------------------------------------------------- drawing


def render_board(t_has_late_throw: bool, draw_h_rules: bool = True,
                 layout: dict[str, float] | None = None) -> np.ndarray:
    """Render the board.

    ``draw_h_rules`` controls whether horizontal rules are painted between the
    player rows. Setting it False leaves the rows separated by colour banding
    alone, which is how some real boards look, and is a harder case for the
    structure-based frame test in ``detect.py``.

    ``layout`` overrides the geometry, so the pipeline can be tested against
    proportions that differ from the ones in ``config``.
    """
    g = layout or geometry()
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Backdrop with a soft vertical gradient, like the real capture.
    for y in range(H):
        f = y / H
        img[y, :] = [int(DEEP_BLUE[c] * (0.55 + 0.45 * (1 - f))) for c in range(3)]

    x0, x1, y0, y1 = g["x0"], g["x1"], g["y0"], g["y1"]

    # Title strip: lane number over the initials column, bowler name to its right.
    centred(img, LANE, (x0, g["title_top"], g["init_w"], g["title_h"]), 1.8, GOLD, 3)
    cv2.putText(img, BOWLER, (int(x0 + g["init_w"] + 12),
                              int(g["title_top"] + g["title_h"] * 0.82)),
                FONT, 1.25, GOLD, 3, cv2.LINE_AA)

    # Header strip.
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y0 + g["header_h"])),
                  MID_BLUE, -1)

    col_x = [x0, x0 + g["init_w"]]
    col_x += [x0 + g["init_w"] + k * g["frame_w"] for k in range(1, cfg.N_FRAMES + 1)]
    col_x.append(x1)

    for f in range(cfg.N_FRAMES):
        centred(img, str(f + 1), (col_x[f + 1], y0, g["frame_w"], g["header_h"]),
                0.7, WHITE)
    centred(img, "TTL", (col_x[11], y0, g["ttl_w"], g["header_h"]), 0.7, WHITE)

    # Player rows.
    sub_h = g["sub_h"]
    row_top = y0 + g["header_h"]
    for p, (initial, throws, totals) in enumerate(PLAYERS):
        active = p == ACTIVE_ROW
        if active and t_has_late_throw:
            throws = throws + [T_LATE_THROW]
        shown_total = (
            T_LATE_TOTAL if active and t_has_late_throw
            else (totals[-1] if totals else 0)
        )

        ty = row_top + 2 * p * sub_h
        sy = ty + sub_h

        throw_bg = YELLOW if active else MID_BLUE
        score_bg = PALE if active else LIGHT_BLUE
        ink = INK if active else WHITE

        cv2.rectangle(img, (int(x0), int(ty)), (int(x1), int(sy)), throw_bg, -1)
        cv2.rectangle(img, (int(x0), int(sy)), (int(x1), int(sy + sub_h)), score_bg, -1)
        if active:
            cv2.rectangle(img, (int(col_x[11]), int(ty)),
                          (int(x1), int(sy + sub_h)), RED, -1)

        centred(img, initial, (x0, ty, g["init_w"], sub_h), 1.0, ink)

        for f, throw in enumerate(throws):
            draw_throws(img, throw, (col_x[f + 1], ty, g["frame_w"], sub_h), 0.95, ink)
        for f, total in enumerate(totals):
            centred(img, str(total), (col_x[f + 1], sy, g["frame_w"], sub_h), 0.95, ink)

        centred(img, str(shown_total), (col_x[11], sy, g["ttl_w"], sub_h),
                0.95, WHITE if active else WHITE)
        centred(img, "0", (col_x[11], ty, g["ttl_w"], sub_h), 0.95, WHITE)

    # Grid rules: vertical between every column, horizontal between player rows.
    for x in col_x[1:12]:
        cv2.line(img, (int(x), int(y0)), (int(x), int(y1)), WHITE, 2, cv2.LINE_AA)
    if draw_h_rules:
        for k in range(cfg.N_PLAYER_ROWS + 1):
            y = row_top + 2 * k * sub_h
            cv2.line(img, (int(x0), int(y)), (int(x1), int(y)), WHITE, 2, cv2.LINE_AA)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y0)), WHITE, 2, cv2.LINE_AA)

    return img


def render_pins(width: int, height: int) -> np.ndarray:
    """The pin-rack animation: no table structure at all."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (200, 90, 25)
    rows = [(4, 0.30), (3, 0.46), (2, 0.62), (1, 0.78)]
    r = max(6, int(min(width, height) * 0.045))
    for count, fy in rows:
        span = count * r * 3
        for i in range(count):
            cx = int(width / 2 - span / 2 + r * 1.5 + i * r * 3)
            cy = int(height * fy)
            cv2.circle(img, (cx, cy), r, WHITE, -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy - int(r * 1.4)), (int(r * 0.55), r),
                        0, 0, 360, WHITE, -1, cv2.LINE_AA)
    return img


def with_inset(board: np.ndarray) -> np.ndarray:
    """Paste a pin-animation window over the middle-right of the table."""
    out = board.copy()
    g = geometry()
    x = int(g["x0"] + g["init_w"] + 4 * g["frame_w"])
    y = int(g["y0"] + g["header_h"])
    w = int(3.6 * g["frame_w"])
    h = int(3.2 * g["sub_h"])
    out[y:y + h, x:x + w] = render_pins(w, h)
    cv2.rectangle(out, (x, y), (x + w, y + h), DEEP_BLUE, 3)
    return out


# ----------------------------------------------------------------------- main

# The clip's timeline: 100 frames, 4 seconds at 25 fps. "late" means player T's
# fourth throw has appeared.
SEGMENTS: list[tuple[str, int]] = [
    ("board", 30),        # early state
    ("pins", 10),         # animation only, no table
    ("board_late", 20),
    ("inset_late", 15),   # table with an animation window pasted over it
    ("board_late", 25),
]


def state_at(frame_index: int) -> dict[str, object]:
    """What the clip is showing at ``frame_index``."""
    cursor = 0
    for kind, count in SEGMENTS:
        if frame_index < cursor + count:
            return {
                "kind": kind,
                "has_board": kind != "pins",
                "late": kind.endswith("late"),
                "occluded": kind.startswith("inset"),
            }
        cursor += count
    return {"kind": "past_end", "has_board": False, "late": True, "occluded": False}


def build(out_path: Path, fps: int = 25) -> tuple[Path, int]:
    """Write the clip. Returns the path and the total frame count."""
    frames = {
        "board": render_board(False),
        "board_late": render_board(True),
        "pins": render_pins(W, H),
    }
    frames["inset_late"] = with_inset(frames["board_late"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {out_path}")

    total = 0
    try:
        for kind, count in SEGMENTS:
            for _ in range(count):
                writer.write(frames[kind])
                total += 1
    finally:
        writer.release()
    return out_path, total


def expected() -> dict[str, object]:
    """Ground truth for the generated clip, for the tests to assert against."""
    final = {
        "J": (["X", "5-", "-7", "4-"], [15, 20, 27, 31], 31),
        "V": (["8-", "3-", "71", "8o1"], [8, 11, 19, 28], 28),
        "P": (["X", "4/", "9-", "6-"], [20, 39, 48, 54], 54),
        "T": (["61", "1/", "8-", "3"], [7, 25, 33], T_LATE_TOTAL),
    }
    return {"lane": LANE, "bowler": BOWLER, "players": final}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic.mp4")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    path, n = build(Path(args.out), args.fps)
    print(f"wrote {path} ({n} frames, {W}x{H} @ {args.fps} fps)")

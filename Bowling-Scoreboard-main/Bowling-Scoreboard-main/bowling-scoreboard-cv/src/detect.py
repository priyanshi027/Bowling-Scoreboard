"""Stage 1 - locate the scoreboard in a frame and rectify it.

The video interleaves three kinds of frame:

* the full Brunswick scoreboard table,
* a full-screen pin animation with no table at all,
* the table with a small pin-animation window pasted over part of it.

``detect_board`` separates the first and third from the second, returns a
perspective-rectified copy of the table so every later stage can work in fixed
canonical pixel coordinates, and reports where an inset window is covering the
table so those cells can be ignored for that frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as cfg
from .imgutil import count_lines, line_masks, runs, to_gray, uniform_run


@dataclass
class BoardDetection:
    """Result of trying to find the table in one frame."""

    ok: bool
    reason: str = ""
    quad: np.ndarray | None = None       # 4x2 float32, original frame coords
    matrix: np.ndarray | None = None     # perspective transform to canonical
    warp: np.ndarray | None = None       # canonical BGR table
    title: np.ndarray | None = None      # rectified strip above the table
    n_h_lines: int = 0
    n_v_lines: int = 0
    geometry: object | None = None    # TableGeometry when measured
    occluders: list[tuple[int, int, int, int]] = field(default_factory=list)


# --------------------------------------------------------------------- helpers


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    by_y = pts[np.argsort(pts[:, 1])]
    top, bottom = by_y[:2], by_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_skew_deg(quad: np.ndarray) -> float:
    """Largest deviation of the quad's edges from axis alignment, in degrees."""
    tl, tr, br, bl = quad
    top = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    bottom = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    left = np.degrees(np.arctan2(bl[1] - tl[1], bl[0] - tl[0])) - 90.0
    right = np.degrees(np.arctan2(br[1] - tr[1], br[0] - tr[0])) - 90.0
    return float(max(abs(top), abs(bottom), abs(left), abs(right)))


# ------------------------------------------------------------------- detection


def classify_frame(frame: np.ndarray) -> tuple[bool, int, int]:
    """Cheap test for "does this frame contain the table at all?".

    Returns ``(is_board, n_horizontal_lines, n_vertical_lines)``.
    """
    small = _downscale(frame)
    gray = to_gray(small)
    h_mask, v_mask = line_masks(gray)
    n_h = count_lines(h_mask, axis=0)
    n_v = count_lines(v_mask, axis=1)
    return (n_h >= cfg.MIN_H_LINES and n_v >= cfg.MIN_V_LINES), n_h, n_v


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= cfg.DETECT_SCALE:
        return frame
    scale = cfg.DETECT_SCALE / w
    return cv2.resize(frame, (cfg.DETECT_SCALE, int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def detect_board(frame: np.ndarray) -> BoardDetection:
    """Find and rectify the scoreboard table in ``frame``.

    Outlining strategies, tried strongest first and each *verified* before
    being accepted, so a strategy that misfires falls through instead of
    producing a silently wrong grid:

    1. the combined horizontal and vertical rule mask, when enough horizontal
       rules exist to bound the table.
    2. the column rules alone.

    A third strategy based on image periodicity lives in ``structure.py``. It is
    the only approach that got anywhere on the real capture, but it is not
    converged and is deliberately not wired in - see that module's notes.

    Verification is deliberately cheap and semantic: rectify, map the grid, and
    require that the grid found its own columns and rows in the warp. That is a
    real check, because ``grid`` derives them from the image independently.
    """
    h0, w0 = frame.shape[:2]
    small = _downscale(frame)
    scale = w0 / small.shape[1]
    gray = to_gray(small)

    h_mask, v_mask = line_masks(gray)
    n_h = count_lines(h_mask, axis=0)
    n_v = count_lines(v_mask, axis=1)

    candidates: list[tuple[str, np.ndarray | None]] = []
    geometry = None                     # see structure.py - not yet production ready

    columns = uniform_run(_line_centres(v_mask, axis=1))
    if n_v >= cfg.MIN_V_LINES and len(columns) >= cfg.MIN_UNIFORM_COLS:
        if n_h >= cfg.MIN_H_LINES:
            candidates.append(("structure", _quad_from_structure(h_mask, v_mask, small.shape)))
        candidates.append(("columns", _quad_from_columns(columns, v_mask, small.shape)))

    if not candidates:
        return BoardDetection(False, "no table structure", n_h_lines=n_h, n_v_lines=n_v)

    frame_area = small.shape[0] * small.shape[1]
    last_reason = "no candidate outline survived verification"

    for source, quad_small in candidates:
        if quad_small is None:
            continue
        if cv2.contourArea(quad_small.astype(np.float32)) < cfg.MIN_BOARD_AREA_FRAC * frame_area:
            last_reason = "table too small"
            continue
        skew = _quad_skew_deg(quad_small)
        if skew > cfg.MAX_SKEW_DEG:
            last_reason = f"implausible skew {skew:.1f} deg"
            continue

        quad = quad_small * scale
        dst = np.array(
            [[0, 0], [cfg.CANON_W - 1, 0],
             [cfg.CANON_W - 1, cfg.CANON_H - 1], [0, cfg.CANON_H - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad, dst)
        warp = cv2.warpPerspective(frame, matrix, (cfg.CANON_W, cfg.CANON_H),
                                   flags=cv2.INTER_CUBIC)

        trusted = source == "periodicity"
        if not _grid_agrees(warp, trusted):
            last_reason = f"grid did not agree with the {source} outline"
            continue

        det = BoardDetection(True, f"ok ({source})", quad=quad, matrix=matrix, warp=warp,
                             title=_rectify_title(frame, quad),
                             n_h_lines=n_h, n_v_lines=n_v,
                             geometry=geometry if trusted else None)
        det.occluders = find_occluders(warp)
        return det

    return BoardDetection(False, last_reason, n_h_lines=n_h, n_v_lines=n_v)


def _grid_agrees(warp: np.ndarray, trusted: bool) -> bool:
    """Cheap semantic check that an outline produced a usable cell map.

    For a measured layout the edges are exact by construction, so re-detecting
    them proves nothing; instead each player's initial cell must hold exactly one
    mark, which a misplaced outline will not satisfy. For the mask-based outlines
    the check is that the grid can find its own columns and rows in the warp.
    """
    from . import grid as grid_mod          # local import: grid imports segment
    from . import segment

    gmap = grid_mod.build_grid(warp, trusted_layout=trusted)
    if not trusted:
        return gmap.cols_detected and gmap.rows_detected

    singles = 0
    for cell in gmap.cells:
        if cell.role != grid_mod.INITIAL:
            continue
        if len(segment.extract_glyphs(grid_mod.crop(warp, cell))) == 1:
            singles += 1
    return singles >= cfg.MIN_INITIALS_READ


def _line_centres(mask: np.ndarray, axis: int) -> list[float]:
    """Centre positions of the line runs in a morphology mask."""
    profile = (mask > 0).sum(axis=0 if axis == 1 else 1).astype(np.float32)
    if profile.max() <= 0:
        return []
    hot = profile > 0.35 * profile.max()
    return [(a + b - 1) / 2.0 for a, b in runs(hot)]


def _quad_from_structure(h_mask: np.ndarray, v_mask: np.ndarray,
                         shape: tuple[int, ...]) -> np.ndarray | None:
    """Outline the table from its combined rules. Preferred when both exist."""
    structure = cv2.dilate(
        cv2.bitwise_or(h_mask, v_mask),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours, _ = cv2.findContours(structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    return order_quad(cv2.boxPoints(cv2.minAreaRect(biggest)))


def _quad_from_columns(columns: list[float], v_mask: np.ndarray,
                       shape: tuple[int, ...]) -> np.ndarray | None:
    """Outline the table from the column rules alone.

    Needed when the rows are separated by colour banding rather than by drawn
    rules, because then the horizontal extent cannot come from a horizontal mask.
    The outermost rules bound the *frame* columns only, so the initials column
    and the total column are added back using their widths relative to one frame
    column - a ratio fixed by the layout in ``config``.
    """
    if len(columns) < 2:
        return None

    gap = float(np.median(np.diff(columns)))
    if gap <= 0:
        return None

    frame_frac = (1.0 - cfg.INITIAL_COL_FRAC - cfg.TTL_COL_FRAC) / cfg.N_FRAMES
    left = columns[0] - gap * (cfg.INITIAL_COL_FRAC / frame_frac)
    right = columns[-1] + gap * (cfg.TTL_COL_FRAC / frame_frac)

    # The rules span the table vertically, so their ink gives the top and bottom.
    rows = np.nonzero((v_mask > 0).any(axis=1))[0]
    if rows.size == 0:
        return None
    top, bottom = float(rows.min()), float(rows.max())

    height, width = shape[0], shape[1]
    left = max(0.0, left)
    right = min(float(width - 1), right)
    top = max(0.0, top)
    bottom = min(float(height - 1), bottom)
    if right - left < 8 or bottom - top < 8:
        return None

    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]],
                    dtype=np.float32)


def _rectify_title(frame: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Rectify the strip directly above the table (lane number + bowler name)."""
    tl, tr, br, bl = quad
    left_h = bl - tl
    right_h = br - tr
    band = cfg.TITLE_HEIGHT_FRAC
    src = np.array([tl - left_h * band, tr - right_h * band, tr, tl], dtype=np.float32)

    # Clamp to the frame so a board flush with the top edge does not sample
    # outside the image.
    h, w = frame.shape[:2]
    src[:, 0] = np.clip(src[:, 0], 0, w - 1)
    src[:, 1] = np.clip(src[:, 1], 0, h - 1)
    if abs(src[0][1] - src[3][1]) < 4:
        return None

    out_w, out_h = cfg.CANON_W, max(8, int(cfg.CANON_H * band))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (out_w, out_h),
                               flags=cv2.INTER_CUBIC)


def find_occluders(warp: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate inset windows pasted over the table.

    An inset reads as a large blob with no internal grid rules. Individual empty
    cells look similar, so a blob only counts as an occluder when it spans
    several cells in *both* axes - something a single empty cell cannot do.
    """
    gray = to_gray(warp)
    h, w = gray.shape[:2]

    h_mask, v_mask = line_masks(gray)
    structure = cv2.dilate(
        cv2.bitwise_or(h_mask, v_mask),
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    # Regions with no grid structure at all.
    empty = cv2.bitwise_not(structure)
    empty = cv2.morphologyEx(
        empty, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    )

    cell_w = w * (1.0 - cfg.INITIAL_COL_FRAC - cfg.TTL_COL_FRAC) / cfg.N_FRAMES
    cell_h = h * (1.0 - cfg.HEADER_HEIGHT_FRAC) / (cfg.N_PLAYER_ROWS * 2)
    min_w = cell_w * cfg.OCCLUDER_MIN_COLS
    min_h = cell_h * cfg.OCCLUDER_MIN_ROWS

    boxes: list[tuple[int, int, int, int]] = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(empty, 8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bw >= min_w and bh >= min_h:
            boxes.append((int(x), int(y), int(bw), int(bh)))
    return boxes


def box_overlap_frac(box: tuple[int, int, int, int],
                     other: tuple[int, int, int, int]) -> float:
    """Fraction of ``box``'s area covered by ``other``."""
    bx, by, bw, bh = box
    ox, oy, ow, oh = other
    ix = max(0, min(bx + bw, ox + ow) - max(bx, ox))
    iy = max(0, min(by + bh, oy + oh) - max(by, oy))
    area = bw * bh
    return (ix * iy) / area if area else 0.0

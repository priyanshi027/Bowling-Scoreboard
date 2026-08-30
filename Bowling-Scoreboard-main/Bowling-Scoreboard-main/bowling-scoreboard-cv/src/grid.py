"""Stage 2 - map the rectified table onto its cells.

Column boundaries come from the table's own vertical rules; row boundaries are
anchored on the four player-initial glyphs in the left column, which sit at the
vertical centre of each player's throw sub-row. Both fall back to the measured
proportions in ``config`` when the image evidence is too weak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as cfg
from . import segment
from .imgutil import line_masks, runs, to_gray, uniform_run

# Cell roles.
INITIAL = "initial"
THROWS = "throws"
SCORE = "score"
TOTAL = "total"


@dataclass(frozen=True)
class Cell:
    """One addressable region of the table."""

    key: str
    role: str
    box: tuple[int, int, int, int]      # x, y, w, h in canonical coords
    player: int
    frame: int | None = None            # 1-based frame number, None for initial/total


@dataclass
class GridMap:
    col_edges: list[int] = field(default_factory=list)   # 13 x-positions
    row_edges: list[int] = field(default_factory=list)   # 9 y-positions
    cells: list[Cell] = field(default_factory=list)
    cols_detected: bool = False
    rows_detected: bool = False

    def by_key(self, key: str) -> Cell | None:
        for cell in self.cells:
            if cell.key == key:
                return cell
        return None


# ------------------------------------------------------------------- columns


def _proportional_col_edges(width: int) -> list[float]:
    left = cfg.INITIAL_COL_FRAC * width
    right = (1.0 - cfg.TTL_COL_FRAC) * width
    step = (right - left) / cfg.N_FRAMES
    edges = [0.0, left]
    edges += [left + k * step for k in range(1, cfg.N_FRAMES)]
    edges += [right, float(width)]
    return edges


def _vertical_line_centres(warp: np.ndarray) -> list[float]:
    """Column-rule positions inside the rectified table.

    Rules sitting on the table's outer border are dropped: they bound the table,
    not a column, and leaving them in would corrupt the even-spacing test.
    """
    gray = to_gray(warp)
    width = gray.shape[1]
    _, v_mask = line_masks(gray)
    profile = (v_mask > 0).sum(axis=0).astype(np.float32)
    if profile.max() <= 0:
        return []

    hot = profile > cfg.GRID_LINE_PROMINENCE * profile.max()
    margin = cfg.EDGE_RULE_MARGIN * width
    return [
        centre
        for centre in ((a + b - 1) / 2.0 for a, b in runs(hot))
        if margin <= centre <= width - 1 - margin
    ]


def find_columns(warp: np.ndarray) -> tuple[list[int], bool]:
    """Locate the 13 column edges of the rectified table.

    Preferred path: the eleven interior rules are equally spaced, so the longest
    evenly spaced run of detected rules *is* the answer, and the outer two edges
    are the table's own borders. This needs no assumption about how wide the
    initials or total columns are.

    That matters: measuring those widths off a reference frame and trusting the
    numbers is fragile - being wrong by more than a fraction of a column width
    makes every cell miss its contents. The proportional prior is only used when
    the rules cannot be found.
    """
    width = warp.shape[1]
    centres = _vertical_line_centres(warp)

    interior = uniform_run(centres)
    if len(interior) == cfg.N_FRAMES + 1:
        edges = [0.0] + list(interior) + [float(width)]
        return [int(round(e)) for e in edges], True

    # Fall back to snapping the measured prior onto whatever rules were found.
    prior = _proportional_col_edges(width)
    if len(centres) < 4:
        return [int(round(e)) for e in prior], False

    col_w = (prior[11] - prior[1]) / cfg.N_FRAMES
    tol = cfg.GRID_COL_TOLERANCE * col_w

    snapped = list(prior)
    hits = 0
    for i in range(1, 12):                      # interior edges only
        nearest = min(centres, key=lambda c: abs(c - prior[i]))
        if abs(nearest - prior[i]) <= tol:
            snapped[i] = nearest
            hits += 1

    # Keep the edges monotonic even if two priors snapped to the same rule.
    for i in range(1, len(snapped)):
        snapped[i] = max(snapped[i], snapped[i - 1] + 2)
    snapped[-1] = float(width)

    return [int(round(e)) for e in snapped], hits >= 8


# ---------------------------------------------------------------------- rows


def _proportional_row_edges(height: int) -> list[float]:
    header = cfg.HEADER_HEIGHT_FRAC * height
    sub_h = (height - header) / (cfg.N_PLAYER_ROWS * 2)
    return [header + k * sub_h for k in range(cfg.N_PLAYER_ROWS * 2 + 1)]


def _initial_anchors(warp: np.ndarray, initial_col_w: int, sub_h_prior: float,
                     header_prior: float) -> list[float] | None:
    """Vertical centres of the four player initials, in canonical coords.

    Each initial is located in its own small crop rather than in one pass over
    the whole column. The active player's row is drawn dark-on-yellow while the
    others are white-on-blue, and a single global threshold over the full column
    cannot serve both; a per-cell crop has a uniform background, so the polarity
    check in ``binarize_fg`` gets it right every time.
    """
    height = warp.shape[0]
    x_end = max(4, initial_col_w)
    half = 0.55 * sub_h_prior

    found: list[tuple[int, float]] = []
    for p in range(cfg.N_PLAYER_ROWS):
        centre = header_prior + (2 * p + 0.5) * sub_h_prior
        y0 = int(max(0, centre - half))
        y1 = int(min(height, centre + half))
        crop = warp[y0:y1, 0:x_end]
        if crop.size == 0:
            continue

        mask = segment.binarize_fg(crop)
        if not mask.any():
            continue

        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best: tuple[int, float] | None = None
        for i in range(1, n):
            _, _, bw, bh, area = stats[i]
            if bw < 3 or area < cfg.MIN_GLYPH_AREA_PX:
                continue
            if not (0.25 * sub_h_prior <= bh <= 0.90 * sub_h_prior):
                continue
            if best is None or area > best[0]:
                best = (int(area), float(centroids[i][1]))
        if best is not None:
            found.append((p, y0 + best[1]))

    if len(found) < 3:
        return None

    # Fit centre = intercept + slope * player_index. The slope is one player row,
    # i.e. two sub-rows, so a missing anchor does not bias the geometry.
    rows = np.array([p for p, _ in found], dtype=np.float64)
    centres = np.array([y for _, y in found], dtype=np.float64)
    slope, intercept = np.polyfit(rows, centres, 1)
    if slope <= 0:
        return None

    return [float(intercept + slope * p) for p in range(cfg.N_PLAYER_ROWS)]


def find_rows(warp: np.ndarray, initial_col_w: int) -> tuple[list[int], bool]:
    """Derive the 9 sub-row boundaries, preferring initial-glyph anchors."""
    height = warp.shape[0]
    prior = _proportional_row_edges(height)
    sub_h_prior = prior[1] - prior[0]

    anchors = _initial_anchors(warp, initial_col_w, sub_h_prior, prior[0])
    if anchors is None:
        return [int(round(e)) for e in prior], False

    # Consecutive initials are one player row apart, i.e. two sub-rows.
    spacing = float(np.median(np.diff(anchors))) if len(anchors) > 1 else 2 * sub_h_prior
    sub_h = spacing / 2.0
    top = anchors[0] - sub_h / 2.0

    plausible = (
        0.70 * sub_h_prior <= sub_h <= 1.35 * sub_h_prior
        and abs(top - prior[0]) <= 0.15 * height
        and top + 8 * sub_h <= height + 0.05 * height
    )
    if not plausible:
        return [int(round(e)) for e in prior], False

    edges = [top + k * sub_h for k in range(cfg.N_PLAYER_ROWS * 2 + 1)]
    edges[-1] = min(edges[-1], height)
    return [int(round(e)) for e in edges], True


# ---------------------------------------------------------------- assembly


def build_grid(warp: np.ndarray, trusted_layout: bool = False) -> GridMap:
    """Full cell map for a rectified table.

    ``trusted_layout`` says the warp was built from measured table geometry, so
    the column rules and sub-rows already sit exactly where the canonical
    proportions put them. Re-detecting them in the warp would only add noise -
    and on a board with faint rules it fails outright, which is precisely the
    case the measured geometry exists to handle.
    """
    if trusted_layout:
        col_edges = [int(round(e)) for e in _proportional_col_edges(warp.shape[1])]
        row_edges = [int(round(e)) for e in _proportional_row_edges(warp.shape[0])]
        cols_ok = rows_ok = True
    else:
        col_edges, cols_ok = find_columns(warp)
        row_edges, rows_ok = find_rows(warp, col_edges[1])

    cells: list[Cell] = []
    for p in range(cfg.N_PLAYER_ROWS):
        throw_top = row_edges[2 * p]
        throw_bot = row_edges[2 * p + 1]
        score_bot = row_edges[2 * p + 2]

        # Player initial: spans both sub-rows of the left column but is drawn on
        # the throws line, so read it from there.
        cells.append(
            Cell(
                key=f"p{p}.initial",
                role=INITIAL,
                box=(col_edges[0], throw_top, col_edges[1] - col_edges[0], throw_bot - throw_top),
                player=p,
            )
        )

        for f in range(1, cfg.N_FRAMES + 1):
            x0, x1 = col_edges[f], col_edges[f + 1]
            cells.append(
                Cell(
                    key=f"p{p}.f{f}.throws",
                    role=THROWS,
                    box=(x0, throw_top, x1 - x0, throw_bot - throw_top),
                    player=p,
                    frame=f,
                )
            )
            cells.append(
                Cell(
                    key=f"p{p}.f{f}.score",
                    role=SCORE,
                    box=(x0, throw_bot, x1 - x0, score_bot - throw_bot),
                    player=p,
                    frame=f,
                )
            )

        x0, x1 = col_edges[11], col_edges[12]
        cells.append(
            Cell(
                key=f"p{p}.total",
                role=TOTAL,
                box=(x0, throw_bot, x1 - x0, score_bot - throw_bot),
                player=p,
            )
        )

    return GridMap(col_edges, row_edges, cells, cols_ok, rows_ok)


def crop(warp: np.ndarray, cell: Cell) -> np.ndarray:
    x, y, w, h = cell.box
    return warp[max(0, y):y + h, max(0, x):x + w]

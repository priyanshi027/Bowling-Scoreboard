"""EXPERIMENTAL - table geometry from image periodicity. Not wired in.

Written while investigating why the shipped detector fails on the real capture,
and kept because the investigation itself is the useful part. ``detect.py`` does
**not** call this; it is not converged.

What the real frames showed (see ``data/real/``):

* The board's rows are separated by **colour banding**, not by drawn horizontal
  rules. A morphological horizontal-line mask finds two or three lines, and the
  ones it does find are the video progress bar.
* The column rules are thin and low contrast against the pale score rows. A
  binarised vertical-line mask finds four of eleven, which is why the shipped
  detector reports "no table structure" on the real board.
* Summing |Sobel x| down each column and running a comb filter over the profile
  *does* recover the column period reliably - 114.9px on both board frames - and
  the pin-animation frame yields nothing, so the discrimination works.
* The four player initials are a good row anchor: they were found at y = 146,
  281, 409, 525 against a truth of 147, 278, 409, 530.

Where it stands: period detection is solid, but pinning the *phase* - which of
the comb's teeth is the first column rule - is not. Every discriminator tried
worked on one board and failed the other:

* gradient energy at the teeth: the rule between frame ten and the total column
  is almost edgeless on the real board, so an energy vote slides the whole
  lattice one column right;
* blob count in the strip left of a candidate tooth: correct on the real board
  (five blobs at the right cut, nine at a wrong one), but the synthetic board
  fills its initials column with the row background colours, so band boundaries
  register as blobs;
* suppressing persistent columns instead of taking a percentile: fixed the
  synthetic, broke the real board's phase.

Anyone picking this up: the period and the row anchors are trustworthy and worth
keeping. The phase needs a cue that does not depend on either board's incidental
styling - the header row's text baseline is the most promising untried idea,
since it exists on both and spans the frame columns only.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as cfg
from .imgutil import uniform_run


@dataclass
class TableGeometry:
    """Where the table is, in the coordinates of the image it was found in."""

    col_rules: list[float]      # N_FRAMES + 1 interior column rules, left to right
    row_centres: list[float]    # centre of each player's throw sub-row
    period: float               # one frame column, in pixels
    row_pitch: float            # one player row (two sub-rows), in pixels
    comb_score: float           # mean normalised gradient along the comb
    comb_contrast: float        # comb energy against the background level

    @property
    def sub_row(self) -> float:
        return self.row_pitch / 2.0

    def quad(self) -> np.ndarray:
        """Table outline, ordered top-left, top-right, bottom-right, bottom-left.

        Chosen so that the rectified table puts the column rules and sub-rows
        exactly where the canonical layout in ``config`` expects them. The
        geometry drives the canvas rather than the other way round, so the cell
        map is correct without assuming the board's proportions in advance.
        """
        first, last = self.col_rules[0], self.col_rules[-1]
        span = 1.0 - cfg.INITIAL_COL_FRAC - cfg.TTL_COL_FRAC
        width = (last - first) / span
        x_left = first - cfg.INITIAL_COL_FRAC * width

        # The eight sub-rows occupy everything below the header strip.
        height = 8 * self.sub_row / (1.0 - cfg.HEADER_HEIGHT_FRAC)
        first_centre_frac = cfg.HEADER_HEIGHT_FRAC + 0.5 * (1.0 - cfg.HEADER_HEIGHT_FRAC) / 8
        y_top = self.row_centres[0] - first_centre_frac * height

        return np.array(
            [[x_left, y_top], [x_left + width, y_top],
             [x_left + width, y_top + height], [x_left, y_top + height]],
            dtype=np.float32,
        )


# ------------------------------------------------------------------- profiles


def _smooth(profile: np.ndarray, k: int = 9) -> np.ndarray:
    return cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32), (1, k), 0).ravel()


def column_profile(gray: np.ndarray) -> np.ndarray:
    """Vertical-edge energy summed down each column, with the baseline removed.

    Summing raw gradient is not enough. A board densely filled with digits has
    vertical edges in nearly every column, so the profile comes out broadly high
    and the rule peaks sit on a large pedestal - the comb then scores almost any
    lattice about equally and picks an arbitrary one. Subtracting a broadly
    smoothed copy leaves only what is locally peaked, which is what a rule is.
    """
    raw = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
    fine = _smooth(raw, cfg.PROFILE_SMOOTH)
    baseline = _smooth(raw, cfg.PROFILE_BASELINE)
    return np.clip(fine - baseline, 0.0, None)


def _best_comb(profile: np.ndarray) -> tuple[float, float, float, float] | None:
    """Best ``(score, contrast, period, offset)`` for an evenly spaced comb.

    Scored by the *mean* along the comb, not the sum. A comb at half the true
    period has twice as many teeth but must place half of them on empty ground,
    so its mean falls - which is what keeps the search off sub-multiples of the
    real period.
    """
    prof = profile.astype(np.float32)
    peak = float(prof.max())
    if peak <= 0:
        return None
    prof = prof / peak
    background = float(prof.mean())
    if background <= 0:
        return None

    n = len(prof)
    lo = max(4.0, n * cfg.COMB_PERIOD_MIN_FRAC)
    hi = n * cfg.COMB_PERIOD_MAX_FRAC
    if hi <= lo:
        return None

    best: tuple[float, float, float, float] | None = None
    for period in np.arange(lo, hi, cfg.COMB_PERIOD_STEP):
        for offset in np.arange(0.0, period, 1.0):
            xs = np.arange(offset, n - 1, period)
            if len(xs) < cfg.COMB_MIN_TEETH:
                continue
            score = float(prof[np.round(xs).astype(int)].mean())
            if best is None or score > best[0]:
                best = (score, score / background, float(period), float(offset))
    return best


# --------------------------------------------------------------------- anchors


def _contrast_map(patch: np.ndarray) -> np.ndarray:
    """Deviation from a heavily blurred copy - high wherever ink sits.

    Polarity-agnostic on purpose: the active player's initial is dark on yellow
    while the others are light on blue.
    """
    patch = patch.astype(np.float32)
    return np.abs(patch - cv2.GaussianBlur(patch, (0, 0), 9))


def _blob_spans(profile: np.ndarray, rel: float, min_len: int) -> list[tuple[int, int]]:
    hot = _smooth(profile, 15) > rel * float(_smooth(profile, 15).max() or 1.0)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(hot):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                spans.append((start, i))
            start = None
    if start is not None and len(hot) - start >= min_len:
        spans.append((start, len(hot)))
    return spans


def _ink_by_row(patch: np.ndarray) -> np.ndarray:
    """How much ink each row of ``patch`` holds.

    Measured from *horizontal* gradient only. Letters are built from vertical
    strokes and light it up; the colour step where one row band meets the next
    runs the full width of the strip and has no horizontal gradient at all, so it
    contributes nothing. Using an isotropic contrast measure instead makes every
    band boundary look like a letter.

    Reduced across the row by a high percentile rather than a maximum, so the
    board's own left border - a hot column in every row - is likewise ignored.
    """
    gx = np.abs(cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    if gx.shape[1] < 8:
        return gx.max(axis=1)

    # Suppress columns that are hot in most rows - the board's left border and any
    # rule inside the strip - then the row maximum is the letter and nothing else.
    persistent = np.median(gx, axis=0)
    return np.clip(gx - persistent[None, :], 0.0, None).max(axis=1)


def row_anchors(gray: np.ndarray, x_end: int) -> tuple[list[float], float] | None:
    """Centres of the player initials in the strip left of ``x_end``.

    Returns ``(centres, pitch)``. The lane number sits above the table in the
    same strip and shows up here too, so the four blobs that form an evenly
    spaced run are selected and anything else discarded.
    """
    if x_end < 8:
        return None
    strip = gray[:, :x_end]
    if strip.size == 0:
        return None

    spans = _blob_spans(_ink_by_row(strip), cfg.ANCHOR_REL_THRESHOLD, cfg.ANCHOR_MIN_RUN)
    centres = [(a + b) / 2.0 for a, b in spans]
    if not cfg.N_PLAYER_ROWS <= len(centres) <= cfg.ANCHOR_MAX_BLOBS:
        return None

    run = uniform_run(centres, tolerance=cfg.ANCHOR_SPACING_TOLERANCE)
    if len(run) < cfg.N_PLAYER_ROWS:
        return None
    run = run[: cfg.N_PLAYER_ROWS]
    return run, float(np.median(np.diff(run)))


# ----------------------------------------------------------------------- entry


def find_table(gray: np.ndarray) -> TableGeometry | None:
    """Locate the table, or return ``None`` if this image does not hold one."""
    height, width = gray.shape[:2]

    comb = _best_comb(column_profile(gray))
    if comb is None:
        return None
    score, contrast, period, offset = comb

    teeth = [offset + k * period for k in range(int((width - 1 - offset) / period) + 1)]
    if len(teeth) < cfg.N_FRAMES + 1:
        return None

    # Phase and row anchors are solved together. The initials column is whatever
    # lies left of the first column rule, and it is the only column holding ink
    # on the throw sub-rows but none on the score sub-rows - so a strip cut at
    # the correct rule yields exactly four evenly spaced blobs.
    #
    # Cut too far right and the strip reaches into frame one, whose score cells
    # add four more blobs that merge into one mass. Cut too far left and the
    # letters are missed. The largest tooth that still gives a clean four is the
    # first rule.
    for index in range(min(cfg.PHASE_MAX_CANDIDATES, len(teeth) - cfg.N_FRAMES), -1, -1):
        cut = teeth[index]
        if cut < cfg.ANCHOR_MIN_RUN:
            continue
        anchors = row_anchors(gray, int(cut))
        if anchors is None:
            continue
        centres, pitch = anchors

        rules = teeth[index:index + cfg.N_FRAMES + 1]
        if len(rules) < cfg.N_FRAMES + 1:
            continue

        # The row pitch and the column period both describe the same table, so a
        # wildly different aspect means the anchors are not really initials.
        if not (cfg.MIN_PITCH_OVER_PERIOD <= pitch / period <= cfg.MAX_PITCH_OVER_PERIOD):
            continue
        if centres[-1] + pitch > height * 1.25:
            continue

        return TableGeometry(
            col_rules=list(rules),
            row_centres=centres,
            period=period,
            row_pitch=pitch,
            comb_score=score,
            comb_contrast=contrast,
        )
    return None

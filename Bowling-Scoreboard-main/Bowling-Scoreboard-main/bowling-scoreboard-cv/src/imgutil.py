"""Small image helpers shared across the pipeline stages."""

from __future__ import annotations

import cv2
import numpy as np

from . import config as cfg


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def runs(flags: np.ndarray | list[bool]) -> list[tuple[int, int]]:
    """Contiguous ``True`` spans of a boolean sequence as ``(start, end_exclusive)``."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(flags):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(flags)))
    return spans


def bright_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels that are locally brighter than their surroundings.

    The grid rules and the glyphs are both light-on-dark over most of the board,
    so a local contrast threshold picks up the table structure without needing a
    global brightness assumption that the yellow active row would break.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -6
    )


def line_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isolate long horizontal and long vertical strokes."""
    h, w = gray.shape[:2]
    mask = bright_mask(gray)

    h_len = max(12, int(w * cfg.H_KERNEL_FRAC))
    v_len = max(12, int(h * cfg.V_KERNEL_FRAC))

    h_mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    )
    v_mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    )
    return h_mask, v_mask


def count_lines(mask: np.ndarray, axis: int) -> int:
    """Count distinct line runs in a morphology mask.

    ``axis=0`` collapses rows (counts horizontal lines), ``axis=1`` collapses
    columns (counts vertical lines).
    """
    profile = (mask > 0).sum(axis=1 if axis == 0 else 0).astype(np.float32)
    if profile.max() <= 0:
        return 0
    hot = profile > 0.35 * profile.max()
    return len(runs(hot))


def uniform_run(centres: list[float], tolerance: float | None = None) -> list[float]:
    """Longest run of evenly spaced positions.

    The frame columns are equal width, so their rules form an arithmetic
    sequence. Picking the longest such run discards a window border or a stray
    edge that happens to survive the morphology.
    """
    tolerance = cfg.COL_SPACING_TOLERANCE if tolerance is None else tolerance
    n = len(centres)
    if n < 3:
        return []

    best: list[float] = []
    for i in range(n):
        for j in range(i + 3, n + 1):
            gaps = np.diff(centres[i:j])
            median = float(np.median(gaps))
            if median <= 0:
                continue
            if np.all(np.abs(gaps - median) <= tolerance * median):
                if j - i > len(best):
                    best = centres[i:j]
    return best

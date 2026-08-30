"""Stage 3 - turn a rectified cell into a left-to-right list of glyph bitmaps.

Cells come in two polarities: white text on blue for idle players, and dark text
on yellow/white for the player who is up. Rather than special-casing the active
row, the background is inferred per cell from its border pixels, which works for
either polarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as cfg
from .imgutil import runs, to_gray

# A cell whose pixels are this uniform holds no text.
_EMPTY_STD = 7.0


@dataclass
class Glyph:
    """One connected mark inside a cell."""

    bitmap: np.ndarray                 # GLYPH_SIZE square, float32 in [0, 1]
    box: tuple[int, int, int, int]     # x, y, w, h within the cell crop
    area: int

    @property
    def cx(self) -> float:
        return self.box[0] + self.box[2] / 2.0


def binarize_fg(img: np.ndarray) -> np.ndarray:
    """Return a 0/255 mask of the *text* in ``img``, whatever its polarity.

    Otsu splits the crop into two classes; whichever class dominates the border
    ring is the background, so the other class is the ink.
    """
    gray = to_gray(img)
    if gray.size == 0 or float(gray.std()) < _EMPTY_STD:
        return np.zeros(gray.shape, dtype=np.uint8)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ring = np.ones(gray.shape, dtype=bool)
    b = max(1, min(gray.shape) // 8)
    ring[b:-b, b:-b] = False
    if not ring.any():
        ring[:] = True

    # If the border is mostly "white" under this threshold, the ink is the dark
    # class, so invert.
    if (binary[ring] > 0).mean() > 0.5:
        binary = cv2.bitwise_not(binary)

    # Close single-pixel gaps that video compression opens up in thin strokes.
    return cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    """Scale a cropped glyph mask into a fixed square, preserving aspect ratio.

    Aspect ratio has to survive normalisation or "1" and "-" become the same
    picture, so the glyph is fitted into the square rather than stretched to it.
    """
    size = cfg.GLYPH_SIZE
    canvas = np.zeros((size, size), dtype=np.float32)
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return canvas

    inner = size - 4
    scale = inner / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)

    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized.astype(np.float32) / 255.0
    return canvas


Box = tuple[int, int, int, int]


def _is_flat_mark(box: Box) -> bool:
    """True for a wide, short mark - the board's "-" for a miss."""
    _, _, w, h = box
    return h >= 2 and w >= cfg.FLAT_MARK_MIN_W and w >= cfg.FLAT_MARK_ASPECT * h


def _contained(a: Box, b: Box) -> bool:
    """True when one box sits almost entirely inside the other."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    smaller = min(aw * ah, bw * bh)
    return smaller > 0 and inter >= cfg.CONTAIN_FRAC * smaller


def _merge_contained(boxes: list[Box]) -> list[Box]:
    """Union boxes where one nests inside the other.

    The board draws a split as a digit inside a circle: two components occupying
    the same space, which have to become one glyph. Testing *containment* rather
    than horizontal overlap keeps neighbouring digits apart - a narrow "1" beside
    a "5" overlaps very little in area even when their x-spans touch.
    """
    boxes = sorted(boxes, key=lambda b: -(b[2] * b[3]))     # largest first
    out: list[Box] = []
    for box in boxes:
        for i, kept in enumerate(out):
            if _contained(box, kept):
                x = min(kept[0], box[0])
                y = min(kept[1], box[1])
                out[i] = (
                    x, y,
                    max(kept[0] + kept[2], box[0] + box[2]) - x,
                    max(kept[1] + kept[3], box[1] + box[3]) - y,
                )
                break
        else:
            out.append(box)
    return sorted(out, key=lambda b: b[0])


def _split_wide(mask: np.ndarray, box: Box) -> list[Box]:
    """Cut a component that is too wide to be a single glyph.

    Blur plus the morphological close can fuse touching digits into one
    component. Rather than guessing how many glyphs are inside from the aspect
    ratio - unreliable, because glyph widths vary a lot between characters and
    fonts - the cuts are placed where the ink profile genuinely thins out. The
    threshold is relaxed in steps so a lightly fused pair still separates.
    """
    x, y, w, h = box
    if h <= 0 or w / h <= cfg.SPLIT_MIN_RATIO:
        return [box]

    profile = (mask[y:y + h, x:x + w] > 0).sum(axis=0).astype(np.float64)
    median = float(np.median(profile))
    if median <= 0:
        return [box]

    min_part = max(cfg.MIN_GLYPH_W_PX, int(round(cfg.SPLIT_MIN_PART_FRAC * h)))
    if w < 2 * min_part:
        return [box]

    cuts: list[int] = []
    for fraction in cfg.SPLIT_VALLEY_FRACS:
        cuts = _valley_cuts(profile, median * fraction, min_part)
        if cuts:
            break
    if not cuts:
        return [box]

    parts: list[Box] = []
    for a, b in zip([0] + cuts, cuts + [w]):
        sub = mask[y:y + h, x + a:x + b]
        ys, xs = np.nonzero(sub)
        if xs.size == 0 or (xs.max() - xs.min() + 1) < cfg.MIN_GLYPH_W_PX:
            continue
        parts.append(
            (x + a + int(xs.min()), y + int(ys.min()),
             int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        )
    return parts or [box]


def _valley_cuts(profile: np.ndarray, threshold: float, min_part: int) -> list[int]:
    """Column indices where the ink profile dips below ``threshold``.

    Only interior dips count, and consecutive cuts are kept at least
    ``min_part`` apart so a single ragged valley yields one cut, not several.
    """
    width = len(profile)
    cuts: list[int] = []
    for start, end in runs(profile <= threshold):
        centre = (start + end - 1) // 2
        if centre < min_part or centre > width - 1 - min_part:
            continue
        if cuts and centre - cuts[-1] < min_part:
            continue
        cuts.append(int(centre))
    return cuts


def extract_glyphs(cell: np.ndarray) -> list[Glyph]:
    """Segment a cell crop into its glyphs, ordered left to right."""
    h, w = cell.shape[:2]
    if h < 6 or w < 4:
        return []

    pad_y = int(round(h * cfg.CELL_PAD))
    pad_x = int(round(w * cfg.CELL_PAD))
    crop = cell[pad_y:h - pad_y, pad_x:w - pad_x]
    if crop.size == 0:
        return []

    mask = binarize_fg(crop)
    if not mask.any():
        return []

    ch = crop.shape[0]
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    tall: list[Box] = []
    flat: list[Box] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        box = (int(x), int(y), int(bw), int(bh))
        if area < cfg.MIN_GLYPH_AREA_PX or bw < cfg.MIN_GLYPH_W_PX:
            continue
        if cfg.MIN_GLYPH_H_FRAC * ch <= bh <= cfg.MAX_GLYPH_H_FRAC * ch:
            tall.append(box)
        elif _is_flat_mark(box):
            flat.append(box)

    boxes: list[Box] = []
    for box in _merge_contained(tall):
        boxes.extend(_split_wide(mask, box))
    boxes.extend(flat)

    glyphs: list[Glyph] = []
    for x, y, bw, bh in boxes:
        sub = mask[y:y + bh, x:x + bw]
        if sub.size == 0:
            continue
        glyphs.append(
            Glyph(
                bitmap=normalize_glyph(sub),
                box=(x + pad_x, y + pad_y, bw, bh),
                area=int((sub > 0).sum()),
            )
        )
    return sorted(glyphs, key=lambda g: g.cx)

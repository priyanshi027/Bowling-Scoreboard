"""Stage 4 - classify glyph bitmaps against a labelled template bank.

The board draws from a tiny, fixed alphabet in a single fixed font, so a
nearest-neighbour match against reference bitmaps is both more accurate and far
cheaper than a general-purpose OCR engine - and it needs no model downloads.

The bank is produced once per capture by ``calibrate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfg
from .segment import Glyph

UNKNOWN = "?"

# A mean-absolute-difference above this means the glyph does not resemble any
# template well enough to name, regardless of how the runner-up scored.
MAX_GLYPH_DIST = 0.28


@dataclass
class Prediction:
    label: str
    confidence: float
    distance: float


class TemplateBank:
    """Labelled reference bitmaps plus nearest-neighbour lookup."""

    def __init__(self, labels: list[str], matrix: np.ndarray):
        if len(labels) != matrix.shape[0]:
            raise ValueError("labels and matrix disagree on template count")
        self.labels = labels
        self.matrix = matrix.astype(np.float32)

    # ------------------------------------------------------------- persistence

    @classmethod
    def load(cls, path: str | Path) -> "TemplateBank":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no template bank at {path}\n"
                "Build one first:  python main.py calibrate --video <path>"
            )
        data = np.load(path, allow_pickle=False)
        return cls(list(data["labels"]), data["matrix"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=np.array(self.labels), matrix=self.matrix)

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, np.ndarray]]) -> "TemplateBank":
        labels = [label for label, _ in pairs]
        matrix = np.stack([bmp.astype(np.float32).ravel() for _, bmp in pairs])
        return cls(labels, matrix)

    # ---------------------------------------------------------- classification

    def predict(self, bitmap: np.ndarray) -> Prediction:
        """Nearest template, with a normalised-margin confidence."""
        vec = bitmap.astype(np.float32).ravel()
        if vec.size != self.matrix.shape[1]:
            raise ValueError("glyph size does not match the template bank")

        dists = np.abs(self.matrix - vec[None, :]).mean(axis=1)
        best = int(np.argmin(dists))
        label = self.labels[best]
        d_best = float(dists[best])

        if d_best > MAX_GLYPH_DIST:
            return Prediction(UNKNOWN, 0.0, d_best)

        other = [d for d, lbl in zip(dists, self.labels) if lbl != label]
        d_other = float(min(other)) if other else 1.0
        confidence = (d_other - d_best) / (d_other + d_best + 1e-6)
        return Prediction(label, float(np.clip(confidence, 0.0, 1.0)), d_best)


def classify_cell(bank: TemplateBank, glyphs: list[Glyph]) -> tuple[str, float]:
    """Read a whole cell as a string plus a mean confidence.

    Glyphs below ``MIN_CONFIDENCE`` are emitted as ``?`` rather than dropped, so
    a misread stays visible to the validation stage instead of silently changing
    the length of the string.
    """
    if not glyphs:
        return "", 1.0                       # a confidently empty cell

    labels: list[str] = []
    confs: list[float] = []
    for glyph in glyphs:
        pred = bank.predict(glyph.bitmap)
        if pred.confidence < cfg.MIN_CONFIDENCE:
            labels.append(UNKNOWN)
            confs.append(0.0)
        else:
            labels.append(pred.label)
            confs.append(pred.confidence)

    return "".join(labels), float(np.mean(confs))

"""Video input helpers: metadata probing and frame iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def describe(self) -> str:
        return (
            f"{self.path.name}: {self.width}x{self.height} @ {self.fps:.2f} fps, "
            f"{self.n_frames} frames ({self.duration_s:.1f}s)"
        )


def probe(path: str | Path) -> VideoInfo:
    """Read basic stream metadata without decoding the whole file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")
    try:
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path,
    every: int = 1,
    start: int = 0,
    limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` pairs.

    Sequential decoding with a skip counter rather than ``CAP_PROP_POS_FRAMES``
    seeking, which is unreliable on inter-frame-compressed files.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")

    emitted = 0
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx < start or (idx - start) % every:
                continue
            yield idx, frame
            emitted += 1
            if limit is not None and emitted >= limit:
                break
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    """Decode a single frame by index (used by the ``preview`` command)."""
    for _, frame in iter_frames(path, every=1, start=index, limit=1):
        return frame
    raise IndexError(f"frame {index} is past the end of {path}")

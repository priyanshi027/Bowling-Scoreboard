"""Stage 5 - fuse per-frame readings into one answer.

Two things make this more than averaging:

* Any single frame can be misread because of compression noise or because an
  inset window is covering part of the table, so readings are pooled across
  frames and voted on by confidence.
* The board is *live*: throws appear as the game is bowled, so the most common
  value over the whole clip is not the final value. Cells are therefore resolved
  from a window at the end of the clip, and the transitions are kept separately
  as a timeline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import config as cfg


@dataclass
class Observation:
    frame_index: int
    text: str
    confidence: float


@dataclass
class Reading:
    """The resolved value of one cell."""

    text: str
    confidence: float
    share: float                 # winning fraction of the confidence-weighted vote
    n_observations: int
    first_seen: int | None       # frame index where this value first appeared


@dataclass
class Transition:
    """A cell taking on a new value partway through the clip."""

    key: str
    frame_index: int
    old: str
    new: str


class Aggregator:
    """Collects per-frame cell readings and resolves them."""

    def __init__(self, tail_frames: int = 12, min_persist: int = 2):
        # How many of the most recent board frames define the "final" state, and
        # how many consecutive observations a new value needs before it is
        # believed rather than treated as a one-frame glitch.
        self.tail_frames = tail_frames
        self.min_persist = min_persist
        self._obs: dict[str, list[Observation]] = defaultdict(list)
        self._frames: list[int] = []

    # ------------------------------------------------------------------ input

    def add(self, key: str, text: str, confidence: float, frame_index: int) -> None:
        self._obs[key].append(Observation(frame_index, text, confidence))

    def note_frame(self, frame_index: int) -> None:
        """Record that ``frame_index`` was a usable board frame."""
        if not self._frames or self._frames[-1] != frame_index:
            self._frames.append(frame_index)

    @property
    def board_frames(self) -> list[int]:
        return list(self._frames)

    # ----------------------------------------------------------------- output

    def resolve(self, key: str) -> Reading:
        """Vote on one cell, weighting by confidence and favouring the endgame."""
        obs = self._obs.get(key, [])
        if not obs:
            return Reading("", 1.0, 1.0, 0, None)

        cutoff = self._tail_cutoff()
        window = [o for o in obs if o.frame_index >= cutoff] or obs

        weights: dict[str, float] = defaultdict(float)
        for o in window:
            # Every observation carries a small floor weight so that a cell read
            # with low confidence in every frame still resolves to something.
            weights[o.text] += 0.1 + o.confidence

        total = sum(weights.values())
        text, weight = max(weights.items(), key=lambda kv: kv[1])
        share = weight / total if total else 0.0

        if share < cfg.VOTE_MIN_SHARE and len(weights) > 1:
            # No clear winner: prefer the most recent reading over a coin flip.
            text = window[-1].text
            weight = weights[text]
            share = weight / total if total else 0.0

        matching = [o for o in window if o.text == text]
        confidence = sum(o.confidence for o in matching) / len(matching)
        first = next((o.frame_index for o in obs if o.text == text), None)

        return Reading(text, confidence, share, len(obs), first)

    def resolve_all(self) -> dict[str, Reading]:
        return {key: self.resolve(key) for key in self._obs}

    def transitions(self) -> list[Transition]:
        """Value changes that persisted, in chronological order."""
        out: list[Transition] = []
        for key, obs in self._obs.items():
            current = ""
            run_value: str | None = None
            run_len = 0
            for o in obs:
                if o.text == run_value:
                    run_len += 1
                else:
                    run_value, run_len = o.text, 1
                if run_len == self.min_persist and run_value != current:
                    out.append(Transition(key, o.frame_index, current, run_value))
                    current = run_value
        return sorted(out, key=lambda t: (t.frame_index, t.key))

    # ---------------------------------------------------------------- internals

    def _tail_cutoff(self) -> int:
        if not self._frames:
            return 0
        tail = self._frames[-self.tail_frames:]
        return tail[0]

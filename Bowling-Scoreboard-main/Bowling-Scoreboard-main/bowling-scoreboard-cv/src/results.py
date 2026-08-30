"""The extraction result and its serialisable form."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .aggregate import Reading, Transition
from .bowling import PlayerScore
from .video import VideoInfo


@dataclass
class ExtractionResult:
    video: VideoInfo
    players: list[PlayerScore] = field(default_factory=list)
    lane: str = ""
    bowler: str = ""
    readings: dict[str, Reading] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    frames_total: int = 0
    frames_with_board: int = 0
    frames_occluded: int = 0
    last_board_frame: int | None = None

    # ------------------------------------------------------------------ metrics

    @property
    def detection_rate(self) -> float:
        return self.frames_with_board / self.frames_total if self.frames_total else 0.0

    @property
    def mean_glyph_confidence(self) -> float:
        vals = [r.confidence for r in self.readings.values() if r.text]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def rules_agreement(self) -> float:
        """Share of checkable frames where printed and computed totals matched."""
        checked = sum(p.checked for p in self.players)
        if not checked:
            return 1.0
        bad = sum(len(p.mismatched_frames) for p in self.players)
        return (checked - bad) / checked

    @property
    def frames_checked(self) -> int:
        return sum(p.checked for p in self.players)

    # ------------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": {
                "file": self.video.path.name,
                "resolution": f"{self.video.width}x{self.video.height}",
                "fps": round(self.video.fps, 3),
                "frames": self.video.n_frames,
                "duration_s": round(self.video.duration_s, 2),
            },
            "scoreboard": {
                "lane": self.lane,
                "bowler_up": self.bowler,
                "players": [self._player_dict(p) for p in self.players],
            },
            "quality": {
                "frames_sampled": self.frames_total,
                "frames_with_scoreboard": self.frames_with_board,
                "detection_rate": round(self.detection_rate, 4),
                "frames_with_occlusion": self.frames_occluded,
                "mean_glyph_confidence": round(self.mean_glyph_confidence, 4),
                "frames_cross_checked": self.frames_checked,
                "scoring_rule_agreement": round(self.rules_agreement, 4),
            },
            "timeline": [
                {
                    "frame": t.frame_index,
                    "time_s": round(t.frame_index / self.video.fps, 2)
                    if self.video.fps else None,
                    "cell": t.key,
                    "from": t.old,
                    "to": t.new,
                }
                for t in self.transitions
            ],
        }

    @staticmethod
    def _player_dict(player: PlayerScore) -> dict[str, Any]:
        return {
            "initial": player.initial,
            "frames": [
                {
                    "frame": f.index,
                    "throws": f.raw,
                    "rolls": f.rolls,
                    "split": f.split,
                    "complete": f.complete,
                    "frame_score": f.frame_score,
                    "running_total": f.cumulative,
                    "printed_total": f.printed,
                }
                for f in player.frames
            ],
            "total_settled": player.total,
            "total_shown": player.provisional_total,
            "total_printed": player.printed_total,
            "total_agrees": player.total_agrees,
            "mismatched_frames": player.mismatched_frames,
            "repaired_frames": player.repaired_frames,
            "frame_agreement": round(player.agreement, 4),
        }

    def csv_rows(self) -> list[list[str]]:
        header = (
            ["player"]
            + [f"f{i}_throws" for i in range(1, 11)]
            + [f"f{i}_total" for i in range(1, 11)]
            + ["total_shown", "total_printed", "totals_agree"]
        )
        rows = [header]
        for p in self.players:
            rows.append(
                [p.initial]
                + [f.raw for f in p.frames]
                + ["" if f.cumulative is None else str(f.cumulative) for f in p.frames]
                + [
                    "" if p.provisional_total is None else str(p.provisional_total),
                    "" if p.printed_total is None else str(p.printed_total),
                    "" if p.total_agrees is None else str(p.total_agrees).lower(),
                ]
            )
        return rows

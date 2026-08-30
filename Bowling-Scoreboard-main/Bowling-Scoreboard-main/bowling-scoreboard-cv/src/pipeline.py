"""Pipeline orchestration - one pass over the video producing one result.

    frame -> detect table -> rectify -> cell grid -> segment glyphs
          -> classify -> vote across frames -> apply bowling rules -> report
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from . import bowling
from . import config as cfg
from . import grid as grid_mod
from . import report as report_mod
from . import segment
from . import video as video_mod
from .aggregate import Aggregator
from .classify import UNKNOWN, TemplateBank, classify_cell
from .detect import box_overlap_frac, detect_board
from .results import ExtractionResult

Progress = Callable[[str], None]

LANE_KEY = "title.lane"
BOWLER_KEY = "title.bowler"


def read_title(title: np.ndarray | None, bank: TemplateBank) -> tuple[str, str]:
    """Read the lane number and the name of the bowler who is up.

    The lane number sits above the initials column and the name to its right, so
    a single x threshold separates them.
    """
    if title is None or title.size == 0:
        return "", ""

    glyphs = segment.extract_glyphs(title)
    if not glyphs:
        return "", ""

    boundary = cfg.INITIAL_COL_FRAC * title.shape[1]
    lane: list[str] = []
    name: list[str] = []
    for glyph in glyphs:
        pred = bank.predict(glyph.bitmap)
        if pred.confidence < cfg.MIN_CONFIDENCE or pred.label == UNKNOWN:
            continue
        (lane if glyph.cx < boundary else name).append(pred.label)

    return "".join(lane), "".join(name)


def extract(
    video_path: str | Path,
    bank: TemplateBank,
    every: int = cfg.SAMPLE_EVERY_N_FRAMES,
    limit: int | None = None,
    annotate_path: str | Path | None = None,
    repair: bool = True,
    tail_frames: int = 12,
    progress: Progress | None = None,
) -> ExtractionResult:
    """Run the whole pipeline over ``video_path``."""
    info = video_mod.probe(video_path)
    result = ExtractionResult(video=info)
    agg = Aggregator(tail_frames=tail_frames)

    writer: report_mod.AnnotatedVideo | None = None
    out_size: tuple[int, int] | None = None

    try:
        for index, frame in video_mod.iter_frames(video_path, every=every, limit=limit):
            result.frames_total += 1
            det = detect_board(frame)

            gmap = None
            texts: dict[str, str] = {}

            if det.ok and det.warp is not None:
                result.frames_with_board += 1
                result.last_board_frame = index
                if det.occluders:
                    result.frames_occluded += 1

                agg.note_frame(index)
                gmap = grid_mod.build_grid(det.warp, det.geometry is not None)

                for cell in gmap.cells:
                    if any(
                        box_overlap_frac(cell.box, occ) > cfg.OCCLUDER_CELL_COVER
                        for occ in det.occluders
                    ):
                        continue
                    glyphs = segment.extract_glyphs(grid_mod.crop(det.warp, cell))
                    text, confidence = classify_cell(bank, glyphs)
                    agg.add(cell.key, text, confidence, index)
                    texts[cell.key] = text

                lane, bowler = read_title(det.title, bank)
                agg.add(LANE_KEY, lane, 1.0 if lane else 0.0, index)
                agg.add(BOWLER_KEY, bowler, 1.0 if bowler else 0.0, index)

            if annotate_path is not None:
                if writer is None:
                    out_size = report_mod.output_size(frame)
                    writer = report_mod.AnnotatedVideo(
                        annotate_path, out_size, info.fps / max(every, 1)
                    )
                assert out_size is not None
                writer.add(report_mod.compose_frame(frame, det, gmap, texts, out_size))

            if progress and result.frames_total % 25 == 0:
                progress(
                    f"  frame {index:>5}  "
                    f"board in {result.frames_with_board}/{result.frames_total}"
                )
    finally:
        if writer is not None:
            writer.close()

    if result.frames_with_board == 0:
        raise RuntimeError(
            "the scoreboard was never detected in this video.\n"
            "Inspect a single frame to see why:\n"
            "  python main.py preview --video <path> --frame 30 --debug"
        )

    _finish(result, agg, repair=repair)
    return result


def _finish(result: ExtractionResult, agg: Aggregator, repair: bool) -> None:
    """Resolve the votes and apply the scoring rules."""
    result.readings = agg.resolve_all()
    result.transitions = [
        t for t in agg.transitions() if t.key not in (LANE_KEY, BOWLER_KEY)
    ]
    result.lane = result.readings.get(LANE_KEY, _blank()).text
    result.bowler = result.readings.get(BOWLER_KEY, _blank()).text

    def text(key: str) -> str:
        reading = result.readings.get(key)
        return reading.text if reading else ""

    for p in range(cfg.N_PLAYER_ROWS):
        throws = [text(f"p{p}.f{f}.throws") for f in range(1, cfg.N_FRAMES + 1)]
        scores = [text(f"p{p}.f{f}.score") for f in range(1, cfg.N_FRAMES + 1)]
        result.players.append(
            bowling.build_player(
                initial=text(f"p{p}.initial"),
                throw_texts=throws,
                score_texts=scores,
                total_text=text(f"p{p}.total"),
                repair=repair,
            )
        )


def _blank():
    from .aggregate import Reading

    return Reading("", 0.0, 0.0, 0, None)


def preview(video_path: str | Path, frame_index: int, bank: TemplateBank | None,
            debug_dir: str | Path | None = None) -> dict[str, object]:
    """Inspect a single frame. Used to diagnose detection or grid problems."""
    frame = video_mod.read_frame(video_path, frame_index)
    det = detect_board(frame)

    gmap = grid_mod.build_grid(det.warp, det.geometry is not None) if det.ok and det.warp is not None else None
    texts: dict[str, str] = {}
    lane = bowler = ""

    if gmap is not None and bank is not None and det.warp is not None:
        for cell in gmap.cells:
            glyphs = segment.extract_glyphs(grid_mod.crop(det.warp, cell))
            texts[cell.key], _ = classify_cell(bank, glyphs)
        lane, bowler = read_title(det.title, bank)

    written: list[Path] = []
    if debug_dir is not None:
        written = report_mod.save_stage_images(debug_dir, frame, det, gmap, texts)

    return {
        "frame_index": frame_index,
        "detected": det.ok,
        "reason": det.reason,
        "h_lines": det.n_h_lines,
        "v_lines": det.n_v_lines,
        "occluders": det.occluders,
        "cols_detected": gmap.cols_detected if gmap else False,
        "rows_detected": gmap.rows_detected if gmap else False,
        "col_edges": gmap.col_edges if gmap else [],
        "row_edges": gmap.row_edges if gmap else [],
        "lane": lane,
        "bowler": bowler,
        "texts": texts,
        "debug_images": written,
    }

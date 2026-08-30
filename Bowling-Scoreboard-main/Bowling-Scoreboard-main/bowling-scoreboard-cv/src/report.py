"""Stage 7 - human- and machine-readable output.

Produces the four artefacts the write-up needs: a console scoreboard, a JSON and
CSV dump, an annotated video showing the detection running, and a set of labelled
stills of each pipeline stage for the documentation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg
from . import grid as grid_mod
from . import segment
from .bowling import tokenize
from .detect import BoardDetection
from .imgutil import line_masks
from .grid import GridMap
from .results import ExtractionResult

GREEN = (0, 220, 0)
RED = (0, 0, 235)
YELLOW = (0, 220, 255)
WHITE = (245, 245, 245)
GREY = (120, 120, 120)
PANEL_BG = (28, 20, 14)
PANEL_RULE = (72, 60, 48)


# --------------------------------------------------------------- text rendering


def pretty_throws(raw: str) -> str:
    """Render a throws cell for display, e.g. ``8o1`` -> ``(8)1``."""
    return "".join(f"({t[:-1]})" if t.endswith("o") else t for t in tokenize(raw))


def render_table(result: ExtractionResult) -> str:
    """The scoreboard as a fixed-width table."""
    cw = 7
    lines: list[str] = []

    head = f"Lane {result.lane or '?'}".ljust(12) + f"Now bowling: {result.bowler or '?'}"
    lines.append(head)

    header = "     |" + "|".join(str(i).center(cw) for i in range(1, 11)) + "|" + "TTL".center(cw)
    rule = "-" * len(header)
    lines.append(rule)
    lines.append(header)
    lines.append(rule)

    for player in result.players:
        throws = "|".join(pretty_throws(f.raw).center(cw) for f in player.frames)
        total = player.provisional_total
        lines.append(
            f" {player.initial or '?':<3} |" + throws + "|"
            + ("" if total is None else str(total)).center(cw)
        )

        def cell(value: int | None) -> str:
            return ("" if value is None else str(value)).center(cw)

        lines.append("     |" + "|".join(cell(f.cumulative) for f in player.frames)
                     + "|" + " " * cw)
        lines.append(rule)

    lines.append("")
    lines.append(
        f"scoreboard detected in {result.frames_with_board}/{result.frames_total} "
        f"sampled frames ({result.detection_rate:.0%})"
        + (f", {result.frames_occluded} partly occluded" if result.frames_occluded else "")
    )
    lines.append(f"mean glyph confidence   {result.mean_glyph_confidence:.3f}")
    lines.append(
        f"scoring-rule cross-check {result.rules_agreement:.1%} "
        f"of {result.frames_checked} frames where the board printed a running total"
    )
    for player in result.players:
        if player.mismatched_frames:
            lines.append(
                f"  ! {player.initial}: frames {player.mismatched_frames} disagreed with "
                f"the rules; corrected to the computed value"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------ file output


def write_json(result: ExtractionResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(result: ExtractionResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(result.csv_rows())
    return path


def write_text(result: ExtractionResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_table(result) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------- drawing


def draw_detection(frame: np.ndarray, det: BoardDetection) -> np.ndarray:
    """Original frame with the detected table outlined."""
    out = frame.copy()
    if det.ok and det.quad is not None:
        cv2.polylines(out, [det.quad.astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
        label = "SCOREBOARD DETECTED"
        colour = GREEN
    else:
        label = f"NO SCOREBOARD ({det.reason})"
        colour = RED
    cv2.putText(out, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    return out


def draw_grid(warp: np.ndarray, gmap: GridMap) -> np.ndarray:
    """Rectified table with the derived cell boundaries drawn on."""
    out = warp.copy()
    h, w = out.shape[:2]
    for x in gmap.col_edges:
        cv2.line(out, (x, 0), (x, h - 1), GREEN, 1, cv2.LINE_AA)
    for y in gmap.row_edges:
        cv2.line(out, (0, y), (w - 1, y), GREEN, 1, cv2.LINE_AA)

    tag = ("cols:detected" if gmap.cols_detected else "cols:fallback") + "  " + (
        "rows:detected" if gmap.rows_detected else "rows:fallback"
    )
    cv2.putText(out, tag, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)
    return out


def draw_readings(warp: np.ndarray, gmap: GridMap, texts: dict[str, str],
                  occluders: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """Render the extracted values as a clean table.

    Drawn on a flat background rather than over a dimmed copy of the source, so
    the panel shows what was *read* rather than a double exposure of the board.
    """
    h, w = warp.shape[:2]
    out = np.full((h, w, 3), PANEL_BG, dtype=np.uint8)

    for x in gmap.col_edges:
        cv2.line(out, (x, 0), (x, h - 1), PANEL_RULE, 1)
    for y in gmap.row_edges:
        cv2.line(out, (0, y), (w - 1, y), PANEL_RULE, 1)

    # Frame numbers along the top, above the first player row.
    for f in range(1, cfg.N_FRAMES + 1):
        x0, x1 = gmap.col_edges[f], gmap.col_edges[f + 1]
        label = str(f)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(out, label, (x0 + (x1 - x0 - tw) // 2, max(14, gmap.row_edges[0] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)

    for occ in occluders or []:
        x, y, bw, bh = occ
        cv2.rectangle(out, (x, y), (x + bw, y + bh), RED, 2)
        cv2.putText(out, "occluded", (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, RED, 1, cv2.LINE_AA)

    for cell in gmap.cells:
        text = texts.get(cell.key, "")
        if not text:
            continue
        x, y, bw, bh = cell.box
        shown = pretty_throws(text) if cell.role == grid_mod.THROWS else text
        colour = YELLOW if cell.role in (grid_mod.THROWS, grid_mod.INITIAL) else WHITE
        scale = 0.62 if bw > 70 else 0.48
        (tw, th), _ = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(out, shown, (x + max(2, (bw - tw) // 2), y + (bh + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
    return out


def compose_frame(frame: np.ndarray, det: BoardDetection, gmap: GridMap | None,
                  texts: dict[str, str], out_size: tuple[int, int]) -> np.ndarray:
    """Stack the annotated input over the annotated extraction, at a fixed size.

    A fixed output size is required because every frame goes to the same video
    writer, including frames where no board was found.
    """
    out_w, out_h = out_size
    top = draw_detection(frame, det)
    top = cv2.resize(top, (out_w, int(out_w * top.shape[0] / top.shape[1])))

    panel_h = out_h - top.shape[0]
    if panel_h <= 0:
        return cv2.resize(top, (out_w, out_h))

    if det.ok and det.warp is not None and gmap is not None:
        panel = draw_readings(det.warp, gmap, texts, det.occluders)
        panel = cv2.resize(panel, (out_w, panel_h))
    else:
        panel = np.zeros((panel_h, out_w, 3), dtype=np.uint8)
        cv2.putText(panel, "no scoreboard in this frame", (24, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREY, 2, cv2.LINE_AA)

    return np.vstack([top, panel])


def output_size(frame: np.ndarray, width: int = 1280) -> tuple[int, int]:
    top_h = int(width * frame.shape[0] / frame.shape[1])
    panel_h = int(width * cfg.CANON_H / cfg.CANON_W)
    return width, top_h + panel_h


class AnnotatedVideo:
    """Fixed-size MP4 writer for the demo clip."""

    def __init__(self, path: str | Path, size: tuple[int, int], fps: float):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.size = size
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open video writer for {path}")

    def add(self, image: np.ndarray) -> None:
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size)
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> "AnnotatedVideo":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------- debug stage dump


def save_stage_images(out_dir: str | Path, frame: np.ndarray, det: BoardDetection,
                      gmap: GridMap | None, texts: dict[str, str] | None = None) -> list[Path]:
    """Write one image per pipeline stage - the screenshots for the write-up."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(name: str, image: np.ndarray) -> None:
        path = out / name
        cv2.imwrite(str(path), image)
        written.append(path)

    save("01_input_frame.png", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_mask, v_mask = line_masks(gray)
    structure = cv2.cvtColor(cv2.bitwise_or(h_mask, v_mask), cv2.COLOR_GRAY2BGR)
    structure[(h_mask > 0)] = GREEN
    structure[(v_mask > 0)] = YELLOW
    save("02_table_structure.png", structure)

    save("03_detected_board.png", draw_detection(frame, det))

    if det.ok and det.warp is not None:
        save("04_rectified_table.png", det.warp)
        if gmap is not None:
            save("05_cell_grid.png", draw_grid(det.warp, gmap))
            save("06_binarised_cells.png", _cell_montage(det.warp, gmap))
            if texts:
                save("07_extracted_values.png",
                     draw_readings(det.warp, gmap, texts, det.occluders))
    return written


def _cell_montage(warp: np.ndarray, gmap: GridMap, max_cells: int = 24) -> np.ndarray:
    """Grid of segmented cells above their binarised masks."""
    tile_w, tile_h = 110, 56
    cells = [c for c in gmap.cells if c.role in (grid_mod.THROWS, grid_mod.SCORE)][:max_cells]
    cols = 6
    rows = (len(cells) + cols - 1) // cols

    sheet = np.full((rows * tile_h * 2, cols * tile_w, 3), 25, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        crop = grid_mod.crop(warp, cell)
        if crop.size == 0:
            continue
        mask = segment.binarize_fg(crop)
        y0 = r * tile_h * 2
        x0 = c * tile_w
        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = cv2.resize(crop, (tile_w, tile_h))
        sheet[y0 + tile_h:y0 + 2 * tile_h, x0:x0 + tile_w] = cv2.cvtColor(
            cv2.resize(mask, (tile_w, tile_h)), cv2.COLOR_GRAY2BGR
        )
    return sheet

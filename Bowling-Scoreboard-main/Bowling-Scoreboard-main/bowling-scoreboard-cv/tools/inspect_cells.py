#!/usr/bin/env python3
"""Dump per-cell segmentation detail for a single frame.

    python tools/inspect_cells.py --video data/synthetic.mp4 --frame 5
    python tools/inspect_cells.py --video data/synthetic.mp4 --frame 5 --cells p0.f1.score

Prints one line per glyph with its bounding box, and optionally writes each
cell's crop and binarised mask so the segmentation can be eyeballed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import grid as grid_mod  # noqa: E402
from src import segment, video  # noqa: E402
from src.detect import detect_board  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/synthetic.mp4")
    ap.add_argument("--frame", type=int, default=5)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="cell keys to inspect (default: all non-empty)")
    ap.add_argument("--dump", default=None, help="directory for crop/mask images")
    args = ap.parse_args()

    frame = video.read_frame(args.video, args.frame)
    det = detect_board(frame)
    if not det.ok or det.warp is None:
        print(f"no board: {det.reason}")
        return 1

    gmap = grid_mod.build_grid(det.warp)
    dump = Path(args.dump) if args.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    for cell in gmap.cells:
        if args.cells and cell.key not in args.cells:
            continue
        crop = grid_mod.crop(det.warp, cell)
        glyphs = segment.extract_glyphs(crop)
        if not glyphs and not args.cells:
            continue

        boxes = " ".join(f"(x={g.box[0]},y={g.box[1]},w={g.box[2]},h={g.box[3]})"
                         for g in glyphs)
        print(f"{cell.key:<18} {cell.box!s:<28} n={len(glyphs)}  {boxes}")

        if dump:
            cv2.imwrite(str(dump / f"{cell.key}_crop.png"), crop)
            cv2.imwrite(str(dump / f"{cell.key}_mask.png"), segment.binarize_fg(crop))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

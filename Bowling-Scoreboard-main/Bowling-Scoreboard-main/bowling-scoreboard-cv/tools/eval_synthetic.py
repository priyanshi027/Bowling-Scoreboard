#!/usr/bin/env python3
"""Measure accuracy on the synthetic clip, where ground truth is known.

Reports two numbers that mean different things:

* per-frame accuracy - how often a single frame's reading of a cell is right.
  This is the raw strength of detection, segmentation and classification.
* final accuracy - how often the aggregated answer is right, after voting across
  frames. This is what the pipeline actually outputs, and it should be higher.

The gap between the two is what the temporal voting stage buys.

    python tools/eval_synthetic.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg  # noqa: E402
from src import grid as grid_mod  # noqa: E402
from src import pipeline, segment, video  # noqa: E402
from src.classify import TemplateBank, classify_cell  # noqa: E402
from src.detect import box_overlap_frac, detect_board  # noqa: E402

import make_synthetic  # noqa: E402
from build_synthetic_bank import expected_cells  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/synthetic.mp4")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()

    bank = TemplateBank.load(Path(args.templates) / "bank.npz")

    # ---- per-frame accuracy
    right = wrong = 0
    errors: Counter[str] = Counter()

    for index, frame in video.iter_frames(args.video, every=args.every):
        state = make_synthetic.state_at(index)
        if not state["has_board"]:
            continue
        det = detect_board(frame)
        if not det.ok or det.warp is None:
            continue

        wanted = expected_cells(bool(state["late"]))
        gmap = grid_mod.build_grid(det.warp)
        for cell in gmap.cells:
            if any(box_overlap_frac(cell.box, occ) > cfg.OCCLUDER_CELL_COVER
                   for occ in det.occluders):
                continue
            truth = wanted.get(cell.key, "")
            got, _ = classify_cell(bank, segment.extract_glyphs(grid_mod.crop(det.warp, cell)))
            if got == truth:
                right += 1
            else:
                wrong += 1
                errors[f"{cell.key}: expected {truth!r} got {got!r}"] += 1

    total = right + wrong
    print(f"per-frame cell accuracy   {right}/{total} = {right / total:.2%}")
    if errors:
        print(f"\n{len(errors)} distinct per-frame error(s), most frequent first:")
        for message, count in errors.most_common(12):
            print(f"  x{count:<4} {message}")

    # ---- final aggregated accuracy
    result = pipeline.extract(args.video, bank, every=args.every)
    final = expected_cells(True)
    f_right = f_wrong = 0
    f_errors: list[str] = []
    for key, truth in final.items():
        got = result.readings.get(key)
        text = got.text if got else ""
        if text == truth:
            f_right += 1
        else:
            f_wrong += 1
            f_errors.append(f"{key}: expected {truth!r} got {text!r}")

    f_total = f_right + f_wrong
    print(f"\nfinal cell accuracy       {f_right}/{f_total} = {f_right / f_total:.2%}")
    for message in f_errors:
        print(f"  {message}")

    print(f"\nmean glyph confidence     {result.mean_glyph_confidence:.3f}")
    print(f"scoring-rule agreement    {result.rules_agreement:.2%} "
          f"over {result.frames_checked} checkable frames")
    return 0 if f_wrong == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a template bank for the synthetic clip from its known ground truth.

Only usable on ``tools/make_synthetic.py`` output. Because the content of every
cell is known, each segmented glyph can be labelled by its position instead of
by shape matching: a cell expected to read "8o1" contributes its first glyph as
"8o" and its second as "1". That is exact, and it doubles as a check on the
segmenter - a cell that yields the wrong number of glyphs is reported rather
than quietly mislabelled.

Real footage uses the manual pass instead (``main.py calibrate``), because there
the board's font is not known in advance.

    python tools/build_synthetic_bank.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402
from src import grid as grid_mod  # noqa: E402
from src import segment, video  # noqa: E402
from src.bowling import tokenize  # noqa: E402
from src.classify import TemplateBank  # noqa: E402
from src.detect import box_overlap_frac, detect_board  # noqa: E402

import make_synthetic  # noqa: E402  (same directory)

TEMPLATES_PER_LABEL = 8
WITHIN_LABEL_DIST = 0.05


def expected_cells(late: bool) -> dict[str, str]:
    """Cell key -> the text that cell shows, for the given clip state.

    ``late`` selects the state after player T's fourth throw has appeared, which
    also changes T's printed total. Resolving this per frame matters: T's total
    cell is the only one drawn white-on-red, so skipping it would leave the bank
    with no template for a digit in that rendering.
    """
    out: dict[str, str] = {}
    for p, (initial, throws, totals) in enumerate(make_synthetic.PLAYERS):
        active = p == make_synthetic.ACTIVE_ROW
        if active and late:
            throws = throws + [make_synthetic.T_LATE_THROW]

        out[f"p{p}.initial"] = initial
        for f, throw in enumerate(throws, start=1):
            out[f"p{p}.f{f}.throws"] = throw
        for f, total in enumerate(totals, start=1):
            out[f"p{p}.f{f}.score"] = str(total)

        if active:
            out[f"p{p}.total"] = str(
                make_synthetic.T_LATE_TOTAL if late else totals[-1]
            )
        else:
            out[f"p{p}.total"] = str(totals[-1]) if totals else ""
    return out


def collect_labelled(video_path: str, every: int) -> tuple[list[tuple[str, np.ndarray]],
                                                           list[str]]:
    """Gather ``(label, bitmap)`` pairs plus a list of segmentation mismatches."""
    pairs: list[tuple[str, np.ndarray]] = []
    problems: list[str] = []
    title_done = False

    for index, frame in video.iter_frames(video_path, every=every):
        state = make_synthetic.state_at(index)
        if not state["has_board"]:
            continue

        det = detect_board(frame)
        if not det.ok or det.warp is None:
            continue

        wanted = expected_cells(bool(state["late"]))
        gmap = grid_mod.build_grid(det.warp)

        for cell in gmap.cells:
            text = wanted.get(cell.key)
            if not text:
                continue
            if any(box_overlap_frac(cell.box, occ) > cfg.OCCLUDER_CELL_COVER
                   for occ in det.occluders):
                continue

            tokens = tokenize(text)
            glyphs = segment.extract_glyphs(grid_mod.crop(det.warp, cell))
            if len(glyphs) != len(tokens):
                problems.append(
                    f"frame {index} {cell.key}: expected {tokens} "
                    f"({len(tokens)} glyphs) but segmented {len(glyphs)}"
                )
                continue
            for token, glyph in zip(tokens, glyphs):
                pairs.append((token, glyph.bitmap))

        # The title strip: lane number then the bowler's name. It never changes,
        # so one frame is enough.
        if not title_done and det.title is not None and det.title.size:
            tokens = list(make_synthetic.LANE + make_synthetic.BOWLER)
            glyphs = segment.extract_glyphs(det.title)
            if len(glyphs) == len(tokens):
                pairs.extend(zip(tokens, (g.bitmap for g in glyphs)))
                title_done = True
            else:
                problems.append(
                    f"frame {index} title: expected {len(tokens)} glyphs, "
                    f"got {len(glyphs)}"
                )

    return pairs, problems


def reduce_to_templates(pairs: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Keep the most common distinct variants of each label.

    The same character is drawn several ways on this board - white on blue, dark
    on yellow, white on red - and blur makes split digits differ again. Every
    such variant needs a template, so all instances are grouped and the largest
    groups are kept. Stopping as soon as N distinct shapes are seen would bias
    the bank towards whichever cells happen to be visited first and leave later
    variants unmatched.
    """
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, bitmap in pairs:
        by_label[label].append(bitmap)

    out: list[tuple[str, np.ndarray]] = []
    for label, bitmaps in sorted(by_label.items()):
        groups: list[dict[str, object]] = []
        for bitmap in bitmaps:
            best_i, best_d = -1, 1e9
            for i, group in enumerate(groups):
                d = float(np.abs(group["exemplar"] - bitmap).mean())
                if d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0 and best_d <= WITHIN_LABEL_DIST:
                groups[best_i]["total"] = groups[best_i]["total"] + bitmap
                groups[best_i]["count"] = int(groups[best_i]["count"]) + 1
            else:
                groups.append({"exemplar": bitmap.copy(),
                               "total": bitmap.copy(), "count": 1})

        groups.sort(key=lambda g: -int(g["count"]))
        for group in groups[:TEMPLATES_PER_LABEL]:
            out.append((label, np.asarray(group["total"]) / int(group["count"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/synthetic.mp4")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--strict", action="store_true",
                    help="fail if any cell segmented to the wrong glyph count")
    args = ap.parse_args()

    pairs, problems = collect_labelled(args.video, args.every)
    if not pairs:
        print("error: no labelled glyphs collected", file=sys.stderr)
        return 1

    if problems:
        print(f"{len(problems)} segmentation mismatch(es):")
        for line in problems[:12]:
            print(f"  {line}")
        if len(problems) > 12:
            print(f"  ... and {len(problems) - 12} more")
        if args.strict:
            return 1
    else:
        print("segmentation matched the expected glyph count in every cell")

    templates = reduce_to_templates(pairs)
    bank = TemplateBank.from_pairs(templates)

    path = Path(args.templates) / "bank.npz"
    bank.save(path)

    counts: dict[str, int] = defaultdict(int)
    for label, _ in templates:
        counts[label] += 1
    summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"\n{len(pairs)} labelled glyphs -> {len(templates)} templates")
    print(f"per label: {summary}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

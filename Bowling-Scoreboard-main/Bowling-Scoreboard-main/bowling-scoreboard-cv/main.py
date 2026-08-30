#!/usr/bin/env python3
"""Scoreboard data extraction from video - command line entry point.

    python main.py probe     --video data/bowling_scoreboard.mp4
    python main.py calibrate --video data/bowling_scoreboard.mp4
    python main.py calibrate --finalize
    python main.py extract   --video data/bowling_scoreboard.mp4
    python main.py preview    --video data/bowling_scoreboard.mp4 --frame 30 --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import calibrate as calibrate_mod       # noqa: E402
from src import config as cfg                    # noqa: E402
from src import pipeline, report, video          # noqa: E402
from src.classify import TemplateBank            # noqa: E402

DEFAULT_VIDEO = ROOT / "data" / "bowling_scoreboard.mp4"
TEMPLATES_DIR = ROOT / "templates"
OUTPUT_DIR = ROOT / "output"


def log(message: str = "") -> None:
    print(message, flush=True)


# ------------------------------------------------------------------- commands


def cmd_probe(args: argparse.Namespace) -> int:
    log(video.probe(args.video).describe())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.finalize:
        bank_path, labels = calibrate_mod.finalize(TEMPLATES_DIR)
        log(f"wrote {bank_path.relative_to(ROOT)}")
        log(f"alphabet ({len(labels)}): {' '.join(labels)}")
        log("")
        log("Now run:  python main.py extract --video <path>")
        return 0

    log(f"scanning {Path(args.video).name} for glyph shapes ...")
    n, mosaic, labels_path = calibrate_mod.run_clustering(
        args.video, TEMPLATES_DIR, every=args.every
    )
    log(f"found {n} recurring shapes")
    log("")
    log("Next:")
    log(f"  1. open  {mosaic.relative_to(ROOT)}")
    log(f"  2. type each shape's character into {labels_path.relative_to(ROOT)}")
    log("     digits, X (strike), / (spare), - (miss), '8o' for a circled 8,")
    log("     capital letters for names. Leave noise clusters blank.")
    log("  3. run  python main.py calibrate --finalize")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    bank = TemplateBank.load(TEMPLATES_DIR / calibrate_mod.BANK_FILE)
    info = video.probe(args.video)
    log(info.describe())
    log(f"sampling every {args.every} frame(s), {len(bank.labels)} templates loaded")
    log("")

    out_dir = Path(args.out)
    annotate = None if args.no_video else out_dir / "annotated.mp4"

    result = pipeline.extract(
        args.video,
        bank,
        every=args.every,
        limit=args.limit,
        annotate_path=annotate,
        repair=not args.no_repair,
        progress=log if args.verbose else None,
    )

    log(report.render_table(result))
    log("")

    json_path = report.write_json(result, out_dir / "scoreboard.json")
    csv_path = report.write_csv(result, out_dir / "scoreboard.csv")
    txt_path = report.write_text(result, out_dir / "scoreboard.txt")

    written = [json_path, csv_path, txt_path]
    if annotate is not None:
        written.append(annotate)

    if args.stills:
        # The last frame that showed the board: the final state, and never a
        # frame caught mid-change.
        idx = result.last_board_frame
        if idx is None:
            idx = info.n_frames // 2
        stage_info = pipeline.preview(args.video, idx, bank, out_dir / "stages")
        log(f"stage stills taken from frame {idx}")
        written.extend(stage_info["debug_images"])  # type: ignore[arg-type]

    log("written:")
    for path in written:
        log(f"  {Path(path).relative_to(ROOT)}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    bank = None
    bank_path = TEMPLATES_DIR / calibrate_mod.BANK_FILE
    if bank_path.exists():
        bank = TemplateBank.load(bank_path)
    else:
        log("(no template bank yet - showing detection and grid only)")

    debug_dir = Path(args.out) / "stages" if args.debug else None
    info = pipeline.preview(args.video, args.frame, bank, debug_dir)

    log(f"frame {info['frame_index']}")
    log(f"  detected      {info['detected']}  ({info['reason']})")
    log(f"  lines         {info['h_lines']} horizontal, {info['v_lines']} vertical")
    log(f"  columns       {'detected' if info['cols_detected'] else 'fallback'}")
    log(f"  rows          {'detected' if info['rows_detected'] else 'fallback'}")
    log(f"  occluders     {info['occluders']}")
    if bank is not None:
        log(f"  lane / bowler {info['lane']!r} / {info['bowler']!r}")
        log("")
        for p in range(cfg.N_PLAYER_ROWS):
            throws = [
                info["texts"].get(f"p{p}.f{f}.throws", "")  # type: ignore[union-attr]
                for f in range(1, cfg.N_FRAMES + 1)
            ]
            scores = [
                info["texts"].get(f"p{p}.f{f}.score", "")  # type: ignore[union-attr]
                for f in range(1, cfg.N_FRAMES + 1)
            ]
            initial = info["texts"].get(f"p{p}.initial", "?")  # type: ignore[union-attr]
            log(f"  {initial:<2} throws {throws}")
            log(f"     scores {scores}")

    for path in info["debug_images"]:  # type: ignore[union-attr]
        log(f"  wrote {Path(path).relative_to(ROOT)}")
    return 0


# ----------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Extract bowling scoreboard data from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_video(p: argparse.ArgumentParser) -> None:
        p.add_argument("--video", default=str(DEFAULT_VIDEO),
                       help="input video (default: data/bowling_scoreboard.mp4)")

    p_probe = sub.add_parser("probe", help="print video metadata")
    add_video(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    p_cal = sub.add_parser("calibrate", help="build the glyph template bank")
    add_video(p_cal)
    p_cal.add_argument("--every", type=int, default=4,
                       help="sample every Nth frame while clustering (default: 4)")
    p_cal.add_argument("--finalize", action="store_true",
                       help="turn the labelled clusters into templates/bank.npz")
    p_cal.set_defaults(func=cmd_calibrate)

    p_ext = sub.add_parser("extract", help="extract the scoreboard")
    add_video(p_ext)
    p_ext.add_argument("--every", type=int, default=cfg.SAMPLE_EVERY_N_FRAMES,
                       help="sample every Nth frame (default: %(default)s)")
    p_ext.add_argument("--limit", type=int, default=None,
                       help="stop after this many sampled frames")
    p_ext.add_argument("--out", default=str(OUTPUT_DIR), help="output directory")
    p_ext.add_argument("--no-video", action="store_true",
                       help="skip writing output/annotated.mp4")
    p_ext.add_argument("--no-repair", action="store_true",
                       help="report rule mismatches without correcting them")
    p_ext.add_argument("--stills", action="store_true",
                       help="also write per-stage stills for the write-up")
    p_ext.add_argument("--verbose", action="store_true", help="print progress")
    p_ext.set_defaults(func=cmd_extract)

    p_prev = sub.add_parser("preview", help="inspect one frame")
    add_video(p_prev)
    p_prev.add_argument("--frame", type=int, default=0, help="frame index")
    p_prev.add_argument("--out", default=str(OUTPUT_DIR), help="output directory")
    p_prev.add_argument("--debug", action="store_true",
                        help="write per-stage images to <out>/stages")
    p_prev.set_defaults(func=cmd_preview)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

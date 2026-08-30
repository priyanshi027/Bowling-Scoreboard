"""End-to-end pipeline test against a synthetic clip with known ground truth.

Covers detection, rectification, grid mapping, glyph segmentation,
classification, temporal voting and the scoring cross-check together. The clip
and its template bank are generated once per session into a temporary directory,
so the test needs no binary fixtures in the repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src import pipeline  # noqa: E402
from src.classify import TemplateBank  # noqa: E402

import make_synthetic  # noqa: E402


def _run(script: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / script), *args],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        raise AssertionError(f"{script} failed:\n{result.stdout}\n{result.stderr}")


@pytest.fixture(scope="session")
def clip(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    workdir = tmp_path_factory.mktemp("synthetic")
    video_path = workdir / "synthetic.mp4"
    make_synthetic.build(video_path)

    templates = workdir / "templates"
    templates.mkdir()
    _run("build_synthetic_bank.py", "--video", str(video_path),
         "--templates", str(templates), "--strict")

    bank = TemplateBank.load(templates / "bank.npz")
    result = pipeline.extract(video_path, bank, every=2)
    return {"video": video_path, "bank": bank, "result": result,
            "templates": templates}


@pytest.fixture(scope="session")
def result(clip: dict[str, object]):
    return clip["result"]


# ------------------------------------------------------------------- detection


def test_board_detected_in_most_frames(result):
    # The clip contains 10 frames of pin animation with no table at all.
    assert result.frames_with_board >= 40
    assert result.detection_rate >= 0.85


def test_pin_animation_frames_are_rejected(clip):
    # Frame 35 falls inside the animation-only stretch.
    info = pipeline.preview(clip["video"], 35, clip["bank"])
    assert info["detected"] is False


def test_grid_is_derived_from_the_image_not_the_fallback(clip):
    info = pipeline.preview(clip["video"], 5, clip["bank"])
    assert info["detected"] is True
    assert info["cols_detected"] is True
    assert info["rows_detected"] is True


def test_inset_window_is_reported_as_an_occluder(clip):
    # Frames 60-74 have a pin-animation window pasted over the table.
    info = pipeline.preview(clip["video"], 66, clip["bank"])
    assert info["detected"] is True
    assert len(info["occluders"]) >= 1


# ---------------------------------------------------------------------- title


def test_lane_and_bowler_are_read(result):
    assert result.lane == make_synthetic.LANE
    assert result.bowler == make_synthetic.BOWLER


# ------------------------------------------------------------------ scoreboard


@pytest.mark.parametrize(
    "index,initial,throws,totals,shown",
    [
        (0, "J", ["X", "5-", "-7", "4-"], [15, 20, 27, 31], 31),
        (1, "V", ["8-", "3-", "71", "8o1"], [8, 11, 19, 28], 28),
        (2, "P", ["X", "4/", "9-", "6-"], [20, 39, 48, 54], 54),
        (3, "T", ["61", "1/", "8-", "3"], [7, 25, 33], 36),
    ],
)
def test_player_row_matches_ground_truth(result, index, initial, throws, totals, shown):
    player = result.players[index]
    assert player.initial == initial
    assert [f.raw for f in player.frames[: len(throws)]] == throws
    assert [f.cumulative for f in player.frames[: len(totals)]] == totals
    assert player.provisional_total == shown
    assert player.printed_total == shown
    assert player.total_agrees is True


def test_unplayed_frames_are_blank(result):
    for player in result.players:
        for frame in player.frames[4:]:
            assert frame.raw == ""
            assert frame.rolls == []
            assert frame.cumulative is None


def test_split_is_flagged_from_the_circled_digit(result):
    v = result.players[1]
    assert v.frames[3].split is True
    assert [f.split for f in v.frames[:3]] == [False, False, False]


def test_live_board_resolves_to_the_final_state(result):
    # Player T's fourth throw only appears partway through the clip; the result
    # must reflect the end state, not the most common state.
    assert result.players[3].frames[3].raw == "3"
    assert result.players[3].provisional_total == 36


def test_late_throw_shows_up_in_the_timeline(result):
    changes = [t for t in result.transitions if t.key == "p3.f4.throws"]
    assert changes, "expected a recorded transition for the late throw"
    assert changes[-1].new == "3"


# ------------------------------------------------------------------- quality


def test_scoring_rules_agree_everywhere(result):
    assert result.frames_checked >= 12
    assert result.rules_agreement == 1.0
    for player in result.players:
        assert player.mismatched_frames == []
        assert player.repaired_frames == []


def test_glyph_confidence_is_healthy(result):
    assert result.mean_glyph_confidence > 0.5


# ---------------------------------------------------------------- serialisation


def test_json_shape(result):
    payload = result.to_dict()
    assert payload["scoreboard"]["lane"] == make_synthetic.LANE
    assert len(payload["scoreboard"]["players"]) == 4
    assert payload["quality"]["scoring_rule_agreement"] == 1.0
    j = payload["scoreboard"]["players"][0]
    assert j["initial"] == "J"
    assert j["frames"][0]["rolls"] == [10]
    assert j["frames"][0]["running_total"] == 15


def test_csv_rows(result):
    rows = result.csv_rows()
    assert rows[0][0] == "player"
    assert [r[0] for r in rows[1:]] == ["J", "V", "P", "T"]
    assert rows[1][1] == "X"                 # J, frame 1 throws


# ----------------------------------------------------------- calibration path


def test_manual_calibration_path_produces_a_usable_bank(clip, tmp_path):
    """The unsupervised clustering pass should isolate the alphabet.

    The synthetic board draws 22 distinct characters. Clusters may split a
    character across several entries, which is harmless, so this only asserts a
    sane lower bound and that the artefacts round-trip.
    """
    from src import calibrate

    templates = tmp_path / "cal"
    n_clusters, mosaic, labels = calibrate.run_clustering(
        clip["video"], templates, every=6
    )
    assert n_clusters >= 22
    assert mosaic.exists() and labels.exists()

    with pytest.raises(ValueError, match="no labels filled in"):
        calibrate.finalize(templates)


# --------------------------------------------------------------- robustness
#
# These guard the two failure modes that would break the pipeline on real
# footage whose layout differs from the reference frames. Both were live defects
# during development: a board with no horizontal rules was not detected at all,
# and a board whose column widths differed from the config fractions read at
# barely one percent accuracy.


def _shifted_layout() -> dict[str, float]:
    """Geometry that deliberately disagrees with the fractions in config."""
    base = make_synthetic.geometry()
    alt = dict(base)
    alt["init_w"] = base["table_w"] * 0.17        # config says 0.116
    alt["ttl_w"] = base["table_w"] * 0.12         # config says 0.085
    alt["frame_w"] = (base["table_w"] - alt["init_w"] - alt["ttl_w"]) / 10
    alt["header_h"] = base["table_h"] * 0.12      # config says 0.075
    alt["sub_h"] = (base["table_h"] - alt["header_h"]) / 8
    return alt


def _segmentation_agreement(image) -> float:
    """Share of cells that segment to the expected number of glyphs.

    Measures the grid on its own, independently of whether the template bank
    happens to recognise the glyphs.
    """
    from src import grid as grid_mod
    from src import segment
    from src.bowling import tokenize
    from src.detect import detect_board

    sys.path.insert(0, str(ROOT / "tools"))
    from build_synthetic_bank import expected_cells

    det = detect_board(image)
    assert det.ok, f"board not detected: {det.reason}"
    gmap = grid_mod.build_grid(det.warp)
    truth = expected_cells(True)

    hits = 0
    for cell in gmap.cells:
        expected = tokenize(truth.get(cell.key, ""))
        found = segment.extract_glyphs(grid_mod.crop(det.warp, cell))
        hits += len(found) == len(expected)
    return hits / len(gmap.cells)


def test_board_with_colour_banding_only_is_still_detected():
    """Rows separated by colour alone, with no horizontal rules drawn.

    Detection must not depend on horizontal rules, because the column rules are
    the reliable signal and some boards draw no long horizontal edges at all.
    """
    from src import grid as grid_mod
    from src.detect import detect_board

    image = make_synthetic.render_board(True, draw_h_rules=False)
    det = detect_board(image)
    assert det.ok, det.reason
    assert det.n_h_lines < cfg_min_h(), "this case is only meaningful without h rules"

    gmap = grid_mod.build_grid(det.warp)
    assert gmap.cols_detected is True
    assert gmap.rows_detected is True
    assert _segmentation_agreement(image) >= 0.95


def cfg_min_h() -> int:
    from src import config

    return config.MIN_H_LINES


def test_grid_adapts_to_a_layout_that_disagrees_with_config():
    """Column widths unlike the reference frames must still map correctly.

    The column edges are read off the table's own rules rather than assumed from
    the config fractions, so changing those fractions in the *image* must not
    move the cells.
    """
    from src import grid as grid_mod
    from src.detect import detect_board

    layout = _shifted_layout()
    image = make_synthetic.render_board(True, draw_h_rules=False, layout=layout)

    det = detect_board(image)
    assert det.ok, det.reason
    gmap = grid_mod.build_grid(det.warp)
    assert gmap.cols_detected is True
    assert _segmentation_agreement(image) >= 0.95


def test_shifted_layout_with_rules_is_mostly_correct():
    """Same shifted layout, but with horizontal rules present.

    This path takes a different outline and is the weaker of the two: the glyph
    splitter over-segments a few stretched score cells. Pinned at its measured
    level so a regression shows up, and documented as a known limit.
    """
    image = make_synthetic.render_board(True, draw_h_rules=True,
                                        layout=_shifted_layout())
    assert _segmentation_agreement(image) >= 0.85


@pytest.mark.parametrize("name", ["pins", "black", "noise", "flat"])
def test_non_board_images_are_rejected(name):
    """Loosening the detection test must not admit false positives."""
    import numpy as np

    from src.detect import detect_board

    images = {
        "pins": make_synthetic.render_pins(make_synthetic.W, make_synthetic.H),
        "black": np.zeros((720, 1280, 3), dtype=np.uint8),
        "noise": (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype("uint8"),
        "flat": np.full((720, 1280, 3), (200, 90, 25), dtype=np.uint8),
    }
    det = detect_board(images[name])
    assert det.ok is False, f"{name} was wrongly detected as a scoreboard"


# ------------------------------------------------------- real capture fixtures
#
# Frames lifted from the actual bowling_scoreboard.mp4 (browser screenshots of
# the player, with the control overlay cropped off). These are the only real
# pixels available, and they currently FAIL, which is the single most important
# fact about this pipeline. Marked xfail rather than deleted so the failure stays
# visible and flips to a pass when detection is fixed.

REAL_DIR = ROOT / "data" / "real"


@pytest.mark.skipif(not (REAL_DIR / "board3.png").exists(),
                    reason="real capture fixtures not present")
@pytest.mark.xfail(strict=False,
                   reason="the real board has no drawn horizontal rules and its "
                          "column rules are too faint for the binarised line mask; "
                          "see src/structure.py for the investigation")
@pytest.mark.parametrize("name", ["board3.png", "board5.png"])
def test_real_capture_frames_are_detected(name):
    import cv2

    from src.detect import detect_board

    det = detect_board(cv2.imread(str(REAL_DIR / name)))
    assert det.ok, det.reason


@pytest.mark.skipif(not (REAL_DIR / "board4.png").exists(),
                    reason="real capture fixtures not present")
def test_real_animation_frame_is_rejected():
    """The pin-animation frame from the real clip must not be taken for a board."""
    import cv2

    from src.detect import detect_board

    det = detect_board(cv2.imread(str(REAL_DIR / "board4.png")))
    assert det.ok is False

"""Scoring-rule tests.

The expected values in ``test_reference_frame_*`` are read straight off the
reference frame of the sample video, so these tests pin the scoring engine to
the ground truth the pipeline has to reproduce.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bowling  # noqa: E402


def cumulative(throws: list[str]) -> list[int | None]:
    frames = [bowling.parse_frame(t, i + 1) for i, t in enumerate(throws)]
    bowling.score_frames(frames)
    return [f.cumulative for f in frames]


def pad(throws: list[str]) -> list[str]:
    return throws + [""] * (bowling.N_FRAMES - len(throws))


# ----------------------------------------------------------------- tokenising


def test_tokenize_plain():
    assert bowling.tokenize("X") == ["X"]
    assert bowling.tokenize("5-") == ["5", "-"]
    assert bowling.tokenize("4/") == ["4", "/"]


def test_tokenize_circled_digit_is_one_token():
    assert bowling.tokenize("8o1") == ["8o", "1"]
    assert bowling.tokenize("1o") == ["1o"]


def test_split_flag_set_from_circled_digit():
    frame = bowling.parse_frame("8o1", 4)
    assert frame.split is True
    assert frame.rolls == [8, 1]


# --------------------------------------------------------------- throw parsing


def test_spare_mark_completes_the_rack():
    assert bowling.parse_frame("4/", 2).rolls == [4, 6]
    assert bowling.parse_frame("1/", 2).rolls == [1, 9]
    assert bowling.parse_frame("-/", 2).rolls == [0, 10]


def test_miss_is_zero():
    assert bowling.parse_frame("5-", 1).rolls == [5, 0]
    assert bowling.parse_frame("-7", 3).rolls == [0, 7]


def test_strike_is_a_single_roll():
    frame = bowling.parse_frame("X", 1)
    assert frame.rolls == [10]
    assert frame.complete is True


def test_unreadable_glyph_marks_frame_unreadable():
    frame = bowling.parse_frame("?1", 1)
    assert frame.readable is False
    assert frame.rolls == []


def test_impossible_frame_rejected():
    # Two throws cannot fell more than ten pins.
    assert bowling.parse_frame("77", 1).readable is False


def test_tenth_frame_variants():
    assert bowling.parse_frame("XXX", 10).rolls == [10, 10, 10]
    assert bowling.parse_frame("X7/", 10).rolls == [10, 7, 3]
    assert bowling.parse_frame("9/X", 10).rolls == [9, 1, 10]
    assert bowling.parse_frame("9-", 10).rolls == [9, 0]


def test_tenth_frame_completeness():
    assert bowling.parse_frame("9-", 10).complete is True
    assert bowling.parse_frame("9/", 10).complete is False    # bonus ball owed
    assert bowling.parse_frame("X", 10).complete is False


# ------------------------------------------------------------------- scoring


def test_perfect_game_is_300():
    assert cumulative(bowling.perfect_game())[-1] == 300


def test_all_spares_five_pin_is_150():
    assert cumulative(["5/"] * 9 + ["5/5"])[-1] == 150


def test_all_open_nines_is_90():
    assert cumulative(["9-"] * 10)[-1] == 90


def test_gutter_game_is_zero():
    assert cumulative(["--"] * 10)[-1] == 0


def test_pending_strike_has_no_total_yet():
    # A strike in frame 1 with nothing after it cannot be scored.
    assert cumulative(pad(["X"])) == [None] * 10


def test_frames_after_a_pending_frame_are_also_pending():
    assert cumulative(pad(["X", "X"])) == [None] * 10


# --------------------------------------------- ground truth from the sample video


def test_reference_frame_player_j():
    # J: X | 5- | -7 | 4-  ->  15 20 27 31
    assert cumulative(pad(["X", "5-", "-7", "4-"]))[:4] == [15, 20, 27, 31]


def test_reference_frame_player_v():
    # V: 8- | 3- | 71 | (8)1  ->  8 11 19 28
    assert cumulative(pad(["8-", "3-", "71", "8o1"]))[:4] == [8, 11, 19, 28]


def test_reference_frame_player_p():
    # P: X | 4/ | 9- | 6-  ->  20 39 48 54
    assert cumulative(pad(["X", "4/", "9-", "6-"]))[:4] == [20, 39, 48, 54]


def test_reference_frame_player_t_mid_frame():
    # T: 61 | 1/ | 8- | 3(in progress)  ->  7 25 33, TTL 36
    throws = pad(["61", "1/", "8-", "3"])
    assert cumulative(throws)[:3] == [7, 25, 33]

    player = bowling.build_player("T", throws, ["7", "25", "33"] + [""] * 7, "36")
    assert player.total == 33                 # settled frames only
    assert player.provisional_total == 36     # plus the 3 pins already down
    assert player.total_agrees is True


def test_reference_frame_player_j_totals_agree():
    player = bowling.build_player(
        "J", pad(["X", "5-", "-7", "4-"]), ["15", "20", "27", "31"] + [""] * 6, "31"
    )
    assert player.mismatched_frames == []
    assert player.provisional_total == 31
    assert player.total_agrees is True
    assert player.agreement == 1.0


# ---------------------------------------------------------------- validation


def test_misread_printed_total_is_repaired():
    # Throws say 15 after a strike + 5; pretend the printed digits read "16".
    player = bowling.build_player(
        "J", pad(["X", "5-"]), ["16", "20"] + [""] * 8, "20", repair=True
    )
    assert player.mismatched_frames == [1]
    assert player.repaired_frames == [1]
    assert player.frames[0].printed == 15      # corrected from the rules
    assert player.agreement < 1.0


def test_repair_can_be_disabled():
    player = bowling.build_player(
        "J", pad(["X", "5-"]), ["16", "20"] + [""] * 8, "20", repair=False
    )
    assert player.mismatched_frames == [1]
    assert player.repaired_frames == []
    assert player.frames[0].printed == 16      # left as read


def test_empty_row_is_not_an_error():
    player = bowling.build_player("Z", pad([]), [""] * 10, "")
    assert player.total is None
    assert player.provisional_total is None
    assert player.mismatched_frames == []
    assert player.agreement == 1.0

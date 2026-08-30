"""Tunable constants for the scoreboard extraction pipeline.

Everything that might need adjusting for a different capture lives here so the
algorithm modules stay free of magic numbers.

The fractions below were measured off reference frames of the sample video and
act as *seeds and fallbacks*: ``grid.py`` prefers boundaries it detects in the
image itself and only falls back to these numbers when detection is weak.
"""

from __future__ import annotations

# ---------------------------------------------------------------- board layout
# The table is rectified to this canonical size before any parsing, so every
# downstream measurement can be expressed in fixed pixel units. The aspect ratio
# matches the table in the sample capture (~2.65:1).
CANON_W = 1440
CANON_H = 544

# The canonical warp covers the table only: a column-header strip ("1 2 ... TTL")
# on top, then the player rows.
HEADER_HEIGHT_FRAC = 0.075   # header strip, fraction of canonical height

# 4 player rows below the header. Each is split into a "throws" sub-row on top
# and a running-"score" sub-row underneath, of equal height.
N_PLAYER_ROWS = 4
THROW_SUBROW_FRAC = 0.50

# 12 columns: player initial, frames 1..10, TTL.
N_FRAMES = 10
# Widths of the two outer columns, as fractions of the canonical width. Measured
# off real frames: the initials column is about 1.6 and the total column about
# 1.8 frame-column widths, i.e. 0.119 and 0.134 of the table. An earlier estimate
# of 0.085 for the total column was too small and clipped the TTL digits.
INITIAL_COL_FRAC = 0.125     # left column with the player's initial letter
TTL_COL_FRAC = 0.135         # right column with the running total

# The title strip above the table carries the lane number and the name of the
# bowler who is up. Height is expressed as a fraction of the table height.
TITLE_HEIGHT_FRAC = 0.13

# ------------------------------------------------------- detection thresholds
DETECT_SCALE = 640           # frames are downscaled to this width for detection
# The column rules carry the detection. Horizontal rules are optional, because
# some boards separate their rows by colour banding and draw almost no long
# horizontal edges; when they are present they give a more precise outline.
MIN_V_LINES = 6              # a board frame shows ~12 long vertical rules
MIN_UNIFORM_COLS = 6         # ...of which this many must be evenly spaced
COL_SPACING_TOLERANCE = 0.25  # how far a gap may stray from the median
MIN_H_LINES = 3              # at or above this, use the more precise outline

# Comb filter over the vertical-gradient projection - the primary way the column
# rules are found. The period range is bounded so the search cannot settle on a
# sub-multiple of the true spacing, and contrast is the comb's mean energy
# against the profile's own background level.
PROFILE_SMOOTH = 9                # narrow smoothing of the column profile
PROFILE_BASELINE = 101            # broad smoothing subtracted as the baseline
COMB_PERIOD_MIN_FRAC = 1 / 22.0   # of image width
COMB_PERIOD_MAX_FRAC = 1 / 9.0
COMB_PERIOD_STEP = 0.5
COMB_MIN_TEETH = 6
COMB_MIN_CONTRAST = 1.6           # reported, not used to reject
PHASE_MAX_CANDIDATES = 4          # leftmost teeth tried as the first column rule

# The row pitch and the column period describe the same table, so their ratio is
# bounded. Guards against mistaking arbitrary blobs for player initials.
# Measured at 1.14 on the real board and 1.09 on the synthetic one. Kept tight:
# a loose bound admits a lattice of *sub*-rows at the wrong phase, whose ratio is
# about half.
MIN_PITCH_OVER_PERIOD = 0.80
MAX_PITCH_OVER_PERIOD = 1.70

# Player-initial anchors, which fix the row geometry.
ANCHOR_REL_THRESHOLD = 0.45   # of peak local contrast
ANCHOR_MIN_RUN = 8            # rows, minimum height of an initial's blob
ANCHOR_SPACING_TOLERANCE = 0.20
ANCHOR_INK_PERCENTILE = 92    # per-row ink statistic; ignores border columns
# The initials column carries ink on the throw sub-rows only, so a correctly cut
# strip yields four blobs, or five once the lane number above the table is
# counted. A strip reaching into frame one picks up the score rows too and yields
# roughly nine - which is the signal that the cut is wrong.
ANCHOR_MAX_BLOBS = 6
H_KERNEL_FRAC = 0.35         # horizontal morphology kernel, fraction of width
V_KERNEL_FRAC = 0.20         # vertical morphology kernel, fraction of height
MIN_BOARD_AREA_FRAC = 0.15   # board must cover >=15% of the frame
MAX_SKEW_DEG = 12.0          # beyond this the quad is rejected as a bad fit
MIN_INITIALS_READ = 3        # player initials that must segment cleanly to
                             # accept a measured layout

# An inset animation window occludes part of the board in some frames. A blob of
# structure-free background spanning at least this many cells in both axes is
# treated as an occluder rather than as genuinely empty cells.
OCCLUDER_MIN_COLS = 2
OCCLUDER_MIN_ROWS = 2
OCCLUDER_CELL_COVER = 0.35   # a cell this covered by an occluder is skipped

# ------------------------------------------------------------- grid refinement
GRID_LINE_PROMINENCE = 0.30  # column-profile peak height, relative to max
GRID_COL_TOLERANCE = 0.35    # accept detected columns within this much of the
                             # expected uniform spacing (as a fraction of it)
EDGE_RULE_MARGIN = 0.015     # rules this close to the table edge are its border,
                             # not a column separator

# --------------------------------------------------------- glyph segmentation
GLYPH_SIZE = 28              # normalised glyph bitmaps are GLYPH_SIZE square
CELL_PAD = 0.06              # trim this fraction off each cell edge (grid rules)
MIN_GLYPH_H_FRAC = 0.28      # component height relative to cell height
MAX_GLYPH_H_FRAC = 1.00
MIN_GLYPH_W_PX = 2
MIN_GLYPH_AREA_PX = 12

# "-" is far shorter than a digit and would fail the height test, so wide flat
# marks are admitted on aspect ratio instead.
FLAT_MARK_ASPECT = 1.5       # width / height for a mark to count as flat
FLAT_MARK_MIN_W = 5

# A circled digit is drawn as a ring around a digit. The two components are
# rejoined when one box sits almost entirely inside the other - containment,
# rather than horizontal overlap, so that neighbouring digits are never merged.
CONTAIN_FRAC = 0.70

# Compression blur and the morphological close can fuse touching digits into one
# component. Anything materially wider than a single glyph is cut apart where its
# ink profile thins out, trying each threshold in turn until a cut is found.
SPLIT_MIN_RATIO = 1.15         # width / height above which a box is suspect
SPLIT_MIN_PART_FRAC = 0.20     # narrowest allowed piece, as a fraction of height
SPLIT_VALLEY_FRACS = (0.25, 0.40, 0.55)  # of the median ink column

# ------------------------------------------------------------- classification
# Glyph alphabet. A trailing "o" marks the circled form of a digit, which the
# Brunswick board uses to flag a split.
DIGITS = [str(d) for d in range(10)]
CIRCLED = [f"{d}o" for d in range(10)]
MARKS = ["X", "/", "-"]
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALPHABET = DIGITS + CIRCLED + MARKS + LETTERS

MIN_CONFIDENCE = 0.35        # below this a glyph is dropped as unreadable

# ---------------------------------------------------------------- calibration
# Clustering radius in normalised glyph space. Deliberately tight: measured
# distances between instances of the *same* character reach about 0.05, while
# genuinely different characters can come as close as 0.08 ("0" against "8",
# "T" against "7"). Splitting one character across two clusters costs one extra
# label and gives the classifier more templates, whereas merging two characters
# silently corrupts every reading - so the threshold errs tight.
CAL_CLUSTER_DIST = 0.07
CAL_MIN_CLUSTER = 4          # discard clusters seen fewer times than this
CAL_MAX_CLUSTERS = 96
CAL_MOSAIC_COLS = 8
CAL_TILE = 96                # mosaic tile size in pixels

# ------------------------------------------------------------------ aggregation
# The board is append-only: a cell that has been written never blanks out. A
# reading needs this share of the confidence-weighted vote to be accepted.
VOTE_MIN_SHARE = 0.30
SAMPLE_EVERY_N_FRAMES = 2    # default temporal subsampling for extraction

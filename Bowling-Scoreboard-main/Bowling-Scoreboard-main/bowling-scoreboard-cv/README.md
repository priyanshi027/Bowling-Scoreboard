# Bowling Scoreboard Extraction from Video

A computer-vision pipeline that watches a video of a ten-pin bowling scoreboard
and extracts the scoreboard as structured data — every player's throws, their
running totals, the lane number and who is up — then **verifies the reading
against the rules of bowling**.

Built with OpenCV and NumPy only. No OCR engine, no neural network, no model
downloads, no network access at run time.

```
Lane 6      Now bowling: TARUN
---------------------------------------------------------------------------------------------
     |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   10  |  TTL
---------------------------------------------------------------------------------------------
 J   |   X   |   5-  |   -7  |   4-  |       |       |       |       |       |       |   31
     |   15  |   20  |   27  |   31  |       |       |       |       |       |       |
---------------------------------------------------------------------------------------------
 V   |   8-  |   3-  |   71  |  (8)1 |       |       |       |       |       |       |   28
     |   8   |   11  |   19  |   28  |       |       |       |       |       |       |
---------------------------------------------------------------------------------------------
 P   |   X   |   4/  |   9-  |   6-  |       |       |       |       |       |       |   54
     |   20  |   39  |   48  |   54  |       |       |       |       |       |       |
---------------------------------------------------------------------------------------------
 T   |   61  |   1/  |   8-  |   3   |       |       |       |       |       |       |   36
     |   7   |   25  |   33  |       |       |       |       |       |       |       |
---------------------------------------------------------------------------------------------

scoreboard detected in 45/50 sampled frames (90%), 8 partly occluded
mean glyph confidence   0.862
scoring-rule cross-check 100.0% of 15 frames where the board printed a running total
```

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the clip at `data/bowling_scoreboard.mp4`, then:

```bash
# 1. confirm the file decodes
python main.py probe --video data/bowling_scoreboard.mp4

# 2. check the board is being found and the grid lands on the cells
python main.py preview --video data/bowling_scoreboard.mp4 --frame 30 --debug

# 3. one-time calibration (see "Calibration" below — about two minutes)
python main.py calibrate --video data/bowling_scoreboard.mp4
#    ... edit templates/labels.json ...
python main.py calibrate --finalize

# 4. extract
python main.py extract --video data/bowling_scoreboard.mp4 --stills
```

Outputs land in `output/`:

| File | Contents |
| --- | --- |
| `scoreboard.json` | full result: throws, rolls, per-frame scores, totals, quality metrics, change timeline |
| `scoreboard.csv` | flat one-row-per-player table |
| `scoreboard.txt` | the console table above |
| `annotated.mp4` | demo video: detection on top, extracted values below |
| `stages/01..07_*.png` | one still per pipeline stage, for the write-up |

### Try it with no video at all

The repository can generate its own test clip with known ground truth, which is
handy for verifying the install:

```bash
python tools/make_synthetic.py --out data/synthetic.mp4
python tools/build_synthetic_bank.py          # labels it from ground truth
python main.py extract --video data/synthetic.mp4
python tools/eval_synthetic.py                # accuracy report
```

---

## How it works

Seven stages. Each is a module in `src/` and can be inspected on its own.

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Detect | `detect.py` | Find the table, reject frames that have none, rectify it |
| 2. Map | `grid.py` | Locate the 12 columns and 8 sub-rows, build the cell map |
| 3. Segment | `segment.py` | Cut each cell into individual glyph bitmaps |
| 4. Classify | `classify.py` | Name each glyph by nearest-neighbour against a template bank |
| 5. Fuse | `aggregate.py` | Vote across frames; track how the board changed over time |
| 6. Verify | `bowling.py` | Recompute every total from the throws and cross-check |
| 7. Report | `report.py` | JSON, CSV, console table, annotated video, stage stills |

### 1. Detecting the board

The clip interleaves three kinds of frame: the full table, a full-screen pin
animation with no table at all, and the table with a pin-animation window pasted
over part of it.

A frame is scored on **table structure** rather than colour or template
matching. A morphological opening with a long horizontal kernel and another with
a long vertical kernel isolate the grid rules.

The **column rules carry the detection**: there are eleven of them, strongly
drawn and *evenly spaced* — a signature ordinary imagery does not produce. The
longest evenly-spaced run of detected rules is what qualifies a frame. Horizontal
rules are treated as a bonus, not a requirement, because some boards separate
their rows by colour banding alone and draw almost no long horizontal edges. (An
earlier version demanded three horizontal rules and failed to detect such a board
at all.)

The table is then outlined one of two ways. When horizontal rules are present,
the combined rule mask gives a precise outline. When they are not, the outline is
reconstructed from the column rules — their ink gives the vertical extent, and
the initials and total columns are added back using their widths relative to one
frame column. A `minAreaRect` gives four corners and a perspective transform maps
them onto a fixed 1440×544 canvas, so every later stage works in stable pixel
coordinates. A quad skewed beyond a threshold is rejected rather than warped,
since that means the fit is wrong.

Rejection was checked against pin animation, pure noise, black and flat colour —
none are mistaken for a board.

**Occlusion.** The inset animation window is found as a large region containing
no grid rules. A single empty cell looks the same, so a region only counts as an
occluder if it spans several cells in *both* axes — something one empty cell
cannot do. Occluded cells are skipped for that frame only, and recovered from
the frames where they are visible.

### 2. Mapping the grid

Column boundaries are read off **the table's own rules**, not assumed. The eleven
interior rules are equally spaced, so the longest evenly-spaced run of detected
rules *is* the answer, and the outer two edges are the table's own borders. Rules
lying on the table border are discarded first — they bound the table, not a
column.

This matters more than it sounds. An earlier version snapped a proportional prior
(measured off a reference frame) onto the detected rules. On a board whose column
widths differed from those measurements by more than a fraction of a column
width, the snap failed, the grid fell back to the wrong prior, and accuracy
collapsed from 98% to **1%**. Deriving the edges from the image removes that
dependency; the prior now only applies if the rules cannot be found at all.

Row boundaries are anchored on the four **player initials** in the left column,
which sit at the vertical centre of each player's throw sub-row. Each initial is
located in its own small crop, then a least-squares line through the four centres
gives the row pitch and offset — so one missed initial does not skew the geometry.

Both fall back to fixed proportions if the image evidence is too weak, and
`preview` reports which path was taken (`cols:detected` / `cols:fallback`).

### 3. Segmenting glyphs

Cells come in two polarities: white on blue for idle players, dark on
yellow/white for the player who is up. Instead of special-casing the active row,
the background is inferred **per cell** from its border ring, so either polarity
works. This is also why the initials are located one crop at a time — a single
global threshold over the whole column cannot serve both.

Two details that cost real accuracy:

- **The miss mark.** `-` is far shorter than a digit and fails a height filter,
  so wide flat marks are admitted on aspect ratio instead.
- **Fused digits.** Compression blur plus the morphological close can weld
  touching digits into one component — `15` arrives as a single blob. Guessing
  the glyph count from the aspect ratio is unreliable (in some fonts `1` is as
  wide as any other digit), so the component is cut where its **ink column
  profile genuinely thins out**, with the threshold relaxed in steps until a cut
  is found.
- **Circled digits.** The board marks a split as a digit inside a ring. Those are
  two components in the same place, rejoined by testing *containment* — one box
  nested inside the other. Testing horizontal overlap instead would wrongly weld
  neighbouring digits together.

### 4. Classifying

The board draws from a tiny fixed alphabet — `0-9`, `X`, `/`, `-`, circled
digits, and letters for names — in a single fixed font. For that, a
nearest-neighbour match against reference bitmaps beats a general OCR engine on
both accuracy and speed, and needs nothing downloaded.

Each glyph is normalised into a 28×28 square **with its aspect ratio preserved**
(stretch it to fill and `1` and `-` become the same picture), then matched by
mean absolute difference. Confidence is the normalised margin between the best
and the best differently-labelled template, so a near-tie scores low. Matches
worse than an absolute distance ceiling are reported as `?` rather than guessed —
a visible unknown is better than a plausible wrong digit.

### 5. Fusing across frames

Two separate problems:

- **Noise.** Any single frame can misread. Readings are pooled and voted on,
  weighted by confidence.
- **The board is live.** Throws appear as the game is bowled — in the sample
  clip, player T gains a fourth throw partway through. So the most common value
  across the whole clip is *not* the final value. Cells are resolved from a
  window at the **end** of the clip, and the transitions are reported separately
  as a timeline in the JSON.

### 6. Verifying against the rules

This is the part a pure OCR pipeline cannot do. The board prints both the throws
and the running totals, and the rules of ten-pin bowling link them. Recomputing
each total from the throws and comparing gives a genuine correctness signal
rather than a confidence score.

`bowling.py` implements strikes, spares, the ten-pin bonus structure and the
tenth-frame third ball. Where the throws parse cleanly but a printed total
disagrees, the computed value is preferred and the frame is flagged
(`--no-repair` reports without correcting). Frames whose bonus balls have not
been thrown yet stay unscored rather than being guessed.

The printed `TTL` is checked against a **provisional** total: settled frames plus
pins already down in an open in-progress frame. Pins from an unresolved strike or
spare are not added, matching the board, which holds the total back until the
bonus balls are thrown.

---

## Calibration

The one manual step, done once per capture. The board's font is not known in
advance, so the alphabet has to be pinned down from the footage itself.

```bash
python main.py calibrate --video data/bowling_scoreboard.mp4
```

This segments every glyph in the clip and groups identical shapes without
supervision, then writes `templates/clusters.png` — a numbered contact sheet —
and `templates/labels.json`. Type the character each numbered shape shows:

```json
{
  "clusters": {
    "0": { "count": 144, "seen_in": ["score", "throws"], "label": "1" },
    "1": { "count": 117, "seen_in": ["throws"],          "label": "-" },
    "11": { "count": 30, "seen_in": ["score"],           "label": "0" },
    "26": { "count": 12, "seen_in": ["throws"],          "label": "8o" }
  }
}
```

Conventions: a digit for a pin count, `X` strike, `/` spare, `-` miss, a digit
plus lowercase `o` for a circled (split) digit, capital letters for names. Leave
a label empty to discard a cluster as noise. Then:

```bash
python main.py calibrate --finalize
```

You will usually see **more clusters than characters** — the same digit drawn
white-on-blue, dark-on-yellow and white-on-red lands in different clusters. That
is deliberate. The clustering radius is set tight because the costs are
asymmetric: splitting one character across two clusters costs one extra label and
gives the classifier more templates, whereas merging two characters silently
corrupts every reading. Measured distances between instances of the same
character reach ~0.05, while `0` vs `8` and `T` vs `7` come as close as ~0.08.

Label every cluster, including duplicates — a variant left unlabelled becomes a
`?` in the output. Re-running the clustering pass preserves labels you have
already typed.

---

## Commands

```
python main.py probe     --video PATH
python main.py preview   --video PATH --frame N [--debug]
python main.py calibrate  --video PATH [--every N] | --finalize
python main.py extract   --video PATH [options]
```

`extract` options:

| Flag | Effect |
| --- | --- |
| `--every N` | sample every Nth frame (default 2) |
| `--limit N` | stop after N sampled frames |
| `--out DIR` | output directory (default `output/`) |
| `--stills` | also write the per-stage stills |
| `--no-video` | skip `annotated.mp4` |
| `--no-repair` | report rule mismatches without correcting them |
| `--verbose` | print progress |

`preview` is the tool to reach for when something looks wrong — it reports
whether the board was found, whether the grid was detected or fell back, and
where any occluders are, and `--debug` writes the intermediate images.

---

## Accuracy

Measured on the synthetic clip, which has exact ground truth
(`python tools/eval_synthetic.py`):

| Metric | Result |
| --- | --- |
| Per-frame cell accuracy | 3799 / 3800 = **99.97%** |
| Final cell accuracy after voting | 39 / 39 = **100%** |
| Mean glyph confidence | 0.862 |
| Scoring-rule agreement | 100% over 15 checkable frames |
| Board detected | 45 / 50 sampled frames (the other 5 are animation-only) |

The single per-frame miss is the one frame on which the board changes value
mid-render.

### Robustness

The synthetic board can be re-rendered with a layout that deliberately disagrees
with the fractions in `config.py`, and with its horizontal rules removed. Cells
correctly segmented (grid quality, independent of the template bank):

| Variant | Before the detection rewrite | Now |
| --- | --- | --- |
| Reference layout | 100% | 100% |
| No horizontal rules (colour banding only) | **not detected at all** | 100% |
| Column widths unlike `config` | **1.1%** | 100% |
| Both, with rules present | — | 87.5% |

False positives: pin animation, pure noise, black and flat colour are all
rejected.

### Tested against the real capture - and it fails

`data/real/` holds three frames lifted from the actual `bowling_scoreboard.mp4`.
**The detector does not find the board in them.** This is the most important fact
in this README, and it is pinned as an `xfail` test rather than hidden.

What the real frames revealed:

| Assumption | Reality on the real board |
| --- | --- |
| Rows separated by drawn horizontal rules | **Colour banding only.** A horizontal line mask finds 2-3 lines, and they are the video progress bar |
| Column rules strong enough to binarise | **Too faint.** A vertical line mask finds 4 of 11; the pale score rows kill the contrast |
| Board fills the frame | Occupies the upper ~75%, with blue below |
| `TTL_COL_FRAC = 0.085` | Actually ~0.135 - the estimate would have clipped the TTL digits. Corrected |

What *did* work on real pixels, in `src/structure.py` (experimental, not wired in):
summing `|Sobel x|` down each column and running a comb filter over the profile
recovers the column period reliably - **114.9px on both board frames** - and the
player initials anchor the rows to **y = 146, 281, 409, 525 against a truth of
147, 278, 409, 530**. The pin-animation frame yields nothing, so the
discrimination works too. What is not solved is the *phase*: which comb tooth is
the first column rule. Every discriminator tried fixed one board and broke the
other; the module documents each attempt.

So the honest position is that the periodicity approach is the right direction
and is half-built, and the shipped detector needs it before it will read this
board.

### What the synthetic numbers do and do not show

Being precise, because it matters:

- **The full pipeline has never run end to end on the real capture.** Every
  accuracy number above comes from a clip this repository generates.
- **The synthetic clip's layout is derived from this project's own config**
  (`tools/make_synthetic.py:63-79` reads `CANON_W`, `INITIAL_COL_FRAC`, and the
  rest). So the reference-layout result is partly self-confirming. That is exactly
  why the shifted-layout row above exists — it is the test that is *not* circular.
- **The glyph font is not the board's font.** The synthetic draws with a built-in
  OpenCV font. The template-bank design is font-agnostic by construction, but
  thresholds tuned against one font (`FLAT_MARK_ASPECT`, `SPLIT_MIN_RATIO`,
  `CAL_CLUSTER_DIST`) may need adjusting for another.
- **A correct final table does not by itself prove the OCR was correct**, because
  stage 6 can reconstruct a total from the throws. During development two cells
  failed to read in *every* frame and the printed table still looked right. That
  is why `eval_synthetic.py` scores raw cell readings separately.

## Tests

```bash
python -m pytest tests/ -q          # 50 pass, 2 xfail, ~9s
```

The two `xfail`s are the real capture frames. They are expected failures today
and will flip to passes when detection handles a colour-banded board.

- `tests/test_bowling.py` — the scoring engine: strikes, spares, the tenth
  frame, a 300 game, mid-frame states, and the four player rows from the sample
  footage as ground truth.
- `tests/test_pipeline_synthetic.py` — end to end on a generated clip: detection,
  rejection of animation-only frames, occlusion handling, grid detection, the
  full scoreboard, split flagging, the live-board case, and serialisation.
- The robustness cases above are pinned as regression tests, including the
  87.5% one, so a change that degrades them fails the suite rather than passing
  quietly.

No binary fixtures — the test clip is generated at run time.

---

## Layout

```
main.py                   CLI
src/
  config.py               all tunable constants, documented
  video.py                decoding and frame sampling
  detect.py               stage 1 - find and rectify the table
  grid.py                 stage 2 - columns, rows, cell map
  segment.py              stage 3 - cell to glyph bitmaps
  classify.py             stage 4 - template bank and matching
  calibrate.py            build the template bank (clustering + labelling)
  aggregate.py            stage 5 - voting and change timeline
  bowling.py              stage 6 - scoring rules and verification
  report.py               stage 7 - output artefacts
  results.py              result model and serialisation
  imgutil.py              shared image helpers
tools/
  make_synthetic.py       generate a ground-truth test clip
  build_synthetic_bank.py label that clip from ground truth
  eval_synthetic.py       accuracy report
  inspect_cells.py        per-cell segmentation dump
tests/
```

## Limitations

- **Calibration is per-capture.** A board with a different font needs the
  labelling pass again. The bank in `templates/bank.npz` is not portable.
- **Fixed layout.** Four player rows and ten frames are assumed
  (`N_PLAYER_ROWS`, `N_FRAMES` in `config.py`). A board with a different shape
  needs those changed.
- **Names need letter templates.** The bowler's name only reads if its letters
  were labelled during calibration; unlabelled letters come back as `?`.
- **Mild perspective only.** Quads skewed past `MAX_SKEW_DEG` are rejected rather
  than warped. Heavy off-axis footage would need corner refinement.
- **Does not currently read the real capture.** See "Tested against the real
  capture" above. `src/structure.py` is the route in; the phase problem is the
  one thing left to solve.
- **The glyph splitter can over-segment stretched cells.** On a board whose
  proportions differ markedly from the canvas aspect, a two-digit score can split
  into three. `SPLIT_MIN_RATIO` in `config.py` is the knob.
- **The rules check needs printed totals.** It can only verify frames where the
  board shows a running total; a board that only prints throws gets no
  cross-check.

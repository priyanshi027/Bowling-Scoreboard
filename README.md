# Bowling Scoreboard Extraction from Video

A classical computer vision pipeline for extracting structured ten-pin bowling scoreboard data directly from video using **OpenCV and NumPy**.

The system reads:

* 🎳 Each player's throws
* 🔢 Running totals
* 🛣️ Lane number
* 👤 Current player / bowler
* ⏱️ Changes in the scoreboard over time

No OCR engine or neural network is used. The approach relies entirely on **classical computer vision, geometric constraints, template matching, and bowling-score verification rules**.

---

## 🎯 Objective

Read a ten-pin bowling scoreboard from video and convert it into structured data containing:

```text
Player
├── Throws
├── Running totals
├── Lane number
└── Current bowler
```

The scoreboard is treated as a structured visual system rather than a generic text-recognition problem.

---

## 🧠 Approach

### Classical Computer Vision Only

The implementation uses:

* **OpenCV**
* **NumPy**
* Image morphology
* Connected-component analysis
* Perspective transformation
* Binary segmentation
* Nearest-neighbour classification
* Temporal voting
* Bowling-score verification

No:

* ❌ OCR engine
* ❌ Neural network
* ❌ Deep learning model
* ❌ Runtime model downloads

The reason is that the scoreboard has a **small alphabet, fixed font, rigid layout, and predictable geometry**. These constraints provide strong information that can be exploited directly.

---

# 🔍 Method

## 1. Detection

Frames are evaluated based on **table structure rather than colour**.

Directional morphological operations are used to isolate the scoreboard's grid rules.

The key detection cue is the presence of **eleven evenly spaced column rules**.

This periodic structure is highly characteristic of the bowling scoreboard and is unlikely to occur naturally in ordinary imagery.

### Detection Pipeline

```text
Video Frame
     │
     ▼
Directional Morphology
     │
     ▼
Grid Rule Detection
     │
     ▼
Eleven Column Rules?
     │
   ┌─┴─┐
   │   │
  Yes  No
   │   │
   ▼   └──► Reject Frame
Scoreboard Detected
   │
   ▼
Perspective Rectification
```

Frames without the expected table structure, such as **pin animations**, are rejected.

---

## 2. Perspective Rectification

Once the scoreboard is detected, it is transformed onto a fixed-size canvas using a **perspective transform**.

This ensures that all subsequent processing happens in stable coordinates.

```text
Original Frame
      │
      ▼
Detected Scoreboard
      │
      ▼
Perspective Transform
      │
      ▼
Fixed Canvas
      │
      ▼
Stable Cell Coordinates
```

This eliminates variations caused by:

* Camera perspective
* Rotation
* Position
* Scale
* Minor viewpoint changes

---

# 📐 3. Grid Mapping

Cell boundaries are **measured from the scoreboard itself** rather than being hard-coded.

### Column Mapping

The scoreboard's own vertical rules determine the column boundaries.

The eleven column rules provide a reliable geometric reference.

### Row Mapping

Rows are anchored using the **four player initials**, which occur at the centre of each throw sub-row.

A least-squares fit is applied through the detected anchors.

This makes the system robust to a missed anchor because one incorrect detection does not significantly distort the entire geometry.

### Why Not Assume Fixed Proportions?

Assuming fixed cell proportions is fragile.

Even a small geometric error can cause the extracted region to shift by a fraction of a column width, resulting in:

```text
Correct Geometry
      │
      ▼
 ┌────┬────┬────┐
 │ A  │ 7  │ 8  │
 └────┴────┴────┘

Small Geometry Error
      │
      ▼
 ┌────┬────┬────┐
 │ A7 │ 8  │    │
 └────┴────┴────┘
```

Therefore, measured boundaries are preferred over assumed proportions.

---

# 🖼️ 4. Cell Segmentation

Each scoreboard cell is binarised independently.

The polarity is inferred **locally** because different rows can have different visual polarity.

For example:

* Active player's row → **dark text on light background**
* Other rows → **light text on dark background**

Therefore, a single global threshold polarity is unreliable.

### Segmentation Pipeline

```text
Scoreboard Cell
      │
      ▼
Local Polarity Estimation
      │
      ▼
Binary Image
      │
      ▼
Connected Components
      │
      ▼
Glyph Candidates
```

---

## 🔗 Component Corrections

Connected-component segmentation requires two important corrections.

### Circled Digit / Split Marker

A circled digit can appear as multiple connected components even though it represents one logical glyph.

The components are therefore rejoined using **containment relationships**.

```text
┌─────────┐
│    8    │
│  (   )  │
└─────────┘

      ↓

One logical glyph
```

### Fused Digits

Blur can cause two touching digits to become a single connected component.

The component is therefore analysed using its **ink profile**.

A thin region in the profile can indicate the correct location to split the fused glyph.

```text
Fused Component

███████│██████
███████│██████
██████ │ █████
██████ │ █████
       ▲
       │
    Split Point
```

---

# 🔤 5. Glyph Classification

Classification uses **nearest-neighbour matching** against reference bitmaps.

The reference bitmaps are constructed once per capture through **unsupervised clustering of glyphs observed in the video**.

This allows the system to learn the actual glyph appearance used by that particular scoreboard capture without requiring an external OCR model.

### Classification Pipeline

```text
Extracted Glyph
      │
      ▼
Normalisation
      │
      ▼
Reference Glyphs
      │
      ▼
Nearest-Neighbour Matching
      │
      ▼
Best Match
      │
      ▼
Confidence Evaluation
```

---

## 📏 Aspect-Ratio Preservation

Glyph aspect ratio is preserved during normalisation.

This is important because narrow and flat marks can become indistinguishable if every glyph is forced into the same shape.

For example:

```text
Original Glyph

████
████
████

      ↓ Preserve aspect ratio

████
████
████
```

Instead of unnecessarily stretching it into a square representation.

---

## 🎯 Confidence

Classification confidence is based on the **margin between the best and second-best matches**.

Conceptually:

```text
Confidence = Distance(second_best)
             -
             Distance(best)
```

A large margin indicates a reliable classification.

A small margin indicates ambiguity.

Low-confidence matches are therefore reported as:

```text
UNKNOWN
```

rather than being incorrectly guessed.

---

# ⏱️ 6. Temporal Fusion

Individual video frames can contain recognition errors.

Instead of trusting a single frame, observations are pooled over time and combined using **confidence-weighted voting**.

```text
Frame 1 → 8   (High confidence)
Frame 2 → 8   (High confidence)
Frame 3 → 3   (Low confidence)
Frame 4 → 8   (High confidence)
Frame 5 → 8   (High confidence)

             ↓

        Final Reading
             8
```

---

## 🎳 Handling a Live Scoreboard

The scoreboard is not static.

Throws appear progressively as the game is played.

Therefore, simply selecting the most frequent value across the entire video can produce an incorrect final result.

Instead, cells are resolved by processing observations **from the end of the video backwards**.

The changes in each cell are retained as a timeline.

```text
Time ───────────────────────────────►

T1        T2        T3        T4
│         │         │         │
▼         ▼         ▼         ▼

Empty     7         7         7
                    │
                    └──► New throw

Empty     Empty     8         8
                              │
                              └──► Final state
```

This allows the system to distinguish between:

* Initial state
* Intermediate score
* Newly recorded throw
* Final score

---

# ✅ 7. Bowling-Rule Verification

The most important validation step is **score verification using the rules of ten-pin bowling**.

The scoreboard contains both:

1. Individual throws
2. Running totals

The system independently recomputes the running totals from the extracted throws.

```text
Detected Throws
      │
      ▼
Bowling Score Rules
      │
      ▼
Recomputed Totals
      │
      ├──────────────┐
      ▼              ▼
Printed Total    Recomputed Total
      │              │
      └──────┬───────┘
             ▼
         Comparison
             │
       ┌─────┴─────┐
       ▼           ▼
      Match      Mismatch
       │           │
       ▼           ▼
    Valid       Re-evaluate
```

If the printed and recomputed totals disagree, the corresponding reading can be reconsidered.

This provides an **independent correctness check** instead of relying only on visual classification confidence.

---

# 🏗️ Complete Pipeline

```text
                    VIDEO
                      │
                      ▼
              Frame Extraction
                      │
                      ▼
             ┌─────────────────┐
             │ Scoreboard      │
             │ Detection       │
             └─────────────────┘
                      │
                      ▼
             Grid Rule Detection
                      │
                      ▼
              11 Column Rules
                      │
                      ▼
          Perspective Rectification
                      │
                      ▼
               Grid Mapping
                      │
                      ▼
               Cell Extraction
                      │
                      ▼
             Local Binarisation
                      │
                      ▼
          Connected Components
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Split Components    Merge Components
             │                 │
             └────────┬────────┘
                      ▼
               Glyph Extraction
                      │
                      ▼
             Glyph Classification
                      │
                      ▼
              Confidence Scoring
                      │
                      ▼
              Temporal Fusion
                      │
                      ▼
             Bowling Score Rules
                      │
                      ▼
                 Verification
                      │
                      ▼
             Structured Output
```

---

# 📊 Output

The final output is structured around each player and contains their bowling information.

Example conceptual output:

```json
{
  "lane": 12,
  "players": [
    {
      "name": "AB",
      "throws": [8, 1, 10, 7, 2],
      "running_totals": [9, 29, 38, 47]
    },
    {
      "name": "CD",
      "throws": [10, 8, 1, 9],
      "running_totals": [19, 28, 38]
    }
  ],
  "current_bowler": "AB"
}
```

> The exact output schema can be adapted to the scoreboard format used by the capture.

---

# 🛠️ Technology Stack

| Technology                   | Purpose                              |
| ---------------------------- | ------------------------------------ |
| **Python**                   | Implementation                       |
| **OpenCV**                   | Image processing and computer vision |
| **NumPy**                    | Numerical operations                 |
| **Morphological Operations** | Grid detection                       |
| **Perspective Transform**    | Scoreboard rectification             |
| **Connected Components**     | Glyph segmentation                   |
| **Nearest Neighbour**        | Glyph classification                 |
| **Clustering**               | Reference glyph generation           |
| **Temporal Voting**          | Video-level recognition              |
| **Bowling Rules**            | Score verification                   |

---

# 🚫 Why No OCR or Neural Network?

A general OCR system is unnecessary for this problem.

The scoreboard provides strong prior information:

* Fixed layout
* Fixed font
* Small character alphabet
* Known grid structure
* Predictable cell locations
* Repeated glyphs
* Known bowling-score rules

Therefore, the problem can be treated as:

> **Structured visual recognition rather than general text recognition.**

This makes a classical computer vision solution:

* Easier to inspect
* Easier to debug
* Deterministic
* Lightweight
* Independent of downloaded models
* Suitable for offline execution

---

# 💡 Why This Design?

The central idea is to exploit **structure before appearance**.

Instead of asking:

> "What text does this image contain?"

the pipeline asks:

> "Does this frame contain the expected scoreboard geometry, and what glyphs occur inside the cells defined by that geometry?"

This dramatically reduces the search space.

The system combines three independent sources of information:

### 1. Geometry

The scoreboard's rigid grid determines **where** information should be.

### 2. Appearance

Glyph segmentation and nearest-neighbour matching determine **what** each cell contains.

### 3. Domain Rules

Bowling scoring determines whether the extracted information is **internally consistent**.

Together:

```text
        Geometry
           │
           ▼
       Cell Location
           │
           ▼
       Glyph Reading
           │
           ▼
    Temporal Consistency
           │
           ▼
     Bowling Validation
           │
           ▼
      Reliable Output
```

---

# 🔬 Key Design Decisions

| Problem                             | Solution                          |
| ----------------------------------- | --------------------------------- |
| Pin animations / irrelevant frames  | Detect scoreboard grid structure  |
| Perspective distortion              | Perspective transform             |
| Unknown cell boundaries             | Measure boundaries from grid      |
| Missed player anchor                | Least-squares row fitting         |
| Different row polarity              | Local polarity inference          |
| Circled glyph split into components | Containment-based merging         |
| Touching digits                     | Ink-profile based splitting       |
| Glyph appearance variation          | Per-capture reference clustering  |
| Narrow / flat glyphs                | Preserve aspect ratio             |
| Ambiguous recognition               | Confidence margin                 |
| Single-frame errors                 | Temporal fusion                   |
| Live scoreboard updates             | End-of-clip resolution + timeline |
| Incorrect totals                    | Bowling-rule verification         |

---

# ⚙️ Design Principles

### Structure Over Colour

The scoreboard's geometry is more reliable than its colour.

### Measurement Over Assumptions

Cell boundaries are extracted from the image rather than relying on fixed proportions.

### Local Decisions Over Global Thresholds

Each cell determines its own segmentation polarity.

### Unknown Over Guessing

Ambiguous glyphs are rejected rather than forced into a class.

### Temporal Evidence Over Single Frames

Multiple observations are combined to reduce frame-level errors.

### Rules Over Confidence

Visual confidence alone is insufficient. Bowling-score arithmetic provides an independent validation mechanism.

---

# 🚀 Advantages

* ✅ Classical computer vision only
* ✅ No OCR dependency
* ✅ No neural networks
* ✅ No model downloads
* ✅ Works offline
* ✅ Lightweight
* ✅ Explainable pipeline
* ✅ Deterministic processing
* ✅ Uses scoreboard geometry
* ✅ Handles perspective distortion
* ✅ Handles live score changes
* ✅ Provides confidence estimates
* ✅ Performs independent score verification

---

# 📌 Limitations

The approach is intentionally specialised for a structured bowling scoreboard.

It assumes that:

* The scoreboard layout remains sufficiently consistent.
* The scoreboard font belongs to the expected family.
* Grid lines remain detectable.
* Characters are not heavily occluded.
* Video quality is sufficient for segmentation.
* The bowling rules represented by the scoreboard are standard ten-pin scoring.

For substantially different scoreboard designs, the geometric and glyph models may need to be adapted.

---

# 🔮 Future Improvements

Potential extensions include:

* [ ] Automatic lane-number extraction
* [ ] More robust player-name recognition
* [ ] Improved fused-glyph splitting
* [ ] Sub-pixel grid estimation
* [ ] Better motion-blur handling
* [ ] Automatic video-quality assessment
* [ ] Real-time processing
* [ ] Export to CSV
* [ ] Export to JSON
* [ ] Web-based scoreboard visualisation
* [ ] Automated test suite with annotated videos
* [ ] Benchmarking against OCR-based approaches

---

# 📁 Suggested Project Structure

```text
bowling-scoreboard/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── detection.py
│   ├── rectification.py
│   ├── grid_mapping.py
│   ├── segmentation.py
│   ├── glyph_classifier.py
│   ├── temporal_fusion.py
│   ├── bowling_rules.py
│   └── pipeline.py
│
├── data/
│   ├── input/
│   └── output/
│
├── references/
│   └── glyphs/
│
├── tests/
│
└── examples/
```

---

# ▶️ Getting Started

## Installation

```bash
git clone <repository-url>
cd bowling-scoreboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
opencv-python
numpy
```

---

## Run

```bash
python src/pipeline.py --input data/input/bowling.mp4
```

Example output:

```text
Lane: 12

Player: AB
Throws: 8 1 10 7 2
Totals: 9 29 38 47

Player: CD
Throws: 10 8 1 9
Totals: 19 28 38

Current Bowler: AB
```

---

# 🧪 Verification Strategy

The pipeline should be evaluated at multiple levels rather than using only final-score accuracy.

### Detection

* Scoreboard detection rate
* False-positive frame rate

### Grid Mapping

* Column-boundary error
* Row-boundary error
* Cell localisation accuracy

### Segmentation

* Glyph component detection
* Split/merge accuracy

### Classification

* Per-glyph accuracy
* Unknown rate
* Confidence distribution

### Temporal Fusion

* Per-cell final accuracy
* Correction rate over individual frames

### End-to-End

* Throw accuracy
* Running-total accuracy
* Player identification accuracy
* Lane identification accuracy
* Final scoreboard accuracy

---

# 🏆 Core Insight

The most important idea behind this project is:

> **Use the scoreboard's structure and bowling rules as additional information sources instead of treating the problem as generic OCR.**

A single frame may be ambiguous.

A glyph may be blurry.

A component may be incorrectly segmented.

But when **geometry + appearance + temporal information + bowling rules** agree, the resulting extraction becomes substantially more reliable and, importantly, **checkable**.

---

# 📄 License

Add your preferred license here, for example:

```text
MIT License
```

---

# 👩‍💻 Author

**Priyanshi Singh**

Computer Vision | OpenCV | Python | C++ | Machine Learning | DSA

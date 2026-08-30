"""One-time setup - build the labelled glyph template bank for a capture.

Run in two passes:

    python main.py calibrate --video data/bowling_scoreboard.mp4
        Sweeps the clip, segments every glyph it can find, groups identical
        shapes, and writes templates/clusters.png plus templates/labels.json.

    <edit templates/labels.json, naming each cluster>

    python main.py calibrate --finalize
        Turns the labelled clusters into templates/bank.npz.

Grouping is unsupervised, so the manual step is naming roughly fifteen shapes
once - not labelling thousands of glyphs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg
from . import grid as grid_mod
from . import segment
from . import video as video_mod
from .classify import TemplateBank
from .detect import box_overlap_frac, detect_board

CLUSTER_IMAGE = "clusters.png"
CLUSTER_DATA = "clusters.npz"
LABELS_FILE = "labels.json"
BANK_FILE = "bank.npz"


@dataclass
class Cluster:
    exemplar: np.ndarray                  # first member; the matching reference
    total: np.ndarray                     # running sum, for a smoothed template
    count: int = 0
    roles: set[str] = field(default_factory=set)

    @property
    def centroid(self) -> np.ndarray:
        return self.total / max(self.count, 1)


def _cluster_glyphs(items: list[tuple[np.ndarray, str]]) -> list[Cluster]:
    """Greedy single-pass clustering in normalised glyph space.

    Every instance of a given character is essentially the same bitmap, so shapes
    separate at a fixed radius with no need for k-means or a guess at the number
    of classes.

    Membership is decided against each cluster's first exemplar rather than its
    running centroid. A centroid drifts as it absorbs members, and a drifting
    centroid can wander close enough to a neighbouring character to start
    swallowing it - which is how "0" ends up inside "8".
    """
    clusters: list[Cluster] = []
    for bitmap, role in items:
        vec = bitmap.astype(np.float32)
        best_i, best_d = -1, 1e9
        for i, cluster in enumerate(clusters):
            d = float(np.abs(cluster.exemplar - vec).mean())
            if d < best_d:
                best_i, best_d = i, d

        if best_i >= 0 and best_d <= cfg.CAL_CLUSTER_DIST:
            clusters[best_i].total += vec
            clusters[best_i].count += 1
            clusters[best_i].roles.add(role)
        elif len(clusters) < cfg.CAL_MAX_CLUSTERS:
            clusters.append(
                Cluster(exemplar=vec.copy(), total=vec.copy(), count=1, roles={role})
            )

    clusters = [c for c in clusters if c.count >= cfg.CAL_MIN_CLUSTER]
    clusters.sort(key=lambda c: -c.count)
    return clusters


def collect_glyphs(video_path: str | Path, every: int = 4,
                   max_frames: int | None = None) -> list[tuple[np.ndarray, str]]:
    """Segment every glyph in the clip, tagged with the role of its cell."""
    items: list[tuple[np.ndarray, str]] = []
    used = 0
    for index, frame in video_mod.iter_frames(video_path, every=every):
        det = detect_board(frame)
        if not det.ok or det.warp is None:
            continue
        used += 1
        gmap = grid_mod.build_grid(det.warp, det.geometry is not None)
        for cell in gmap.cells:
            if any(box_overlap_frac(cell.box, occ) > cfg.OCCLUDER_CELL_COVER
                   for occ in det.occluders):
                continue
            for glyph in segment.extract_glyphs(grid_mod.crop(det.warp, cell)):
                items.append((glyph.bitmap, cell.role))

        # The title strip contributes the letters of the bowler's name and the
        # lane digits, which appear nowhere else on the board.
        if det.title is not None and det.title.size:
            for glyph in segment.extract_glyphs(det.title):
                items.append((glyph.bitmap, "title"))

        if max_frames is not None and used >= max_frames:
            break

    if not items:
        raise RuntimeError(
            "no glyphs found - the scoreboard was never detected.\n"
            "Try:  python main.py preview --video <path> --frame 30 --debug"
        )
    return items


def _mosaic(clusters: list[Cluster]) -> np.ndarray:
    """Contact sheet of cluster exemplars, each captioned with its id."""
    tile = cfg.CAL_TILE
    label_h = 22
    cols = min(cfg.CAL_MOSAIC_COLS, max(1, len(clusters)))
    rows = (len(clusters) + cols - 1) // cols

    sheet = np.full((rows * (tile + label_h), cols * tile, 3), 30, dtype=np.uint8)
    for i, cluster in enumerate(clusters):
        r, c = divmod(i, cols)
        img = cv2.resize((cluster.exemplar * 255).astype(np.uint8), (tile, tile),
                         interpolation=cv2.INTER_NEAREST)
        y0 = r * (tile + label_h)
        x0 = c * tile
        sheet[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(sheet, f"{i} (n={cluster.count})", (x0 + 3, y0 + tile + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return sheet


def run_clustering(video_path: str | Path, templates_dir: str | Path,
                   every: int = 4) -> tuple[int, Path, Path]:
    """First calibration pass. Returns ``(n_clusters, mosaic_path, labels_path)``."""
    templates = Path(templates_dir)
    templates.mkdir(parents=True, exist_ok=True)

    items = collect_glyphs(video_path, every=every)
    clusters = _cluster_glyphs(items)
    if not clusters:
        raise RuntimeError("glyphs were found but none recurred often enough to cluster")

    mosaic_path = templates / CLUSTER_IMAGE
    cv2.imwrite(str(mosaic_path), _mosaic(clusters))

    np.savez_compressed(
        templates / CLUSTER_DATA,
        centroids=np.stack([c.centroid for c in clusters]),
        exemplars=np.stack([c.exemplar for c in clusters]),
        counts=np.array([c.count for c in clusters]),
    )

    labels_path = templates / LABELS_FILE
    existing = _load_existing_labels(labels_path)
    payload = {
        "_help": (
            f"Open {CLUSTER_IMAGE} and type the character each numbered shape shows. "
            "Use X for a strike, / for a spare, - for a miss, a digit for a pin "
            "count, a digit followed by lowercase o for a circled (split) digit, "
            "and a capital letter for a player initial. Leave a label empty to "
            "discard that cluster as noise. Then run: "
            "python main.py calibrate --finalize"
        ),
        "clusters": {
            str(i): {
                "count": int(c.count),
                "seen_in": sorted(c.roles),
                "label": existing.get(str(i), ""),
            }
            for i, c in enumerate(clusters)
        },
    }
    labels_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(clusters), mosaic_path, labels_path


def _load_existing_labels(path: Path) -> dict[str, str]:
    """Preserve labels already typed in, so re-clustering is not destructive."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v.get("label", "") for k, v in data.get("clusters", {}).items()}
    except (json.JSONDecodeError, AttributeError):
        return {}


def finalize(templates_dir: str | Path) -> tuple[Path, list[str]]:
    """Second calibration pass: labelled clusters -> template bank."""
    templates = Path(templates_dir)
    data_path = templates / CLUSTER_DATA
    labels_path = templates / LABELS_FILE
    for path in (data_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} is missing - run the clustering pass first:\n"
                "  python main.py calibrate --video <path>"
            )

    data = np.load(data_path, allow_pickle=False)
    centroids = data["centroids"]
    spec = json.loads(labels_path.read_text(encoding="utf-8"))["clusters"]

    pairs: list[tuple[str, np.ndarray]] = []
    for key, entry in spec.items():
        label = str(entry.get("label", "")).strip()
        if not label:
            continue
        index = int(key)
        if not 0 <= index < len(centroids):
            continue
        if label not in cfg.ALPHABET:
            raise ValueError(
                f"cluster {key}: {label!r} is not in the alphabet. "
                "Expected a digit, X, /, -, a digit plus lowercase o, or A-Z."
            )
        pairs.append((label, centroids[index]))

    if not pairs:
        raise ValueError(f"no labels filled in yet - edit {labels_path}")

    bank_path = templates / BANK_FILE
    TemplateBank.from_pairs(pairs).save(bank_path)
    return bank_path, sorted({label for label, _ in pairs})

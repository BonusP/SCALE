"""S2 - discover the series palette in CIELAB and build soft colour layers.

Legend swatches seed the palette but do not define it: a swatch is often a blend of
the marker and its connecting line, so the seeds are refined against the actual ink.
Layers are deliberately *soft* - a pixel joins every palette entry within
``soft_layer_delta_e`` of its best match - because near-hue pairs (blue vs violet)
cannot be separated per pixel.  The final colour call is made per glyph, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .canvas import Canvas
from .config import ColorLabParams


@dataclass
class Palette:
    centers_lab: np.ndarray
    """(K, 3) float32 refined centres in CIELAB."""
    centers_rgb: np.ndarray
    """(K, 3) uint8."""
    match_centers_lab: np.ndarray
    """(K, 3) float32 used to *label* an assembled chain.

    These are the legend swatches themselves when a legend exists.  Refined centres are
    right for deciding which pixels belong to a layer, but for naming a curve only the
    relative order matters, and the unrefined swatches keep that order honest - a badly
    blended swatch is still nearer its own curve than to anyone else's.
    """
    display_rgb: np.ndarray
    """(K, 3) uint8 - legend swatch when available, so overlays match the printed figure."""
    labels: list[str | None]
    """Legend text per entry, when a legend seeded it."""
    seed_rgb: list[list[int] | None]
    source: str
    layers: np.ndarray
    """(K, H, W) bool soft membership masks."""
    best_index: np.ndarray
    """(H, W) int32 argmin palette index, -1 where the pixel is not curve ink."""
    contested: np.ndarray
    """(H, W) bool - curve ink whose top two palette matches are within the soft margin."""

    @property
    def size(self) -> int:
        return int(self.centers_lab.shape[0])

    def pairwise_delta_e(self) -> np.ndarray:
        diff = self.centers_lab[:, None, :] - self.centers_lab[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def to_dict(self) -> dict[str, Any]:
        matrix = self.pairwise_delta_e()
        off_diagonal = matrix + np.eye(self.size) * 1e6
        return {
            "size": self.size,
            "source": self.source,
            "entries": [
                {
                    "palette_index": index,
                    "legend_label": self.labels[index],
                    "seed_rgb": self.seed_rgb[index],
                    "refined_rgb": [int(v) for v in self.centers_rgb[index]],
                    "display_rgb": [int(v) for v in self.display_rgb[index]],
                    "refined_lab": [round(float(v), 2) for v in self.centers_lab[index]],
                    "layer_px": int(self.layers[index].sum()),
                    "nearest_other_delta_e": round(float(off_diagonal[index].min()), 2),
                }
                for index in range(self.size)
            ],
            "min_pairwise_delta_e": round(float(off_diagonal.min()), 2) if self.size > 1 else None,
        }


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB)[0]


def _legend_seeds(legend_items: list[dict[str, Any]] | None) -> tuple[np.ndarray | None, list[str | None]]:
    seeds: list[list[int]] = []
    labels: list[str | None] = []
    for item in legend_items or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("swatch_color_rgb") or item.get("target_color_rgb")
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            continue
        try:
            rgb = [max(0, min(255, int(round(float(v))))) for v in raw]
        except (TypeError, ValueError):
            continue
        seeds.append(rgb)
        label = item.get("legend_label")
        labels.append(label.strip() if isinstance(label, str) and label.strip() else None)
    if not seeds:
        return None, []
    return np.asarray(seeds, dtype=np.uint8), labels


def _kmeans_centers(samples: np.ndarray, k: int) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _compactness, _labels, centers = cv2.kmeans(samples, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)


def _auto_k(samples: np.ndarray, params: ColorLabParams) -> np.ndarray:
    """Pick k by the elbow of within-cluster distortion; charts rarely need more than 8 series."""
    best: tuple[float, np.ndarray] | None = None
    previous = None
    for k in range(1, min(params.palette_max_k, len(samples)) + 1):
        centers = _kmeans_centers(samples, k)
        distances = np.linalg.norm(samples[:, None, :] - centers[None], axis=2).min(axis=1)
        distortion = float(distances.mean())
        if previous is not None and previous - distortion < 1.5:
            break
        previous = distortion
        best = (distortion, centers)
    return best[1] if best else _kmeans_centers(samples, 1)


def build_palette(
    canvas: Canvas,
    legend_items: list[dict[str, Any]] | None,
    params: ColorLabParams,
) -> Palette:
    samples = canvas.unmixed_lab[canvas.curve_ink]
    if samples.size == 0:
        raise ValueError("no curve ink left after canvas cleaning")
    weights = canvas.alpha[canvas.curve_ink]

    seed_rgb, labels = _legend_seeds(legend_items)
    if seed_rgb is not None and len(seed_rgb) >= 1:
        centers = _rgb_to_lab(seed_rgb).copy()
        source = "legend_swatch_seeded_lloyd"
        seeds_out: list[list[int] | None] = [[int(v) for v in row] for row in seed_rgb]
    else:
        centers = _auto_k(samples, params)
        source = "unsupervised_kmeans_lab"
        labels = [None] * len(centers)
        seeds_out = [None] * len(centers)

    # alpha-weighted Lloyd refinement: pull the legend blend onto the real ink colour.
    # Anchored to the seed, because two near-hue series (a blue and a violet) otherwise
    # collapse onto one another and the extracted curves get each other's legend label.
    for _ in range(max(0, params.palette_refine_iterations)):
        distances = np.linalg.norm(samples[:, None, :] - centers[None], axis=2)
        assignment = distances.argmin(axis=1)
        moved = 0.0
        for index in range(len(centers)):
            selected = assignment == index
            if not selected.any():
                continue
            weight = weights[selected][:, None]
            updated = (samples[selected] * weight).sum(axis=0) / max(1e-6, float(weight.sum()))
            others = [centers[other] for other in range(len(centers)) if other != index]
            if others and min(float(np.linalg.norm(updated - other)) for other in others) < params.palette_min_separation_delta_e:
                # Taking this step would let two series share a colour. A badly blended
                # legend swatch is worth fixing; merging two legend entries is not.
                continue
            moved = max(moved, float(np.linalg.norm(updated - centers[index])))
            centers[index] = updated
        if moved < 0.25:
            break

    distances = np.linalg.norm(samples[:, None, :] - centers[None], axis=2)
    order = np.argsort(distances, axis=1)
    best = distances[np.arange(len(samples)), order[:, 0]]
    runner_up = (
        distances[np.arange(len(samples)), order[:, 1]]
        if len(centers) > 1
        else np.full(len(samples), np.inf, dtype=np.float32)
    )

    height, width = canvas.height, canvas.width
    coords = np.nonzero(canvas.curve_ink)
    best_index = np.full((height, width), -1, np.int32)
    matched = best <= params.palette_max_match_delta_e
    best_index[coords[0][matched], coords[1][matched]] = order[matched, 0]

    contested = np.zeros((height, width), bool)
    ambiguous = matched & ((runner_up - best) < params.soft_layer_delta_e)
    contested[coords[0][ambiguous], coords[1][ambiguous]] = True

    layers = np.zeros((len(centers), height, width), bool)
    within = distances <= (best[:, None] + params.soft_layer_delta_e)
    within &= distances <= params.palette_max_match_delta_e
    for index in range(len(centers)):
        selected = within[:, index]
        layers[index][coords[0][selected], coords[1][selected]] = True

    if seed_rgb is not None:
        match_centers = _rgb_to_lab(seed_rgb).astype(np.float32)
        display_rgb = np.asarray(seed_rgb, dtype=np.uint8)
    else:
        match_centers = centers.astype(np.float32)
        display_rgb = _lab_to_rgb(centers)

    return Palette(
        centers_lab=centers.astype(np.float32),
        centers_rgb=_lab_to_rgb(centers),
        match_centers_lab=match_centers,
        display_rgb=display_rgb,
        labels=list(labels) + [None] * (len(centers) - len(labels)),
        seed_rgb=list(seeds_out) + [None] * (len(centers) - len(seeds_out)),
        source=source,
        layers=layers,
        best_index=best_index,
        contested=contested,
    )

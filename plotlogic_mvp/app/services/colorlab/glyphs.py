"""S3/S4 - find marker glyphs by scale-space template matching, then type them.

Markers are the measurements: one glyph is one data point.  Detection runs on every
soft colour layer with a filled-disk and a ring template, so filled and hollow markers
are found by the same pass.  The marker radius is chosen by reconstruction IoU rather
than from a distance transform, because a distance transform on a hollow marker
measures the stroke width, not the marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .canvas import Canvas
from .config import ColorLabParams
from .palette import Palette

FILLED = "filled"
OPEN = "open"
AMBIGUOUS = "ambiguous"


@dataclass
class Glyph:
    x: float
    y: float
    radius: float
    palette_index: int
    glyph_type: str
    fill_ratio: float
    ring_ratio: float
    score: float
    colour_delta_e: float
    colour_margin_delta_e: float
    ink_px: int
    lab: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Median CIELAB of this glyph's own ink - the vote that decides its series colour."""
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_px": round(self.x, 2),
            "y_px": round(self.y, 2),
            "radius_px": round(self.radius, 2),
            "palette_index": self.palette_index,
            "glyph_type": self.glyph_type,
            "fill_ratio": round(self.fill_ratio, 3),
            "ring_ratio": round(self.ring_ratio, 3),
            "score": round(self.score, 3),
            "colour_delta_e": round(self.colour_delta_e, 2),
            "colour_margin_delta_e": round(self.colour_margin_delta_e, 2),
            "ink_px": self.ink_px,
            "flags": list(self.flags),
        }


@dataclass
class GlyphDetection:
    glyphs: list[Glyph]
    radius: float
    radius_scores: dict[int, float]
    reconstruction_iou: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "glyph_count": len(self.glyphs),
            "radius_px": round(self.radius, 2),
            "reconstruction_iou": round(self.reconstruction_iou, 3),
            "radius_search": {str(k): round(v, 4) for k, v in sorted(self.radius_scores.items())},
            "glyphs": [glyph.to_dict() for glyph in self.glyphs],
        }


def _disk(radius: int) -> np.ndarray:
    """Slightly inset disk: a filled marker stays fully covered even if the radius
    estimate is a pixel optimistic."""
    size = 2 * radius + 1
    kernel = np.zeros((size, size), np.float32)
    cv2.circle(kernel, (radius, radius), max(1, int(round(radius * 0.85))), 1.0, -1)
    return kernel


def _ring(radius: int) -> np.ndarray:
    """One-pixel-thin circle.  Run against ink dilated by one pixel this measures the
    *angular* hit rate, so a hollow marker is found whatever its outline thickness -
    an area-based ring kernel would miss thin outlines entirely."""
    size = 2 * radius + 1
    kernel = np.zeros((size, size), np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, 1)
    return kernel


def _coverage(mask_f32: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    total = float(kernel.sum())
    if total <= 0:
        return np.zeros_like(mask_f32)
    return cv2.filter2D(mask_f32, -1, kernel / total, borderType=cv2.BORDER_CONSTANT)


def stroke_scale(mask: np.ndarray) -> float:
    """Half-width of the thinnest strokes - connecting lines and hollow marker outlines.

    The marker search starts above this so a plain line can never score as a marker.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dilated = cv2.dilate(distance, np.ones((3, 3), np.uint8))
    ridge = distance[(distance >= dilated - 1e-6) & mask & (distance > 0.5)]
    return float(np.median(ridge)) if ridge.size else 1.0


def _pick_peaks(
    score: np.ndarray,
    valid: np.ndarray,
    radius: int,
    params: ColorLabParams,
) -> list[tuple[int, int, float]]:
    spacing = max(1, int(round(radius * params.nms_spacing_factor)))
    # A 3x3 local max keeps every marker in a dense chain; a window as wide as the NMS
    # spacing would suppress a whole run of equally-good markers down to one.
    dilated = cv2.dilate(score, np.ones((3, 3), np.uint8))
    candidate = (score >= dilated - 1e-6) & (score > params.peak_score_threshold) & valid
    ys, xs = np.nonzero(candidate)
    if ys.size == 0:
        return []
    values = score[ys, xs]
    order = np.argsort(-values)
    claimed = np.zeros(score.shape, np.uint8)
    kept: list[tuple[int, int, float]] = []
    for index in order:
        y, x = int(ys[index]), int(xs[index])
        if claimed[y, x]:
            continue
        kept.append((y, x, float(values[index])))
        cv2.circle(claimed, (x, y), spacing, 1, -1)
    return kept


def _detect_at_radius(
    palette: Palette,
    radius: int,
    params: ColorLabParams,
) -> tuple[list[tuple[int, int, float, float, float, int]], np.ndarray]:
    """Return raw peaks for one radius plus the reconstruction of the ink they explain."""
    disk_kernel, ring_kernel = _disk(radius), _ring(radius)
    height, width = palette.best_index.shape
    reconstruction = np.zeros((height, width), np.uint8)
    peaks: list[tuple[int, int, float, float, float, int]] = []
    for index in range(palette.size):
        layer = palette.layers[index]
        if not layer.any():
            continue
        layer_f32 = layer.astype(np.float32)
        tolerant = cv2.dilate(layer.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        disk_response = _coverage(layer_f32, disk_kernel)
        ring_response = _coverage(tolerant, ring_kernel)
        score = np.maximum(disk_response, ring_response)
        # A hollow marker's centre is background, so candidates live near ink, not on it.
        valid = cv2.dilate(
            layer.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
        ).astype(bool)
        for y, x, value in _pick_peaks(score, valid, radius, params):
            disk_value = float(disk_response[y, x])
            ring_value = float(ring_response[y, x])
            peaks.append((y, x, value, disk_value, ring_value, index))
            if disk_value >= ring_value:
                cv2.circle(reconstruction, (x, y), radius, 1, -1)
            else:
                cv2.circle(reconstruction, (x, y), radius, 1, max(2, int(round(radius * 0.45))))
    return peaks, reconstruction.astype(bool)


def _select_radius(
    canvas: Canvas,
    palette: Palette,
    params: ColorLabParams,
) -> tuple[int, dict[int, float], float]:
    target = canvas.curve_ink
    target_px = int(target.sum())
    scores: dict[int, float] = {}
    # Anything at or below the stroke scale is a line, not a marker: searching there
    # makes the detector tile the connecting lines with tiny disks.
    lower = max(params.radius_search_min, int(np.ceil(stroke_scale(target) * params.radius_min_stroke_factor)))
    best: tuple[float, int] = (-1.0, lower)
    for radius in range(lower, params.radius_search_max + 1):
        _peaks, reconstruction = _detect_at_radius(palette, radius, params)
        intersection = int((reconstruction & target).sum())
        union = int((reconstruction | target).sum())
        iou = intersection / max(1, union)
        scores[radius] = iou
        if iou > best[0]:
            best = (iou, radius)
    if target_px == 0:
        return params.radius_search_min, scores, 0.0
    return best[1], scores, best[0]


def _footprint_indices(radius: float, core: bool, params: ColorLabParams) -> tuple[np.ndarray, np.ndarray]:
    reach = int(np.ceil(radius * params.ring_outer_factor)) + 1
    grid = np.arange(-reach, reach + 1)
    dy, dx = np.meshgrid(grid, grid, indexing="ij")
    distance = np.sqrt(dy * dy + dx * dx)
    if core:
        selected = distance <= radius * params.core_radius_factor
    else:
        selected = (distance >= radius * params.ring_inner_factor) & (distance <= radius * params.ring_outer_factor)
    return dy[selected], dx[selected]


def detect_glyphs(canvas: Canvas, palette: Palette, params: ColorLabParams) -> GlyphDetection:
    radius, radius_scores, iou = _select_radius(canvas, palette, params)
    raw_peaks, _reconstruction = _detect_at_radius(palette, radius, params)
    raw_peaks.sort(key=lambda item: -item[2])

    # A glyph found on two overlapping soft layers is one glyph; keep the strongest.
    dedupe_radius = max(1.0, radius * params.dedupe_distance_factor)
    claimed = np.zeros(canvas.curve_ink.shape, np.uint8)
    unique: list[tuple[int, int, float, float, float, int]] = []
    for peak in raw_peaks:
        y, x = peak[0], peak[1]
        if claimed[y, x]:
            continue
        unique.append(peak)
        cv2.circle(claimed, (x, y), int(round(dedupe_radius)), 1, -1)

    core_dy, core_dx = _footprint_indices(radius, True, params)
    ring_dy, ring_dx = _footprint_indices(radius, False, params)
    height, width = canvas.curve_ink.shape
    glyphs: list[Glyph] = []

    for y, x, score, _disk_value, _ring_value, layer_index in unique:
        core_y, core_x = y + core_dy, x + core_dx
        ring_y, ring_x = y + ring_dy, x + ring_dx
        core_valid = (core_y >= 0) & (core_y < height) & (core_x >= 0) & (core_x < width)
        ring_valid = (ring_y >= 0) & (ring_y < height) & (ring_x >= 0) & (ring_x < width)
        core_y, core_x = core_y[core_valid], core_x[core_valid]
        ring_y, ring_x = ring_y[ring_valid], ring_x[ring_valid]
        if core_y.size == 0:
            continue

        core_ink = canvas.curve_ink[core_y, core_x]
        ring_ink = canvas.curve_ink[ring_y, ring_x] if ring_y.size else np.zeros(0, bool)
        fill_ratio = float(core_ink.mean())
        ring_ratio = float(ring_ink.mean()) if ring_ink.size else 0.0

        if fill_ratio >= params.filled_min_fill_ratio:
            glyph_type = FILLED
            sample_y, sample_x = core_y[core_ink], core_x[core_ink]
        elif fill_ratio < params.open_max_fill_ratio and ring_ratio >= params.open_min_ring_ratio:
            glyph_type = OPEN
            sample_y, sample_x = ring_y[ring_ink], ring_x[ring_ink]
        else:
            glyph_type = AMBIGUOUS
            merged_y = np.concatenate([core_y[core_ink], ring_y[ring_ink]]) if ring_y.size else core_y[core_ink]
            merged_x = np.concatenate([core_x[core_ink], ring_x[ring_ink]]) if ring_x.size else core_x[core_ink]
            sample_y, sample_x = merged_y, merged_x

        if sample_y.size < params.min_glyph_ink_px:
            continue

        # Sub-pixel centre: alpha-weighted centroid of the glyph's own ink.
        weights = canvas.alpha[sample_y, sample_x].astype(np.float64)
        total = float(weights.sum())
        centre_x = float((sample_x * weights).sum() / total) if total > 0 else float(x)
        centre_y = float((sample_y * weights).sum() / total) if total > 0 else float(y)

        # Colour is decided here, by ~100-pixel vote, not per pixel.  Only the most
        # opaque half of the glyph's ink votes: anti-aliased edge pixels carry the
        # paper colour mixed in and would drag a shaded 3-D marker off its own hue.
        opacity = canvas.alpha[sample_y, sample_x]
        core = opacity >= np.median(opacity)
        lab = canvas.unmixed_lab[sample_y[core], sample_x[core]]
        median_lab = np.median(lab, axis=0).astype(np.float32)
        distances = np.linalg.norm(palette.centers_lab - median_lab[None], axis=1)
        order = np.argsort(distances)
        palette_index = int(order[0])
        delta_e = float(distances[order[0]])
        margin = float(distances[order[1]] - distances[order[0]]) if palette.size > 1 else float("inf")

        flags: list[str] = []
        if margin < params.colour_margin_delta_e:
            flags.append("colour_ambiguous")
        if glyph_type == AMBIGUOUS:
            flags.append("glyph_type_ambiguous")
        if palette_index != layer_index:
            flags.append("relabelled_by_glyph_vote")

        glyphs.append(
            Glyph(
                x=centre_x,
                y=centre_y,
                radius=float(radius),
                palette_index=palette_index,
                glyph_type=glyph_type,
                fill_ratio=fill_ratio,
                ring_ratio=ring_ratio,
                score=float(score),
                colour_delta_e=delta_e,
                colour_margin_delta_e=margin if np.isfinite(margin) else 999.0,
                ink_px=int(sample_y.size),
                lab=(float(median_lab[0]), float(median_lab[1]), float(median_lab[2])),
                flags=flags,
            )
        )

    glyphs.sort(key=lambda glyph: (glyph.palette_index, glyph.x))
    return GlyphDetection(
        glyphs=glyphs,
        radius=float(radius),
        radius_scores=radius_scores,
        reconstruction_iou=float(iou),
    )

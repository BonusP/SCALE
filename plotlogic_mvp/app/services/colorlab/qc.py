"""S8 - QC gates and overlays.

The point of the gates is that a bad extraction must announce itself.  A silently
plausible CSV built from a broken trace is worse than no CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .assemble import Series
from .canvas import Canvas
from .config import ColorLabParams
from .glyphs import FILLED, GlyphDetection
from .palette import Palette


def build_qc_report(
    canvas: Canvas,
    palette: Palette,
    detection: GlyphDetection,
    series_list: list[Series],
    params: ColorLabParams,
) -> dict[str, Any]:
    ink_px = max(1, int(canvas.ink.sum()))
    curve_px = max(1, int(canvas.curve_ink.sum()))
    contested_ratio = float(canvas.curve_ink.sum() and palette.contested.sum() / curve_px)

    radii = [glyph.radius for glyph in detection.glyphs]
    radius_cv = float(np.std(radii) / np.mean(radii)) if radii and np.mean(radii) > 0 else 0.0

    if series_list:
        xs = [x for series in series_list for x, _y in series.points_px]
        x_coverage = (max(xs) - min(xs)) / max(1.0, float(canvas.width)) if xs else 0.0
    else:
        x_coverage = 0.0

    ambiguous = sum(1 for glyph in detection.glyphs if "colour_ambiguous" in glyph.flags)
    ambiguous_ratio = ambiguous / max(1, len(detection.glyphs))

    metrics = {
        "structural_ink_ratio": round(int(canvas.structural.sum()) / ink_px, 4),
        "contested_pixel_ratio": round(contested_ratio, 4),
        "glyph_radius_cv": round(radius_cv, 4),
        "x_coverage": round(x_coverage, 4),
        "colour_ambiguous_ratio": round(ambiguous_ratio, 4),
        "reconstruction_iou": round(detection.reconstruction_iou, 4),
        "glyph_count": len(detection.glyphs),
        "series_count": len(series_list),
        "min_points_per_series": min((len(s.points_px) for s in series_list), default=0),
    }

    gates: list[dict[str, Any]] = []

    def gate(name: str, value: float, limit: float, comparison: str, message: str) -> None:
        failed = value > limit if comparison == "max" else value < limit
        gates.append(
            {"gate": name, "value": value, "limit": limit, "comparison": comparison, "passed": not failed, "message": message}
        )

    gate("structural_ink_ratio", metrics["structural_ink_ratio"], params.qc_max_structural_ratio, "max",
         "structural-ink removal may have eaten straight curve segments")
    gate("contested_pixel_ratio", metrics["contested_pixel_ratio"], params.qc_max_contested_ratio, "max",
         "palette entries are too close to separate reliably")
    gate("glyph_radius_cv", metrics["glyph_radius_cv"], params.qc_max_radius_cv, "max",
         "marker radius estimate is unstable or marker sizes are mixed")
    gate("x_coverage", metrics["x_coverage"], params.qc_min_x_coverage, "min",
         "detected series cover only part of the plot width")
    gate("colour_ambiguous_ratio", metrics["colour_ambiguous_ratio"], params.qc_max_colour_ambiguous_ratio, "max",
         "many glyphs sit between two palette colours")
    gate("min_points_per_series", metrics["min_points_per_series"], params.min_points_per_series, "min",
         "at least one series has too few points to be trustworthy")

    failed = [item for item in gates if not item["passed"]]
    return {
        "schema_version": 1,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [item["gate"] for item in failed],
        "low_confidence": bool(failed),
    }


def _bgr(colour_rgb: list[int]) -> tuple[int, int, int]:
    return int(colour_rgb[2]), int(colour_rgb[1]), int(colour_rgb[0])


def draw_colorlab_overlay(canvas: Canvas, series_list: list[Series], out_path: Path) -> None:
    base = canvas.rgb[:, :, ::-1].copy()
    faded = cv2.addWeighted(base, 0.35, np.full_like(base, 255), 0.65, 0)
    for series in series_list:
        colour = _bgr(series.colour_rgb)
        points = [(int(round(x)), int(round(y))) for x, y in series.points_px]
        if len(points) > 1:
            cv2.polylines(faded, [np.array(points, np.int32)], False, colour, 2, cv2.LINE_AA)
        marker = cv2.MARKER_CROSS if series.glyph_type == FILLED else cv2.MARKER_TILTED_CROSS
        for (x, y), flags in zip(points, series.point_flags):
            cv2.drawMarker(faded, (x, y), (30, 30, 30) if "outlier" in flags else colour, marker, 9, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), faded)


def draw_layer_map(canvas: Canvas, palette: Palette, out_path: Path) -> None:
    image = np.full((canvas.height, canvas.width, 3), 255, np.uint8)
    for index in range(palette.size):
        mask = (palette.best_index == index)
        image[mask] = palette.centers_rgb[index][::-1]
    image[canvas.structural] = (200, 200, 200)
    cv2.imwrite(str(out_path), image)


def draw_legacy_overlay(canvas: Canvas, traces: list[dict[str, Any]], out_path: Path) -> None:
    """Same rendering treatment as the colorlab overlay so the two are comparable."""
    base = canvas.rgb[:, :, ::-1].copy()
    faded = cv2.addWeighted(base, 0.35, np.full_like(base, 255), 0.65, 0)
    for trace in traces:
        raw = trace.get("series_color_rgb") or [220, 20, 60]
        colour = (int(raw[2]), int(raw[1]), int(raw[0]))
        points = [
            (int(round(p[0])), int(round(p[1])))
            for p in trace.get("points") or []
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        if len(points) > 1:
            cv2.polylines(faded, [np.array(points, np.int32)], False, colour, 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), faded)

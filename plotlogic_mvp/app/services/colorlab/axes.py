"""S6 - turn crop pixels into data values using the bboxes the vision/OCR pass found.

The tick bboxes and their OCR'd numbers already live in ``axis_mapping.json`` in
*panel* pixel space; the glyphs live in *crop* pixel space.  This module joins the two
and picks a linear or log10 model per axis by comparing fit residuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

ROLES = ("bottom", "left", "right")


@dataclass
class AxisModel:
    role: str
    kind: str
    """``linear``, ``log10`` or ``none``."""
    slope: float
    intercept: float
    r_squared: float
    pair_count: int
    label: str | None
    issues: list[str]

    def value(self, panel_px: float) -> float | None:
        if self.kind == "none":
            return None
        raw = self.slope * panel_px + self.intercept
        if self.kind == "log10":
            if raw > 300 or raw < -300:
                return None
            return float(10.0**raw)
        return float(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "kind": self.kind,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": round(self.r_squared, 5),
            "pair_count": self.pair_count,
            "axis_label": self.label,
            "issues": list(self.issues),
        }


@dataclass
class AxisFrame:
    offset_xy: tuple[int, int]
    models: dict[str, AxisModel]
    is_dual_y: bool
    x_role: str
    left_role: str
    right_role: str | None

    def x_value(self, crop_x: float) -> float | None:
        return self.models["bottom"].value(crop_x + self.offset_xy[0])

    def y_value(self, crop_y: float, axis: str) -> float | None:
        model = self.models.get(axis)
        return model.value(crop_y + self.offset_xy[1]) if model else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_offset_xy": list(self.offset_xy),
            "is_dual_y": self.is_dual_y,
            "quantities": {"x": self.x_role, "left": self.left_role, "right": self.right_role},
            "axes": {role: model.to_dict() for role, model in self.models.items()},
        }


def _fit(pixels: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    if len(pixels) < 2:
        return 0.0, 0.0, 0.0
    spread = float(((pixels - pixels.mean()) ** 2).sum())
    if spread == 0:
        return 0.0, 0.0, 0.0
    slope = float(((pixels - pixels.mean()) * (values - values.mean())).sum() / spread)
    intercept = float(values.mean() - slope * pixels.mean())
    predicted = slope * pixels + intercept
    residual = float(((values - predicted) ** 2).sum())
    total = float(((values - values.mean()) ** 2).sum())
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    return slope, intercept, r_squared


def _build_model(role: str, axis: dict[str, Any] | None) -> AxisModel:
    issues: list[str] = []
    label = None
    pairs: list[dict[str, float]] = []
    if isinstance(axis, dict):
        label = axis.get("axis_label") if isinstance(axis.get("axis_label"), str) else None
        for pair in axis.get("pairs") or []:
            if isinstance(pair, dict) and isinstance(pair.get("pixel"), (int, float)) and isinstance(pair.get("value"), (int, float)):
                if math.isfinite(float(pair["pixel"])) and math.isfinite(float(pair["value"])):
                    pairs.append({"pixel": float(pair["pixel"]), "value": float(pair["value"])})
    if len(pairs) < 2:
        issues.append("INSUFFICIENT_CALIBRATION_PAIRS")
        return AxisModel(role, "none", 0.0, 0.0, 0.0, len(pairs), label, issues)

    pixels = np.array([pair["pixel"] for pair in pairs], dtype=np.float64)
    values = np.array([pair["value"] for pair in pairs], dtype=np.float64)
    slope, intercept, r_squared = _fit(pixels, values)

    if (values > 0).all() and len(pairs) >= 3:
        log_slope, log_intercept, log_r2 = _fit(pixels, np.log10(values))
        if log_r2 > r_squared + 0.01 and log_r2 > 0.999:
            issues.append("LOG_AXIS_DETECTED")
            return AxisModel(role, "log10", log_slope, log_intercept, log_r2, len(pairs), label, issues)

    if r_squared < 0.99:
        issues.append("POOR_AXIS_FIT")
    return AxisModel(role, "linear", slope, intercept, r_squared, len(pairs), label, issues)


def build_axis_frame(
    axis_mapping: dict[str, Any],
    crop_meta: dict[str, Any],
    strategy: dict[str, Any] | None,
) -> AxisFrame:
    offset = crop_meta.get("crop_offset_xy") or [0, 0]
    offset_xy = (int(offset[0]), int(offset[1]))
    axes = axis_mapping.get("axes") if isinstance(axis_mapping.get("axes"), dict) else {}
    models = {role: _build_model(role, axes.get(role)) for role in ROLES}

    strategy = strategy or {}
    right_usable = models["right"].kind != "none"
    is_dual_y = bool(strategy.get("right_y_role")) and right_usable
    return AxisFrame(
        offset_xy=offset_xy,
        models=models,
        is_dual_y=is_dual_y,
        x_role=str(strategy.get("x_axis_role") or "x"),
        left_role=str(strategy.get("left_y_role") or "y"),
        right_role=str(strategy["right_y_role"]) if is_dual_y else None,
    )

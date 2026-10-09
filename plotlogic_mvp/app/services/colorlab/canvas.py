"""S0/S1 - build the curve-only canvas and un-blend anti-aliased ink.

The plot crop is ink alpha-composited over white paper.  Recovering that alpha and
inverting the blend turns every faint edge pixel back into its saturated series
colour, which is what makes a single perceptual threshold work later on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ColorLabParams


@dataclass
class Canvas:
    rgb: np.ndarray
    """Original crop, RGB uint8."""
    masked_rgb: np.ndarray
    """Crop with exclusion bboxes painted white, RGB uint8."""
    alpha: np.ndarray
    """Per-pixel ink coverage in [0, 1], float32."""
    unmixed_rgb: np.ndarray
    """Ink colour with the white paper divided out, RGB uint8."""
    unmixed_lab: np.ndarray
    """CIELAB of ``unmixed_rgb``, float32."""
    ink: np.ndarray
    """Boolean ink mask before structural removal."""
    structural: np.ndarray
    """Boolean mask of axis frame / ticks / gridlines."""
    curve_ink: np.ndarray
    """Boolean ink mask with structural ink removed - the 'curves only' image."""
    height: int
    width: int
    exclusion_bboxes: list[list[int]]

    def stats(self) -> dict[str, Any]:
        ink_px = int(self.ink.sum())
        return {
            "size_wh": [self.width, self.height],
            "ink_px": ink_px,
            "ink_ratio": round(ink_px / float(self.width * self.height), 4),
            "structural_px": int(self.structural.sum()),
            "structural_ratio_of_ink": round(int(self.structural.sum()) / max(1, ink_px), 4),
            "curve_ink_px": int(self.curve_ink.sum()),
            "exclusion_bbox_count": len(self.exclusion_bboxes),
        }


def _clamp_bbox(bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return None
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _structural_mask(ink_u8: np.ndarray, params: ColorLabParams) -> np.ndarray:
    """Axis frames, tick marks and gridlines are long unbroken runs; curves are not."""
    height, width = ink_u8.shape
    len_h = max(params.structural_kernel_min_px, int(width * params.structural_kernel_fraction))
    len_v = max(params.structural_kernel_min_px, int(height * params.structural_kernel_fraction))
    horiz = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (len_h, 1)))
    vert = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, len_v)))
    merged = cv2.bitwise_or(horiz, vert)
    radius = max(0, int(params.structural_dilate_px))
    if radius:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        merged = cv2.dilate(merged, kernel)
    return merged.astype(bool)


def build_canvas(
    crop_path: Path,
    exclusion_bboxes: list[Any],
    params: ColorLabParams,
) -> Canvas:
    bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read crop image: {crop_path}")
    rgb = bgr[:, :, ::-1].copy()
    height, width = rgb.shape[:2]

    masked = rgb.astype(np.float32)
    kept_bboxes: list[list[int]] = []
    for raw in exclusion_bboxes or []:
        bbox = _clamp_bbox(raw, width, height)
        if bbox is None:
            continue
        kept_bboxes.append(bbox)
        x1, y1, x2, y2 = bbox
        masked[y1:y2, x1:x2] = 255.0

    # P = a*C + (1-a)*255 with white paper.  Saturated plot inks have a near-zero
    # minimum channel, so the darkest channel carries the coverage.
    alpha = (255.0 - masked.min(axis=2)) / 255.0
    safe_alpha = np.clip(alpha, 1e-3, 1.0)[..., None]
    unmixed = np.clip((masked - (1.0 - safe_alpha) * 255.0) / safe_alpha, 0.0, 255.0).astype(np.uint8)

    ink = alpha > params.ink_alpha_threshold
    structural = _structural_mask(ink.astype(np.uint8), params)
    curve_ink = ink & ~structural

    return Canvas(
        rgb=rgb,
        masked_rgb=masked.astype(np.uint8),
        alpha=alpha.astype(np.float32),
        unmixed_rgb=unmixed,
        unmixed_lab=cv2.cvtColor(unmixed, cv2.COLOR_RGB2LAB).astype(np.float32),
        ink=ink,
        structural=structural,
        curve_ink=curve_ink,
        height=height,
        width=width,
        exclusion_bboxes=kept_bboxes,
    )


def write_debug_images(canvas: Canvas, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_only = np.where(canvas.curve_ink[..., None], canvas.masked_rgb, 255).astype(np.uint8)
    written = {
        "canvas.png": canvas.masked_rgb,
        "curve_only.png": curve_only,
        "unmixed.png": np.where(canvas.curve_ink[..., None], canvas.unmixed_rgb, 255).astype(np.uint8),
    }
    for name, image in written.items():
        cv2.imwrite(str(out_dir / name), image[:, :, ::-1])
    cv2.imwrite(str(out_dir / "alpha.png"), (canvas.alpha * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "structural_mask.png"), (canvas.structural * 255).astype(np.uint8))
    return {
        "canvas": "canvas.png",
        "curve_only": "curve_only.png",
        "unmixed": "unmixed.png",
        "alpha": "alpha.png",
        "structural_mask": "structural_mask.png",
    }

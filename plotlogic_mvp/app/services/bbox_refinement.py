from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _bbox(raw: Any) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in raw):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in raw]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _normalize_axis_role(role: Any) -> str:
    normalized = str(role or "").strip().lower()
    aliases = {
        "right_y": "right",
        "y_right": "right",
        "right_axis": "right",
        "left_y": "left",
        "y_left": "left",
        "bottom_x": "bottom",
        "x_bottom": "bottom",
        "x": "bottom",
    }
    return aliases.get(normalized, normalized)


def _ink_bbox(image: Image.Image, search_bbox: list[int], *, min_alpha_pixels: int = 8) -> list[int] | None:
    width, height = image.size
    clipped = _clamp_bbox(search_bbox, width, height)
    if not clipped:
        return None
    x1, y1, x2, y2 = clipped
    rgb = np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"))
    if rgb.size == 0:
        return None
    gray = rgb.mean(axis=2)
    channel_spread = rgb.max(axis=2) - rgb.min(axis=2)
    # Catch black text and colored legend glyphs while ignoring white paper.
    mask = (gray < 205) | ((gray < 245) & (channel_spread > 22))
    ys, xs = np.where(mask)
    if len(xs) < min_alpha_pixels:
        return None
    return [int(xs.min() + x1), int(ys.min() + y1), int(xs.max() + x1 + 1), int(ys.max() + y1 + 1)]


def _union(bboxes: list[list[int] | None]) -> list[int] | None:
    boxes = [bbox for bbox in bboxes if bbox]
    if not boxes:
        return None
    return [
        min(bbox[0] for bbox in boxes),
        min(bbox[1] for bbox in boxes),
        max(bbox[2] for bbox in boxes),
        max(bbox[3] for bbox in boxes),
    ]


def _pad(bbox: list[int], pad_x: int, pad_y: int, width: int, height: int) -> list[int] | None:
    return _clamp_bbox([bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y], width, height)


def _refine_text_bbox(image: Image.Image, bbox: list[int], *, pad_x: int = 10, pad_y: int = 6) -> list[int] | None:
    width, height = image.size
    search = _pad(bbox, pad_x, pad_y, width, height)
    if not search:
        return None
    refined = _ink_bbox(image, search)
    if not refined:
        return bbox
    return _pad(refined, 2, 2, width, height) or refined


def _refine_right_axis_label(image: Image.Image, axis: dict[str, Any], plot_bbox: list[int] | None) -> None:
    label_bbox = _bbox(axis.get("label_bbox_panel_px"))
    if not label_bbox:
        return
    width, height = image.size
    axis_line = _bbox(axis.get("axis_line_bbox_panel_px"))
    left_bound = label_bbox[0] - 18
    if plot_bbox:
        left_bound = max(left_bound, plot_bbox[2] + 28)
    elif axis_line:
        left_bound = max(left_bound, axis_line[2] + 22)
    search = _clamp_bbox(
        [
            int(left_bound),
            label_bbox[1] - 28,
            min(width, label_bbox[2] + 46),
            label_bbox[3] + 28,
        ],
        width,
        height,
    )
    if not search:
        return
    refined = _ink_bbox(image, search, min_alpha_pixels=16)
    if refined:
        axis["label_bbox_panel_px"] = _pad(refined, 3, 3, width, height) or refined


def refine_chart_element_bboxes(panel_image: Image.Image, vision: dict[str, Any]) -> dict[str, Any]:
    plot_bbox = _bbox(vision.get("plot_frame_bbox_panel_px"))

    for axis in vision.get("axes", []):
        if not isinstance(axis, dict):
            continue
        role = _normalize_axis_role(axis.get("axis_role"))
        axis["axis_role"] = role
        if role == "right":
            _refine_right_axis_label(panel_image, axis, plot_bbox)
        else:
            label_bbox = _bbox(axis.get("label_bbox_panel_px"))
            if label_bbox:
                refined = _refine_text_bbox(panel_image, label_bbox, pad_x=12, pad_y=8)
                if refined:
                    axis["label_bbox_panel_px"] = refined

    legend_boxes: list[list[int] | None] = []
    for item in vision.get("legend_items", []):
        if not isinstance(item, dict):
            continue
        label_bbox = _bbox(item.get("label_bbox_panel_px"))
        if label_bbox:
            refined = _refine_text_bbox(panel_image, label_bbox, pad_x=12, pad_y=8)
            if refined:
                item["label_bbox_panel_px"] = refined
                legend_boxes.append(refined)
        swatch_bbox = _bbox(item.get("swatch_bbox_panel_px"))
        if swatch_bbox:
            refined = _refine_text_bbox(panel_image, swatch_bbox, pad_x=5, pad_y=5)
            if refined:
                item["swatch_bbox_panel_px"] = refined
                legend_boxes.append(refined)

    legend_region = _bbox(vision.get("legend_region_bbox_panel_px"))
    if legend_boxes:
        width, height = panel_image.size
        union = _union([legend_region, *legend_boxes])
        if union:
            vision["legend_region_bbox_panel_px"] = _pad(union, 5, 5, width, height) or union

    return vision

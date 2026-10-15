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


def _union_bboxes(bboxes: list[list[int]]) -> list[int] | None:
    valid = [_bbox(bbox) for bbox in bboxes]
    valid = [bbox for bbox in valid if bbox]
    if not valid:
        return None
    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]


def _iter_component_bboxes(panel: dict[str, Any]) -> list[list[int]]:
    bboxes: list[list[int]] = []
    plot_bbox = _bbox(panel.get("plot_frame_bbox_panel_px"))
    if plot_bbox:
        bboxes.append(plot_bbox)
    for axis in panel.get("axes", []):
        if not isinstance(axis, dict):
            continue
        for key in ("axis_line_bbox_panel_px", "label_bbox_panel_px"):
            bbox = _bbox(axis.get(key))
            if bbox:
                bboxes.append(bbox)
        for tick in axis.get("ticks", []):
            if not isinstance(tick, dict):
                continue
            for key in ("tick_bbox_panel_px", "label_bbox_panel_px"):
                bbox = _bbox(tick.get(key))
                if bbox:
                    bboxes.append(bbox)
    legend_region = _bbox(panel.get("legend_region_bbox_panel_px"))
    if legend_region:
        bboxes.append(legend_region)
    for item in panel.get("legend_items", []):
        if not isinstance(item, dict):
            continue
        for key in ("label_bbox_panel_px", "swatch_bbox_panel_px"):
            bbox = _bbox(item.get(key))
            if bbox:
                bboxes.append(bbox)
    for region in panel.get("plot_annotation_regions", []):
        if isinstance(region, dict):
            bbox = _bbox(region.get("bbox_panel_px"))
            if bbox:
                bboxes.append(bbox)
    return bboxes


def _bbox_inside(inner: list[int], outer: list[int], tolerance: int = 4) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _geometry_appears_panel_local(panel: dict[str, Any], initial_bbox: list[int]) -> bool:
    bboxes = _iter_component_bboxes(panel)
    if not bboxes:
        return True
    initial_w = max(1, initial_bbox[2] - initial_bbox[0])
    initial_h = max(1, initial_bbox[3] - initial_bbox[1])
    local_bounds = [0, 0, initial_w, initial_h]

    # The plot frame is the most reliable signal. If it fits the explicit panel
    # bbox but not the local crop bounds, all panel geometry is source-space.
    plot_bbox = _bbox(panel.get("plot_frame_bbox_panel_px"))
    if plot_bbox:
        plot_in_initial = _bbox_inside(plot_bbox, initial_bbox)
        plot_in_local = _bbox_inside(plot_bbox, local_bounds)
        if plot_in_initial and not plot_in_local:
            return False
        if plot_in_local and not plot_in_initial:
            return True

    source_only = sum(1 for bbox in bboxes if _bbox_inside(bbox, initial_bbox) and not _bbox_inside(bbox, local_bounds))
    local_only = sum(1 for bbox in bboxes if _bbox_inside(bbox, local_bounds) and not _bbox_inside(bbox, initial_bbox))
    if source_only:
        return False
    if local_only:
        return True
    return True


def _bbox_to_source_space(
    raw: Any,
    initial_bbox: list[int],
    image_width: int,
    image_height: int,
    *,
    panel_local_geometry: bool,
) -> list[int] | None:
    bbox = _bbox(raw)
    if not bbox:
        return None
    if not panel_local_geometry:
        return _clamp_bbox(bbox, image_width, image_height)

    initial_w = max(1, initial_bbox[2] - initial_bbox[0])
    initial_h = max(1, initial_bbox[3] - initial_bbox[1])
    local_bounds = [0, 0, initial_w, initial_h]
    source_bounds = [0, 0, image_width, image_height]

    if _bbox_inside(bbox, local_bounds):
        shifted = [
            bbox[0] + initial_bbox[0],
            bbox[1] + initial_bbox[1],
            bbox[2] + initial_bbox[0],
            bbox[3] + initial_bbox[1],
        ]
        return _clamp_bbox(shifted, image_width, image_height)
    if _bbox_inside(bbox, source_bounds):
        return bbox
    return None


def _component_bboxes_source_space(
    panel: dict[str, Any],
    initial_bbox: list[int],
    image_width: int,
    image_height: int,
    *,
    panel_local_geometry: bool,
) -> list[list[int]]:
    bboxes: list[list[int]] = []
    for bbox in _iter_component_bboxes(panel):
        source_bbox = _bbox_to_source_space(
            bbox,
            initial_bbox,
            image_width,
            image_height,
            panel_local_geometry=panel_local_geometry,
        )
        if source_bbox:
            bboxes.append(source_bbox)
    return bboxes


def _complete_chart_bbox_from_frame(
    frame: list[int],
    component_bboxes: list[list[int]],
    width: int,
    height: int,
) -> list[int] | None:
    frame_w = max(1, frame[2] - frame[0])
    frame_h = max(1, frame[3] - frame[1])
    union = _union_bboxes([frame, *component_bboxes]) or frame

    complete = [
        min(union[0], frame[0] - max(90, int(round(frame_w * 0.30)))),
        min(union[1], frame[1] - max(60, int(round(frame_h * 0.24)))),
        max(union[2], frame[2] + max(150, int(round(frame_w * 0.45)))),
        max(union[3], frame[3] + max(180, int(round(frame_h * 0.72)))),
    ]
    return _clamp_bbox(complete, width, height)


def _complete_chart_bbox_from_components(
    frame: list[int],
    component_bboxes: list[list[int]],
    width: int,
    height: int,
) -> list[int] | None:
    outside_frame = [
        bbox
        for bbox in component_bboxes
        if (
            bbox[0] < frame[0] - 4
            or bbox[1] < frame[1] - 4
            or bbox[2] > frame[2] + 4
            or bbox[3] > frame[3] + 4
        )
    ]
    if len(outside_frame) < 2:
        return None

    frame_w = max(1, frame[2] - frame[0])
    frame_h = max(1, frame[3] - frame[1])
    union = _union_bboxes([frame, *component_bboxes])
    if not union:
        return None

    union_w = max(1, union[2] - union[0])
    union_h = max(1, union[3] - union[1])
    pad_x = max(10, int(round(frame_w * 0.035)), int(round(union_w * 0.02)))
    pad_y = max(10, int(round(frame_h * 0.035)), int(round(union_h * 0.02)))
    complete = [
        union[0] - pad_x,
        union[1] - pad_y,
        union[2] + pad_x,
        union[3] + pad_y,
    ]

    # Component bboxes from the vision pass are better panel boundaries than a
    # large fixed expansion. Keep the crop compact so multi-panel figures do not
    # bleed into neighboring charts, while still allowing normal axis labels.
    limits = [
        frame[0] - max(85, int(round(frame_w * 0.25))),
        frame[1] - max(60, int(round(frame_h * 0.18))),
        frame[2] + max(120, int(round(frame_w * 0.30))),
        frame[3] + max(140, int(round(frame_h * 0.34))),
    ]
    limited = [
        max(complete[0], limits[0]),
        max(complete[1], limits[1]),
        min(complete[2], limits[2]),
        min(complete[3], limits[3]),
    ]
    return _clamp_bbox(limited, width, height)


def _intersection_area(a: list[int], b: list[int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _frame_matches_hint(frame: list[int], hint: list[int]) -> bool:
    frame_area = max(1, _area(frame))
    hint_area = max(1, _area(hint))
    overlap = _intersection_area(frame, hint) / min(frame_area, hint_area)
    if overlap >= 0.45:
        return True
    frame_w = max(1, frame[2] - frame[0])
    frame_h = max(1, frame[3] - frame[1])
    hint_w = max(1, hint[2] - hint[0])
    hint_h = max(1, hint[3] - hint[1])
    center_dx = abs((frame[0] + frame[2]) / 2 - (hint[0] + hint[2]) / 2)
    center_dy = abs((frame[1] + frame[3]) / 2 - (hint[1] + hint[3]) / 2)
    return center_dx <= max(frame_w, hint_w) * 0.20 and center_dy <= max(frame_h, hint_h) * 0.20


def _groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    grouped: list[tuple[int, int]] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value <= prev + 1:
            prev = value
            continue
        grouped.append((start, prev))
        start = prev = value
    grouped.append((start, prev))
    return grouped


def _group_center(group: tuple[int, int]) -> int:
    return int(round((group[0] + group[1]) / 2))


def _best_frame(gray: np.ndarray, search_bbox: list[int], initial_bbox: list[int]) -> list[int] | None:
    sx1, sy1, sx2, sy2 = search_bbox
    roi = gray[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    mask = roi < 180
    roi_h, roi_w = mask.shape
    if roi_h < 80 or roi_w < 80:
        return None

    vertical_start = max(0, int(round(roi_h * 0.12)))
    vertical_mask = mask[vertical_start:, :]
    col_counts = vertical_mask.sum(axis=0)
    min_vertical = max(80, int(round(vertical_mask.shape[0] * 0.38)))
    col_indices = [idx for idx, count in enumerate(col_counts) if count >= min_vertical]
    col_groups = _groups(col_indices)
    if len(col_groups) < 2:
        return None

    initial_w = max(1, initial_bbox[2] - initial_bbox[0])
    initial_center = (initial_bbox[0] + initial_bbox[2]) / 2
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_score = -1.0
    for left in col_groups:
        for right in col_groups:
            left_x = _group_center(left) + sx1
            right_x = _group_center(right) + sx1
            frame_w = right_x - left_x
            if frame_w < max(120, initial_w * 0.45):
                continue
            left_peak = int(col_counts[left[0] : left[1] + 1].max())
            right_peak = int(col_counts[right[0] : right[1] + 1].max())
            pair_center = (left_x + right_x) / 2
            center_penalty = abs(pair_center - initial_center)
            score = min(left_peak, right_peak) * 3.0 + frame_w * 0.2 - center_penalty * 0.5
            if score > best_score:
                best_score = score
                best_pair = (left, right)
    if not best_pair:
        return None

    left_x = _group_center(best_pair[0]) + sx1
    right_x = _group_center(best_pair[1]) + sx1
    local_left = max(0, left_x - sx1)
    local_right = min(roi_w, right_x - sx1 + 1)
    if local_right <= local_left:
        return None

    horizontal_mask = mask[:, local_left:local_right]
    row_counts = horizontal_mask.sum(axis=1)
    frame_w = max(1, local_right - local_left)
    min_horizontal = max(80, int(round(frame_w * 0.72)))
    row_indices = [idx for idx, count in enumerate(row_counts) if count >= min_horizontal]
    row_groups = _groups(row_indices)
    if len(row_groups) < 2:
        return None

    best_rows: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_row_score = -1.0
    expected_top_y = initial_bbox[1] + max(40, int(round((initial_bbox[3] - initial_bbox[1]) * 0.20)))
    min_top_y = expected_top_y
    for top in row_groups:
        for bottom in row_groups:
            top_y = _group_center(top) + sy1
            bottom_y = _group_center(bottom) + sy1
            frame_h = bottom_y - top_y
            if top_y < min_top_y:
                continue
            if frame_h < 120:
                continue
            if frame_h > (initial_bbox[3] - initial_bbox[1]) * 0.85:
                continue
            if frame_h > (sy2 - sy1) * 0.8:
                continue
            score = frame_h - abs(top_y - expected_top_y) * 1.2
            if score > best_row_score:
                best_row_score = score
                best_rows = (top, bottom)
    if not best_rows:
        return None

    top_y = _group_center(best_rows[0]) + sy1
    bottom_y = _group_center(best_rows[1]) + sy1
    if right_x <= left_x or bottom_y <= top_y:
        return None
    return [left_x, top_y, right_x, bottom_y]


def repair_chart_panel_geometry(
    image: Image.Image,
    panel: dict[str, Any],
    initial_bbox: list[int],
) -> dict[str, Any]:
    width, height = image.size
    clipped = _clamp_bbox(initial_bbox, width, height) or [0, 0, width, height]
    panel_type = str(panel.get("panel_type") or "").strip().lower()
    if panel_type != "chart":
        return {"panel_bbox": clipped, "plot_frame_source_px": None, "repair_applied": False}

    panel_local_geometry = _geometry_appears_panel_local(panel, clipped)
    component_bboxes = _component_bboxes_source_space(
        panel,
        clipped,
        width,
        height,
        panel_local_geometry=panel_local_geometry,
    )
    plot_frame_hint = _bbox_to_source_space(
        panel.get("plot_frame_bbox_panel_px"),
        clipped,
        width,
        height,
        panel_local_geometry=panel_local_geometry,
    )
    bbox_w = max(1, clipped[2] - clipped[0])
    bbox_h = max(1, clipped[3] - clipped[1])
    search_bbox = [
        clipped[0] - max(40, int(round(bbox_w * 0.08))),
        clipped[1] - max(30, int(round(bbox_h * 0.08))),
        clipped[2] + max(160, int(round(bbox_w * 0.55))),
        clipped[3] + max(180, int(round(bbox_h * 0.65))),
    ]
    search_bbox = _clamp_bbox(search_bbox, width, height)
    if not search_bbox:
        return {"panel_bbox": clipped, "plot_frame_source_px": None, "repair_applied": False}

    gray = np.asarray(image.convert("L"))
    frame = _best_frame(gray, search_bbox, clipped)
    detected_frame = frame
    if frame and plot_frame_hint and not _frame_matches_hint(frame, plot_frame_hint):
        detected_frame = None
    frame_source = detected_frame or plot_frame_hint
    if not frame_source:
        component_union = _union_bboxes(component_bboxes)
        if not component_union:
            return {"panel_bbox": clipped, "plot_frame_source_px": None, "repair_applied": False}
        pad_x = max(45, int(round((component_union[2] - component_union[0]) * 0.10)))
        pad_y = max(55, int(round((component_union[3] - component_union[1]) * 0.14)))
        repaired = _clamp_bbox(
            [
                component_union[0] - pad_x,
                component_union[1] - pad_y,
                component_union[2] + pad_x,
                component_union[3] + pad_y,
            ],
            width,
            height,
        ) or clipped
        return {
            "panel_bbox": repaired,
            "plot_frame_source_px": None,
            "repair_applied": repaired != clipped,
        }

    repaired = (
        _complete_chart_bbox_from_components(frame_source, component_bboxes, width, height)
        or _complete_chart_bbox_from_frame(frame_source, component_bboxes, width, height)
        or clipped
    )
    return {
        "panel_bbox": repaired,
        "plot_frame_source_px": detected_frame,
        "repair_applied": repaired != clipped,
    }

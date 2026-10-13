from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .. import config
from .bbox_refinement import refine_chart_element_bboxes
from .chart_geometry import repair_chart_panel_geometry
from .categorical_tracing import categorical_series  # categorical 처리를 위한 수정
from .codex_direct import read_categorical_table, to_categorical_series_data, validation_report
from .ocr_backend import read_text_cached
from .ocr_geometry import build_ocr_text_geometry
from .visual_qa import ensure_visual_qa_artifacts


if str(config.PLOTLOGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PLOTLOGIC_ROOT))

from core.data_extraction_pipeline_cmsl import run_data_extraction_for_panel as run_cmsl_extraction  # noqa: E402
from .curve_axis_association import assign_curve_axes


def _bbox(raw: Any) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in raw):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in raw]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_to_crop(raw: Any, plot_bbox: list[int]) -> list[int] | None:
    bbox = _bbox(raw)
    if not bbox:
        return None
    ox, oy = plot_bbox[0], plot_bbox[1]
    return [bbox[0] - ox, bbox[1] - oy, bbox[2] - ox, bbox[3] - oy]


def _normalize_legend_axis_hint(raw: Any) -> str | None:
    normalized = str(raw or "").strip().lower()
    aliases = {
        "left": "left_y",
        "left_y": "left_y",
        "y_left": "left_y",
        "right": "right_y",
        "right_y": "right_y",
        "y_right": "right_y",
    }
    return aliases.get(normalized)


def _normalize_axis_role(role: Any) -> str:
    normalized = str(role or "").strip().lower()
    aliases = {
        "x": "bottom",
        "bottom_x": "bottom",
        "x_bottom": "bottom",
        "bottom_axis": "bottom",
        "left_y": "left",
        "y_left": "left",
        "left_axis": "left",
        "right_y": "right",
        "y_right": "right",
        "right_axis": "right",
    }
    return aliases.get(normalized, normalized)


def _union_bboxes(bboxes: list[list[int]]) -> list[int] | None:
    valid = [_bbox(bbox) for bbox in bboxes]
    valid = [bbox for bbox in valid if bbox]
    if not valid:
        return None
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def _pad_bbox(bbox: list[int], pad_x: int, pad_y: int) -> list[int]:
    return [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y]


def _is_legend_annotation(region: dict[str, Any]) -> bool:
    text = f"{region.get('annotation_text') or ''} {region.get('reason') or ''}".lower()
    return "legend" in text


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _offset_bbox(raw: Any, offset_xy: list[int]) -> list[int] | None:
    bbox = _bbox(raw)
    if not bbox:
        return None
    ox, oy = offset_xy
    return [bbox[0] - ox, bbox[1] - oy, bbox[2] - ox, bbox[3] - oy]


def _derive_panel_geometry(vision: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    width, height = image.size
    panel_bbox_hint = _bbox(vision.get("panel_bbox_figure_px"))
    if panel_bbox_hint:
        initial = _clamp_bbox(panel_bbox_hint, width, height) or [0, 0, width, height]
        return repair_chart_panel_geometry(image, vision, initial)

    bboxes: list[list[int]] = []
    plot_bbox = _bbox(vision.get("plot_frame_bbox_panel_px"))
    if plot_bbox:
        bboxes.append(plot_bbox)
    for axis in vision.get("axes", []):
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
    for item in vision.get("legend_items", []):
        if isinstance(item, dict):
            for key in ("label_bbox_panel_px", "swatch_bbox_panel_px"):
                bbox = _bbox(item.get(key))
                if bbox:
                    bboxes.append(bbox)
    legend_region = _bbox(vision.get("legend_region_bbox_panel_px"))
    if legend_region:
        bboxes.append(legend_region)
    for region in vision.get("plot_annotation_regions", []):
        if isinstance(region, dict):
            bbox = _bbox(region.get("bbox_panel_px"))
            if bbox:
                bboxes.append(bbox)

    panel_bbox = _union_bboxes(bboxes) or [0, 0, width, height]
    pad_x = max(12, int(round((panel_bbox[2] - panel_bbox[0]) * 0.04)))
    pad_y = max(12, int(round((panel_bbox[3] - panel_bbox[1]) * 0.04)))
    padded = [
        panel_bbox[0] - pad_x,
        panel_bbox[1] - pad_y,
        panel_bbox[2] + pad_x,
        panel_bbox[3] + pad_y,
    ]
    panel_bbox = _clamp_bbox(padded, width, height) or [0, 0, width, height]
    return repair_chart_panel_geometry(image, vision, panel_bbox)


def _normalize_axis_roles(vision: dict[str, Any]) -> None:
    for axis in vision.get("axes", []):
        if isinstance(axis, dict):
            axis["axis_role"] = _normalize_axis_role(axis.get("axis_role"))


def _axis_dimension(role: str) -> str:
    role = _normalize_axis_role(role)
    return "x" if role == "bottom" else "y"


def _offset_tick_position(role: str, value: Any, offset_xy: list[int]) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value) - float(offset_xy[0] if _axis_dimension(role) == "x" else offset_xy[1])


def _offset_axis_line_coordinate(role: str, value: Any, offset_xy: list[int]) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return _offset_tick_position(role, value, offset_xy) or 0.0


def _iter_geometry_bboxes(vision: dict[str, Any]) -> list[list[int]]:
    bboxes: list[list[int]] = []
    plot_bbox = _bbox(vision.get("plot_frame_bbox_panel_px"))
    if plot_bbox:
        bboxes.append(plot_bbox)
    for axis in vision.get("axes", []):
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
    legend_region = _bbox(vision.get("legend_region_bbox_panel_px"))
    if legend_region:
        bboxes.append(legend_region)
    for item in vision.get("legend_items", []):
        if not isinstance(item, dict):
            continue
        for key in ("label_bbox_panel_px", "swatch_bbox_panel_px"):
            bbox = _bbox(item.get(key))
            if bbox:
                bboxes.append(bbox)
    for region in vision.get("plot_annotation_regions", []):
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


def _geometry_appears_panel_local(vision: dict[str, Any], panel_bbox: list[int]) -> bool:
    bboxes = _iter_geometry_bboxes(vision)
    if not bboxes:
        return True
    panel_w = max(1, panel_bbox[2] - panel_bbox[0])
    panel_h = max(1, panel_bbox[3] - panel_bbox[1])
    local_bounds = [0, 0, panel_w, panel_h]
    source_like = sum(1 for bbox in bboxes if _bbox_inside(bbox, panel_bbox))
    local_like = sum(1 for bbox in bboxes if _bbox_inside(bbox, local_bounds))
    outside_local = any(not _bbox_inside(bbox, local_bounds) for bbox in bboxes)
    if outside_local and source_like:
        return False
    if panel_bbox[0] or panel_bbox[1]:
        return local_like >= source_like
    return True


def _vision_to_panel_coordinates(
    vision: dict[str, Any],
    offset_xy: list[int],
    *,
    preserve_panel_local_geometry: bool = False,
) -> dict[str, Any]:
    adjusted = json.loads(json.dumps(vision))
    panel_bbox = _offset_bbox(adjusted.get("panel_bbox_figure_px"), offset_xy)
    if panel_bbox:
        adjusted["panel_bbox_figure_px"] = panel_bbox
    if preserve_panel_local_geometry:
        _normalize_axis_roles(adjusted)
        return adjusted

    plot_bbox = _offset_bbox(adjusted.get("plot_frame_bbox_panel_px"), offset_xy)
    if plot_bbox:
        adjusted["plot_frame_bbox_panel_px"] = plot_bbox
    origin = adjusted.get("origin_panel_px")
    if isinstance(origin, list) and len(origin) == 2:
        adjusted["origin_panel_px"] = [
            int(round(origin[0] - offset_xy[0])),
            int(round(origin[1] - offset_xy[1])),
        ]
    for axis in adjusted.get("axes", []):
        if not isinstance(axis, dict):
            continue
        role = _normalize_axis_role(axis.get("axis_role"))
        axis["axis_role"] = role
        for key in ("axis_line_bbox_panel_px", "label_bbox_panel_px"):
            bbox = _offset_bbox(axis.get(key), offset_xy)
            if bbox:
                axis[key] = bbox
        axis["axis_line_coordinate_px"] = _offset_axis_line_coordinate(
            role,
            axis.get("axis_line_coordinate_px"),
            offset_xy,
        )
        for tick in axis.get("ticks", []):
            if not isinstance(tick, dict):
                continue
            for key in ("tick_bbox_panel_px", "label_bbox_panel_px"):
                bbox = _offset_bbox(tick.get(key), offset_xy)
                if bbox:
                    tick[key] = bbox
            tick_position = _offset_tick_position(role, tick.get("tick_position_px"), offset_xy)
            if tick_position is not None:
                tick["tick_position_px"] = tick_position
    legend_bbox = _offset_bbox(adjusted.get("legend_region_bbox_panel_px"), offset_xy)
    if legend_bbox:
        adjusted["legend_region_bbox_panel_px"] = legend_bbox
    for item in adjusted.get("legend_items", []):
        if not isinstance(item, dict):
            continue
        for key in ("label_bbox_panel_px", "swatch_bbox_panel_px"):
            bbox = _offset_bbox(item.get(key), offset_xy)
            if bbox:
                item[key] = bbox
    for region in adjusted.get("plot_annotation_regions", []):
        if isinstance(region, dict):
            bbox = _offset_bbox(region.get("bbox_panel_px"), offset_xy)
            if bbox:
                region["bbox_panel_px"] = bbox
    return adjusted


def _remap_tick_positions_to_frame(axis: dict[str, Any], frame: list[int]) -> None:
    ticks = [tick for tick in axis.get("ticks", []) if isinstance(tick, dict)]
    values = [
        float(tick["label_numeric_value"])
        for tick in ticks
        if isinstance(tick.get("label_numeric_value"), (int, float))
    ]
    if len(values) < 2:
        return
    role = _normalize_axis_role(axis.get("axis_role"))
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return
    if role == "bottom" and len(values) >= 4:
        ordered = sorted(set(values))
        steps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        if steps:
            median_step = sorted(steps)[len(steps) // 2]
            old_positions = [
                float(tick["tick_position_px"])
                for tick in ticks
                if isinstance(tick.get("tick_position_px"), (int, float))
            ]
            old_span = max(old_positions) - min(old_positions) if len(old_positions) >= 2 else 0
            frame_span = max(1, frame[2] - frame[0])
            if old_span / frame_span < 0.75:
                max_value += median_step
    for tick in ticks:
        value = tick.get("label_numeric_value")
        if not isinstance(value, (int, float)):
            continue
        fraction = (float(value) - min_value) / (max_value - min_value)
        if role == "bottom":
            tick["tick_position_px"] = frame[0] + fraction * (frame[2] - frame[0])
        else:
            tick["tick_position_px"] = frame[3] - fraction * (frame[3] - frame[1])


def _apply_detected_plot_frame(vision: dict[str, Any], frame_crop_px: list[int]) -> None:
    vision["plot_frame_bbox_panel_px"] = frame_crop_px
    vision["origin_panel_px"] = [frame_crop_px[0], frame_crop_px[3]]
    for axis in vision.get("axes", []):
        if not isinstance(axis, dict):
            continue
        role = _normalize_axis_role(axis.get("axis_role"))
        axis["axis_role"] = role
        if role == "bottom":
            axis["axis_line_coordinate_px"] = frame_crop_px[3]
            axis["axis_line_bbox_panel_px"] = [frame_crop_px[0], frame_crop_px[3] - 1, frame_crop_px[2], frame_crop_px[3] + 2]
        elif role == "left":
            axis["axis_line_coordinate_px"] = frame_crop_px[0]
            axis["axis_line_bbox_panel_px"] = [frame_crop_px[0] - 2, frame_crop_px[1], frame_crop_px[0] + 2, frame_crop_px[3]]
        elif role == "right":
            axis["axis_line_coordinate_px"] = frame_crop_px[2]
            axis["axis_line_bbox_panel_px"] = [frame_crop_px[2] - 2, frame_crop_px[1], frame_crop_px[2] + 2, frame_crop_px[3]]
        _remap_tick_positions_to_frame(axis, frame_crop_px)


def prepare_dataset(
    job_id: str,
    paper_id: int,
    chart_id: str,
    image_path: Path,
    vision: dict[str, Any],
) -> tuple[Path, str, str, Path, dict[str, Any], list[int]]:
    dataset_root = config.DATA_ROOT / "jobs" / job_id / "plotlogic_dataset"
    panel_dir = dataset_root / "panels" / f"{paper_id:04d}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_name = f"{chart_id}_chart{image_path.suffix.lower() or '.png'}"
    panel_path = panel_dir / panel_name
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        has_explicit_panel_bbox = _bbox(vision.get("panel_bbox_figure_px")) is not None
        panel_geometry = _derive_panel_geometry(vision, rgb)
        panel_bbox = list(panel_geometry["panel_bbox"])
        rgb.crop(tuple(panel_bbox)).save(panel_path)
    rel_path = panel_path.relative_to(dataset_root).as_posix()
    detected_frame = panel_geometry.get("plot_frame_source_px")
    preserve_panel_local_geometry = (
        has_explicit_panel_bbox
        and not detected_frame
        and _geometry_appears_panel_local(vision, panel_bbox)
    )
    adjusted_vision = _vision_to_panel_coordinates(
        vision,
        [panel_bbox[0], panel_bbox[1]],
        preserve_panel_local_geometry=preserve_panel_local_geometry,
    )
    if isinstance(detected_frame, list) and len(detected_frame) == 4:
        frame_crop = [
            int(detected_frame[0] - panel_bbox[0]),
            int(detected_frame[1] - panel_bbox[1]),
            int(detected_frame[2] - panel_bbox[0]),
            int(detected_frame[3] - panel_bbox[1]),
        ]
        _apply_detected_plot_frame(adjusted_vision, frame_crop)
    with Image.open(panel_path) as panel_image:
        refine_chart_element_bboxes(panel_image.convert("RGB"), adjusted_vision)
    return dataset_root, chart_id, rel_path, panel_path, adjusted_vision, panel_bbox


def overrides_from_vision(panel_id: str, panel_rel_path: str, vision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    plot_bbox = _bbox(vision.get("plot_frame_bbox_panel_px")) or [0, 0, 1, 1]
    axis_labels = vision.get("axis_labels") if isinstance(vision.get("axis_labels"), dict) else {}
    axes = [axis for axis in vision.get("axes", []) if isinstance(axis, dict)]

    axis_bbox_by_role: dict[str, list[int] | None] = {}
    tick_axes: list[dict[str, Any]] = []
    for axis in axes:
        role = _normalize_axis_role(axis.get("axis_role"))
        axis["axis_role"] = role
        if role not in {"left", "bottom", "right"}:
            continue
        axis_line_bbox = _bbox(axis.get("axis_line_bbox_panel_px"))
        label_bbox = _bbox(axis.get("label_bbox_panel_px"))
        tick_label_bboxes = [
            bbox
            for tick in axis.get("ticks", [])
            if isinstance(tick, dict)
            for bbox in [_bbox(tick.get("label_bbox_panel_px"))]
            if bbox
        ]
        axis_bbox_by_role[role] = _union_bboxes([b for b in [axis_line_bbox, label_bbox, *tick_label_bboxes] if b]) or axis_line_bbox

        candidates = []
        matches = []
        tick_positions = []
        for tick in axis.get("ticks", []):
            if not isinstance(tick, dict):
                continue
            label_bbox_tick = _bbox(tick.get("label_bbox_panel_px"))
            tick_position = tick.get("tick_position_px")
            if not label_bbox_tick or not isinstance(tick_position, (int, float)):
                continue
            tick_positions.append(float(tick_position))
            candidates.append(
                {
                    "text": str(tick.get("label_text") or ""),
                    "normalized_text": str(tick.get("label_text") or ""),
                    "numeric_value": tick.get("label_numeric_value"),
                    "bbox_xyxy": label_bbox_tick,
                    "ocr_source": "codex-vision",
                    "ocr_confidence": tick.get("confidence"),
                }
            )
            matches.append(
                {
                    "tick_position_px": float(tick_position),
                    "candidate_bbox_xyxy": label_bbox_tick,
                }
            )
        tick_axes.append(
            {
                "axis_role": role,
                "axis_bbox_xyxy": axis_bbox_by_role[role] or [0, 0, 1, 1],
                "search_bbox_xyxy": axis_bbox_by_role[role] or [0, 0, 1, 1],
                "axis_line_coordinate_px": axis.get("axis_line_coordinate_px") or 0,
                "tick_positions_px": tick_positions,
                "tick_count": len(tick_positions),
                "tick_label_candidates": candidates,
                "tick_matches": matches,
                "issues": [] if len(matches) >= 2 else ["INSUFFICIENT_SUPPORT"],
            }
        )

    plot_panel = {
        "panel_id": panel_id,
        "panel_path": panel_rel_path,
        "panel_kind": "chart",
        "review_status": "scratch",
        "gt_bbox_xyxy": plot_bbox,
        "annotator": "plotlogic_mvp_vision",
        "note": "job-local scratch override",
    }
    axes_panel = {
        "panel_id": panel_id,
        "panel_path": panel_rel_path,
        "panel_kind": "chart",
        # is_target_dual_y is scoped to the legacy current_voltage_power_dual_y family only (see the
        # "backward compatibility" rule in the prompt), so it is false for dual_y_generic_line_chart
        # panels even though they genuinely have two y axes. Fall back to "a right axis bbox was
        # actually returned" so every downstream reader of axes_panel["is_dual_y"] (axis_mapping's
        # y_axes list, chart_extraction_strategy's evidence, etc.) reflects real geometry rather than
        # that one family-specific flag.
        "is_dual_y": bool(vision.get("is_target_dual_y") or vision.get("is_dual_y")) or bool(axis_bbox_by_role.get("right")),
        "review_status": "scratch",
        "axis_left_bbox_xyxy": axis_bbox_by_role.get("left"),
        "axis_bottom_bbox_xyxy": axis_bbox_by_role.get("bottom"),
        "axis_right_bbox_xyxy": axis_bbox_by_role.get("right"),
        "axis_bottom_label": axis_labels.get("x", ""),
        "axis_left_label": axis_labels.get("left_y", ""),
        "axis_right_label": axis_labels.get("right_y", ""),
        # Which extraction_family Codex matched this panel to (e.g. current_voltage_power_dual_y,
        # dual_y_generic_line_chart, single_axis_line_chart). build_chart_extraction_strategy() uses
        # this to decide which family-specific behavior to turn on, instead of assuming every
        # right-axis chart is a current-voltage-power chart.
        "extraction_family": str(vision.get("extraction_family") or ""),
        "annotator": "plotlogic_mvp_vision",
        "note": "job-local scratch override",
    }
    ticks_panel = {
        "panel_id": panel_id,
        "panel_path": panel_rel_path,
        "panel_kind": "chart",
        "axes": tick_axes,
    }

    legend_items: list[dict[str, Any]] = []
    legend_item_exclusions: list[list[int]] = []
    for item in vision.get("legend_items", []):
        if not isinstance(item, dict):
            continue
        label_bbox_crop = _bbox_to_crop(item.get("label_bbox_panel_px"), plot_bbox)
        swatch_bbox_crop = _bbox_to_crop(item.get("swatch_bbox_panel_px"), plot_bbox)
        legend_items.append(
            {
                "legend_label": str(item.get("legend_label") or ""),
                "label_bbox_crop_px": label_bbox_crop or [0, 0, 0, 0],
                "swatch_color_rgb": item.get("swatch_color_rgb") or [0, 0, 0],
                "swatch_bbox_crop_px": swatch_bbox_crop or [0, 0, 0, 0],
                "marker_shape": item.get("marker_shape"),
                "confidence": item.get("confidence") or 0,
                # Vision's own left_y/right_y call for this series, straight from the legend text.
                # Lets the core engine bind a curve to its axis without guessing from pixel shape.
                "axis_binding_hint": _normalize_legend_axis_hint(item.get("axis_binding_hint")),
            }
        )
        item_bbox = _union_bboxes([bbox for bbox in [label_bbox_crop, swatch_bbox_crop] if bbox])
        if item_bbox:
            legend_item_exclusions.append(_pad_bbox(item_bbox, 2, 2))

    exclusions = list(legend_item_exclusions)
    for region in vision.get("plot_annotation_regions", []):
        if isinstance(region, dict):
            if _is_legend_annotation(region):
                continue
            bbox = _bbox_to_crop(region.get("bbox_panel_px"), plot_bbox)
            if bbox:
                exclusions.append(bbox)

    return plot_panel, axes_panel, ticks_panel, legend_items, exclusions


def export_curve_series_csv(curve_series_path: Path, csv_path: Path) -> None:
    payload = json.loads(curve_series_path.read_text(encoding="utf-8"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "panel_id",
                "trace_id",
                "series_id",
                "legend_label",
                "curve_role",
                "axis_binding",
                "point_index",
                "x",
                "y",
            ]
        )
        for series in payload.get("series", []):
            # 시리즈·조각 구분: CSV는 각 조각의 원래 trace_id를 유지합니다.
            for segment in series.get("segments") or [series]:
                for index, point in enumerate(segment.get("points_data") or []):
                    if not isinstance(point, list) or len(point) < 2:
                        continue
                    writer.writerow(
                        [
                            payload.get("panel_id"),
                            segment.get("trace_id"),
                            series.get("series_id"),
                            series.get("legend_label"),
                            series.get("curve_role"),
                            series.get("axis_binding"),
                            index,
                            point[0],
                            point[1],
                        ]
                    )


def export_categorical_csv(curve_series_path: Path, csv_path: Path) -> None:
    """CSV for categorical-line charts (a line/scatter chart with a categorical x axis).

    These have no curve, axis binding, or point ordering to speak of — a category name and its
    value is the whole data point — so this uses its own column layout instead of
    export_curve_series_csv's curve-oriented one (trace_id/curve_role/axis_binding/x/y). The source
    curve_series_data.json is still produced by to_categorical_series_data() in the same
    schema_version/panel_id/series[].points_data shape the other extraction methods use; only the
    CSV rendering differs, because "x"/"y" would be a misleading header for a category name and its
    value.
    """
    payload = json.loads(curve_series_path.read_text(encoding="utf-8"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel_id", "series_id", "legend_label", "category", "value"])
        for series in payload.get("series", []):
            for point in series.get("points_data") or []:
                if not isinstance(point, list) or len(point) < 2:
                    continue
                writer.writerow(
                    [
                        payload.get("panel_id"),
                        series.get("series_id"),
                        series.get("legend_label"),
                        point[0],
                        point[1],
                    ]
                )


# categorical 처리를 위한 수정: retain full pixel-extraction artifacts before table conversion.
def run_categorical_extraction(job_id, paper_id, chart_id, image_path, vision):
    result = run_extraction(job_id, paper_id, chart_id, image_path, vision, True)
    out = result["result_dir"]
    inputs = json.loads((out / "mvp_extraction_inputs.json").read_text(encoding="utf-8"))
    mapping = json.loads((out / "axis_mapping.json").read_text(encoding="utf-8"))
    traces = json.loads((out / "curve_traces_panel_px.json").read_text(encoding="utf-8"))
    # Read only accepted traces, grouping same-label fragments at each category position.
    accepted = [trace for trace in traces.get("traces", []) if trace.get("final_trace_status") == "accepted"]
    panel_vision = inputs["panel_vision"]
    series = categorical_series(chart_id, panel_vision, accepted, mapping)
    (out / "categorical_pixel_table.json").write_text(json.dumps(series, indent=2, ensure_ascii=False), encoding="utf-8")
    table = {"issues": series["issues"]}
    fallback = bool(series["issues"] or not series["series"])
    if fallback:
        table = read_categorical_table(result["panel_path"], overlay_path=out / "overlay.png",
                                       trace_context={"pixel_table": series, "traces": accepted})
        (out / "codex_categorical_table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
        codex_series = to_categorical_series_data(chart_id, table)
        if codex_series["series"]:
            series = codex_series
            series["source"] = "categorical_codex_after_pixel_tracing"
            series["categories"] = json.loads((out / "categorical_pixel_table.json").read_text(encoding="utf-8"))["categories"]
            colors = {item.get("legend_label"): item.get("series_color_rgb") for item in accepted}
            for item in series["series"]:
                item["series_color_rgb"] = colors.get(item["legend_label"])
        else:
            table["issues"] = list(table.get("issues", [])) + series["issues"]
    # Category order comes from image positions, then from Codex's item order if geometry is absent.
    order = [item["label"] for item in series.get("categories", [])]
    for item in series["series"]:
        for category, _value in item["points_data"]:
            if category not in order:
                order.append(category)
        item["points_data"].sort(key=lambda point: order.index(point[0]))
    series["category_order"] = order
    validation = validation_report(series, table)
    validation["checks"]["pixel_trace_count"] = len(accepted)
    validation["categorical_codex_fallback"] = fallback
    for name, payload in [("curve_series_data.json", series), ("validation_report.json", validation)]:
        (out / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    mapping["axes"]["bottom"]["categories"] = series.get("categories", [])
    mapping["axes"]["bottom"]["model"] = None
    mapping["axes"]["bottom"]["issues"] = []
    mapping["issues"] = [issue for issue in mapping["issues"] if not issue.startswith("bottom:")]
    (out / "axis_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    inputs["extraction"] = "categorical_pixel_tracing_with_codex_fallback"
    (out / "mvp_extraction_inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False), encoding="utf-8")
    # categorical 처리를 위한 수정: record table artifacts and method in the existing run manifest.
    result["manifest"]["extraction"] = inputs["extraction"]
    result["manifest"]["artifacts"]["categorical_pixel_table"] = "categorical_pixel_table.json"
    if fallback:
        result["manifest"]["artifacts"]["codex_categorical_table"] = "codex_categorical_table.json"
    (out / "run_manifest.json").write_text(json.dumps(result["manifest"], indent=2, ensure_ascii=False), encoding="utf-8")
    export_categorical_csv(out / "curve_series_data.json", result["csv_path"])
    result["series_count"] = len(series["series"])
    result["issues"] = validation["issues"]
    result["visual_qa"] = ensure_visual_qa_artifacts(out)
    return result


# CMSL extraction: no color-count limit, VLM axis association after tracking.
def run_extraction(
    job_id: str,
    paper_id: int,
    chart_id: str,
    image_path: Path,
    vision: dict[str, Any],
    use_ocr_text: bool = False,
) -> dict[str, Any]:
    # Snapshot the codex output before panel cropping, coordinate remapping and ink refinement
    # so the debug view can show the uncorrected bboxes.
    raw_codex_vision = json.loads(json.dumps(vision))
    dataset_root, panel_id, panel_rel_path, panel_path, panel_vision, panel_bbox = prepare_dataset(
        job_id,
        paper_id,
        chart_id,
        image_path,
        vision,
    )
    ocr_text_report: dict[str, Any] | None = None
    if use_ocr_text:
        ocr = read_text_cached(panel_path, panel_path.with_suffix(panel_path.suffix + ".ocr.json"))
        # CMSL uses individual OCR annotation boxes.
        applied = build_ocr_text_geometry(panel_vision, ocr_items=ocr.get("items") or [], refine_annotations=True)
        panel_vision = applied["panel_vision"]
        ocr_text_report = {
            "backend": ocr.get("backend"),
            "error": ocr.get("error"),
            "item_count": len(ocr.get("items") or []),
            "summary": applied["summary"],
            "issues": applied["issues"],
        }
    plot_panel, axes_panel, ticks_panel, legend_items, exclusions = overrides_from_vision(panel_id, panel_rel_path, panel_vision)
    out_root = config.DATA_ROOT / "jobs" / job_id / "plotlogic_runs"
    manifest = run_cmsl_extraction(
        dataset_root=dataset_root,
        panel_id=panel_id,
        out_root=out_root,
        manual_axis_pairs=None,
        max_colors=None,
        exclusion_bboxes_crop_px=exclusions,
        exclusion_source="plotlogic_mvp_vision",
        legend_items=legend_items,
        plot_panel_override=plot_panel,
        axes_panel_override=axes_panel,
        ticks_panel_override=ticks_panel,
        axis_binding_resolver=assign_curve_axes,
    )
    result_dir = out_root / panel_id
    debug_inputs = {
        "schema_version": 1,
        "job_id": job_id,
        "paper_id": paper_id,
        "panel_id": panel_id,
        "source_figure_image": str(image_path),
        "panel_path": str(panel_path),
        "panel_rel_path": panel_rel_path,
        "panel_crop_bbox_source_image_px": panel_bbox,
        "panel_vision_raw_codex": raw_codex_vision,
        "panel_vision_raw_codex_coordinate_space": (
            "panel_local_px"
            if _geometry_appears_panel_local(raw_codex_vision, panel_bbox)
            else "figure_px"
        ),
        "panel_vision": panel_vision,
        "ocr_text_applied": ocr_text_report,
        "plot_panel_override": plot_panel,
        "axes_panel_override": axes_panel,
        "ticks_panel_override": ticks_panel,
        "legend_items_crop_px": legend_items,
        "exclusion_bboxes_crop_px": exclusions,
    }
    (result_dir / "mvp_extraction_inputs.json").write_text(
        json.dumps(debug_inputs, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    csv_path = result_dir / "curve_series_data.csv"
    curve_series_path = result_dir / "curve_series_data.json"
    if curve_series_path.exists():
        export_curve_series_csv(curve_series_path, csv_path)
    visual_qa = ensure_visual_qa_artifacts(result_dir)
    validation = json.loads((result_dir / "validation_report.json").read_text(encoding="utf-8")) if (result_dir / "validation_report.json").exists() else {}
    checks = validation.get("checks") if isinstance(validation.get("checks"), dict) else {}
    return {
        "manifest": manifest,
        "result_dir": result_dir,
        "panel_path": panel_path,
        "overlay_path": result_dir / "overlay.png",
        "csv_path": csv_path,
        "series_count": int(checks.get("final_series_count") or 0),
        "visual_qa": visual_qa,
        "issues": (validation.get("issues") or []) + [f"PANEL_CROP_BBOX_SOURCE_IMAGE:{panel_bbox}"],
    }

# ###

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _center_x(bbox: list[int]) -> float:
    return (bbox[0] + bbox[2]) / 2


def _center_y(bbox: list[int]) -> float:
    return (bbox[1] + bbox[3]) / 2


def _bbox_inside(inner: list[int], outer: list[int], tolerance: int = 4) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _out_of_image(bbox: list[int], width: int, height: int, tolerance: int = 2) -> bool:
    return bbox[0] < -tolerance or bbox[1] < -tolerance or bbox[2] > width + tolerance or bbox[3] > height + tolerance


def _add_issue(panel: dict[str, Any], issue: str) -> None:
    issues = panel.setdefault("issues", [])
    if not isinstance(issues, list):
        issues = []
        panel["issues"] = issues
    if issue not in issues:
        issues.append(issue)


def _iter_component_bboxes(panel: dict[str, Any]) -> list[tuple[str, list[int] | None, Any]]:
    components: list[tuple[str, list[int] | None, Any]] = [
        ("plot_frame_bbox_panel_px", _bbox(panel.get("plot_frame_bbox_panel_px")), panel.get("plot_frame_bbox_panel_px")),
        ("legend_region_bbox_panel_px", _bbox(panel.get("legend_region_bbox_panel_px")), panel.get("legend_region_bbox_panel_px")),
    ]
    for axis_index, axis in enumerate(panel.get("axes", []), start=1):
        if not isinstance(axis, dict):
            continue
        role = str(axis.get("axis_role") or f"axis{axis_index}").strip().lower()
        components.append((f"axis:{role}:axis_line_bbox_panel_px", _bbox(axis.get("axis_line_bbox_panel_px")), axis.get("axis_line_bbox_panel_px")))
        components.append((f"axis:{role}:label_bbox_panel_px", _bbox(axis.get("label_bbox_panel_px")), axis.get("label_bbox_panel_px")))
        for tick_index, tick in enumerate(axis.get("ticks", []), start=1):
            if not isinstance(tick, dict):
                continue
            components.append((f"axis:{role}:tick{tick_index}:tick_bbox_panel_px", _bbox(tick.get("tick_bbox_panel_px")), tick.get("tick_bbox_panel_px")))
            components.append((f"axis:{role}:tick{tick_index}:label_bbox_panel_px", _bbox(tick.get("label_bbox_panel_px")), tick.get("label_bbox_panel_px")))
    for item_index, item in enumerate(panel.get("legend_items", []), start=1):
        if not isinstance(item, dict):
            continue
        components.append((f"legend_item:{item_index}:label_bbox_panel_px", _bbox(item.get("label_bbox_panel_px")), item.get("label_bbox_panel_px")))
        components.append((f"legend_item:{item_index}:swatch_bbox_panel_px", _bbox(item.get("swatch_bbox_panel_px")), item.get("swatch_bbox_panel_px")))
    for region_index, region in enumerate(panel.get("plot_annotation_regions", []), start=1):
        if isinstance(region, dict):
            components.append((f"annotation:{region_index}:bbox_panel_px", _bbox(region.get("bbox_panel_px")), region.get("bbox_panel_px")))
    return components


def _component_union(panel: dict[str, Any]) -> list[int] | None:
    boxes = [bbox for _, bbox, _ in _iter_component_bboxes(panel) if bbox]
    if not boxes:
        return None
    return [
        min(bbox[0] for bbox in boxes),
        min(bbox[1] for bbox in boxes),
        max(bbox[2] for bbox in boxes),
        max(bbox[3] for bbox in boxes),
    ]


def _axis_by_role(panel: dict[str, Any]) -> dict[str, dict[str, Any]]:
    axes: dict[str, dict[str, Any]] = {}
    for axis in panel.get("axes", []):
        if not isinstance(axis, dict):
            continue
        role = str(axis.get("axis_role") or "").strip().lower()
        aliases = {
            "x": "bottom",
            "bottom_x": "bottom",
            "x_bottom": "bottom",
            "left_y": "left",
            "y_left": "left",
            "right_y": "right",
            "y_right": "right",
        }
        role = aliases.get(role, role)
        if role:
            axes[role] = axis
    return axes


def _check_axis_alignment(panel: dict[str, Any], plot_bbox: list[int]) -> None:
    axes = _axis_by_role(panel)
    plot_w = max(1, plot_bbox[2] - plot_bbox[0])
    plot_h = max(1, plot_bbox[3] - plot_bbox[1])
    x_tol = max(18, int(round(plot_w * 0.06)))
    y_tol = max(18, int(round(plot_h * 0.06)))

    left = _bbox(axes.get("left", {}).get("axis_line_bbox_panel_px")) if axes.get("left") else None
    right = _bbox(axes.get("right", {}).get("axis_line_bbox_panel_px")) if axes.get("right") else None
    bottom = _bbox(axes.get("bottom", {}).get("axis_line_bbox_panel_px")) if axes.get("bottom") else None
    if left and abs(_center_x(left) - plot_bbox[0]) > x_tol:
        _add_issue(panel, "BBOX_AXIS_LEFT_NOT_ALIGNED_WITH_PLOT_FRAME")
    if right and abs(_center_x(right) - plot_bbox[2]) > x_tol:
        _add_issue(panel, "BBOX_AXIS_RIGHT_NOT_ALIGNED_WITH_PLOT_FRAME")
    if bottom and abs(_center_y(bottom) - plot_bbox[3]) > y_tol:
        _add_issue(panel, "BBOX_AXIS_BOTTOM_NOT_ALIGNED_WITH_PLOT_FRAME")

    for role, axis in axes.items():
        for tick in axis.get("ticks", []):
            if not isinstance(tick, dict):
                continue
            pos = tick.get("tick_position_px")
            if not isinstance(pos, (int, float)):
                _add_issue(panel, f"BBOX_TICK_POSITION_MISSING:{role}")
                continue
            if role == "bottom" and not (plot_bbox[0] - x_tol <= float(pos) <= plot_bbox[2] + x_tol):
                _add_issue(panel, "BBOX_BOTTOM_TICK_POSITION_OUTSIDE_PLOT_FRAME")
            if role in {"left", "right"} and not (plot_bbox[1] - y_tol <= float(pos) <= plot_bbox[3] + y_tol):
                _add_issue(panel, f"BBOX_{role.upper()}_TICK_POSITION_OUTSIDE_PLOT_FRAME")


def validate_figure_analysis(image_path: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError:
        return analysis

    panels = analysis.get("panels")
    if not isinstance(panels, list):
        return analysis

    for panel_index, panel in enumerate(panels, start=1):
        if not isinstance(panel, dict):
            continue
        panel_label = str(panel.get("panel_label") or panel_index)
        panel_bbox = _bbox(panel.get("panel_bbox_figure_px"))
        is_chart = bool(panel.get("is_chart")) or str(panel.get("panel_type") or "").strip().lower() == "chart"
        is_target = bool(panel.get("is_extraction_target") or panel.get("is_target_dual_y"))

        if not panel_bbox:
            _add_issue(panel, f"BBOX_INVALID_PANEL_BBOX:{panel_label}")
            continue
        if _out_of_image(panel_bbox, width, height):
            _add_issue(panel, f"BBOX_PANEL_OUT_OF_IMAGE:{panel_label}")

        component_union = _component_union(panel)
        if component_union and not _bbox_inside(component_union, panel_bbox, tolerance=10):
            _add_issue(panel, "BBOX_PANEL_TOO_TIGHT_FOR_CHART_COMPONENTS")

        for name, bbox, raw in _iter_component_bboxes(panel):
            if raw in (None, [0, 0, 0, 0]):
                continue
            if not bbox:
                _add_issue(panel, f"BBOX_INVALID:{name}")
                continue
            if _out_of_image(bbox, width, height):
                _add_issue(panel, f"BBOX_OUT_OF_IMAGE:{name}")
            local_bounds = [0, 0, max(1, panel_bbox[2] - panel_bbox[0]), max(1, panel_bbox[3] - panel_bbox[1])]
            if panel_bbox[0] or panel_bbox[1]:
                is_coordinate_anchor = name == "plot_frame_bbox_panel_px" or name.endswith(":axis_line_bbox_panel_px")
                if is_coordinate_anchor and _bbox_inside(bbox, local_bounds) and not _bbox_inside(bbox, panel_bbox):
                    _add_issue(panel, f"BBOX_COORD_SPACE_SUSPECT_PANEL_LOCAL:{name}")

        plot_bbox = _bbox(panel.get("plot_frame_bbox_panel_px"))
        if is_chart and not plot_bbox:
            _add_issue(panel, "BBOX_MISSING_PLOT_FRAME_FOR_CHART")
        if plot_bbox:
            if not _bbox_inside(plot_bbox, panel_bbox, tolerance=12):
                _add_issue(panel, "BBOX_PLOT_FRAME_OUTSIDE_PANEL_BBOX")
            _check_axis_alignment(panel, plot_bbox)

        if is_target:
            family = str(panel.get("extraction_family") or panel.get("chart_family") or "")
            # Categorical-line charts are read directly by codex (see codex_direct.py), not traced
            # from axis geometry, so the axis-presence checks below do not apply to them.
            if family not in {"categorical_line_chart"}:
                # Only a genuinely dual-y target needs a right axis; a single-axis chart never has one.
                is_dual_y_target = bool(panel.get("is_target_dual_y")) or "dual_y" in family
                required_roles = ("left", "bottom", "right") if is_dual_y_target else ("left", "bottom")
                axes = _axis_by_role(panel)
                for required_role in required_roles:
                    if required_role not in axes:
                        _add_issue(panel, f"BBOX_TARGET_MISSING_{required_role.upper()}_AXIS")
                if not panel.get("legend_items"):
                    _add_issue(panel, "BBOX_TARGET_MISSING_LEGEND_ITEMS")

    analysis["bbox_validation"] = {
        "schema_version": 1,
        "image_width": width,
        "image_height": height,
        "checked": True,
    }
    return analysis

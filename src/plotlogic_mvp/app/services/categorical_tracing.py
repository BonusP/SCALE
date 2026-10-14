"""categorical 처리를 위한 수정: convert pixel traces to category/value tables."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def categorical_series(panel_id: str, vision: dict, traces: list[dict], mapping: dict) -> dict:
    # categorical 처리를 위한 수정: category coordinates stay in panel pixels; no numeric x fit.
    bottom = next((axis for axis in vision.get("axes", []) if axis.get("axis_role") == "bottom"), {})
    categories = []
    for tick in bottom.get("ticks", []):
        label, pixel = str(tick.get("label_text") or "").strip(), tick.get("tick_position_px")
        if label and isinstance(pixel, (int, float)) and math.isfinite(pixel):
            categories.append({"label": label, "pixel": float(pixel)})
    categories.sort(key=lambda item: item["pixel"])
    if len({item["label"] for item in categories}) != len(categories):
        return {"panel_id": panel_id, "series": [], "categories": [], "issues": ["DUPLICATE_CATEGORY_LABELS"]}
    groups: dict[Any, list] = defaultdict(list)
    for trace in traces:
        groups[(trace.get("legend_label") or trace.get("series_id"), trace.get("axis_binding_hint"))].append(trace)
    series, issues = [], []
    if not categories:
        issues.append("MISSING_CATEGORY_POSITIONS")
    for (label, binding), fragments in groups.items():
        y_axis = "right" if binding == "right_y" else "left"
        model = mapping.get("axes", {}).get(y_axis, {}).get("model")
        if not model or binding not in {"left_y", "right_y"}:
            issues.append(f"MISSING_CATEGORY_Y_MAPPING:{label}")
            continue
        points, missing = [], []
        for category in categories:
            x = category["pixel"]
            samples = []
            for fragment in fragments:
                pixels = sorted(fragment.get("points", []))
                # Only read a locally supported position; never bridge a large hole or extrapolate.
                near = [(px, py) for px, py in pixels if abs(px - x) <= 3]
                if near:
                    samples.append(min(near, key=lambda point: abs(point[0] - x))[1])
            if not samples or max(samples) - min(samples) > 5:
                missing.append(category["label"])
                continue
            y = sum(samples) / len(samples)
            points.append([category["label"], model["slope"] * y + model["intercept"]])
        if missing:
            issues.append(f"UNRESOLVED_CATEGORY_VALUES:{label}:{','.join(missing)}")
        if points:
            first = fragments[0]
            series.append({
                "trace_id": first["trace_id"], "series_id": first["series_id"],
                "legend_label": label, "curve_role": "category_value", "axis_binding": binding,
                "series_color_rgb": first.get("series_color_rgb"), "points_data": points,
                "source_pixel_trace_refs": [item["trace_id"] for item in fragments], "issues": [],
            })
    expected = {item.get("legend_label") for item in vision.get("legend_items", []) if item.get("legend_label")}
    if expected - {item["legend_label"] for item in series}:
        issues.append("UNRESOLVED_CATEGORY_SERIES")
    return {"schema_version": 1, "panel_id": panel_id, "source": "categorical_pixel_tracing",
            "axis_labels": vision.get("axis_labels", {}), "categories": categories,
            "series": series, "issues": issues}

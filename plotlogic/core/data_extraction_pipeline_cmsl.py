from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

# ###수정 차이 08~10: 양자화 bin과 background/grid mask를 배열로 계산합니다.
import numpy as np
# ###

from PIL import Image, ImageDraw, ImageFilter, ImageStat


RGB = tuple[int, int, int]
Point = tuple[int, int]
ROLE_RECOLOR_RGB: dict[str, RGB] = {
    "voltage_curve": (0, 114, 178),
    "power_density_curve": (213, 94, 0),
    "data_curve": (0, 158, 115),
    "unknown_curve": (110, 110, 110),
}
ROLE_RECOLOR_PALETTES: dict[str, list[RGB]] = {
    "voltage_curve": [(0, 114, 178), (86, 180, 233), (0, 158, 115), (0, 80, 140)],
    "power_density_curve": [(213, 94, 0), (230, 159, 0), (204, 121, 167), (160, 70, 0)],
    "data_curve": [(0, 158, 115), (0, 120, 160), (120, 120, 0), (80, 120, 180)],
    "unknown_curve": [(110, 110, 110), (80, 80, 80), (140, 140, 140), (50, 50, 50)],
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _reset_panel_out_dir(panel_out: Path, out_root: Path) -> None:
    resolved_panel_out = panel_out.resolve()
    resolved_out_root = out_root.resolve()
    if resolved_panel_out == resolved_out_root or resolved_out_root not in resolved_panel_out.parents:
        raise ValueError(f"refusing to reset output directory outside out_root: {panel_out}")
    if panel_out.exists():
        shutil.rmtree(panel_out)
    panel_out.mkdir(parents=True, exist_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _index_by_panel_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    panels = payload.get("panels")
    if not isinstance(panels, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for panel in panels:
        if isinstance(panel, dict) and isinstance(panel.get("panel_id"), str):
            out[panel["panel_id"]] = panel
    return out


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, width, height]
    return [x1, y1, x2, y2]


def _normalize_exclusion_bboxes_crop_px(
    bboxes: list[list[int]] | None,
    width: int,
    height: int,
) -> list[list[int]]:
    normalized: list[list[int]] = []
    if not bboxes:
        return normalized
    for bbox in bboxes:
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not all(isinstance(value, (int, float)) for value in bbox):
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append([x1, y1, x2, y2])
    return normalized


def _point_in_bboxes(x: int, y: int, bboxes: list[list[int]]) -> bool:
    return any(x1 <= x < x2 and y1 <= y < y2 for x1, y1, x2, y2 in bboxes)


def _distance(a: RGB, b: RGB) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _saturation(rgb: RGB) -> float:
    r, g, b = [v / 255.0 for v in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx <= 0:
        return 0.0
    return (mx - mn) / mx


def _brightness(rgb: RGB) -> float:
    return sum(rgb) / 3.0


def _rgb_hsv(rgb: RGB) -> tuple[float, float, float]:
    r, g, b = [value / 255.0 for value in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    delta = mx - mn
    if delta == 0:
        hue = 0.0
    elif mx == r:
        hue = (60 * ((g - b) / delta) + 360) % 360
    elif mx == g:
        hue = 60 * ((b - r) / delta) + 120
    else:
        hue = 60 * ((r - g) / delta) + 240
    saturation = 0.0 if mx == 0 else delta / mx
    return hue, saturation, mx


def _hue_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _is_gray_like(rgb: RGB) -> bool:
    return _rgb_hsv(rgb)[1] < 0.12


def _matches_gray_seed(color: RGB, seed: RGB, *, max_distance: float = 80.0) -> bool:
    """Strict gray matching avoids pulling pastel anti-alias pixels from colored curves."""
    channel_spread = max(color) - min(color)
    brightness = _brightness(color)
    seed_brightness = _brightness(seed)
    if channel_spread > 24:
        return False
    # ###수정 추가 수정: 검정 범례 seed에는 밝기 55 하한을 적용하지 않습니다.
    minimum_brightness = 0 if seed_brightness < 55 else 55
    if brightness < minimum_brightness or brightness > 235:
        return False
    # ###
    if abs(brightness - seed_brightness) > max(45.0, max_distance * 0.65):
        return False
    return _distance(color, seed) <= max_distance


# ###수정 차이 11: RGB 병합 기준만 50으로 변경하고 HSV 조건은 유지합니다.
def _same_color_family(a: RGB, b: RGB) -> bool:
    if _distance(a, b) <= 50:
        return True
    hue_a, saturation_a, value_a = _rgb_hsv(a)
    hue_b, saturation_b, value_b = _rgb_hsv(b)
    if min(saturation_a, saturation_b) < 0.25:
        return False
    if _hue_distance(hue_a, hue_b) > 12:
        return False
    return abs(value_a - value_b) <= 0.45
# ###


# ###수정 차이 13: legend와의 RGB 거리 50 초과를 제외하고 보조 조건은 유지합니다.
def _matches_legend_seed(color: RGB, seed: RGB) -> bool:
    seed_hue, seed_saturation, _seed_value = _rgb_hsv(seed)
    color_hue, color_saturation, _color_value = _rgb_hsv(color)
    distance = _distance(color, seed)
    if distance > 50:
        return False
    if seed_saturation < 0.12:
        return _matches_gray_seed(color, seed, max_distance=80)

    if color_saturation < 0.18:
        return False

    hue_delta = _hue_distance(color_hue, seed_hue)
    if distance <= 55:
        return True
    if hue_delta <= 18 and distance <= 130:
        return True
    if hue_delta <= 34 and distance <= 88:
        return True
    return False
# ###


# ###수정 차이 06·07: 3-bit bin의 midpoint RGB 대표값을 사용합니다.
def _quantize(rgb: RGB, step: int = 32) -> RGB:
    if step != 32:
        raise ValueError("CMSL quantization uses fixed 3-bit bins (step=32)")
    return tuple((v // 32) * 32 + 16 for v in rgb)  # type: ignore[return-value]
# ###


# ###수정 차이 08: 모든 crop pixel의 최빈 양자화 bin을 background로 지정합니다.
def _build_background_mask(crop: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint16)
    bins = rgb // 32
    encoded = bins[:, :, 0] * 64 + bins[:, :, 1] * 8 + bins[:, :, 2]
    counts = np.bincount(encoded.ravel(), minlength=512)
    dominant = int(np.argmax(counts))
    mask = Image.fromarray(np.where(encoded == dominant, 255, 0).astype(np.uint8))
    q = [dominant // 64, (dominant // 8) % 8, dominant % 8]
    return mask, {
        "dominant_background_bin": q,
        "dominant_background_rgb": [32 * value + 16 for value in q],
        "background_pixel_count": int(counts[dominant]),
        "quantization_step": 32,
        "color_bin_count": 512,
    }
# ###


# ###수정 차이 09: 연속된 grid 후보 행·열을 하나의 group으로 병합합니다.
def _group_adjacent_indices(indices: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for index in sorted(set(indices)):
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups
# ###


# ###수정 차이 09: 행·열의 foreground 점유율이 80%를 초과하면 grid로 제외합니다.
def _detect_grid_mask(background_mask: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    foreground = np.asarray(background_mask) == 0
    height, width = foreground.shape
    rows = np.flatnonzero(foreground.sum(axis=1) > width * 0.80).tolist()
    columns = np.flatnonzero(foreground.sum(axis=0) > height * 0.80).tolist()
    row_groups = _group_adjacent_indices(rows)
    column_groups = _group_adjacent_indices(columns)
    grid = np.zeros((height, width), dtype=np.uint8)
    for group in row_groups:
        grid[group, :] = 255
    for group in column_groups:
        grid[:, group] = 255
    return Image.fromarray(grid), {
        "grid_occupancy_threshold": 0.80,
        "grid_comparison": "strictly_greater_than",
        "horizontal_grid_groups": row_groups,
        "vertical_grid_groups": column_groups,
        "grid_pixel_count": int(np.count_nonzero(grid)),
    }
# ###


# ###수정 차이 08~10: background와 grid 제외 mask를 통합합니다.
def _build_non_curve_mask(crop: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    background, background_info = _build_background_mask(crop)
    grid, grid_info = _detect_grid_mask(background)
    excluded = np.maximum(np.asarray(background), np.asarray(grid))
    return background, grid, Image.fromarray(excluded), {**background_info, **grid_info}
# ###



def _linear_model(pairs: list[dict[str, float]]) -> dict[str, Any] | None:
    if len(pairs) < 2:
        return None
    xs = [pair["pixel"] for pair in pairs]
    ys = [pair["value"] for pair in pairs]
    n = float(len(pairs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return {
        "type": "linear",
        "slope": slope,
        "intercept": intercept,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def _apply_linear(model: dict[str, Any], pixel: float) -> float:
    return float(model["slope"]) * pixel + float(model["intercept"])


def _normalize_manual_pairs(
    manual_axis_pairs: dict[str, Any] | None,
    panel_id: str,
) -> dict[str, list[dict[str, float]]]:
    if not manual_axis_pairs:
        return {}
    raw = manual_axis_pairs.get(panel_id, manual_axis_pairs)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict[str, float]]] = {}
    for role, pairs in raw.items():
        if role not in {"left", "bottom", "right"} or not isinstance(pairs, list):
            continue
        parsed: list[dict[str, float]] = []
        for pair in pairs:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and isinstance(pair[0], (int, float))
                and isinstance(pair[1], (int, float))
            ):
                parsed.append({"pixel": float(pair[0]), "value": float(pair[1])})
            elif (
                isinstance(pair, dict)
                and isinstance(pair.get("pixel"), (int, float))
                and isinstance(pair.get("value"), (int, float))
            ):
                parsed.append({"pixel": float(pair["pixel"]), "value": float(pair["value"])})
        if parsed:
            out[role] = parsed
    return out


def _pairs_from_tick_axis(axis: dict[str, Any]) -> list[dict[str, float]]:
    candidates = axis.get("tick_label_candidates")
    matches = axis.get("tick_matches")
    if not isinstance(candidates, list) or not isinstance(matches, list):
        return []
    value_by_bbox: dict[tuple[int, int, int, int], float] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        bbox = candidate.get("bbox_xyxy")
        value = candidate.get("numeric_value")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, int) for v in bbox)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            value_by_bbox[tuple(bbox)] = float(value)

    pairs: list[dict[str, float]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        bbox = match.get("candidate_bbox_xyxy")
        tick_position = match.get("tick_position_px")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, int) for v in bbox)
            and isinstance(tick_position, (int, float))
        ):
            value = value_by_bbox.get(tuple(bbox))
            if value is not None:
                pairs.append({"pixel": float(tick_position), "value": value})
    return pairs


def build_axis_mapping(
    axes_panel: dict[str, Any] | None,
    ticks_panel: dict[str, Any] | None,
    *,
    panel_id: str,
    manual_axis_pairs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manual_pairs = _normalize_manual_pairs(manual_axis_pairs, panel_id)
    axes_by_role: dict[str, dict[str, Any]] = {}
    if ticks_panel and isinstance(ticks_panel.get("axes"), list):
        for axis in ticks_panel["axes"]:
            if isinstance(axis, dict) and axis.get("axis_role") in {"left", "bottom", "right"}:
                axes_by_role[axis["axis_role"]] = axis

    mapping: dict[str, Any] = {
        "schema_version": 1,
        "panel_id": panel_id,
        "coordinate_space": "panel_px",
        "x_axis": "bottom",
        "y_axes": ["left", "right"] if axes_panel and axes_panel.get("is_dual_y") else ["left"],
        "axes": {},
        "issues": [],
    }
    for role in ("bottom", "left", "right"):
        source = "gt_axis_ticks"
        pairs = _pairs_from_tick_axis(axes_by_role.get(role, {}))
        if role in manual_pairs:
            pairs = manual_pairs[role]
            source = "manual_axis_pairs"
        model = _linear_model(pairs)
        axis_issues: list[str] = []
        if len(pairs) < 2:
            axis_issues.append("INSUFFICIENT_CALIBRATION_PAIRS")
            mapping["issues"].append(f"{role}:INSUFFICIENT_CALIBRATION_PAIRS")
        mapping["axes"][role] = {
            "role": role,
            "pixel_dimension": "x" if role == "bottom" else "y",
            "pair_source": source,
            "pairs": pairs,
            "model": model,
            "axis_label": (axes_panel or {}).get(f"axis_{role}_label"),
            "issues": axis_issues,
        }
    return mapping


def build_chart_extraction_strategy(
    axes_panel: dict[str, Any] | None,
    axis_mapping: dict[str, Any],
) -> dict[str, Any]:
    has_right = bool(axes_panel and axes_panel.get("axis_right_bbox_xyxy"))
    is_dual_y = bool(axes_panel and axes_panel.get("is_dual_y")) or has_right
    axis_labels_present = any(
        isinstance((axes_panel or {}).get(key), str) and (axes_panel or {}).get(key).strip()
        for key in ("axis_bottom_label", "axis_left_label", "axis_right_label")
    )
    issues = [] if axis_labels_present else ["AXIS_LABEL_OCR_NOT_RUN"]

    # Which extraction_family Vision actually matched this panel to. Only
    # current_voltage_power_dual_y turns on the C-V-P-specific assumptions everywhere else in this
    # module (voltage decreases, power density peaks, P=V*I fit, same-color curve separation) — every
    # other family, including any other dual-y chart, goes through the lenient path below that trusts
    # legend/color axis_binding_hint instead of curve shape. A caller that never passes
    # extraction_family (older GT fixtures, existing tests) keeps the historical default: any dual-y
    # panel is still treated as current_voltage_power_dual_y, so nothing about that path changes.
    extraction_family = str((axes_panel or {}).get("extraction_family") or "").strip()
    is_cvp = extraction_family == "current_voltage_power_dual_y" or (not extraction_family and is_dual_y)
    if is_cvp:
        family = "dual_y_power_density"
    elif extraction_family:
        family = extraction_family
    else:
        family = "single_y_line_chart"

    confidence = 0.55 if is_dual_y else 0.4
    if is_cvp:
        x_axis_role, left_y_role, right_y_role = "current_density", "voltage", "power_density"
        expected_curve_roles = [
            {"curve_role": "voltage_curve", "shape": "mostly_decreasing_data_y"},
            {"curve_role": "power_density_curve", "shape": "rise_peak_fall_data_y"},
        ]
    elif is_dual_y:
        x_axis_role, left_y_role, right_y_role = "x", "y_left", "y_right"
        expected_curve_roles = [{"curve_role": "data_curve", "shape": "unknown"}]
    else:
        x_axis_role, left_y_role, right_y_role = "x", "y", None
        expected_curve_roles = [{"curve_role": "data_curve", "shape": "unknown"}]

    return {
        "schema_version": 1,
        "chart_family": family,
        "confidence": confidence,
        "evidence": {
            "is_dual_y_gt": bool(axes_panel and axes_panel.get("is_dual_y")),
            "has_right_axis_bbox": has_right,
            "extraction_family": extraction_family or None,
            "axis_mapping_roles": list(axis_mapping.get("axes", {}).keys()),
        },
        "x_axis_role": x_axis_role,
        "left_y_role": left_y_role,
        "right_y_role": right_y_role,
        "color_groups_represent": "condition_or_series",
        "expected_curve_roles": expected_curve_roles,
        "marker_fill_is_hint_only": True,
        "issues": issues,
    }


def _image_variance(image: Image.Image) -> float:
    return float(ImageStat.Stat(image).var[0])


def _plot_sharpness_metrics(crop: Image.Image) -> dict[str, Any]:
    gray = crop.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    laplacian = gray.filter(
        ImageFilter.Kernel(
            (3, 3),
            [0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1,
            offset=128,
        )
    )
    return {
        "edge_mean": float(ImageStat.Stat(edges).mean[0]),
        "edge_variance": _image_variance(edges),
        "laplacian_variance": _image_variance(laplacian),
    }


def _candidate_color_separation_metrics(series_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    min_seed_distance: float | None = None
    min_palette_distance: float | None = None
    closest_palette_pair: dict[str, Any] | None = None
    ambiguous_palette_pair_count = 0

    for left_index, left in enumerate(series_candidates):
        left_seed = _candidate_seed_color(left)
        left_colors = _candidate_target_colors(left)
        for right in series_candidates[left_index + 1 :]:
            right_seed = _candidate_seed_color(right)
            if left_seed is not None and right_seed is not None:
                seed_distance = _distance(left_seed, right_seed)
                min_seed_distance = (
                    seed_distance
                    if min_seed_distance is None
                    else min(min_seed_distance, seed_distance)
                )
            for left_color in left_colors:
                for right_color in _candidate_target_colors(right):
                    palette_distance = _distance(left_color, right_color)
                    if palette_distance <= 35:
                        ambiguous_palette_pair_count += 1
                    if min_palette_distance is None or palette_distance < min_palette_distance:
                        min_palette_distance = palette_distance
                        closest_palette_pair = {
                            "left_series_id": left.get("series_id"),
                            "right_series_id": right.get("series_id"),
                            "left_rgb": list(left_color),
                            "right_rgb": list(right_color),
                            "distance": palette_distance,
                        }

    return {
        "series_candidate_count": len(series_candidates),
        "min_seed_distance": min_seed_distance,
        "min_palette_distance": min_palette_distance,
        "closest_palette_pair": closest_palette_pair,
        "ambiguous_palette_pair_count": ambiguous_palette_pair_count,
    }


def assess_plot_image_quality(crop: Image.Image, series_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    sharpness = _plot_sharpness_metrics(crop)
    color_separation = _candidate_color_separation_metrics(series_candidates)
    issues: list[str] = []
    if sharpness["laplacian_variance"] < 900 or sharpness["edge_variance"] < 2000:
        issues.append("LOW_IMAGE_SHARPNESS")
    min_palette_distance = color_separation.get("min_palette_distance")
    if isinstance(min_palette_distance, (int, float)) and min_palette_distance < 35:
        issues.append("LOW_COLOR_SEPARATION_BETWEEN_SERIES")
    if color_separation.get("ambiguous_palette_pair_count"):
        issues.append("AMBIGUOUS_ANTIALIAS_COLOR_PAIRS")
    return {
        "schema_version": 1,
        "sharpness": sharpness,
        "color_separation": color_separation,
        "issues": issues,
    }


def _sharpen_crop_for_segmentation(crop: Image.Image) -> Image.Image:
    return crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=3))


# ###수정 차이 10: 제외 mask를 반영하고 최소 빈도 1과 색상 수 무제한을 적용합니다.
def detect_color_candidates(
    crop: Image.Image,
    *,
    max_colors: int | None = None,
    min_pixels: int = 1,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
    legend_items: list[dict[str, Any]] | None = None,
    non_curve_mask: Image.Image | None = None,
) -> list[dict[str, Any]]:
    rgb_crop = crop.convert("RGB")
    if non_curve_mask is None:
        _background, _grid, non_curve_mask, _info = _build_non_curve_mask(rgb_crop)
    blocked = non_curve_mask.load()
    exclusions = _normalize_exclusion_bboxes_crop_px(
        exclusion_bboxes_crop_px,
        rgb_crop.width,
        rgb_crop.height,
    )
    legend_seeds: list[dict[str, Any]] = []
    for item in legend_items or []:
        if not isinstance(item, dict):
            continue
        seed = _legend_item_rgb(item)
        label = item.get("legend_label")
        if seed is None:
            continue
        raw_axis_hint = item.get("axis_binding_hint")
        # ###수정 추가 수정: 같은 RGB의 V/P 범례를 한 mask로 묶되 각각의 이름과 축 후보를 보존합니다.
        option = {"legend_label": label.strip() if isinstance(label, str) and label.strip() else None,
                  "swatch_color_rgb": list(seed), "axis_binding_hint": raw_axis_hint if raw_axis_hint in ("left_y", "right_y") else None,
                  "marker_shape": item.get("marker_shape")}
        shared_group = next((group for group in legend_seeds if group["seed"] == seed), None)
        if shared_group is not None:
            shared_group["legend_options"].append(option)
            shared_group["legend_label"] = None
            shared_group["axis_binding_hint"] = None
            continue
        legend_seeds.append(
            {
                "seed": seed,
                "legend_label": label.strip() if isinstance(label, str) and label.strip() else None,
                "axis_binding_hint": raw_axis_hint if raw_axis_hint in ("left_y", "right_y") else None,
                "colors": [],
                "count": 0,
                "counts": {},
                "legend_options": [option],
            }
        )
        # ###
    has_gray_legend_seed = any(_rgb_hsv(group["seed"])[1] < 0.12 for group in legend_seeds)

    counts: Counter[RGB] = Counter()
    src = rgb_crop.load()
    for y in range(rgb_crop.height):
        for x in range(rgb_crop.width):
            if _point_in_bboxes(x, y, exclusions):
                continue
            if blocked[x, y] > 0:
                continue
            rgb = src[x, y]
            brightness = _brightness(rgb)
            # ###수정 추가 수정: 검정 범례와 일치하는 dark pixel은 기존 밝기 30 필터에서 보존합니다.
            dark_legend_match = any(_brightness(group["seed"]) < 55 and _matches_gray_seed(rgb, group["seed"]) for group in legend_seeds)
            if (brightness < 30 and not dark_legend_match) or brightness > 245:
                continue
            # ###
            saturation = _saturation(rgb)
            if saturation < 0.28:
                if not has_gray_legend_seed:
                    continue
                if not any(_matches_gray_seed(rgb, group["seed"], max_distance=80) for group in legend_seeds):
                    continue
            counts[_quantize(rgb)] += 1

    groups: list[dict[str, Any]] = []
    for color, count in counts.most_common():
        if count < min_pixels:
            continue
        if legend_seeds:
            distances = [
                (_distance(color, group["seed"]), index, group)
                for index, group in enumerate(legend_seeds)
                if _matches_legend_seed(color, group["seed"])
            ]
            if not distances:
                continue
            dist, _index, group = min(distances, key=lambda item: item[0])
            group["colors"].append(color)
            group["count"] += count
            group["counts"][color] = count
            continue
        matched_group = None
        for group in groups:
            if any(_same_color_family(color, palette_color) for palette_color in group["colors"]):
                matched_group = group
                break
        if matched_group is None:
            groups.append({"colors": [color], "count": count, "counts": {color: count}})
        else:
            matched_group["colors"].append(color)
            matched_group["count"] += count
            matched_group["counts"][color] = count

    seeded_groups = [group for group in legend_seeds if group["count"] >= min_pixels]
    unseeded_groups = sorted(groups, key=lambda group: group["count"], reverse=True)
    groups = seeded_groups + unseeded_groups
    selected: list[dict[str, Any]] = []
    for group in (groups if max_colors is None else groups[:max_colors]):
        ordered_colors = sorted(group["colors"], key=lambda color: group["counts"][color], reverse=True)
        representative = ordered_colors[0]
        series_id = f"color_{len(selected) + 1:02d}"
        selected.append(
            {
                "series_id": series_id,
                "legend_label": None,
                "target_color_rgb": list(representative),
                "target_colors_rgb": [list(color) for color in ordered_colors[:12]],
                "color_distance": 45 if len(ordered_colors) > 1 else 70,
                "pixel_count_estimate": group["count"],
                "series_mask_ref": f"masks/{series_id}.png",
                "candidate_axis_binding_hint": None,
                "color_family_member_count": len(ordered_colors),
                "color_family_method": "legend_seed_nearest_color" if "seed" in group else "hue_distance_plus_rgb_distance",
            }
        )
        if group.get("legend_label"):
            selected[-1]["legend_label"] = group["legend_label"]
            selected[-1]["legend_seed_rgb"] = list(group["seed"])
        if group.get("axis_binding_hint"):
            selected[-1]["candidate_axis_binding_hint"] = group["axis_binding_hint"]
        # ###수정 추가 수정: same-color 범례 목록은 추출 curve별 VLM 판정까지 유지합니다.
        selected[-1]["legend_options"] = group.get("legend_options", [])
        if "seed" in group:
            selected[-1]["legend_seed_rgb"] = list(group["seed"])
        # ###
    return selected
# ###


def _legend_item_rgb(item: dict[str, Any]) -> RGB | None:
    raw = item.get("swatch_color_rgb")
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    if not all(isinstance(value, (int, float)) for value in raw):
        return None
    return tuple(max(0, min(255, int(round(value)))) for value in raw)  # type: ignore[return-value]


# ###수정 차이 13: 후속 legend label matching의 RGB 거리도 50으로 제한합니다.
def match_legend_items_to_series_candidates(
    series_candidates: list[dict[str, Any]],
    legend_items: list[dict[str, Any]] | None,
    *,
    max_color_distance: float = 50.0,
) -> dict[str, Any]:
    parsed_legend_items: list[dict[str, Any]] = []
    for index, item in enumerate(legend_items or []):
        if not isinstance(item, dict):
            continue
        rgb = _legend_item_rgb(item)
        label = item.get("legend_label")
        if rgb is None or not isinstance(label, str) or not label.strip():
            continue
        raw_axis_hint = item.get("axis_binding_hint")
        parsed_legend_items.append(
            {
                "legend_item_index": index,
                "legend_label": label.strip(),
                "swatch_color_rgb": list(rgb),
                "marker_shape": item.get("marker_shape"),
                "confidence": item.get("confidence"),
                "axis_binding_hint": raw_axis_hint if raw_axis_hint in ("left_y", "right_y") else None,
            }
        )

    candidate_pairs: list[tuple[float, int, int]] = []
    for candidate_index, candidate in enumerate(series_candidates):
        candidate_rgbs = _candidate_target_colors(candidate)
        if not candidate_rgbs:
            continue
        for legend_index, legend_item in enumerate(parsed_legend_items):
            dist = min(
                _distance(candidate_rgb, tuple(legend_item["swatch_color_rgb"]))  # type: ignore[arg-type]
                for candidate_rgb in candidate_rgbs
            )
            if dist <= max_color_distance:
                candidate_pairs.append((dist, candidate_index, legend_index))

    # ###수정 추가 수정: 후보와 범례를 1:1로 제한하지 않고 같은 색의 모든 범례를 보존합니다.
    used_candidates: set[int] = set()
    used_legend_items: set[int] = set()
    for dist, candidate_index, legend_index in sorted(candidate_pairs):
        if legend_index in used_legend_items:
            continue
        legend_item = parsed_legend_items[legend_index]
        candidate = series_candidates[candidate_index]
        options = candidate.setdefault("legend_options", [])
        if not any(option.get("legend_label") == legend_item["legend_label"] for option in options):
            options.append({key: legend_item.get(key) for key in ("legend_label", "swatch_color_rgb", "axis_binding_hint", "marker_shape")})
        candidate["legend_label"] = options[0].get("legend_label") if len(options) == 1 else None
        hints = {option.get("axis_binding_hint") for option in options}
        candidate["candidate_axis_binding_hint"] = next(iter(hints)) if len(hints) == 1 else None
        candidate["legend_match"] = {
            "source": "codex_legend_vision",
            "legend_item_index": legend_item["legend_item_index"],
            "legend_label": legend_item["legend_label"],
            "swatch_color_rgb": legend_item["swatch_color_rgb"],
            "color_distance": dist,
            "marker_shape": legend_item.get("marker_shape"),
            "confidence": legend_item.get("confidence"),
        }
        used_candidates.add(candidate_index)
        used_legend_items.add(legend_index)
    # ###

    issues: list[str] = []
    for legend_index, legend_item in enumerate(parsed_legend_items):
        if legend_index not in used_legend_items:
            issues.append(f"LEGEND_ITEM_UNMATCHED_BY_COLOR:{legend_item['legend_label']}")

    return {
        "source": "codex_legend_vision" if parsed_legend_items else "none",
        "legend_item_count": len(parsed_legend_items),
        "matched_series_count": len(used_candidates),
        "max_color_distance": max_color_distance,
        "issues": issues,
    }
# ###


def _parse_rgb_list(raw: Any) -> RGB | None:
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    if not all(isinstance(value, (int, float)) for value in raw):
        return None
    return tuple(max(0, min(255, int(round(value)))) for value in raw)  # type: ignore[return-value]


def _candidate_target_colors(candidate: dict[str, Any]) -> list[RGB]:
    colors: list[RGB] = []
    raw_palette = candidate.get("target_colors_rgb")
    if isinstance(raw_palette, list):
        for raw_color in raw_palette:
            rgb = _parse_rgb_list(raw_color)
            if rgb is not None and rgb not in colors:
                colors.append(rgb)
    rgb = _parse_rgb_list(candidate.get("target_color_rgb"))
    if rgb is not None and rgb not in colors:
        colors.append(rgb)
    return colors


def _candidate_seed_color(candidate: dict[str, Any]) -> RGB | None:
    seed = _parse_rgb_list(candidate.get("legend_seed_rgb"))
    if seed is not None:
        return seed
    legend_match = candidate.get("legend_match")
    if isinstance(legend_match, dict):
        seed = _parse_rgb_list(legend_match.get("swatch_color_rgb"))
        if seed is not None:
            return seed
    return _parse_rgb_list(candidate.get("target_color_rgb"))


def _candidate_assignment_score(rgb: RGB, seed: RGB) -> float:
    if _is_gray_like(seed):
        return _distance(rgb, seed)
    hue, saturation, value = _rgb_hsv(rgb)
    seed_hue, seed_saturation, seed_value = _rgb_hsv(seed)
    return (
        _distance(rgb, seed) * 0.45
        + _hue_distance(hue, seed_hue) * 4.0
        + abs(saturation - seed_saturation) * 35.0
        + abs(value - seed_value) * 35.0
    )


# ###수정 차이 10: curve mask 생성에서도 background·grid를 제외합니다.
def _mask_for_color(
    image: Image.Image,
    targets: RGB | list[RGB],
    threshold: float,
    *,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
    include_faint_hue: bool = False,
    non_curve_mask: Image.Image | None = None,
) -> tuple[Image.Image, set[Point]]:
    rgb = image.convert("RGB")
    exclusions = _normalize_exclusion_bboxes_crop_px(
        exclusion_bboxes_crop_px,
        rgb.width,
        rgb.height,
    )
    blocked = non_curve_mask.load() if non_curve_mask is not None else None
    mask = Image.new("L", rgb.size, 0)
    mask_pixels = mask.load()
    points: set[Point] = set()
    threshold2 = threshold * threshold
    target_colors = targets if isinstance(targets, list) else [targets]
    target_hsvs = [(target, *_rgb_hsv(target)) for target in target_colors]
    src = rgb.load()
    for y in range(rgb.height):
        for x in range(rgb.width):
            if _point_in_bboxes(x, y, exclusions):
                continue
            if blocked is not None and blocked[x, y] > 0:
                continue
            r, g, b = src[x, y]
            rgb_value = (r, g, b)
            distance_match = any(
                _matches_gray_seed(rgb_value, target, max_distance=max(60.0, threshold))
                if _is_gray_like(target)
                else _saturation(rgb_value) >= 0.10
                and (r - target[0]) ** 2 + (g - target[1]) ** 2 + (b - target[2]) ** 2 <= threshold2
                for target in target_colors
            )
            hue_match = False
            if include_faint_hue:
                hue, saturation, value = _rgb_hsv(rgb_value)
                # saturation 하한: 기존 0.13 → 0.20으로 수정해 저채도 가장자리의 다른 시리즈 유입을 줄입니다. hue 허용 차이 26°는 유지합니다.
                if saturation >= 0.20 and value <= 0.995 and max(rgb_value) < 253:
                    hue_match = any(
                        target_saturation >= 0.2 and _hue_distance(hue, target_hue) <= 26
                        for _target, target_hue, target_saturation, _target_value in target_hsvs
                    )
            if distance_match or hue_match:
                mask_pixels[x, y] = 255
                points.add((x, y))
    return mask, points
# ###


# ###수정 차이 10: mask 보강 후에도 background·grid 제외 영역을 유지합니다.
def build_solid_color_filter_source(
    crop: Image.Image,
    series_candidates: list[dict[str, Any]],
    *,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
    non_curve_mask: Image.Image | None = None,
) -> Image.Image:
    """Create a denoised crop where accepted curve pixels are recolored to one solid series color."""
    rgb = crop.convert("RGB")
    out = Image.new("RGB", rgb.size, "white")
    out_pixels = out.load()
    prepared_candidates: list[dict[str, Any]] = []
    for candidate in series_candidates:
        colors = _candidate_target_colors(candidate)
        if not colors:
            continue
        representative = _parse_rgb_list(candidate.get("target_color_rgb")) or colors[0]
        seed = _candidate_seed_color(candidate) or representative
        include_faint_hue = not any(_is_gray_like(color) for color in colors)
        mask, _points = _mask_for_color(
            rgb,
            colors,
            float(candidate.get("color_distance", 70)),
            exclusion_bboxes_crop_px=exclusion_bboxes_crop_px,
            include_faint_hue=include_faint_hue,
            non_curve_mask=non_curve_mask,
        )
        solid_mask = _strengthen_binary_mask(mask, radius_px=1)
        if non_curve_mask is not None:
            solid_mask = Image.fromarray(np.where(np.asarray(non_curve_mask) > 0, 0, np.asarray(solid_mask)).astype(np.uint8))
        prepared_candidates.append(
            {
                "representative": representative,
                "seed": seed,
                "mask_pixels": solid_mask.load(),
            }
        )

    if not prepared_candidates:
        return out

    src_pixels = rgb.load()
    for y in range(rgb.height):
        for x in range(rgb.width):
            best: tuple[float, RGB] | None = None
            pixel = src_pixels[x, y]
            for candidate in prepared_candidates:
                if candidate["mask_pixels"][x, y] <= 0:
                    continue
                score = _candidate_assignment_score(pixel, candidate["seed"])
                if best is None or score < best[0]:
                    best = (score, candidate["representative"])
            if best is not None:
                out_pixels[x, y] = best[1]
    return out
# ###


def _points_from_mask(mask: Image.Image) -> set[Point]:
    binary = mask.convert("L")
    pixels = binary.load()
    return {
        (x, y)
        for y in range(binary.height)
        for x in range(binary.width)
        if pixels[x, y] > 0
    }


def _strengthen_binary_mask(mask: Image.Image, *, radius_px: int = 1) -> Image.Image:
    """Connect faint same-color curve fragments without changing the source color model."""
    radius_px = max(0, int(radius_px))
    if radius_px == 0:
        return mask.convert("L")
    kernel = radius_px * 2 + 1
    binary = mask.convert("L")
    return binary.filter(ImageFilter.MaxFilter(kernel)).filter(ImageFilter.MinFilter(kernel))


def _cluster_y_values(y_values: list[int], max_gap_px: int = 3) -> list[int]:
    if not y_values:
        return []
    sorted_y = sorted(set(y_values))
    clusters: list[list[int]] = [[sorted_y[0]]]
    for y in sorted_y[1:]:
        if y - clusters[-1][-1] <= max_gap_px:
            clusters[-1].append(y)
        else:
            clusters.append([y])
    return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]


# ###수정 차이 15·16: 연결 간격 10/10 px와 Euclidean 점수를 사용합니다.
def _tracks_from_mask_points(
    points: set[Point],
    *,
    image_width: int,
    min_width_px: int,
    max_x_gap_px: int = 10,
    max_y_gap_px: int = 10,
    # ###수정 추가 수정: support 보간용 조각 수집은 길이 필터 전에 수행합니다.
    retain_fragments: bool = False,
    min_coverage_fraction: float = 0.08,
    # ###
) -> list[list[Point]]:
    by_x: dict[int, list[int]] = defaultdict(list)
    for x, y in points:
        by_x[x].append(y)

    tracks: list[list[Point]] = []
    active: dict[int, Point] = {}
    for x in sorted(by_x):
        clusters = _cluster_y_values(by_x[x])
        assigned_tracks: set[int] = set()
        for y in clusters:
            best_idx = None
            best_score = None
            for idx, (last_x, last_y) in active.items():
                if idx in assigned_tracks:
                    continue
                if x - last_x > max_x_gap_px:
                    continue
                dy = abs(y - last_y)
                if dy > max_y_gap_px:
                    continue
                score = math.hypot(x - last_x, dy)
                if best_score is None or score < best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                tracks.append([(x, y)])
                active[len(tracks) - 1] = (x, y)
                assigned_tracks.add(len(tracks) - 1)
            else:
                tracks[best_idx].append((x, y))
                active[best_idx] = (x, y)
                assigned_tracks.add(best_idx)
        active = {idx: pt for idx, pt in active.items() if x - pt[0] <= max_x_gap_px}

    filtered: list[list[Point]] = []
    for track in tracks:
        xs = [p[0] for p in track]
        width = max(xs) - min(xs) + 1 if xs else 0
        if retain_fragments or (len(track) >= 5 and width >= min_width_px and width >= image_width * min_coverage_fraction):
            filtered.append(track)
    filtered.sort(key=lambda track: (-(max(p[0] for p in track) - min(p[0] for p in track)), min(p[1] for p in track)))
    return filtered
# ###


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _merge_track_group(tracks: list[list[Point]]) -> list[Point]:
    by_x: dict[int, list[int]] = defaultdict(list)
    for track in tracks:
        for x, y in track:
            by_x[x].append(y)
    return [(x, int(round(sum(ys) / len(ys)))) for x, ys in sorted(by_x.items())]


def _tracks_are_duplicate_edges(
    a: list[Point],
    b: list[Point],
    *,
    max_median_y_distance_px: float,
    min_overlap_ratio: float = 0.55,
) -> bool:
    a_by_x = {x: y for x, y in a}
    b_by_x = {x: y for x, y in b}
    common_xs = sorted(set(a_by_x).intersection(b_by_x))
    if not common_xs:
        return False
    overlap_ratio = len(common_xs) / max(1, min(len(a_by_x), len(b_by_x)))
    if overlap_ratio < min_overlap_ratio:
        return False
    distances = [abs(a_by_x[x] - b_by_x[x]) for x in common_xs]
    return _median(distances) <= max_median_y_distance_px


def _merge_near_duplicate_tracks(
    tracks: list[list[Point]],
    *,
    max_median_y_distance_px: float,
) -> list[list[Point]]:
    groups: list[list[list[Point]]] = []
    for track in sorted(tracks, key=lambda item: -(max(x for x, _y in item) - min(x for x, _y in item))):
        matched_idx = None
        for idx, group in enumerate(groups):
            representative = _merge_track_group(group)
            if _tracks_are_duplicate_edges(
                representative,
                track,
                max_median_y_distance_px=max_median_y_distance_px,
            ):
                matched_idx = idx
                break
        if matched_idx is None:
            groups.append([track])
        else:
            groups[matched_idx].append(track)
    merged = [_merge_track_group(group) for group in groups]
    merged.sort(key=lambda track: (-(max(p[0] for p in track) - min(p[0] for p in track)), min(p[1] for p in track)))
    return merged


def _smooth_track_points(points: list[Point], *, window: int = 7) -> list[Point]:
    if len(points) < 3:
        return sorted(points)
    ordered = sorted(points)
    radius = max(1, window // 2)
    smoothed: list[Point] = []
    for idx, (x, _y) in enumerate(ordered):
        lo = max(0, idx - radius)
        hi = min(len(ordered), idx + radius + 1)
        ys = [ordered[j][1] for j in range(lo, hi)]
        smoothed.append((x, int(round(sum(ys) / len(ys)))))
    return smoothed


def _smooth_float_values(values: list[float], *, window: int) -> list[float]:
    if len(values) < 3:
        return values[:]
    radius = max(1, window // 2)
    smoothed: list[float] = []
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def _isotonic_regression(values: list[float], *, increasing: bool) -> list[float]:
    if not values:
        return []
    working = values[:] if increasing else [-value for value in values]
    blocks: list[dict[str, float]] = []
    for value in working:
        blocks.append({"sum": value, "weight": 1.0, "count": 1.0})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left["sum"] / left["weight"] <= right["sum"] / right["weight"]:
                break
            merged = {
                "sum": left["sum"] + right["sum"],
                "weight": left["weight"] + right["weight"],
                "count": left["count"] + right["count"],
            }
            blocks[-2:] = [merged]
    out: list[float] = []
    for block in blocks:
        mean = block["sum"] / block["weight"]
        out.extend([mean] * int(block["count"]))
    return out if increasing else [-value for value in out]


def _smooth_final_curve_points(points: list[list[int]], *, curve_role: str) -> tuple[list[list[int]], dict[str, Any]]:
    by_x: dict[int, list[int]] = defaultdict(list)
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        by_x[int(point[0])].append(int(point[1]))
    ordered = [(x, sum(ys) / len(ys)) for x, ys in sorted(by_x.items())]
    if len(ordered) < 20:
        return points, {
            "method": "none_short_trace",
            "input_point_count": len(points),
            "output_point_count": len(points),
        }

    xs = [x for x, _y in ordered]
    ys = [y for _x, y in ordered]
    window = max(7, min(21, int(round(len(ys) * 0.07))))
    if window % 2 == 0:
        window += 1
    smoothed = _smooth_float_values(ys, window=window)
    # Final trace smoothing should reduce pixel jitter without re-fitting the chart shape.
    # Monotone/single-peak projection is useful as a classifier hint, but it can create
    # artificial horizontal plateaus when the extracted mask is noisy or partly merged.
    blend = 0.65
    smoothed = [raw * (1.0 - blend) + local * blend for raw, local in zip(ys, smoothed)]
    method = "local_moving_average_shape_preserving"

    smoothed_points = [[int(x), int(round(y))] for x, y in zip(xs, smoothed)]
    return smoothed_points, {
        "method": method,
        "input_point_count": len(points),
        "output_point_count": len(smoothed_points),
        "window": window,
    }


def _smoothed_final_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trace in traces:
        copied = json.loads(json.dumps(trace))
        role = str(copied.get("curve_role") or "")
        if role in {"voltage_curve", "power_density_curve"}:
            smoothed_points, smoothing = _smooth_final_curve_points(
                copied.get("points") or [],
                curve_role=role,
            )
            copied["points"] = smoothed_points
            features = copied.get("features") if isinstance(copied.get("features"), dict) else {}
            features["final_trace_smoothing"] = smoothing
            copied["features"] = features
        out.append(copied)
    return out


def _centerline_from_thick_stroke(
    points: list[Point],
    *,
    image_size: tuple[int, int],
    stroke_width: int = 7,
) -> list[Point]:
    if len(points) < 2:
        return sorted(points)
    smoothed = _smooth_track_points(points, window=7)
    stroke = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(stroke)
    draw.line(smoothed, fill=255, width=max(3, stroke_width), joint="curve")
    pixels = stroke.load()
    centerline: list[Point] = []
    for x in range(stroke.width):
        ys = [y for y in range(stroke.height) if pixels[x, y] > 0]
        if not ys:
            continue
        centerline.append((x, int(round((min(ys) + max(ys)) / 2))))
    if len(centerline) < 2:
        return smoothed
    min_raw_x = min(x for x, _y in points)
    max_raw_x = max(x for x, _y in points)
    return [(x, y) for x, y in centerline if min_raw_x <= x <= max_raw_x]


def _mask_pixels_near_trace(mask: Image.Image, trace_points: list[Point], *, width_px: int) -> Image.Image:
    source = mask.convert("L")
    corridor = Image.new("L", source.size, 0)
    if len(trace_points) >= 2:
        draw = ImageDraw.Draw(corridor)
        draw.line(trace_points, fill=255, width=max(3, width_px), joint="curve")
    source_pixels = source.load()
    corridor_pixels = corridor.load()
    out = Image.new("L", source.size, 0)
    out_pixels = out.load()
    for y in range(source.height):
        for x in range(source.width):
            if source_pixels[x, y] > 0 and corridor_pixels[x, y] > 0:
                out_pixels[x, y] = 255
    return out


def _contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value > previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _fit_line_xy(points: list[Point]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    xs = [float(x) for x, _y in points]
    ys = [float(y) for _x, y in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _median_float(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _voltage_spline_anchors(points: list[Point]) -> list[tuple[float, float]] | None:
    if len(points) < 20:
        return None
    ordered = _smooth_track_points(points, window=9)
    xs = [x for x, _y in ordered]
    min_x = min(xs)
    max_x = max(xs)
    x_span = max_x - min_x
    if x_span < 20:
        return None

    fractions = [0.0, 0.38, 0.68, 1.0] if x_span >= 80 else [0.0, 0.5, 1.0]
    band = max(3.0, x_span * 0.055)
    anchors: list[tuple[float, float]] = []
    for fraction in fractions:
        target_x = min_x + x_span * fraction
        if fraction == 0.0:
            candidates = [(x, y) for x, y in ordered if x <= min_x + band]
        elif fraction == 1.0:
            candidates = [(x, y) for x, y in ordered if x >= max_x - band]
        else:
            candidates = [(x, y) for x, y in ordered if abs(x - target_x) <= band]
        if not candidates:
            candidates = sorted(ordered, key=lambda point: abs(point[0] - target_x))[: max(3, len(ordered) // 20)]
        anchors.append(
            (
                _median_float([float(x) for x, _y in candidates]),
                _median_float([float(y) for _x, y in candidates]),
            )
        )

    unique: list[tuple[float, float]] = []
    for x, y in sorted(anchors):
        if unique and abs(x - unique[-1][0]) < 1:
            continue
        if unique and y < unique[-1][1]:
            y = unique[-1][1]
        unique.append((x, y))
    if len(unique) < 3 or unique[-1][1] - unique[0][1] < 12:
        return None
    return unique


def _monotone_cubic_tangents(anchors: list[tuple[float, float]]) -> list[float]:
    secants: list[float] = []
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        dx = x1 - x0
        secants.append(0.0 if dx == 0 else (y1 - y0) / dx)
    tangents: list[float] = []
    for index in range(len(anchors)):
        if index == 0:
            tangent = secants[0]
        elif index == len(anchors) - 1:
            tangent = secants[-1]
        else:
            left = secants[index - 1]
            right = secants[index]
            if left <= 0 or right <= 0:
                tangent = 0.0
            else:
                tangent = (left + right) / 2.0
                tangent = min(tangent, 3.0 * min(left, right))
        tangents.append(max(0.0, tangent))
    return tangents


def _monotone_cubic_y(anchors: list[tuple[float, float]], tangents: list[float], x: float) -> float:
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    segment_index = 0
    for index in range(len(anchors) - 1):
        if anchors[index][0] <= x <= anchors[index + 1][0]:
            segment_index = index
            break
    x0, y0 = anchors[segment_index]
    x1, y1 = anchors[segment_index + 1]
    dx = x1 - x0
    if dx <= 0:
        return y0
    t = (x - x0) / dx
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    y = h00 * y0 + h10 * dx * tangents[segment_index] + h01 * y1 + h11 * dx * tangents[segment_index + 1]
    lower = min(y0, y1)
    upper = max(y0, y1)
    return max(lower, min(upper, y))


def _refine_voltage_centerline_by_spline_branch(
    clusters_by_x: list[tuple[int, list[tuple[int, int, float, int]]]],
    points: list[Point],
    *,
    image_height: int,
) -> list[Point]:
    if len(points) < 20:
        return points
    current = _smooth_track_points(points, window=9)
    original_len = len(points)
    for _ in range(4):
        anchors = _voltage_spline_anchors(current)
        if not anchors:
            return _refine_voltage_centerline_by_linear_branch(
                clusters_by_x,
                points,
                image_height=image_height,
            )
        tangents = _monotone_cubic_tangents(anchors)
        residual_limit = max(14.0, image_height * 0.04)
        refined: list[Point] = []
        for x, clusters in clusters_by_x:
            expected_y = _monotone_cubic_y(anchors, tangents, float(x))
            selected = min(
                clusters,
                key=lambda cluster: abs(cluster[2] - expected_y) - min(cluster[3], 6) * 0.35,
            )
            if abs(selected[2] - expected_y) <= residual_limit:
                refined.append((x, int(round(selected[2]))))
        if len(refined) < max(20, original_len * 0.5):
            return _refine_voltage_centerline_by_linear_branch(
                clusters_by_x,
                points,
                image_height=image_height,
            )
        next_current = _smooth_track_points(refined, window=9)
        if len(next_current) == len(current):
            current = next_current
            break
        current = next_current
    if len(current) < max(20, original_len * 0.55):
        return points
    return current


def _refine_voltage_centerline_by_linear_branch(
    clusters_by_x: list[tuple[int, list[tuple[int, int, float, int]]]],
    points: list[Point],
    *,
    image_height: int,
) -> list[Point]:
    """Remove marker-loop jumps by reselecting clusters along the quasi-linear voltage branch."""
    if len(points) < 20:
        return points
    kept = points
    fit: tuple[float, float] | None = None
    for _ in range(4):
        fit = _fit_line_xy(kept)
        if fit is None:
            return points
        slope, intercept = fit
        if slope <= 0:
            return points
        residual_limit = max(18.0, image_height * 0.045)
        next_kept = [
            (x, y)
            for x, y in points
            if abs(y - (slope * x + intercept)) <= residual_limit
        ]
        if len(next_kept) < max(20, len(points) * 0.45):
            return points
        if len(next_kept) == len(kept):
            kept = next_kept
            break
        kept = next_kept

    fit = _fit_line_xy(kept)
    if fit is None:
        return points
    slope, intercept = fit
    if slope <= 0:
        return points

    residual_limit = max(16.0, image_height * 0.04)
    refined: list[Point] = []
    for x, clusters in clusters_by_x:
        expected_y = slope * x + intercept
        selected = min(clusters, key=lambda cluster: abs(cluster[2] - expected_y))
        if abs(selected[2] - expected_y) <= residual_limit:
            refined.append((x, int(round(selected[2]))))

    if len(refined) < max(20, len(points) * 0.6):
        return points
    return _smooth_track_points(refined, window=11)


def _monotonic_voltage_centerline_from_mask(mask: Image.Image) -> list[Point]:
    """Extract the filled-marker voltage branch from same-color C-V-P traces."""
    binary = mask.convert("L")
    clusters_by_x: list[tuple[int, list[tuple[int, int, float, int]]]] = []
    pixels = binary.load()
    for x in range(binary.width):
        ys = [y for y in range(binary.height) if pixels[x, y] > 0]
        clusters: list[tuple[int, int, float, int]] = []
        for y1, y2 in _contiguous_runs(ys):
            height = y2 - y1 + 1
            if height >= 1:
                clusters.append((y1, y2, (y1 + y2) / 2.0, height))
        if clusters:
            clusters_by_x.append((x, clusters))

    if len(clusters_by_x) < max(20, int(binary.width * 0.25)):
        return []

    xs = [x for x, _clusters in clusters_by_x]
    min_x = min(xs)
    max_x = max(xs)
    x_span = max_x - min_x
    if x_span < max(20, int(binary.width * 0.25)):
        return []

    left_band = [(x, clusters) for x, clusters in clusters_by_x if x <= min_x + x_span * 0.15]
    right_band = [(x, clusters) for x, clusters in clusters_by_x if x >= max_x - x_span * 0.15]
    if not left_band or not right_band:
        return []

    start_y_values = [min(clusters, key=lambda cluster: cluster[2])[2] for _x, clusters in left_band]
    end_y_values = [max(clusters, key=lambda cluster: cluster[2])[2] for _x, clusters in right_band]
    start_y = sorted(start_y_values)[len(start_y_values) // 2]
    end_y = sorted(end_y_values)[len(end_y_values) // 2]
    if end_y - start_y < max(20, binary.height * 0.18):
        return []

    points: list[Point] = []
    for x, clusters in clusters_by_x:
        expected_y = start_y + (end_y - start_y) * ((x - min_x) / max(1, x_span))
        selected = min(
            clusters,
            key=lambda cluster: abs(cluster[2] - expected_y) - min(cluster[3], 6) * 0.35,
        )
        points.append((x, int(round(selected[2]))))

    smoothed = _smooth_track_points(points, window=11)
    return _refine_voltage_centerline_by_spline_branch(
        clusters_by_x,
        smoothed,
        image_height=binary.height,
    )


def classify_curve_role(
    points: list[Point],
    *,
    is_dual_y_power_density: bool,
    axis_binding_hint: str | None = None,
) -> dict[str, Any]:
    if len(points) < 5:
        return {"curve_role": "unknown_curve", "axis_binding_hint": None, "confidence": 0.0, "features": {}}
    ordered = sorted(points)
    n = len(ordered)
    head = ordered[: max(1, n // 5)]
    tail = ordered[-max(1, n // 5) :]
    first_y = sum(p[1] for p in head) / len(head)
    last_y = sum(p[1] for p in tail) / len(tail)
    min_idx, min_point = min(enumerate(ordered), key=lambda item: item[1][1])
    interior_min = 0.15 * n <= min_idx <= 0.85 * n
    y_range = max(p[1] for p in ordered) - min(p[1] for p in ordered)
    left_drop_to_peak = first_y - min_point[1]
    right_drop_to_peak = last_y - min_point[1]
    data_y_decreases = last_y - first_y

    features = {
        "first_y_px": first_y,
        "last_y_px": last_y,
        "min_y_px": min_point[1],
        "min_y_x_px": min_point[0],
        "y_range_px": y_range,
        "interior_min_y_px": interior_min,
        "data_y_decreasing_score_px": data_y_decreases,
        "rise_peak_fall_score_px": min(left_drop_to_peak, right_drop_to_peak),
    }

    # Vision already read which axis this curve's legend/color maps to (see series_candidates[*].
    # candidate_axis_binding_hint, threaded through from legend_items[*].axis_binding_hint). Trust
    # that over guessing from pixel shape: it works for any dual-y chart, not only the
    # monotonic-decreasing/single-peak shapes a current_voltage_power_dual_y curve happens to have.
    # Only the same-color-overlap case (two curves sharing one color, no hint possible) still falls
    # through to the shape heuristics below.
    if axis_binding_hint in ("left_y", "right_y"):
        if is_dual_y_power_density:
            role = "voltage_curve" if axis_binding_hint == "left_y" else "power_density_curve"
        else:
            role = "data_curve"
        return {"curve_role": role, "axis_binding_hint": axis_binding_hint, "confidence": 0.6, "features": features}

    if not is_dual_y_power_density:
        return {"curve_role": "data_curve", "axis_binding_hint": "left_y", "confidence": 0.4, "features": features}

    if interior_min and left_drop_to_peak > 8 and right_drop_to_peak > 8:
        return {
            "curve_role": "power_density_curve",
            "axis_binding_hint": "right_y",
            "confidence": min(0.95, 0.55 + min(left_drop_to_peak, right_drop_to_peak) / 80),
            "features": features,
        }
    if data_y_decreases > 8:
        return {
            "curve_role": "voltage_curve",
            "axis_binding_hint": "left_y",
            "confidence": min(0.9, 0.5 + data_y_decreases / 120),
            "features": features,
        }
    return {
        "curve_role": "unknown_curve",
        "axis_binding_hint": None,
        "confidence": 0.25,
        "features": features,
    }


def _refine_dual_y_trace_roles(traces: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        series_id = trace.get("series_id")
        if isinstance(series_id, str):
            grouped[series_id].append(trace)

    for group in grouped.values():
        if len(group) < 2:
            continue

        def power_score(trace: dict[str, Any]) -> float:
            features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
            if not features.get("interior_min_y_px"):
                return 0.0
            return float(features.get("rise_peak_fall_score_px") or 0.0)

        def voltage_score(trace: dict[str, Any]) -> float:
            features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
            return float(features.get("data_y_decreasing_score_px") or 0.0)

        power_trace = max(group, key=power_score)
        if power_score(power_trace) > 6:
            power_trace["curve_role"] = "power_density_curve"
            power_trace["axis_binding_hint"] = "right_y"
            power_trace["confidence"] = max(float(power_trace.get("confidence") or 0.0), 0.68)
            power_trace["role_refinement"] = {
                "source": "dual_y_same_color_trace_pairing",
                "score": power_score(power_trace),
            }

        voltage_candidates = [trace for trace in group if trace is not power_trace]
        if not voltage_candidates:
            continue
        voltage_trace = max(voltage_candidates, key=voltage_score)
        if voltage_score(voltage_trace) > 4:
            voltage_trace["curve_role"] = "voltage_curve"
            voltage_trace["axis_binding_hint"] = "left_y"
            voltage_trace["confidence"] = max(float(voltage_trace.get("confidence") or 0.0), 0.62)
            voltage_trace["role_refinement"] = {
                "source": "dual_y_same_color_trace_pairing",
                "score": voltage_score(voltage_trace),
            }


def _trace_final_rejection_reason(trace: dict[str, Any], *, is_dual_y_power_density: bool) -> str | None:
    points = trace.get("points")
    point_count = len(points) if isinstance(points, list) else 0
    if point_count < 20:
        return "TOO_FEW_POINTS"

    if not is_dual_y_power_density:
        return None

    role = trace.get("curve_role")
    axis_hint = trace.get("axis_binding_hint")
    if role not in {"voltage_curve", "power_density_curve"}:
        return "UNSUPPORTED_CURVE_ROLE"
    if role == "voltage_curve" and axis_hint != "left_y":
        return "VOLTAGE_AXIS_BINDING_MISMATCH"
    if role == "power_density_curve" and axis_hint != "right_y":
        return "POWER_DENSITY_AXIS_BINDING_MISMATCH"

    features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
    if role == "voltage_curve" and float(features.get("data_y_decreasing_score_px") or 0.0) < 8:
        return "VOLTAGE_SHAPE_NOT_MONOTONIC"
    if role == "power_density_curve" and float(features.get("rise_peak_fall_score_px") or 0.0) < 6:
        return "POWER_DENSITY_SHAPE_NOT_PEAKED"

    if float(trace.get("confidence") or 0.0) < 0.6:
        return "LOW_ROLE_CONFIDENCE"
    return None


def _score_final_trace(trace: dict[str, Any]) -> float:
    features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
    point_count = len(trace.get("points") or [])
    confidence = float(trace.get("confidence") or 0.0)
    role = trace.get("curve_role")
    method_bonus = 0.0
    if trace.get("extraction_method") == "red_curve_after_voltage_exclusion_centerline":
        method_bonus = 250.0
    elif trace.get("extraction_method") == "same_color_filled_marker_monotonic_voltage_centerline":
        method_bonus = 360.0
    elif trace.get("extraction_method") == "same_color_after_power_density_exclusion_voltage_centerline":
        method_bonus = 320.0
    elif trace.get("extraction_method") == "same_color_voltage_fragment_merge_centerline":
        method_bonus = 170.0
    if role == "power_density_curve":
        shape_score = float(features.get("rise_peak_fall_score_px") or 0.0)
        lvp_bonus = float(features.get("lvp_physics_fit_score") or 0.0)
    elif role == "voltage_curve":
        shape_score = float(features.get("data_y_decreasing_score_px") or 0.0)
        lvp_bonus = 0.0
    else:
        shape_score = 0.0
        lvp_bonus = 0.0
    return point_count + confidence * 100.0 + shape_score + method_bonus + lvp_bonus


def _trace_points_to_data_for_axis(
    trace: dict[str, Any],
    axis_mapping: dict[str, Any],
    *,
    crop_offset_xy: list[int] | None,
    axis_hint: str,
) -> list[list[float]]:
    x_model = axis_mapping.get("axes", {}).get("bottom", {}).get("model")
    y_role = "right" if axis_hint == "right_y" else "left"
    y_model = axis_mapping.get("axes", {}).get(y_role, {}).get("model")
    if not x_model or not y_model:
        return []
    ox, oy = crop_offset_xy if isinstance(crop_offset_xy, list) and len(crop_offset_xy) == 2 else [0, 0]
    points_data: list[list[float]] = []
    for point in trace.get("points", []):
        if not isinstance(point, list) or len(point) < 2:
            continue
        x_panel = float(point[0]) + float(ox)
        y_panel = float(point[1]) + float(oy)
        points_data.append([_apply_linear(x_model, x_panel), _apply_linear(y_model, y_panel)])
    return points_data


def _lvp_power_fit_metrics(
    voltage_trace: dict[str, Any],
    power_trace: dict[str, Any],
    axis_mapping: dict[str, Any],
    *,
    crop_offset_xy: list[int] | None,
) -> dict[str, Any] | None:
    voltage_points = _trace_points_to_data_for_axis(
        voltage_trace,
        axis_mapping,
        crop_offset_xy=crop_offset_xy,
        axis_hint="left_y",
    )
    power_points = _trace_points_to_data_for_axis(
        power_trace,
        axis_mapping,
        crop_offset_xy=crop_offset_xy,
        axis_hint="right_y",
    )
    voltage_points = [point for point in voltage_points if len(point) >= 2]
    power_points = [point for point in power_points if len(point) >= 2]
    if len(voltage_points) < 20 or len(power_points) < 20:
        return None

    voltage_model = _linear_model(
        [{"pixel": float(point[0]), "value": float(point[1])} for point in voltage_points]
    )
    if not voltage_model:
        return None

    residuals: list[float] = []
    fitted_count = 0
    fitted_peak = 0.0
    for point in _dedupe_points_by_x(power_points):
        x_value = float(point[0])
        measured_power = float(point[1])
        voltage_at_x = max(0.0, _apply_linear(voltage_model, x_value))
        expected_power = x_value * voltage_at_x / 1000.0
        if not math.isfinite(expected_power):
            continue
        residuals.append(measured_power - expected_power)
        fitted_peak = max(fitted_peak, expected_power)
        fitted_count += 1
    if fitted_count < max(15, int(round(len(power_points) * 0.35))) or fitted_peak <= 0:
        return None

    median_abs_residual = _median_absolute(residuals)
    residual_over_peak = median_abs_residual / fitted_peak
    coverage_ratio = fitted_count / max(1, len(_dedupe_points_by_x(power_points)))
    # Good physics agreement should dominate shape-only power selection, but not rescue unusable masks.
    fit_score = max(0.0, 900.0 * (1.0 - min(residual_over_peak, 1.0))) + 100.0 * coverage_ratio
    return {
        "method": "lvp_candidate_power_score_from_voltage_fit",
        "formula": "power_density_W_cm2 = current_density_mA_cm2 * voltage_V / 1000",
        "source_voltage_trace_id": voltage_trace.get("trace_id"),
        "source_power_trace_id": power_trace.get("trace_id"),
        "source_voltage_fit": voltage_model,
        "candidate_power_point_count": len(power_points),
        "fitted_point_count": fitted_count,
        "median_abs_residual": median_abs_residual,
        "median_abs_residual_over_peak": residual_over_peak,
        "coverage_ratio": coverage_ratio,
        "score": fit_score,
    }


def _annotate_power_candidates_with_lvp_fit(
    voltage_trace: dict[str, Any] | None,
    power_candidates: list[dict[str, Any]],
    axis_mapping: dict[str, Any] | None,
    *,
    crop_offset_xy: list[int] | None,
) -> None:
    if not voltage_trace or not axis_mapping:
        return
    for power_trace in power_candidates:
        metrics = _lvp_power_fit_metrics(
            voltage_trace,
            power_trace,
            axis_mapping,
            crop_offset_xy=crop_offset_xy,
        )
        if not metrics:
            continue
        features = power_trace.get("features")
        if not isinstance(features, dict):
            features = {}
            power_trace["features"] = features
        features["lvp_physics_fit"] = metrics
        features["lvp_physics_fit_score"] = metrics["score"]
        power_trace["physics_validation"] = {
            "method": metrics["method"],
            "source_voltage_trace_id": metrics["source_voltage_trace_id"],
            "median_abs_residual_over_peak": metrics["median_abs_residual_over_peak"],
            "coverage_ratio": metrics["coverage_ratio"],
            "score": metrics["score"],
        }


def select_final_traces_for_data_series(
    traces: list[dict[str, Any]],
    *,
    is_dual_y_power_density: bool,
    axis_mapping: dict[str, Any] | None = None,
    crop_offset_xy: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep exploratory traces separate from the final data-series artifact."""
    for trace in traces:
        trace["final_trace_status"] = "candidate"
        trace.pop("final_trace_reject_reason", None)

    if not is_dual_y_power_density:
        for trace in traces:
            reason = _trace_final_rejection_reason(trace, is_dual_y_power_density=False)
            if reason:
                trace["final_trace_status"] = "rejected"
                trace["final_trace_reject_reason"] = reason
            else:
                trace["final_trace_status"] = "accepted"
        return [trace for trace in traces if trace.get("final_trace_status") == "accepted"], []

    eligible_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[str] = []
    for trace in traces:
        reason = _trace_final_rejection_reason(trace, is_dual_y_power_density=True)
        if reason:
            trace["final_trace_status"] = "rejected"
            trace["final_trace_reject_reason"] = reason
            continue
        role = str(trace.get("curve_role"))
        eligible_by_role[role].append(trace)

    selected: list[dict[str, Any]] = []
    has_series_ids = any(
        isinstance(trace.get("series_id"), str)
        for traces_for_role in eligible_by_role.values()
        for trace in traces_for_role
    )
    eligible_series_ids = (
        sorted(
            {
                str(trace.get("series_id"))
                for traces_for_role in eligible_by_role.values()
                for trace in traces_for_role
                if isinstance(trace.get("series_id"), str)
            }
        )
        if has_series_ids
        else ["__all__"]
    )
    for series_id in eligible_series_ids:
        voltage_candidates = [
            trace
            for trace in eligible_by_role.get("voltage_curve", [])
            if not has_series_ids or trace.get("series_id") == series_id
        ]
        power_candidates = [
            trace
            for trace in eligible_by_role.get("power_density_curve", [])
            if not has_series_ids or trace.get("series_id") == series_id
        ]

        voltage_winner: dict[str, Any] | None = None
        if not voltage_candidates:
            issues.append(f"{series_id}:voltage_curve:NO_FINAL_QUALITY_TRACE")
        else:
            voltage_winner = max(voltage_candidates, key=_score_final_trace)
            voltage_winner["final_trace_status"] = "accepted"
            selected.append(voltage_winner)
            for trace in voltage_candidates:
                if trace is voltage_winner:
                    continue
                trace["final_trace_status"] = "rejected"
                trace["final_trace_reject_reason"] = "LOWER_SCORING_DUPLICATE_ROLE_TRACE"

        _annotate_power_candidates_with_lvp_fit(
            voltage_winner,
            power_candidates,
            axis_mapping,
            crop_offset_xy=crop_offset_xy,
        )
        if not power_candidates:
            issues.append(f"{series_id}:power_density_curve:NO_FINAL_QUALITY_TRACE")
        else:
            power_winner = max(power_candidates, key=_score_final_trace)
            power_winner["final_trace_status"] = "accepted"
            selected.append(power_winner)
            for trace in power_candidates:
                if trace is power_winner:
                    continue
                trace["final_trace_status"] = "rejected"
                trace["final_trace_reject_reason"] = "LOWER_SCORING_DUPLICATE_ROLE_TRACE"

    if not selected and traces:
        fallback = max(traces, key=_score_final_trace)
        fallback["final_trace_status"] = "accepted"
        fallback["final_trace_reject_reason"] = "FALLBACK_BEST_AVAILABLE"
        selected.append(fallback)
        issues.append("FALLBACK_BEST_AVAILABLE_TRACE_USED")

    return selected, issues


def _merge_voltage_fragment_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if trace.get("curve_role") != "voltage_curve" or trace.get("axis_binding_hint") != "left_y":
            continue
        if trace.get("extraction_method") != "color_mask_column_tracking_thick_stroke_centerline":
            continue
        series_id = trace.get("series_id")
        if not isinstance(series_id, str):
            continue
        features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
        if float(features.get("data_y_decreasing_score_px") or 0.0) < 8:
            continue
        if len(trace.get("points") or []) < 20:
            continue
        grouped[series_id].append(trace)

    merged: list[dict[str, Any]] = []
    for series_id, candidates in grouped.items():
        if len(candidates) < 2:
            continue
        by_x: dict[int, list[int]] = defaultdict(list)
        best_coverage = 0
        for trace in candidates:
            xs: list[int] = []
            for point in trace.get("points", []):
                if not isinstance(point, list) or len(point) < 2:
                    continue
                x, y = int(point[0]), int(point[1])
                by_x[x].append(y)
                xs.append(x)
            if xs:
                best_coverage = max(best_coverage, max(xs) - min(xs))
        if len(by_x) < 20:
            continue
        averaged = [(x, int(round(sum(ys) / len(ys)))) for x, ys in sorted(by_x.items())]
        if averaged[-1][0] - averaged[0][0] < best_coverage + max(15, int(best_coverage * 0.15)):
            continue
        points = _smooth_track_points(averaged, window=7)
        role = classify_curve_role(points, is_dual_y_power_density=True)
        if role.get("curve_role") != "voltage_curve" or role.get("axis_binding_hint") != "left_y":
            continue
        features = role.get("features") if isinstance(role.get("features"), dict) else {}
        if float(features.get("data_y_decreasing_score_px") or 0.0) < 20:
            continue
        source_ids = [str(trace.get("trace_id")) for trace in candidates]
        first = candidates[0]
        merged.append(
            {
                "trace_id": f"{series_id}_voltage_fragment_merged_trace",
                "series_id": series_id,
                "legend_label": first.get("legend_label"),
                "curve_role": "voltage_curve",
                "axis_binding_hint": "left_y",
                "series_color_rgb": first.get("series_color_rgb"),
                "points": [[x, y] for x, y in points],
                "confidence": max(float(role.get("confidence") or 0.0), 0.82),
                "extraction_method": "same_color_voltage_fragment_merge_centerline",
                "features": {
                    **features,
                    "source_fragment_trace_ids": source_ids,
                    "centerline_point_count": len(points),
                    "centerline_method": "merged_voltage_fragments_by_x_average",
                },
                "coordinate_space": first.get("coordinate_space") or "crop_px",
                "mask_ref": first.get("mask_ref"),
            }
        )
    return merged


# ###수정 차이 17: stroke centerline과 특수 재추출을 우회하여 greedy track 좌표를 사용합니다.
def extract_curve_traces(
    crop: Image.Image,
    series_candidates: list[dict[str, Any]],
    *,
    masks_dir: Path,
    is_dual_y_power_density: bool,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
    refine_dual_y_roles: bool = True,
    non_curve_mask: Image.Image | None = None,
    # ###수정 추가 수정: grid/범례 제외 영역을 보간 support에서도 침범하지 않습니다.
    tracking_forbidden_mask: Image.Image | None = None,
    # ###
) -> tuple[list[dict[str, Any]], list[str]]:
    traces: list[dict[str, Any]] = []
    issues: list[str] = []
    masks_dir.mkdir(parents=True, exist_ok=True)
    # ###수정 추가 수정: 짧지만 유효한 curve 조각은 기존 12% 폭 필터에서 미리 버리지 않고 VLM에 전달합니다.
    min_width = max(8, int(crop.width * 0.02))
    # ###

    for candidate in series_candidates:
        colors = _candidate_target_colors(candidate)
        if not colors:
            issues.append(f"{candidate['series_id']}:INVALID_TARGET_COLOR")
            continue
        mask, points = _mask_for_color(
            crop,
            colors,
            float(candidate.get("color_distance", 70)),
            exclusion_bboxes_crop_px=exclusion_bboxes_crop_px,
            non_curve_mask=non_curve_mask,
        )
        mask_path = masks_dir / f"{candidate['series_id']}.png"
        mask.save(mask_path)
        # ###수정 추가 수정: 관측 mask와 별도로 bounded gap/marker support를 만들고 10/10 greedy tracking에 사용합니다.
        from .cmsl_curve_support import build_tracking_support, repair_crossing_tracks
        support, inferred, support_info = build_tracking_support(mask, tracker=_tracks_from_mask_points, forbidden_mask=tracking_forbidden_mask)
        support_dir = masks_dir.parent / "tracking_support"
        support_dir.mkdir(parents=True, exist_ok=True)
        support.save(support_dir / f"{candidate['series_id']}.png")
        inferred.save(support_dir / f"{candidate['series_id']}_inferred.png")
        _write_json_atomic(support_dir / f"{candidate['series_id']}.json", support_info)
        points = _points_from_mask(support)
        # ###
        # ###수정 추가 수정: CMSL 후보의 최소 폭은 2%로 완화하고 최종 20점·VLM validity 검사를 유지합니다.
        tracks = _tracks_from_mask_points(points, image_width=crop.width, min_width_px=min_width, min_coverage_fraction=0.02)
        # ###
        raw_track_count = len(tracks)
        tracks = _merge_near_duplicate_tracks(
            tracks,
            max_median_y_distance_px=max(10, crop.height * 0.05),
        )
        # ###수정 추가 수정: 교차점의 branch 전환을 좌표 보존 tail 교환으로 바로잡습니다.
        tracks, crossing_swaps = repair_crossing_tracks(tracks)
        support_info["crossing_tail_swap_count"] = crossing_swaps
        _write_json_atomic(support_dir / f"{candidate['series_id']}.json", support_info)
        # ###
        if not tracks:
            issues.append(f"{candidate['series_id']}:ZERO_TRACE_TRACKS")
        for idx, track in enumerate(tracks[:4], start=1):
            centerline = sorted(track)
            role = classify_curve_role(
                centerline,
                is_dual_y_power_density=is_dual_y_power_density,
                axis_binding_hint=candidate.get("candidate_axis_binding_hint"),
            )
            forced_curve_role = candidate.get("forced_curve_role")
            if isinstance(forced_curve_role, str):
                role = {
                    **role,
                    "curve_role": forced_curve_role,
                    "axis_binding_hint": candidate.get("forced_axis_binding_hint"),
                    "confidence": max(float(role.get("confidence") or 0.0), 0.7),
                }
            traces.append(
                {
                    "trace_id": f"{candidate['series_id']}_trace_{idx:02d}",
                    "series_id": candidate["series_id"],
                    "legend_label": candidate.get("legend_label"),
                    # ###수정 추가 수정: 동일 색의 여러 범례를 curve별 axis/label 판정 입력으로 전달합니다.
                    "legend_options": candidate.get("legend_options", []),
                    # ###
                    "curve_role": role["curve_role"],
                    "axis_binding_hint": role["axis_binding_hint"],
                    "series_color_rgb": candidate["target_color_rgb"],
                    "points": [[x, y] for x, y in sorted(centerline)],
                    "confidence": role["confidence"],
                    "extraction_method": "cmsl_greedy_euclidean_column_tracking",
                    "features": {
                        **role["features"],
                        "raw_track_point_count": len(track),
                        "raw_track_count_before_edge_merge": raw_track_count,
                        "track_count_after_edge_merge": len(tracks),
                        "centerline_point_count": len(centerline),
                        "centerline_method": "greedy_track_points_without_stroke_reconstruction",
                        # ###수정 추가 수정: inferred support의 존재와 원본 mask 위치를 trace에 명시합니다.
                        "tracking_support_ref": f"tracking_support/{candidate['series_id']}.json",
                        "inferred_support_pixel_count": support_info["inferred_pixel_count"],
                        "crossing_tail_swap_count": crossing_swaps,
                        # ###
                    },
                    "coordinate_space": "crop_px",
                    "mask_ref": str(mask_path.name),
                }
            )
    if is_dual_y_power_density and refine_dual_y_roles:
        _refine_dual_y_trace_roles(traces)
    return traces, issues
# ###


def _best_voltage_trace_for_exclusion(
    traces: list[dict[str, Any]],
    *,
    series_id: str | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for trace in traces:
        if series_id and trace.get("series_id") != series_id:
            continue
        if trace.get("curve_role") != "voltage_curve" or trace.get("axis_binding_hint") != "left_y":
            continue
        features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
        if float(trace.get("confidence") or 0.0) < 0.75:
            continue
        if len(trace.get("points") or []) < 20:
            continue
        if float(features.get("data_y_decreasing_score_px") or 0.0) < 35:
            continue
        candidates.append(trace)
    if not candidates:
        return None
    return max(candidates, key=_score_final_trace)


def _best_power_density_trace_for_exclusion(
    traces: list[dict[str, Any]],
    *,
    series_id: str | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for trace in traces:
        if series_id and trace.get("series_id") != series_id:
            continue
        if trace.get("curve_role") != "power_density_curve" or trace.get("axis_binding_hint") != "right_y":
            continue
        features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
        if len(trace.get("points") or []) < 20:
            continue
        rise_peak_fall_score = float(features.get("rise_peak_fall_score_px") or 0.0)
        method = trace.get("extraction_method")
        if method == "same_color_after_voltage_exclusion_power_density_centerline":
            points = trace.get("points") or []
            xs = [point[0] for point in points if isinstance(point, list) and len(point) >= 2]
            x_coverage = max(xs) - min(xs) if xs else 0
            if rise_peak_fall_score < 8 or x_coverage < 80 or float(trace.get("confidence") or 0.0) < 0.8:
                continue
        elif rise_peak_fall_score < 12:
            continue
        candidates.append(trace)
    if not candidates:
        return None

    def score(trace: dict[str, Any]) -> float:
        points = trace.get("points") or []
        xs = [point[0] for point in points if isinstance(point, list) and len(point) >= 2]
        x_coverage = max(xs) - min(xs) if xs else 0
        features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
        peak_score = float(features.get("rise_peak_fall_score_px") or 0.0)
        return x_coverage + peak_score * 2.0 + len(points) * 0.2

    second_pass_candidates = [
        trace
        for trace in candidates
        if trace.get("extraction_method")
        in {
            "red_curve_after_voltage_exclusion_centerline",
            "same_color_after_voltage_exclusion_power_density_centerline",
        }
    ]
    return max(second_pass_candidates or candidates, key=score)


def _red_curve_points_after_voltage_exclusion(
    crop: Image.Image,
    voltage_trace: dict[str, Any],
    *,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
) -> tuple[Image.Image, set[Point]]:
    rgb = crop.convert("RGB")
    voltage_exclusion = Image.new("L", rgb.size, 0)
    draw = ImageDraw.Draw(voltage_exclusion)
    voltage_points = [tuple(point) for point in voltage_trace.get("points", [])]
    if len(voltage_points) >= 2:
        draw.line(
            voltage_points,
            fill=255,
            width=max(11, int(round(rgb.height * 0.045))),
            joint="curve",
        )
    for bbox in _normalize_exclusion_bboxes_crop_px(exclusion_bboxes_crop_px, rgb.width, rgb.height):
        draw.rectangle(tuple(bbox), fill=255)

    mask = Image.new("L", rgb.size, 0)
    src = rgb.load()
    excluded = voltage_exclusion.load()
    mask_pixels = mask.load()
    points: set[Point] = set()
    for y in range(rgb.height):
        for x in range(rgb.width):
            if excluded[x, y]:
                continue
            r, g, b = src[x, y]
            brightness = _brightness((r, g, b))
            if r >= 180 and r - g >= 35 and r - b >= 35 and brightness < 248:
                mask_pixels[x, y] = 255
                points.add((x, y))
    return mask, points


def _mask_after_trace_exclusion(
    crop: Image.Image,
    colors: list[RGB],
    color_distance: float,
    exclusion_trace: dict[str, Any],
    *,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
    width_ratio: float = 0.05,
    include_faint_hue: bool = False,
) -> tuple[Image.Image, set[Point]]:
    mask, points = _mask_for_color(
        crop,
        colors,
        color_distance,
        exclusion_bboxes_crop_px=exclusion_bboxes_crop_px,
        include_faint_hue=include_faint_hue,
    )
    exclusion = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(exclusion)
    trace_points = [tuple(point) for point in exclusion_trace.get("points", [])]
    if len(trace_points) >= 2:
        draw.line(
            trace_points,
            fill=255,
            width=max(11, int(round(mask.height * width_ratio))),
            joint="curve",
        )
    excluded = exclusion.load()
    mask_pixels = mask.load()
    kept: set[Point] = set()
    for x, y in points:
        if excluded[x, y] > 0:
            mask_pixels[x, y] = 0
            continue
        kept.add((x, y))
    return mask, kept


def _best_power_density_centerline_from_track_subsets(
    tracks: list[list[Point]],
    *,
    image_size: tuple[int, int],
    stroke_width: int,
) -> tuple[list[Point], dict[str, Any] | None, list[int]] | None:
    candidates: list[tuple[float, list[Point], dict[str, Any], list[int]]] = []
    indexed_tracks = list(enumerate(tracks))
    max_subset_size = min(4, len(indexed_tracks))
    for subset_size in range(max_subset_size, 1, -1):
        for subset in combinations(indexed_tracks, subset_size):
            points: set[Point] = set()
            indexes: list[int] = []
            for index, track in subset:
                indexes.append(index)
                points.update(track)
            if not points:
                continue
            centerline = _centerline_from_thick_stroke(
                points,
                image_size=image_size,
                stroke_width=stroke_width,
            )
            role = classify_curve_role(centerline, is_dual_y_power_density=True)
            if role["curve_role"] != "power_density_curve":
                continue
            xs = [x for x, _y in centerline]
            x_coverage = max(xs) - min(xs) if xs else 0
            features = role.get("features") if isinstance(role.get("features"), dict) else {}
            rise_score = float(features.get("rise_peak_fall_score_px") or 0.0)
            score = len(centerline) + x_coverage * 0.2 + rise_score * 8.0
            candidates.append((score, centerline, role, indexes))
    if not candidates:
        return None
    _score, centerline, role, indexes = max(candidates, key=lambda item: item[0])
    return centerline, role, indexes


def _filter_power_density_tracks(
    tracks: list[list[Point]],
    *,
    image_height: int,
) -> list[list[Point]]:
    filtered: list[list[Point]] = []
    for track in tracks:
        ys = [y for _x, y in track]
        xs = [x for x, _y in track]
        if not xs or not ys:
            continue
        y_span = max(ys) - min(ys)
        x_span = max(xs) - min(xs)
        # Voltage leftovers near the top-left can be picked up by faint-hue matching.
        if max(ys) < image_height * 0.20 and y_span < image_height * 0.08:
            continue
        if x_span < 20 and len(track) < 20:
            continue
        filtered.append(track)
    return filtered or tracks


def _split_track_by_x_gap(track: list[Point], *, max_gap_px: int) -> list[list[Point]]:
    if not track:
        return []
    segments: list[list[Point]] = []
    current: list[Point] = []
    previous_x: int | None = None
    for point in sorted(track):
        x, _y = point
        if previous_x is not None and x - previous_x > max_gap_px and current:
            segments.append(current)
            current = []
        current.append(point)
        previous_x = x
    if current:
        segments.append(current)
    return segments


def _trim_endpoint_sparse_islands(
    track: list[Point],
    *,
    image_width: int,
    max_gap_px: int = 8,
) -> tuple[list[Point], list[list[int]]]:
    """Drop tiny endpoint islands that would otherwise be bridged into a false curve."""
    segments = _split_track_by_x_gap(track, max_gap_px=max_gap_px)
    if len(segments) <= 1:
        return track, []

    trimmed_ranges: list[list[int]] = []
    min_remaining_width = max(20, int(round(image_width * 0.18)))
    max_endpoint_span = max(4, int(round(image_width * 0.04)))
    max_endpoint_points = max(6, int(round(image_width * 0.04)))

    def segment_x_range(segment: list[Point]) -> tuple[int, int]:
        xs = [x for x, _y in segment]
        return min(xs), max(xs)

    def is_small_endpoint(segment: list[Point]) -> bool:
        x1, x2 = segment_x_range(segment)
        return (x2 - x1 + 1) <= max_endpoint_span or len(segment) <= max_endpoint_points

    kept = segments[:]
    while len(kept) > 1 and is_small_endpoint(kept[0]):
        candidate = kept[1:]
        kept_x1 = min(segment_x_range(segment)[0] for segment in candidate)
        kept_x2 = max(segment_x_range(segment)[1] for segment in candidate)
        if kept_x2 - kept_x1 + 1 < min_remaining_width:
            break
        x1, x2 = segment_x_range(kept[0])
        trimmed_ranges.append([x1, x2])
        kept = candidate
    while len(kept) > 1 and is_small_endpoint(kept[-1]):
        candidate = kept[:-1]
        kept_x1 = min(segment_x_range(segment)[0] for segment in candidate)
        kept_x2 = max(segment_x_range(segment)[1] for segment in candidate)
        if kept_x2 - kept_x1 + 1 < min_remaining_width:
            break
        x1, x2 = segment_x_range(kept[-1])
        trimmed_ranges.append([x1, x2])
        kept = candidate

    if not trimmed_ranges:
        return track, []
    return [point for segment in kept for point in segment], trimmed_ranges


def _classify_power_density_tracks(
    tracks: list[list[Point]],
    *,
    image_size: tuple[int, int],
    stroke_width: int,
) -> list[dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    for index, track in enumerate(tracks):
        trimmed_track, trimmed_endpoint_ranges = _trim_endpoint_sparse_islands(
            track,
            image_width=image_size[0],
        )
        centerline = _centerline_from_thick_stroke(
            trimmed_track,
            image_size=image_size,
            stroke_width=stroke_width,
        )
        role = classify_curve_role(centerline, is_dual_y_power_density=True)
        classified.append(
            {
                "track_index": index,
                "track": trimmed_track,
                "raw_track_point_count": len(track),
                "trimmed_endpoint_island_ranges_x": trimmed_endpoint_ranges,
                "centerline": centerline,
                "role": role,
            }
        )
    return classified


def _score_power_residual_track(item: dict[str, Any], *, image_width: int) -> float:
    centerline = item.get("centerline") if isinstance(item.get("centerline"), list) else []
    if len(centerline) < 20:
        return 0.0
    xs = [point[0] for point in centerline if isinstance(point, tuple) and len(point) >= 2]
    if not xs:
        return 0.0
    x_coverage = max(xs) - min(xs)
    if x_coverage < max(35, image_width * 0.18):
        return 0.0
    role = item.get("role") if isinstance(item.get("role"), dict) else {}
    features = role.get("features") if isinstance(role.get("features"), dict) else {}
    rise_score = float(features.get("rise_peak_fall_score_px") or 0.0)
    curve_role = role.get("curve_role")
    if curve_role != "power_density_curve" and not (x_coverage >= image_width * 0.45 and rise_score >= 15):
        return 0.0
    role_bonus = 260.0 if curve_role == "power_density_curve" else 120.0
    return x_coverage * 3.0 + len(centerline) * 0.3 + rise_score * 2.0 + role_bonus


# ###수정 차이 15: 추가 tracking 호출의 연결 간격도 10/10 px로 통일합니다.
def extract_power_density_after_voltage_exclusion(
    crop: Image.Image,
    series_candidates: list[dict[str, Any]],
    traces: list[dict[str, Any]] | None = None,
    *,
    masks_dir: Path,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if traces is None:
        traces = series_candidates
        series_candidates = []

    if not series_candidates:
        voltage_trace = _best_voltage_trace_for_exclusion(traces)
        if voltage_trace is None:
            return [], ["POWER_DENSITY_SECOND_PASS_SKIPPED:NO_STABLE_VOLTAGE_TRACE"]
        return _extract_red_power_density_after_voltage_exclusion(
            crop,
            voltage_trace,
            masks_dir=masks_dir,
            exclusion_bboxes_crop_px=exclusion_bboxes_crop_px,
        )

    extracted: list[dict[str, Any]] = []
    issues: list[str] = []
    masks_dir.mkdir(parents=True, exist_ok=True)
    min_width = max(8, int(crop.width * 0.12))
    for candidate in series_candidates:
        series_id = candidate.get("series_id")
        if not isinstance(series_id, str):
            continue
        voltage_series_id = candidate.get("voltage_series_id")
        voltage_trace = _best_voltage_trace_for_exclusion(
            traces,
            series_id=voltage_series_id if isinstance(voltage_series_id, str) else series_id,
        )
        if voltage_trace is None:
            issues.append(f"{series_id}:POWER_DENSITY_SECOND_PASS_SKIPPED:NO_STABLE_VOLTAGE_TRACE")
            continue
        colors = _candidate_target_colors(candidate)
        if not colors:
            issues.append(f"{series_id}:POWER_DENSITY_SECOND_PASS_SKIPPED:INVALID_TARGET_COLOR")
            continue
        mask, points = _mask_after_trace_exclusion(
            crop,
            colors,
            float(candidate.get("color_distance", 70)),
            voltage_trace,
            exclusion_bboxes_crop_px=exclusion_bboxes_crop_px,
            width_ratio=0.06,
            include_faint_hue=False,
        )
        mask_path = masks_dir / f"{series_id}_power_density_after_voltage_exclusion.png"
        mask.save(mask_path)
        enhanced_mask = _strengthen_binary_mask(mask, radius_px=1)
        enhanced_mask_path = masks_dir / f"{series_id}_power_density_after_voltage_exclusion_enhanced.png"
        enhanced_mask.save(enhanced_mask_path)
        enhanced_points = _points_from_mask(enhanced_mask)
        tracks = _tracks_from_mask_points(
            enhanced_points,
            image_width=crop.width,
            min_width_px=max(8, int(crop.width * 0.08)),
            max_x_gap_px=10,
            max_y_gap_px=10,
        )
        tracks = _filter_power_density_tracks(tracks, image_height=crop.height)
        raw_track_count = len(tracks)
        tracks = _merge_near_duplicate_tracks(
            tracks,
            max_median_y_distance_px=max(10, crop.height * 0.05),
        )
        if not tracks:
            issues.append(f"{series_id}:POWER_DENSITY_SECOND_PASS_ZERO_TRACE_TRACKS")
            continue

        stroke_width = max(5, int(round(crop.height * 0.035)))
        classified_tracks = _classify_power_density_tracks(
            tracks,
            image_size=crop.size,
            stroke_width=stroke_width,
        )
        residual_candidates = [
            {
                **item,
                "residual_power_track_score": _score_power_residual_track(item, image_width=crop.width),
            }
            for item in classified_tracks
        ]
        residual_candidates = [
            item for item in residual_candidates if item["residual_power_track_score"] > 0
        ]
        if residual_candidates:
            best_item = max(residual_candidates, key=lambda item: item["residual_power_track_score"])
            original_role = best_item["role"]
            centerline = best_item["centerline"]
            extracted.append(
                {
                    "trace_id": f"{series_id}_power_density_after_voltage_exclusion_trace_best",
                    "series_id": series_id,
                    "legend_label": candidate.get("legend_label"),
                    "curve_role": "power_density_curve",
                    "axis_binding_hint": "right_y",
                    "series_color_rgb": candidate.get("target_color_rgb"),
                    "points": [[x, y] for x, y in sorted(centerline)],
                    "confidence": max(float(original_role.get("confidence") or 0.0), 0.82),
                    "extraction_method": "same_color_after_voltage_exclusion_power_density_centerline",
                    "features": {
                        **original_role["features"],
                        "source_voltage_trace_id": voltage_trace.get("trace_id"),
                        "raw_track_count_before_edge_merge": raw_track_count,
                        "track_count_after_edge_merge": len(tracks),
                        "selected_power_like_track_indexes": [best_item["track_index"]],
                        "rejected_non_power_track_indexes": [
                            item["track_index"]
                            for item in classified_tracks
                            if item["track_index"] != best_item["track_index"]
                        ],
                        "original_residual_track_role": original_role.get("curve_role"),
                        "residual_power_track_score": best_item["residual_power_track_score"],
                        "residual_track_scores": {
                            str(item["track_index"]): item["residual_power_track_score"]
                            for item in residual_candidates
                        },
                        "trimmed_endpoint_island_ranges_x_by_track": {
                            str(item["track_index"]): item["trimmed_endpoint_island_ranges_x"]
                            for item in classified_tracks
                            if item["trimmed_endpoint_island_ranges_x"]
                        },
                        "centerline_point_count": len(centerline),
                        "centerline_method": "best_scoring_residual_track_after_voltage_exclusion",
                        "mask_enhancement": "binary_close_radius_1_before_track_linking",
                        "enhanced_mask_ref": enhanced_mask_path.name,
                        "voltage_exclusion_width_px": max(11, int(round(crop.height * 0.06))),
                    },
                    "coordinate_space": "crop_px",
                    "mask_ref": str(mask_path.name),
                }
            )
            continue
        power_tracks = [
            item
            for item in classified_tracks
            if item["role"].get("curve_role") == "power_density_curve"
        ]
        merge_source_tracks = power_tracks or classified_tracks
        all_track_points: set[Point] = set()
        for item in merge_source_tracks:
            all_track_points.update(item["track"])
        merged_centerline = _centerline_from_thick_stroke(
            all_track_points,
            image_size=crop.size,
            stroke_width=stroke_width,
        )
        merged_role = classify_curve_role(merged_centerline, is_dual_y_power_density=True)
        if merged_role["curve_role"] == "power_density_curve":
            extracted.append(
                {
                    "trace_id": f"{series_id}_power_density_after_voltage_exclusion_trace_merged",
                    "series_id": series_id,
                    "legend_label": candidate.get("legend_label"),
                    "curve_role": "power_density_curve",
                    "axis_binding_hint": "right_y",
                    "series_color_rgb": candidate.get("target_color_rgb"),
                    "points": [[x, y] for x, y in sorted(merged_centerline)],
                    "confidence": max(float(merged_role.get("confidence") or 0.0), 0.88),
                    "extraction_method": "same_color_after_voltage_exclusion_power_density_centerline",
                    "features": {
                        **merged_role["features"],
                        "source_voltage_trace_id": voltage_trace.get("trace_id"),
                        "raw_track_count_before_edge_merge": raw_track_count,
                        "track_count_after_edge_merge": len(tracks),
                        "selected_power_like_track_indexes": [
                            item["track_index"] for item in merge_source_tracks
                        ],
                        "rejected_non_power_track_indexes": [
                            item["track_index"]
                            for item in classified_tracks
                            if item not in merge_source_tracks
                        ],
                        "trimmed_endpoint_island_ranges_x_by_track": {
                            str(item["track_index"]): item["trimmed_endpoint_island_ranges_x"]
                            for item in classified_tracks
                            if item["trimmed_endpoint_island_ranges_x"]
                        },
                        "centerline_point_count": len(merged_centerline),
                        "centerline_method": "merged_same_color_power_density_tracks_after_track_role_filter",
                        "mask_enhancement": "binary_close_radius_1_before_track_linking",
                        "enhanced_mask_ref": enhanced_mask_path.name,
                        "voltage_exclusion_width_px": max(11, int(round(crop.height * 0.06))),
                    },
                    "coordinate_space": "crop_px",
                    "mask_ref": str(mask_path.name),
                }
            )
            continue

        subset_power = _best_power_density_centerline_from_track_subsets(
            tracks,
            image_size=crop.size,
            stroke_width=max(5, int(round(crop.height * 0.035))),
        )
        if subset_power:
            subset_centerline, subset_role, subset_indexes = subset_power
            extracted.append(
                {
                    "trace_id": f"{series_id}_power_density_after_voltage_exclusion_trace_subset",
                    "series_id": series_id,
                    "legend_label": candidate.get("legend_label"),
                    "curve_role": "power_density_curve",
                    "axis_binding_hint": "right_y",
                    "series_color_rgb": candidate.get("target_color_rgb"),
                    "points": [[x, y] for x, y in sorted(subset_centerline)],
                    "confidence": max(float(subset_role.get("confidence") or 0.0), 0.62),
                    "extraction_method": "same_color_after_voltage_exclusion_power_density_centerline",
                    "features": {
                        **subset_role["features"],
                        "source_voltage_trace_id": voltage_trace.get("trace_id"),
                        "selected_track_indexes": subset_indexes,
                        "raw_track_count_before_edge_merge": raw_track_count,
                        "track_count_after_edge_merge": len(tracks),
                        "centerline_point_count": len(subset_centerline),
                        "centerline_method": "selected_power_density_track_subset_after_voltage_exclusion",
                        "mask_enhancement": "binary_close_radius_1_before_track_linking",
                        "enhanced_mask_ref": enhanced_mask_path.name,
                        "voltage_exclusion_width_px": max(11, int(round(crop.height * 0.06))),
                    },
                    "coordinate_space": "crop_px",
                    "mask_ref": str(mask_path.name),
                }
            )
            continue

        fragments: list[dict[str, Any]] = []
        for idx, track in enumerate(tracks[:2], start=1):
            centerline = _centerline_from_thick_stroke(
                track,
                image_size=crop.size,
                stroke_width=max(5, int(round(crop.height * 0.035))),
            )
            role = classify_curve_role(centerline, is_dual_y_power_density=True)
            if role["curve_role"] != "power_density_curve":
                continue
            fragments.append(
                {
                    "trace_id": f"{series_id}_power_density_after_voltage_exclusion_trace_{idx:02d}",
                    "series_id": series_id,
                    "legend_label": candidate.get("legend_label"),
                    "curve_role": "power_density_curve",
                    "axis_binding_hint": "right_y",
                    "series_color_rgb": candidate.get("target_color_rgb"),
                    "points": [[x, y] for x, y in sorted(centerline)],
                    "confidence": max(float(role.get("confidence") or 0.0), 0.85),
                    "extraction_method": "same_color_after_voltage_exclusion_power_density_centerline",
                    "features": {
                        **role["features"],
                        "source_voltage_trace_id": voltage_trace.get("trace_id"),
                        "raw_track_point_count": len(track),
                        "raw_track_count_before_edge_merge": raw_track_count,
                        "track_count_after_edge_merge": len(tracks),
                        "centerline_point_count": len(centerline),
                        "centerline_method": "same_color_voltage_excluded_power_branch_fit",
                        "mask_enhancement": "binary_close_radius_1_before_track_linking",
                        "enhanced_mask_ref": enhanced_mask_path.name,
                        "voltage_exclusion_width_px": max(11, int(round(crop.height * 0.06))),
                    },
                    "coordinate_space": "crop_px",
                    "mask_ref": str(mask_path.name),
                }
            )

        if not fragments:
            issues.append(f"{series_id}:POWER_DENSITY_SECOND_PASS_NO_POWER_DENSITY_SHAPE")
            continue
        if len(fragments) == 1:
            extracted.extend(fragments)
            continue

        by_x: dict[int, list[int]] = defaultdict(list)
        for trace in fragments:
            for point in trace.get("points", []):
                if not isinstance(point, list) or len(point) < 2:
                    continue
                x, y = int(point[0]), int(point[1])
                by_x[x].append(y)
        averaged = [(x, int(round(sum(ys) / len(ys)))) for x, ys in sorted(by_x.items())]
        merged_points = _smooth_track_points(averaged, window=9)
        role = classify_curve_role(merged_points, is_dual_y_power_density=True)
        extracted.append(
            {
                "trace_id": f"{series_id}_power_density_after_voltage_exclusion_trace_merged",
                "series_id": series_id,
                "legend_label": candidate.get("legend_label"),
                "curve_role": "power_density_curve",
                "axis_binding_hint": "right_y",
                "series_color_rgb": candidate.get("target_color_rgb"),
                "points": [[x, y] for x, y in merged_points],
                "confidence": max(float(role.get("confidence") or 0.0), 0.88),
                "extraction_method": "same_color_after_voltage_exclusion_power_density_centerline",
                "features": {
                    **role["features"],
                    "source_voltage_trace_id": voltage_trace.get("trace_id"),
                    "source_fragment_trace_ids": [trace["trace_id"] for trace in fragments],
                    "raw_track_count_before_edge_merge": raw_track_count,
                    "track_count_after_edge_merge": len(tracks),
                    "centerline_point_count": len(merged_points),
                    "centerline_method": "merged_same_color_power_density_fragments_column_topline",
                    "mask_enhancement": "binary_close_radius_1_before_track_linking",
                    "enhanced_mask_ref": enhanced_mask_path.name,
                    "voltage_exclusion_width_px": max(11, int(round(crop.height * 0.06))),
                },
                "coordinate_space": "crop_px",
                "mask_ref": str(mask_path.name),
            }
        )
    if not extracted and not issues:
        issues.append("POWER_DENSITY_SECOND_PASS_SKIPPED:NO_SERIES_CANDIDATES")
    return extracted, issues
# ###


# ###수정 차이 15: 추가 tracking 호출의 연결 간격도 10/10 px로 통일합니다.
def _extract_red_power_density_after_voltage_exclusion(
    crop: Image.Image,
    voltage_trace: dict[str, Any],
    *,
    masks_dir: Path,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if voltage_trace is None:
        return [], ["POWER_DENSITY_SECOND_PASS_SKIPPED:NO_STABLE_VOLTAGE_TRACE"]

    mask, points = _red_curve_points_after_voltage_exclusion(
        crop,
        voltage_trace,
        exclusion_bboxes_crop_px=exclusion_bboxes_crop_px,
    )
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / "power_density_after_voltage_exclusion.png"
    mask.save(mask_path)

    min_width = max(8, int(crop.width * 0.12))
    tracks = _tracks_from_mask_points(
        points,
        image_width=crop.width,
        min_width_px=min_width,
        max_x_gap_px=10,
        max_y_gap_px=10,
    )
    raw_track_count = len(tracks)
    tracks = _merge_near_duplicate_tracks(
        tracks,
        max_median_y_distance_px=max(10, crop.height * 0.05),
    )
    if not tracks:
        return [], ["POWER_DENSITY_SECOND_PASS_ZERO_TRACE_TRACKS"]

    legend_label = voltage_trace.get("legend_label")
    extracted: list[dict[str, Any]] = []
    for idx, track in enumerate(tracks[:2], start=1):
        centerline = _centerline_from_thick_stroke(
            track,
            image_size=crop.size,
            stroke_width=max(5, int(round(crop.height * 0.035))),
        )
        role = classify_curve_role(centerline, is_dual_y_power_density=True)
        if role["curve_role"] != "power_density_curve":
            continue
        extracted.append(
            {
                "trace_id": f"power_density_after_voltage_exclusion_trace_{idx:02d}",
                "series_id": "power_density_after_voltage_exclusion",
                "legend_label": legend_label,
                "curve_role": "power_density_curve",
                "axis_binding_hint": "right_y",
                "series_color_rgb": list(ROLE_RECOLOR_RGB["power_density_curve"]),
                "points": [[x, y] for x, y in sorted(centerline)],
                "confidence": max(float(role.get("confidence") or 0.0), 0.85),
                "extraction_method": "red_curve_after_voltage_exclusion_centerline",
                "features": {
                    **role["features"],
                    "source_voltage_trace_id": voltage_trace.get("trace_id"),
                    "raw_track_point_count": len(track),
                    "raw_track_count_before_edge_merge": raw_track_count,
                    "track_count_after_edge_merge": len(tracks),
                    "centerline_point_count": len(centerline),
                    "centerline_method": "smooth_polyline_thick_stroke_column_midpoint",
                    "voltage_exclusion_width_px": max(11, int(round(crop.height * 0.045))),
                },
                "coordinate_space": "crop_px",
                "mask_ref": str(mask_path.name),
            }
        )

    if not extracted:
        return [], ["POWER_DENSITY_SECOND_PASS_NO_POWER_DENSITY_SHAPE"]
    if len(extracted) > 1:
        by_x: dict[int, list[int]] = defaultdict(list)
        for trace in extracted:
            for point in trace.get("points", []):
                if not isinstance(point, list) or len(point) < 2:
                    continue
                x, y = int(point[0]), int(point[1])
                by_x[x].append(y)
        averaged = [(x, int(round(sum(ys) / len(ys)))) for x, ys in sorted(by_x.items())]
        merged_points = _smooth_track_points(averaged, window=9)
        role = classify_curve_role(merged_points, is_dual_y_power_density=True)
        merged = {
            "trace_id": "power_density_after_voltage_exclusion_trace_merged",
            "series_id": "power_density_after_voltage_exclusion",
            "legend_label": legend_label,
            "curve_role": "power_density_curve",
            "axis_binding_hint": "right_y",
            "series_color_rgb": list(ROLE_RECOLOR_RGB["power_density_curve"]),
            "points": [[x, y] for x, y in merged_points],
            "confidence": max(float(role.get("confidence") or 0.0), 0.88),
            "extraction_method": "red_curve_after_voltage_exclusion_centerline",
            "features": {
                **role["features"],
                "source_voltage_trace_id": voltage_trace.get("trace_id"),
                "source_fragment_trace_ids": [trace["trace_id"] for trace in extracted],
                "raw_track_count_before_edge_merge": raw_track_count,
                "track_count_after_edge_merge": len(tracks),
                "centerline_point_count": len(merged_points),
                "centerline_method": "merged_power_density_fragments_column_topline",
                "voltage_exclusion_width_px": max(11, int(round(crop.height * 0.045))),
            },
            "coordinate_space": "crop_px",
            "mask_ref": str(mask_path.name),
        }
        return [merged], []
    return extracted, []
# ###


def extract_voltage_after_power_density_exclusion(
    crop: Image.Image,
    series_candidates: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    masks_dir: Path,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    extracted: list[dict[str, Any]] = []
    issues: list[str] = []
    masks_dir.mkdir(parents=True, exist_ok=True)
    for candidate in series_candidates:
        series_id = candidate.get("series_id")
        if not isinstance(series_id, str):
            continue
        power_trace = _best_power_density_trace_for_exclusion(traces, series_id=series_id)
        if power_trace is None and candidate.get("legend_label"):
            same_label_traces = [
                trace
                for trace in traces
                if trace.get("legend_label") == candidate.get("legend_label")
            ]
            power_trace = _best_power_density_trace_for_exclusion(same_label_traces)
        if power_trace is None:
            issues.append(f"{series_id}:VOLTAGE_AFTER_POWER_EXCLUSION_SKIPPED:NO_STABLE_POWER_TRACE")
            continue
        colors = _candidate_target_colors(candidate)
        if not colors:
            continue
        mask, _points = _mask_after_trace_exclusion(
            crop,
            colors,
            float(candidate.get("color_distance", 70)),
            power_trace,
            exclusion_bboxes_crop_px=exclusion_bboxes_crop_px,
            width_ratio=0.045,
        )
        mask_path = masks_dir / f"{candidate['series_id']}_after_power_density_exclusion.png"
        mask.save(mask_path)
        centerline = _monotonic_voltage_centerline_from_mask(mask)
        if not centerline:
            issues.append(f"{candidate['series_id']}:NO_VOLTAGE_TRACE_AFTER_POWER_DENSITY_EXCLUSION")
            continue
        role = classify_curve_role(centerline, is_dual_y_power_density=True)
        features = role.get("features") if isinstance(role.get("features"), dict) else {}
        if (
            role.get("curve_role") != "voltage_curve"
            or role.get("axis_binding_hint") != "left_y"
            or float(features.get("data_y_decreasing_score_px") or 0.0) < 8
        ):
            issues.append(
                f"{candidate['series_id']}:VOLTAGE_AFTER_POWER_EXCLUSION_REJECTED:NON_MONOTONIC_VOLTAGE_SHAPE"
            )
            continue
        extracted.append(
            {
                "trace_id": f"{candidate['series_id']}_voltage_after_power_density_exclusion_trace",
                "series_id": candidate["series_id"],
                "legend_label": candidate.get("legend_label"),
                "curve_role": "voltage_curve",
                "axis_binding_hint": "left_y",
                "series_color_rgb": candidate["target_color_rgb"],
                "points": [[x, y] for x, y in sorted(centerline)],
                "confidence": max(float(role.get("confidence") or 0.0), 0.9),
                "extraction_method": "same_color_after_power_density_exclusion_voltage_centerline",
                "features": {
                    **role["features"],
                    "source_power_density_trace_id": power_trace.get("trace_id"),
                    "centerline_point_count": len(centerline),
                    "centerline_method": "same_color_power_excluded_monotonic_branch_fit",
                    "power_density_exclusion_width_px": max(11, int(round(crop.height * 0.045))),
                },
                "coordinate_space": "crop_px",
                "mask_ref": str(mask_path.name),
            }
        )
    if not extracted and not issues:
        issues.append("VOLTAGE_AFTER_POWER_EXCLUSION_SKIPPED:NO_MATCHING_SERIES_CANDIDATE")
    return extracted, issues


def _role_color(curve_role: Any) -> RGB:
    return ROLE_RECOLOR_RGB.get(str(curve_role), ROLE_RECOLOR_RGB["unknown_curve"])


def _safe_recolor_id(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    safe = "".join(char if char.isalnum() else "_" for char in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"


def _role_recolor_assignments(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    role_counts: Counter[str] = Counter()
    seen_ids: Counter[str] = Counter()
    assignments: list[dict[str, Any]] = []
    for trace in traces:
        role = str(trace.get("curve_role") or "unknown_curve")
        source_series_id = str(trace.get("series_id") or trace.get("trace_id") or "series")
        palette = ROLE_RECOLOR_PALETTES.get(role, ROLE_RECOLOR_PALETTES["unknown_curve"])
        role_index = role_counts[role]
        role_counts[role] += 1
        base_id = f"role_{_safe_recolor_id(source_series_id)}_{_safe_recolor_id(role)}"
        seen_ids[base_id] += 1
        recolor_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"
        assignments.append(
            {
                "trace": trace,
                "recolor_series_id": recolor_id,
                "source_series_id": source_series_id,
                "source_trace_id": trace.get("trace_id"),
                "legend_label": trace.get("legend_label"),
                "curve_role": role,
                "axis_binding_hint": trace.get("axis_binding_hint"),
                "rgb": palette[role_index % len(palette)],
            }
        )
    return assignments


def _draw_role_recolored_crop(crop: Image.Image, traces: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    recolored = Image.new("RGB", crop.size, "white")
    draw = ImageDraw.Draw(recolored)
    role_counts: Counter[str] = Counter()
    trace_recolors: list[dict[str, Any]] = []
    for assignment in _role_recolor_assignments(traces):
        trace = assignment["trace"]
        role = str(assignment["curve_role"])
        rgb = assignment["rgb"]
        points = [tuple(point) for point in trace.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill=rgb, width=3)
        for x, y in points:
            draw.point((int(x), int(y)), fill=rgb)
        role_counts[role] += 1
        trace_recolors.append(
            {
                "recolor_series_id": assignment["recolor_series_id"],
                "source_series_id": assignment["source_series_id"],
                "source_trace_id": assignment["source_trace_id"],
                "legend_label": assignment["legend_label"],
                "curve_role": role,
                "axis_binding_hint": assignment["axis_binding_hint"],
                "rgb": list(rgb),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recolored.save(out_path)
    return {
        "role_trace_counts": dict(role_counts),
        "role_colors_rgb": {role: list(rgb) for role, rgb in ROLE_RECOLOR_RGB.items()},
        "trace_recolors": trace_recolors,
    }


def _role_recolor_candidates(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for assignment in _role_recolor_assignments(traces):
        role = str(assignment["curve_role"])
        axis_hint = "right_y" if role == "power_density_curve" else "left_y"
        label_parts = [
            assignment.get("legend_label"),
            assignment.get("source_series_id"),
            role,
        ]
        candidates.append(
            {
                "series_id": assignment["recolor_series_id"],
                "legend_label": " / ".join(
                    str(part) for part in label_parts if isinstance(part, str) and part.strip()
                ),
                "target_color_rgb": list(assignment["rgb"]),
                "color_distance": 20,
                "forced_curve_role": role,
                "forced_axis_binding_hint": axis_hint,
                "source_series_id": assignment["source_series_id"],
                "source_trace_id": assignment["source_trace_id"],
            }
        )
    return candidates


def _extract_role_recolored_traces_voltage_first(
    crop: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    masks_dir: Path,
    is_dual_y_power_density: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not is_dual_y_power_density:
        return extract_curve_traces(
            crop,
            candidates,
            masks_dir=masks_dir,
            is_dual_y_power_density=False,
            refine_dual_y_roles=False,
        )

    voltage_candidates = [
        candidate for candidate in candidates if candidate.get("forced_curve_role") == "voltage_curve"
    ]
    power_candidates = [
        candidate for candidate in candidates if candidate.get("forced_curve_role") == "power_density_curve"
    ]
    other_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("forced_curve_role") not in {"voltage_curve", "power_density_curve"}
    ]

    traces: list[dict[str, Any]] = []
    issues: list[str] = []
    if voltage_candidates:
        voltage_traces, voltage_issues = extract_curve_traces(
            crop,
            voltage_candidates,
            masks_dir=masks_dir,
            is_dual_y_power_density=True,
            refine_dual_y_roles=False,
        )
        traces.extend(voltage_traces)
        issues.extend(voltage_issues)

    voltage_candidate_by_source = {
        candidate.get("source_series_id"): candidate.get("series_id")
        for candidate in voltage_candidates
        if isinstance(candidate.get("source_series_id"), str) and isinstance(candidate.get("series_id"), str)
    }
    prepared_power_candidates: list[dict[str, Any]] = []
    for candidate in power_candidates:
        copied = dict(candidate)
        voltage_series_id = voltage_candidate_by_source.get(candidate.get("source_series_id"))
        if isinstance(voltage_series_id, str):
            copied["voltage_series_id"] = voltage_series_id
        prepared_power_candidates.append(copied)

    if prepared_power_candidates and traces:
        power_traces, power_issues = extract_power_density_after_voltage_exclusion(
            crop,
            prepared_power_candidates,
            traces,
            masks_dir=masks_dir,
        )
        traces.extend(power_traces)
        issues.extend(power_issues)
    elif prepared_power_candidates:
        issues.append("ROLE_POWER_PASS_SKIPPED:NO_ROLE_VOLTAGE_TRACE")

    if other_candidates:
        other_traces, other_issues = extract_curve_traces(
            crop,
            other_candidates,
            masks_dir=masks_dir,
            is_dual_y_power_density=True,
            refine_dual_y_roles=False,
        )
        traces.extend(other_traces)
        issues.extend(other_issues)

    return traces, issues


def _final_role_traces_for_recolor(final_traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Role-pass images should show every accepted semantic curve, not every exploratory trace."""
    selected: list[dict[str, Any]] = []
    for trace in final_traces:
        role = str(trace.get("curve_role") or "")
        if role not in ROLE_RECOLOR_RGB:
            continue
        selected.append(trace)
    return selected


def _to_panel_traces(traces: list[dict[str, Any]], offset_xy: list[int]) -> list[dict[str, Any]]:
    ox, oy = offset_xy
    out: list[dict[str, Any]] = []
    for trace in traces:
        copied = {k: v for k, v in trace.items() if k != "points"}
        copied["coordinate_space"] = "panel_px"
        copied["source_trace_id"] = trace["trace_id"]
        copied["points"] = [[int(x) + ox, int(y) + oy] for x, y in trace["points"]]
        out.append(copied)
    return out


# ###수정 차이 18: VLM의 axis 대응이 미확정이면 수치 변환을 건너뜁니다.
# ###수정 추가 수정: VLM 판정 이후 좌표 support로 최종 곡선을 선택하며 기존 C-V-P shape 탈락은 적용하지 않습니다.
def _select_cmsl_traces_after_axis_binding(traces: list[dict[str, Any]], *, is_dual_y_power_density: bool) -> tuple[list[dict[str, Any]], list[str]]:
    accepted: list[dict[str, Any]] = []
    issues = []
    for trace in traces:
        trace["final_trace_status"] = "rejected"
        if trace.get("axis_is_curve") is False:
            trace["final_trace_reject_reason"] = "VLM_NOT_A_COHERENT_CURVE"
            continue
        if trace.get("axis_binding_hint") not in {"left_y", "right_y"}:
            trace["final_trace_reject_reason"] = "UNRESOLVED_CURVE_AXIS"
            issues.append(f"UNRESOLVED_CURVE_AXIS:{trace['trace_id']}")
            continue
        if len(trace.get("points", [])) < 20:
            trace["final_trace_reject_reason"] = "TOO_FEW_POINTS"
            continue
        # 정상 curve 조각 보존: 같은 series·y축의 단일 조각 선택 제한 제거
        trace["final_trace_status"] = "accepted"
        trace.pop("final_trace_reject_reason", None)
        accepted.append(trace)
    return accepted, issues
# ###


def _convert_trace_to_data(trace: dict[str, Any], axis_mapping: dict[str, Any]) -> dict[str, Any]:
    x_model = axis_mapping.get("axes", {}).get("bottom", {}).get("model")
    axis_hint = trace.get("axis_binding_hint") or "unresolved"
    y_role = "right" if axis_hint == "right_y" else "left"
    y_model = axis_mapping.get("axes", {}).get(y_role, {}).get("model") if axis_hint in {"left_y", "right_y"} else None
    issues: list[str] = [] if axis_hint in {"left_y", "right_y"} else ["UNRESOLVED_CURVE_AXIS"]
    if not x_model:
        issues.append("MISSING_X_AXIS_MAPPING")
    if not y_model:
        issues.append(f"MISSING_{y_role.upper()}_AXIS_MAPPING")
    points_data: list[list[float]] = []
    if x_model and y_model:
        for x, y in trace.get("points", []):
            points_data.append([_apply_linear(x_model, float(x)), _apply_linear(y_model, float(y))])
    return {
        "trace_id": trace["trace_id"],
        "series_id": trace["series_id"],
        "legend_label": trace.get("legend_label"),
        "curve_role": trace.get("curve_role"),
        "axis_binding": axis_hint,
        "points_data": points_data,
        "source_pixel_trace_ref": trace["trace_id"],
        "series_color_rgb": trace.get("series_color_rgb"),  # 시리즈·조각 구분: 원래 색상 유지
        "conversion_confidence": 0.0 if issues else trace.get("confidence", 0.0),
        "issues": issues,
    }
# ###


# 시리즈·조각 구분: 좌표 유지, 빈 구간 연결 방지
def _group_curve_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (row["series_id"], row["axis_binding"])
        if key not in grouped:
            grouped[key] = {**row, "trace_id": f"{key[0]}_{key[1]}", "segments": [],
                            "points_data": [], "source_pixel_trace_refs": [], "issues": []}
        series = grouped[key]
        series["segments"].append({"trace_id": row["trace_id"], "points_data": row["points_data"],
                                   "source_pixel_trace_ref": row.get("source_pixel_trace_ref"),
                                   "conversion_confidence": row.get("conversion_confidence"),
                                   "issues": row.get("issues", [])})
        series["points_data"].extend(row["points_data"])
        series["source_pixel_trace_refs"].append(row.get("source_pixel_trace_ref"))
        series["issues"].extend(row.get("issues", []))
    for series in grouped.values():
        series.pop("source_pixel_trace_ref", None)
        series["segment_count"] = len(series["segments"])
        series["issues"] = list(dict.fromkeys(series["issues"]))
    return list(grouped.values())


def _dedupe_points_by_x(points: list[list[float]]) -> list[list[float]]:
    by_x: dict[float, list[float]] = defaultdict(list)
    for point in points:
        if len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        if math.isfinite(x) and math.isfinite(y):
            by_x[x].append(y)
    return [[x, sum(ys) / len(ys)] for x, ys in sorted(by_x.items())]


def _interpolate_y(points: list[list[float]], x_value: float) -> float | None:
    ordered = _dedupe_points_by_x(points)
    if len(ordered) < 2:
        return None
    if x_value < ordered[0][0] or x_value > ordered[-1][0]:
        return None
    left_index = 0
    right_index = len(ordered) - 1
    while left_index <= right_index:
        mid = (left_index + right_index) // 2
        if ordered[mid][0] < x_value:
            left_index = mid + 1
        else:
            right_index = mid - 1
    if left_index < len(ordered) and ordered[left_index][0] == x_value:
        return ordered[left_index][1]
    upper = left_index
    lower = upper - 1
    if lower < 0 or upper >= len(ordered):
        return None
    x0, y0 = ordered[lower]
    x1, y1 = ordered[upper]
    if x1 == x0:
        return y0
    ratio = (x_value - x0) / (x1 - x0)
    return y0 + (y1 - y0) * ratio


def _median_absolute(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(abs(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _apply_lvp_power_density_fit(series: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """For L-V-P charts, smooth power density using P(W/cm2)=J(mA/cm2)*V(V)/1000."""
    by_series_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in series:
        series_id = item.get("series_id")
        if isinstance(series_id, str):
            by_series_id[series_id].append(item)

    summaries: list[dict[str, Any]] = []
    for series_id, items in by_series_id.items():
        # 시리즈·조각 구분: 여러 조각 중 임의의 한 조각으로 물리 보정을 적용하지 않습니다.
        if any(sum(item.get("axis_binding") == axis for item in items) > 1 for axis in ("left_y", "right_y")):
            summaries.append({"series_id": series_id, "applied": False, "reason": "SEGMENTED_CURVE_PRESERVE_COORDINATES"})
            continue
        voltage = next((item for item in items if item.get("curve_role") == "voltage_curve"), None)
        power = next((item for item in items if item.get("curve_role") == "power_density_curve"), None)
        if not voltage or not power:
            continue
        voltage_points = voltage.get("points_data")
        power_points = power.get("points_data")
        if not isinstance(voltage_points, list) or not isinstance(power_points, list):
            continue
        voltage_points = [point for point in voltage_points if isinstance(point, list) and len(point) >= 2]
        power_points = [point for point in power_points if isinstance(point, list) and len(point) >= 2]
        if len(voltage_points) < 20 or len(power_points) < 20:
            continue
        voltage_model = _linear_model(
            [{"pixel": float(point[0]), "value": float(point[1])} for point in voltage_points]
        )
        if not voltage_model:
            continue
        fitted: list[list[float]] = []
        residuals: list[float] = []
        for point in _dedupe_points_by_x(power_points):
            x_value = float(point[0])
            measured_power = float(point[1])
            voltage_at_x = max(0.0, _apply_linear(voltage_model, x_value))
            fitted_power = x_value * voltage_at_x / 1000.0
            if not math.isfinite(fitted_power):
                continue
            fitted.append([x_value, fitted_power])
            residuals.append(measured_power - fitted_power)

        if len(fitted) < max(20, len(power_points) * 0.45):
            continue
        median_abs_residual = _median_absolute(residuals)
        max_fitted = max((point[1] for point in fitted), default=0.0)
        if max_fitted <= 0:
            continue
        if median_abs_residual / max_fitted > 0.35:
            continue

        power["points_data"] = fitted
        power["fit"] = {
            "method": "lvp_physics_power_density_from_linear_voltage_fit",
            "formula": "power_density_W_cm2 = current_density_mA_cm2 * voltage_V / 1000",
            "source_voltage_trace_id": voltage.get("trace_id"),
            "source_power_trace_id": power.get("trace_id"),
            "source_voltage_fit": voltage_model,
            "raw_power_point_count": len(power_points),
            "fitted_point_count": len(fitted),
            "median_abs_residual": median_abs_residual,
            "median_abs_residual_over_peak": median_abs_residual / max_fitted,
        }
        summaries.append(
            {
                "series_id": series_id,
                "power_trace_id": power.get("trace_id"),
                "source_voltage_trace_id": voltage.get("trace_id"),
                "method": "lvp_physics_power_density_from_linear_voltage_fit",
                "fitted_point_count": len(fitted),
                "median_abs_residual_over_peak": median_abs_residual / max_fitted,
            }
        )
    return series, summaries


def _overlay_stroke_style(image_size: tuple[int, int]) -> dict[str, int]:
    short_side = max(1, min(image_size))
    # Keep visual QA overlays legible without swamping low-resolution chart crops.
    line_width = max(2, min(6, int(round(short_side * 0.006))))
    marker_radius = max(1, min(4, int(round(line_width * 0.65))))
    return {"line_width": line_width, "marker_radius": marker_radius}


def _draw_overlay(panel_image: Image.Image, traces: list[dict[str, Any]], out_path: Path) -> None:
    overlay = panel_image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    stroke = _overlay_stroke_style(overlay.size)
    line_width = stroke["line_width"]
    marker_radius = stroke["marker_radius"]
    overlay_palette: list[RGB] = [
        (6, 182, 212),
        (34, 197, 94),
        (249, 115, 22),
        (168, 85, 247),
        (250, 204, 21),
        (239, 68, 68),
    ]
    for index, trace in enumerate(traces):
        rgb = overlay_palette[index % len(overlay_palette)]
        points = [tuple(point) for point in trace.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill=rgb, width=line_width, joint="curve")
        for x, y in points[:: max(1, len(points) // 80)]:
            radius = marker_radius
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def _draw_crop_trace_overlay(crop: Image.Image, traces: list[dict[str, Any]], out_path: Path) -> None:
    overlay = crop.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    role_colors = {
        "voltage_curve": (0, 114, 178, 230),
        "power_density_curve": (213, 94, 0, 230),
        "data_curve": (0, 158, 115, 230),
        "unknown_curve": (90, 90, 90, 220),
    }
    line_width = max(2, min(5, int(round(min(crop.size) * 0.008))))
    for trace in traces:
        role = str(trace.get("curve_role") or "unknown_curve")
        color = role_colors.get(role, role_colors["unknown_curve"])
        points = [tuple(point) for point in trace.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill=color, width=line_width, joint="curve")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[Point],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash_px: float,
    gap_px: float,
) -> None:
    if len(points) < 2:
        return
    dash_px = max(1.0, float(dash_px))
    gap_px = max(1.0, float(gap_px))
    period = dash_px + gap_px
    phase = 0.0
    for start, end in zip(points, points[1:]):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        distance = 0.0
        while distance < length:
            cycle_pos = phase % period
            remaining_cycle = period - cycle_pos
            step = min(length - distance, remaining_cycle)
            if cycle_pos < dash_px:
                draw_start = distance
                draw_end = min(distance + step, distance + dash_px - cycle_pos)
                if draw_end > draw_start:
                    t0 = draw_start / length
                    t1 = draw_end / length
                    draw.line(
                        (
                            x1 + dx * t0,
                            y1 + dy * t0,
                            x1 + dx * t1,
                            y1 + dy * t1,
                        ),
                        fill=fill,
                        width=width,
                    )
            distance += step
            phase += step


def _draw_trace_comparison_legend(draw: ImageDraw.ImageDraw) -> None:
    rows = [
        ((100, 116, 139, 170), "raw candidate"),
        ((220, 38, 38, 170), "raw rejected"),
        ((15, 23, 42, 210), "raw accepted before final"),
        ((0, 114, 178, 245), "final voltage"),
        ((213, 94, 0, 245), "final power"),
    ]
    x = 8
    y = 8
    row_h = 15
    width = 172
    height = row_h * len(rows) + 10
    draw.rounded_rectangle((x, y, x + width, y + height), radius=5, fill=(255, 255, 255, 218), outline=(203, 213, 225, 230))
    for index, (color, label) in enumerate(rows):
        yy = y + 9 + index * row_h
        if "raw" in label:
            _draw_dashed_polyline(
                draw,
                [(x + 8, yy + 4), (x + 38, yy + 4)],
                fill=color,
                width=2,
                dash_px=5,
                gap_px=4,
            )
        else:
            draw.line((x + 8, yy + 4, x + 38, yy + 4), fill=color, width=3)
        draw.text((x + 45, yy - 2), label, fill=(15, 23, 42, 245))


def _draw_raw_vs_final_trace_overlay(
    crop: Image.Image,
    raw_traces: list[dict[str, Any]],
    final_traces: list[dict[str, Any]],
    out_path: Path,
) -> None:
    overlay = Image.blend(crop.convert("RGB"), Image.new("RGB", crop.size, "white"), 0.18)
    draw = ImageDraw.Draw(overlay, "RGBA")
    raw_width = max(1, min(4, int(round(min(crop.size) * 0.005))))
    final_width = max(3, min(7, int(round(min(crop.size) * 0.011))))
    marker_radius = max(3, min(7, int(round(min(crop.size) * 0.009))))
    for trace in raw_traces:
        points = [tuple(point) for point in trace.get("points", [])]
        if len(points) >= 2:
            status = str(trace.get("final_trace_status") or "candidate")
            if status == "accepted":
                color = (15, 23, 42, 190)
                dash = 8
                gap = 5
            elif status == "rejected":
                color = (220, 38, 38, 150)
                dash = 5
                gap = 5
            else:
                color = (100, 116, 139, 120)
                dash = 4
                gap = 6
            _draw_dashed_polyline(draw, points, fill=color, width=raw_width, dash_px=dash, gap_px=gap)
    for trace in final_traces:
        role = str(trace.get("curve_role") or "")
        color = (0, 114, 178, 245) if role == "voltage_curve" else (213, 94, 0, 245)
        points = [tuple(point) for point in trace.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill=(255, 255, 255, 210), width=final_width + 3, joint="curve")
            draw.line(points, fill=color, width=final_width, joint="curve")
            for marker_index, (x, y) in enumerate((points[0], points[-1])):
                outline = (15, 23, 42, 230)
                fill = (255, 255, 255, 245) if marker_index == 0 else color
                draw.ellipse(
                    (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
                    fill=fill,
                    outline=outline,
                    width=2,
                )
    _draw_trace_comparison_legend(draw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


# ###수정 차이 08~10·17·18: mask 적용 후 greedy trace를 보존하고 수치 변환 전에 axis를 판정합니다.
def run_data_extraction_for_panel(
    *,
    dataset_root: Path,
    panel_id: str,
    out_root: Path,
    manual_axis_pairs: dict[str, Any] | None = None,
    max_colors: int | None = None,
    exclusion_bboxes_crop_px: list[list[int]] | None = None,
    exclusion_source: str | None = None,
    legend_items: list[dict[str, Any]] | None = None,
    plot_panel_override: dict[str, Any] | None = None,
    axes_panel_override: dict[str, Any] | None = None,
    ticks_panel_override: dict[str, Any] | None = None,
    axis_binding_resolver: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    links = dataset_root / "links"
    gt_plots = _index_by_panel_id(_read_json(links / "gt_plots.json", {}))
    gt_axes = _index_by_panel_id(_read_json(links / "gt_axes.json", {}))
    gt_ticks = _index_by_panel_id(_read_json(links / "gt_axis_ticks.json", {}))

    plot_panel = plot_panel_override or gt_plots.get(panel_id)
    axes_panel = axes_panel_override or gt_axes.get(panel_id)
    ticks_panel = ticks_panel_override or gt_ticks.get(panel_id)
    if not plot_panel:
        raise ValueError(f"panel_id not found in gt_plots.json: {panel_id}")
    if plot_panel_override is None and plot_panel.get("review_status") != "accept":
        raise ValueError(f"panel_id is not an accepted plot: {panel_id}")

    panel_path = dataset_root / str(plot_panel["panel_path"])
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    panel_out = out_root / panel_id
    _reset_panel_out_dir(panel_out, out_root)
    with Image.open(panel_path) as image:
        panel_image = image.convert("RGB")
    plot_bbox = _clamp_bbox(plot_panel.get("gt_bbox_xyxy") or [0, 0, panel_image.width, panel_image.height], panel_image.width, panel_image.height)
    crop = panel_image.crop(tuple(plot_bbox))
    crop_path = panel_out / "plot_crop.png"
    crop.save(crop_path)
    initial_quality = assess_plot_image_quality(crop, [])
    use_sharpened_segmentation = "LOW_IMAGE_SHARPNESS" in initial_quality["issues"]
    segmentation_crop = _sharpen_crop_for_segmentation(crop) if use_sharpened_segmentation else crop
    if use_sharpened_segmentation:
        segmentation_crop.save(panel_out / "sharpened_plot_crop.png")
    normalized_exclusions = _normalize_exclusion_bboxes_crop_px(
        exclusion_bboxes_crop_px,
        crop.width,
        crop.height,
    )

    crop_meta = {
        "schema_version": 1,
        "panel_id": panel_id,
        "source_panel_path": str(plot_panel["panel_path"]),
        "plot_bbox_xyxy_panel_px": plot_bbox,
        "crop_offset_xy": [plot_bbox[0], plot_bbox[1]],
        "crop_size_wh": [plot_bbox[2] - plot_bbox[0], plot_bbox[3] - plot_bbox[1]],
        "coordinate_space": "crop_px",
    }
    _write_json_atomic(panel_out / "crop_meta.json", crop_meta)
    _write_json_atomic(
        panel_out / "exclusion_regions.json",
        {
            "schema_version": 1,
            "panel_id": panel_id,
            "coordinate_space": "crop_px",
            "source": exclusion_source or "none",
            "bboxes": normalized_exclusions,
            "issues": [] if normalized_exclusions else ["NO_EXCLUSION_REGIONS"],
        },
    )

    background_mask, grid_mask, non_curve_mask, preprocessing = _build_non_curve_mask(segmentation_crop)
    background_mask.save(panel_out / "background_mask.png")
    grid_mask.save(panel_out / "grid_mask.png")
    non_curve_mask.save(panel_out / "non_curve_mask.png")
    _write_json_atomic(panel_out / "cmsl_preprocessing.json", preprocessing)
    _write_json_atomic(panel_out / "cmsl_settings.json", {
        "modification_note": "###수정 차이 03·04·06~11·13·15~18 최소 실험",
        "quantization_step": 32, "quantized_representative": "32*(X//32)+16",
        "color_bins": 512, "grid_occupancy": 0.80, "min_pixels": 1,
        "max_colors": max_colors, "rgb_group_distance": 50, "legend_max_distance": 50,
        "max_x_gap_px": 10, "max_y_gap_px": 10, "tracking_score": "euclidean",
        # ###수정 추가 수정: 추가 보완 규칙과 VLM 선행 선택을 effective settings에 기록합니다.
        "additional_revision": "missing_curves_2026-10-06",
        "axis_association_scope": "before_semantic_role_filter",
        "same_color_legend_policy": "retain_options_assign_per_curve",
        "support_max_gap_px": 24, "support_alignment_tolerance_px": 5.0,
        "support_inferred_pixels_recorded": True,
        "minimum_track_width_fraction": 0.02, "final_minimum_point_count": 20,
        # ###
        "preserved_differences": [5, 12, 14, 19],
        "preserved_helpers": ["HSV grouping alternatives", "brightness/saturation gates with black seed exception", "mask strengthening", "short track filtering", "duplicate merging"],
    })

    axis_mapping = build_axis_mapping(
        axes_panel,
        ticks_panel,
        panel_id=panel_id,
        manual_axis_pairs=manual_axis_pairs,
    )
    _write_json_atomic(panel_out / "axis_mapping.json", axis_mapping)

    strategy = build_chart_extraction_strategy(axes_panel, axis_mapping)
    _write_json_atomic(panel_out / "chart_extraction_strategy.json", strategy)

    series_candidates = detect_color_candidates(
        segmentation_crop,
        max_colors=max_colors,
        exclusion_bboxes_crop_px=normalized_exclusions,
        legend_items=legend_items,
        non_curve_mask=non_curve_mask,
    )
    if not series_candidates:
        series_candidates = []
    legend_matching = match_legend_items_to_series_candidates(series_candidates, legend_items)
    series_issues = ["ZERO_COLOR_CANDIDATES"] if not series_candidates else []
    series_issues.extend(legend_matching["issues"])
    if legend_matching["legend_item_count"] and normalized_exclusions:
        series_source = "color_quantization_with_legend_ocr_and_exclusion_regions"
    elif legend_matching["legend_item_count"]:
        series_source = "color_quantization_with_legend_ocr"
    elif normalized_exclusions:
        series_source = "color_quantization_with_exclusion_regions"
        series_issues.append("LEGEND_ITEMS_NOT_AVAILABLE")
    else:
        series_source = "color_quantization_no_legend_ocr"
        series_issues.append("LEGEND_OCR_NOT_RUN")
    _write_json_atomic(
        panel_out / "series_candidates.json",
        {
            "schema_version": 1,
            "panel_id": panel_id,
            "source": series_source,
            "items": series_candidates,
            "legend_matching": legend_matching,
            "issues": series_issues,
        },
    )
    image_quality = assess_plot_image_quality(crop, series_candidates)
    image_quality["preprocessing"] = {
        "segmentation_source": "sharpened_plot_crop.png" if use_sharpened_segmentation else "plot_crop.png",
        "sharpened_for_segmentation": use_sharpened_segmentation,
        "initial_sharpness_issues": initial_quality["issues"],
    }
    _write_json_atomic(panel_out / "image_quality_report.json", image_quality)
    color_filter_source = build_solid_color_filter_source(
        segmentation_crop,
        series_candidates,
        exclusion_bboxes_crop_px=normalized_exclusions,
        non_curve_mask=non_curve_mask,
    )
    color_filter_source_path = panel_out / "color_filter_source.png"
    color_filter_source.save(color_filter_source_path)

    # ###수정 추가 수정: 보간 support는 grid와 명시적인 범례/텍스트 제외 영역을 넘지 않습니다.
    tracking_forbidden_mask = grid_mask.copy()
    forbidden_draw = ImageDraw.Draw(tracking_forbidden_mask)
    for x1, y1, x2, y2 in normalized_exclusions:
        forbidden_draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
    # ###
    traces_crop, trace_issues = extract_curve_traces(
        color_filter_source,
        series_candidates,
        masks_dir=panel_out / "masks",
        is_dual_y_power_density=strategy["chart_family"] == "dual_y_power_density",
        exclusion_bboxes_crop_px=normalized_exclusions,
        non_curve_mask=non_curve_mask,
        # ###수정 추가 수정: 관측 background 제외와 보간 금지 영역을 구분합니다.
        tracking_forbidden_mask=tracking_forbidden_mask,
        refine_dual_y_roles=False,
        # ###
    )
    # ###수정 추가 수정: 길이 조건만 확인한 모든 후보의 축·범례를 VLM에 먼저 대응시킵니다.
    support_reports = []
    for candidate in series_candidates:
        support_path = panel_out / "tracking_support" / f"{candidate['series_id']}.json"
        if support_path.exists():
            support_reports.append({"series_id": candidate["series_id"], **_read_json(support_path, {})})
    _write_json_atomic(panel_out / "tracking_support.json", {"schema_version": 1, "items": support_reports,
                       "modification_note": "###수정 추가 수정: 관측/보간 support를 구분하고 10/10 tracker 제한을 유지합니다."})
    geometric_candidates = [trace for trace in traces_crop if len(trace.get("points", [])) >= 20]
    is_dual_y = len(axis_mapping.get("y_axes", [])) == 2
    if is_dual_y:
        if axis_binding_resolver is None:
            raise ValueError("Dual-axis CMSL extraction requires a VLM axis-binding resolver")
        binding_report = axis_binding_resolver(
            panel_image=panel_image,
            traces=_to_panel_traces(geometric_candidates, crop_meta["crop_offset_xy"]),
            axis_mapping=axis_mapping,
            out_dir=panel_out,
        )
    else:
        binding_report = {
            "schema_version": 1, "source": "single_y_axis",
            "assignments": [{"trace_id": trace["trace_id"], "axis_binding": "left_y", "confidence": 1.0,
                             "reason": "Only one y axis", "is_curve": True} for trace in geometric_candidates],
            "issues": [],
        }
    binding_report["input_trace_scope"] = "geometric_candidates_before_role_filter"
    assignments = {item["trace_id"]: item for item in binding_report.get("assignments", [])}
    for trace in traces_crop:
        assignment = assignments.get(trace["trace_id"], {})
        trace["initial_curve_role"] = trace.get("curve_role")
        trace["axis_binding_hint"] = assignment.get("axis_binding") or "unresolved"
        trace["axis_is_curve"] = assignment.get("is_curve", True)
        trace["axis_assignment_confidence"] = assignment.get("confidence", 0.0)
        trace["axis_assignment_reason"] = assignment.get("reason", "No assignment")
        if trace["axis_binding_hint"] in {"left_y", "right_y"}:
            trace["curve_role"] = ("voltage_curve" if trace["axis_binding_hint"] == "left_y" else "power_density_curve") if strategy["chart_family"] == "dual_y_power_density" else "data_curve"
        options = trace.get("legend_options", [])
        labels = {option.get("legend_label") for option in options if option.get("legend_label")}
        label = assignment.get("legend_label")
        if label in labels:
            trace["legend_label"] = label
        elif options:
            matching = [option for option in options if option.get("axis_binding_hint") in {None, trace["axis_binding_hint"]}]
            matching_labels = {option.get("legend_label") for option in matching if option.get("legend_label")}
            trace["legend_label"] = next(iter(matching_labels)) if len(matching_labels) == 1 else None
    accepted_final_traces_crop, final_trace_issues = _select_cmsl_traces_after_axis_binding(
        traces_crop, is_dual_y_power_density=strategy["chart_family"] == "dual_y_power_density")
    final_trace_issues.extend(binding_report.get("issues", []))
    _write_json_atomic(panel_out / "curve_axis_association.json", binding_report)
    _write_json_atomic(panel_out / "curve_traces_crop_px.json", {
        "schema_version": 1, "panel_id": panel_id, "coordinate_space": "crop_px",
        "source_crop_ref": "plot_crop.png", "traces": traces_crop, "issues": trace_issues,
    })
    _write_json_atomic(panel_out / "final_curve_traces_crop_px.json", {
        "schema_version": 1, "panel_id": panel_id, "coordinate_space": "crop_px",
        "source_trace_ref": "curve_traces_crop_px.json", "source": "axis_bound_greedy_tracks_with_recorded_gap_support",
        "traces": accepted_final_traces_crop, "issues": final_trace_issues,
    })
    _draw_crop_trace_overlay(crop, accepted_final_traces_crop, panel_out / "final_trace_overlay.png")
    _draw_raw_vs_final_trace_overlay(crop, traces_crop, accepted_final_traces_crop, panel_out / "raw_vs_final_trace_overlay.png")
    role_recolor_summary = _draw_role_recolored_crop(crop, _final_role_traces_for_recolor(accepted_final_traces_crop), panel_out / "role_recolored_crop.png")
    role_second_pass_traces: list[dict[str, Any]] = []
    role_second_pass_issues: list[str] = []
    _write_json_atomic(panel_out / "role_recolored_curve_traces_crop_px.json", {
        "schema_version": 1, "panel_id": panel_id, "coordinate_space": "crop_px",
        "source": "axis_bound_curve_recolor_no_second_pass", "enabled": False,
        "role_recolor_summary": role_recolor_summary, "traces": [], "issues": [],
    })
    traces_panel = _to_panel_traces(traces_crop, crop_meta["crop_offset_xy"])
    final_traces_panel = _to_panel_traces(accepted_final_traces_crop, crop_meta["crop_offset_xy"])
    _write_json_atomic(panel_out / "curve_traces_panel_px.json", {
        "schema_version": 1, "panel_id": panel_id, "coordinate_space": "panel_px",
        "source_crop_bbox_xyxy_panel_px": plot_bbox, "traces": traces_panel, "issues": trace_issues,
    })
    # ###

    converted_series = [_convert_trace_to_data(trace, axis_mapping) for trace in final_traces_panel]
    data_postprocess: list[dict[str, Any]] = []
    if strategy["chart_family"] == "dual_y_power_density":
        converted_series, data_postprocess = _apply_lvp_power_density_fit(converted_series)
    # 시리즈·조각 구분: 좌표 유지, 빈 구간 연결 방지
    converted_series = _group_curve_segments(converted_series)
    data_series = {
        "schema_version": 1,
        "panel_id": panel_id,
        "source_panel_path": str(plot_panel["panel_path"]),
        "plot_bbox_xyxy_panel_px": plot_bbox,
        "axis_mapping_ref": "axis_mapping.json",
        "series": converted_series,
        "postprocess": data_postprocess,
        "issues": list(axis_mapping.get("issues", []))
        + trace_issues
        + final_trace_issues
        + [f"IMAGE_QUALITY:{issue}" for issue in image_quality.get("issues", [])],
    }
    _write_json_atomic(panel_out / "curve_series_data.json", data_series)
    _draw_overlay(panel_image, final_traces_panel, panel_out / "overlay.png")

    validation = {
        "schema_version": 1,
        "panel_id": panel_id,
        "checks": {
            "plot_crop_exists": crop_path.exists(),
            "color_filter_source_exists": color_filter_source_path.exists(),
            "series_candidate_count": len(series_candidates),
            "curve_trace_count": len(traces_crop),
            "x_axis_mapping_ready": bool(axis_mapping["axes"]["bottom"]["model"]),
            "left_y_axis_mapping_ready": bool(axis_mapping["axes"]["left"]["model"]),
            "right_y_axis_mapping_ready": bool(axis_mapping["axes"]["right"]["model"]),
            "exclusion_region_count": len(normalized_exclusions),
            "legend_item_count": legend_matching["legend_item_count"],
            "legend_matched_series_count": legend_matching["matched_series_count"],
            "role_recolored_trace_count": len(role_second_pass_traces),
            "final_series_count": len(converted_series),
            "segment_count": len(final_traces_panel),
            "image_laplacian_variance": round(float(image_quality["sharpness"]["laplacian_variance"]), 3),
            "min_palette_distance": (
                round(float(image_quality["color_separation"]["min_palette_distance"]), 3)
                if isinstance(image_quality["color_separation"].get("min_palette_distance"), (int, float))
                else None
            ),
            "sharpened_for_segmentation": use_sharpened_segmentation,
        },
        "issues": data_series["issues"] + strategy.get("issues", []) + role_second_pass_issues,
    }
    _write_json_atomic(panel_out / "validation_report.json", validation)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "panel_id": panel_id,
        "artifacts": {
            "crop_meta": "crop_meta.json",
            "plot_crop": "plot_crop.png",
            "exclusion_regions": "exclusion_regions.json",
            "axis_mapping": "axis_mapping.json",
            "chart_extraction_strategy": "chart_extraction_strategy.json",
            "color_filter_source": "color_filter_source.png",
            "series_candidates": "series_candidates.json",
            "image_quality_report": "image_quality_report.json",
            "sharpened_plot_crop": "sharpened_plot_crop.png" if use_sharpened_segmentation else None,
            "curve_traces_crop_px": "curve_traces_crop_px.json",
            "final_curve_traces_crop_px": "final_curve_traces_crop_px.json",
            "final_trace_overlay": "final_trace_overlay.png",
            "raw_vs_final_trace_overlay": "raw_vs_final_trace_overlay.png",
            "role_recolored_crop": "role_recolored_crop.png",
            "role_recolored_curve_traces_crop_px": "role_recolored_curve_traces_crop_px.json",
            "curve_traces_panel_px": "curve_traces_panel_px.json",
            "curve_series_data": "curve_series_data.json",
            "validation_report": "validation_report.json",
            "overlay": "overlay.png",
        },
    }
    # ###수정 추가 수정: support 보간 증거를 GUI debug artifact 목록에 연결합니다.
    manifest["artifacts"]["tracking_support"] = "tracking_support.json"
    for candidate in series_candidates:
        series_id = candidate["series_id"]
        manifest["artifacts"][f"tracking_support_{series_id}"] = f"tracking_support/{series_id}.png"
        manifest["artifacts"][f"inferred_support_{series_id}"] = f"tracking_support/{series_id}_inferred.png"
    # ###
    _write_json_atomic(panel_out / "run_manifest.json", manifest)
    return manifest
# ###

from __future__ import annotations

import json
from pathlib import Path
import pytest

from PIL import Image, ImageDraw

from core.data_extraction_pipeline import (
    _draw_role_recolored_crop,
    _apply_lvp_power_density_fit,
    _extract_role_recolored_traces_voltage_first,
    _merge_voltage_fragment_traces,
    _mask_for_color,
    _overlay_stroke_style,
    _role_recolor_candidates,
    _monotonic_voltage_centerline_from_mask,
    _smooth_final_curve_points,
    _trim_endpoint_sparse_islands,
    assess_plot_image_quality,
    build_solid_color_filter_source,
    classify_curve_role,
    detect_color_candidates,
    extract_power_density_after_voltage_exclusion,
    extract_voltage_after_power_density_exclusion,
    extract_curve_traces,
    match_legend_items_to_series_candidates,
    run_data_extraction_for_panel,
    select_final_traces_for_data_series,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_classify_curve_role_separates_voltage_and_power_shapes() -> None:
    voltage_points = [(x, 20 + x // 2) for x in range(0, 100, 5)]
    power_points = [(x, int(72 - 42 * (1 - abs(x - 50) / 50))) for x in range(0, 100, 5)]

    voltage = classify_curve_role(voltage_points, is_dual_y_power_density=True)
    power = classify_curve_role(power_points, is_dual_y_power_density=True)

    assert voltage["curve_role"] == "voltage_curve"
    assert voltage["axis_binding_hint"] == "left_y"
    assert power["curve_role"] == "power_density_curve"
    assert power["axis_binding_hint"] == "right_y"


def test_voltage_centerline_uses_curved_spline_branch() -> None:
    mask = Image.new("L", (190, 130), 0)
    draw = ImageDraw.Draw(mask)

    def voltage_y(x: int) -> int:
        return int(round(18 + 0.34 * x + 0.0016 * (x - 90) ** 2))

    voltage_points = [(x, voltage_y(x)) for x in range(10, 176)]
    draw.line(voltage_points, fill=255, width=5)
    # A nearby non-voltage branch should not force the voltage fit back to a straight chord.
    draw.line([(20, 112), (70, 64), (105, 38), (145, 70), (175, 118)], fill=255, width=2)

    centerline = _monotonic_voltage_centerline_from_mask(mask)

    assert len(centerline) > 120
    by_x = {x: y for x, y in centerline}
    xs = sorted(by_x)
    x0, x1 = xs[0], xs[-1]
    y0, y1 = by_x[x0], by_x[x1]

    def chord_y(x: int) -> float:
        return y0 + (y1 - y0) * ((x - x0) / max(1, x1 - x0))

    sampled = [x for x in range(max(20, x0), min(170, x1), 10) if x in by_x]
    spline_error = sum(abs(by_x[x] - voltage_y(x)) for x in sampled) / len(sampled)
    chord_error = sum(abs(chord_y(x) - voltage_y(x)) for x in sampled) / len(sampled)

    assert spline_error < 5.0
    assert spline_error < chord_error * 0.7


def test_voltage_centerline_keeps_thin_left_endpoint_before_power_crossing() -> None:
    mask = Image.new("L", (280, 220), 0)
    draw = ImageDraw.Draw(mask)
    draw.line([(0, 14), (55, 72), (115, 128), (180, 178), (240, 207)], fill=255, width=2)
    draw.line([(5, 198), (38, 106), (75, 78), (115, 96), (158, 150), (210, 207)], fill=255, width=5)
    draw.line([(2, 170), (7, 150), (11, 185)], fill=255, width=1)

    centerline = _monotonic_voltage_centerline_from_mask(mask)

    by_x = {x: y for x, y in centerline}
    assert min(by_x) <= 3
    assert max(by_x) >= 235
    assert by_x[10] < 35
    assert by_x[60] < by_x[120] < by_x[200]


def test_final_power_trace_smoothing_reduces_zigzag() -> None:
    points = []
    for x in range(0, 121):
        base_y = 95 - 55 * (1 - abs(x - 60) / 60)
        jitter = 5 if x % 6 in {0, 1} else -4 if x % 6 in {3, 4} else 0
        points.append([x, int(round(base_y + jitter))])

    smoothed, meta = _smooth_final_curve_points(points, curve_role="power_density_curve")

    def sign_changes(values: list[int]) -> int:
        signs: list[int] = []
        for left, right in zip(values, values[1:]):
            diff = right - left
            if diff:
                signs.append(1 if diff > 0 else -1)
        return sum(1 for left, right in zip(signs, signs[1:]) if left != right)

    raw_changes = sign_changes([point[1] for point in points])
    smooth_changes = sign_changes([point[1] for point in smoothed])

    def longest_flat_run(values: list[int]) -> int:
        longest = 1
        current = 1
        for left, right in zip(values, values[1:]):
            if left == right:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    assert meta["method"] == "local_moving_average_shape_preserving"
    assert len(smoothed) == len(points)
    assert smooth_changes < raw_changes
    assert longest_flat_run([point[1] for point in smoothed]) <= 3


def test_final_voltage_selection_prefers_monotonic_branch_over_fragment_merge() -> None:
    monotonic_trace = {
        "trace_id": "color_01_voltage_monotonic_trace",
        "series_id": "color_01",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.9,
        "extraction_method": "same_color_filled_marker_monotonic_voltage_centerline",
        "points": [[x, 20 + x] for x in range(100)],
        "features": {"data_y_decreasing_score_px": 90.0},
    }
    fragment_merge_trace = {
        "trace_id": "color_01_voltage_fragment_merged_trace",
        "series_id": "color_01",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.9,
        "extraction_method": "same_color_voltage_fragment_merge_centerline",
        "points": [[x, 18 + x] for x in range(160)],
        "features": {"data_y_decreasing_score_px": 95.0},
    }
    power_trace = {
        "trace_id": "color_01_power_trace",
        "series_id": "color_01",
        "curve_role": "power_density_curve",
        "axis_binding_hint": "right_y",
        "confidence": 0.8,
        "extraction_method": "same_color_after_voltage_exclusion_power_density_centerline",
        "points": [[x, 90 - abs(x - 60)] for x in range(120)],
        "features": {"rise_peak_fall_score_px": 30.0},
    }

    selected, issues = select_final_traces_for_data_series(
        [fragment_merge_trace, monotonic_trace, power_trace],
        is_dual_y_power_density=True,
    )

    assert not issues
    voltage = next(trace for trace in selected if trace["curve_role"] == "voltage_curve")
    assert voltage["trace_id"] == "color_01_voltage_monotonic_trace"


def test_lvp_power_density_fit_uses_matching_voltage_curve() -> None:
    voltage_points = [[float(x), 1.0 - x * 0.0002] for x in range(0, 2100, 100)]
    noisy_power_points = [
        [float(x), (x * (1.0 - x * 0.0002) / 1000.0) + (0.03 if x % 300 == 0 else -0.02)]
        for x in range(0, 2100, 100)
    ]
    series = [
        {
            "trace_id": "voltage_trace",
            "series_id": "color_02",
            "curve_role": "voltage_curve",
            "points_data": voltage_points,
        },
        {
            "trace_id": "power_trace",
            "series_id": "color_02",
            "curve_role": "power_density_curve",
            "points_data": noisy_power_points,
        },
    ]

    fitted, summary = _apply_lvp_power_density_fit(series)

    power = next(item for item in fitted if item["curve_role"] == "power_density_curve")
    assert summary
    assert power["fit"]["method"] == "lvp_physics_power_density_from_linear_voltage_fit"
    assert power["points_data"][0] == [0.0, 0.0]
    assert power["points_data"][10] == pytest.approx([1000.0, 0.8])
    assert power["points_data"][-1] == pytest.approx([2000.0, 1.2])


def test_color_candidates_ignore_exclusion_regions() -> None:
    crop = Image.new("RGB", (80, 50), "white")
    draw = ImageDraw.Draw(crop)
    draw.rectangle([50, 5, 72, 18], fill=(240, 40, 40))

    without_exclusion = detect_color_candidates(crop, max_colors=4, min_pixels=20)
    with_exclusion = detect_color_candidates(
        crop,
        max_colors=4,
        min_pixels=20,
        exclusion_bboxes_crop_px=[[48, 3, 74, 20]],
    )

    assert without_exclusion
    assert with_exclusion == []


def test_color_candidates_group_similar_hue_variants_into_one_filter() -> None:
    crop = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(crop)
    draw.rectangle([5, 8, 30, 30], fill=(255, 0, 0))
    draw.rectangle([34, 8, 58, 30], fill=(255, 48, 48))
    draw.rectangle([62, 8, 86, 30], fill=(240, 120, 120))
    draw.rectangle([90, 8, 112, 30], fill=(40, 90, 240))

    candidates = detect_color_candidates(crop, max_colors=4, min_pixels=20)

    assert len(candidates) == 2
    red_candidate = candidates[0]
    assert red_candidate["series_id"] == "color_01"
    assert red_candidate["color_family_member_count"] >= 3
    assert len(red_candidate["target_colors_rgb"]) >= 3
    assert candidates[1]["series_id"] == "color_02"


def test_legend_seeded_color_candidates_keep_close_legend_colors_separate() -> None:
    crop = Image.new("RGB", (140, 70), "white")
    draw = ImageDraw.Draw(crop)
    draw.rectangle([5, 8, 30, 30], fill=(255, 0, 0))
    draw.rectangle([34, 8, 58, 30], fill=(255, 48, 48))
    draw.rectangle([75, 8, 100, 30], fill=(220, 70, 70))
    draw.rectangle([104, 8, 128, 30], fill=(210, 95, 95))

    candidates = detect_color_candidates(
        crop,
        max_colors=4,
        min_pixels=20,
        legend_items=[
            {"legend_label": "520", "swatch_color_rgb": [255, 0, 0]},
            {"legend_label": "600", "swatch_color_rgb": [220, 70, 70]},
        ],
    )

    assert len(candidates) == 2
    assert [candidate["legend_label"] for candidate in candidates] == ["520", "600"]
    assert candidates[0]["color_family_method"] == "legend_seed_nearest_color"
    assert candidates[1]["color_family_method"] == "legend_seed_nearest_color"
    assert candidates[0]["legend_seed_rgb"] == [255, 0, 0]
    assert candidates[1]["legend_seed_rgb"] == [220, 70, 70]


def test_gray_legend_seed_does_not_absorb_pastel_antialias_colors() -> None:
    crop = Image.new("RGB", (160, 70), "white")
    draw = ImageDraw.Draw(crop)
    draw.rectangle([5, 8, 34, 32], fill=(192, 192, 192))
    draw.rectangle([40, 8, 68, 32], fill=(168, 168, 168))
    draw.rectangle([85, 8, 112, 32], fill=(240, 168, 168))
    draw.rectangle([118, 8, 146, 32], fill=(168, 168, 255))

    candidates = detect_color_candidates(
        crop,
        max_colors=4,
        min_pixels=20,
        legend_items=[
            {"legend_label": "gray", "swatch_color_rgb": [200, 200, 200]},
        ],
    )

    assert len(candidates) == 1
    assert candidates[0]["legend_label"] == "gray"
    palette = candidates[0]["target_colors_rgb"]
    assert [192, 192, 192] in palette
    assert [168, 168, 168] in palette
    assert [240, 168, 168] not in palette
    assert [168, 168, 255] not in palette


def test_solid_color_filter_source_keeps_gray_separate_from_colored_curves() -> None:
    crop = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(crop)
    draw.line([(5, 20), (110, 20)], fill=(176, 176, 176), width=2)
    draw.line([(5, 38), (110, 38)], fill=(240, 96, 96), width=2)
    draw.point([(30, 37), (31, 39), (80, 37), (81, 39)], fill=(210, 176, 176))

    candidates = [
        {
            "series_id": "color_01",
            "target_color_rgb": [176, 176, 176],
            "target_colors_rgb": [[176, 176, 176]],
            "color_distance": 70,
        },
        {
            "series_id": "color_02",
            "target_color_rgb": [240, 96, 96],
            "target_colors_rgb": [[240, 96, 96], [210, 176, 176]],
            "color_distance": 70,
        },
    ]

    solid = build_solid_color_filter_source(crop, candidates)
    pixels = solid.load()

    assert pixels[30, 20] == (176, 176, 176)
    assert pixels[30, 38] == (240, 96, 96)
    assert pixels[30, 37] == (240, 96, 96)


def test_solid_color_filter_source_assigns_overlapping_red_orange_to_nearest_legend_seed() -> None:
    crop = Image.new("RGB", (140, 80), "white")
    draw = ImageDraw.Draw(crop)
    draw.line([(10, 22), (125, 22)], fill=(216, 72, 96), width=2)
    draw.line([(10, 46), (125, 46)], fill=(240, 144, 96), width=2)
    # This red-ish antialias color appears in the orange palette on compressed charts.
    draw.point([(60, 22), (61, 22)], fill=(240, 144, 144))

    candidates = [
        {
            "series_id": "color_01",
            "legend_label": "650C",
            "legend_seed_rgb": [228, 72, 93],
            "target_color_rgb": [216, 72, 96],
            "target_colors_rgb": [[216, 72, 96], [240, 144, 144]],
            "color_distance": 45,
        },
        {
            "series_id": "color_02",
            "legend_label": "600C",
            "legend_seed_rgb": [255, 154, 59],
            "target_color_rgb": [240, 144, 96],
            "target_colors_rgb": [[240, 144, 96], [240, 144, 144]],
            "color_distance": 45,
        },
    ]

    solid = build_solid_color_filter_source(crop, candidates)
    pixels = solid.load()

    assert pixels[60, 22] == (216, 72, 96)
    assert pixels[60, 46] == (240, 144, 96)


def test_image_quality_flags_low_color_separation() -> None:
    crop = Image.new("RGB", (80, 60), "white")
    candidates = [
        {
            "series_id": "color_01",
            "target_color_rgb": [240, 72, 72],
            "target_colors_rgb": [[240, 72, 72]],
        },
        {
            "series_id": "color_02",
            "target_color_rgb": [240, 72, 48],
            "target_colors_rgb": [[240, 72, 48]],
        },
    ]

    quality = assess_plot_image_quality(crop, candidates)

    assert quality["color_separation"]["min_palette_distance"] == pytest.approx(24.0)
    assert "LOW_COLOR_SEPARATION_BETWEEN_SERIES" in quality["issues"]
    assert "AMBIGUOUS_ANTIALIAS_COLOR_PAIRS" in quality["issues"]


def test_overlay_stroke_scales_down_for_low_resolution_images() -> None:
    small = _overlay_stroke_style((260, 180))
    medium = _overlay_stroke_style((640, 420))
    large = _overlay_stroke_style((1600, 1000))

    assert small == {"line_width": 2, "marker_radius": 1}
    assert medium["line_width"] >= small["line_width"]
    assert large["line_width"] <= 6
    assert large["line_width"] > small["line_width"]


def test_colored_legend_seed_does_not_absorb_other_series_pastel_colors() -> None:
    crop = Image.new("RGB", (180, 80), "white")
    draw = ImageDraw.Draw(crop)
    draw.rectangle([5, 8, 35, 32], fill=(240, 72, 96))
    draw.rectangle([42, 8, 72, 32], fill=(240, 168, 192))
    draw.rectangle([88, 8, 118, 32], fill=(168, 120, 240))
    draw.rectangle([125, 8, 155, 32], fill=(192, 168, 240))

    candidates = detect_color_candidates(
        crop,
        max_colors=4,
        min_pixels=20,
        legend_items=[
            {"legend_label": "PLNC-GDC", "swatch_color_rgb": [244, 78, 111]},
            {"legend_label": "PLNC", "swatch_color_rgb": [155, 124, 231]},
        ],
    )

    assert len(candidates) == 2
    by_label = {candidate["legend_label"]: candidate for candidate in candidates}
    pink_palette = by_label["PLNC-GDC"]["target_colors_rgb"]
    purple_palette = by_label["PLNC"]["target_colors_rgb"]
    assert [240, 168, 192] in pink_palette
    assert [240, 168, 192] not in purple_palette
    assert [192, 168, 240] in purple_palette
    assert [192, 168, 240] not in pink_palette


def test_legend_items_match_color_series_ids() -> None:
    series_candidates = [
        {
            "series_id": "color_01",
            "legend_label": None,
            "target_color_rgb": [240, 48, 48],
        },
        {
            "series_id": "color_02",
            "legend_label": None,
            "target_color_rgb": [48, 96, 240],
        },
    ]

    matching = match_legend_items_to_series_candidates(
        series_candidates,
        [
            {
                "legend_label": "700 C",
                "swatch_color_rgb": [238, 42, 44],
                "marker_shape": "circle",
                "confidence": 0.92,
            },
            {
                "legend_label": "800 C",
                "swatch_color_rgb": [42, 90, 232],
                "marker_shape": "square",
                "confidence": 0.9,
            },
        ],
    )

    assert matching["matched_series_count"] == 2
    assert series_candidates[0]["series_id"] == "color_01"
    assert series_candidates[0]["legend_label"] == "700 C"
    assert series_candidates[0]["legend_match"]["source"] == "codex_legend_vision"
    assert series_candidates[1]["series_id"] == "color_02"
    assert series_candidates[1]["legend_label"] == "800 C"


def test_dual_y_same_color_traces_are_refined_to_left_and_right_roles(tmp_path: Path) -> None:
    crop = Image.new("RGB", (140, 100), "white")
    draw = ImageDraw.Draw(crop)
    color = (240, 40, 40)
    draw.line([(8, 20), (45, 35), (85, 55), (125, 74)], fill=color, width=2)
    draw.line([(8, 82), (45, 44), (70, 28), (95, 45), (125, 82)], fill=color, width=2)

    traces, issues = extract_curve_traces(
        crop,
        [
            {
                "series_id": "color_01",
                "legend_label": "700 C",
                "target_color_rgb": list(color),
                "color_distance": 35,
            }
        ],
        masks_dir=tmp_path / "masks",
        is_dual_y_power_density=True,
    )

    assert not issues
    roles = {(trace["curve_role"], trace["axis_binding_hint"]) for trace in traces}
    assert ("voltage_curve", "left_y") in roles
    assert ("power_density_curve", "right_y") in roles


def test_trace_points_use_thick_stroke_centerline(tmp_path: Path) -> None:
    crop = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(crop)
    color = (240, 40, 40)
    for jitter in (-2, 0, 2):
        draw.line(
            [(8, 20 + jitter), (35, 32 + jitter), (70, 42 + jitter), (110, 55 + jitter)],
            fill=color,
            width=1,
        )

    traces, issues = extract_curve_traces(
        crop,
        [
            {
                "series_id": "color_01",
                "legend_label": "centerline",
                "target_color_rgb": list(color),
                "color_distance": 35,
            }
        ],
        masks_dir=tmp_path / "masks",
        is_dual_y_power_density=False,
    )

    assert not issues
    assert len(traces) == 1
    trace = traces[0]
    xs = [point[0] for point in trace["points"]]
    assert len(xs) == len(set(xs))
    assert trace["extraction_method"] == "color_mask_column_tracking_thick_stroke_centerline"
    assert trace["features"]["centerline_method"] == "smooth_polyline_thick_stroke_column_midpoint"
    assert trace["features"]["centerline_point_count"] <= trace["features"]["raw_track_point_count"]


def test_duplicate_upper_lower_edge_tracks_are_merged_before_data_points(tmp_path: Path) -> None:
    crop = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(crop)
    color = (240, 40, 40)
    upper = [(8, 28), (35, 35), (70, 42), (110, 50)]
    lower = [(8, 36), (35, 43), (70, 50), (110, 58)]
    draw.line(upper, fill=color, width=1)
    draw.line(lower, fill=color, width=1)

    traces, issues = extract_curve_traces(
        crop,
        [
            {
                "series_id": "color_01",
                "legend_label": "hollow-edge",
                "target_color_rgb": list(color),
                "color_distance": 35,
            }
        ],
        masks_dir=tmp_path / "masks",
        is_dual_y_power_density=False,
    )

    assert not issues
    assert len(traces) == 1
    trace = traces[0]
    assert trace["features"]["raw_track_count_before_edge_merge"] >= 2
    assert trace["features"]["track_count_after_edge_merge"] == 1
    y_by_x = {x: y for x, y in trace["points"]}
    assert 30 <= y_by_x[8] <= 34
    assert 52 <= y_by_x[110] <= 56


def test_dual_y_final_series_keeps_best_right_axis_curve_despite_open_marker_fragments() -> None:
    good_voltage = {
        "trace_id": "color_01_trace_01",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.9,
        "points": [[x, 20 + x] for x in range(50)],
        "features": {
            "raw_track_count_before_edge_merge": 1,
            "track_count_after_edge_merge": 1,
            "data_y_decreasing_score_px": 50,
        },
    }
    fragmented_power = {
        "trace_id": "color_02_trace_01",
        "curve_role": "power_density_curve",
        "axis_binding_hint": "right_y",
        "confidence": 0.95,
        "points": [[x, 70 - abs(x - 25)] for x in range(50)],
        "features": {
            "raw_track_count_before_edge_merge": 5,
            "track_count_after_edge_merge": 2,
            "rise_peak_fall_score_px": 40,
        },
    }
    duplicate_voltage = {
        "trace_id": "color_03_trace_01",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.8,
        "points": [[x, 25 + x] for x in range(50)],
        "features": {
            "raw_track_count_before_edge_merge": 4,
            "track_count_after_edge_merge": 3,
            "data_y_decreasing_score_px": 45,
        },
    }

    selected, issues = select_final_traces_for_data_series(
        [good_voltage, fragmented_power, duplicate_voltage],
        is_dual_y_power_density=True,
    )

    assert [trace["trace_id"] for trace in selected] == ["color_01_trace_01", "color_02_trace_01"]
    assert good_voltage["final_trace_status"] == "accepted"
    assert fragmented_power["final_trace_status"] == "accepted"
    assert duplicate_voltage["final_trace_status"] == "rejected"
    assert issues == []


def test_dual_y_final_series_scores_power_by_voltage_physics_fit() -> None:
    axis_mapping = {
        "axes": {
            "bottom": {"model": {"slope": 10.0, "intercept": 0.0}},
            "left": {"model": {"slope": -0.01, "intercept": 1.2}},
            "right": {"model": {"slope": -0.01, "intercept": 1.2}},
        }
    }
    voltage_points = []
    good_power_points = []
    bad_power_points = []
    for x in range(0, 101):
        current = x * 10.0
        voltage = 1.0 - current * 0.0005
        expected_power = current * voltage / 1000.0
        voltage_points.append([x, int(round(120 - voltage * 100))])
        good_power_points.append([x, int(round(120 - expected_power * 100))])
        shifted_power = expected_power * 0.55 + 0.22
        bad_power_points.append([x, int(round(120 - shifted_power * 100))])

    voltage = {
        "trace_id": "voltage",
        "series_id": "color_01",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.92,
        "points": voltage_points,
        "features": {"data_y_decreasing_score_px": 50},
    }
    good_power = {
        "trace_id": "good_power",
        "series_id": "color_01",
        "curve_role": "power_density_curve",
        "axis_binding_hint": "right_y",
        "confidence": 0.65,
        "points": good_power_points,
        "features": {"rise_peak_fall_score_px": 24},
    }
    bad_power = {
        "trace_id": "bad_power",
        "series_id": "color_01",
        "curve_role": "power_density_curve",
        "axis_binding_hint": "right_y",
        "confidence": 0.96,
        "points": bad_power_points,
        "features": {"rise_peak_fall_score_px": 45},
    }

    selected, issues = select_final_traces_for_data_series(
        [voltage, bad_power, good_power],
        is_dual_y_power_density=True,
        axis_mapping=axis_mapping,
        crop_offset_xy=[0, 0],
    )

    assert [trace["trace_id"] for trace in selected] == ["voltage", "good_power"]
    assert issues == []
    assert good_power["final_trace_status"] == "accepted"
    assert bad_power["final_trace_status"] == "rejected"
    assert good_power["physics_validation"]["median_abs_residual_over_peak"] < 0.05
    assert (
        good_power["features"]["lvp_physics_fit_score"]
        > bad_power["features"]["lvp_physics_fit_score"]
    )


def test_power_density_second_pass_excludes_stable_voltage_trace(tmp_path: Path) -> None:
    crop = Image.new("RGB", (140, 100), "white")
    draw = ImageDraw.Draw(crop)
    voltage_color = (255, 0, 0)
    power_color = (255, 150, 150)
    voltage_points = [(8, 18), (45, 35), (85, 58), (125, 82)]
    power_points = [(8, 88), (45, 44), (70, 20), (95, 42), (125, 84)]
    draw.line(voltage_points, fill=voltage_color, width=3)
    draw.line(power_points, fill=power_color, width=2)

    voltage_trace = {
        "trace_id": "color_01_trace_01",
        "series_id": "color_01",
        "legend_label": "520",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.9,
        "points": [[x, int(18 + (82 - 18) * (x - 8) / (125 - 8))] for x in range(8, 126)],
        "features": {
            "raw_track_count_before_edge_merge": 1,
            "track_count_after_edge_merge": 1,
            "data_y_decreasing_score_px": 64,
        },
    }

    traces, issues = extract_power_density_after_voltage_exclusion(
        crop,
        [voltage_trace],
        masks_dir=tmp_path / "masks",
    )

    assert issues == []
    assert len(traces) == 1
    trace = traces[0]
    assert trace["curve_role"] == "power_density_curve"
    assert trace["axis_binding_hint"] == "right_y"
    assert trace["legend_label"] == "520"
    assert trace["extraction_method"] == "red_curve_after_voltage_exclusion_centerline"


def test_same_color_power_second_pass_rejects_non_power_fragments(tmp_path: Path) -> None:
    crop = Image.new("RGB", (220, 150), "white")
    draw = ImageDraw.Draw(crop)
    red = (240, 72, 96)

    power_points = [(70, 132), (100, 70), (130, 24), (160, 70), (190, 132)]
    spurious_non_power_fragment = [(0, 30), (1, 30), (50, 40), (55, 40)]
    draw.line(power_points, fill=red, width=3)
    draw.line(spurious_non_power_fragment, fill=red, width=3)

    voltage_trace = {
        "trace_id": "color_01_voltage_monotonic_trace",
        "series_id": "color_01",
        "legend_label": "PLNC-GDC",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.9,
        "points": [[x, int(10 + (120 * (x - 70) / 120))] for x in range(70, 191)],
        "features": {
            "data_y_decreasing_score_px": 120,
        },
    }
    candidates = [
        {
            "series_id": "color_01",
            "legend_label": "PLNC-GDC",
            "target_color_rgb": list(red),
            "target_colors_rgb": [list(red)],
            "color_distance": 35,
        }
    ]

    traces, issues = extract_power_density_after_voltage_exclusion(
        crop,
        candidates,
        [voltage_trace],
        masks_dir=tmp_path / "masks",
    )

    assert issues == []
    assert len(traces) == 1
    trace = traces[0]
    assert trace["curve_role"] == "power_density_curve"
    assert trace["features"]["selected_power_like_track_indexes"]
    assert trace["features"]["rejected_non_power_track_indexes"]
    assert min(x for x, _y in trace["points"]) > 60


def test_power_residual_mask_is_subset_of_base_mask(tmp_path: Path) -> None:
    crop = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(crop)
    red = (240, 72, 96)
    faint_red = (240, 168, 176)
    draw.line([(20, 22), (70, 52), (130, 92)], fill=red, width=3)
    draw.line([(20, 108), (70, 42), (100, 20), (130, 44), (160, 108)], fill=faint_red, width=2)

    voltage_trace = {
        "trace_id": "color_01_voltage_monotonic_trace",
        "series_id": "color_01",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.9,
        "points": [[x, int(22 + (70 * (x - 20) / 110))] for x in range(20, 131)],
        "features": {"data_y_decreasing_score_px": 70},
    }
    candidates = [
        {
            "series_id": "color_01",
            "target_color_rgb": list(red),
            "target_colors_rgb": [list(red)],
            "color_distance": 40,
        }
    ]

    extract_power_density_after_voltage_exclusion(
        crop,
        candidates,
        [voltage_trace],
        masks_dir=tmp_path / "masks",
    )

    base_mask, _points = _mask_for_color(crop, [red], 40)
    residual_mask = Image.open(tmp_path / "masks" / "color_01_power_density_after_voltage_exclusion.png").convert("L")
    base_pixels = list(base_mask.convert("L").tobytes())
    residual_pixels = list(residual_mask.tobytes())

    assert sum(1 for value in residual_pixels if value > 0) <= sum(1 for value in base_pixels if value > 0)
    assert all(base > 0 for base, residual in zip(base_pixels, residual_pixels) if residual > 0)


def test_voltage_branch_mask_is_written_as_base_subset(tmp_path: Path) -> None:
    crop = Image.new("RGB", (160, 110), "white")
    draw = ImageDraw.Draw(crop)
    blue = (48, 72, 144)
    draw.line([(8, 18), (52, 38), (104, 68), (148, 96)], fill=blue, width=4)
    draw.line([(8, 100), (55, 52), (86, 22), (118, 52), (148, 98)], fill=blue, width=2)
    candidates = [
        {
            "series_id": "color_01",
            "legend_label": "800 C",
            "target_color_rgb": list(blue),
            "target_colors_rgb": [list(blue)],
            "color_distance": 35,
        }
    ]

    traces, issues = extract_curve_traces(
        crop,
        candidates,
        masks_dir=tmp_path / "masks",
        is_dual_y_power_density=True,
        refine_dual_y_roles=False,
    )

    assert issues == []
    voltage_trace = next(trace for trace in traces if trace["trace_id"] == "color_01_voltage_monotonic_trace")
    assert voltage_trace["mask_ref"] == "color_01_voltage_branch.png"
    base_mask = Image.open(tmp_path / "masks" / "color_01.png").convert("L")
    voltage_mask = Image.open(tmp_path / "masks" / "color_01_voltage_branch.png").convert("L")
    base_pixels = list(base_mask.tobytes())
    voltage_pixels = list(voltage_mask.tobytes())

    assert sum(1 for value in voltage_pixels if value > 0) > 0
    assert sum(1 for value in voltage_pixels if value > 0) < sum(1 for value in base_pixels if value > 0)
    assert all(base > 0 for base, voltage in zip(base_pixels, voltage_pixels) if voltage > 0)


def test_power_density_endpoint_sparse_islands_are_trimmed() -> None:
    main_curve = [(x, 120 - abs(x - 80)) for x in range(12, 336)]
    track = [(0, 240), (1, 240), *main_curve, (348, 260), (349, 260)]

    trimmed, ranges = _trim_endpoint_sparse_islands(track, image_width=350)

    assert ranges == [[0, 1], [348, 349]]
    assert min(x for x, _y in trimmed) == 12
    assert max(x for x, _y in trimmed) == 335
    assert len(trimmed) == len(main_curve)


def test_voltage_second_pass_excludes_stable_power_density_trace(tmp_path: Path) -> None:
    crop = Image.new("RGB", (140, 100), "white")
    draw = ImageDraw.Draw(crop)
    red = (255, 0, 0)
    voltage_points = [(8, 18), (45, 35), (85, 58), (125, 82)]
    power_points = [(8, 88), (45, 44), (70, 20), (95, 42), (125, 84)]
    draw.line(voltage_points, fill=red, width=3)
    draw.line(power_points, fill=red, width=2)

    power_trace = {
        "trace_id": "power_density_after_voltage_exclusion_trace_01",
        "series_id": "power_density_after_voltage_exclusion",
        "legend_label": "520",
        "curve_role": "power_density_curve",
        "axis_binding_hint": "right_y",
        "confidence": 0.9,
        "points": [[x, int(88 - 68 * (1 - ((x - 70) / 62) ** 2))] for x in range(8, 126)],
        "features": {
            "rise_peak_fall_score_px": 40,
        },
    }
    candidates = [
        {
            "series_id": "color_01",
            "legend_label": "520",
            "target_color_rgb": list(red),
            "color_distance": 80,
        }
    ]

    traces, issues = extract_voltage_after_power_density_exclusion(
        crop,
        candidates,
        [power_trace],
        masks_dir=tmp_path / "masks",
    )

    assert issues == []
    assert len(traces) == 1
    trace = traces[0]
    assert trace["curve_role"] == "voltage_curve"
    assert trace["axis_binding_hint"] == "left_y"
    assert trace["legend_label"] == "520"
    assert trace["extraction_method"] == "same_color_after_power_density_exclusion_voltage_centerline"
    assert trace["features"]["source_power_density_trace_id"] == power_trace["trace_id"]


def test_voltage_second_pass_rejects_non_monotonic_residual(tmp_path: Path) -> None:
    crop = Image.new("RGB", (320, 120), "white")
    draw = ImageDraw.Draw(crop)
    red = (255, 0, 0)
    residual_power_points = [(170, 108), (205, 54), (240, 22), (275, 54), (310, 108)]
    draw.line(residual_power_points, fill=red, width=3)

    power_trace = {
        "trace_id": "power_density_after_voltage_exclusion_trace_01",
        "series_id": "color_01",
        "legend_label": "1000C",
        "curve_role": "power_density_curve",
        "axis_binding_hint": "right_y",
        "confidence": 0.9,
        "points": [[x, int(108 - 86 * (1 - ((x - 70) / 62) ** 2))] for x in range(8, 133)],
        "features": {
            "rise_peak_fall_score_px": 70,
        },
    }
    candidates = [
        {
            "series_id": "color_01",
            "legend_label": "1000C",
            "target_color_rgb": list(red),
            "color_distance": 80,
        }
    ]

    traces, issues = extract_voltage_after_power_density_exclusion(
        crop,
        candidates,
        [power_trace],
        masks_dir=tmp_path / "masks",
    )

    assert traces == []
    assert issues == ["color_01:NO_VOLTAGE_TRACE_AFTER_POWER_DENSITY_EXCLUSION"]


def test_final_selection_rejects_voltage_trace_with_power_like_shape() -> None:
    power_like_voltage = {
        "trace_id": "color_01_voltage_after_power_density_exclusion_trace",
        "series_id": "color_01",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "confidence": 0.95,
        "points": [[x, 80 - abs(x - 40)] for x in range(80)],
        "features": {
            "data_y_decreasing_score_px": -10,
            "rise_peak_fall_score_px": 35,
        },
    }
    stable_power = {
        "trace_id": "color_01_power_density_after_voltage_exclusion_trace_merged",
        "series_id": "color_01",
        "curve_role": "power_density_curve",
        "axis_binding_hint": "right_y",
        "confidence": 0.9,
        "points": [[x, 80 - abs(x - 40)] for x in range(80)],
        "features": {
            "rise_peak_fall_score_px": 35,
        },
    }

    selected, issues = select_final_traces_for_data_series(
        [power_like_voltage, stable_power],
        is_dual_y_power_density=True,
    )

    assert [trace["trace_id"] for trace in selected] == [
        "color_01_power_density_after_voltage_exclusion_trace_merged"
    ]
    assert power_like_voltage["final_trace_reject_reason"] == "VOLTAGE_SHAPE_NOT_MONOTONIC"
    assert issues == ["color_01:voltage_curve:NO_FINAL_QUALITY_TRACE"]


def test_voltage_fragments_merge_into_longer_candidate() -> None:
    left_fragment = {
        "trace_id": "color_01_trace_01",
        "series_id": "color_01",
        "legend_label": "1000C",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "series_color_rgb": [240, 40, 40],
        "points": [[x, 20 + x // 2] for x in range(5, 66)],
        "confidence": 0.75,
        "extraction_method": "color_mask_column_tracking_thick_stroke_centerline",
        "features": {"data_y_decreasing_score_px": 30},
        "coordinate_space": "crop_px",
        "mask_ref": "color_01.png",
    }
    right_fragment = {
        "trace_id": "color_01_trace_02",
        "series_id": "color_01",
        "legend_label": "1000C",
        "curve_role": "voltage_curve",
        "axis_binding_hint": "left_y",
        "series_color_rgb": [240, 40, 40],
        "points": [[x, 20 + x // 2] for x in range(105, 166)],
        "confidence": 0.75,
        "extraction_method": "color_mask_column_tracking_thick_stroke_centerline",
        "features": {"data_y_decreasing_score_px": 30},
        "coordinate_space": "crop_px",
        "mask_ref": "color_01.png",
    }

    merged = _merge_voltage_fragment_traces([left_fragment, right_fragment])

    assert len(merged) == 1
    trace = merged[0]
    assert trace["trace_id"] == "color_01_voltage_fragment_merged_trace"
    assert trace["curve_role"] == "voltage_curve"
    assert trace["features"]["source_fragment_trace_ids"] == ["color_01_trace_01", "color_01_trace_02"]
    xs = [point[0] for point in trace["points"]]
    assert min(xs) == 5
    assert max(xs) == 165


def test_role_recolor_candidates_preserve_series_and_role_granularity(tmp_path: Path) -> None:
    traces = []
    for series_id in ("color_01", "color_02"):
        traces.append(
            {
                "trace_id": f"{series_id}_voltage",
                "series_id": series_id,
                "legend_label": series_id,
                "curve_role": "voltage_curve",
                "axis_binding_hint": "left_y",
                "points": [[x, 10 + x] for x in range(5, 35)],
            }
        )
        traces.append(
            {
                "trace_id": f"{series_id}_power",
                "series_id": series_id,
                "legend_label": series_id,
                "curve_role": "power_density_curve",
                "axis_binding_hint": "right_y",
                "points": [[x, 40 - abs(x - 20)] for x in range(5, 35)],
            }
        )

    summary = _draw_role_recolored_crop(Image.new("RGB", (50, 50), "white"), traces, tmp_path / "role.png")
    candidates = _role_recolor_candidates(traces)

    assert summary["role_trace_counts"] == {"voltage_curve": 2, "power_density_curve": 2}
    assert len(summary["trace_recolors"]) == 4
    assert len(candidates) == 4
    assert {candidate["source_series_id"] for candidate in candidates} == {"color_01", "color_02"}
    assert {
        (candidate["source_series_id"], candidate["forced_curve_role"])
        for candidate in candidates
    } == {
        ("color_01", "voltage_curve"),
        ("color_01", "power_density_curve"),
        ("color_02", "voltage_curve"),
        ("color_02", "power_density_curve"),
    }


def test_role_recolored_second_pass_uses_voltage_first_power_residual(tmp_path: Path) -> None:
    crop = Image.new("RGB", (180, 130), "white")
    draw = ImageDraw.Draw(crop)
    voltage_rgb = (0, 114, 178)
    power_rgb = (213, 94, 0)
    draw.line([(10, 20), (70, 48), (140, 92)], fill=voltage_rgb, width=4)
    draw.line([(10, 118), (55, 58), (85, 26), (120, 58), (160, 116)], fill=power_rgb, width=4)
    candidates = [
        {
            "series_id": "role_color_01_voltage_curve",
            "source_series_id": "color_01",
            "target_color_rgb": list(voltage_rgb),
            "color_distance": 24,
            "forced_curve_role": "voltage_curve",
            "forced_axis_binding_hint": "left_y",
        },
        {
            "series_id": "role_color_01_power_density_curve",
            "source_series_id": "color_01",
            "target_color_rgb": list(power_rgb),
            "color_distance": 24,
            "forced_curve_role": "power_density_curve",
            "forced_axis_binding_hint": "right_y",
        },
    ]

    traces, issues = _extract_role_recolored_traces_voltage_first(
        crop,
        candidates,
        masks_dir=tmp_path / "role_masks",
        is_dual_y_power_density=True,
    )

    assert not issues
    assert (tmp_path / "role_masks" / "role_color_01_voltage_curve.png").exists()
    assert (
        tmp_path
        / "role_masks"
        / "role_color_01_power_density_curve_power_density_after_voltage_exclusion.png"
    ).exists()
    power_trace = next(trace for trace in traces if trace.get("curve_role") == "power_density_curve")
    assert power_trace["features"]["source_voltage_trace_id"].startswith("role_color_01_voltage_curve")
    assert power_trace["trace_id"].startswith("role_color_01_power_density_curve_power_density")


def test_run_data_extraction_for_panel_writes_expected_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    panel_id = "p1_F01-P01_chart"
    panel_rel = "panels/p1/p1_F01-P01_chart.png"
    panel_path = dataset_root / panel_rel
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    plot_bbox = [20, 10, 140, 100]
    draw.rectangle(plot_bbox, outline="black")
    red_points = [(20 + x, 25 + x // 2) for x in range(5, 110, 4)]
    blue_points = [(20 + x, int(88 - 48 * (1 - abs(x - 55) / 55))) for x in range(5, 110, 4)]
    draw.line(red_points, fill=(240, 40, 40), width=2)
    draw.line(blue_points, fill=(30, 100, 230), width=2)
    image.save(panel_path)

    _write_json(
        dataset_root / "links" / "gt_plots.json",
        {
            "schema_version": 1,
            "panels": [
                {
                    "panel_id": panel_id,
                    "panel_path": panel_rel,
                    "panel_kind": "chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": plot_bbox,
                }
            ],
        },
    )
    _write_json(
        dataset_root / "links" / "gt_axes.json",
        {
            "schema_version": 1,
            "panels": [
                {
                    "panel_id": panel_id,
                    "panel_path": panel_rel,
                    "panel_kind": "chart",
                    "is_dual_y": True,
                    "review_status": "accept",
                    "axis_left_bbox_xyxy": [0, 10, 22, 100],
                    "axis_bottom_bbox_xyxy": [20, 98, 140, 119],
                    "axis_right_bbox_xyxy": [138, 10, 159, 100],
                }
            ],
        },
    )
    _write_json(dataset_root / "links" / "gt_axis_ticks.json", {"schema_version": 1, "panels": []})

    manual_pairs = {
        panel_id: {
            "bottom": [[20, 0.0], [140, 4.5]],
            "left": [[10, 1.2], [100, 0.0]],
            "right": [[10, 2.4], [100, 0.0]],
        }
    }
    manifest = run_data_extraction_for_panel(
        dataset_root=dataset_root,
        panel_id=panel_id,
        out_root=tmp_path / "out",
        manual_axis_pairs=manual_pairs,
        max_colors=4,
        exclusion_bboxes_crop_px=[[90, 5, 118, 22]],
        exclusion_source="test_legend",
    )

    panel_out = tmp_path / "out" / panel_id
    assert manifest["panel_id"] == panel_id
    assert (panel_out / "plot_crop.png").exists()
    assert (panel_out / "axis_mapping.json").exists()
    assert (panel_out / "chart_extraction_strategy.json").exists()
    assert (panel_out / "color_filter_source.png").exists()
    assert (panel_out / "image_quality_report.json").exists()
    assert (panel_out / "series_candidates.json").exists()
    assert (panel_out / "exclusion_regions.json").exists()
    assert (panel_out / "curve_traces_crop_px.json").exists()
    assert (panel_out / "final_curve_traces_crop_px.json").exists()
    assert (panel_out / "final_trace_overlay.png").exists()
    assert (panel_out / "raw_vs_final_trace_overlay.png").exists()
    assert (panel_out / "role_recolored_crop.png").exists()
    assert (panel_out / "role_recolored_curve_traces_crop_px.json").exists()
    assert (panel_out / "curve_series_data.json").exists()
    assert (panel_out / "overlay.png").exists()

    traces = json.loads((panel_out / "curve_traces_crop_px.json").read_text(encoding="utf-8"))
    roles = {trace["curve_role"] for trace in traces["traces"]}
    assert "voltage_curve" in roles
    assert "power_density_curve" in roles
    final_traces = json.loads((panel_out / "final_curve_traces_crop_px.json").read_text(encoding="utf-8"))
    assert final_traces["source"] == "accepted_traces_with_shape_preserving_final_smoothing"

    exclusions = json.loads((panel_out / "exclusion_regions.json").read_text(encoding="utf-8"))
    assert exclusions["bboxes"] == [[90, 5, 118, 22]]
    role_traces = json.loads((panel_out / "role_recolored_curve_traces_crop_px.json").read_text(encoding="utf-8"))
    assert role_traces["source"] == "shape_role_recolored_second_pass_filtering"
    assert role_traces["input_trace_scope"] == "final_accepted_role_traces"
    assert role_traces["traces"]
    image_quality = json.loads((panel_out / "image_quality_report.json").read_text(encoding="utf-8"))
    assert "laplacian_variance" in image_quality["sharpness"]
    validation = json.loads((panel_out / "validation_report.json").read_text(encoding="utf-8"))
    assert "image_laplacian_variance" in validation["checks"]
    assert "min_palette_distance" in validation["checks"]


def test_run_data_extraction_for_panel_resets_stale_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    panel_id = "p1_F01-P01_chart"
    panel_rel = "panels/p1/p1_F01-P01_chart.png"
    panel_path = dataset_root / panel_rel
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    plot_bbox = [10, 10, 90, 70]
    draw.rectangle(plot_bbox, outline="black")
    draw.line([(15, 20), (35, 30), (60, 44), (85, 58)], fill=(240, 40, 40), width=2)
    image.save(panel_path)

    _write_json(
        dataset_root / "links" / "gt_plots.json",
        {
            "schema_version": 1,
            "panels": [
                {
                    "panel_id": panel_id,
                    "panel_path": panel_rel,
                    "panel_kind": "chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": plot_bbox,
                }
            ],
        },
    )
    _write_json(
        dataset_root / "links" / "gt_axes.json",
        {
            "schema_version": 1,
            "panels": [
                {
                    "panel_id": panel_id,
                    "panel_path": panel_rel,
                    "panel_kind": "chart",
                    "is_dual_y": False,
                    "review_status": "accept",
                    "axis_left_bbox_xyxy": [0, 10, 12, 70],
                    "axis_bottom_bbox_xyxy": [10, 68, 90, 79],
                    "axis_right_bbox_xyxy": None,
                }
            ],
        },
    )
    _write_json(dataset_root / "links" / "gt_axis_ticks.json", {"schema_version": 1, "panels": []})

    out_root = tmp_path / "out"
    stale_file = out_root / panel_id / "stale_artifact.txt"
    stale_nested = out_root / panel_id / "old_masks" / "stale.png"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("old", encoding="utf-8")
    stale_nested.parent.mkdir(parents=True, exist_ok=True)
    stale_nested.write_text("old", encoding="utf-8")

    run_data_extraction_for_panel(
        dataset_root=dataset_root,
        panel_id=panel_id,
        out_root=out_root,
        manual_axis_pairs={
            panel_id: {
                "bottom": [[10, 0.0], [90, 1.0]],
                "left": [[10, 1.0], [70, 0.0]],
            }
        },
        max_colors=2,
    )

    assert not stale_file.exists()
    assert not stale_nested.exists()
    assert (out_root / panel_id / "run_manifest.json").exists()

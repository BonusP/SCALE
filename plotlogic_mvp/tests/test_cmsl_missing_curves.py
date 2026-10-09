# ###수정 추가 수정: black 후보·same-color 범례·VLM 선행 선택과 gap support의 회귀를 검증합니다.
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from app import config

sys.path.insert(0, str(config.PLOTLOGIC_ROOT))
from core import data_extraction_pipeline_cmsl as core
from core.cmsl_curve_support import build_tracking_support, repair_crossing_tracks
from app.services.curve_axis_association import validate_assignments


def test_black_legend_pixels_survive_candidate_filter():
    image = Image.new("RGB", (160, 100), "white")
    ImageDraw.Draw(image).line((8, 80, 150, 20), fill="black", width=2)
    legend = [{"legend_label": "black", "swatch_color_rgb": [0, 0, 0]}]
    candidates = core.detect_color_candidates(image, legend_items=legend)
    assert len(candidates) == 1
    assert candidates[0]["legend_label"] == "black"
    assert candidates[0]["target_color_rgb"] == [16, 16, 16]
    source = core.build_solid_color_filter_source(image, candidates)
    assert source.getpixel((8, 80)) == (16, 16, 16)


def test_same_color_voltage_power_labels_are_not_forced_to_first_legend():
    image = Image.new("RGB", (160, 100), "white")
    ImageDraw.Draw(image).line((8, 80, 150, 20), fill=(220, 0, 0), width=2)
    legend = [{"legend_label": "V--600C", "swatch_color_rgb": [220, 0, 0], "axis_binding_hint": "left_y"},
              {"legend_label": "P--600C", "swatch_color_rgb": [220, 0, 0], "axis_binding_hint": "right_y"}]
    candidates = core.detect_color_candidates(image, legend_items=legend)
    core.match_legend_items_to_series_candidates(candidates, legend)
    assert len(candidates) == 1
    assert candidates[0]["legend_label"] is None
    assert candidates[0]["candidate_axis_binding_hint"] is None
    assert {row["legend_label"] for row in candidates[0]["legend_options"]} == {"V--600C", "P--600C"}


def test_dashed_support_preserves_tracker_gaps_and_records_inferred_pixels():
    mask = Image.new("L", (100, 300))
    draw = ImageDraw.Draw(mask)
    for start in range(5, 80, 12):
        draw.line((start, start * 3, start + 4, (start + 4) * 3), fill=255, width=1)
    points = core._points_from_mask(mask)
    assert not core._tracks_from_mask_points(points, image_width=100, min_width_px=8)
    support, inferred, report = build_tracking_support(mask, tracker=core._tracks_from_mask_points)
    tracks = core._tracks_from_mask_points(core._points_from_mask(support), image_width=100, min_width_px=8)
    assert tracks and max(len(track) for track in tracks) > 60
    assert report["inferred_pixel_count"] > 0
    assert all(abs(x2 - x1) <= 10 and abs(y2 - y1) <= 10 for track in tracks for (x1, y1), (x2, y2) in zip(track, track[1:]))
    assert inferred.getbbox() is not None


def test_support_never_bridges_explicit_exclusion_region():
    mask = Image.new("L", (80, 80))
    draw = ImageDraw.Draw(mask)
    draw.line((5, 40, 25, 40), fill=255)
    draw.line((40, 40, 65, 40), fill=255)
    forbidden = Image.new("L", mask.size)
    ImageDraw.Draw(forbidden).rectangle((28, 0, 37, 79), fill=255)
    support, _inferred, _report = build_tracking_support(mask, tracker=core._tracks_from_mask_points, forbidden_mask=forbidden)
    assert all(support.getpixel((x, 40)) == 0 for x in range(28, 38))


def test_vlm_receives_unknown_power_before_shape_filter_and_updates_legend(tmp_path):
    image = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((5, 20, 170, 105), fill=(0, 0, 220), width=2)
    draw.line((5, 100, 170, 60), fill=(220, 0, 0), width=2)
    image.save(tmp_path / "panel.png")
    panel = {"panel_id": "p", "panel_path": "panel.png", "gt_bbox_xyxy": [0, 0, 180, 120]}
    axes = {"is_dual_y": True, "axis_right_bbox_xyxy": [179, 0, 180, 120], "extraction_family": "current_voltage_power_dual_y", "axis_left_label": "Voltage", "axis_right_label": "Power"}
    pairs = {"bottom": [[0, 0], [180, 1]], "left": [[0, 1], [120, 0]], "right": [[0, 2], [120, 0]]}
    legend = [{"legend_label": "V--red", "swatch_color_rgb": [220, 0, 0], "axis_binding_hint": "left_y"},
              {"legend_label": "P--red", "swatch_color_rgb": [220, 0, 0], "axis_binding_hint": "right_y"},
              {"legend_label": "V--blue", "swatch_color_rgb": [0, 0, 220], "axis_binding_hint": "left_y"}]
    received = []
    def resolver(**kwargs):
        received.extend(kwargs["traces"])
        assignments = []
        for trace in kwargs["traces"]:
            red = trace["series_color_rgb"][0] > 100
            assignments.append({"trace_id": trace["trace_id"], "axis_binding": "right_y" if red else "left_y", "confidence": .9, "is_curve": True, "legend_label": "P--red" if red else "V--blue"})
        return {"assignments": assignments, "issues": []}
    core.run_data_extraction_for_panel(dataset_root=tmp_path, panel_id="p", out_root=tmp_path / "runs", manual_axis_pairs=pairs,
                                     plot_panel_override=panel, axes_panel_override=axes, ticks_panel_override={"axes": []}, legend_items=legend, axis_binding_resolver=resolver)
    assert any(trace["curve_role"] == "unknown_curve" for trace in received)
    result = json.loads((tmp_path / "runs/p/curve_series_data.json").read_text())
    red = next(trace for trace in result["series"] if trace["legend_label"] == "P--red")
    assert red["axis_binding"] == "right_y" and red["curve_role"] == "power_density_curve" and red["points_data"]


def test_vlm_cannot_invent_an_unseen_legend_label():
    payload = {"assignments": [{"trace_id": "a", "axis_binding": "right_y", "confidence": .9, "legend_label": "invented", "is_curve": True}]}
    result = validate_assignments(payload, [{"trace_id": "a", "legend_options": [{"legend_label": "P--red"}]}])
    assert result["assignments"][0]["legend_label"] is None
    assert "INVALID_LEGEND_ASSIGNMENT:a" in result["issues"]


def test_crossing_tail_repair_preserves_coordinates_and_gap_limits():
    a = [(x, x if x <= 25 else 50 - x) for x in range(51)]
    b = [(x, 50 - x if x <= 25 else x) for x in range(51)]
    tracks, swaps = repair_crossing_tracks([a, b])
    assert swaps == 1
    assert sorted(a + b) == sorted(tracks[0] + tracks[1])
    for track in tracks:
        assert all(y2 >= y1 for (_x1, y1), (_x2, y2) in zip(track, track[1:])) or all(y2 <= y1 for (_x1, y1), (_x2, y2) in zip(track, track[1:]))
        assert all(0 < x2 - x1 <= 10 and abs(y2 - y1) <= 10 for (x1, y1), (x2, y2) in zip(track, track[1:]))


def test_coherent_partial_curve_is_not_discarded_by_old_width_filter(tmp_path):
    image = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(image).line((50, 20, 75, 45), fill=(220, 0, 0), width=2)
    candidate = {"series_id": "red", "target_color_rgb": [220, 0, 0], "legend_label": "red"}
    traces, _issues = core.extract_curve_traces(image, [candidate], masks_dir=tmp_path / "masks", is_dual_y_power_density=False)
    assert traces and len(traces[0]["points"]) >= 20
# ###

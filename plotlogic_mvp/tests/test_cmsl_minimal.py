# ###수정 변경된 CMSL 알고리즘과 격리·axis 대응·PDF 업로드 계약을 검증합니다.
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app import config
from app.services.curve_axis_association import assign_curve_axes, validate_assignments
from app.services.portable_pdf import extract_pdf_images
from app.services.run_methods import resolve_method
from app.services.vision import current_prompt_metadata
# ###수정 standalone core 테스트에서도 복사한 실험 core 경로를 사용합니다.
sys.path.insert(0, str(config.PLOTLOGIC_ROOT))
# ###


# ###수정 categorical 복원: 실제 job 분기에서 직접 읽기와 CMSL 경로를 검증합니다.
# chart_fmailiy 다시 4개로 수정
@pytest.mark.parametrize("family,categorical_axis,expected", [
    ("categorical_line_chart", False, "categorical"),  # Axis geometry is optional.
    ("single_axis_line_chart", True, "categorical"),
    ("dual_y_generic_line_chart", True, "categorical"),
    ("current_voltage_power_dual_y", True, "categorical"),
    ("single_axis_line_chart", False, "cmsl"),
    ("dual_y_generic_line_chart", False, "cmsl"),
    ("cmsl_line_chart", False, "skipped"),
    ("scatter_plot", False, "skipped"),
    ("current_voltage_power_dual_y", False, "cmsl"),
    ("bar_chart", True, "skipped"),
])
def test_cmsl_job_routes_categorical_and_numeric_charts(tmp_path, monkeypatch, family, categorical_axis, expected):
    from app.services import pipeline

    image_path = tmp_path / "figure.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    panel = {
        "panel_index": 1, "panel_type": "chart", "panel_bbox_figure_px": [0, 0, 100, 80],
        "is_chart": True, "is_extraction_target": True, "extraction_family": family,
        "chart_family": family, "axes": [],
    }
    if categorical_axis:
        panel["axes"] = [{"axis_role": "bottom", "ticks": [
            {"label_text": "Sample A", "label_numeric_value": None},
            {"label_text": "Sample B", "label_numeric_value": None},
        ]}]
    rows, updates, routes = [], [], []
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(config, "PAPERRAG_POSTPROCESS_ENABLED", False)
    monkeypatch.setattr(pipeline, "add_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "update_job", lambda *args, **kwargs: updates.append(kwargs))
    monkeypatch.setattr(pipeline, "update_paper_extraction", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "upsert_chart", lambda **kwargs: rows.append(kwargs))
    monkeypatch.setattr(pipeline, "paper_snapshot", lambda *args: {"parse": {"status": "success"}})
    monkeypatch.setattr(pipeline, "collect_images", lambda *args: [image_path])
    monkeypatch.setattr(pipeline, "_find_cached_figure_analysis", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "analyze_figure_panels", lambda *args: {"panels": [panel]})
    monkeypatch.setattr(pipeline, "validate_figure_analysis", lambda path, analysis: analysis)

    def extraction(route, *args, **kwargs):
        routes.append((route, kwargs))
        return {"result_dir": tmp_path, "overlay_path": None, "csv_path": tmp_path / "data.csv", "series_count": 1}

    # categorical 처리를 위한 수정
    monkeypatch.setattr(pipeline, "run_categorical_extraction", lambda *args, **kwargs: extraction("categorical", *args, **kwargs))
    monkeypatch.setattr(pipeline, "run_codex_direct_categorical_extraction", lambda *args, **kwargs: extraction("categorical", *args, **kwargs))
    monkeypatch.setattr(pipeline, "run_extraction", lambda *args, **kwargs: extraction("cmsl", *args, **kwargs))
    pipeline.process_job("test-categorical", 1, "cmsl_minimal")

    assert len(rows) == 1
    assert rows[0]["chart_id"].endswith("__cmsl")
    if expected == "skipped":
        assert not routes
        assert rows[0]["status"] == "skipped_non_extractable_chart"
    else:
        assert rows[0]["status"] == "extracted"
        assert [route for route, kwargs in routes] == [expected]
        if expected == "categorical":
            assert rows[0]["extraction_family"] == "categorical_line_chart"
        else:
            assert routes[0][1]["cmsl_minimal"] is True
    assert updates[-1]["status"] == ("completed_with_skips" if expected == "skipped" else "completed")
# ###
from core import data_extraction_pipeline_cmsl as core


def test_quantization_midpoints_cover_512_bins():
    assert core._quantize((0, 31, 32)) == (16, 16, 48)
    assert core._quantize((255, 254, 224)) == (240, 240, 240)
    assert len({core._quantize((r, g, b)) for r in range(0, 256, 32) for g in range(0, 256, 32) for b in range(0, 256, 32)}) == 512


def test_background_is_selected_before_brightness_filtering():
    image = Image.new("RGB", (100, 80), (180, 30, 30))
    ImageDraw.Draw(image).line((5, 60, 95, 20), fill=(10, 80, 210), width=2)
    mask, info = core._build_background_mask(image)
    assert info["dominant_background_bin"] == [5, 0, 0]
    assert mask.getpixel((0, 0)) == 255
    assert mask.getpixel((5, 60)) == 0


def test_grid_strict_80_percent_and_adjacent_groups():
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((0, 10, 80, 10), fill="black")
    draw.line((0, 11, 80, 11), fill="black")
    draw.line((0, 15, 79, 15), fill="black")
    draw.line((20, 0, 20, 79), fill="black")
    draw.line((21, 0, 21, 79), fill="black")
    background, _ = core._build_background_mask(image)
    grid, info = core._detect_grid_mask(background)
    assert info["horizontal_grid_groups"] == [[10, 11]]
    assert info["vertical_grid_groups"] == [[20, 21]]
    assert grid.getpixel((95, 10)) == 255
    assert grid.getpixel((95, 15)) == 0


def test_rgb_50_preserves_hsv_alternative_and_legend_50_is_hard_limit():
    assert core._same_color_family((200, 0, 0), (140, 0, 0))  # HSV alternative survives a 60 RGB distance.
    assert not core._matches_legend_seed((140, 0, 0), (200, 0, 0))
    assert core._matches_legend_seed((180, 0, 0), (200, 0, 0))


def test_grid_mask_cannot_be_reintroduced_by_curve_mask_strengthening():
    image = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(image).line((5, 45, 75, 12), fill=(220, 20, 20), width=3)
    blocked = Image.new("L", image.size)
    ImageDraw.Draw(blocked).line((40, 0, 40, 59), fill=255)
    candidates = core.detect_color_candidates(image, non_curve_mask=blocked)
    source = core.build_solid_color_filter_source(image, candidates, non_curve_mask=blocked)
    assert all(source.getpixel((40, y)) == (255, 255, 255) for y in range(60))


def test_tracking_uses_euclidean_choice_and_fixed_gaps():
    # At x=10, Euclidean distance chooses (9,15), whereas the old score chooses (1,18).
    points = {(0, 18), (1, 18), (3, 0), (4, 3), (5, 6), (6, 9), (7, 12), (8, 14), (9, 15)} | {(x, 20) for x in range(10, 31)}
    tracks = core._tracks_from_mask_points(points, image_width=31, min_width_px=1)
    chosen = next(track for track in tracks if (10, 20) in track)
    assert (9, 15) in chosen
    assert (1, 18) not in chosen
    separated = {(x, 10) for x in range(0, 15)} | {(x, 10) for x in range(25, 45)}
    assert len(core._tracks_from_mask_points(separated, image_width=45, min_width_px=1)) == 2


def test_axis_validation_does_not_silently_bind_unknown_or_duplicate_ids():
    inputs = [{"trace_id": "a"}, {"trace_id": "b"}, {"trace_id": "c"}]
    result = validate_assignments({"assignments": [
        {"trace_id": "a", "axis_binding": "right_y", "confidence": .9, "reason": "legend"},
        {"trace_id": "b", "axis_binding": "left_y", "confidence": .9},
        {"trace_id": "b", "axis_binding": "right_y", "confidence": .9},
        {"trace_id": "unknown", "axis_binding": "left_y", "confidence": .9},
    ], "issues": []}, inputs)
    bindings = {row["trace_id"]: row["axis_binding"] for row in result["assignments"]}
    assert bindings == {"a": "right_y", "b": None, "c": None}


def test_unresolved_axis_does_not_export_wrong_numeric_values():
    trace = {"trace_id": "a", "series_id": "s", "axis_binding_hint": "unresolved", "points": [[4, 5]]}
    model = {"slope": 1, "intercept": 0}
    result = core._convert_trace_to_data(trace, {"axes": {"bottom": {"model": model}, "left": {"model": model}}})
    assert result["points_data"] == []
    assert "UNRESOLVED_CURVE_AXIS" in result["issues"]


def test_registry_and_data_are_isolated():
    exp = Path(__file__).resolve().parents[2]
    assert config.DATA_ROOT == exp / "data"
    assert config.DB_PATH == exp / "data/mvp.sqlite3"
    assert config.PAPERRAG_PAPERS_DIR == exp / "data/papers"
    assert not config.PAPERRAG_POSTPROCESS_ENABLED
    assert resolve_method("cmsl_minimal").chart_id_suffix == "cmsl"
    metadata = current_prompt_metadata("figure_panel_classifier_cmsl.md")
    registry = json.loads(Path(metadata["chart_family_registry"]).read_text(encoding="utf-8"))
    # chart_fmailiy 다시 4개로 수정
    assert Path(metadata["chart_family_registry"]).name == "chart_families.json"
    assert {item["id"] for item in registry["enabled_extraction_families"]} == {
        "single_axis_line_chart", "dual_y_generic_line_chart",
        "current_voltage_power_dual_y", "categorical_line_chart",
    }
    for family in registry["enabled_extraction_families"]:
        assert any("Marker-only scatter" in cue for cue in family["negative_cues"])


def test_dual_axis_pipeline_uses_resolver_before_numeric_conversion(tmp_path):
    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((8, 90, 150, 25), fill=(220, 20, 20), width=2)
    draw.line((8, 20, 150, 85), fill=(20, 50, 220), width=2)
    image.save(tmp_path / "panel.png")
    panel = {"panel_id": "p", "panel_path": "panel.png", "gt_bbox_xyxy": [0, 0, 160, 120]}
    # chart_fmailiy 다시 4개로 수정
    axes = {"is_dual_y": True, "axis_right_bbox_xyxy": [159, 0, 160, 120], "extraction_family": "dual_y_generic_line_chart", "axis_left_label": "left", "axis_right_label": "right"}
    ticks = {"axes": []}
    pairs = {"bottom": [[0, 0], [160, 100]], "left": [[0, 10], [120, 0]], "right": [[0, 20], [120, 0]]}
    received = []
    def resolver(**kwargs):
        received.extend(kwargs["traces"])
        return {"assignments": [{"trace_id": item["trace_id"], "axis_binding": "right_y", "confidence": 1, "reason": "test resolver"} for item in kwargs["traces"]], "issues": []}
    core.run_data_extraction_for_panel(dataset_root=tmp_path, panel_id="p", out_root=tmp_path / "runs", manual_axis_pairs=pairs, plot_panel_override=panel, axes_panel_override=axes, ticks_panel_override=ticks, axis_binding_resolver=resolver)
    result = json.loads((tmp_path / "runs/p/curve_series_data.json").read_text())
    assert received and result["series"]
    assert all(item["axis_binding"] == "right_y" and item["points_data"] for item in result["series"])
    assert (tmp_path / "runs/p/grid_mask.png").exists()
    final = json.loads((tmp_path / "runs/p/final_curve_traces_crop_px.json").read_text())
    assert all(item["extraction_method"] == "cmsl_greedy_euclidean_column_tracking" for item in final["traces"])


def test_vlm_failure_is_saved_and_leaves_axis_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_CODEX", False)
    trace = {"trace_id": "s", "points": [[2, 2], [10, 8]], "legend_label": "x"}
    report = assign_curve_axes(panel_image=Image.new("RGB", (30, 20), "white"), traces=[trace], axis_mapping={"axes": {}}, out_dir=tmp_path)
    assert report["assignments"][0]["axis_binding"] is None
    assert "CODEX_DISABLED" in report["issues"]
    assert (tmp_path / "curve_axis_overlay.png").exists()


def test_portable_parser_keeps_three_pdf_images_separate(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[4] / "output/pdf/cmsl_chart_gui_test_3_images.pdf"
    monkeypatch.setattr(config, "PAPERRAG_PAPERS_DIR", tmp_path)
    folder = tmp_path / "0001"
    folder.mkdir()
    (folder / "source.pdf").write_bytes(source.read_bytes())
    report = extract_pdf_images(1)
    assert report["ok"] and report["image_count"] == 3
    detail = json.loads((folder / "detail_info.json").read_text())
    assert [Image.open(folder / item["path"]).size for item in detail["figures"]] == [(485, 430), (575, 497), (1022, 651)]
# ###

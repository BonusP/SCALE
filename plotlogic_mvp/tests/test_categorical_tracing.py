# categorical 처리를 위한 수정: verify pixel artifacts, table conversion and fallback.
import json

import pytest
from PIL import Image, ImageDraw

from app.services.categorical_tracing import categorical_series
from app.services import plotlogic_bridge as bridge


def test_category_positions_preserve_labels_and_merge_fragments():
    vision = {"axes": [{"axis_role": "bottom", "ticks": [
        {"label_text": "Q2", "tick_position_px": 30},
        {"label_text": "Q1", "tick_position_px": 10},
    ]}], "legend_items": [{"legend_label": "A"}]}
    traces = [{"trace_id": name, "series_id": "a", "legend_label": "A", "axis_binding_hint": "left_y", "points": pts}
              for name, pts in [("first", [[10, 20]]), ("second", [[30, 40]])]]
    result = categorical_series("p", vision, traces, {"axes": {"left": {"model": {"slope": -2, "intercept": 100}}}})
    assert not result["issues"]
    assert result["series"][0]["points_data"] == [["Q1", 60], ["Q2", 20]]
    assert len(result["series"]) == 1


def test_ambiguous_category_does_not_average_different_curves():
    vision = {"axes": [{"axis_role": "bottom", "ticks": [{"label_text": "Q1", "tick_position_px": 10}]}]}
    traces = [{"trace_id": str(y), "series_id": "a", "legend_label": "A", "axis_binding_hint": "left_y", "points": [[10, y]]}
              for y in (20, 50)]
    result = categorical_series("p", vision, traces, {"axes": {"left": {"model": {"slope": 1, "intercept": 0}}}})
    assert not result["series"]
    assert result["issues"]


@pytest.mark.parametrize("geometry", [True, False])
def test_categorical_run_produces_pixel_artifacts_and_table(tmp_path, monkeypatch, geometry):
    panel_id = "categorical_test"
    path = tmp_path / "panel.png"
    image = Image.new("RGB", (200, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 40, 170, 40), fill=(40, 100, 210), width=3)
    draw.line((20, 95, 170, 95), fill=(210, 40, 40), width=3)
    image.save(path)
    labels = [{"legend_label": "A", "swatch_color_rgb": [40, 100, 210], "axis_binding_hint": "left_y"},
              {"legend_label": "B", "swatch_color_rgb": [210, 40, 40], "axis_binding_hint": "left_y"}]
    vision = {"axis_labels": {"x": "Quarter", "left_y": "Value"}, "legend_items": labels,
              "axes": [{"axis_role": "bottom", "ticks": [
                  {"label_text": label, "tick_position_px": x} for label, x in [("Q1", 20), ("Q2", 90), ("Q3", 160)]
              ]}] if geometry else []}
    monkeypatch.setattr(bridge.config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(bridge, "prepare_dataset", lambda *args: (tmp_path, panel_id, "panel.png", path, vision, [0, 0, 200, 140]))
    monkeypatch.setattr(bridge, "read_text_cached", lambda *args: {"items": []})
    monkeypatch.setattr(bridge, "build_ocr_text_geometry", lambda *args, **kwargs: {"panel_vision": vision, "summary": {}, "issues": []})
    ticks = {"axes": [{"axis_role": "left", "tick_label_candidates": [
        {"numeric_value": 140, "bbox_xyxy": [0, 0, 1, 1]}, {"numeric_value": 0, "bbox_xyxy": [0, 139, 1, 140]}],
        "tick_matches": [{"candidate_bbox_xyxy": [0, 0, 1, 1], "tick_position_px": 0},
                         {"candidate_bbox_xyxy": [0, 139, 1, 140], "tick_position_px": 140}]}]}
    monkeypatch.setattr(bridge, "overrides_from_vision", lambda *args: (
        {"panel_id": panel_id, "panel_path": "panel.png", "gt_bbox_xyxy": [0, 0, 200, 140]},
        {"extraction_family": "categorical_line_chart", "axis_left_label": "Value"}, ticks, labels, []))
    calls = []
    def fallback(panel_path, **kwargs):
        calls.append(kwargs)
        assert kwargs["overlay_path"].exists()
        assert kwargs["trace_context"]["traces"]
        return {"value_axis_label": "Value", "chart_type": "line", "issues": [], "items": [
            {"category": "Q1", "group_label": label, "value": value, "confidence": 1}
            for label, value in [("A", 100), ("B", 45)]]}
    monkeypatch.setattr(bridge, "read_categorical_table", fallback)
    result = bridge.run_categorical_extraction("j", 3, panel_id, path, vision)
    out = result["result_dir"]
    for name in ["color_filter_source.png", "curve_traces_panel_px.json", "overlay.png", "categorical_pixel_table.json", "curve_series_data.csv"]:
        assert (out / name).exists()
    series = json.loads((out / "curve_series_data.json").read_text(encoding="utf-8"))
    assert len(series["series"]) == 2
    assert isinstance(series["series"][0]["points_data"][0][0], str)
    assert bool(calls) is not geometry
    if geometry:
        assert series["category_order"] == ["Q1", "Q2", "Q3"]
        assert series["series"][0]["points_data"][0][1] == pytest.approx(100, abs=1)

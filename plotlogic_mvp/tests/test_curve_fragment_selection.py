# 정상 curve 조각 보존: 전압·전력 조각과 기존 제외 조건 검증
import copy
import sys

import pytest

from app import config

sys.path.insert(0, str(config.PLOTLOGIC_ROOT))
from core.data_extraction_pipeline_cmsl import _select_cmsl_traces_after_axis_binding
from core.data_extraction_pipeline_cmsl import _group_curve_segments, _apply_lvp_power_density_fit


def fragment(trace_id, axis, count=25, offset=0):
    return {"trace_id": trace_id, "series_id": "color_01", "axis_binding_hint": axis,
            "axis_is_curve": True, "points": [[offset + x, x] for x in range(count)]}


@pytest.mark.parametrize("dual_y", [True, False])
def test_preserve_all_valid_fragments_without_changing_coordinates(dual_y):
    traces = [fragment("v_first", "left_y"), fragment("v_second", "left_y", 50, 60),
              fragment("p_first", "right_y"), fragment("p_second", "right_y", 40, 70)]
    traces[0]["final_trace_reject_reason"] = "SHORTER_SAME_SERIES_AXIS_SEGMENT"
    original = copy.deepcopy(traces)
    accepted, issues = _select_cmsl_traces_after_axis_binding(traces, is_dual_y_power_density=dual_y)
    assert [trace["trace_id"] for trace in accepted] == [trace["trace_id"] for trace in original]
    assert [trace["points"] for trace in accepted] == [trace["points"] for trace in original]
    assert all(trace["final_trace_status"] == "accepted" and "final_trace_reject_reason" not in trace for trace in accepted)
    assert not issues


def test_existing_curve_axis_and_minimum_length_filters_remain():
    traces = [fragment("not_curve", "left_y"), fragment("unresolved", "unresolved"),
              fragment("short", "right_y", 19), fragment("valid", "left_y", 20)]
    traces[0]["axis_is_curve"] = False
    accepted, issues = _select_cmsl_traces_after_axis_binding(traces, is_dual_y_power_density=True)
    assert [trace["trace_id"] for trace in accepted] == ["valid"]
    assert [trace["final_trace_reject_reason"] for trace in traces[:3]] == [
        "VLM_NOT_A_COHERENT_CURVE", "UNRESOLVED_CURVE_AXIS", "TOO_FEW_POINTS"]
    assert issues == ["UNRESOLVED_CURVE_AXIS:unresolved"]


# 시리즈·조각 구분: 좌표 유지, 빈 구간 연결 방지
def test_grouped_series_preserves_segments_and_csv_trace_ids(tmp_path):
    import json
    import csv
    from app.services.plotlogic_bridge import export_curve_series_csv
    rows = [{"trace_id": trace, "series_id": "red", "axis_binding": axis,
             "legend_label": "red", "curve_role": "voltage_curve" if axis == "left_y" else "power_density_curve",
             "points_data": points, "source_pixel_trace_ref": trace, "issues": []}
            for trace, axis, points in [("a", "left_y", [[0, 1], [1, .9]]),
                                        ("b", "left_y", [[5, .8], [6, .7]]),
                                        ("c", "right_y", [[0, 0], [1, .1]])]]
    original = copy.deepcopy(rows)
    fitted, report = _apply_lvp_power_density_fit(rows)
    assert fitted == original
    assert report[0]["reason"] == "SEGMENTED_CURVE_PRESERVE_COORDINATES"
    grouped = _group_curve_segments(rows)
    assert len(grouped) == 2
    assert grouped[0]["segment_count"] == 2
    assert [s["points_data"] for s in grouped[0]["segments"]] == [rows[0]["points_data"], rows[1]["points_data"]]
    path = tmp_path / "series.json"
    path.write_text(json.dumps({"panel_id": "p", "series": grouped}))
    csv_path = tmp_path / "series.csv"
    export_curve_series_csv(path, csv_path)
    with csv_path.open() as file:
        exported = list(csv.DictReader(file))
    assert [row["trace_id"] for row in exported] == ["a", "a", "b", "b", "c", "c"]
    assert len(exported) == sum(len(row["points_data"]) for row in rows)


def test_evaluation_does_not_interpolate_across_segment_gap():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("segment_evaluation", Path(__file__).resolve().parents[2] / "evaluate.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    series = {"series": [{"legend_label": "A", "axis_binding": "left_y", "segments": [
        {"points_data": [[0, 10], [1, 11]]}, {"points_data": [[5, 15], [6, 16]]}]}]}
    prediction = evaluation.load_pred({"series_data": series})
    sampled = evaluation.resample_dense(prediction, {"A": {"0.5": 10.5, "3": 13, "5.5": 15.5}})
    assert sampled == {"A": {"0.5": 10.5, "5.5": 15.5}}

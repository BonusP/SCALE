from pathlib import Path
import json
from typing import cast

from labeling_ui.lib.gt_data import (
    build_merged_panel_list,
    compute_iou,
    load_gt_plots,
    load_localize_report,
    merge_panel_data,
)


def test_load_localize_report_returns_dict(tmp_path: Path) -> None:
    report_path = tmp_path / "phase2_localize_report.json"
    report_data = {
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "0001_F02-P01_chart",
                "panel_path": "panels/0001/0001_F02-P01_chart.png",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 70, 260, 152]},
            }
        ],
    }
    _ = report_path.write_text(json.dumps(report_data), encoding="utf-8")

    result = load_localize_report(report_path)
    panels = cast(list[dict[str, object]], result["panels"])

    assert result["generated_at_utc"] == "2026-02-27T00:00:00Z"
    assert len(panels) == 1
    assert panels[0]["panel_id"] == "0001_F02-P01_chart"


def test_load_gt_plots_returns_default_on_missing(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt_plots.json"

    result = load_gt_plots(gt_path)
    panels = cast(list[dict[str, object]], result["panels"])

    assert result["schema_version"] == 1
    assert panels == []


def test_load_gt_plots_returns_existing_data(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt_plots.json"
    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "0001_F02-P01_chart",
                "panel_path": "panels/0001/0001_F02-P01_chart.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [22, 68, 258, 150],
                "annotator": "alice",
                "note": "axis ok",
            }
        ],
    }
    _ = gt_path.write_text(json.dumps(gt_data), encoding="utf-8")

    result = load_gt_plots(gt_path)
    panels = cast(list[dict[str, object]], result["panels"])

    assert result["schema_version"] == 1
    assert len(panels) == 1


def test_merge_panel_data_combines_pred_and_gt() -> None:
    pred_panel: dict[str, object] = {
        "panel_id": "0001_F02-P01_chart",
        "panel_path": "panels/0001/0001_F02-P01_chart.png",
        "panel_kind": "chart",
        "decision": "ACCEPT",
        "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 70, 260, 152]},
    }
    gt_panel: dict[str, object] = {
        "panel_id": "0001_F02-P01_chart",
        "panel_path": "panels/0001/0001_F02-P01_chart.png",
        "panel_kind": "chart",
        "review_status": "accept",
        "gt_bbox_xyxy": [22, 68, 258, 150],
        "annotator": "alice",
        "note": "axis ok",
    }

    result = merge_panel_data(pred_panel, gt_panel)

    assert result["panel_id"] == "0001_F02-P01_chart"
    assert result["panel_path"] == "panels/0001/0001_F02-P01_chart.png"
    assert result["panel_kind"] == "chart"
    assert result["review_status"] == "accept"
    assert result["gt_bbox_xyxy"] == [22, 68, 258, 150]
    assert result["pred_bbox_xyxy"] == [20, 70, 260, 152]
    assert result["annotator"] == "alice"
    assert result["note"] == "axis ok"


def test_merge_panel_data_handles_missing_gt() -> None:
    pred_panel: dict[str, object] = {
        "panel_id": "0001_F02-P01_chart",
        "panel_path": "panels/0001/0001_F02-P01_chart.png",
        "panel_kind": "chart",
        "decision": "ACCEPT",
        "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 70, 260, 152]},
    }

    result = merge_panel_data(pred_panel, None)

    assert result["panel_id"] == "0001_F02-P01_chart"
    assert result["pred_bbox_xyxy"] == [20, 70, 260, 152]
    assert result["gt_bbox_xyxy"] is None
    assert result["review_status"] is None


def test_build_merged_panel_list(tmp_path: Path) -> None:
    report_path = tmp_path / "phase2_localize_report.json"
    report_data = {
        "panels": [
            {
                "panel_id": "0001_F02-P01_chart",
                "panel_path": "panels/0001/0001_F02-P01_chart.png",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 70, 260, 152]},
            },
            {
                "panel_id": "0001_F03-P01_chart",
                "panel_path": "panels/0001/0001_F03-P01_chart.png",
                "panel_kind": "chart",
                "decision": "REJECT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [10, 10, 100, 100]},
            },
        ]
    }
    _ = report_path.write_text(json.dumps(report_data), encoding="utf-8")

    gt_path = tmp_path / "gt_plots.json"
    gt_data = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "0001_F02-P01_chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [22, 68, 258, 150],
                "annotator": "alice",
                "note": "ok",
            }
        ],
    }
    _ = gt_path.write_text(json.dumps(gt_data), encoding="utf-8")

    result = build_merged_panel_list(report_path, gt_path)

    assert len(result) == 2
    assert result[0]["panel_id"] == "0001_F02-P01_chart"
    assert result[0]["gt_bbox_xyxy"] == [22, 68, 258, 150]
    assert result[0]["review_status"] == "accept"
    assert result[1]["panel_id"] == "0001_F03-P01_chart"
    assert result[1]["gt_bbox_xyxy"] is None


def test_build_merged_panel_list_marks_orphan_figure(tmp_path: Path) -> None:
    links_dir = tmp_path / "links"
    links_dir.mkdir(parents=True, exist_ok=True)

    report_path = tmp_path / "phase2_localize_report.json"
    report_data = {
        "panels": [
            {
                "panel_id": "0001_F01-P01_chart",
                "figure_id": "0001_F01",
                "panel_path": "panels/0001/0001_F01-P01_chart.png",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 70, 260, 152]},
            },
            {
                "panel_id": "0001_F99-P01_chart",
                "figure_id": "0001_F99",
                "panel_path": "panels/0001/0001_F99-P01_chart.png",
                "panel_kind": "chart",
                "decision": "WARN",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [10, 10, 100, 100]},
            },
        ]
    }
    _ = report_path.write_text(json.dumps(report_data), encoding="utf-8")

    gt_path = links_dir / "gt_plots.json"
    _ = gt_path.write_text(json.dumps({"schema_version": 1, "panels": []}), encoding="utf-8")

    figure_index_path = links_dir / "figure_index.json"
    _ = figure_index_path.write_text(
        json.dumps(
            {
                "0001_F01": {
                    "figure_id": "0001_F01",
                    "paper_id": "0001",
                    "figure_path": "figures/0001/0001_F01__src.png",
                    "figure_class": None,
                }
            }
        ),
        encoding="utf-8",
    )

    result = build_merged_panel_list(report_path, gt_path)

    by_id = {item["panel_id"]: item for item in result}
    assert by_id["0001_F01-P01_chart"]["is_orphan_figure"] is False
    assert by_id["0001_F99-P01_chart"]["is_orphan_figure"] is True


def test_compute_iou_returns_correct_value() -> None:
    box_a = [0, 0, 100, 100]
    box_b = [50, 50, 150, 150]

    result = compute_iou(box_a, box_b)

    assert abs(result - 0.14285714285714285) < 0.0001


def test_compute_iou_no_overlap() -> None:
    box_a = [0, 0, 50, 50]
    box_b = [100, 100, 150, 150]

    result = compute_iou(box_a, box_b)

    assert result == 0.0


def test_compute_iou_identical() -> None:
    box_a = [10, 20, 100, 200]
    box_b = [10, 20, 100, 200]

    result = compute_iou(box_a, box_b)

    assert result == 1.0

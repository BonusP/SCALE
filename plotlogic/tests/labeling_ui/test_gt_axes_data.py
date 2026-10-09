import json
from pathlib import Path

from labeling_ui.lib.gt_axes_data import build_merged_axis_panel_list


def test_build_merged_axis_panel_list_normalizes_literal_pending_to_none(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "phase2_gt_axes_seed_v1.json"
    gt_path = tmp_path / "gt_axes.json"

    seed_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-05T16:35:20Z",
                "summary": {"seed_rows_total": 1},
                "panels": [
                    {
                        "panel_id": "0001_F01-P04_chart",
                        "panel_path": "panels/0001/0001_F01-P04_chart.png",
                        "panel_kind": "chart",
                        "is_dual_y": False,
                        "is_dual_x": False,
                        "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
                        "axis_left_bbox_xyxy": [8, 10, 18, 100],
                        "axis_bottom_bbox_xyxy": [10, 92, 100, 102],
                        "axis_right_bbox_xyxy": None,
                        "review_status": "pending",
                        "note": "seed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    gt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "panels": [
                    {
                        "panel_id": "0001_F01-P04_chart",
                        "panel_path": "panels/0001/0001_F01-P04_chart.png",
                        "panel_kind": "chart",
                        "is_dual_y": False,
                        "is_dual_x": True,
                        "review_status": "pending",
                        "axis_left_bbox_xyxy": None,
                        "axis_bottom_bbox_xyxy": None,
                        "axis_right_bbox_xyxy": None,
                        "annotator": "",
                        "note": "needs review",
                        "created_at_utc": "2026-03-16T08:20:35Z",
                        "updated_at_utc": "2026-03-16T08:20:35Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = build_merged_axis_panel_list(seed_path, gt_path)

    assert len(rows) == 1
    assert rows[0]["review_status"] is None
    assert rows[0]["is_dual_y"] is False
    assert rows[0]["is_dual_x"] is True


def test_build_merged_axis_panel_list_synthesizes_right_axis_for_dual_y_pending(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "phase2_gt_axes_seed_v1.json"
    gt_path = tmp_path / "gt_axes.json"

    seed_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-05T16:35:20Z",
                "summary": {"seed_rows_total": 1},
                "panels": [
                    {
                        "panel_id": "0001_F01-P05_chart",
                        "panel_path": "panels/0001/0001_F01-P05_chart.png",
                        "panel_kind": "chart",
                        "is_dual_y": False,
                        "image_size_wh": [400, 300],
                        "source_pred_plot_frame_bbox_xyxy": [80, 40, 320, 260],
                        "axis_left_bbox_xyxy": [36, 31, 90, 267],
                        "axis_bottom_bbox_xyxy": [75, 251, 327, 299],
                        "axis_right_bbox_xyxy": None,
                        "review_status": "pending",
                        "note": "seed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    gt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "panels": [
                    {
                        "panel_id": "0001_F01-P05_chart",
                        "panel_path": "panels/0001/0001_F01-P05_chart.png",
                        "panel_kind": "chart",
                        "is_dual_y": True,
                        "review_status": "pending",
                        "axis_left_bbox_xyxy": None,
                        "axis_bottom_bbox_xyxy": None,
                        "axis_right_bbox_xyxy": None,
                        "annotator": "",
                        "note": "double y",
                        "created_at_utc": "2026-03-16T08:20:35Z",
                        "updated_at_utc": "2026-03-16T08:20:35Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = build_merged_axis_panel_list(seed_path, gt_path)

    assert len(rows) == 1
    assert rows[0]["review_status"] is None
    assert rows[0]["is_dual_y"] is True
    assert rows[0]["axis_left_bbox_xyxy"] == [36, 31, 90, 267]
    assert rows[0]["axis_bottom_bbox_xyxy"] == [75, 251, 327, 299]
    assert rows[0]["seed_axis_right_bbox_xyxy"] == [310, 31, 363, 267]
    assert rows[0]["axis_right_bbox_xyxy"] == [310, 31, 363, 267]

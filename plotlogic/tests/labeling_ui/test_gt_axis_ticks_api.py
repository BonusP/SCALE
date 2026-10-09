import json
from pathlib import Path

def test_get_gt_axis_ticks_returns_payload(tmp_path: Path) -> None:
    from labeling_ui.routes import gt_axis_ticks

    ticks_path = tmp_path / "gt_axis_ticks.json"
    ticks_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at_utc": "2026-03-21T00:00:00Z",
                "source": "accepted-axis-bbox-tick-detector-v2",
                "summary": {"accepted_panels_total": 1, "panels_total": 1, "axes_total": 2},
                "panels": [
                    {
                        "panel_id": "p1_F01-P01_chart",
                        "panel_path": "panels/p1/p1_F01-P01_chart.png",
                        "panel_kind": "chart",
                        "axes": [
                            {
                                "axis_role": "left",
                                "axis_bbox_xyxy": [10, 10, 20, 80],
                                "search_bbox_xyxy": [0, 10, 20, 80],
                                "axis_line_coordinate_px": 18,
                                "tick_positions_px": [20, 40, 60],
                                "tick_count": 3,
                                "tick_label_candidates": [],
                                "tick_matches": [],
                                "issues": ["OCR_EMPTY"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    gt_axis_ticks.configure(get_gt_axis_ticks_path=lambda: ticks_path)

    data = gt_axis_ticks.get_gt_axis_ticks()
    assert data["schema_version"] == 2
    assert data["summary"]["panels_total"] == 1
    assert data["panels"][0]["panel_id"] == "p1_F01-P01_chart"
    assert data["panels"][0]["axes"][0]["search_bbox_xyxy"] == [0, 10, 20, 80]
    assert data["panels"][0]["axes"][0]["tick_matches"] == []
    assert data["panels"][0]["axes"][0]["axis_line_coordinate_px"] == 18
    assert data["panels"][0]["axes"][0]["issues"] == ["OCR_EMPTY"]

import json
from pathlib import Path

from fastapi.testclient import TestClient

from labeling_ui.app import app


client = TestClient(app)


def _configure_tick_review_router(tmp_path: Path) -> tuple[Path, Path]:
    from labeling_ui.lib.gt_axis_tick_review_data import (
        build_merged_tick_review_panel_list,
        load_gt_axis_tick_reviews,
    )
    from labeling_ui.lib.jsonio import append_jsonl_line, write_json_atomic
    from labeling_ui.routes import gt_axis_tick_reviews

    ticks_path = tmp_path / "gt_axis_ticks.json"
    reviews_path = tmp_path / "gt_axis_tick_reviews.json"
    changelog_path = tmp_path / "gt_axis_tick_review_changelog_v1.jsonl"

    ticks_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at_utc": "2026-04-10T00:00:00Z",
                "source": "accepted-axis-bbox-tick-detector-v2",
                "summary": {"accepted_panels_total": 1, "panels_total": 1, "axes_total": 2},
                "panels": [
                    {
                        "panel_id": "0001_F01-P01_chart",
                        "panel_path": "panels/0001/0001_F01-P01_chart.png",
                        "panel_kind": "chart",
                        "axes": [
                            {
                                "axis_role": "left",
                                "axis_bbox_xyxy": [8, 10, 18, 100],
                                "search_bbox_xyxy": [0, 10, 18, 100],
                                "axis_line_coordinate_px": 16,
                                "tick_positions_px": [24, 44, 64],
                                "tick_count": 3,
                                "tick_label_candidates": [
                                    {
                                        "text": "30",
                                        "normalized_text": "30",
                                        "numeric_value": 30.0,
                                        "bbox_xyxy": [0, 18, 14, 28],
                                        "ocr_source": "codex-exec",
                                        "ocr_confidence": 0.92,
                                    }
                                ],
                                "tick_matches": [
                                    {
                                        "tick_position_px": 24,
                                        "candidate_bbox_xyxy": [0, 18, 14, 28],
                                    }
                                ],
                                "issues": [],
                            },
                            {
                                "axis_role": "bottom",
                                "axis_bbox_xyxy": [10, 92, 100, 102],
                                "search_bbox_xyxy": [10, 92, 100, 120],
                                "axis_line_coordinate_px": 96,
                                "tick_positions_px": [20, 40],
                                "tick_count": 2,
                                "tick_label_candidates": [],
                                "tick_matches": [],
                                "issues": ["ZERO_MATCHES"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reviews_path.write_text(
        json.dumps({"schema_version": 1, "panels": []}),
        encoding="utf-8",
    )

    gt_axis_tick_reviews.configure(
        get_gt_axis_ticks_path=lambda: ticks_path,
        get_gt_axis_tick_reviews_path=lambda: reviews_path,
        get_gt_axis_tick_review_changelog_path=lambda: changelog_path,
        load_gt_axis_tick_reviews=load_gt_axis_tick_reviews,
        build_merged_tick_review_panel_list=build_merged_tick_review_panel_list,
        write_json_atomic=write_json_atomic,
        append_jsonl_line=append_jsonl_line,
    )
    return reviews_path, changelog_path


def test_get_gt_axis_tick_reviews_list_returns_merged_meta(tmp_path: Path) -> None:
    _configure_tick_review_router(tmp_path)

    response = client.get("/api/gt-axis-tick-reviews")
    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["total"] == 1
    assert data["meta"]["pending"] == 1
    assert data["meta"]["defer"] == 0
    assert data["meta"]["problem_panels_total"] == 1
    assert data["panels"][0]["panel_id"] == "0001_F01-P01_chart"
    assert data["panels"][0]["review_status"] is None
    assert data["panels"][0]["work_bucket"] is None
    assert data["panels"][0]["problem_axes_count"] == 1
    assert data["panels"][0]["axes"][1]["issues"] == ["ZERO_MATCHES"]


def test_put_gt_axis_tick_review_persists_and_logs(tmp_path: Path) -> None:
    reviews_path, changelog_path = _configure_tick_review_router(tmp_path)

    payload = {
        "review_status": "accept",
        "work_bucket": "defer",
        "annotator": "alice",
        "note": "tick pairing ok",
    }
    response = client.put("/api/gt-axis-tick-reviews/0001_F01-P01_chart", json=payload)
    assert response.status_code == 200
    assert response.json()["review_status"] == "accept"

    reviews_data = json.loads(reviews_path.read_text(encoding="utf-8"))
    assert len(reviews_data["panels"]) == 1
    assert reviews_data["panels"][0]["panel_id"] == "0001_F01-P01_chart"
    assert reviews_data["panels"][0]["review_status"] == "accept"
    assert reviews_data["panels"][0]["work_bucket"] == "defer"
    assert reviews_data["panels"][0]["note"] == "tick pairing ok"

    lines = [line for line in changelog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    change = json.loads(lines[0])
    assert change["source"] == "api/gt-axis-tick-reviews"
    assert change["after"]["review_status"] == "accept"
    assert change["after"]["work_bucket"] == "defer"


def test_get_gt_axis_tick_reviews_list_uses_review_rows_when_sidecar_is_empty(
    tmp_path: Path,
) -> None:
    reviews_path, _ = _configure_tick_review_router(tmp_path)
    ticks_path = tmp_path / "gt_axis_ticks.json"

    ticks_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at_utc": "2026-04-10T00:00:00Z",
                "source": "accepted-axis-bbox-tick-detector-v2",
                "summary": {"accepted_panels_total": 0, "panels_total": 0, "axes_total": 0},
                "panels": [],
            }
        ),
        encoding="utf-8",
    )
    reviews_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": "2026-04-10T00:00:00Z",
                "updated_at_utc": "2026-04-10T00:00:00Z",
                "panels": [
                    {
                        "panel_id": "0001_F01-P99_chart",
                        "panel_path": "panels/0001/0001_F01-P99_chart.png",
                        "panel_kind": "chart",
                        "review_status": "pending",
                        "work_bucket": "defer",
                        "annotator": "",
                        "note": "",
                        "created_at_utc": "2026-04-10T00:00:00Z",
                        "updated_at_utc": "2026-04-10T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/api/gt-axis-tick-reviews")
    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["total"] == 1
    assert data["meta"]["pending"] == 1
    assert data["meta"]["defer"] == 1
    assert data["panels"][0]["panel_id"] == "0001_F01-P99_chart"
    assert data["panels"][0]["review_status"] is None
    assert data["panels"][0]["work_bucket"] == "defer"
    assert data["panels"][0]["axes"] == []

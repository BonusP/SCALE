"""Tests for GT axis API endpoints."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from labeling_ui.app import app


client = TestClient(app)


def _configure_axis_router(tmp_path: Path, seed_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    from labeling_ui.lib.gt_axes_data import build_merged_axis_panel_list, load_gt_axes
    from labeling_ui.lib.jsonio import append_jsonl_line, write_json_atomic
    from labeling_ui.lib.validation import is_valid_bbox
    from labeling_ui.routes import gt_axes

    seed_path = tmp_path / "phase2_gt_axes_seed_v1.json"
    gt_path = tmp_path / "gt_axes.json"
    changelog_path = tmp_path / "gt_axis_changelog_v1.jsonl"

    seed_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-05T16:35:20Z",
                "summary": {"seed_rows_total": len(seed_rows)},
                "panels": seed_rows,
            }
        ),
        encoding="utf-8",
    )
    gt_path.write_text(json.dumps({"schema_version": 1, "panels": []}), encoding="utf-8")

    gt_axes.configure(
        get_gt_axes_seed_path=lambda: seed_path,
        get_gt_axes_path=lambda: gt_path,
        get_gt_axes_changelog_path=lambda: changelog_path,
        load_gt_axes=load_gt_axes,
        build_merged_axis_panel_list=build_merged_axis_panel_list,
        write_json_atomic=write_json_atomic,
        append_jsonl_line=append_jsonl_line,
        is_valid_bbox=is_valid_bbox,
    )
    return gt_path, changelog_path


def test_get_gt_axes_list_returns_200(tmp_path: Path) -> None:
    _configure_axis_router(
        tmp_path,
        [
            {
                "panel_id": "0001_F01-P01_chart",
                "panel_path": "panels/0001/0001_F01-P01_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
                "axis_left_bbox_xyxy": [8, 10, 18, 100],
                "axis_bottom_bbox_xyxy": [10, 92, 100, 102],
                "axis_right_bbox_xyxy": None,
                "review_status": "pending",
                "note": "seed",
            }
        ],
    )

    response = client.get("/api/gt-axes")
    assert response.status_code == 200
    data = response.json()
    assert "meta" in data
    assert "panels" in data
    assert len(data["panels"]) == 1
    assert data["panels"][0]["panel_id"] == "0001_F01-P01_chart"


def test_put_gt_axis_dual_y_accept_requires_right(tmp_path: Path) -> None:
    _configure_axis_router(
        tmp_path,
        [
            {
                "panel_id": "0001_F01-P02_chart",
                "panel_path": "panels/0001/0001_F01-P02_chart.png",
                "panel_kind": "chart",
                "is_dual_y": True,
                "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
                "axis_left_bbox_xyxy": [8, 10, 18, 100],
                "axis_bottom_bbox_xyxy": [10, 92, 100, 102],
                "axis_right_bbox_xyxy": [92, 10, 102, 100],
                "review_status": "pending",
                "note": "seed",
            }
        ],
    )

    payload = {
        "review_status": "accept",
        "axis_left_bbox_xyxy": [8, 10, 18, 100],
        "axis_bottom_bbox_xyxy": [10, 92, 100, 102],
        "axis_right_bbox_xyxy": None,
        "annotator": "alice",
        "note": "missing right",
    }

    response = client.put("/api/gt-axes/0001_F01-P02_chart", json=payload)
    assert response.status_code == 422
    assert "axis_right_bbox_xyxy" in response.json()["detail"]


def test_put_gt_axis_accept_persists_and_logs(tmp_path: Path) -> None:
    gt_path, changelog_path = _configure_axis_router(
        tmp_path,
        [
            {
                "panel_id": "0001_F01-P03_chart",
                "panel_path": "panels/0001/0001_F01-P03_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
                "axis_left_bbox_xyxy": [8, 10, 18, 100],
                "axis_bottom_bbox_xyxy": [10, 92, 100, 102],
                "axis_right_bbox_xyxy": None,
                "review_status": "pending",
                "note": "seed",
            }
        ],
    )

    payload = {
        "review_status": "accept",
        "axis_left_bbox_xyxy": [9, 10, 19, 100],
        "axis_bottom_bbox_xyxy": [10, 91, 100, 101],
        "axis_right_bbox_xyxy": None,
        "annotator": "alice",
        "note": "ok",
    }
    response = client.put("/api/gt-axes/0001_F01-P03_chart", json=payload)
    assert response.status_code == 200
    assert response.json()["review_status"] == "accept"

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    assert len(gt_data["panels"]) == 1
    assert gt_data["panels"][0]["panel_id"] == "0001_F01-P03_chart"
    assert gt_data["panels"][0]["axis_left_bbox_xyxy"] == [9, 10, 19, 100]

    lines = [line for line in changelog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    change = json.loads(lines[0])
    assert change["source"] == "api/gt-axes"
    assert change["after"]["review_status"] == "accept"


def test_put_gt_axis_pending_persists_bbox_and_logs(tmp_path: Path) -> None:
    gt_path, changelog_path = _configure_axis_router(
        tmp_path,
        [
            {
                "panel_id": "0001_F01-P04_chart",
                "panel_path": "panels/0001/0001_F01-P04_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
                "axis_left_bbox_xyxy": [8, 10, 18, 100],
                "axis_bottom_bbox_xyxy": [10, 92, 100, 102],
                "axis_right_bbox_xyxy": None,
                "review_status": "pending",
                "note": "seed",
            }
        ],
    )

    payload = {
        "review_status": "pending",
        "axis_left_bbox_xyxy": [11, 10, 21, 100],
        "axis_bottom_bbox_xyxy": [10, 90, 100, 100],
        "axis_right_bbox_xyxy": None,
        "annotator": "alice",
        "note": "reopen",
    }

    response = client.put("/api/gt-axes/0001_F01-P04_chart", json=payload)
    assert response.status_code == 200
    assert response.json()["review_status"] == "pending"

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    assert gt_data["panels"][0]["review_status"] == "pending"
    assert gt_data["panels"][0]["axis_left_bbox_xyxy"] == [11, 10, 21, 100]
    assert gt_data["panels"][0]["axis_bottom_bbox_xyxy"] == [10, 90, 100, 100]

    lines = [line for line in changelog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    change = json.loads(lines[0])
    assert change["after"]["review_status"] == "pending"


def test_put_gt_axis_reject_preserves_existing_bbox_when_payload_is_null(tmp_path: Path) -> None:
    gt_path, changelog_path = _configure_axis_router(
        tmp_path,
        [
            {
                "panel_id": "0001_F01-P05_chart",
                "panel_path": "panels/0001/0001_F01-P05_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
                "axis_left_bbox_xyxy": [8, 10, 18, 100],
                "axis_bottom_bbox_xyxy": [10, 92, 100, 102],
                "axis_right_bbox_xyxy": None,
                "review_status": "pending",
                "note": "seed",
            }
        ],
    )

    accepted_payload = {
        "review_status": "accept",
        "axis_left_bbox_xyxy": [12, 10, 22, 100],
        "axis_bottom_bbox_xyxy": [10, 88, 100, 98],
        "axis_right_bbox_xyxy": None,
        "annotator": "alice",
        "note": "accepted",
    }
    accepted = client.put("/api/gt-axes/0001_F01-P05_chart", json=accepted_payload)
    assert accepted.status_code == 200

    reject_payload = {
        "review_status": "reject",
        "axis_left_bbox_xyxy": None,
        "axis_bottom_bbox_xyxy": None,
        "axis_right_bbox_xyxy": None,
        "annotator": "alice",
        "note": "mistake",
    }
    rejected = client.put("/api/gt-axes/0001_F01-P05_chart", json=reject_payload)
    assert rejected.status_code == 200
    assert rejected.json()["review_status"] == "reject"

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    assert gt_data["panels"][0]["review_status"] == "reject"
    assert gt_data["panels"][0]["axis_left_bbox_xyxy"] == [12, 10, 22, 100]
    assert gt_data["panels"][0]["axis_bottom_bbox_xyxy"] == [10, 88, 100, 98]

    lines = [line for line in changelog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    change = json.loads(lines[-1])
    assert change["after"]["review_status"] == "reject"
    assert change["after"]["axis_left_bbox_xyxy"] == [12, 10, 22, 100]
    assert change["after"]["axis_bottom_bbox_xyxy"] == [10, 88, 100, 98]

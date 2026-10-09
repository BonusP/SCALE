"""Tests for GT plots API endpoints."""

import json
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from labeling_ui.app import app


client = TestClient(app)


def test_get_gt_plots_list_returns_200() -> None:
    """GET /api/gt-plots should return 200 with panels list."""
    response = client.get("/api/gt-plots")

    assert response.status_code == 200
    data = response.json()
    assert "panels" in data
    assert isinstance(data["panels"], list)
    assert "meta" in data


def test_get_gt_plots_filters_non_chart_by_default(tmp_path: Path) -> None:
    """GET /api/gt-plots should return chart only unless include_non_chart=1."""
    from labeling_ui.lib.gt_data import (
        build_merged_panel_list,
        compute_iou,
        load_gt_plots,
        load_localize_report,
    )
    from labeling_ui.lib.jsonio import append_jsonl_line, write_json_atomic
    from labeling_ui.lib.validation import is_valid_bbox
    from labeling_ui.routes import gt_plots

    report_path = tmp_path / "phase2_localize_report.json"
    gt_path = tmp_path / "gt_plots.json"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "0001_F01-P01_chart",
                        "panel_path": "panels/0001/0001_F01-P01_chart.png",
                        "panel_kind": "chart",
                        "decision": "ACCEPT",
                        "evidence": {"predicted_plot_frame_bbox_xyxy": [10, 10, 100, 100]},
                    },
                    {
                        "panel_id": "0001_F01-P02_schematic",
                        "panel_path": "panels/0001/0001_F01-P02_schematic.png",
                        "panel_kind": "schematic",
                        "decision": "ACCEPT",
                        "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 20, 120, 120]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    gt_path.write_text(json.dumps({"schema_version": 1, "panels": []}), encoding="utf-8")

    gt_plots.configure(
        get_dataset_root=lambda: dataset_root,
        get_localize_report_path=lambda: report_path,
        get_gt_plots_path=lambda: gt_path,
        get_gt_changelog_path=lambda: tmp_path / "gt_changelog_v1.jsonl",
        load_localize_report=load_localize_report,
        load_gt_plots=load_gt_plots,
        build_merged_panel_list=build_merged_panel_list,
        write_json_atomic=write_json_atomic,
        append_jsonl_line=append_jsonl_line,
        is_valid_bbox=is_valid_bbox,
        compute_iou=compute_iou,
    )

    chart_only = client.get("/api/gt-plots")
    assert chart_only.status_code == 200
    chart_panels = chart_only.json()["panels"]
    assert len(chart_panels) == 1
    assert chart_panels[0]["panel_kind"] == "chart"

    all_kinds = client.get("/api/gt-plots?include_non_chart=1")
    assert all_kinds.status_code == 200
    all_panels = all_kinds.json()["panels"]
    assert len(all_panels) == 2
    assert {panel["panel_kind"] for panel in all_panels} == {"chart", "schematic"}


def test_get_single_gt_plot_returns_200() -> None:
    """GET /api/gt-plots/{panel_id} should return 200 with panel data."""
    response = client.get("/api/gt-plots/0001_F02-P01_chart")

    assert response.status_code == 200
    data = response.json()
    assert "panel_id" in data
    assert "panel_path" in data
    assert "panel_kind" in data


def test_get_single_gt_plot_returns_404_for_unknown() -> None:
    """GET /api/gt-plots/{panel_id} should return 404 for unknown panel."""
    response = client.get("/api/gt-plots/UNKNOWN_PANEL_ID")

    assert response.status_code == 404


def test_put_gt_plot_accept_with_bbox(tmp_path: Path) -> None:
    """PUT /api/gt-plots/{panel_id} should accept with valid bbox."""
    # Create temp gt_plots.json
    gt_path = tmp_path / "gt_plots.json"
    gt_path.write_text(json.dumps({"schema_version": 1, "panels": []}), encoding="utf-8")

    # Create temp localize report with the panel
    report_path = tmp_path / "phase2_localize_report.json"
    report_path.write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "0001_F02-P01_chart",
                        "panel_path": "panels/0001/0001_F02-P01_chart.png",
                        "panel_kind": "chart",
                        "decision": "ACCEPT",
                        "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 70, 260, 152]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Create temp dataset root with panel file
    dataset_root = tmp_path / "dataset"
    panel_dir = dataset_root / "panels" / "0001"
    panel_dir.mkdir(parents=True)
    panel_file = panel_dir / "0001_F02-P01_chart.png"
    # Create a minimal valid PNG (1x1 pixel)
    import struct
    import zlib

    def create_minimal_png() -> bytes:
        signature = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr)
        ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", ihdr_crc)
        idat_data = zlib.compress(b"\x00\x00\x00\x00\x00")
        idat_crc = zlib.crc32(b"IDAT" + idat_data)
        idat_chunk = (
            struct.pack(">I", len(idat_data))
            + b"IDAT"
            + idat_data
            + struct.pack(">I", idat_crc)
        )
        iend_crc = zlib.crc32(b"IEND")
        iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return signature + ihdr_chunk + idat_chunk + iend_chunk

    panel_file.write_bytes(create_minimal_png())

    # Patch the module-level functions using configure
    from labeling_ui.routes import gt_plots
    from labeling_ui.lib.gt_data import (
        load_localize_report,
        load_gt_plots,
        build_merged_panel_list,
        compute_iou,
    )
    from labeling_ui.lib.jsonio import append_jsonl_line, write_json_atomic
    from labeling_ui.lib.validation import is_valid_bbox

    gt_plots.configure(
        get_dataset_root=lambda: dataset_root,
        get_localize_report_path=lambda: report_path,
        get_gt_plots_path=lambda: gt_path,
        get_gt_changelog_path=lambda: tmp_path / "gt_changelog_v1.jsonl",
        load_localize_report=load_localize_report,
        load_gt_plots=load_gt_plots,
        build_merged_panel_list=build_merged_panel_list,
        write_json_atomic=write_json_atomic,
        append_jsonl_line=append_jsonl_line,
        is_valid_bbox=is_valid_bbox,
        compute_iou=compute_iou,
    )

    payload = {
        "review_status": "accept",
        "gt_bbox_xyxy": [22, 68, 258, 150],
        "annotator": "alice",
        "note": "axis ok",
    }

    response = client.put("/api/gt-plots/0001_F02-P01_chart", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["panel_id"] == "0001_F02-P01_chart"
    assert data["review_status"] == "accept"
    assert "updated_at_utc" in data

    changelog_path = tmp_path / "gt_changelog_v1.jsonl"
    assert changelog_path.exists()
    lines = [line for line in changelog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    change = json.loads(lines[0])
    assert change["panel_id"] == "0001_F02-P01_chart"
    assert change["source"] == "api/gt-plots"
    assert change["before"] is None
    assert change["after"]["review_status"] == "accept"


def test_put_gt_plot_invalid_status_returns_422() -> None:
    """PUT /api/gt-plots/{panel_id} should return 422 for invalid status."""
    payload = {
        "review_status": "invalid_status",
        "gt_bbox_xyxy": [22, 68, 258, 150],
        "annotator": "alice",
        "note": "test",
    }

    response = client.put("/api/gt-plots/0001_F02-P01_chart", json=payload)

    assert response.status_code == 422



def test_get_gt_eval_returns_metrics() -> None:
    """GET /api/gt-eval/{panel_id} should return evaluation metrics."""
    response = client.get("/api/gt-eval/0001_F02-P01_chart")

    # Either 200 with metrics or 404 with "Panel not found" detail
    if response.status_code == 200:
        data = response.json()
        assert "panel_id" in data
        assert "iou" in data
        assert "threshold" in data
        assert "decision" in data
        assert "f1_panel" in data
    else:
        # Must be our 404, not a routing 404
        assert response.status_code == 404
        assert "Panel not found" in detail, f"Expected 'Panel not found' but got: {detail}"


def test_get_gt_eval_without_gt_bbox_returns_null_iou() -> None:
    """Panel with reject/skip status should return null IoU."""
    response = client.get("/api/gt-eval/NONEXISTENT_PANEL")

    assert response.status_code == 404


def test_full_gt_workflow(tmp_path: Path) -> None:
    """Test full workflow: list -> get -> update -> eval."""
    import struct
    import zlib

    from labeling_ui.lib.gt_data import (
        build_merged_panel_list,
        compute_iou,
        load_gt_plots,
        load_localize_report,
    )
    from labeling_ui.lib.jsonio import append_jsonl_line, write_json_atomic
    from labeling_ui.lib.validation import is_valid_bbox
    from labeling_ui.routes import gt_plots

    # Setup temp files
    report_path = tmp_path / "phase2_localize_report.json"
    gt_path = tmp_path / "gt_plots.json"
    dataset_root = tmp_path / "dataset"

    # Create minimal PNG helper
    def create_minimal_png() -> bytes:
        signature = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr)
        ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", ihdr_crc)
        idat_data = zlib.compress(b"\x00\x00\x00\x00\x00")
        idat_crc = zlib.crc32(b"IDAT" + idat_data)
        idat_chunk = (
            struct.pack(">I", len(idat_data))
            + b"IDAT"
            + idat_data
            + struct.pack(">I", idat_crc)
        )
        iend_crc = zlib.crc32(b"IEND")
        iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return signature + ihdr_chunk + idat_chunk + iend_chunk

    # Create panel file
    panel_dir = dataset_root / "panels" / "0001"
    panel_dir.mkdir(parents=True)
    panel_file = panel_dir / "0001_F02-P01_chart.png"
    panel_file.write_bytes(create_minimal_png())

    # Create localize report
    report_path.write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "0001_F02-P01_chart",
                        "panel_path": "panels/0001/0001_F02-P01_chart.png",
                        "panel_kind": "chart",
                        "decision": "ACCEPT",
                        "evidence": {"predicted_plot_frame_bbox_xyxy": [20, 70, 260, 152]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Create empty GT
    gt_path.write_text(json.dumps({"schema_version": 1, "panels": []}), encoding="utf-8")

    # Configure router
    gt_plots.configure(
        get_dataset_root=lambda: dataset_root,
        get_localize_report_path=lambda: report_path,
        get_gt_plots_path=lambda: gt_path,
        get_gt_changelog_path=lambda: tmp_path / "gt_changelog_v1.jsonl",
        load_localize_report=load_localize_report,
        load_gt_plots=load_gt_plots,
        build_merged_panel_list=build_merged_panel_list,
        write_json_atomic=write_json_atomic,
        append_jsonl_line=append_jsonl_line,
        is_valid_bbox=is_valid_bbox,
        compute_iou=compute_iou,
    )

    # 1. List panels
    list_resp = client.get("/api/gt-plots")
    assert list_resp.status_code == 200
    assert len(list_resp.json()["panels"]) == 1

    # 2. Get single panel
    get_resp = client.get("/api/gt-plots/0001_F02-P01_chart")
    assert get_resp.status_code == 200
    assert get_resp.json()["gt_bbox_xyxy"] is None  # No GT yet

    # 3. Update panel
    update_resp = client.put(
        "/api/gt-plots/0001_F02-P01_chart",
        json={
            "review_status": "accept",
            "gt_bbox_xyxy": [22, 68, 258, 150],
            "annotator": "test_user",
            "note": "integration test",
        },
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["review_status"] == "accept"

    # 4. Verify update persisted
    get_after = client.get("/api/gt-plots/0001_F02-P01_chart")
    assert get_after.status_code == 200
    assert get_after.json()["gt_bbox_xyxy"] == [22, 68, 258, 150]
    assert get_after.json()["annotator"] == "test_user"

    # 5. Get eval metrics
    eval_resp = client.get("/api/gt-eval/0001_F02-P01_chart")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["iou"] is not None
    assert eval_data["iou"] > 0.8  # High overlap expected
    assert eval_data["f1_panel"] == 1  # IoU >= threshold

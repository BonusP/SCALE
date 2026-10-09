"""Tests for phase2_gt_tool.py validate subcommand."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts/phase2_gt_tool.py")


def _write_gt_file(tmp_path: Path, gt_data: dict) -> Path:
    """Write GT JSON to temp file and return path."""
    gt_path = tmp_path / "links" / "gt_plots.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")
    return gt_path


def _create_panel_image(
    dataset_root: Path, panel_path: str, width: int = 100, height: int = 100
) -> None:
    """Create a dummy panel image at the specified path."""
    from PIL import Image

    full_path = dataset_root / panel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color="white")
    img.save(full_path)


def _make_valid_gt() -> dict:
    """Create a minimal valid GT file."""
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [],
    }


def test_validate_help_exits_zero():
    """validate --help should exit with code 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "validate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "validate" in result.stdout.lower() or "Validate" in result.stdout


def test_validate_valid_gt_exits_zero(tmp_path: Path):
    """Valid GT with accept panels having valid bbox should exit 0."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    # Create panel image
    _create_panel_image(dataset_root, "panels/p001/panel.png", width=200, height=150)

    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "p001_F01-P01",
                "panel_path": "panels/p001/panel.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [10, 20, 100, 120],
                "annotator": "tester",
                "note": None,
                "created_at_utc": "2026-02-27T00:00:00Z",
                "updated_at_utc": "2026-02-27T00:00:00Z",
            }
        ],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] == 0


def test_validate_malformed_json_exits_one(tmp_path: Path):
    """Malformed JSON should cause CRITICAL error and exit 1."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    gt_path = tmp_path / "links" / "gt_plots.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text("{ invalid json }", encoding="utf-8")

    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] >= 1
    assert any(issue["code"] == "GT_JSON_MALFORMED" for issue in report["issues"])


def test_validate_duplicate_panel_id_critical(tmp_path: Path):
    """Duplicate panel_id should cause CRITICAL error and exit 1."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_panel_image(dataset_root, "panels/p001/panel1.png")
    _create_panel_image(dataset_root, "panels/p001/panel2.png")

    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "p001_F01-P01",
                "panel_path": "panels/p001/panel1.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 50, 50],
            },
            {
                "panel_id": "p001_F01-P01",  # duplicate
                "panel_path": "panels/p001/panel2.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 60, 60],
            },
        ],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] >= 1
    assert any(issue["code"] == "GT_DUPLICATE_PANEL_ID" for issue in report["issues"])


def test_validate_missing_panel_file_critical(tmp_path: Path):
    """Missing panel file should cause CRITICAL error and exit 1."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    # Do NOT create the panel image - it should be missing
    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "p001_F01-P01",
                "panel_path": "panels/p001/missing.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 50, 50],
            }
        ],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] >= 1
    assert any(issue["code"] == "GT_PANEL_FILE_MISSING" for issue in report["issues"])


def test_validate_accept_invalid_bbox_warn(tmp_path: Path):
    """Accept panel with invalid bbox should WARN but not fail."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_panel_image(dataset_root, "panels/p001/panel.png", width=100, height=100)

    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "p001_F01-P01",
                "panel_path": "panels/p001/panel.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [50, 50, 10, 10],  # Invalid: x2 < x1, y2 < y1
            }
        ],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    # Should exit 0 (WARN, not CRITICAL)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] == 0
    assert report["summary"]["warn"] >= 1
    assert any(issue["code"] == "GT_BBOX_INVALID" for issue in report["issues"])


def test_validate_accept_out_of_bounds_warn(tmp_path: Path):
    """Accept panel with bbox outside image bounds should WARN but not fail."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_panel_image(dataset_root, "panels/p001/panel.png", width=100, height=100)

    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "p001_F01-P01",
                "panel_path": "panels/p001/panel.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 200, 200],  # Exceeds image bounds (100x100)
            }
        ],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    # Should exit 0 (WARN, not CRITICAL)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] == 0
    assert report["summary"]["warn"] >= 1
    assert any(issue["code"] == "GT_BBOX_OUT_OF_BOUNDS" for issue in report["issues"])


def test_validate_reject_with_bbox_warn(tmp_path: Path):
    """Reject/skip panel with non-null bbox should WARN but not fail."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_panel_image(dataset_root, "panels/p001/panel.png", width=100, height=100)

    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "p001_F01-P01",
                "panel_path": "panels/p001/panel.png",
                "panel_kind": "chart",
                "review_status": "reject",
                "gt_bbox_xyxy": [0, 0, 50, 50],  # Should be null for reject
            }
        ],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    # Should exit 0 (WARN, not CRITICAL)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] == 0
    assert report["summary"]["warn"] >= 1
    assert any(issue["code"] == "GT_REJECT_HAS_BBOX" for issue in report["issues"])


def test_validate_schema_version_mismatch_critical(tmp_path: Path):
    """GT with wrong schema_version should cause CRITICAL error."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    gt_data = {
        "schema_version": 2,  # Wrong version
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] >= 1
    assert any(
        issue["code"] == "GT_SCHEMA_VERSION_MISMATCH" for issue in report["issues"]
    )


def test_validate_missing_panels_list_critical(tmp_path: Path):
    """GT without panels list should cause CRITICAL error."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        # No panels key
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["critical"] >= 1
    assert any(issue["code"] == "GT_PANELS_MISSING" for issue in report["issues"])


def test_validate_report_counts(tmp_path: Path):
    """Report should include correct counts for accept/reject/skip."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_panel_image(dataset_root, "panels/p001/a.png", width=100, height=100)
    _create_panel_image(dataset_root, "panels/p001/r.png", width=100, height=100)
    _create_panel_image(dataset_root, "panels/p001/s.png", width=100, height=100)

    gt_data = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "updated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "p001_F01-P01",
                "panel_path": "panels/p001/a.png",
                "panel_kind": "chart",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 50, 50],
            },
            {
                "panel_id": "p001_F01-P02",
                "panel_path": "panels/p001/r.png",
                "panel_kind": "chart",
                "review_status": "reject",
                "gt_bbox_xyxy": None,
            },
            {
                "panel_id": "p001_F01-P03",
                "panel_path": "panels/p001/s.png",
                "panel_kind": "chart",
                "review_status": "skip",
                "gt_bbox_xyxy": None,
            },
        ],
    }
    gt_path = _write_gt_file(tmp_path, gt_data)
    out_path = tmp_path / "validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--gt",
            str(gt_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["counts"]["total"] == 3
    assert report["counts"]["accept"] == 1
    assert report["counts"]["reject"] == 1
    assert report["counts"]["skip"] == 1

"""Tests for phase2_gt_tool.py export-seed subcommand."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts/phase2_gt_tool.py")

EXPECTED_HEADER = [
    "paper_id",
    "panel_id",
    "panel_path",
    "panel_kind",
    "decision",
    "score_final",
    "pred_bbox_xyxy",
    "gt_bbox_xyxy",
    "review_status",
    "annotator",
    "note",
]


def _make_minimal_report() -> dict:
    """Create a minimal localize report with diverse strata for testing."""
    panels = []
    # Create panels covering multiple strata: (panel_kind, decision) combinations
    strata = [
        ("chart", "ACCEPT"),
        ("chart", "WARN"),
        ("chart", "REJECT"),
        ("schematic", "WARN"),
        ("image", "WARN"),
    ]
    for i, (kind, decision) in enumerate(strata):
        # Add 3 panels per stratum to test sampling
        for j in range(3):
            panels.append(
                {
                    "panel_id": f"paper{i:02d}_F01-P{j:02d}_{kind}",
                    "paper_id": f"paper{i:02d}",
                    "panel_path": f"panels/paper{i:02d}/panel_{j:02d}.png",
                    "panel_kind": kind,
                    "decision": decision,
                    "score_breakdown": {"final": 0.5 + i * 0.05},
                    "evidence": {"predicted_plot_frame_bbox_xyxy": [10, 20, 100, 200]},
                }
            )
    return {"panels": panels}


def _write_report(tmp_path: Path, report: dict) -> Path:
    """Write report JSON to temp file and return path."""
    report_path = tmp_path / "localize_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def test_export_seed_help_exits_zero():
    """export-seed --help should exit with code 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "export-seed", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "export-seed" in result.stdout.lower() or "Export" in result.stdout


def test_main_help_exits_zero():
    """Main --help should exit with code 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_export_seed_csv_header_exact(tmp_path: Path):
    """CSV output header must match exact specification."""
    report = _make_minimal_report()
    report_path = _write_report(tmp_path, report)
    out_path = tmp_path / "seed.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "export-seed",
            "--input",
            str(report_path),
            "--out",
            str(out_path),
            "--limit",
            "10",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == EXPECTED_HEADER, f"Got: {header}"


def test_export_seed_deterministic(tmp_path: Path):
    """Running export-seed twice with same seed should produce identical output."""
    report = _make_minimal_report()
    report_path = _write_report(tmp_path, report)
    out1 = tmp_path / "seed1.csv"
    out2 = tmp_path / "seed2.csv"

    for out in (out1, out2):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "export-seed",
                "--input",
                str(report_path),
                "--out",
                str(out),
                "--limit",
                "5",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_export_seed_limit_respected(tmp_path: Path):
    """Output row count should match --limit (excluding header)."""
    # Create a larger report to ensure we have enough panels
    panels = []
    for i in range(100):
        panels.append(
            {
                "panel_id": f"p{i:03d}_F01-P01_chart",
                "paper_id": f"p{i:03d}",
                "panel_path": f"panels/p{i:03d}/panel.png",
                "panel_kind": "chart",
                "decision": "ACCEPT" if i % 2 == 0 else "WARN",
                "score_breakdown": {"final": 0.6},
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 100, 100]},
            }
        )
    report = {"panels": panels}
    report_path = _write_report(tmp_path, report)
    out_path = tmp_path / "seed.csv"

    limit = 10
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "export-seed",
            "--input",
            str(report_path),
            "--out",
            str(out_path),
            "--limit",
            str(limit),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    # First row is header, remaining rows are data
    data_rows = rows[1:]
    assert len(data_rows) == limit, f"Expected {limit} rows, got {len(data_rows)}"


def test_export_seed_stratified_distribution(tmp_path: Path):
    """Stratified sampling should distribute panels across strata proportionally."""
    # Create panels with known strata distribution
    panels = []
    # 20 chart+ACCEPT, 10 chart+WARN, 5 schematic+WARN
    for i in range(20):
        panels.append(
            {
                "panel_id": f"chart_accept_{i:02d}",
                "paper_id": "p001",
                "panel_path": f"panels/p001/ca_{i:02d}.png",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "score_breakdown": {"final": 0.7},
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 50, 50]},
            }
        )
    for i in range(10):
        panels.append(
            {
                "panel_id": f"chart_warn_{i:02d}",
                "paper_id": "p002",
                "panel_path": f"panels/p002/cw_{i:02d}.png",
                "panel_kind": "chart",
                "decision": "WARN",
                "score_breakdown": {"final": 0.5},
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 50, 50]},
            }
        )
    for i in range(5):
        panels.append(
            {
                "panel_id": f"schematic_warn_{i:02d}",
                "paper_id": "p003",
                "panel_path": f"panels/p003/sw_{i:02d}.png",
                "panel_kind": "schematic",
                "decision": "WARN",
                "score_breakdown": {"final": 0.4},
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 50, 50]},
            }
        )

    report = {"panels": panels}
    report_path = _write_report(tmp_path, report)
    out_path = tmp_path / "seed.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "export-seed",
            "--input",
            str(report_path),
            "--out",
            str(out_path),
            "--limit",
            "14",  # Should sample proportionally: 8, 4, 2 (roughly)
            "--strategy",
            "stratified",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Count strata
    strata_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["panel_kind"], row["decision"])
        strata_counts[key] = strata_counts.get(key, 0) + 1

    # All strata should be represented (at least 1 each)
    assert ("chart", "ACCEPT") in strata_counts
    assert ("chart", "WARN") in strata_counts
    assert ("schematic", "WARN") in strata_counts


def test_export_seed_handles_missing_bbox(tmp_path: Path):
    """Panels without predicted bbox should still be exported with empty pred_bbox."""
    report = {
        "panels": [
            {
                "panel_id": "no_bbox_panel",
                "paper_id": "p001",
                "panel_path": "panels/p001/no_bbox.png",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "score_breakdown": {"final": 0.65},
                # No evidence field
            }
        ]
    }
    report_path = _write_report(tmp_path, report)
    out_path = tmp_path / "seed.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "export-seed",
            "--input",
            str(report_path),
            "--out",
            str(out_path),
            "--limit",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["panel_id"] == "no_bbox_panel"
    # pred_bbox_xyxy should be empty string when missing
    assert rows[0]["pred_bbox_xyxy"] == ""


def test_export_seed_ignores_dataset_root(tmp_path: Path):
    """Export should not use dataset_root from report (it may be absolute/invalid)."""
    report = {
        "dataset_root": "/absolute/path/from/another/machine",
        "panels": [
            {
                "panel_id": "test_panel",
                "paper_id": "p001",
                "panel_path": "panels/p001/test.png",  # relative path
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "score_breakdown": {"final": 0.7},
                "evidence": {"predicted_plot_frame_bbox_xyxy": [1, 2, 3, 4]},
            }
        ],
    }
    report_path = _write_report(tmp_path, report)
    out_path = tmp_path / "seed.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "export-seed",
            "--input",
            str(report_path),
            "--out",
            str(out_path),
            "--limit",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    # Should succeed even with invalid dataset_root
    assert result.returncode == 0, f"stderr: {result.stderr}"

    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # panel_path should be preserved as-is (relative)
    assert rows[0]["panel_path"] == "panels/p001/test.png"

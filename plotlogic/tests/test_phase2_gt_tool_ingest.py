"""Tests for phase2_gt_tool ingest subcommand."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import os

# Path to the script
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "phase2_gt_tool.py"
PROJECT_ROOT = Path(__file__).parent.parent


def run_tool(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the phase2_gt_tool.py script with given arguments."""
    env = os.environ.copy()
    # Add project root to PYTHONPATH for labeling_ui module
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)] + args,
        capture_output=True,
        text=True,
        env=env,
    )

@pytest.fixture
def tmp_dataset_root(tmp_path: Path) -> Path:
    """Create a temporary dataset root directory structure."""
    dataset_root = tmp_path / "dataset"
    links_dir = dataset_root / "links"
    links_dir.mkdir(parents=True)
    return dataset_root


@pytest.fixture
def sample_localize_report(tmp_path: Path) -> Path:
    """Create a sample localize report JSON file."""
    report = {
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "panels": [
            {
                "panel_id": "0001_F01-P01_chart",
                "paper_id": "0001",
                "panel_path": "panels/0001/0001_F01-P01_chart.png",
                "panel_kind": "chart",
            },
            {
                "panel_id": "0002_F01-P01_chart",
                "paper_id": "0002",
                "panel_path": "panels/0002/0002_F01-P01_chart.png",
                "panel_kind": "chart",
            },
            {
                "panel_id": "0003_F01-P01_schematic",
                "paper_id": "0003",
                "panel_path": "panels/0003/0003_F01-P01_schematic.png",
                "panel_kind": "schematic",
            },
        ],
    }
    report_path = tmp_path / "localize_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def write_csv_sheet(path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV file."""
    fieldnames = [
        "panel_id",
        "review_status",
        "gt_bbox_xyxy",
        "annotator",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestIngestHelp:
    def test_ingest_help_exits_zero(self) -> None:
        """Test that ingest --help exits with code 0."""
        result = run_tool(["ingest", "--help"])
        assert result.returncode == 0
        assert "ingest" in result.stdout.lower() or "annotated" in result.stdout.lower()


class TestIngestBasics:
    def test_ingest_creates_new_file(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test ingest creates a new gt_plots.json file."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",
                    "annotator": "alice",
                    "note": "looks good",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        assert out_path.exists()

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert len(data["panels"]) == 1
        assert data["panels"][0]["panel_id"] == "0001_F01-P01_chart"
        assert data["panels"][0]["review_status"] == "accept"
        assert data["panels"][0]["gt_bbox_xyxy"] == [10, 20, 100, 200]

    def test_ingest_upsert_updates_existing(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test ingest updates existing gt_plots.json."""
        out_path = tmp_dataset_root / "links" / "gt_plots.json"

        # Create initial file
        initial_data = {
            "schema_version": 1,
            "generated_at_utc": "2026-02-26T00:00:00Z",
            "updated_at_utc": "2026-02-26T00:00:00Z",
            "panels": [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "panel_path": "panels/0001/0001_F01-P01_chart.png",
                    "panel_kind": "chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": [10, 20, 100, 200],
                    "annotator": "alice",
                    "note": "initial",
                    "created_at_utc": "2026-02-26T00:00:00Z",
                    "updated_at_utc": "2026-02-26T00:00:00Z",
                }
            ],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(initial_data), encoding="utf-8")

        # Create new sheet with updated annotation
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[15, 25, 150, 250]",
                    "annotator": "bob",
                    "note": "updated bbox",
                },
            ],
        )

        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data["panels"]) == 1
        # Verify bbox was updated
        assert data["panels"][0]["gt_bbox_xyxy"] == [15, 25, 150, 250]
        assert data["panels"][0]["annotator"] == "bob"
        assert data["panels"][0]["note"] == "updated bbox"


class TestIngestStatusHandling:
    def test_ingest_reject_forces_bbox_null(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that reject status forces gt_bbox_xyxy to null."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "reject",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",  # Should be ignored
                    "annotator": "alice",
                    "note": "bad quality",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["panels"][0]["review_status"] == "reject"
        assert data["panels"][0]["gt_bbox_xyxy"] is None

    def test_ingest_skip_forces_bbox_null(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that skip status forces gt_bbox_xyxy to null."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "skip",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",  # Should be ignored
                    "annotator": "alice",
                    "note": "skip for now",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["panels"][0]["review_status"] == "skip"
        assert data["panels"][0]["gt_bbox_xyxy"] is None

    def test_ingest_invalid_status_skipped(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that rows with invalid status are skipped."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "invalid_status",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",
                    "annotator": "alice",
                    "note": "test",
                },
                {
                    "panel_id": "0002_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[5, 10, 50, 100]",
                    "annotator": "bob",
                    "note": "valid",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        # Only the valid row should be included
        assert len(data["panels"]) == 1
        assert data["panels"][0]["panel_id"] == "0002_F01-P01_chart"


class TestIngestDuplicates:
    def test_ingest_duplicate_panel_id_last_wins(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that duplicate panel_id rows use last-row-wins."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",
                    "annotator": "alice",
                    "note": "first",
                },
                {
                    "panel_id": "0001_F01-P01_chart",  # Duplicate
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[20, 30, 200, 300]",
                    "annotator": "bob",
                    "note": "second (should win)",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data["panels"]) == 1
        assert data["panels"][0]["gt_bbox_xyxy"] == [20, 30, 200, 300]
        assert data["panels"][0]["annotator"] == "bob"
        assert data["panels"][0]["note"] == "second (should win)"


class TestIngestBackup:
    def test_ingest_backup_created(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that backup file is created when --backup is set."""
        out_path = tmp_dataset_root / "links" / "gt_plots.json"

        # Create initial file
        initial_data = {
            "schema_version": 1,
            "generated_at_utc": "2026-02-26T00:00:00Z",
            "updated_at_utc": "2026-02-26T00:00:00Z",
            "panels": [],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(initial_data), encoding="utf-8")

        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",
                    "annotator": "alice",
                    "note": "test",
                },
            ],
        )

        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
                "--backup",  # Explicitly enable backup
            ]
        )

        assert result.returncode == 0

        # Check backup file exists
        backup_files = list(out_path.parent.glob("gt_plots.json.bak.*"))
        assert len(backup_files) == 1

        # Verify backup content is the original
        backup_data = json.loads(backup_files[0].read_text(encoding="utf-8"))
        assert backup_data["panels"] == []


class TestIngestPanelDataFromReport:
    def test_ingest_uses_panel_path_from_report(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that panel_path and panel_kind come from localize report, not CSV."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",
                    "annotator": "alice",
                    "note": "test",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        panel = data["panels"][0]
        # These should come from the localize report
        assert panel["panel_path"] == "panels/0001/0001_F01-P01_chart.png"
        assert panel["panel_kind"] == "chart"


class TestIngestStatusNormalization:
    def test_ingest_normalizes_status_to_lowercase(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that review_status is normalized to lowercase."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "ACCEPT",  # Uppercase
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",
                    "annotator": "alice",
                    "note": "test",
                },
                {
                    "panel_id": "0002_F01-P01_chart",
                    "review_status": "Reject",  # Mixed case
                    "gt_bbox_xyxy": "[0, 0, 0, 0]",
                    "annotator": "bob",
                    "note": "test",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        statuses = [p["review_status"] for p in data["panels"]]
        assert statuses == ["accept", "reject"]


class TestIngestBboxValidation:
    def test_ingest_accept_requires_valid_bbox(
        self, tmp_dataset_root: Path, sample_localize_report: Path
    ) -> None:
        """Test that accept status requires valid bbox JSON array."""
        sheet_path = sample_localize_report.parent / "review_sheet.csv"
        write_csv_sheet(
            sheet_path,
            [
                {
                    "panel_id": "0001_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[10, 20, 100, 200]",
                    "annotator": "alice",
                    "note": "valid",
                },
                {
                    "panel_id": "0002_F01-P01_chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "not valid json",  # Invalid
                    "annotator": "bob",
                    "note": "invalid json",
                },
                {
                    "panel_id": "0003_F01-P01_schematic",
                    "review_status": "accept",
                    "gt_bbox_xyxy": "[1, 2, 3]",  # Only 3 ints
                    "annotator": "charlie",
                    "note": "wrong length",
                },
            ],
        )

        out_path = tmp_dataset_root / "links" / "gt_plots.json"
        result = run_tool(
            [
                "ingest",
                "--dataset-root",
                str(tmp_dataset_root),
                "--input",
                str(sample_localize_report),
                "--sheet",
                str(sheet_path),
                "--out",
                str(out_path),
            ]
        )

        assert result.returncode == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        # Only the first panel with valid bbox should be included
        assert len(data["panels"]) == 1
        assert data["panels"][0]["panel_id"] == "0001_F01-P01_chart"

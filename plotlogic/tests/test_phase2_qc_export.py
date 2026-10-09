import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_phase2_qc_writes_expected_markdown_sections(tmp_path: Path) -> None:
    input_path = tmp_path / "phase2_localize_report.json"
    output_path = tmp_path / "phase2_qc.md"
    report = {
        "generated_at_utc": "2026-02-25T04:00:00Z",
        "panels_total": 3,
        "critical_count": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P01_chart",
                "paper_id": "p1",
                "decision": "ACCEPT",
                "critical_reasons": [],
            },
            {
                "panel_id": "p1_F01-P02_chart",
                "paper_id": "p1",
                "decision": "WARN",
                "critical_reasons": [],
            },
            {
                "panel_id": "p2_F05-P03_chart",
                "paper_id": "p2",
                "decision": "REJECT",
                "critical_reasons": ["AREA_RATIO_OUT_OF_RANGE", "AXIS_LEFT_MISSING"],
            },
        ],
    }
    input_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_phase2_qc.py"),
            "--input",
            str(input_path),
            "--out",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    markdown = output_path.read_text(encoding="utf-8")
    assert "# Phase 2 QC Summary" in markdown
    assert "ACCEPT: 1" in markdown
    assert "WARN: 1" in markdown
    assert "REJECT: 1" in markdown
    assert "AREA_RATIO_OUT_OF_RANGE" in markdown
    assert "p2_F05-P03_chart" in markdown

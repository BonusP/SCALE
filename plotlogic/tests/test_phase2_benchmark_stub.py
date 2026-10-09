import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_phase2_benchmark_stub_without_gt(tmp_path: Path) -> None:
    input_path = tmp_path / "phase2_localize_report.json"
    output_path = tmp_path / "phase2_benchmark_stub.json"
    report = {
        "generated_at_utc": "2026-02-26T10:00:00Z",
        "critical_count": 0,
        "panels": [
            {
                "panel_id": "0001_F01-P01_chart",
                "paper_id": "0001",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "score_breakdown": {"final": 0.8},
                "evidence": {"axis_left_detected": True},
                "critical_reasons": [],
            },
            {
                "panel_id": "0002_F03-P01_image",
                "paper_id": "0002",
                "panel_kind": "image",
                "decision": "WARN",
                "score_breakdown": {"final": 0.55},
                "evidence": {"axis_left_detected": False},
                "critical_reasons": [],
            },
        ],
    }
    input_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_phase2_benchmark_stub.py"),
            "--dataset-root",
            str(tmp_path),
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
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "GT_FREE"
    assert payload["summary"]["panels_total"] == 2
    assert payload["summary"]["decision_counts"] == {"ACCEPT": 1, "WARN": 1, "REJECT": 0}
    assert payload["gt_stub"]["gt_present"] is False
    assert payload["metrics"]["panel_roi_iou"]["status"] == "PENDING_BBOX_EXPORT"


def test_export_phase2_benchmark_stub_with_gt(tmp_path: Path) -> None:
    input_path = tmp_path / "phase2_localize_report.json"
    gt_path = tmp_path / "gt_plots.json"
    output_path = tmp_path / "phase2_benchmark_stub.json"

    report = {
        "generated_at_utc": "2026-02-26T10:00:00Z",
        "critical_count": 1,
        "panels": [
            {
                "panel_id": "0001_F01-P01_chart",
                "paper_id": "0001",
                "panel_kind": "chart",
                "decision": "REJECT",
                "score_breakdown": {"final": 0.2},
                "evidence": {"axis_left_detected": False},
                "critical_reasons": ["AXIS_LEFT_MISSING"],
            }
        ],
    }
    input_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    gt_payload = {"panels": [{"panel_id": "0001_F01-P01_chart"}, {"panel_id": "0009_F10-P02_chart"}]}
    gt_path.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_phase2_benchmark_stub.py"),
            "--dataset-root",
            str(tmp_path),
            "--input",
            str(input_path),
            "--gt",
            str(gt_path),
            "--out",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "GT_ATTACHED_STUB"
    assert payload["gt_stub"]["gt_present"] is True
    assert payload["gt_stub"]["gt_panel_total"] == 2
    assert payload["gt_stub"]["matched_panel_total"] == 1
    assert payload["gt_stub"]["gt_coverage_ratio"] == 0.5

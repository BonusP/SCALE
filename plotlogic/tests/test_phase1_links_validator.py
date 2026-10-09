import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")


def _run_validator(dataset_root: Path, out_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_phase1_links.py"),
            "--dataset-root",
            str(dataset_root),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_phase1_links_exit_code_follows_critical_issues(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "papers" / "p1.pdf")
    _touch(dataset_root / "figures" / "p1" / "p1_F01__src.png")
    _touch(dataset_root / "panels" / "p1" / "p1_F01-P01_chart.png")

    _write_json(
        dataset_root / "links" / "paper_index.json",
        {"p1": {"paper_id": "p1", "pdf_path": "papers/p1.pdf", "upstage_layout_path": None}},
    )
    _write_json(
        dataset_root / "links" / "figure_index.json",
        {
            "p1_F01": {
                "figure_id": "p1_F01",
                "paper_id": "p1",
                "figure_path": "figures/p1/p1_F01__src.png",
            }
        },
    )
    _write_json(
        dataset_root / "links" / "panel_index.json",
        {
            "panels": [
                {
                    "panel_id": "p1_F01-P01_chart",
                    "paper_id": "p1",
                    "figure_id": "p1_F01",
                    "panel_path": "panels/p1/p1_F01-P01_chart.png",
                }
            ]
        },
    )
    _write_json(
        dataset_root / "links" / "component_index.json",
        {
            "p1_F01-P01_chart-C01": {
                "panel_id": "p1_F01-P01_chart",
                "bbox_xyxy": [0, 0, 10, 10],
                "coord_space": "panel_px",
            }
        },
    )

    ok_out = tmp_path / "ok_report.json"
    ok_result = _run_validator(dataset_root, ok_out)
    assert ok_result.returncode == 0, ok_result.stderr
    ok_report = json.loads(ok_out.read_text(encoding="utf-8"))
    assert ok_report["summary"]["critical"] == 0

    _write_json(
        dataset_root / "links" / "component_index.json",
        {
            "p1_F01-P01_chart-C01": {
                "panel_id": "missing_panel_id",
                "bbox_xyxy": [0, 0, 10, 10],
                "coord_space": "panel_px",
            }
        },
    )
    fail_out = tmp_path / "fail_report.json"
    fail_result = _run_validator(dataset_root, fail_out)
    assert fail_result.returncode == 1
    fail_report = json.loads(fail_out.read_text(encoding="utf-8"))
    codes = {issue["code"] for issue in fail_report["issues"]}
    assert "COMPONENT_UNKNOWN_PANEL" in codes

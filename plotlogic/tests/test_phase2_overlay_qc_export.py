import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_phase2_overlay_qc_generates_manifest_and_image(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P01_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), color="white").save(panel_path)

    report = {
        "generated_at_utc": "2026-02-26T09:00:00Z",
        "panels": [
            {
                "panel_id": "p1_F01-P01_chart",
                "paper_id": "p1",
                "panel_path": "panels/p1/p1_F01-P01_chart.png",
                "decision": "WARN",
                "evidence": {
                    "predicted_plot_frame_bbox_xyxy": [20, 12, 98, 66],
                },
            }
        ],
    }

    input_path = tmp_path / "phase2_localize_report.json"
    out_dir = tmp_path / "overlays"
    manifest_path = tmp_path / "phase2_overlay_manifest.json"
    input_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_phase2_overlay_qc.py"),
            "--dataset-root",
            str(dataset_root),
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--manifest-out",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["panels_total"] == 1
    assert manifest["summary"]["generated"] == 1
    assert manifest["summary"]["skipped_no_bbox"] == 0
    overlay_path = Path(manifest["items"][0]["overlay_path"])
    assert overlay_path.exists()

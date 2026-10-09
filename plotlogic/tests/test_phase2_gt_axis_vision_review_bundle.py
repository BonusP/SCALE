import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_phase2_gt_axis_vision_review_bundle_writes_expected_sections(tmp_path: Path) -> None:
    gt_axes_path = tmp_path / "gt_axes.json"
    gt_axes_seed_path = tmp_path / "phase2_gt_axes_seed_v1.json"
    out_path = tmp_path / "phase2_gt_axis_vision_review_bundle_v1.json"

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1",
                "panel_path": "panels/p1.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [10, 10, 30, 100],
                "axis_bottom_bbox_xyxy": [20, 90, 120, 110],
                "axis_right_bbox_xyxy": None,
                "note": "accepted",
                "updated_at_utc": "2026-03-26T10:00:00Z",
            },
            {
                "panel_id": "p2",
                "panel_path": "panels/p2.png",
                "panel_kind": "chart",
                "is_dual_y": True,
                "review_status": "pending",
                "axis_left_bbox_xyxy": [11, 10, 31, 100],
                "axis_bottom_bbox_xyxy": [21, 90, 121, 110],
                "axis_right_bbox_xyxy": [110, 10, 130, 100],
                "note": "pending",
                "updated_at_utc": "2026-03-26T10:01:00Z",
            },
        ],
    }
    gt_axes_seed = {
        "panels": [
            {
                "panel_id": "p1",
                "image_size_wh": [140, 120],
                "source_pred_plot_frame_bbox_xyxy": [20, 20, 120, 100],
                "axis_left_bbox_xyxy": [12, 20, 20, 100],
            },
            {
                "panel_id": "p2",
                "image_size_wh": [160, 120],
                "source_pred_plot_frame_bbox_xyxy": [20, 20, 140, 100],
                "axis_left_bbox_xyxy": [10, 20, 28, 100],
            },
        ]
    }
    gt_axes_path.write_text(json.dumps(gt_axes, indent=2), encoding="utf-8")
    gt_axes_seed_path.write_text(json.dumps(gt_axes_seed, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_phase2_gt_axis_vision_review_bundle.py"),
            "--gt-axes",
            str(gt_axes_path),
            "--gt-axes-seed",
            str(gt_axes_seed_path),
            "--out",
            str(out_path),
            "--reference-limit",
            "10",
            "--rule-version",
            "v8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["rule_version"] == "v8"
    assert len(payload["accepted_reference_panels"]) == 1
    assert len(payload["pending_panels"]) == 1
    assert payload["accepted_reference_panels"][0]["bucket_hint"] == "tick_weak"
    assert payload["pending_panels"][0]["bucket_hint"] == "dual_y"

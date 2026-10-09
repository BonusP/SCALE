import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "apply_phase2_gt_axes_pending_postprocess_v9.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gt_axes_pending_postprocess_v9", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_postprocess_keeps_origin_inside_and_widens_bottom() -> None:
    module = _load_module()
    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1",
                "review_status": "pending",
                "axis_left_bbox_xyxy": [12, 12, 20, 90],
                "axis_bottom_bbox_xyxy": [18, 84, 98, 96],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "updated_at_utc": "2026-03-27T00:00:00Z",
            }
        ],
    }
    seed = {
        "panels": [
            {
                "panel_id": "p1",
                "image_size_wh": [120, 110],
                "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 90],
            }
        ]
    }

    updated, events = module.apply_postprocess(
        gt_axes,
        seed,
        actor="codex-rule-v9",
        source="bulk-rebox/pending-axis-bbox-v9",
    )
    row = updated["panels"][0]
    assert row["axis_left_bbox_xyxy"][0] <= 10 <= row["axis_left_bbox_xyxy"][2]
    assert row["axis_left_bbox_xyxy"][1] <= 90 <= row["axis_left_bbox_xyxy"][3]
    assert row["axis_bottom_bbox_xyxy"][0] <= 10 <= row["axis_bottom_bbox_xyxy"][2]
    assert row["axis_bottom_bbox_xyxy"][1] <= 90 <= row["axis_bottom_bbox_xyxy"][3]
    assert row["axis_bottom_bbox_xyxy"][2] >= 100 + round((100 - 10) * 0.037)
    assert len(events) == 1

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "apply_phase2_gt_axis_vision_verdict.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gt_axis_vision_apply", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_verdicts_updates_pending_rows_and_preserves_status() -> None:
    module = _load_module()
    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1",
                "review_status": "pending",
                "is_dual_y": False,
                "axis_left_bbox_xyxy": [1, 1, 2, 2],
                "axis_bottom_bbox_xyxy": [3, 3, 4, 4],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "old",
                "updated_at_utc": "2026-03-27T00:00:00Z",
            },
            {
                "panel_id": "p2",
                "review_status": "accept",
                "is_dual_y": False,
                "axis_left_bbox_xyxy": [9, 9, 10, 10],
                "axis_bottom_bbox_xyxy": [11, 11, 12, 12],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "updated_at_utc": "2026-03-27T00:00:00Z",
            },
        ],
    }
    verdict_artifact = {
        "model": "gpt-5.4",
        "verdicts": [
            {
                "panel_id": "p1",
                "decision": "reject_axis_assignment",
                "bucket_confirmed": "dual_y",
                "issues": [],
                "recommended_left_bbox_xyxy": [10, 11, 14, 30],
                "recommended_bottom_bbox_xyxy": [20, 21, 60, 26],
                "recommended_right_bbox_xyxy": [30, 31, 34, 52],
                "rationale": "dual y",
                "confidence": 0.9,
            },
            {
                "panel_id": "p2",
                "decision": "adjust",
                "bucket_confirmed": "tick_rich",
                "issues": [],
                "recommended_left_bbox_xyxy": [40, 41, 42, 43],
                "recommended_bottom_bbox_xyxy": [50, 51, 52, 53],
                "recommended_right_bbox_xyxy": None,
                "rationale": "should not apply",
                "confidence": 0.9,
            },
        ],
    }

    updated, events, skipped = module.apply_verdicts(
        gt_axes,
        verdict_artifact,
        actor="codex-vision-review",
        source="vision-review/codex-exec",
    )

    row1 = updated["panels"][0]
    assert row1["review_status"] == "pending"
    assert row1["is_dual_y"] is True
    assert row1["axis_right_bbox_xyxy"] == [30, 31, 34, 52]
    assert "[vision:gpt-5.4] reject_axis_assignment: dual y" in row1["note"]
    row2 = updated["panels"][1]
    assert row2["axis_left_bbox_xyxy"] == [9, 9, 10, 10]
    assert len(events) == 1
    assert events[0]["panel_id"] == "p1"
    assert skipped == []


def test_apply_verdicts_skips_swapped_role_geometry() -> None:
    module = _load_module()
    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1",
                "review_status": "pending",
                "is_dual_y": False,
                "axis_left_bbox_xyxy": [1, 1, 2, 20],
                "axis_bottom_bbox_xyxy": [3, 30, 40, 35],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "updated_at_utc": "2026-03-27T00:00:00Z",
            }
        ],
    }
    verdict_artifact = {
        "model": "gpt-5.4-mini",
        "verdicts": [
            {
                "panel_id": "p1",
                "decision": "adjust",
                "bucket_confirmed": "tick_rich",
                "issues": [],
                "recommended_left_bbox_xyxy": [0, 100, 300, 120],
                "recommended_bottom_bbox_xyxy": [20, 0, 40, 200],
                "recommended_right_bbox_xyxy": None,
                "rationale": "swapped roles",
                "confidence": 0.8,
            }
        ],
    }

    updated, events, skipped = module.apply_verdicts(
        gt_axes,
        verdict_artifact,
        actor="codex-vision-review",
        source="vision-review/codex-exec",
    )

    row = updated["panels"][0]
    assert row["axis_left_bbox_xyxy"] == [1, 1, 2, 20]
    assert row["axis_bottom_bbox_xyxy"] == [3, 30, 40, 35]
    assert events == []
    assert skipped == [
        {
            "panel_id": "p1",
            "reason": "left bbox is not a vertical strip; bottom bbox is not a horizontal band",
        }
    ]

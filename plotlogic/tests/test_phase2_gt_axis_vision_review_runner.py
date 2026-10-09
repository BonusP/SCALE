import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_phase2_gt_axis_vision_review.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gt_axis_vision_review_runner", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_request_payload_embeds_target_and_reference_images(tmp_path: Path) -> None:
    module = _load_module()
    panel_dir = tmp_path / "panels" / "0001"
    panel_dir.mkdir(parents=True)
    (panel_dir / "target.png").write_bytes(b"fakepng")
    (panel_dir / "ref.png").write_bytes(b"fakepng")

    bundle = {
        "rule_version": "v8",
        "rules": ["Anchor axis bbox to the source plot frame."],
    }
    target = {
        "panel_id": "target",
        "panel_path": "panels/0001/target.png",
        "is_dual_y": False,
        "bucket_hint": "tick_rich",
        "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
        "axis_left_bbox_xyxy": [5, 10, 20, 100],
        "axis_bottom_bbox_xyxy": [10, 90, 100, 110],
        "axis_right_bbox_xyxy": None,
        "note": None,
    }
    reference = {
        "panel_id": "ref",
        "panel_path": "panels/0001/ref.png",
        "is_dual_y": False,
        "bucket_hint": "tick_rich",
        "source_pred_plot_frame_bbox_xyxy": [10, 10, 100, 100],
        "axis_left_bbox_xyxy": [6, 10, 22, 100],
        "axis_bottom_bbox_xyxy": [10, 90, 100, 112],
        "axis_right_bbox_xyxy": None,
        "note": "accepted",
    }

    payload = module.build_request_payload(
        bundle,
        target,
        [reference],
        dataset_root=tmp_path,
        model="gpt-5.4",
        max_output_tokens=700,
        reasoning_effort="medium",
    )

    assert payload["model"] == "gpt-5.4"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    user_content = payload["input"][1]["content"]
    image_items = [item for item in user_content if item["type"] == "input_image"]
    assert len(image_items) == 2
    assert image_items[0]["image_url"].startswith("data:image/png;base64,")
    assert image_items[1]["detail"] == "low"


def test_parse_verdict_response_reads_structured_json_output() -> None:
    module = _load_module()
    response_payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "panel_id": "p1",
                                "decision": "adjust",
                                "bucket_confirmed": "tick_rich",
                                "issues": ["BOTTOM_RIGHT_TICK_CLIPPED"],
                                "recommended_left_bbox_xyxy": [1, 2, 3, 4],
                                "recommended_bottom_bbox_xyxy": [5, 6, 7, 8],
                                "recommended_right_bbox_xyxy": None,
                                "rationale": "Need more right margin.",
                                "confidence": 0.81,
                            }
                        ),
                    }
                ],
            }
        ]
    }

    verdict = module.parse_verdict_response(response_payload)
    assert verdict["panel_id"] == "p1"
    assert verdict["decision"] == "adjust"
    assert verdict["recommended_bottom_bbox_xyxy"] == [5, 6, 7, 8]


def test_select_reference_panels_prefers_matching_bucket_and_dual_flag() -> None:
    module = _load_module()
    target = {"panel_id": "t", "bucket_hint": "tick_rich", "is_dual_y": False}
    refs = [
        {"panel_id": "a", "bucket_hint": "tick_weak", "is_dual_y": False},
        {"panel_id": "b", "bucket_hint": "tick_rich", "is_dual_y": False},
        {"panel_id": "c", "bucket_hint": "dual_y", "is_dual_y": True},
    ]
    selected = module.select_reference_panels(refs, target, reference_count=2)
    assert [row["panel_id"] for row in selected] == ["b", "a"]


def test_build_codex_prompt_mentions_target_and_references() -> None:
    module = _load_module()
    bundle = {"rule_version": "v8", "rules": ["Include tick marks and tick labels."]}
    target = {
        "panel_id": "target",
        "bucket_hint": "tick_rich",
        "is_dual_y": False,
        "source_pred_plot_frame_bbox_xyxy": [1, 2, 3, 4],
        "axis_left_bbox_xyxy": [1, 2, 3, 4],
        "axis_bottom_bbox_xyxy": [5, 6, 7, 8],
        "axis_right_bbox_xyxy": None,
        "note": "",
    }
    ref = {
        "panel_id": "ref",
        "bucket_hint": "tick_weak",
        "is_dual_y": False,
        "source_pred_plot_frame_bbox_xyxy": [1, 2, 3, 4],
        "axis_left_bbox_xyxy": [1, 2, 3, 4],
        "axis_bottom_bbox_xyxy": [5, 6, 7, 8],
        "axis_right_bbox_xyxy": None,
        "note": "accepted",
    }
    prompt = module.build_codex_prompt(bundle, target, [ref])
    assert "TARGET PANEL: target" in prompt
    assert "ACCEPTED REFERENCE: ref" in prompt
    assert "Return exactly one JSON object" in prompt
    assert "prefer full x tick-label capture" in prompt

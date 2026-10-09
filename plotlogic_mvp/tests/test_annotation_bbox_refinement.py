# OCR annotation bbox 보정을 위한 수정: prevent broad exclusions from erasing curves.
from app.services.ocr_geometry import build_ocr_text_geometry


def vision(regions):
    return {"plot_frame_bbox_panel_px": [0, 0, 600, 300], "axes": [],
            "legend_items": [], "plot_annotation_regions": regions}


def test_broad_annotation_splits_into_individual_ocr_boxes():
    region = {"annotation_text": "point value labels", "bbox_panel_px": [10, 10, 590, 290]}
    original = vision([region])
    items = [{"text": "85.5", "bbox": [20, 30, 50, 45], "confidence": .99},
             {"text": "86.1", "bbox": [450, 70, 480, 85], "confidence": .99}]
    result = build_ocr_text_geometry(original, items, refine_annotations=True)
    regions = result["panel_vision"]["plot_annotation_regions"]
    assert [item["bbox_panel_px"] for item in regions] == [item["bbox"] for item in items]
    assert all(item["codex_bbox_panel_px"] == region["bbox_panel_px"] for item in regions)
    assert original["plot_annotation_regions"] == [region]
    assert result["summary"]["text_boxes_replaced"]["annotation_text"] == 2


def test_large_unconfirmed_region_is_not_used_as_exclusion():
    result = build_ocr_text_geometry(vision([{"bbox_panel_px": [10, 10, 590, 290]}]), [], refine_annotations=True)
    region = result["panel_vision"]["plot_annotation_regions"][0]
    assert region["bbox_panel_px"] is None
    assert region["codex_bbox_panel_px"] == [10, 10, 590, 290]
    assert "BROAD_ANNOTATION_EXCLUSION_SKIPPED:0" in result["issues"]


def test_small_boxes_and_non_plot_ui_are_preserved():
    regions = [{"bbox_panel_px": [10, 10, 40, 25]},
               {"annotation_text": "toolbar UI", "bbox_panel_px": [550, 0, 600, 280]}]
    result = build_ocr_text_geometry(vision(regions), [], refine_annotations=True)
    assert [item["bbox_panel_px"] for item in result["panel_vision"]["plot_annotation_regions"]] == [item["bbox_panel_px"] for item in regions]


def test_refinement_is_opt_in_and_rejects_low_confidence_or_broad_ocr():
    original = vision([{"bbox_panel_px": [10, 10, 590, 290]}])
    items = [{"text": "noise", "bbox": [20, 30, 50, 45], "confidence": .01},
             {"text": "merged row", "bbox": [20, 60, 550, 90], "confidence": .99}]
    assert build_ocr_text_geometry(original, items)["panel_vision"] == original
    result = build_ocr_text_geometry(original, items, refine_annotations=True)
    assert result["panel_vision"]["plot_annotation_regions"][0]["bbox_panel_px"] is None

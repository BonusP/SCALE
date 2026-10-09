import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "scripts" / "export_phase2_gt_axis_ticks.py"
SPEC = importlib.util.spec_from_file_location("export_phase2_gt_axis_ticks", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
export_axis_ticks = MODULE.export_axis_ticks
prune_overlapping_matched_candidates = MODULE._prune_overlapping_matched_candidates
correct_decimal_sequence_outliers = MODULE._correct_decimal_sequence_outliers
correct_linear_sequence_outliers = MODULE._correct_linear_sequence_outliers
recover_bottom_candidates_from_tick_windows = MODULE._recover_bottom_candidates_from_tick_windows
tighten_candidate_bbox_to_numeric_ink = MODULE._tighten_candidate_bbox_to_numeric_ink
extend_bbox_for_leading_sign = MODULE._extend_bbox_for_leading_sign


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _mock_ocr_runner(results_by_axis_tick: dict[tuple[str, int], tuple[str, float]]):
    def runner(crop_requests, *, model: str, cwd: Path, timeout_seconds: int):
        del model, cwd, timeout_seconds
        outputs = {}
        for request in crop_requests:
            text, confidence = results_by_axis_tick.get(
                (request["axis_role"], request["tick_position_px"]),
                ("", 0.0),
            )
            outputs[request["candidate_id"]] = {
                "text": text,
                "confidence": confidence,
            }
        return outputs

    return runner


def _mock_bbox_runner(results_by_request_id: dict[str, list[dict[str, object]]]):
    def runner(axis_requests, *, model: str, cwd: Path, timeout_seconds: int):
        del model, cwd, timeout_seconds
        outputs = {}
        for request in axis_requests:
            outputs[request["request_id"]] = results_by_request_id.get(request["request_id"], [])
        return outputs

    return runner


def test_export_axis_ticks_detects_left_and_bottom_tick_positions(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P01_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (120, 100), color=255)
    draw = ImageDraw.Draw(image)

    # left axis and ticks
    draw.line((20, 10, 20, 80), fill=0, width=1)
    for y in (20, 40, 60):
        draw.line((20, y, 27, y), fill=0, width=1)

    # bottom axis and ticks
    draw.line((20, 80, 90, 80), fill=0, width=1)
    for x in (30, 50, 70):
        draw.line((x, 73, x, 80), fill=0, width=1)

    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P01_chart",
                "panel_path": "panels/p1/p1_F01-P01_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [0, 10, 30, 82],
                "axis_bottom_bbox_xyxy": [20, 72, 92, 100],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    out_path = dataset_root / "links" / "gt_axis_ticks.json"
    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=out_path,
        ocr_backend="none",
    )

    assert out_path.exists()
    assert payload["schema_version"] == 2
    assert payload["summary"]["panels_total"] == 1
    assert payload["summary"]["axes_total"] == 2

    panel = payload["panels"][0]
    assert panel["panel_id"] == "p1_F01-P01_chart"
    by_role = {axis["axis_role"]: axis for axis in panel["axes"]}

    assert by_role["left"]["search_bbox_xyxy"] == [0, 10, 30, 82]
    assert by_role["bottom"]["search_bbox_xyxy"] == [20, 72, 92, 100]
    assert by_role["left"]["tick_positions_px"] == [20, 40, 60]
    assert by_role["bottom"]["tick_positions_px"] == [30, 50, 70]
    assert by_role["left"]["tick_count"] == 3
    assert by_role["bottom"]["tick_count"] == 3
    assert by_role["left"]["axis_line_coordinate_px"] == 20
    assert by_role["bottom"]["axis_line_coordinate_px"] == 80
    assert by_role["left"]["tick_label_candidates"] == []
    assert by_role["bottom"]["tick_label_candidates"] == []
    assert by_role["left"]["tick_matches"] == []
    assert by_role["bottom"]["tick_matches"] == []
    assert "ZERO_MATCHES" in by_role["left"]["issues"]
    assert "ZERO_MATCHES" in by_role["bottom"]["issues"]


def test_export_axis_ticks_clips_negative_left_axis_bbox_for_detection(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P01b_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (120, 100), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((26, 4, 26, 80), fill=0, width=1)
    for y in (16, 36, 56, 76):
        draw.line((26, y, 33, y), fill=0, width=1)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P01b_chart",
                "panel_path": "panels/p1/p1_F01-P01b_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [12, -6, 40, 82],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["axis_role"] == "left"
    assert left_axis["tick_positions_px"] == [16, 36, 56, 76]
    assert left_axis["tick_count"] == 4


def test_export_axis_ticks_left_tick_detection_ignores_near_axis_data_blob(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P01c_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (140, 110), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((50, 10, 50, 90), fill=0, width=1)
    for y in (22, 44, 66):
        draw.line((50, y, 57, y), fill=0, width=1)
    # A dark data-like blob near the axis but not attached to the axis line.
    draw.rectangle((38, 32, 47, 38), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P01c_chart",
                "panel_path": "panels/p1/p1_F01-P01c_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [20, 10, 60, 92],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["tick_positions_px"] == [22, 44, 66]
    assert left_axis["tick_count"] == 3


def test_export_axis_ticks_left_tick_detection_penalizes_data_line_touching_axis(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P01c2_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (150, 110), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((50, 10, 50, 90), fill=0, width=1)
    for y in (22, 66):
        draw.line((50, y, 57, y), fill=0, width=1)
    # Data line touching the axis and extending deep into the plot frame.
    draw.line((50, 44, 84, 44), fill=0, width=1)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P01c2_chart",
                "panel_path": "panels/p1/p1_F01-P01c2_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [20, 10, 90, 92],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["tick_positions_px"] == [22, 66]
    assert left_axis["tick_count"] == 2


def test_export_axis_ticks_recovers_left_labels_when_ticks_are_missing(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P01d_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (150, 130), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((58, 10, 58, 110), fill=0, width=1)
    # No visible tick stubs, only label-like blobs.
    for y in (24, 54, 84):
        draw.rectangle((24, y, 30, y + 11), fill=0)
        draw.rectangle((34, y, 48, y + 11), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P01d_chart",
                "panel_path": "panels/p1/p1_F01-P01d_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [20, 10, 62, 112],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["tick_positions_px"] == [30, 60, 90]
    assert len(left_axis["tick_matches"]) == 3
    assert [row["tick_position_px"] for row in left_axis["tick_matches"]] == [30, 60, 90]
    assert all(match["candidate_bbox_xyxy"][2] < 58 for match in left_axis["tick_matches"])


def test_export_axis_ticks_collects_ocr_label_candidates_and_normalizes_numeric_text(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P02_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (140, 110), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((24, 10, 24, 86), fill=0, width=1)
    draw.line((24, 86, 110, 86), fill=0, width=1)
    for y in (24, 44, 64):
        draw.line((24, y, 30, y), fill=0, width=1)
    for x in (36, 58, 80):
        draw.line((x, 80, x, 86), fill=0, width=1)
    for y in (18, 38, 58):
        draw.rectangle((0, y, 4, y + 10), fill=0)
        draw.rectangle((8, y, 18, y + 10), fill=0)
    for x in (30, 52, 74):
        draw.rectangle((x, 92, x + 4, 104), fill=0)
        draw.rectangle((x + 8, 92, x + 12, 104), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P02_chart",
                "panel_path": "panels/p1/p1_F01-P02_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [0, 10, 40, 88],
                "axis_bottom_bbox_xyxy": [24, 78, 112, 110],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)
    out_path = dataset_root / "links" / "gt_axis_ticks.json"
    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=out_path,
        ocr_runner=_mock_ocr_runner(
            {
                ("left", 24): ("30", 0.93),
                ("left", 44): ("20", 0.91),
                ("left", 64): ("10", 0.89),
                ("bottom", 36): ("400", 0.96),
                ("bottom", 58): ("500", 0.94),
                ("bottom", 80): ("600", 0.92),
            }
        ),
    )

    panel = payload["panels"][0]
    by_role = {axis["axis_role"]: axis for axis in panel["axes"]}

    left_labels = by_role["left"]["tick_label_candidates"]
    bottom_labels = by_role["bottom"]["tick_label_candidates"]

    assert by_role["left"]["search_bbox_xyxy"] == [0, 10, 40, 88]
    assert by_role["bottom"]["search_bbox_xyxy"] == [24, 78, 112, 110]
    assert by_role["left"]["tick_positions_px"] == [24, 44, 64]
    assert by_role["bottom"]["tick_positions_px"] == [36, 58, 80]
    assert [item["normalized_text"] for item in left_labels] == ["30", "20", "10"]
    assert [item["numeric_value"] for item in left_labels] == [30.0, 20.0, 10.0]
    assert all(item["ocr_source"] == "tesseract" for item in left_labels)
    assert all(item["bbox_xyxy"][0] == 0 for item in left_labels)
    assert [item["bbox_xyxy"][1] for item in left_labels] == [18, 38, 58]
    assert [item["normalized_text"] for item in bottom_labels] == ["400", "500", "600"]
    assert [item["numeric_value"] for item in bottom_labels] == [400.0, 500.0, 600.0]
    assert all(item["ocr_source"] == "tesseract" for item in bottom_labels)
    assert by_role["bottom"]["tick_positions_px"] == [36, 58, 80]
    assert [item["bbox_xyxy"][0] for item in bottom_labels] == [28, 50, 72]
    assert all(item["bbox_xyxy"][1] == 90 for item in bottom_labels)
    assert [item["tick_position_px"] for item in by_role["left"]["tick_matches"]] == [24, 44, 64]
    assert [item["tick_position_px"] for item in by_role["bottom"]["tick_matches"]] == [36, 58, 80]
    assert by_role["left"]["issues"] == []
    assert by_role["bottom"]["issues"] == []


def test_export_axis_ticks_detects_label_candidate_boxes_without_ocr_sidecar(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P03_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (160, 120), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((34, 12, 34, 90), fill=0, width=1)
    draw.line((34, 90, 128, 90), fill=0, width=1)
    for y in (28, 56, 84):
        draw.line((34, y, 40, y), fill=0, width=1)
    for x in (46, 76, 106):
        draw.line((x, 84, x, 90), fill=0, width=1)

    for y in (24, 52, 80):
        draw.rectangle((8, y, 12, y + 9), fill=0)
        draw.rectangle((16, y, 22, y + 9), fill=0)
    for x in (44, 74, 104):
        draw.rectangle((x, 96, x + 4, 106), fill=0)
        draw.rectangle((x + 8, 96, x + 14, 106), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P03_chart",
                "panel_path": "panels/p1/p1_F01-P03_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [4, 12, 40, 92],
                "axis_bottom_bbox_xyxy": [34, 82, 130, 118],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    out_path = dataset_root / "links" / "gt_axis_ticks.json"
    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=out_path,
        ocr_backend="none",
    )

    panel = payload["panels"][0]
    by_role = {axis["axis_role"]: axis for axis in panel["axes"]}
    left_labels = by_role["left"]["tick_label_candidates"]
    bottom_labels = by_role["bottom"]["tick_label_candidates"]

    assert len(left_labels) == 3
    assert len(bottom_labels) == 3
    assert by_role["left"]["tick_positions_px"] == [28, 56, 84]
    assert by_role["bottom"]["tick_positions_px"] == [46, 76, 106]
    assert len(by_role["left"]["tick_matches"]) == 3
    assert len(by_role["bottom"]["tick_matches"]) == 3
    assert all(item["text"] == "" for item in left_labels + bottom_labels)
    assert all(item["numeric_value"] is None for item in left_labels + bottom_labels)
    assert all(item["ocr_source"] == "none" for item in left_labels + bottom_labels)
    assert all(label["bbox_xyxy"][2] <= 26 for label in left_labels)
    assert all(label["bbox_xyxy"][1] >= 94 for label in bottom_labels)
    assert "OCR_EMPTY" in by_role["left"]["issues"]
    assert "INSUFFICIENT_SUPPORT" in by_role["bottom"]["issues"]


def test_export_axis_ticks_filters_axis_title_like_boxes_and_keeps_tick_labels(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P04_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (180, 220), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((34, 12, 34, 100), fill=0, width=1)
    draw.line((34, 100, 150, 100), fill=0, width=1)
    for y in (32, 60, 88):
        draw.line((34, y, 40, y), fill=0, width=1)
    for x in (50, 86, 122):
        draw.line((x, 94, x, 100), fill=0, width=1)

    for y in (28, 56, 84):
        draw.rectangle((12, y, 16, y + 9), fill=0)
        draw.rectangle((20, y, 26, y + 9), fill=0)
    for x in (46, 82, 118):
        draw.rectangle((x, 108, x + 4, 118), fill=0)
        draw.rectangle((x + 8, 108, x + 14, 118), fill=0)

    # Axis-title-like blocks: too tall / too wide to be treated as tick labels.
    draw.rectangle((0, 22, 4, 94), fill=0)
    draw.rectangle((54, 124, 126, 132), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P04_chart",
                "panel_path": "panels/p1/p1_F01-P04_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [4, 12, 40, 102],
                "axis_bottom_bbox_xyxy": [34, 92, 152, 132],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    panel = payload["panels"][0]
    by_role = {axis["axis_role"]: axis for axis in panel["axes"]}
    left_labels = by_role["left"]["tick_label_candidates"]
    bottom_labels = by_role["bottom"]["tick_label_candidates"]

    assert len(left_labels) == 3
    assert len(bottom_labels) == 3
    assert by_role["left"]["tick_positions_px"] == [32, 60, 88]
    assert by_role["bottom"]["tick_positions_px"] == [50, 86, 122]
    assert len(by_role["left"]["tick_matches"]) == 3
    assert len(by_role["bottom"]["tick_matches"]) == 3
    assert all(label["bbox_xyxy"][0] >= 6 for label in left_labels)
    assert all(label["bbox_xyxy"][3] <= 120 for label in bottom_labels)


def test_export_axis_ticks_detects_bottom_ticks_below_axis_without_expanding_search_bbox(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P04b_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (150, 130), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((28, 84, 112, 84), fill=0, width=1)
    for x in (40, 68, 96):
        draw.line((x, 84, x, 91), fill=0, width=1)
    for x in (36, 64, 92):
        draw.rectangle((x, 96, x + 4, 108), fill=0)
        draw.rectangle((x + 8, 96, x + 14, 108), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P04b_chart",
                "panel_path": "panels/p1/p1_F01-P04b_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": None,
                "axis_bottom_bbox_xyxy": [28, 78, 112, 110],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    bottom_axis = payload["panels"][0]["axes"][0]
    assert bottom_axis["axis_role"] == "bottom"
    assert bottom_axis["tick_positions_px"] == [40, 68, 96]
    assert bottom_axis["search_bbox_xyxy"] == [28, 78, 112, 110]
    assert len(bottom_axis["tick_label_candidates"]) == 3
    assert [row["tick_position_px"] for row in bottom_axis["tick_matches"]] == [40, 68, 96]
    assert all(label["bbox_xyxy"][1] >= 94 for label in bottom_axis["tick_label_candidates"])
    assert all(label["bbox_xyxy"][3] > 100 for label in bottom_axis["tick_label_candidates"])


def test_export_axis_ticks_bottom_does_not_rescue_blank_line_fragments(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P04d_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (170, 130), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((28, 84, 140, 84), fill=0, width=1)
    for x in (40, 68, 96, 124):
        draw.line((x, 84, x, 91), fill=0, width=1)

    # Valid label-like blobs only at 68 and 124.
    for x in (62, 118):
        draw.rectangle((x, 96, x + 4, 108), fill=0)
        draw.rectangle((x + 8, 96, x + 14, 108), fill=0)

    # Thin horizontal fragments near the axis line that should not become candidates.
    for x in (34, 86):
        draw.rectangle((x, 84, x + 20, 86), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P04d_chart",
                "panel_path": "panels/p1/p1_F01-P04d_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": None,
                "axis_bottom_bbox_xyxy": [28, 78, 140, 110],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    bottom_axis = payload["panels"][0]["axes"][0]
    assert bottom_axis["axis_role"] == "bottom"
    assert bottom_axis["tick_positions_px"] == [40, 68, 96, 124]
    assert len(bottom_axis["tick_label_candidates"]) == 2
    assert [row["tick_position_px"] for row in bottom_axis["tick_matches"]] == [68, 124]
    assert all(label["bbox_xyxy"][1] >= 94 for label in bottom_axis["tick_label_candidates"])
    assert all(label["bbox_xyxy"][3] > 100 for label in bottom_axis["tick_label_candidates"])


def test_export_axis_ticks_merges_split_left_decimal_label_fragments(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P04c_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (130, 130), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((56, 10, 56, 110), fill=0, width=1)
    for y in (30, 55, 80, 105):
        draw.line((56, y, 63, y), fill=0, width=1)

    draw.rectangle((24, 22, 30, 34), fill=0)
    draw.rectangle((34, 22, 50, 34), fill=0)
    draw.rectangle((24, 47, 38, 59), fill=0)
    draw.rectangle((42, 47, 50, 59), fill=0)
    draw.rectangle((24, 72, 38, 84), fill=0)
    draw.rectangle((42, 72, 50, 84), fill=0)
    draw.rectangle((24, 97, 30, 109), fill=0)
    draw.rectangle((34, 97, 50, 109), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P04c_chart",
                "panel_path": "panels/p1/p1_F01-P04c_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [24, 10, 64, 112],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_runner=_mock_ocr_runner(
            {
                ("left", 30): ("0.20", 0.95),
                ("left", 55): ("0.15", 0.95),
                ("left", 80): ("0.10", 0.95),
                ("left", 105): ("0.05", 0.95),
            }
        ),
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert [row["tick_position_px"] for row in left_axis["tick_matches"]] == [30, 55, 80, 105]
    assert [row["text"] for row in left_axis["tick_label_candidates"]] == ["0.20", "0.15", "0.10", "0.05"]
    assert [row["numeric_value"] for row in left_axis["tick_label_candidates"]] == [0.2, 0.15, 0.1, 0.05]
    assert left_axis["issues"] == []


def test_export_axis_ticks_merges_bottom_digit_fragments_for_same_tick(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P04e_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (150, 130), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((28, 84, 112, 84), fill=0, width=1)
    for x in (40, 68, 96):
        draw.line((x, 84, x, 91), fill=0, width=1)

    draw.rectangle((62, 96, 65, 108), fill=0)
    draw.rectangle((70, 96, 73, 108), fill=0)
    draw.rectangle((88, 96, 92, 108), fill=0)
    draw.rectangle((96, 96, 102, 108), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P04e_chart",
                "panel_path": "panels/p1/p1_F01-P04e_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": None,
                "axis_bottom_bbox_xyxy": [28, 78, 112, 110],
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_runner=_mock_ocr_runner(
            {
                ("bottom", 68): ("11", 0.95),
                ("bottom", 96): ("20", 0.95),
            }
        ),
    )

    bottom_axis = payload["panels"][0]["axes"][0]
    assert [row["tick_position_px"] for row in bottom_axis["tick_matches"]] == [68, 96]
    assert [row["text"] for row in bottom_axis["tick_label_candidates"]] == ["11", "20"]
    assert len(bottom_axis["tick_label_candidates"]) == 2
    assert bottom_axis["issues"] == []


def test_prune_overlapping_left_candidates_prefers_non_empty_numeric_bbox() -> None:
    candidates = [
        {
            "text": "-0.6",
            "normalized_text": "-0.6",
            "numeric_value": -0.6,
            "bbox_xyxy": [36, 83, 70, 100],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.92,
        },
        {
            "text": "",
            "normalized_text": "",
            "numeric_value": None,
            "bbox_xyxy": [36, 90, 69, 116],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.0,
        },
        {
            "text": "-0.4",
            "normalized_text": "-0.4",
            "numeric_value": -0.4,
            "bbox_xyxy": [36, 113, 69, 130],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.9,
        },
    ]
    matches = [
        {"tick_position_px": 91, "candidate_bbox_xyxy": [36, 83, 70, 100]},
        {"tick_position_px": 103, "candidate_bbox_xyxy": [36, 90, 69, 116]},
        {"tick_position_px": 122, "candidate_bbox_xyxy": [36, 113, 69, 130]},
    ]

    pruned_candidates, pruned_matches = prune_overlapping_matched_candidates(
        "left",
        candidates,
        matches,
    )

    assert [row["bbox_xyxy"] for row in pruned_candidates] == [
        [36, 83, 70, 100],
        [36, 113, 69, 130],
    ]
    assert [row["tick_position_px"] for row in pruned_matches] == [91, 122]


def test_correct_decimal_sequence_outliers_restores_missing_decimal_places() -> None:
    candidates = [
        {
            "text": "0.20",
            "normalized_text": "0.20",
            "numeric_value": 0.2,
            "bbox_xyxy": [29, 15, 66, 30],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.75,
        },
        {
            "text": "15",
            "normalized_text": "15",
            "numeric_value": 15.0,
            "bbox_xyxy": [45, 65, 66, 81],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.44,
        },
        {
            "text": "10",
            "normalized_text": "10",
            "numeric_value": 10.0,
            "bbox_xyxy": [45, 115, 66, 131],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.28,
        },
        {
            "text": "0.05",
            "normalized_text": "0.05",
            "numeric_value": 0.05,
            "bbox_xyxy": [29, 166, 66, 181],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.4,
        },
    ]
    matches = [
        {"tick_position_px": 23, "candidate_bbox_xyxy": [29, 15, 66, 30]},
        {"tick_position_px": 73, "candidate_bbox_xyxy": [45, 65, 66, 81]},
        {"tick_position_px": 123, "candidate_bbox_xyxy": [45, 115, 66, 131]},
        {"tick_position_px": 174, "candidate_bbox_xyxy": [29, 166, 66, 181]},
    ]

    corrected = correct_decimal_sequence_outliers("left", candidates, matches)

    assert [row["text"] for row in corrected] == ["0.20", "0.15", "0.10", "0.05"]
    assert [row["numeric_value"] for row in corrected] == [0.2, 0.15, 0.1, 0.05]
    assert corrected[1]["ocr_source"] == "tesseract-sequence-corrected"
    assert corrected[2]["ocr_source"] == "tesseract-sequence-corrected"


def test_tesseract_signed_candidate_preferred_when_confidence_is_close() -> None:
    # Mirrors the -2 vs 2 disagreement seen in real crops.
    def choose(attempt_results):
        best_text = ""
        best_confidence = 0.0
        best_score = (-1, -1, -1.0)
        for text, confidence in attempt_results:
            score = (
                1 if MODULE._parse_numeric_value(text) is not None else 0,
                1 if text and text != "-" else 0,
                confidence,
            )
            if score > best_score:
                best_score = score
                best_text = text
                best_confidence = confidence
        if best_text and not best_text.startswith(("+", "-")):
            best_value = float(best_text)
            signed_candidates = []
            for text, confidence in attempt_results:
                if not text.startswith(("+", "-")):
                    continue
                if abs(abs(float(text)) - abs(best_value)) > 1e-6:
                    continue
                signed_candidates.append((text, confidence))
            if signed_candidates:
                signed_text, signed_confidence = max(signed_candidates, key=lambda row: row[1])
                if signed_confidence >= max(0.3, best_confidence - 0.18):
                    best_text = signed_text
                    best_confidence = signed_confidence
        return best_text, best_confidence

    text, confidence = choose([("-2", 0.58), ("2", 0.72)])
    assert text == "-2"
    assert confidence == 0.58


def test_correct_linear_sequence_outliers_restores_missing_sign() -> None:
    candidates = [
        {
            "text": "-10",
            "normalized_text": "-10",
            "numeric_value": -10.0,
            "bbox_xyxy": [10, 10, 20, 20],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.88,
        },
        {
            "text": "5",
            "normalized_text": "5",
            "numeric_value": 5.0,
            "bbox_xyxy": [10, 30, 20, 40],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.61,
        },
        {
            "text": "0",
            "normalized_text": "0",
            "numeric_value": 0.0,
            "bbox_xyxy": [10, 50, 20, 60],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.80,
        },
        {
            "text": "5",
            "normalized_text": "5",
            "numeric_value": 5.0,
            "bbox_xyxy": [10, 70, 20, 80],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.79,
        },
        {
            "text": "10",
            "normalized_text": "10",
            "numeric_value": 10.0,
            "bbox_xyxy": [10, 90, 20, 100],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.82,
        },
    ]
    matches = [
        {"tick_position_px": 10, "candidate_bbox_xyxy": [10, 10, 20, 20]},
        {"tick_position_px": 30, "candidate_bbox_xyxy": [10, 30, 20, 40]},
        {"tick_position_px": 50, "candidate_bbox_xyxy": [10, 50, 20, 60]},
        {"tick_position_px": 70, "candidate_bbox_xyxy": [10, 70, 20, 80]},
        {"tick_position_px": 90, "candidate_bbox_xyxy": [10, 90, 20, 100]},
    ]

    corrected = correct_linear_sequence_outliers(candidates, matches)

    assert [row["text"] for row in corrected] == ["-10", "-5", "0", "5", "10"]
    assert corrected[1]["ocr_source"] == "tesseract-linear-corrected"


def test_correct_linear_sequence_outliers_restores_integer_digit_outlier() -> None:
    candidates = [
        {
            "text": "100",
            "normalized_text": "100",
            "numeric_value": 100.0,
            "bbox_xyxy": [10, 10, 20, 20],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.86,
        },
        {
            "text": "80",
            "normalized_text": "80",
            "numeric_value": 80.0,
            "bbox_xyxy": [10, 30, 20, 40],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.79,
        },
        {
            "text": "1",
            "normalized_text": "1",
            "numeric_value": 1.0,
            "bbox_xyxy": [10, 50, 20, 60],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.48,
        },
        {
            "text": "40",
            "normalized_text": "40",
            "numeric_value": 40.0,
            "bbox_xyxy": [10, 70, 20, 80],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.77,
        },
        {
            "text": "20",
            "normalized_text": "20",
            "numeric_value": 20.0,
            "bbox_xyxy": [10, 90, 20, 100],
            "ocr_source": "tesseract",
            "ocr_confidence": 0.74,
        },
    ]
    matches = [
        {"tick_position_px": 10, "candidate_bbox_xyxy": [10, 10, 20, 20]},
        {"tick_position_px": 30, "candidate_bbox_xyxy": [10, 30, 20, 40]},
        {"tick_position_px": 50, "candidate_bbox_xyxy": [10, 50, 20, 60]},
        {"tick_position_px": 70, "candidate_bbox_xyxy": [10, 70, 20, 80]},
        {"tick_position_px": 90, "candidate_bbox_xyxy": [10, 90, 20, 100]},
    ]

    corrected = correct_linear_sequence_outliers(candidates, matches)

    assert [row["text"] for row in corrected] == ["100", "80", "60", "40", "20"]
    assert corrected[2]["ocr_source"] == "tesseract-linear-corrected"


def test_recover_bottom_candidates_from_tick_windows_when_raw_candidates_collapse() -> None:
    image = Image.new("L", (240, 120), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((20, 84, 220, 84), fill=0, width=1)
    for x in (40, 80, 120, 160, 200):
        draw.line((x, 84, x, 91), fill=0, width=1)
    # Label-like blobs close enough to the axis that a wide dilation would merge them.
    for x in (68, 148):
        draw.rectangle((x, 89, x + 7, 103), fill=0)
        draw.rectangle((x + 10, 89, x + 19, 103), fill=0)

    gray = np.asarray(image, dtype=np.float32) / 255.0
    recovered = recover_bottom_candidates_from_tick_windows(
        gray_image=gray,
        axis_bbox_xyxy=[20, 78, 220, 108],
        tick_positions=[40, 80, 120, 160, 200],
        axis_line_coordinate=84,
    )

    assert len(recovered) == 2
    assert [int((row["bbox_xyxy"][0] + row["bbox_xyxy"][2]) / 2) for row in recovered] == [80, 160]
    assert all((row["bbox_xyxy"][3] - row["bbox_xyxy"][1]) >= 15 for row in recovered)


def test_tighten_candidate_bbox_to_numeric_ink_excludes_axis_side_line_fragment() -> None:
    image = Image.new("L", (120, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 27, 20), fill=0)
    draw.rectangle((30, 10, 37, 20), fill=0)
    draw.line((42, 8, 42, 24), fill=0, width=1)

    gray = np.asarray(image, dtype=np.float32) / 255.0
    tightened = tighten_candidate_bbox_to_numeric_ink(
        gray,
        [20, 8, 43, 24],
        "left",
        axis_line_coordinate=42,
    )

    assert tightened[0] == 20
    assert tightened[2] < 42


def test_extend_bbox_for_leading_sign_keeps_minus_to_the_left() -> None:
    image = Image.new("L", (80, 40), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 15, 15, 16), fill=0)
    draw.rectangle((20, 10, 24, 20), fill=0)
    draw.rectangle((28, 10, 32, 20), fill=0)

    gray = np.asarray(image, dtype=np.float32) / 255.0
    expanded = extend_bbox_for_leading_sign(
        gray,
        [20, 10, 33, 21],
        "left",
        [8, 8, 40, 24],
    )

    assert expanded[0] <= 12


def test_export_axis_ticks_retries_left_with_label_first_recovery_when_tick_count_is_too_low(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P01e_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (150, 140), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((58, 10, 58, 120), fill=0, width=1)
    for y in (24, 84):
        draw.line((58, y, 65, y), fill=0, width=1)
    for y in (24, 54, 84, 114):
        draw.rectangle((24, y, 30, y + 11), fill=0)
        draw.rectangle((34, y, 48, y + 11), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P01e_chart",
                "panel_path": "panels/p1/p1_F01-P01e_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [20, 10, 62, 122],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["tick_positions_px"] == [30, 60, 90, 118]
    assert len(left_axis["tick_matches"]) == 4


def test_export_axis_ticks_drops_unmatched_ticks_from_tick_positions(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P05_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (150, 120), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((30, 12, 30, 92), fill=0, width=1)
    for y in (24, 44, 64, 84):
        draw.line((30, y, 36, y), fill=0, width=1)
    for y in (20, 40, 60):
        draw.rectangle((6, y, 10, y + 9), fill=0)
        draw.rectangle((14, y, 20, y + 9), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P05_chart",
                "panel_path": "panels/p1/p1_F01-P05_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [0, 12, 36, 92],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["tick_positions_px"] == [24, 44, 64, 84]
    assert left_axis["tick_count"] == 4
    assert len(left_axis["tick_label_candidates"]) == 3
    assert left_axis["tick_matches"] == [
        {"tick_position_px": 24, "candidate_bbox_xyxy": [6, 20, 15, 30]},
        {"tick_position_px": 44, "candidate_bbox_xyxy": [6, 40, 15, 50]},
        {"tick_position_px": 64, "candidate_bbox_xyxy": [6, 60, 15, 70]},
    ]


def test_export_axis_ticks_uses_vision_bbox_fallback_when_heuristic_candidates_are_missing(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P07_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (120, 100), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((20, 10, 20, 80), fill=0, width=1)
    for y in (20, 40, 60):
        draw.line((20, y, 27, y), fill=0, width=1)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P07_chart",
                "panel_path": "panels/p1/p1_F01-P07_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [0, 10, 30, 82],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
        bbox_runner=_mock_bbox_runner(
            {
                "left:0:10:30:82": [
                    {"tick_position_px": 10, "bbox_xyxy": [2, 8, 12, 18]},
                    {"tick_position_px": 30, "bbox_xyxy": [2, 28, 12, 38]},
                    {"tick_position_px": 50, "bbox_xyxy": [2, 48, 12, 58]},
                ]
            }
        ),
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["tick_matches"] == [
        {"tick_position_px": 20, "candidate_bbox_xyxy": [2, 18, 12, 28]},
        {"tick_position_px": 40, "candidate_bbox_xyxy": [2, 38, 12, 48]},
        {"tick_position_px": 60, "candidate_bbox_xyxy": [2, 58, 12, 68]},
    ]


def test_export_axis_ticks_uses_monotonic_global_matching_instead_of_greedy(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_path = dataset_root / "panels" / "p1" / "p1_F01-P06_chart.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (180, 220), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((62, 12, 62, 198), fill=0, width=1)
    for y in (27, 59, 91, 124, 140, 156, 172, 188):
        draw.line((62, y, 68, y), fill=0, width=1)

    # Label-like groups corresponding to later ticks.
    for top in (22, 54, 86, 118, 150, 183):
        draw.rectangle((32, top, 36, top + 9), fill=0)
        draw.rectangle((40, top, 46, top + 9), fill=0)
    image.save(panel_path)

    gt_axes = {
        "schema_version": 1,
        "panels": [
            {
                "panel_id": "p1_F01-P06_chart",
                "panel_path": "panels/p1/p1_F01-P06_chart.png",
                "panel_kind": "chart",
                "is_dual_y": False,
                "review_status": "accept",
                "axis_left_bbox_xyxy": [26, 8, 72, 198],
                "axis_bottom_bbox_xyxy": None,
                "axis_right_bbox_xyxy": None,
                "annotator": "",
                "note": "",
                "created_at_utc": "2026-03-21T00:00:00Z",
                "updated_at_utc": "2026-03-21T00:00:00Z",
            }
        ],
    }
    _write_json(dataset_root / "links" / "gt_axes.json", gt_axes)

    payload = export_axis_ticks(
        dataset_root=dataset_root,
        gt_axes_path=dataset_root / "links" / "gt_axes.json",
        out_path=dataset_root / "links" / "gt_axis_ticks.json",
        ocr_backend="none",
    )

    left_axis = payload["panels"][0]["axes"][0]
    assert left_axis["tick_matches"] == [
        {"tick_position_px": 27, "candidate_bbox_xyxy": [32, 22, 41, 32]},
        {"tick_position_px": 59, "candidate_bbox_xyxy": [32, 54, 41, 64]},
        {"tick_position_px": 91, "candidate_bbox_xyxy": [32, 86, 41, 96]},
        {"tick_position_px": 124, "candidate_bbox_xyxy": [32, 118, 41, 128]},
        {"tick_position_px": 156, "candidate_bbox_xyxy": [32, 150, 41, 160]},
        {"tick_position_px": 188, "candidate_bbox_xyxy": [32, 183, 41, 193]},
    ]

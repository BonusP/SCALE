import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_phase2_localize_runner_generates_decision_and_critical_reasons(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    panel_index = {
        "panels": [
            {
                "panel_id": "p1_F01-P01_chart",
                "paper_id": "p1",
                "figure_id": "p1_F01",
                "panel_path": "panels/p1/p1_F01-P01_chart.png",
                "candidate_type": "chart",
                "plot_frame_area_ratio": 0.55,
                "plot_frame_boundary_ok": True,
                "axis_left_detected": True,
                "axis_bottom_detected": True,
                "bg_score": 0.90,
                "ct_score": 0.90,
                "tick_score": 0.90,
            },
            {
                "panel_id": "p1_F01-P02_chart",
                "paper_id": "p1",
                "figure_id": "p1_F01",
                "panel_path": "panels/p1/p1_F01-P02_chart.png",
                "candidate_type": "chart",
                "plot_frame_area_ratio": 0.03,
                "plot_frame_boundary_ok": True,
                "axis_left_detected": True,
                "axis_bottom_detected": True,
                "bg_score": 0.95,
                "ct_score": 0.95,
                "tick_score": 0.95,
            },
            {
                "panel_id": "p1_F01-P03_chart",
                "paper_id": "p1",
                "figure_id": "p1_F01",
                "panel_path": "panels/p1/p1_F01-P03_chart.png",
                "candidate_type": "chart",
                "plot_frame_area_ratio": 0.55,
                "plot_frame_boundary_ok": False,
                "axis_left_detected": False,
                "axis_bottom_detected": False,
                "bg_score": 0.80,
                "ct_score": 0.80,
                "tick_score": 0.80,
            },
            {
                "panel_id": "p1_F01-P04_chart",
                "paper_id": "p1",
                "figure_id": "p1_F01",
                "panel_path": "panels/p1/p1_F01-P04_chart.png",
                "candidate_type": "chart",
                "plot_frame_area_ratio": 0.50,
                "plot_frame_aspect_ratio": 3.8,
                "plot_frame_boundary_ok": True,
                "axis_left_detected": True,
                "axis_bottom_detected": True,
                "bg_score": 0.92,
                "ct_score": 0.92,
                "tick_score": 0.92,
            },
        ]
    }
    _write_json(dataset_root / "links" / "panel_index.json", panel_index)
    out_path = tmp_path / "phase2_localize_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_phase2_localize.py"),
            "--dataset-root",
            str(dataset_root),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    by_panel = {item["panel_id"]: item for item in report["panels"]}
    assert report["decision_counts"] == {"ACCEPT": 1, "WARN": 0, "REJECT": 3}
    assert report["critical_count"] == 3
    assert by_panel["p1_F01-P01_chart"]["decision"] == "ACCEPT"
    assert by_panel["p1_F01-P02_chart"]["decision"] == "REJECT"
    assert "AREA_RATIO_OUT_OF_RANGE" in by_panel["p1_F01-P02_chart"]["critical_reasons"]
    assert by_panel["p1_F01-P03_chart"]["decision"] == "REJECT"
    assert "PLOT_FRAME_BOUNDARY_OUTSIDE" in by_panel["p1_F01-P03_chart"]["critical_reasons"]
    assert "AXIS_LEFT_MISSING" in by_panel["p1_F01-P03_chart"]["critical_reasons"]
    assert "AXIS_BOTTOM_MISSING" in by_panel["p1_F01-P03_chart"]["critical_reasons"]
    assert by_panel["p1_F01-P04_chart"]["decision"] == "REJECT"
    assert "ASPECT_RATIO_OUT_OF_RANGE" in by_panel["p1_F01-P04_chart"]["critical_reasons"]


def test_phase2_localize_runner_uses_panel_image_scoring_when_hints_missing(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    panel_image_path = dataset_root / "panels" / "p1" / "p1_F01-P01_chart.png"
    panel_image_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (220, 160), color=240)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 200, 130), outline=10, width=2)
    for x in range(40, 190, 25):
        draw.line((x, 130, x, 125), fill=10, width=1)
    for y in range(30, 130, 20):
        draw.line((30, y, 35, y), fill=10, width=1)
    image.save(panel_image_path)

    panel_index = {
        "panels": [
            {
                "panel_id": "p1_F01-P01_chart",
                "paper_id": "p1",
                "figure_id": "p1_F01",
                "panel_path": "panels/p1/p1_F01-P01_chart.png",
                "candidate_type": "chart",
                "plot_frame_area_ratio": 0.55,
                "plot_frame_boundary_ok": True,
            }
        ]
    }
    _write_json(dataset_root / "links" / "panel_index.json", panel_index)
    out_path = tmp_path / "phase2_localize_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_phase2_localize.py"),
            "--dataset-root",
            str(dataset_root),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    panel = report["panels"][0]
    assert panel["score_source"] == "panel_image"
    assert 0.0 <= panel["score_breakdown"]["bg"] <= 1.0
    assert 0.0 <= panel["score_breakdown"]["ct"] <= 1.0
    assert 0.0 <= panel["score_breakdown"]["tick"] <= 1.0
    image_features = panel["evidence"]["image_features"]
    assert "tick_spacing_cv" in image_features
    assert "rectangularity" in image_features
    assert "orthogonality_error_deg" in image_features
    bbox = image_features["predicted_plot_frame_bbox_xyxy"]
    assert isinstance(bbox, list) and len(bbox) == 4
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]


def test_phase2_localize_runner_keeps_unknown_as_warn_not_reject(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    panel_index = {
        "panels": [
            {
                "panel_id": "p1_F09-P01_unknown",
                "paper_id": "p1",
                "figure_id": "p1_F09",
                "panel_path": "panels/p1/p1_F09-P01_unknown.png",
                "candidate_type": "unknown",
                "plot_frame_area_ratio": 0.01,
                "plot_frame_boundary_ok": False,
                "bg_score": 0.10,
                "ct_score": 0.10,
                "tick_score": 0.10,
            }
        ]
    }
    _write_json(dataset_root / "links" / "panel_index.json", panel_index)
    out_path = tmp_path / "phase2_localize_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_phase2_localize.py"),
            "--dataset-root",
            str(dataset_root),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    panel = report["panels"][0]
    assert panel["panel_kind"] == "unknown"
    assert panel["decision"] == "WARN"
    assert panel["critical_reasons"] == []
    first_issue = panel["issues"][0]
    assert set(first_issue.keys()) >= {
        "code",
        "severity",
        "reason",
        "at",
        "value",
        "stage",
        "timestamp",
    }
    issue_codes = {issue["code"] for issue in panel["issues"]}
    assert "NON_CHART_AREA_RATIO_OUT_OF_RANGE" in issue_codes
    assert "NON_CHART_PLOT_FRAME_BOUNDARY_OUTSIDE" in issue_codes


def test_phase2_localize_runner_applies_non_chart_branch_policy(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    panel_index = {
        "panels": [
            {
                "panel_id": "p2_F03-P01_image",
                "paper_id": "p2",
                "figure_id": "p2_F03",
                "panel_path": "panels/p2/p2_F03-P01_image.png",
                "candidate_type": "image",
                "plot_frame_area_ratio": 0.55,
                "plot_frame_aspect_ratio": 1.2,
                "plot_frame_boundary_ok": True,
                "axis_left_detected": False,
                "axis_bottom_detected": False,
                "bg_score": 0.10,
                "ct_score": 0.10,
                "tick_score": 0.10,
            }
        ]
    }
    _write_json(dataset_root / "links" / "panel_index.json", panel_index)
    out_path = tmp_path / "phase2_localize_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_phase2_localize.py"),
            "--dataset-root",
            str(dataset_root),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    panel = report["panels"][0]
    assert panel["branch"] == "IMAGE"
    assert panel["decision"] == "WARN"
    assert panel["critical_reasons"] == []
    assert panel["evidence"]["allow_reject"] is False
    assert panel["evidence"]["requires_axis"] is False


def test_phase2_localize_runner_uses_ocr_sidecar_for_text_geometry(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    panel_image_path = dataset_root / "panels" / "p3" / "p3_F01-P01_chart.png"
    panel_image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (200, 140), color=245).save(panel_image_path)

    panel_index = {
        "panels": [
            {
                "panel_id": "p3_F01-P01_chart",
                "paper_id": "p3",
                "figure_id": "p3_F01",
                "panel_path": "panels/p3/p3_F01-P01_chart.png",
                "candidate_type": "chart",
                "plot_frame_area_ratio": 0.55,
                "plot_frame_aspect_ratio": 1.4,
                "plot_frame_boundary_ok": True,
                "bg_score": 0.90,
                "ct_score": 0.90,
            }
        ]
    }
    ocr_index = {
        "ocr_blocks": [
            {
                "panel_id": "p3_F01-P01_chart",
                "text": "0.0",
                "bbox_xyxy": [10, 20, 30, 32],
            },
            {
                "panel_id": "p3_F01-P01_chart",
                "text": "1.0",
                "bbox_xyxy": [10, 52, 30, 64],
            },
            {
                "panel_id": "p3_F01-P01_chart",
                "text": "0",
                "bbox_xyxy": [40, 122, 50, 134],
            },
            {
                "panel_id": "p3_F01-P01_chart",
                "text": "10",
                "bbox_xyxy": [86, 122, 102, 134],
            },
        ]
    }

    _write_json(dataset_root / "links" / "panel_index.json", panel_index)
    _write_json(dataset_root / "links" / "ocr_index.json", ocr_index)
    out_path = tmp_path / "phase2_localize_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_phase2_localize.py"),
            "--dataset-root",
            str(dataset_root),
            "--ocr-index",
            str(dataset_root / "links" / "ocr_index.json"),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    panel = report["panels"][0]
    assert panel["score_source"] in {"panel_image+ocr", "ocr", "hints+ocr"}
    assert panel["decision"] == "ACCEPT"
    assert "AXIS_LEFT_MISSING" not in panel["critical_reasons"]
    assert "AXIS_BOTTOM_MISSING" not in panel["critical_reasons"]
    text_features = panel["evidence"]["text_features"]
    assert text_features["left_anchor_detected"] is True
    assert text_features["bottom_anchor_detected"] is True
    assert text_features["text_anchor_score"] >= 0.9

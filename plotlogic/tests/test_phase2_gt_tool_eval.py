"""Tests for phase2_gt_tool eval subcommand."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Import from the module under test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from phase2_gt_tool import compute_iou, cmd_eval


class TestEvalHelp:
    """Test that eval subcommand help exits cleanly."""

    def test_eval_help_exits_zero(self) -> None:
        """--help should exit 0."""
        result = subprocess.run(
            [sys.executable, "scripts/phase2_gt_tool.py", "eval", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Evaluate model predictions" in result.stdout or "eval" in result.stdout


class TestComputeIou:
    """Test IoU computation."""

    def test_eval_iou_computation_identical_boxes(self) -> None:
        """Identical boxes should have IoU = 1.0."""
        box = [10, 20, 50, 60]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_eval_iou_computation_no_overlap(self) -> None:
        """Non-overlapping boxes should have IoU = 0."""
        box_a = [0, 0, 10, 10]
        box_b = [20, 20, 30, 30]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_eval_iou_computation_partial_overlap(self) -> None:
        """Partially overlapping boxes."""
        # box_a: 0,0 to 10,10 (area 100)
        # box_b: 5,5 to 15,15 (area 100)
        # intersection: 5,5 to 10,10 (area 25)
        # union: 100 + 100 - 25 = 175
        # IoU: 25/175 = 0.1428...
        box_a = [0, 0, 10, 10]
        box_b = [5, 5, 15, 15]
        assert compute_iou(box_a, box_b) == pytest.approx(25 / 175)

    def test_eval_iou_computation_one_inside_other(self) -> None:
        """Box inside another."""
        # box_a: 0,0 to 20,20 (area 400)
        # box_b: 5,5 to 10,10 (area 25)
        # intersection: 5,5 to 10,10 (area 25)
        # union: 400
        # IoU: 25/400 = 0.0625
        box_a = [0, 0, 20, 20]
        box_b = [5, 5, 10, 10]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0625)


class TestClassificationMetrics:
    """Test classification metric computation."""

    def test_eval_classification_metrics(self, tmp_path: Path) -> None:
        """Test TP/FP/FN/TN and precision/recall/F1 computation."""
        # Create synthetic localize report
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 100, 100]},
            },
            {
                "panel_id": "p2",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 100, 100]},
            },
            {
                "panel_id": "p3",
                "panel_kind": "chart",
                "decision": "REJECT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 100, 100]},
            },
            {
                "panel_id": "p4",
                "panel_kind": "chart",
                "decision": "REJECT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 100, 100]},
            },
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        # Create GT file
        gt_panels = [
            {
                "panel_id": "p1",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p2",
                "review_status": "reject",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p3",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p4",
                "review_status": "reject",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
        ]
        gt_data = {"panels": gt_panels}
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt_data))

        # Run eval
        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(gt_path),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 0

        # Load and verify output
        eval_result = json.loads(out_path.read_text())
        # p1: y_true=1, y_pred=1 -> TP
        # p2: y_true=0, y_pred=1 -> FP
        # p3: y_true=1, y_pred=0 -> FN
        # p4: y_true=0, y_pred=0 -> TN
        cls = eval_result["classification"]["primary"]
        assert cls["tp"] == 1
        assert cls["fp"] == 1
        assert cls["fn"] == 1
        assert cls["tn"] == 1
        # precision = TP / (TP + FP) = 1/2 = 0.5
        assert cls["precision"] == pytest.approx(0.5)
        # recall = TP / (TP + FN) = 1/2 = 0.5
        assert cls["recall"] == pytest.approx(0.5)
        # f1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert cls["f1"] == pytest.approx(0.5)


class TestMissingPredBbox:
    """Test handling of missing predicted bbox."""

    def test_eval_missing_pred_bbox_iou_zero(self, tmp_path: Path) -> None:
        """Missing pred bbox should result in IoU = 0."""
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {},  # No predicted bbox
            },
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        gt_panels = [
            {
                "panel_id": "p1",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
        ]
        gt_data = {"panels": gt_panels}
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt_data))

        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(gt_path),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 0

        eval_result = json.loads(out_path.read_text())
        # Unconditional IoU should be 0
        assert eval_result["iou"]["unconditional"]["mean"] == pytest.approx(0.0)
        # Conditional should have count 0 (no valid pred bbox)
        assert eval_result["iou"]["conditional"]["count"] == 0


class TestSkipExcluded:
    """Test that skip/blank GT status is excluded from classification."""

    def test_eval_skip_excluded_from_classification(self, tmp_path: Path) -> None:
        """skip/blank review_status should be excluded from classification."""
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {},
            },
            {
                "panel_id": "p2",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {},
            },
            {
                "panel_id": "p3",
                "panel_kind": "chart",
                "decision": "REJECT",
                "evidence": {},
            },
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        gt_panels = [
            {
                "panel_id": "p1",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p2",
                "review_status": "skip",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },  # excluded
            {
                "panel_id": "p3",
                "review_status": "reject",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
        ]
        gt_data = {"panels": gt_panels}
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt_data))

        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(gt_path),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 0

        eval_result = json.loads(out_path.read_text())
        cls = eval_result["classification"]["primary"]
        # p1: y_true=1, y_pred=1 -> TP
        # p2: skip -> EXCLUDED
        # p3: y_true=0, y_pred=0 -> TN
        assert cls["tp"] == 1
        assert cls["tn"] == 1
        assert cls["fp"] == 0
        assert cls["fn"] == 0
        # per_panel should have 3 entries (all panels from pred)
        assert len(eval_result["per_panel"]) == 3


class TestStratifiedByPanelKind:
    """Test stratification by panel_kind."""

    def test_eval_stratified_by_panel_kind(self, tmp_path: Path) -> None:
        """Classification metrics should be stratified by panel_kind."""
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {},
            },
            {
                "panel_id": "p2",
                "panel_kind": "chart",
                "decision": "REJECT",
                "evidence": {},
            },
            {
                "panel_id": "p3",
                "panel_kind": "schematic",
                "decision": "ACCEPT",
                "evidence": {},
            },
            {
                "panel_id": "p4",
                "panel_kind": "schematic",
                "decision": "REJECT",
                "evidence": {},
            },
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        gt_panels = [
            {
                "panel_id": "p1",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p2",
                "review_status": "reject",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p3",
                "review_status": "reject",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p4",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
        ]
        gt_data = {"panels": gt_panels}
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt_data))

        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(gt_path),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 0

        eval_result = json.loads(out_path.read_text())
        by_kind = eval_result["by_panel_kind"]
        assert "chart" in by_kind
        assert "schematic" in by_kind
        # chart: p1=TP, p2=TN
        assert by_kind["chart"]["classification"]["primary"]["tp"] == 1
        assert by_kind["chart"]["classification"]["primary"]["tn"] == 1
        # schematic: p3=FP, p4=FN
        assert by_kind["schematic"]["classification"]["primary"]["fp"] == 1
        assert by_kind["schematic"]["classification"]["primary"]["fn"] == 1


class TestStrictMetrics:
    """Test strict classification metrics."""

    def test_eval_strict_metrics(self, tmp_path: Path) -> None:
        """Strict metrics: y_pred_strict = (decision == ACCEPT)."""
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {},
            },
            {
                "panel_id": "p2",
                "panel_kind": "chart",
                "decision": "WARN",
                "evidence": {},
            },
            {
                "panel_id": "p3",
                "panel_kind": "chart",
                "decision": "REJECT",
                "evidence": {},
            },
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        gt_panels = [
            {
                "panel_id": "p1",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p2",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
            {
                "panel_id": "p3",
                "review_status": "reject",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
        ]
        gt_data = {"panels": gt_panels}
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt_data))

        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(gt_path),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 0

        eval_result = json.loads(out_path.read_text())
        # Primary: ACCEPT/WARN -> 1, REJECT -> 0
        # p1: y_true=1, y_pred=1 -> TP
        # p2: y_true=1, y_pred=1 -> TP (WARN counts as positive in primary)
        # p3: y_true=0, y_pred=0 -> TN
        primary = eval_result["classification"]["primary"]
        assert primary["tp"] == 2
        assert primary["tn"] == 1

        # Strict: ACCEPT -> 1, else -> 0
        # p1: y_true=1, y_pred=1 -> TP
        # p2: y_true=1, y_pred=0 -> FN (WARN doesn't count in strict)
        # p3: y_true=0, y_pred=0 -> TN
        strict = eval_result["classification"]["strict"]
        assert strict["tp"] == 1
        assert strict["fn"] == 1
        assert strict["tn"] == 1


class TestIouConditional:
    """Test conditional IoU (only rows with valid pred bbox)."""

    def test_eval_iou_conditional(self, tmp_path: Path) -> None:
        """Conditional IoU should only include panels with valid pred bbox."""
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 100, 100]},
            },
            {
                "panel_id": "p2",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {},  # No bbox
            },
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        gt_panels = [
            {
                "panel_id": "p1",
                "review_status": "accept",
                "gt_bbox_xyxy": [10, 10, 110, 110],
            },
            {
                "panel_id": "p2",
                "review_status": "accept",
                "gt_bbox_xyxy": [0, 0, 100, 100],
            },
        ]
        gt_data = {"panels": gt_panels}
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt_data))

        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(gt_path),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 0

        eval_result = json.loads(out_path.read_text())
        # Unconditional: p1 IoU + p2 IoU=0
        # p1: [0,0,100,100] vs [10,10,110,110] -> intersection [10,10,100,100] = 90*90=8100
        #     union = 10000 + 10000 - 8100 = 11900 -> IoU = 0.6806...
        # p2: no pred bbox -> IoU = 0
        # mean = (0.6806... + 0) / 2 = 0.3403...
        assert eval_result["iou"]["unconditional"]["mean"] == pytest.approx(
            (8100 / 11900) / 2, rel=0.01
        )
        # Conditional: only p1
        assert eval_result["iou"]["conditional"]["count"] == 1
        assert eval_result["iou"]["conditional"]["mean"] == pytest.approx(
            8100 / 11900, rel=0.01
        )


class TestNoGtFile:
    """Test behavior when GT file doesn't exist."""

    def test_eval_no_gt_file_returns_error(self, tmp_path: Path) -> None:
        """Should return error when GT file doesn't exist."""
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {},
            }
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(tmp_path / "nonexistent_gt.json"),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 1  # Error exit code


class TestPerPanelOutput:
    """Test per_panel output structure."""

    def test_eval_per_panel_output(self, tmp_path: Path) -> None:
        """Each panel should have per-panel metrics in output."""
        pred_panels = [
            {
                "panel_id": "p1",
                "panel_kind": "chart",
                "decision": "ACCEPT",
                "evidence": {"predicted_plot_frame_bbox_xyxy": [0, 0, 100, 100]},
            },
        ]
        pred_report = {"panels": pred_panels}
        pred_path = tmp_path / "pred.json"
        pred_path.write_text(json.dumps(pred_report))

        gt_panels = [
            {
                "panel_id": "p1",
                "review_status": "accept",
                "gt_bbox_xyxy": [10, 10, 110, 110],
            },
        ]
        gt_data = {"panels": gt_panels}
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt_data))

        out_path = tmp_path / "eval.json"
        args = type(
            "Args",
            (),
            {
                "input": str(pred_path),
                "gt": str(gt_path),
                "dataset_root": str(tmp_path),
                "iou_threshold": 0.5,
                "out": str(out_path),
            },
        )()
        result = cmd_eval(args)
        assert result == 0

        eval_result = json.loads(out_path.read_text())
        assert len(eval_result["per_panel"]) == 1
        pp = eval_result["per_panel"][0]
        assert pp["panel_id"] == "p1"
        assert pp["panel_kind"] == "chart"
        assert pp["decision"] == "ACCEPT"
        assert "y_true" in pp
        assert "y_pred_primary" in pp
        assert "y_pred_strict" in pp
        assert "iou" in pp

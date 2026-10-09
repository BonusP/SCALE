import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_apply_phase2_review_sheet_converts_no_problem_to_accept(tmp_path: Path) -> None:
    report_path = tmp_path / "phase2_localize_report.json"
    reviewed_path = tmp_path / "phase2_localize_report_reviewed.json"
    review_sheet_path = tmp_path / "phase2_axis_missing_review_sheet.csv"

    report = {
        "panels_total": 5,
        "decision_counts": {"ACCEPT": 0, "WARN": 0, "REJECT": 5},
        "critical_count": 5,
        "panels": [
            {
                "panel_id": "p1_F01-P01_chart",
                "paper_id": "p1",
                "decision": "REJECT",
                "critical_reasons": ["AXIS_LEFT_MISSING"],
                "issues": [{"code": "AXIS_LEFT_MISSING", "severity": "CRITICAL"}],
            },
            {
                "panel_id": "p1_F01-P02_chart",
                "paper_id": "p1",
                "decision": "REJECT",
                "critical_reasons": ["AXIS_BOTTOM_MISSING"],
                "issues": [{"code": "AXIS_BOTTOM_MISSING", "severity": "CRITICAL"}],
            },
            {
                "panel_id": "p1_F01-P03_chart",
                "paper_id": "p1",
                "decision": "REJECT",
                "critical_reasons": ["AXIS_LEFT_MISSING"],
                "issues": [{"code": "AXIS_LEFT_MISSING", "severity": "CRITICAL"}],
            },
            {
                "panel_id": "p1_F01-P04_chart",
                "paper_id": "p1",
                "decision": "REJECT",
                "critical_reasons": ["AXIS_BOTTOM_MISSING"],
                "issues": [{"code": "AXIS_BOTTOM_MISSING", "severity": "CRITICAL"}],
            },
            {
                "panel_id": "p1_F01-P05_chart",
                "paper_id": "p1",
                "decision": "REJECT",
                "critical_reasons": ["AXIS_LEFT_MISSING"],
                "issues": [{"code": "AXIS_LEFT_MISSING", "severity": "CRITICAL"}],
            },
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    review_sheet_path.write_text(
        "\n".join(
            [
                "paper_id,panel_id,panel_path,decision,critical_reasons,score_final,score_bg,score_ct,score_tick,axis_left_detected,axis_bottom_detected,left_tick_count,bottom_tick_count,prev_review_status,prev_user_opinion,prev_user_note,review_status,user_opinion,user_note",
                "p1,p1_F01-P01_chart,panels/p1/p1_F01-P01_chart.png,REJECT,AXIS_LEFT_MISSING,0.6,0.6,0.6,0.6,FALSE,TRUE,1,5,,,,reviewed,No Problem,looks fine",
                "p1,p1_F01-P02_chart,panels/p1/p1_F01-P02_chart.png,REJECT,AXIS_BOTTOM_MISSING,0.6,0.6,0.6,0.6,TRUE,FALSE,5,1,,,,reviewed,downgrade_warn,bar chart 스타일",
                "p1,p1_F01-P03_chart,panels/p1/p1_F01-P03_chart.png,REJECT,AXIS_LEFT_MISSING,0.6,0.6,0.6,0.6,FALSE,TRUE,1,5,,,,reviewed,not_chart,schematic인데 chart로 분류됨",
                "p1,p1_F01-P04_chart,panels/p1/p1_F01-P04_chart.png,REJECT,AXIS_BOTTOM_MISSING,0.6,0.6,0.6,0.6,TRUE,FALSE,5,1,,,,reviewed,uncertain,전혀 문제 없는 chart",
                "p1,p1_F01-P05_chart,panels/p1/p1_F01-P05_chart.png,REJECT,AXIS_LEFT_MISSING,0.6,0.6,0.6,0.6,FALSE,TRUE,1,5,,Accept,looks fine,,,",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "apply_phase2_review_sheet.py"),
            "--report",
            str(report_path),
            "--review-sheet",
            str(review_sheet_path),
            "--out",
            str(reviewed_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    by_panel = {panel["panel_id"]: panel for panel in reviewed["panels"]}
    assert by_panel["p1_F01-P01_chart"]["decision"] == "ACCEPT"
    assert by_panel["p1_F01-P01_chart"]["critical_reasons"] == []
    assert by_panel["p1_F01-P01_chart"]["original_decision"] == "REJECT"
    assert by_panel["p1_F01-P02_chart"]["decision"] == "WARN"
    assert by_panel["p1_F01-P03_chart"]["decision"] == "WARN"
    assert by_panel["p1_F01-P03_chart"]["manual_reclassify_candidate"] == "SCHEMATIC"
    assert by_panel["p1_F01-P04_chart"]["decision"] == "ACCEPT"
    assert by_panel["p1_F01-P04_chart"]["review"]["inferred_opinion"] == "no_problem"
    assert by_panel["p1_F01-P05_chart"]["decision"] == "ACCEPT"
    assert by_panel["p1_F01-P05_chart"]["review"]["opinion_source"] == "prev_user_opinion"
    assert reviewed["decision_counts"] == {"ACCEPT": 3, "WARN": 2, "REJECT": 0}
    assert reviewed["critical_count"] == 0

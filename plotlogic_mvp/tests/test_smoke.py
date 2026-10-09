import json
from pathlib import Path

import pytest

from PIL import Image
from PIL import ImageDraw

from app.services.codex_direct import to_curve_series_data, validation_report
from app.services.ocr_geometry import build_ocr_text_geometry
from app.services.run_methods import apply_chart_id_suffix, method_for_chart_id, resolve_method
from app.services.bbox_validation import validate_figure_analysis
from app.services.chart_geometry import repair_chart_panel_geometry
from app.services.mask_correction import run_llm_mask_correction
from app.services.plotlogic_bridge import export_curve_series_csv, overrides_from_vision
from app import config
from app.services.vision import (
    BASELINE_PROMPT_TEMPLATE,
    analyze_figure_panels,
    current_prompt_metadata,
    prompt_sha256_for,
    prompt_templates,
)
from app.services.visual_qa import ensure_visual_qa_artifacts


def test_export_curve_series_csv(tmp_path: Path) -> None:
    source = tmp_path / "curve_series_data.json"
    target = tmp_path / "curve_series_data.csv"
    source.write_text(
        """
        {
          "panel_id": "p1",
          "series": [
            {
              "trace_id": "t1",
              "series_id": "s1",
              "legend_label": "750 C",
              "curve_role": "voltage_curve",
              "axis_binding": "left_y",
              "points_data": [[0, 1.0], [1, 0.8]]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    export_curve_series_csv(source, target)

    text = target.read_text(encoding="utf-8")
    assert "panel_id,trace_id,series_id" in text
    assert "p1,t1,s1,750 C,voltage_curve,left_y,0,0,1.0" in text


def test_chart_panel_repair_prefers_component_bounds() -> None:
    image = Image.new("RGB", (1000, 800), "white")
    panel = {
        "panel_type": "chart",
        "plot_frame_bbox_panel_px": [150, 130, 560, 430],
        "axes": [
            {"axis_role": "left", "label_bbox_panel_px": [70, 250, 110, 350]},
            {"axis_role": "right", "label_bbox_panel_px": [590, 200, 650, 330]},
            {"axis_role": "bottom", "label_bbox_panel_px": [260, 470, 450, 520]},
        ],
        "legend_region_bbox_panel_px": [420, 145, 545, 190],
    }

    repaired = repair_chart_panel_geometry(image, panel, [100, 100, 600, 500])
    bbox = repaired["panel_bbox"]

    assert bbox[0] <= 70
    assert bbox[2] >= 650
    assert bbox[3] >= 520
    assert bbox[2] < 710
    assert bbox[3] < 570


def test_bbox_validation_flags_panel_local_axis_bbox(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (900, 700), "white").save(image_path)
    analysis = {
        "panels": [
            {
                "panel_index": 1,
                "panel_label": "(a)",
                "panel_type": "chart",
                "panel_bbox_figure_px": [300, 100, 800, 600],
                "is_chart": True,
                "is_extraction_target": True,
                "is_target_dual_y": True,
                "plot_frame_bbox_panel_px": [30, 40, 460, 420],
                "axes": [
                    {"axis_role": "left", "axis_line_bbox_panel_px": [30, 40, 33, 420], "ticks": []},
                    {"axis_role": "bottom", "axis_line_bbox_panel_px": [30, 420, 460, 423], "ticks": []},
                    {"axis_role": "right", "axis_line_bbox_panel_px": [457, 40, 460, 420], "ticks": []},
                ],
                "legend_items": [],
                "issues": [],
            }
        ]
    }

    validated = validate_figure_analysis(image_path, analysis)
    issues = validated["panels"][0]["issues"]

    assert "BBOX_COORD_SPACE_SUSPECT_PANEL_LOCAL:plot_frame_bbox_panel_px" in issues
    assert "BBOX_PLOT_FRAME_OUTSIDE_PANEL_BBOX" in issues


def test_legend_exclusion_uses_item_boxes_not_whole_region() -> None:
    vision = {
        "is_target_dual_y": True,
        "plot_frame_bbox_panel_px": [100, 50, 500, 350],
        "axis_labels": {"x": "", "left_y": "", "right_y": ""},
        "axes": [],
        "legend_region_bbox_panel_px": [300, 80, 490, 180],
        "legend_items": [
            {
                "legend_label": "A",
                "label_bbox_panel_px": [350, 90, 390, 110],
                "swatch_color_rgb": [255, 0, 0],
                "swatch_bbox_panel_px": [320, 92, 332, 104],
                "marker_shape": "circle",
                "confidence": 0.9,
            },
            {
                "legend_label": "B",
                "label_bbox_panel_px": [350, 130, 390, 150],
                "swatch_color_rgb": [0, 0, 255],
                "swatch_bbox_panel_px": [320, 132, 332, 144],
                "marker_shape": "circle",
                "confidence": 0.9,
            },
        ],
        "plot_annotation_regions": [
            {
                "annotation_text": "legend",
                "bbox_panel_px": [300, 80, 490, 180],
                "reason": "Legend markers and labels should be excluded from curve extraction.",
            },
            {
                "annotation_text": "800 C",
                "bbox_panel_px": [120, 70, 170, 90],
                "reason": "Chart annotation inside plot frame.",
            },
        ],
    }

    _, _, _, _, exclusions = overrides_from_vision("p1", "panel.png", vision)

    assert len(exclusions) == 3
    assert [200, 30, 390, 130] not in exclusions
    assert exclusions[0] == [218, 38, 292, 62]
    assert exclusions[1] == [218, 78, 292, 102]
    assert [20, 20, 70, 40] in exclusions




# chart_fmailiy 다시 4개로 수정
# categorical 처리를 위한 수정
BASELINE_PROMPT_SHA256 = "e9a7a57b9f667503c468f130231ed2445884c19038b307f4608d5465ac511284"
# Bumped four times:
#  - registry_version 2026-09-16.mvp2 added single_axis_line_chart and dual_y_generic_line_chart.
#  - registry_version 2026-09-16.mvp3 added bar_chart and pie_chart (read directly by codex, no
#    pixel tracing).
#  - registry_version 2026-09-16.mvp4 added categorical_line_chart (a line/scatter chart whose x
#    axis is category labels, not numbers — also read directly by codex, like bar_chart).
#  - registry_version 2026-09-17.mvp5 removed bar_chart and pie_chart again: only colored line
#    charts (single-axis or dual-axis) are extraction targets now. A line chart with a categorical
#    x axis still goes through the same direct-read path bar/pie charts used to use. The line
#    families were also tightened to require curves distinguishable by color.
# The baseline .md template body is unchanged every time; only the shared registry it renders
# {chart_family_rules} from moved, which legitimately invalidates old cached classifications so
# they get re-classified against the current family list.


def test_baseline_prompt_is_unchanged() -> None:
    """The baseline prompt is the comparison point, so it must stay byte-identical.

    Every classification stored in data/ was produced by it. Editing it would silently
    move the baseline and make prompt experiments unmeasurable — put changes in a new
    template and select it with PLOTLOGIC_MVP_VISION_PROMPT instead.
    """
    templates = prompt_templates()
    assert templates[0].name == BASELINE_PROMPT_TEMPLATE
    assert prompt_sha256_for(templates[0]) == BASELINE_PROMPT_SHA256


# ###수정 별도 실험의 기본 prompt만 CMSL로 변경되었음을 검증합니다.
def test_default_prompt_is_the_cmsl_experiment() -> None:
    assert config.VISION_PROMPT_TEMPLATE == "figure_panel_classifier_cmsl.md"
    assert current_prompt_metadata()["prompt_sha256"] == prompt_sha256_for(config.PROMPTS_ROOT / config.VISION_PROMPT_TEMPLATE)
# ###


def test_prompt_variant_is_recorded_per_run(tmp_path: Path, monkeypatch) -> None:
    """A run must record which prompt it used, or baseline and variant jobs are indistinguishable."""
    monkeypatch.setattr("app.config.USE_CODEX", False)
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (200, 150), "white").save(image_path)
    variants = [path.name for path in prompt_templates()]
    assert len(variants) > 1, "no variant available to distinguish from the baseline"

    for variant in variants:
        analysis = analyze_figure_panels(image_path, variant)
        metadata = analysis["prompt_metadata"]
        assert metadata["prompt_variant"] == variant
        assert metadata["prompt_sha256"] == prompt_sha256_for(config.PROMPTS_ROOT / variant)

    # Omitting the variant keeps the configured default rather than guessing.
    # ###수정 prompt 생략 시 실험 설정의 CMSL 기본값을 사용해야 합니다.
    assert analyze_figure_panels(image_path)["prompt_metadata"]["prompt_variant"] == config.VISION_PROMPT_TEMPLATE
    # ###


def test_prompt_variants_are_documented() -> None:
    catalogue = (Path(config.PROMPTS_ROOT) / "PROMPTS.md").read_text(encoding="utf-8")
    for path in prompt_templates():
        assert path.name in catalogue, f"{path.name} is missing from prompts/PROMPTS.md"
        assert prompt_sha256_for(path) in catalogue, f"{path.name} sha is missing from prompts/PROMPTS.md"


def test_methods_write_to_separate_chart_ids() -> None:
    """Two methods on one paper must both survive; chart_id is the primary key."""
    base_id = "0009_img001_P01_P01_chart_f044a47c"
    baseline = resolve_method("baseline")
    experiment = resolve_method("ocr_v2")

    baseline_id = apply_chart_id_suffix(base_id, baseline)
    experiment_id = apply_chart_id_suffix(base_id, experiment)

    assert baseline_id == base_id, "baseline chart ids must keep their historical form"
    assert experiment_id != baseline_id
    assert method_for_chart_id(baseline_id) == "baseline"
    assert method_for_chart_id(experiment_id) == "ocr_v2"


def test_baseline_method_uses_baseline_prompt_without_ocr() -> None:
    baseline = resolve_method("baseline")
    assert baseline.prompt_variant == BASELINE_PROMPT_TEMPLATE
    assert baseline.use_ocr_text is False

    experiment = resolve_method("ocr_v2")
    assert experiment.prompt_variant != BASELINE_PROMPT_TEMPLATE
    assert experiment.use_ocr_text is True

    with pytest.raises(ValueError):
        resolve_method("typo")


def test_codex_direct_table_converts_to_pipeline_series() -> None:
    """The direct method must emit the same shape the rest of the app reads."""
    table = {
        "panel_summary": "iv curves",
        "x_axis": {"label": "Current density", "unit": "mA cm^-2", "min_value": 0, "max_value": 700},
        "left_y_axis": {"label": "Voltage", "unit": "V", "min_value": 0, "max_value": 1},
        "right_y_axis": {"label": "Power density", "unit": "mW cm^-2", "min_value": 0, "max_value": 140},
        "series": [
            {
                "legend_label": "700 C",
                "curve_role": "voltage_curve",
                "axis_binding": "left_y",
                "marker_shape": None,
                "confidence": 0.9,
                # Deliberately unsorted with a duplicate x and a non-numeric point.
                "points": [{"x": 80, "y": 0.72}, {"x": 0, "y": 0.86}, {"x": 80, "y": 0.70}, {"x": None, "y": 1}],
                "issues": [],
            },
            {
                "legend_label": "ghost",
                "curve_role": "other",
                "axis_binding": "sideways",
                "marker_shape": None,
                "confidence": 0.1,
                "points": [],
                "issues": ["not visible"],
            },
        ],
        "issues": ["right axis label cropped"],
    }

    series_data = to_curve_series_data("panel_1", table)
    report = validation_report(series_data, table)
    first, second = series_data["series"]

    assert series_data["panel_id"] == "panel_1"
    assert first["points_data"] == [[0.0, 0.86], [80.0, 0.72]], "x must sort and de-duplicate"
    assert second["axis_binding"] == "left_y", "an unknown axis must fall back, not propagate"
    assert report["checks"] == {
        "final_series_count": 2,
        "point_count": 2,
        "series_without_points": 1,
        "left_y_series": 2,
        "right_y_series": 0,
    }
    assert "right axis label cropped" in report["issues"]
    assert "SERIES_WITHOUT_POINTS:ghost" in report["issues"]


def test_codex_direct_is_a_separate_method() -> None:
    direct = resolve_method("codex_direct")
    assert direct.extraction == "codex_direct"
    assert direct.chart_id_suffix == "direct"
    assert resolve_method("baseline").extraction == "plotlogic"
    assert method_for_chart_id("0015_img001_P01_x__direct") == "codex_direct"


def _ocr_geometry_fixture() -> tuple[dict, list[dict]]:
    frame = [60, 30, 360, 240]
    tick_xs = [60, 160, 260, 360]
    # Codex read the third label as 2.5; OCR reads the same box as 2.0.
    codex_values = [0.0, 1.0, 2.5, 3.0]
    vision = {
        "plot_frame_bbox_panel_px": frame,
        "axes": [
            {
                "axis_role": "bottom",
                "axis_line_bbox_panel_px": [60, 239, 360, 242],
                "label_bbox_panel_px": [150, 275, 270, 295],
                "ticks": [
                    {
                        "label_text": str(value),
                        "label_numeric_value": value,
                        "tick_position_px": x,
                        "label_bbox_panel_px": [x - 10, 250, x + 10, 264],
                    }
                    for value, x in zip(codex_values, tick_xs)
                ],
            }
        ],
    }
    ocr_items = [
        {"text": str(value), "bbox": [x - 9, 251, x + 9, 263], "confidence": 0.99, "numeric_value": value}
        for value, x in zip([0.0, 1.0, 2.0, 3.0], tick_xs)
    ]
    ocr_items.append(
        {"text": "Current density", "bbox": [155, 278, 268, 293], "confidence": 0.98, "numeric_value": None}
    )
    return vision, ocr_items


def test_ocr_geometry_replaces_text_boxes_and_disputes_a_misread_label() -> None:
    vision, ocr_items = _ocr_geometry_fixture()

    geom = build_ocr_text_geometry(vision, ocr_items=ocr_items)
    report = geom["axes_report"][0]
    axis = geom["panel_vision"]["axes"][0]

    assert report["ocr"]["matched"] == 4
    assert report["tick_labels_from_ocr"] == 4
    assert geom["summary"]["text_boxes_replaced"] == {"tick_label": 4, "axis_label": 1, "legend_label": 0}
    assert axis["ticks"][0]["label_bbox_panel_px"] == [51, 251, 69, 263]
    assert axis["ticks"][0]["codex_label_bbox_panel_px"] == [50, 250, 70, 264]
    assert axis["label_bbox_panel_px"] == [155, 278, 268, 293]
    assert [tick["ocr_value_agrees"] for tick in axis["ticks"]] == [True, True, False, True]
    assert report["ocr"]["value_mismatches"] == ["codex=2.5/ocr=2.0"]
    assert any("OCR_VALUE_MISMATCH" in issue for issue in geom["issues"])


def test_ocr_geometry_leaves_non_text_geometry_untouched() -> None:
    vision, ocr_items = _ocr_geometry_fixture()

    geom = build_ocr_text_geometry(vision, ocr_items=ocr_items)
    axis = geom["panel_vision"]["axes"][0]

    assert geom["geometry_source"] == "codex_pipeline"
    assert geom["panel_vision"]["plot_frame_bbox_panel_px"] == vision["plot_frame_bbox_panel_px"]
    assert axis["axis_line_bbox_panel_px"] == [60, 239, 360, 242]
    assert [tick["tick_position_px"] for tick in axis["ticks"]] == [60, 160, 260, 360]


def test_visual_qa_flags_fragmented_mask_and_plateau_trace(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    mask_dir = result_dir / "masks"
    mask_dir.mkdir(parents=True)
    mask = Image.new("L", (80, 40), 0)
    draw = ImageDraw.Draw(mask)
    for x in range(2, 78, 5):
        draw.rectangle((x, 10, x + 1, 11), fill=255)
    mask.save(mask_dir / "color_01.png")
    (result_dir / "curve_traces_crop_px.json").write_text(
        '{"traces": [{"trace_id": "raw1", "final_trace_status": "rejected", "points": [[0, 1], [1, 1]]}]}',
        encoding="utf-8",
    )
    plateau_points = [[x, 10] for x in range(45)] + [[45, 12], [46, 13]]
    (result_dir / "final_curve_traces_crop_px.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "trace_id": "final1",
                        "series_id": "color_01",
                        "curve_role": "voltage_curve",
                        "axis_binding_hint": "left_y",
                        "points": plateau_points,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = ensure_visual_qa_artifacts(result_dir)
    mask_qa = (result_dir / "mask_quality_qa.json").read_text(encoding="utf-8")
    trace_qa = (result_dir / "trace_overlay_qa.json").read_text(encoding="utf-8")

    assert summary["status"] == "needs_review"
    assert "FRAGMENTED_OR_MULTI_COMPONENT_MASK" in mask_qa
    assert "LONG_HORIZONTAL_PLATEAU" in trace_qa


def test_mask_correction_reconstructs_small_gaps_and_removes_dots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.config.USE_CODEX", False)
    result_dir = tmp_path / "result"
    mask_dir = result_dir / "masks"
    mask_dir.mkdir(parents=True)
    mask = Image.new("L", (80, 40), 0)
    draw = ImageDraw.Draw(mask)
    draw.line((5, 8, 30, 15), fill=255, width=1)
    draw.line((38, 17, 70, 24), fill=255, width=1)
    draw.point((6, 35), fill=255)
    draw.point((72, 3), fill=255)
    mask.save(mask_dir / "color_01.png")

    summary = run_llm_mask_correction(result_dir)

    assert summary["status"] == "skipped"
    assert summary["bridge_count"] >= 1
    assert summary["removed_component_count"] >= 1
    assert (result_dir / "cleaned_masks" / "color_01.png").exists()
    assert (result_dir / "reconstructed_masks" / "color_01.png").exists()
    assert (result_dir / "mask_reconstruction_comparison.png").exists()

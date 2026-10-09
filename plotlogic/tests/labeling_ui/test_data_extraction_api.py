import json
from pathlib import Path

from PIL import Image, ImageDraw


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _setup_dataset(tmp_path: Path) -> tuple[Path, str]:
    dataset_root = tmp_path / "dataset"
    panel_id = "p1_F01-P01_chart"
    panel_rel = "panels/p1/p1_F01-P01_chart.png"
    panel_path = dataset_root / panel_rel
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (140, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 10, 120, 85], outline="black")
    draw.line([(25, 25), (45, 35), (70, 48), (110, 68)], fill=(240, 40, 40), width=2)
    draw.line([(25, 75), (50, 48), (75, 35), (110, 70)], fill=(40, 80, 230), width=2)
    image.save(panel_path)

    _write_json(
        dataset_root / "links" / "gt_plots.json",
        {
            "schema_version": 1,
            "panels": [
                {
                    "panel_id": panel_id,
                    "panel_path": panel_rel,
                    "panel_kind": "chart",
                    "review_status": "accept",
                    "gt_bbox_xyxy": [20, 10, 120, 85],
                }
            ],
        },
    )
    _write_json(
        dataset_root / "links" / "gt_axes.json",
        {
            "schema_version": 1,
            "panels": [
                {
                    "panel_id": panel_id,
                    "panel_path": panel_rel,
                    "panel_kind": "chart",
                    "is_dual_y": True,
                    "review_status": "accept",
                    "axis_left_bbox_xyxy": [0, 10, 22, 85],
                    "axis_bottom_bbox_xyxy": [20, 84, 120, 99],
                    "axis_right_bbox_xyxy": [118, 10, 139, 85],
                }
            ],
        },
    )
    _write_json(dataset_root / "links" / "gt_axis_ticks.json", {"schema_version": 1, "panels": []})
    _write_json(
        dataset_root / "extractions" / "manual_axis_pairs.json",
        {
            panel_id: {
                "bottom": [[20, 0.0], [120, 1.0]],
                "left": [[10, 1.0], [85, 0.0]],
                "right": [[10, 2.0], [85, 0.0]],
            }
        },
    )
    return dataset_root, panel_id


def test_data_extraction_route_lists_and_runs_panel(tmp_path: Path) -> None:
    from labeling_ui.routes import data_extraction

    dataset_root, panel_id = _setup_dataset(tmp_path)
    out_root = dataset_root / "extractions" / "data_series_runs" / "ui"
    data_extraction.configure(
        get_dataset_root=lambda: dataset_root,
        get_project_root=lambda: tmp_path,
        get_out_root=lambda: out_root,
        get_manual_axis_pairs_path=lambda: dataset_root / "extractions" / "manual_axis_pairs.json",
    )

    listing = data_extraction.list_data_extraction_panels()
    assert listing["meta"]["total"] == 1
    assert listing["panels"][0]["panel_id"] == panel_id
    assert listing["panels"][0]["run_exists"] is False

    result = data_extraction.run_data_extraction(
        data_extraction.DataExtractionRunRequest(panel_id=panel_id, max_colors=4)
    )
    run = result["run"]
    assert run["run_exists"] is True
    assert run["validation"]["checks"]["x_axis_mapping_ready"] is True
    assert run["validation"]["checks"]["right_y_axis_mapping_ready"] is True
    assert (out_root / panel_id / "overlay.png").exists()


def test_data_extraction_run_can_include_codex_vision_step(
    tmp_path: Path, monkeypatch
) -> None:
    from labeling_ui.routes import data_extraction

    dataset_root, panel_id = _setup_dataset(tmp_path)
    out_root = dataset_root / "extractions" / "data_series_runs" / "ui"
    data_extraction.configure(
        get_dataset_root=lambda: dataset_root,
        get_project_root=lambda: tmp_path,
        get_out_root=lambda: out_root,
        get_manual_axis_pairs_path=lambda: dataset_root / "extractions" / "manual_axis_pairs.json",
    )

    def fake_codex(image_path, request):
        return {
            "schema_version": 1,
            "generated_at_utc": "2026-05-04T00:00:00Z",
            "mode": request.mode,
            "model": request.model,
            "source_image_rel": "plot_crop.png",
            "result": {
                "chart_family": "dual_y_power_density",
                "axis_labels": {"title": "", "x": "Current density", "left_y": "Voltage", "right_y": "Power density"},
                "legend_region_bbox_crop_px": [72, 8, 98, 24],
                "legend_items": [
                    {
                        "legend_label": "condition A",
                        "label_bbox_crop_px": [80, 10, 98, 18],
                        "swatch_color_rgb": [240, 40, 40],
                        "swatch_bbox_crop_px": [74, 11, 78, 15],
                        "marker_shape": None,
                        "confidence": 0.8,
                    }
                ],
                "plot_annotation_regions": [
                    {
                        "annotation_text": "note",
                        "bbox_crop_px": [35, 12, 50, 18],
                        "reason": "colored non-data text inside plot",
                        "confidence": 0.75,
                    }
                ],
                "curve_role_rules": [],
                "curve_shape_groups": [
                    {
                        "shape_class": "linear_like",
                        "curve_role_hint": "voltage_curve",
                        "axis_binding_hint": "left_y",
                        "visual_description": "monotonic decreasing voltage curves",
                        "confidence": 0.7,
                    },
                    {
                        "shape_class": "quadratic_peak_like",
                        "curve_role_hint": "power_density_curve",
                        "axis_binding_hint": "right_y",
                        "visual_description": "rise-peak-fall power density curves",
                        "confidence": 0.7,
                    },
                ],
                "issues": [],
            },
            "stdout_tail": "",
        }

    monkeypatch.setattr(data_extraction, "_run_codex_exec", fake_codex)

    result = data_extraction.run_data_extraction(
        data_extraction.DataExtractionRunRequest(
            panel_id=panel_id,
            max_colors=4,
            run_codex_vision=True,
            codex_mode="legend",
        )
    )

    assert result["codex"]["result"]["chart_family"] == "dual_y_power_density"
    assert (out_root / panel_id / "codex_legend_vision.json").exists()
    assert result["run"]["artifacts"]["codex_legend_vision.json"].endswith(
        "codex_legend_vision.json"
    )
    assert result["run"]["artifacts"]["role_recolored_crop.png"].endswith("role_recolored_crop.png")
    assert result["run"]["artifacts"]["exclusion_regions.json"].endswith("exclusion_regions.json")
    assert (out_root / panel_id / "legend_marker_table.png").exists()
    exclusions = json.loads((out_root / panel_id / "exclusion_regions.json").read_text(encoding="utf-8"))
    assert exclusions["bboxes"][0] == [70, 6, 100, 26]
    assert len(exclusions["bboxes"]) == 2
    assert result["run"]["validation"]["checks"]["exclusion_region_count"] == 2
    assert result["run"]["validation"]["checks"]["legend_item_count"] == 1
    assert result["run"]["validation"]["checks"]["legend_matched_series_count"] == 1
    assert result["run"]["validation"]["checks"]["role_recolored_trace_count"] >= 1
    candidates = json.loads((out_root / panel_id / "series_candidates.json").read_text(encoding="utf-8"))
    assert candidates["legend_matching"]["matched_series_count"] == 1
    assert any(item.get("legend_label") == "condition A" for item in candidates["items"])


def test_codex_legend_prompt_requests_full_legend_block() -> None:
    from labeling_ui.routes import data_extraction

    prompt = data_extraction._codex_prompt("legend", None)

    assert "legend_region_bbox_crop_px must enclose the entire visual legend block" in prompt
    assert "Prefer a slightly padded legend region bbox" in prompt
    assert "Return plot_annotation_regions for non-data text inside the plot crop" in prompt
    assert "not the full plot" in prompt


def test_codex_exec_uses_gpt55_and_medium_reasoning(tmp_path: Path, monkeypatch) -> None:
    from labeling_ui.routes import data_extraction

    image_path = tmp_path / "plot_crop.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    captured: dict[str, object] = {}

    data_extraction.configure(
        get_dataset_root=lambda: tmp_path,
        get_project_root=lambda: tmp_path,
        get_out_root=lambda: tmp_path / "out",
        get_manual_axis_pairs_path=lambda: tmp_path / "manual_axis_pairs.json",
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "chart_family": "unknown",
                    "axis_labels": {"title": "", "x": "", "left_y": "", "right_y": ""},
                    "legend_region_bbox_crop_px": [0, 0, 0, 0],
                    "legend_items": [],
                    "plot_annotation_regions": [],
                    "curve_role_rules": [],
                    "curve_shape_groups": [],
                    "issues": [],
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(data_extraction.subprocess, "run", fake_run)

    result = data_extraction._run_codex_exec(image_path, data_extraction.CodexVisionRequest())

    cmd = captured["cmd"]
    assert cmd[cmd.index("--model") + 1] == "gpt-5.5"
    assert "-c" in cmd
    assert 'model_reasoning_effort="medium"' in cmd
    assert result["model"] == "gpt-5.5"
    assert result["reasoning_effort"] == "medium"


def test_data_extraction_run_from_scratch_uses_panel_vision_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    from labeling_ui.routes import data_extraction

    dataset_root, panel_id = _setup_dataset(tmp_path)
    out_root = dataset_root / "extractions" / "data_series_runs" / "ui"
    data_extraction.configure(
        get_dataset_root=lambda: dataset_root,
        get_project_root=lambda: tmp_path,
        get_out_root=lambda: out_root,
        get_manual_axis_pairs_path=lambda: dataset_root / "extractions" / "manual_axis_pairs.json",
    )

    def fake_scratch_codex(image_path, request):
        assert request.mode == "scratch"
        return {
            "schema_version": 1,
            "generated_at_utc": "2026-05-18T00:00:00Z",
            "mode": "scratch",
            "model": request.model,
            "source_image_rel": "panels/p1/p1_F01-P01_chart.png",
            "result": {
                "chart_family": "dual_y_power_density",
                "is_dual_y": True,
                "plot_frame_bbox_panel_px": [20, 10, 120, 85],
                "origin_panel_px": [20, 85],
                "axis_labels": {
                    "title": "",
                    "x": "Current density",
                    "left_y": "Voltage",
                    "right_y": "Power density",
                },
                "axes": [
                    {
                        "axis_role": "bottom",
                        "axis_line_bbox_panel_px": [20, 84, 120, 86],
                        "axis_line_coordinate_px": 85,
                        "label_bbox_panel_px": [48, 88, 98, 98],
                        "ticks": [
                            {
                                "tick_position_px": 20,
                                "tick_bbox_panel_px": [20, 83, 22, 87],
                                "label_text": "0",
                                "label_numeric_value": 0,
                                "label_bbox_panel_px": [16, 88, 24, 98],
                                "confidence": 0.9,
                            },
                            {
                                "tick_position_px": 120,
                                "tick_bbox_panel_px": [118, 83, 120, 87],
                                "label_text": "1",
                                "label_numeric_value": 1,
                                "label_bbox_panel_px": [116, 88, 124, 98],
                                "confidence": 0.9,
                            },
                        ],
                        "confidence": 0.9,
                    },
                    {
                        "axis_role": "left",
                        "axis_line_bbox_panel_px": [19, 10, 21, 85],
                        "axis_line_coordinate_px": 20,
                        "label_bbox_panel_px": [0, 30, 16, 65],
                        "ticks": [
                            {
                                "tick_position_px": 10,
                                "tick_bbox_panel_px": [18, 10, 22, 12],
                                "label_text": "1",
                                "label_numeric_value": 1,
                                "label_bbox_panel_px": [4, 6, 16, 14],
                                "confidence": 0.9,
                            },
                            {
                                "tick_position_px": 85,
                                "tick_bbox_panel_px": [18, 83, 22, 87],
                                "label_text": "0",
                                "label_numeric_value": 0,
                                "label_bbox_panel_px": [4, 80, 16, 90],
                                "confidence": 0.9,
                            },
                        ],
                        "confidence": 0.9,
                    },
                    {
                        "axis_role": "right",
                        "axis_line_bbox_panel_px": [119, 10, 121, 85],
                        "axis_line_coordinate_px": 120,
                        "label_bbox_panel_px": [124, 30, 138, 65],
                        "ticks": [
                            {
                                "tick_position_px": 10,
                                "tick_bbox_panel_px": [118, 10, 122, 12],
                                "label_text": "2",
                                "label_numeric_value": 2,
                                "label_bbox_panel_px": [124, 6, 136, 14],
                                "confidence": 0.9,
                            },
                            {
                                "tick_position_px": 85,
                                "tick_bbox_panel_px": [118, 83, 122, 87],
                                "label_text": "0",
                                "label_numeric_value": 0,
                                "label_bbox_panel_px": [124, 80, 136, 90],
                                "confidence": 0.9,
                            },
                        ],
                        "confidence": 0.9,
                    },
                ],
                "legend_region_bbox_panel_px": [92, 18, 118, 36],
                "legend_items": [
                    {
                        "legend_label": "condition A",
                        "label_bbox_panel_px": [101, 20, 118, 30],
                        "swatch_color_rgb": [240, 40, 40],
                        "swatch_bbox_panel_px": [94, 22, 99, 27],
                        "marker_shape": "circle",
                        "confidence": 0.9,
                    }
                ],
                "plot_annotation_regions": [],
                "curve_role_rules": [],
                "issues": [],
            },
            "stdout_tail": "",
        }

    monkeypatch.setattr(data_extraction, "_run_codex_exec", fake_scratch_codex)

    result = data_extraction.run_data_extraction(
        data_extraction.DataExtractionRunRequest(
            panel_id=panel_id,
            max_colors=4,
            run_from_scratch=True,
        )
    )

    assert result["scratch"]["result"]["is_dual_y"] is True
    assert (out_root / panel_id / "codex_scratch_vision.json").exists()
    assert (out_root / panel_id / "scratch_axis_overlay.png").exists()
    assert result["run"]["artifacts"]["codex_scratch_vision.json"].endswith(
        "codex_scratch_vision.json"
    )
    mapping = json.loads((out_root / panel_id / "axis_mapping.json").read_text(encoding="utf-8"))
    assert mapping["axes"]["bottom"]["pair_source"] == "gt_axis_ticks"
    assert mapping["axes"]["bottom"]["axis_label"] == "Current density"
    assert mapping["axes"]["right"]["model"] is not None


def test_codex_legend_bbox_refinement_uses_image_foreground(tmp_path: Path) -> None:
    from labeling_ui.routes import data_extraction

    image_path = tmp_path / "plot_crop.png"
    image = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([72, 23, 80, 35], fill=(255, 0, 0))
    draw.line([60, 28, 92, 28], fill=(255, 0, 0), width=3)
    draw.rectangle([96, 24, 112, 36], fill=(30, 30, 30))
    image.save(image_path)

    codex = {
        "result": {
            "legend_region_bbox_crop_px": [58, 8, 112, 24],
            "legend_items": [
                {
                    "legend_label": "520",
                    "label_bbox_crop_px": [94, 10, 114, 24],
                    "swatch_color_rgb": [255, 0, 0],
                    "swatch_bbox_crop_px": [58, 12, 92, 22],
                    "marker_shape": "square",
                    "confidence": 0.9,
                }
            ],
            "issues": [],
        }
    }

    refined = data_extraction._refine_codex_legend_bboxes(image_path, codex)
    result = refined["result"]

    assert result["legend_items"][0]["swatch_bbox_crop_px"] == [60, 23, 93, 36]
    assert result["legend_items"][0]["label_bbox_crop_px"] == [96, 24, 113, 37]
    assert result["legend_region_bbox_crop_px"] == [57, 20, 116, 40]
    assert "LEGEND_BBOX_REFINED_FROM_IMAGE_FOREGROUND" in result["issues"]


def test_codex_annotation_bbox_refinement_uses_colored_text_foreground(tmp_path: Path) -> None:
    from labeling_ui.routes import data_extraction

    image_path = tmp_path / "plot_crop.png"
    image = Image.new("RGB", (140, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((48, 32), "457.9", fill=(230, 72, 86))
    image.save(image_path)

    codex = {
        "result": {
            "legend_region_bbox_crop_px": [5, 5, 50, 18],
            "legend_items": [
                {
                    "legend_label": "650C",
                    "label_bbox_crop_px": [24, 7, 48, 16],
                    "swatch_color_rgb": [230, 72, 86],
                    "swatch_bbox_crop_px": [8, 8, 18, 14],
                    "marker_shape": "star",
                    "confidence": 0.9,
                }
            ],
            "plot_annotation_regions": [
                {
                    "annotation_text": "457.9",
                    "bbox_crop_px": [35, 20, 105, 55],
                    "reason": "colored in-plot annotation",
                    "confidence": 0.8,
                }
            ],
            "issues": [],
        }
    }

    refined = data_extraction._refine_codex_legend_bboxes(image_path, codex)
    region = refined["result"]["plot_annotation_regions"][0]

    assert region["bbox_crop_px"][0] > 35
    assert region["bbox_crop_px"][1] > 20
    assert region["bbox_crop_px"][2] < 105
    assert region["bbox_crop_px"][3] < 55
    assert region["detected_text_color_rgb"] == [230, 72, 86]
    assert "PLOT_ANNOTATION_BBOX_REFINED_FROM_IMAGE_FOREGROUND" in refined["result"]["issues"]

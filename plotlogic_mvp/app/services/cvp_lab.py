from __future__ import annotations

import json
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .. import config
from .bbox_refinement import refine_chart_element_bboxes
from .mask_correction import run_llm_mask_correction
from .plotlogic_bridge import export_curve_series_csv, overrides_from_vision
from .visual_qa import ensure_visual_qa_artifacts


if str(config.PLOTLOGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PLOTLOGIC_ROOT))

from core.data_extraction_pipeline import run_data_extraction_for_panel  # noqa: E402


LAB_ROOT = config.DATA_ROOT / "cvp_rule_lab"
FIXTURES_DIR = LAB_ROOT / "fixtures"
RUNS_DIR = LAB_ROOT / "runs"
RULES_DIR = LAB_ROOT / "rules"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:120] or "fixture"


def _fixture_image_info(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
        gray = ImageOps.autocontrast(image.convert("L"))
        small = gray.resize((16, 16), Image.Resampling.LANCZOS)
        values = list(small.getdata())
    average = sum(values) / len(values)
    bits = "".join("1" if value >= average else "0" for value in values)
    return {
        "width": width,
        "height": height,
        "size_label": f"{width}x{height}",
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "perceptual_hash_bits": 256,
        "perceptual_hash": f"{int(bits, 2):064x}",
    }


def _hash_distance(left_hex: str | None, right_hex: str | None) -> int | None:
    if not left_hex or not right_hex:
        return None
    return (int(left_hex, 16) ^ int(right_hex, 16)).bit_count()


def ensure_lab() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    readme = LAB_ROOT / "README.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "# C-V-P Rule Lab",
                    "",
                    "This folder is separated from production MVP jobs.",
                    "Put current-voltage-power chart fixtures under `fixtures/` and run rules into `runs/`.",
                    "Prompt/rule snapshots are copied into `rules/` and each run directory for comparison.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    extraction_rules = RULES_DIR / "cvp_extraction_rules.md"
    if not extraction_rules.exists():
        extraction_rules.write_text(
            "\n".join(
                [
                    "# C-V-P Extraction Rule Notes",
                    "",
                    "- The production chart-family prompt lives in `prompts/chart_families.json`.",
                    "- Panel classification prompt lives in `prompts/figure_panel_classifier.md`.",
                    "- Pixel-level extraction heuristics currently live in `plotlogic/core/data_extraction_pipeline.py`.",
                    "- Use this lab to compare fixtures before changing the production MVP behavior.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def snapshot_rules(destination: Path) -> dict[str, Any]:
    ensure_lab()
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for source in [
        config.PROMPTS_ROOT / "chart_families.json",
        config.PROMPTS_ROOT / "figure_panel_classifier.md",
        config.PROMPTS_ROOT / "visual_qa" / "mask_quality.md",
        config.PROMPTS_ROOT / "visual_qa" / "trace_overlay.md",
        config.PROMPTS_ROOT / "visual_qa" / "mask_correction.md",
        RULES_DIR / "cvp_extraction_rules.md",
    ]:
        if not source.exists():
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append({"source": str(source), "snapshot": str(target)})
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": copied,
    }
    _write_json(destination / "rule_snapshot.json", payload)
    return payload


def _iter_existing_extraction_inputs() -> list[Path]:
    jobs_dir = config.DATA_ROOT / "jobs"
    if not jobs_dir.exists():
        return []
    return sorted(jobs_dir.glob("*/plotlogic_runs/*/mvp_extraction_inputs.json"))


def import_existing_targets() -> dict[str, Any]:
    ensure_lab()
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for input_path in _iter_existing_extraction_inputs():
        inputs = _read_json(input_path, {})
        if not isinstance(inputs, dict):
            continue
        panel_vision = inputs.get("panel_vision") if isinstance(inputs.get("panel_vision"), dict) else {}
        extraction_family = str(panel_vision.get("extraction_family") or "")
        is_target = bool(panel_vision.get("is_extraction_target") or panel_vision.get("is_target_dual_y"))
        if extraction_family != "current_voltage_power_dual_y" and not is_target:
            continue
        panel_path = Path(str(inputs.get("panel_path") or ""))
        if not panel_path.exists():
            skipped.append({"source": str(input_path), "reason": "panel_path_missing"})
            continue
        fixture_id = _safe_id(str(inputs.get("panel_id") or panel_path.stem))
        fixture_dir = FIXTURES_DIR / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture_image = fixture_dir / "panel.png"
        with Image.open(panel_path) as image:
            image.convert("RGB").save(fixture_image)
        metadata = {
            "fixture_id": fixture_id,
            "imported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_job_id": inputs.get("job_id"),
            "source_paper_id": inputs.get("paper_id"),
            "source_panel_id": inputs.get("panel_id"),
            "source_input_path": str(input_path),
            "source_panel_path": str(panel_path),
            "debug_inputs": inputs,
            "notes": [],
        }
        _write_json(fixture_dir / "meta.json", metadata)
        imported.append({"fixture_id": fixture_id, "image_path": str(fixture_image)})
    return {"imported_count": len(imported), "imported": imported, "skipped": skipped}


def _offset_crop_bbox(raw: Any, offset_xy: list[int]) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in raw):
        return None
    ox, oy = offset_xy
    x1, y1, x2, y2 = [int(round(value)) for value in raw]
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def _combined_plotlogic_vision(run_dir: Path) -> dict[str, Any] | None:
    crop_meta = _read_json(run_dir / "crop_meta.json", {})
    scratch = _read_json(run_dir / "codex_scratch_vision.json", {})
    legend = _read_json(run_dir / "codex_legend_vision.json", {})
    vision = scratch.get("result") if isinstance(scratch.get("result"), dict) else None
    if not isinstance(vision, dict):
        return None
    vision = json.loads(json.dumps(vision))
    vision["panel_type"] = "chart"
    vision["is_chart"] = True
    vision["is_extraction_target"] = True
    vision["is_target_dual_y"] = True
    vision["extraction_family"] = "current_voltage_power_dual_y"
    vision["chart_family"] = vision.get("chart_family") or "dual_y_current_voltage_power"
    offset = crop_meta.get("crop_offset_xy") if isinstance(crop_meta.get("crop_offset_xy"), list) else [0, 0]
    legend_result = legend.get("result") if isinstance(legend.get("result"), dict) else {}
    if isinstance(legend_result.get("legend_region_bbox_crop_px"), list):
        vision["legend_region_bbox_panel_px"] = _offset_crop_bbox(legend_result["legend_region_bbox_crop_px"], offset)
    legend_items: list[dict[str, Any]] = []
    for item in legend_result.get("legend_items", []) if isinstance(legend_result.get("legend_items"), list) else []:
        if not isinstance(item, dict):
            continue
        converted = dict(item)
        converted["label_bbox_panel_px"] = _offset_crop_bbox(item.get("label_bbox_crop_px"), offset)
        converted["swatch_bbox_panel_px"] = _offset_crop_bbox(item.get("swatch_bbox_crop_px"), offset)
        legend_items.append(converted)
    if legend_items:
        vision["legend_items"] = legend_items
    regions: list[dict[str, Any]] = []
    for region in legend_result.get("plot_annotation_regions", []) if isinstance(legend_result.get("plot_annotation_regions"), list) else []:
        if not isinstance(region, dict):
            continue
        converted = dict(region)
        converted["bbox_panel_px"] = _offset_crop_bbox(region.get("bbox_crop_px"), offset)
        regions.append(converted)
    if regions:
        vision["plot_annotation_regions"] = regions
    if isinstance(legend_result.get("curve_role_rules"), list):
        vision["curve_role_rules"] = legend_result["curve_role_rules"]
    return vision


def _load_plotlogic_link_maps(dataset_root: Path) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    links = dataset_root / "links"
    axes_panels = _read_json(links / "gt_axes.json", {}).get("panels") or []
    dual_y = {str(item.get("panel_id")) for item in axes_panels if isinstance(item, dict) and item.get("is_dual_y")}
    plots_by_id = {
        str(item.get("panel_id")): item
        for item in (_read_json(links / "gt_plots.json", {}).get("panels") or [])
        if isinstance(item, dict) and item.get("panel_id")
    }
    axes_by_id = {
        str(item.get("panel_id")): item
        for item in axes_panels
        if isinstance(item, dict) and item.get("panel_id")
    }
    ticks_by_id = {
        str(item.get("panel_id")): item
        for item in (_read_json(links / "gt_axis_ticks.json", {}).get("panels") or [])
        if isinstance(item, dict) and item.get("panel_id")
    }
    return dual_y, plots_by_id, axes_by_id, ticks_by_id


def import_plotlogic_phase1_dual_y_cvp() -> dict[str, Any]:
    ensure_lab()
    dataset_root = config.PLOTLOGIC_ROOT / "data" / "datasets" / "phase1_samples"
    runs_root = dataset_root / "extractions" / "data_series_runs" / "ui"
    dual_y_ids, plots_by_id, axes_by_id, ticks_by_id = _load_plotlogic_link_maps(dataset_root)
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    if not runs_root.exists():
        return {"imported_count": 0, "imported": [], "skipped": [{"source": str(runs_root), "reason": "missing"}]}

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        panel_id = run_dir.name
        if panel_id not in dual_y_ids:
            continue
        strategy = _read_json(run_dir / "chart_extraction_strategy.json", {})
        if strategy.get("chart_family") != "dual_y_power_density":
            continue
        curve_series = _read_json(run_dir / "curve_series_data.json", {})
        roles = {
            item.get("curve_role")
            for item in curve_series.get("series", [])
            if isinstance(item, dict)
        }
        if not {"voltage_curve", "power_density_curve"}.issubset(roles):
            skipped.append({"source": str(run_dir), "reason": "missing_voltage_or_power_series"})
            continue
        crop_meta = _read_json(run_dir / "crop_meta.json", {})
        panel_rel_path = crop_meta.get("source_panel_path")
        panel_path = dataset_root / str(panel_rel_path or "")
        if not panel_path.exists():
            skipped.append({"source": str(run_dir), "reason": "panel_image_missing"})
            continue
        vision = _combined_plotlogic_vision(run_dir)
        fixture_id = _safe_id(f"plotlogic_{panel_id}")
        fixture_dir = FIXTURES_DIR / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture_image = fixture_dir / "panel.png"
        with Image.open(panel_path) as image:
            image.convert("RGB").save(fixture_image)
        notes = [
            "Imported from plotlogic/data/datasets/phase1_samples dual-y tags and existing dual_y_power_density extraction runs."
        ]
        if vision is not None:
            plot_panel, axes_panel, ticks_panel, legend_items, exclusions = overrides_from_vision(
                panel_id,
                "panel.png",
                vision,
            )
        else:
            plot_panel = _override_panel_paths(plots_by_id.get(panel_id) or {}, panel_id, "panel.png")
            axes_panel = _override_panel_paths(axes_by_id.get(panel_id) or {}, panel_id, "panel.png")
            ticks_panel = _override_panel_paths(
                ticks_by_id.get(panel_id) or {"panel_id": panel_id, "panel_path": "panel.png", "panel_kind": "chart", "axes": []},
                panel_id,
                "panel.png",
            )
            legend_items = []
            exclusions = []
            notes.append("No codex_scratch_vision artifact was found; fixture uses GT plot/axis metadata only.")
        debug_inputs = {
            "schema_version": 1,
            "job_id": "plotlogic_phase1_samples",
            "paper_id": str(panel_id).split("_", 1)[0],
            "panel_id": panel_id,
            "source_figure_image": str(panel_path),
            "panel_path": str(fixture_image),
            "panel_rel_path": "panel.png",
            "panel_crop_bbox_source_image_px": None,
            "panel_vision": vision or {
                "panel_type": "chart",
                "is_chart": True,
                "is_extraction_target": True,
                "is_target_dual_y": True,
                "extraction_family": "current_voltage_power_dual_y",
                "chart_family": "dual_y_current_voltage_power",
                "issues": ["NO_CODEX_SCRATCH_VISION_AVAILABLE"],
            },
            "plot_panel_override": plot_panel,
            "axes_panel_override": axes_panel,
            "ticks_panel_override": ticks_panel,
            "legend_items_crop_px": legend_items,
            "exclusion_bboxes_crop_px": exclusions,
        }
        metadata = {
            "fixture_id": fixture_id,
            "imported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "plotlogic_phase1_dual_y_cvp",
            "source_panel_id": panel_id,
            "source_panel_path": str(panel_path),
            "source_run_dir": str(run_dir),
            "debug_inputs": debug_inputs,
            "notes": notes,
        }
        _write_json(fixture_dir / "meta.json", metadata)
        imported.append({"fixture_id": fixture_id, "image_path": str(fixture_image), "source_panel_id": panel_id})
    return {"imported_count": len(imported), "imported": imported, "skipped": skipped}


def add_fixture_from_image(image_path: Path, *, original_filename: str, source: str = "manual_upload") -> dict[str, Any]:
    ensure_lab()
    stem = _safe_id(Path(original_filename).stem)
    fixture_id = stem
    counter = 2
    while (FIXTURES_DIR / fixture_id).exists():
        fixture_id = f"{stem}_{counter}"
        counter += 1
    fixture_dir = FIXTURES_DIR / fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_image = fixture_dir / "panel.png"
    with Image.open(image_path) as image:
        image.convert("RGB").save(fixture_image)
    metadata = {
        "fixture_id": fixture_id,
        "imported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source,
        "original_filename": original_filename,
        "debug_inputs": None,
        "notes": [
            "Manual image fixture. Axis/legend geometry is not available unless added later or imported from an MVP extraction target."
        ],
    }
    _write_json(fixture_dir / "meta.json", metadata)
    return {"fixture_id": fixture_id, "image_path": str(fixture_image)}


def list_fixtures() -> list[dict[str, Any]]:
    ensure_lab()
    fixtures: list[dict[str, Any]] = []
    for fixture_dir in sorted(FIXTURES_DIR.iterdir()):
        if not fixture_dir.is_dir():
            continue
        image_path = fixture_dir / "panel.png"
        meta = _read_json(fixture_dir / "meta.json", {})
        if not image_path.exists():
            continue
        fixtures.append(
            {
                "fixture_id": fixture_dir.name,
                "image_path": str(image_path),
                "image_info": _fixture_image_info(image_path),
                "meta": meta,
            }
        )
    return fixtures


def find_near_duplicate_fixture_groups(fixtures: list[dict[str, Any]], *, threshold: int = 48) -> list[dict[str, Any]]:
    items: list[tuple[str, str, dict[str, Any]]] = []
    for fixture in fixtures:
        info = fixture.get("image_info") if isinstance(fixture.get("image_info"), dict) else {}
        image_hash = info.get("perceptual_hash")
        fixture_id = fixture.get("fixture_id")
        if isinstance(fixture_id, str) and isinstance(image_hash, str):
            items.append((fixture_id, image_hash, info))

    parent = {fixture_id: fixture_id for fixture_id, _, _ in items}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pairs: list[dict[str, Any]] = []
    for index, (left_id, left_hash, left_info) in enumerate(items):
        for right_id, right_hash, right_info in items[index + 1 :]:
            distance = _hash_distance(left_hash, right_hash)
            if distance is None or distance > threshold:
                continue
            union(left_id, right_id)
            pairs.append(
                {
                    "left_fixture_id": left_id,
                    "right_fixture_id": right_id,
                    "hash_distance": distance,
                    "left_size": left_info.get("size_label"),
                    "right_size": right_info.get("size_label"),
                }
            )

    grouped: dict[str, list[str]] = {}
    for fixture_id, _, _ in items:
        grouped.setdefault(find(fixture_id), []).append(fixture_id)

    groups: list[dict[str, Any]] = []
    for group_index, fixture_ids in enumerate(
        sorted((sorted(ids) for ids in grouped.values() if len(ids) > 1), key=lambda ids: ids[0]),
        start=1,
    ):
        fixture_set = set(fixture_ids)
        groups.append(
            {
                "group_id": f"near_duplicate_{group_index}",
                "threshold": threshold,
                "fixture_ids": fixture_ids,
                "pairs": [
                    pair
                    for pair in pairs
                    if pair["left_fixture_id"] in fixture_set and pair["right_fixture_id"] in fixture_set
                ],
            }
        )
    return groups


def list_runs() -> list[dict[str, Any]]:
    ensure_lab()
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary = _read_json(run_dir / "run_summary.json", {})
        if summary:
            runs.append(summary)
    return runs


def _override_panel_paths(payload: dict[str, Any], panel_id: str, panel_rel_path: str) -> dict[str, Any]:
    copy = json.loads(json.dumps(payload or {}))
    copy["panel_id"] = panel_id
    copy["panel_path"] = panel_rel_path
    return copy


def _minimal_overrides(panel_id: str, panel_rel_path: str, image_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    with Image.open(image_path) as image:
        width, height = image.size
    plot_panel = {
        "panel_id": panel_id,
        "panel_path": panel_rel_path,
        "panel_kind": "chart",
        "review_status": "scratch",
        "gt_bbox_xyxy": [0, 0, width, height],
        "annotator": "cvp_rule_lab",
        "note": "minimal fixture fallback",
    }
    axes_panel = {
        "panel_id": panel_id,
        "panel_path": panel_rel_path,
        "panel_kind": "chart",
        "is_dual_y": True,
        "review_status": "scratch",
        "axis_left_bbox_xyxy": None,
        "axis_bottom_bbox_xyxy": None,
        "axis_right_bbox_xyxy": None,
        "axis_bottom_label": "",
        "axis_left_label": "",
        "axis_right_label": "",
    }
    ticks_panel = {"panel_id": panel_id, "panel_path": panel_rel_path, "panel_kind": "chart", "axes": []}
    return plot_panel, axes_panel, ticks_panel, [], []


def _fixture_overrides(fixture: dict[str, Any], panel_id: str, panel_rel_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    meta = fixture.get("meta") if isinstance(fixture.get("meta"), dict) else {}
    inputs = meta.get("debug_inputs") if isinstance(meta.get("debug_inputs"), dict) else {}
    image_path = Path(str(fixture["image_path"]))
    if not inputs:
        return _minimal_overrides(panel_id, panel_rel_path, image_path)

    panel_vision = inputs.get("panel_vision") if isinstance(inputs.get("panel_vision"), dict) else None
    if isinstance(panel_vision, dict) and isinstance(panel_vision.get("plot_frame_bbox_panel_px"), list):
        refined_vision = json.loads(json.dumps(panel_vision))
        with Image.open(image_path) as image:
            refine_chart_element_bboxes(image.convert("RGB"), refined_vision)
        plot_panel, axes_panel, ticks_panel, legend_items, exclusions = overrides_from_vision(
            panel_id,
            panel_rel_path,
            refined_vision,
        )
        plot_panel["note"] = "cvp_rule_lab regenerated from refined panel_vision"
        axes_panel["note"] = "cvp_rule_lab regenerated from refined panel_vision"
        ticks_panel["note"] = "cvp_rule_lab regenerated from refined panel_vision"
        return plot_panel, axes_panel, ticks_panel, legend_items, exclusions

    plot_panel = _override_panel_paths(inputs.get("plot_panel_override") or {}, panel_id, panel_rel_path)
    axes_panel = _override_panel_paths(inputs.get("axes_panel_override") or {}, panel_id, panel_rel_path)
    ticks_panel = _override_panel_paths(inputs.get("ticks_panel_override") or {}, panel_id, panel_rel_path)
    legend_items = inputs.get("legend_items_crop_px") if isinstance(inputs.get("legend_items_crop_px"), list) else []
    exclusions = inputs.get("exclusion_bboxes_crop_px") if isinstance(inputs.get("exclusion_bboxes_crop_px"), list) else []
    return plot_panel, axes_panel, ticks_panel, legend_items, exclusions


def _result_summary(run_id: str, fixture_id: str, result_dir: Path, status: str, error: str | None = None) -> dict[str, Any]:
    validation = _read_json(result_dir / "validation_report.json", {}) if result_dir.exists() else {}
    checks = validation.get("checks") if isinstance(validation.get("checks"), dict) else {}
    quality = _read_json(result_dir / "image_quality_report.json", {}) if result_dir.exists() else {}
    visual_qa = _read_json(result_dir / "visual_qa_summary.json", {}) if result_dir.exists() else {}
    mask_correction = _read_json(result_dir / "llm_mask_correction_summary.json", {}) if result_dir.exists() else {}
    return {
        "run_id": run_id,
        "fixture_id": fixture_id,
        "status": status,
        "error": error,
        "result_dir": str(result_dir),
        "overlay_path": str(result_dir / "overlay.png") if (result_dir / "overlay.png").exists() else None,
        "plot_crop_path": str(result_dir / "plot_crop.png") if (result_dir / "plot_crop.png").exists() else None,
        "color_filter_source_path": str(result_dir / "color_filter_source.png") if (result_dir / "color_filter_source.png").exists() else None,
        "final_trace_overlay_path": str(result_dir / "final_trace_overlay.png") if (result_dir / "final_trace_overlay.png").exists() else None,
        "raw_vs_final_trace_overlay_path": str(result_dir / "raw_vs_final_trace_overlay.png") if (result_dir / "raw_vs_final_trace_overlay.png").exists() else None,
        "role_recolored_crop_path": str(result_dir / "role_recolored_crop.png") if (result_dir / "role_recolored_crop.png").exists() else None,
        "llm_mask_correction_contact_sheet_path": str(result_dir / "llm_mask_correction_contact_sheet.png") if (result_dir / "llm_mask_correction_contact_sheet.png").exists() else None,
        "llm_mask_correction_comparison_path": str(result_dir / "llm_mask_correction_comparison.png") if (result_dir / "llm_mask_correction_comparison.png").exists() else None,
        "mask_reconstruction_comparison_path": str(result_dir / "mask_reconstruction_comparison.png") if (result_dir / "mask_reconstruction_comparison.png").exists() else None,
        "csv_path": str(result_dir / "curve_series_data.csv") if (result_dir / "curve_series_data.csv").exists() else None,
        "series_count": int(checks.get("final_series_count") or 0),
        "validation_issues": validation.get("issues") or [],
        "quality_issues": quality.get("issues") or [],
        "visual_qa_status": visual_qa.get("status"),
        "visual_qa_issue_count": int(visual_qa.get("issue_count") or 0),
        "mask_correction_status": mask_correction.get("status"),
        "mask_correction_corrected_count": int(mask_correction.get("corrected_count") or 0),
        "mask_correction_rejected_count": int(mask_correction.get("rejected_count") or 0),
        "mask_reconstruction_bridge_count": int(mask_correction.get("bridge_count") or 0),
        "mask_reconstruction_removed_component_count": int(mask_correction.get("removed_component_count") or 0),
        "checks": checks,
    }


def run_lab() -> dict[str, Any]:
    ensure_lab()
    fixtures = list_fixtures()
    if not fixtures:
        raise ValueError("No C-V-P lab fixtures found. Import existing targets first.")
    run_id = _utc_stamp()
    run_dir = RUNS_DIR / run_id
    results_root = run_dir / "results"
    dataset_root = run_dir / "dataset"
    snapshot_rules(run_dir / "rules")
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_id = str(fixture["fixture_id"])
        panel_id = fixture_id
        image_path = Path(str(fixture["image_path"]))
        panel_rel_path = f"panels/{fixture_id}.png"
        copied_panel = dataset_root / panel_rel_path
        copied_panel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, copied_panel)
        result_dir = results_root / panel_id
        try:
            plot_panel, axes_panel, ticks_panel, legend_items, exclusions = _fixture_overrides(
                fixture,
                panel_id,
                panel_rel_path,
            )
            manifest = run_data_extraction_for_panel(
                dataset_root=dataset_root,
                panel_id=panel_id,
                out_root=results_root,
                max_colors=8,
                exclusion_bboxes_crop_px=exclusions,
                exclusion_source="cvp_rule_lab_fixture",
                legend_items=legend_items,
                plot_panel_override=plot_panel,
                axes_panel_override=axes_panel,
                ticks_panel_override=ticks_panel,
            )
            lab_inputs = {
                "schema_version": 1,
                "run_id": run_id,
                "fixture_id": fixture_id,
                "source_fixture_image": str(image_path),
                "panel_rel_path": panel_rel_path,
                "plot_panel_override": plot_panel,
                "axes_panel_override": axes_panel,
                "ticks_panel_override": ticks_panel,
                "legend_items_crop_px": legend_items,
                "exclusion_bboxes_crop_px": exclusions,
                "source_fixture_meta": fixture.get("meta"),
                "manifest": manifest,
            }
            _write_json(result_dir / "cvp_lab_inputs.json", lab_inputs)
            curve_series = result_dir / "curve_series_data.json"
            if curve_series.exists():
                export_curve_series_csv(curve_series, result_dir / "curve_series_data.csv")
            ensure_visual_qa_artifacts(result_dir)
            results.append(_result_summary(run_id, fixture_id, result_dir, "ok"))
        except Exception as exc:  # pragma: no cover - surfaced in lab UI.
            result_dir.mkdir(parents=True, exist_ok=True)
            _write_json(result_dir / "error.json", {"error": str(exc)})
            results.append(_result_summary(run_id, fixture_id, result_dir, "failed", str(exc)))
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fixture_count": len(fixtures),
        "ok_count": sum(1 for item in results if item["status"] == "ok"),
        "failed_count": sum(1 for item in results if item["status"] != "ok"),
        "run_dir": str(run_dir),
        "results": results,
    }
    _write_json(run_dir / "run_summary.json", summary)
    return summary


def get_run(run_id: str) -> dict[str, Any] | None:
    ensure_lab()
    return _read_json(RUNS_DIR / _safe_id(run_id) / "run_summary.json", None)


def run_result_mask_correction(run_id: str, fixture_id: str) -> dict[str, Any]:
    ensure_lab()
    safe_run_id = _safe_id(run_id)
    safe_fixture_id = _safe_id(fixture_id)
    run_dir = RUNS_DIR / safe_run_id
    result_dir = run_dir / "results" / safe_fixture_id
    if not result_dir.exists():
        raise ValueError("Lab result not found.")
    summary = run_llm_mask_correction(result_dir)
    run_summary_path = run_dir / "run_summary.json"
    run_summary = _read_json(run_summary_path, {})
    if isinstance(run_summary, dict):
        results = run_summary.get("results")
        if isinstance(results, list):
            for index, item in enumerate(results):
                if isinstance(item, dict) and item.get("fixture_id") == safe_fixture_id:
                    results[index] = _result_summary(safe_run_id, safe_fixture_id, result_dir, str(item.get("status") or "ok"), item.get("error"))
                    break
            _write_json(run_summary_path, run_summary)
    return summary

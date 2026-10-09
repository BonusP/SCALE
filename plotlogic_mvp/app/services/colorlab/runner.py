"""Orchestration: read an existing run's artifacts, extract, tabulate, compare.

Reads only. Everything this package produces lands in ``<run_dir>/colorlab/``, so the
legacy pipeline's own files are never touched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import report as report_module
from .assemble import Series, assemble_series, trace_lines
from .axes import build_axis_frame
from .canvas import build_canvas, write_debug_images
from .config import ColorLabParams
from .glyphs import detect_glyphs
from .palette import build_palette
from .qc import build_qc_report, draw_colorlab_overlay, draw_layer_map, draw_legacy_overlay
from .tabulate import build_point_rows, write_points_csv, write_series_json, write_wide_csv

OUTPUT_DIRNAME = "colorlab"

REQUIRED_INPUTS = ("plot_crop.png", "crop_meta.json")
OPTIONAL_INPUTS = (
    "exclusion_regions.json",
    "axis_mapping.json",
    "chart_extraction_strategy.json",
    "mvp_extraction_inputs.json",
    "series_candidates.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default


def _legend_items(run_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Legend swatches seed the palette. Prefer the vision pass's raw swatch colours;
    fall back to the legacy candidate colours, which are already legend-matched."""
    inputs = _read_json(run_dir / "mvp_extraction_inputs.json", {}) or {}
    vision = inputs.get("panel_vision") or inputs.get("panel_vision_raw_codex") or {}
    items = vision.get("legend_items") if isinstance(vision, dict) else None
    if isinstance(items, list) and items:
        return [item for item in items if isinstance(item, dict)], "panel_vision.legend_items"

    candidates = _read_json(run_dir / "series_candidates.json", {}) or {}
    rows = candidates.get("items") if isinstance(candidates, dict) else candidates
    if isinstance(rows, list) and rows:
        return (
            [
                {"legend_label": row.get("legend_label"), "swatch_color_rgb": row.get("target_color_rgb")}
                for row in rows
                if isinstance(row, dict)
            ],
            "series_candidates.json",
        )
    return [], "none"


def _legacy_traces(run_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    for name in ("final_curve_traces_crop_px.json", "curve_traces_crop_px.json"):
        payload = _read_json(run_dir / name)
        if isinstance(payload, dict) and isinstance(payload.get("traces"), list):
            return payload["traces"], name
    return [], None


def run_colorlab(
    run_dir: Path,
    *,
    params: ColorLabParams | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    for name in REQUIRED_INPUTS:
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{run_dir} is missing {name}")

    params = params or ColorLabParams()
    out_dir = run_dir / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_meta = _read_json(run_dir / "crop_meta.json", {}) or {}
    panel_id = str(crop_meta.get("panel_id") or run_dir.name)
    exclusions = (_read_json(run_dir / "exclusion_regions.json", {}) or {}).get("bboxes") or []
    axis_mapping = _read_json(run_dir / "axis_mapping.json", {}) or {}
    strategy = _read_json(run_dir / "chart_extraction_strategy.json", {}) or {}
    legend_items, legend_source = _legend_items(run_dir)

    # S0 / S1
    canvas = build_canvas(run_dir / "plot_crop.png", exclusions, params)
    debug_images = write_debug_images(canvas, out_dir)

    # S2
    palette = build_palette(canvas, legend_items, params)
    draw_layer_map(canvas, palette, out_dir / "layer_map.png")

    # S3 / S4
    detection = detect_glyphs(canvas, palette, params)

    # S6 (axis frame is needed before assembly reports quantities)
    frame = build_axis_frame(axis_mapping, crop_meta, strategy)

    # S5 - glyph mode, with the line tracker as fallback for marker-less charts
    series_list: list[Series] = assemble_series(
        detection,
        palette,
        params,
        is_dual_y=frame.is_dual_y,
        left_role=frame.left_role,
        right_role=frame.right_role,
    )
    xs = [x for series in series_list for x, _ in series.points_px]
    glyph_coverage = (max(xs) - min(xs)) / max(1.0, canvas.width) if xs else 0.0
    mode = "glyph"
    if glyph_coverage < params.line_mode_max_x_coverage:
        mode = "line"
        series_list = trace_lines(
            canvas,
            palette,
            params,
            is_dual_y=frame.is_dual_y,
            left_role=frame.left_role,
            right_role=frame.right_role,
        )

    # S7
    rows = build_point_rows(panel_id, series_list, frame)
    write_points_csv(out_dir / "points_long.csv", rows)
    x_label = frame.models["bottom"].label or frame.x_role
    wide_summary = write_wide_csv(out_dir / "series_wide.csv", rows, series_list, x_label=x_label)

    # S8
    qc = build_qc_report(canvas, palette, detection, series_list, params)
    draw_colorlab_overlay(canvas, series_list, out_dir / "colorlab_overlay.png")

    legacy_traces, legacy_source = _legacy_traces(run_dir)
    if legacy_traces:
        draw_legacy_overlay(canvas, legacy_traces, out_dir / "legacy_overlay.png")

    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "method": "colorlab_glyph_first",
        "panel_id": panel_id,
        "run_dir": str(run_dir),
        "mode": mode,
        "glyph_x_coverage": round(glyph_coverage, 4),
        "params": params.to_dict(),
        "inputs": {
            "crop": "plot_crop.png",
            "exclusion_bbox_count": len(canvas.exclusion_bboxes),
            "legend_source": legend_source,
            "legend_item_count": len(legend_items),
            "axis_mapping": "axis_mapping.json",
            "legacy_traces": legacy_source,
        },
        "canvas": canvas.stats(),
        "palette": palette.to_dict(),
        "detection": {
            key: value for key, value in detection.to_dict().items() if key != "glyphs"
        },
        "axis_frame": frame.to_dict(),
        "series": [series.to_dict() for series in series_list],
        "wide_table": wide_summary,
        "qc": qc,
        "artifacts": {
            **debug_images,
            "layer_map": "layer_map.png",
            "colorlab_overlay": "colorlab_overlay.png",
            "legacy_overlay": "legacy_overlay.png" if legacy_traces else None,
            "points_csv": "points_long.csv",
            "wide_csv": "series_wide.csv",
            "glyphs": "glyphs.json",
            "summary": "colorlab_summary.json",
            "report": "comparison_report.html" if write_report else None,
        },
    }
    write_series_json(out_dir / "colorlab_summary.json", summary)
    write_series_json(out_dir / "glyphs.json", detection.to_dict())

    if write_report:
        report_module.write_comparison_report(
            run_dir=run_dir,
            out_path=out_dir / "comparison_report.html",
            summary=summary,
            rows=rows,
            series_list=series_list,
            frame=frame,
            legacy_traces=legacy_traces,
        )
    return summary

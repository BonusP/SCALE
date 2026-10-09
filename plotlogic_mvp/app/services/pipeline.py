from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .. import config
from ..db import add_event, list_jobs_for_paper, update_job, update_paper_extraction, upsert_chart
from .bbox_validation import validate_figure_analysis
from .chart_geometry import repair_chart_panel_geometry
from .paperrag_bridge import collect_images, paper_snapshot, run_opendataloader_parse, run_paperrag_postprocess
from .plotlogic_bridge import (
    run_codex_direct_categorical_extraction,
    run_categorical_extraction,  # categorical 처리를 위한 수정
    run_codex_direct_extraction,
    run_extraction,
)

# categorical 처리를 위한 수정: CMSL traces categorical pixels before table conversion;
# other methods retain direct reading. A categorical x axis uses label positions, not a numeric fit.
CATEGORICAL_CHART_FAMILIES = {"categorical_line_chart"}

# The families Vision is supposed to use only when the x axis is genuinely numeric. Vision's own
# family choice is not perfectly reliable, so _bottom_axis_is_categorical() below is a second,
# deterministic check on the ticks Vision already returned: if none of them parsed as a number, the
# axis cannot be pixel-calibrated no matter which of these families the panel was filed under, so it
# is rerouted to the categorical path instead of running the pixel pipeline against an axis with no
# usable calibration pairs (which silently produces empty series).
LINE_CHART_FAMILIES = {"current_voltage_power_dual_y", "single_axis_line_chart", "dual_y_generic_line_chart", "cmsl_line_chart"}


def _bottom_axis_is_categorical(panel: dict[str, Any]) -> bool:
    for axis in panel.get("axes", []) or []:
        if not isinstance(axis, dict):
            continue
        role = str(axis.get("axis_role") or "").strip().lower()
        if role not in {"bottom", "x", "x_bottom", "bottom_x"}:
            continue
        ticks = [tick for tick in axis.get("ticks", []) if isinstance(tick, dict)]
        labelled = [tick for tick in ticks if str(tick.get("label_text") or "").strip()]
        if len(labelled) < 2:
            return False
        numeric = [tick for tick in labelled if isinstance(tick.get("label_numeric_value"), (int, float))]
        return len(numeric) == 0
    return False
from .run_methods import apply_chart_id_suffix, resolve_method
from .vision import analyze_figure_panels, current_prompt_metadata, fallback_panel


def _safe_id(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return safe[:80] or "panel"


def _panel_id(paper_id: int, image_index: int, panel_index: int, image_path: Path, panel: dict[str, Any]) -> str:
    label = _safe_id(str(panel.get("panel_label") or f"P{panel_index:02d}"))
    panel_type = _safe_id(str(panel.get("panel_type") or "panel"))
    digest = hashlib.sha1(f"{image_path}:{panel_index}:{label}".encode("utf-8")).hexdigest()[:8]
    return f"{paper_id:04d}_img{image_index:03d}_P{panel_index:02d}_{label}_{panel_type}_{digest}"


def _candidate_filename(paper_id: int, image_index: int, image_path: Path) -> str:
    return f"{paper_id:04d}_img{image_index:03d}_{_safe_id(image_path.stem)}.json"


def _analysis_prompt_sha(analysis: dict[str, Any]) -> str:
    metadata = analysis.get("prompt_metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("prompt_sha256") or "")


def _read_candidate_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _find_cached_figure_analysis(
    *,
    paper_id: int,
    current_job_id: str,
    candidate_name: str,
    current_prompt_sha: str,
) -> tuple[dict[str, Any], Path] | None:
    if not current_prompt_sha:
        return None
    for job in list_jobs_for_paper(paper_id, limit=100):
        previous_job_id = str(job.get("job_id") or "")
        if not previous_job_id or previous_job_id == current_job_id:
            continue
        path = config.DATA_ROOT / "jobs" / previous_job_id / "candidates" / candidate_name
        if not path.exists():
            continue
        analysis = _read_candidate_json(path)
        if not analysis:
            continue
        if _analysis_prompt_sha(analysis) != current_prompt_sha:
            continue
        return analysis, path
    return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _bbox(raw: Any) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in raw):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in raw]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _panel_bbox(panel: dict[str, Any], image: Image.Image) -> list[int]:
    width, height = image.size
    explicit_panel_bbox = _bbox(panel.get("panel_bbox_figure_px"))
    if explicit_panel_bbox:
        initial = _clamp_bbox(explicit_panel_bbox, width, height) or [0, 0, width, height]
        return list(repair_chart_panel_geometry(image, panel, initial)["panel_bbox"])

    bbox = _bbox(panel.get("plot_frame_bbox_panel_px"))
    if not bbox:
        return [0, 0, width, height]
    pad_x = max(6, int(round((bbox[2] - bbox[0]) * 0.015)))
    pad_y = max(6, int(round((bbox[3] - bbox[1]) * 0.015)))
    padded = [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y]
    return _clamp_bbox(padded, width, height) or [0, 0, width, height]


def _save_panel_crop(
    *,
    job_id: str,
    paper_id: int,
    chart_id: str,
    image_path: Path,
    panel: dict[str, Any],
) -> tuple[Path, list[int]]:
    crop_dir = config.DATA_ROOT / "jobs" / job_id / "panel_crops" / f"{paper_id:04d}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".png"
    crop_path = crop_dir / f"{chart_id}{suffix}"
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        bbox = _panel_bbox(panel, rgb)
        rgb.crop(tuple(bbox)).save(crop_path)
    _write_json(
        crop_path.with_suffix(crop_path.suffix + ".json"),
        {
            "source_image": str(image_path),
            "panel_crop_path": str(crop_path),
            "panel_bbox_figure_px": bbox,
            "panel_index": panel.get("panel_index"),
            "panel_label": panel.get("panel_label"),
            "panel_type": panel.get("panel_type"),
            "chart_family": panel.get("chart_family"),
            "is_target_dual_y": bool(panel.get("is_target_dual_y")),
        },
    )
    return crop_path, bbox


def _image_rel_path(paper_id: int, image_path: Path) -> str:
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    try:
        return image_path.resolve().relative_to(paper_dir.resolve()).as_posix()
    except ValueError:
        return image_path.name


def _panels_from_analysis(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    panels = [panel for panel in analysis.get("panels", []) if isinstance(panel, dict)]
    if panels:
        return panels
    return [fallback_panel("NO_PANEL_DETECTED")]


def _panel_name(image_path: Path, panel: dict[str, Any], panel_index: int) -> str:
    label = str(panel.get("panel_label") or f"P{panel_index:02d}").strip()
    panel_type = str(panel.get("panel_type") or "panel").strip()
    return f"{image_path.name} · {label} · {panel_type}"


def _skip_status(panel_type: str, is_chart: bool, is_target: bool) -> str:
    if not is_chart or panel_type != "chart":
        return "skipped_non_chart"
    if not is_target:
        return "skipped_non_extractable_chart"
    return "target_pending_extraction"


def _skip_reason(panel: dict[str, Any], panel_type: str, is_chart: bool, is_target: bool) -> str:
    explicit = str(panel.get("skip_reason") or "").strip()
    if explicit:
        return explicit
    if not is_chart or panel_type != "chart":
        return f"Panel classified as {panel_type or 'non-chart'}, not a chart."
    if not is_target:
        return "Chart is not currently supported by an enabled extraction family."
    return ""


def _upsert_skipped_panel(
    *,
    chart_id: str,
    job_id: str,
    paper_id: int,
    image_path: Path,
    panel_crop_path: Path,
    panel_crop_bbox: list[int],
    panel: dict[str, Any],
    panel_index: int,
    status: str,
    skip_reason: str,
    method: str,
) -> None:
    upsert_chart(
        chart_id=chart_id,
        job_id=job_id,
        method=method,
        paper_id=paper_id,
        image_name=_panel_name(image_path, panel, panel_index),
        image_rel_path=_image_rel_path(paper_id, image_path),
        image_abs_path=str(panel_crop_path),
        status=status,
        panel_type=str(panel.get("panel_type") or "other"),
        extraction_family=str(panel.get("extraction_family") or ""),
        chart_family=panel.get("chart_family") or "unknown",
        is_chart=1 if bool(panel.get("is_chart")) else 0,
        is_target_dual_y=0,
        confidence=float(panel.get("confidence") or 0),
        skip_reason=skip_reason,
        result_dir=None,
        overlay_path=None,
        csv_path=None,
        series_count=0,
        issues=list(panel.get("issues") or []) + [f"PANEL_CROP_BBOX_SOURCE_IMAGE:{panel_crop_bbox}"],
    )


def process_job_sequence(paper_id: int, jobs: list[tuple[str, str]]) -> None:
    """Run several methods over one paper, one after another.

    Sequential on purpose: the methods share the paper's parse output and the
    per-paper extraction status row, so running them at once would have them fight
    over both. A method that blows up must not stop the ones queued behind it.
    """
    for job_id, method_key in jobs:
        try:
            process_job(job_id, paper_id, method_key)
        except Exception as exc:
            update_job(job_id, status="failed", phase="failed", progress=100, error=str(exc))
            add_event(job_id, f"Method {method_key} failed unexpectedly.", level="error", payload={"error": str(exc)})


def process_job(job_id: str, paper_id: int, method_key: str | None = None) -> None:
    try:
        method = resolve_method(method_key)
        prompt_variant = method.prompt_variant
        add_event(
            job_id,
            f"Running method {method.key}: prompt {prompt_variant}, "
            f"OCR text boxes {'on' if method.use_ocr_text else 'off'}.",
        )
        update_job(job_id, status="running", phase="parsing_pdf", progress=5)
        update_paper_extraction(
            paper_id,
            latest_job_id=job_id,
            status="running",
            phase="parsing_pdf",
            error=None,
        )
        snapshot = paper_snapshot(paper_id)
        parse_status = (snapshot.get("parse") or {}).get("status")
        images = collect_images(paper_id)
        if parse_status == "success" and images:
            add_event(job_id, "Using existing PaperRAG parser output and extracted images.", payload=snapshot)
        else:
            add_event(job_id, "Starting PaperRAG opendataloader parse.")
            parse_result = run_opendataloader_parse(paper_id)
            if not parse_result.get("ok"):
                error = str(parse_result.get("error"))
                update_job(job_id, status="failed", phase="failed", progress=100, error=error)
                update_paper_extraction(
                    paper_id,
                    latest_job_id=job_id,
                    status="failed",
                    phase="failed",
                    error=error,
                )
                add_event(job_id, "PaperRAG parse failed.", level="error", payload=parse_result)
                return

        if config.PAPERRAG_POSTPROCESS_ENABLED:
            update_job(job_id, phase="paperrag_postprocess", progress=20)
            update_paper_extraction(
                paper_id,
                latest_job_id=job_id,
                status="running",
                phase="paperrag_postprocess",
                error=None,
            )
            add_event(job_id, "Starting PaperRAG semantic/RAG/index postprocess before PlotLogic extraction.")
            try:
                postprocess = run_paperrag_postprocess(paper_id)
            except Exception as exc:
                add_event(
                    job_id,
                    "PaperRAG postprocess failed unexpectedly.",
                    level="warning",
                    payload={"error": str(exc)},
                )
            else:
                if postprocess.get("status") == "completed":
                    add_event(job_id, "PaperRAG postprocess completed.", payload=postprocess)
                else:
                    add_event(
                        job_id,
                        "PaperRAG postprocess completed with errors.",
                        level="warning",
                        payload=postprocess,
                    )

        update_job(job_id, phase="collecting_images", progress=25)
        images = collect_images(paper_id)
        update_paper_extraction(
            paper_id,
            latest_job_id=job_id,
            status="running",
            phase="collecting_images",
            image_count=len(images),
        )
        add_event(job_id, f"Collected {len(images)} extracted figure images from PaperRAG.", payload=paper_snapshot(paper_id))
        if not images:
            error = "No images extracted from PDF."
            update_job(job_id, status="failed", phase="failed", progress=100, error=error)
            update_paper_extraction(
                paper_id,
                latest_job_id=job_id,
                status="failed",
                phase="failed",
                image_count=0,
                error=error,
            )
            return

        candidate_dir = config.DATA_ROOT / "jobs" / job_id / "candidates"
        panel_count = 0
        target_count = 0
        skipped_count = 0
        candidate_cache_hits = 0
        try:
            current_prompt_sha = str(current_prompt_metadata(prompt_variant).get("prompt_sha256") or "")
        except Exception as exc:
            current_prompt_sha = ""
            add_event(
                job_id,
                "Could not read current vision prompt metadata; figure classification cache disabled.",
                level="warning",
                payload={"error": str(exc)},
            )
        update_job(job_id, chart_count=0, phase="segmenting_panels", progress=35)
        update_paper_extraction(
            paper_id,
            latest_job_id=job_id,
            status="running",
            phase="segmenting_panels",
            image_count=len(images),
        )
        for image_index, image_path in enumerate(images, start=1):
            progress = 35 + int(image_index / max(1, len(images)) * 35)
            update_job(job_id, phase=f"segmenting panels {image_index}/{len(images)}", progress=progress)
            candidate_name = _candidate_filename(paper_id, image_index, image_path)
            candidate_path = candidate_dir / candidate_name
            cached = _find_cached_figure_analysis(
                paper_id=paper_id,
                current_job_id=job_id,
                candidate_name=candidate_name,
                current_prompt_sha=current_prompt_sha,
            )
            if cached:
                figure_analysis, cached_path = cached
                candidate_cache_hits += 1
                reused_payload = json.loads(json.dumps(figure_analysis))
                reused_payload["cache_metadata"] = {
                    "source": "previous_candidate_analysis",
                    "source_candidate": str(cached_path),
                    "prompt_sha256": current_prompt_sha,
                }
                figure_analysis = reused_payload
                add_event(
                    job_id,
                    f"Reusing cached panel classification for figure image {image_index}/{len(images)}: {image_path.name}",
                    payload={"source_candidate": str(cached_path), "candidate": str(candidate_path)},
                )
            else:
                add_event(job_id, f"Segmenting and classifying panels in figure image {image_index}/{len(images)}: {image_path.name}")
                figure_analysis = analyze_figure_panels(image_path, prompt_variant)
            figure_analysis = validate_figure_analysis(image_path, figure_analysis)
            _write_json(candidate_path, figure_analysis)
            panels = _panels_from_analysis(figure_analysis)
            add_event(
                job_id,
                f"Detected {len(panels)} panel candidates in {image_path.name}.",
                payload={"figure_summary": figure_analysis.get("figure_summary"), "panel_count": len(panels)},
            )

            for raw_panel_index, panel in enumerate(panels, start=1):
                panel_index = int(panel.get("panel_index") or raw_panel_index)
                panel_count += 1
                chart_id = apply_chart_id_suffix(
                    _panel_id(paper_id, image_index, panel_index, image_path, panel), method
                )
                panel_crop_path, panel_crop_bbox = _save_panel_crop(
                    job_id=job_id,
                    paper_id=paper_id,
                    chart_id=chart_id,
                    image_path=image_path,
                    panel=panel,
                )
                panel_type = str(panel.get("panel_type") or "other").strip().lower()
                is_chart = bool(panel.get("is_chart")) and panel_type == "chart"
                extraction_family = str(panel.get("extraction_family") or "").strip()
                is_target = bool(panel.get("is_extraction_target") or panel.get("is_target_dual_y")) and bool(extraction_family) and is_chart
                # ###수정 categorical 복원: CMSL에서도 기존 categorical 직접 읽기 경로를 허용합니다.
                if method.key == "cmsl_minimal":
                    # chart_fmailiy 다시 4개로 수정
                    is_target = is_target and extraction_family in {
                        "single_axis_line_chart", "dual_y_generic_line_chart",
                        "current_voltage_power_dual_y", "categorical_line_chart"
                    }
                # ###
                issues = list(panel.get("issues") or [])
                update_job(job_id, chart_count=panel_count, phase=f"classifying panel {panel_count}", progress=progress)

                if not is_target:
                    skipped_count += 1
                    status = _skip_status(panel_type, is_chart, is_target)
                    skip_reason = _skip_reason(panel, panel_type, is_chart, is_target)
                    _upsert_skipped_panel(
                        chart_id=chart_id,
                        job_id=job_id,
                        paper_id=paper_id,
                        image_path=image_path,
                        panel_crop_path=panel_crop_path,
                        panel_crop_bbox=panel_crop_bbox,
                        panel=panel,
                        panel_index=panel_index,
                        status=status,
                        skip_reason=skip_reason,
                        method=method.key,
                    )
                    continue

                target_count += 1
                update_job(job_id, phase=f"extracting target chart {target_count}", progress=min(92, progress + 8))
                chart_family = str(panel.get("chart_family") or "") or "target_dual_y"
                if (
                    extraction_family in LINE_CHART_FAMILIES
                    and _bottom_axis_is_categorical(panel)
                ):
                    issues.append(f"CATEGORICAL_X_AXIS_REROUTED_FROM:{extraction_family}")
                    add_event(
                        job_id,
                        f"Panel {chart_id}: bottom axis ticks have no numeric value, rerouting "
                        f"from {extraction_family} to categorical_line_chart instead of tracing pixels.",
                        level="warning",
                    )
                    extraction_family = "categorical_line_chart"
                    chart_family = "categorical_line_chart"
                try:
                    if extraction_family in CATEGORICAL_CHART_FAMILIES:
                        # categorical 처리를 위한 수정: CMSL also produces color filters and traces.
                        categorical_runner = run_categorical_extraction if method.key == "cmsl_minimal" else run_codex_direct_categorical_extraction
                        extraction = categorical_runner(
                            job_id, paper_id, chart_id, image_path, panel
                        )
                    elif method.extraction == "codex_direct":
                        extraction = run_codex_direct_extraction(
                            job_id, paper_id, chart_id, image_path, panel
                        )
                    # ###수정 CMSL method에서만 실험 core를 실행하여 기존 method의 동작을 보존합니다.
                    elif method.extraction == "cmsl_minimal":
                        extraction = run_extraction(
                            job_id, paper_id, chart_id, image_path, panel, method.use_ocr_text, cmsl_minimal=True
                        )
                    # ###
                    else:
                        extraction = run_extraction(
                            job_id, paper_id, chart_id, image_path, panel, method.use_ocr_text
                        )
                    upsert_chart(
                        chart_id=chart_id,
                        job_id=job_id,
                        method=method.key,
                        paper_id=paper_id,
                        image_name=_panel_name(image_path, panel, panel_index),
                        image_rel_path=_image_rel_path(paper_id, image_path),
                        image_abs_path=str(extraction.get("panel_path") or panel_crop_path),
                        status="extracted",
                        panel_type="chart",
                        extraction_family=extraction_family,
                        chart_family=chart_family,
                        is_chart=1,
                        is_target_dual_y=1,
                        confidence=float(panel.get("confidence") or 0),
                        skip_reason=None,
                        result_dir=str(extraction["result_dir"]),
                        overlay_path=str(extraction["overlay_path"]) if extraction["overlay_path"] else None,
                        csv_path=str(extraction["csv_path"]) if extraction["csv_path"] else None,
                        series_count=int(extraction["series_count"]),
                        issues=issues + list(extraction.get("issues") or []),
                    )
                    add_event(job_id, f"Extracted data for target chart panel {chart_id}.", payload={"series_count": extraction["series_count"]})
                except Exception as exc:
                    skipped_count += 1
                    upsert_chart(
                        chart_id=chart_id,
                        job_id=job_id,
                        method=method.key,
                        paper_id=paper_id,
                        image_name=_panel_name(image_path, panel, panel_index),
                        image_rel_path=_image_rel_path(paper_id, image_path),
                        image_abs_path=str(panel_crop_path),
                        status="extraction_failed",
                        panel_type="chart",
                        extraction_family=extraction_family,
                        chart_family=chart_family,
                        is_chart=1,
                        is_target_dual_y=1,
                        confidence=float(panel.get("confidence") or 0),
                        skip_reason=str(exc),
                        result_dir=None,
                        overlay_path=None,
                        csv_path=None,
                        series_count=0,
                        issues=issues + [f"EXTRACTION_FAILED:{exc}"],
                    )
                    add_event(job_id, f"Extraction failed for target chart panel {chart_id}.", level="error", payload={"error": str(exc)})

        final_status = "completed" if target_count else "completed_with_skips"
        update_job(
            job_id,
            status=final_status,
            phase="completed",
            progress=100,
            chart_count=panel_count,
            target_count=target_count,
            skipped_count=skipped_count,
        )
        update_paper_extraction(
            paper_id,
            latest_job_id=job_id,
            status=final_status,
            phase="completed",
            image_count=len(images),
            target_count=target_count,
            skipped_count=skipped_count,
            error=None,
        )
        add_event(
            job_id,
            "Job completed.",
            payload={
                "figure_image_count": len(images),
                "panel_count": panel_count,
                "target_count": target_count,
                "skipped_count": skipped_count,
                "candidate_cache_hits": candidate_cache_hits,
            },
        )
    except Exception as exc:
        update_job(job_id, status="failed", phase="failed", progress=100, error=str(exc))
        update_paper_extraction(
            paper_id,
            latest_job_id=job_id,
            status="failed",
            phase="failed",
            error=str(exc),
        )
        add_event(job_id, "Job failed unexpectedly.", level="error", payload={"error": str(exc)})

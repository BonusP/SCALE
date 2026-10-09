from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .db import (
    create_job,
    delete_job,
    get_chart,
    get_job,
    get_paper_extraction,
    init_db,
    list_charts_for_paper,
    list_charts,
    list_events,
    list_jobs_for_paper,
    list_jobs,
    list_paper_extractions,
    safe_relative,
)
from .services.ocr_backend import read_text_cached
from .services.ocr_geometry import build_ocr_text_geometry
from .services.run_methods import METHODS, available_methods, resolve_method
from .services.vision import available_prompts, resolve_prompt_variant
from .services.paperrag_bridge import list_canonical_papers, paper_snapshot, register_uploaded_pdf
from .services.pipeline import process_job_sequence
from .services.cvp_lab import (
    add_fixture_from_image as cvp_lab_add_fixture_from_image,
    find_near_duplicate_fixture_groups as cvp_lab_find_near_duplicate_fixture_groups,
    get_run as cvp_lab_get_run,
    import_existing_targets as cvp_lab_import_existing_targets,
    import_plotlogic_phase1_dual_y_cvp as cvp_lab_import_plotlogic_phase1,
    list_fixtures as cvp_lab_list_fixtures,
    list_runs as cvp_lab_list_runs,
    run_lab as cvp_lab_run,
    run_result_mask_correction as cvp_lab_run_result_mask_correction,
)


# ###수정 별도 GUI의 title에 CMSL 실험 실행임을 표시합니다.
app = FastAPI(title="PlotLogic CMSL Minimal Experiment")
# ###
app.mount("/static", StaticFiles(directory=config.PROJECT_ROOT / "app" / "static"), name="static")


@app.on_event("startup")
def startup() -> None:
    config.ensure_dirs()
    init_db()


def _file_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    return f"/api/files?path={Path(path).resolve()}"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"png", "jpg", "jpeg", "webp"}:
        return "image"
    if suffix == "json":
        return "json"
    if suffix == "csv":
        return "csv"
    return suffix or "file"


def _chart_artifacts(chart: dict[str, Any]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    seen: set[Path] = set()

    def add(label: str, path: Path | None) -> None:
        if not path or not path.exists() or not path.is_file():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        artifacts.append({
            "label": label,
            "kind": _artifact_kind(resolved),
            "url": _file_url(resolved) or "",
        })

    result_dir_raw = chart.get("result_dir")
    if not result_dir_raw:
        return artifacts
    result_dir = Path(result_dir_raw)
    manifest_path = result_dir / "run_manifest.json"
    add("run_manifest", manifest_path)

    crop_meta_path = result_dir / "crop_meta.json"
    if crop_meta_path.exists():
        try:
            crop_meta = json.loads(crop_meta_path.read_text(encoding="utf-8"))
            source_panel_path = crop_meta.get("source_panel_path")
            if source_panel_path:
                dataset_root = config.DATA_ROOT / "jobs" / str(chart["job_id"]) / "plotlogic_dataset"
                add("panel_image", dataset_root / str(source_panel_path))
        except json.JSONDecodeError:
            pass

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact_map = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
            for label, rel_path in artifact_map.items():
                add(str(label), result_dir / str(rel_path))
        except json.JSONDecodeError:
            pass

    if chart.get("csv_path"):
        add("curve_series_csv", Path(str(chart["csv_path"])))
    for label, filename in [
        ("mask_quality_qa", "mask_quality_qa.json"),
        ("trace_overlay_qa", "trace_overlay_qa.json"),
        ("visual_qa", "visual_qa_summary.json"),
        ("visual_qa_llm_input", "visual_qa_llm_input.json"),
        ("ocr_panel_text", "mvp_ocr_panel_text.json"),
        # ###수정 CMSL의 마스크·설정·VLM 축 연결 결과를 GUI 디버그 화면에 표시합니다.
        ("cmsl_settings", "cmsl_settings.json"),
        ("cmsl_preprocessing", "cmsl_preprocessing.json"),
        ("background_mask", "background_mask.png"),
        ("grid_mask", "grid_mask.png"),
        ("non_curve_mask", "non_curve_mask.png"),
        ("curve_axis_overlay", "curve_axis_overlay.png"),
        ("curve_axis_association", "curve_axis_association.json"),
        # ###
    ]:
        add(label, result_dir / filename)
    return artifacts


def _chart_payload(chart: dict[str, Any]) -> dict[str, Any]:
    payload = dict(chart)
    payload["image_url"] = _file_url(payload.get("image_abs_path"))
    payload["overlay_url"] = _file_url(payload.get("overlay_path"))
    payload["csv_url"] = (
        f"/api/jobs/{payload['job_id']}/charts/{payload['chart_id']}/export.csv"
        if payload.get("csv_path")
        else None
    )
    payload["debug_url"] = (
        f"/debug/{payload['job_id']}/{payload['chart_id']}"
        if payload.get("result_dir")
        else None
    )
    payload["artifacts"] = _chart_artifacts(payload)
    return payload


def _load_result_json(chart: dict[str, Any], filename: str) -> Any:
    result_dir = chart.get("result_dir")
    if not result_dir:
        return None
    path = Path(result_dir) / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _result_file_url(chart: dict[str, Any], filename: str) -> str | None:
    result_dir = chart.get("result_dir")
    if not result_dir:
        return None
    path = Path(result_dir) / filename
    if not path.exists() or not path.is_file():
        return None
    return _file_url(path)


def _debug_images(chart: dict[str, Any]) -> dict[str, str | None]:
    return {
        "panel_image": _file_url(chart.get("image_abs_path")),
        "source_figure": None,
        "plot_crop": _result_file_url(chart, "plot_crop.png"),
        "overlay": _file_url(chart.get("overlay_path")),
        "color_filter_source": _result_file_url(chart, "color_filter_source.png"),
        "sharpened_plot_crop": _result_file_url(chart, "sharpened_plot_crop.png"),
        "role_recolored_crop": _result_file_url(chart, "role_recolored_crop.png"),
    }


def _candidate_panel_from_chart_id(job_id: str, chart_id: str) -> dict[str, Any] | None:
    """Recover the raw codex panel from the job candidate file (pre-refinement artifacts)."""
    match = re.match(r"^(\d{4})_img(\d{3})_P(\d{2})_", chart_id)
    if not match:
        return None
    paper_id, image_index, panel_index = (int(group) for group in match.groups())
    candidate_dir = config.DATA_ROOT / "jobs" / job_id / "candidates"
    prefix = f"{paper_id:04d}_img{image_index:03d}_"
    for path in sorted(candidate_dir.glob(f"{prefix}*.json")):
        analysis = _read_json(path, {})
        if not isinstance(analysis, dict):
            continue
        panels = [panel for panel in analysis.get("panels", []) if isinstance(panel, dict)]
        for order, panel in enumerate(panels, start=1):
            if int(panel.get("panel_index") or order) == panel_index:
                return {"panel": panel, "source": str(path)}
    return None


def _raw_codex_vision(chart: dict[str, Any], inputs: Any) -> dict[str, Any] | None:
    if isinstance(inputs, dict) and isinstance(inputs.get("panel_vision_raw_codex"), dict):
        return {
            "panel_vision": inputs["panel_vision_raw_codex"],
            "coordinate_space": inputs.get("panel_vision_raw_codex_coordinate_space") or "unknown",
            "panel_crop_bbox_source_image_px": inputs.get("panel_crop_bbox_source_image_px"),
            "source": "mvp_extraction_inputs.json",
        }
    recovered = _candidate_panel_from_chart_id(str(chart.get("job_id")), str(chart.get("chart_id")))
    if not recovered:
        return None
    return {
        "panel_vision": recovered["panel"],
        "coordinate_space": "unknown",
        "panel_crop_bbox_source_image_px": (
            inputs.get("panel_crop_bbox_source_image_px") if isinstance(inputs, dict) else None
        ),
        "source": recovered["source"],
    }


def _ocr_geometry(chart: dict[str, Any], inputs: Any) -> dict[str, Any] | None:
    """Pipeline geometry with OCR text boxes; never break the debug page."""
    if not isinstance(inputs, dict):
        return None
    panel_vision = inputs.get("panel_vision")
    if not isinstance(panel_vision, dict):
        return None
    panel_path = Path(str(inputs.get("panel_path") or chart.get("image_abs_path") or ""))
    if not panel_path.exists() or not panel_path.is_file():
        return None
    ocr: dict[str, Any] = {"items": [], "error": "OCR_NOT_ATTEMPTED"}
    result_dir = chart.get("result_dir")
    if result_dir:
        try:
            ocr = read_text_cached(panel_path, Path(result_dir) / "mvp_ocr_panel_text.json")
        except Exception as exc:
            ocr = {"items": [], "error": f"{type(exc).__name__}: {exc}"}
    try:
        result = build_ocr_text_geometry(panel_vision, ocr_items=ocr.get("items") or [])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    result["ocr"] = {
        "backend": ocr.get("backend"),
        "error": ocr.get("error"),
        "cached": ocr.get("cached"),
        "item_count": len(ocr.get("items") or []),
        "items": ocr.get("items") or [],
    }
    return result


_STATIC_DIR = config.PROJECT_ROOT / "app" / "static"


def _render_page(name: str) -> HTMLResponse:
    """Serve a page with its /static asset URLs stamped by file mtime.

    Without the stamp a browser can pair a cached app.js with a freshly edited
    index.html; the stale script then calls el("...") on markup that no longer
    exists, throws at load, and the page renders nothing at all.
    """
    html = (_STATIC_DIR / name).read_text(encoding="utf-8")

    def stamp(match: re.Match[str]) -> str:
        asset = match.group(1)
        path = _STATIC_DIR / asset
        version = int(path.stat().st_mtime) if path.exists() else 0
        return f"/static/{asset}?v={version}"

    html = re.sub(r"/static/([\w.-]+\.(?:js|css))", stamp, html)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _render_page("index.html")


@app.get("/debug/{job_id}/{chart_id}", response_class=HTMLResponse)
def debug_page(job_id: str, chart_id: str) -> HTMLResponse:
    chart = get_chart(job_id, chart_id)
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found.")
    return _render_page("debug.html")


@app.get("/papers/{paper_id}/results", response_class=HTMLResponse)
def paper_results_page(paper_id: int) -> HTMLResponse:
    snapshot = paper_snapshot(paper_id)
    paper_dir = Path(snapshot["paper_dir"])
    if not paper_dir.exists():
        raise HTTPException(status_code=404, detail="Paper not found.")
    return _render_page("paper_results.html")


@app.get("/cvp-lab", response_class=HTMLResponse)
def cvp_lab_page() -> HTMLResponse:
    return _render_page("cvp_lab.html")


def _cvp_fixture_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    meta = fixture.get("meta") if isinstance(fixture.get("meta"), dict) else {}
    return {
        "fixture_id": fixture.get("fixture_id"),
        "image_path": fixture.get("image_path"),
        "image_url": _file_url(fixture.get("image_path")),
        "image_info": fixture.get("image_info") or {},
        "source": {
            "job_id": meta.get("source_job_id"),
            "paper_id": meta.get("source_paper_id"),
            "panel_id": meta.get("source_panel_id"),
            "kind": meta.get("source"),
            "original_filename": meta.get("original_filename"),
        },
    }


def _cvp_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    for field in (
        "overlay_path",
        "plot_crop_path",
        "color_filter_source_path",
        "final_trace_overlay_path",
        "raw_vs_final_trace_overlay_path",
        "role_recolored_crop_path",
        "llm_mask_correction_contact_sheet_path",
        "llm_mask_correction_comparison_path",
        "mask_reconstruction_comparison_path",
        "csv_path",
    ):
        url_field = field.replace("_path", "_url")
        payload[url_field] = _file_url(payload.get(field))
    result_dir = Path(str(payload.get("result_dir") or ""))
    series_candidates = _read_json(result_dir / "series_candidates.json", {})
    candidate_labels: dict[str, str] = {}
    candidate_colors: dict[str, Any] = {}
    for item in series_candidates.get("items", []) if isinstance(series_candidates.get("items"), list) else []:
        if not isinstance(item, dict):
            continue
        series_id = item.get("series_id")
        if not isinstance(series_id, str):
            continue
        candidate_labels[series_id] = str(item.get("legend_label") or series_id)
        candidate_colors[series_id] = item.get("target_color_rgb")

    def mask_payloads(folder_name: str) -> list[dict[str, Any]]:
        folder = result_dir / folder_name
        if not folder.exists():
            return []
        masks: list[dict[str, Any]] = []
        for path in sorted(folder.glob("*.png")):
            series_id = path.stem
            for suffix in (
                "_power_density_after_voltage_exclusion_enhanced",
                "_power_density_after_voltage_exclusion",
                "_after_power_density_exclusion",
                "_voltage_branch",
            ):
                if series_id.endswith(suffix):
                    series_id = series_id[: -len(suffix)]
                    break
            masks.append(
                {
                    "filename": path.name,
                    "series_id": series_id,
                    "label": candidate_labels.get(series_id, series_id),
                    "target_color_rgb": candidate_colors.get(series_id),
                    "url": _file_url(path),
                }
            )
        return masks

    payload["mask_artifacts"] = mask_payloads("masks")
    payload["role_mask_artifacts"] = mask_payloads("role_masks")
    payload["llm_corrected_mask_artifacts"] = mask_payloads("llm_corrected_masks")
    payload["cleaned_mask_artifacts"] = mask_payloads("cleaned_masks")
    payload["reconstructed_mask_artifacts"] = mask_payloads("reconstructed_masks")
    payload["lab_inputs"] = _read_json(result_dir / "cvp_lab_inputs.json", {})
    payload["series_candidates"] = series_candidates
    payload["axis_mapping"] = _read_json(result_dir / "axis_mapping.json", {})
    payload["curve_series_data"] = _read_json(result_dir / "curve_series_data.json", {})
    payload["curve_traces_crop_px"] = _read_json(result_dir / "curve_traces_crop_px.json", {})
    payload["mask_quality_qa"] = _read_json(result_dir / "mask_quality_qa.json", {})
    payload["trace_overlay_qa"] = _read_json(result_dir / "trace_overlay_qa.json", {})
    payload["visual_qa_summary"] = _read_json(result_dir / "visual_qa_summary.json", {})
    payload["visual_qa_llm_input"] = _read_json(result_dir / "visual_qa_llm_input.json", {})
    payload["llm_mask_correction_summary"] = _read_json(result_dir / "llm_mask_correction_summary.json", {})
    payload["mask_reconstruction_summary"] = _read_json(result_dir / "mask_reconstruction_summary.json", {})
    payload["debug_artifacts"] = [
        {
            "label": label,
            "url": _file_url(result_dir / filename),
        }
        for label, filename in [
            ("lab_inputs", "cvp_lab_inputs.json"),
            ("validation", "validation_report.json"),
            ("image_quality", "image_quality_report.json"),
            ("series_candidates", "series_candidates.json"),
            ("curve_traces", "curve_traces_crop_px.json"),
            ("final_traces", "final_curve_traces_crop_px.json"),
            ("curve_series", "curve_series_data.json"),
            ("mask_quality_qa", "mask_quality_qa.json"),
            ("trace_overlay_qa", "trace_overlay_qa.json"),
            ("visual_qa", "visual_qa_summary.json"),
            ("visual_qa_llm_input", "visual_qa_llm_input.json"),
            ("llm_mask_correction_input", "llm_mask_correction_input.json"),
            ("llm_mask_correction_plan", "llm_mask_correction_plan.json"),
            ("llm_mask_correction_applied", "llm_mask_correction_applied.json"),
            ("llm_mask_correction_summary", "llm_mask_correction_summary.json"),
            ("mask_reconstruction_summary", "mask_reconstruction_summary.json"),
        ]
        if (result_dir / filename).exists()
    ]
    return payload


def _cvp_run_payload(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run:
        return None
    payload = dict(run)
    payload["results"] = [_cvp_result_payload(result) for result in run.get("results", [])]
    return payload


@app.get("/api/cvp-lab")
def api_cvp_lab() -> dict[str, Any]:
    fixtures = cvp_lab_list_fixtures()
    return {
        "fixtures": [_cvp_fixture_payload(item) for item in fixtures],
        "near_duplicate_groups": cvp_lab_find_near_duplicate_fixture_groups(fixtures),
        "runs": [_cvp_run_payload(item) for item in cvp_lab_list_runs()],
    }


@app.post("/api/cvp-lab/import-existing")
def api_cvp_lab_import_existing() -> dict[str, Any]:
    result = cvp_lab_import_existing_targets()
    fixtures = cvp_lab_list_fixtures()
    return {
        **result,
        "fixtures": [_cvp_fixture_payload(item) for item in fixtures],
        "near_duplicate_groups": cvp_lab_find_near_duplicate_fixture_groups(fixtures),
    }


@app.post("/api/cvp-lab/import-plotlogic-phase1")
def api_cvp_lab_import_plotlogic_phase1() -> dict[str, Any]:
    result = cvp_lab_import_plotlogic_phase1()
    fixtures = cvp_lab_list_fixtures()
    return {
        **result,
        "fixtures": [_cvp_fixture_payload(item) for item in fixtures],
        "near_duplicate_groups": cvp_lab_find_near_duplicate_fixture_groups(fixtures),
    }


@app.post("/api/cvp-lab/fixtures")
async def api_cvp_lab_upload_fixtures(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="cvp-lab-fixtures-") as tmpdir:
        tmp_root = Path(tmpdir)
        for file in files:
            filename = file.filename or "fixture.png"
            suffix = Path(filename).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
                skipped.append({"filename": filename, "reason": "unsupported_file_type"})
                continue
            tmp_path = tmp_root / filename
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            try:
                imported.append(cvp_lab_add_fixture_from_image(tmp_path, original_filename=filename))
            except Exception as exc:
                skipped.append({"filename": filename, "reason": str(exc)})
    fixtures = cvp_lab_list_fixtures()
    return {
        "imported_count": len(imported),
        "imported": imported,
        "skipped": skipped,
        "fixtures": [_cvp_fixture_payload(item) for item in fixtures],
        "near_duplicate_groups": cvp_lab_find_near_duplicate_fixture_groups(fixtures),
    }


@app.post("/api/cvp-lab/run")
def api_cvp_lab_run() -> dict[str, Any]:
    try:
        run = cvp_lab_run()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run": _cvp_run_payload(run)}


@app.get("/api/cvp-lab/runs/{run_id}")
def api_cvp_lab_get_run(run_id: str) -> dict[str, Any]:
    run = _cvp_run_payload(cvp_lab_get_run(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Lab run not found.")
    return {"run": run}


@app.post("/api/cvp-lab/runs/{run_id}/results/{fixture_id}/mask-correction")
def api_cvp_lab_mask_correction(run_id: str, fixture_id: str) -> dict[str, Any]:
    try:
        summary = cvp_lab_run_result_mask_correction(run_id, fixture_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run = _cvp_run_payload(cvp_lab_get_run(run_id))
    return {"summary": summary, "run": run}


@app.post("/api/jobs")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # ###수정 업로드 API 설명에도 CMSL 기본 method를 표시합니다.
    method: str | None = Query(None, description="Extraction method key. Defaults to cmsl_minimal."),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file is required.")
    with tempfile.TemporaryDirectory(prefix="plotlogic-mvp-upload-") as tmpdir:
        tmp_pdf = Path(tmpdir) / "upload.pdf"
        with tmp_pdf.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        registered = register_uploaded_pdf(tmp_pdf, file.filename)
    paper_id = int(registered["paper_id"])
    duplicate = bool(registered.get("duplicate"))
    if duplicate:
        latest = get_paper_extraction(paper_id)
        if latest and latest.get("status") in {"queued", "running"}:
            return {
                "job_id": latest.get("latest_job_id"),
                "paper_id": paper_id,
                "status": latest.get("status"),
                "paper": paper_snapshot(paper_id),
                "duplicate": True,
                "duplicate_match_source": registered.get("match_source"),
                "reused_running_job": True,
            }
    try:
        queued = _queue_methods(
            # ###수정 실험 GUI에서 PDF 업로드 기본 method를 cmsl_minimal로 지정합니다.
            paper_id, file.filename, str(registered["title"]), method or "cmsl_minimal", background_tasks
            # ###
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": queued[0]["job_id"],
        "jobs": queued,
        "paper_id": paper_id,
        "status": "queued",
        "paper": paper_snapshot(paper_id),
        "duplicate": duplicate,
        "duplicate_match_source": registered.get("match_source"),
        "reused_running_job": False,
    }


@app.get("/api/jobs")
def api_list_jobs() -> dict[str, Any]:
    return {"jobs": list_jobs()}


@app.delete("/api/jobs/{job_id}")
def api_delete_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Job is still in progress; wait for it to finish or fail first.")

    jobs_root = (config.DATA_ROOT / "jobs").resolve()
    job_dir = (jobs_root / job_id).resolve()
    if job_dir != jobs_root and jobs_root not in job_dir.parents:
        raise HTTPException(status_code=400, detail="Refusing to delete a path outside the jobs directory.")
    if job_dir.exists():
        shutil.rmtree(job_dir)

    delete_job(job_id)
    return {"deleted": True, "job_id": job_id}


@app.get("/api/paperrag/papers")
def api_list_paperrag_papers() -> dict[str, Any]:
    extraction_by_id = list_paper_extractions()
    papers = []
    for paper in list_canonical_papers():
        extraction = extraction_by_id.get(int(paper["paper_id"]))
        papers.append({**paper, "extraction": extraction})
    return {"papers": sorted(papers, key=lambda item: int(item["paper_id"]), reverse=True)}


def _latest_charts_by_id(charts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for chart in sorted(charts, key=lambda item: str(item.get("updated_at") or "")):
        latest[str(chart["chart_id"])] = chart
    return sorted(latest.values(), key=lambda item: str(item.get("image_name") or ""))


@app.get("/api/papers/{paper_id}/results")
def api_paper_results(paper_id: int) -> dict[str, Any]:
    snapshot = paper_snapshot(paper_id)
    paper_dir = Path(snapshot["paper_dir"])
    if not paper_dir.exists():
        raise HTTPException(status_code=404, detail="Paper not found.")
    jobs = list_jobs_for_paper(paper_id)
    all_charts = list_charts_for_paper(paper_id)
    charts = [_chart_payload(chart) for chart in _latest_charts_by_id(all_charts)]
    status_counts: dict[str, int] = {}
    for chart in charts:
        status = str(chart.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "paper": snapshot,
        "extraction": get_paper_extraction(paper_id),
        "jobs": jobs,
        "charts": charts,
        "summary": {
            "chart_count": len(charts),
            "extracted_count": sum(1 for chart in charts if chart.get("status") == "extracted"),
            "csv_count": sum(1 for chart in charts if chart.get("csv_path")),
            # is_target_dual_y is scoped to the legacy current_voltage_power_dual_y family (see
            # plotlogic_bridge.overrides_from_vision), so counting only that would hide
            # single_axis_line_chart / dual_y_generic_line_chart panels that were genuinely
            # recognized and extracted. extraction_family is set for every matched family, so it is
            # the accurate "was this one of our targets" signal across all of them.
            "target_count": sum(1 for chart in charts if chart.get("extraction_family")),
            "status_counts": status_counts,
        },
    }


@app.get("/api/prompts")
def api_list_prompts() -> dict[str, Any]:
    return {"prompts": available_prompts()}


def _queue_methods(
    paper_id: int,
    source_name: str,
    title: str,
    method: str | None,
    background_tasks: BackgroundTasks,
) -> list[dict[str, str]]:
    """Queue one job per method so both results exist for the same paper.

    Omitting `method` runs every method, which is the point: one action gives a
    baseline result and an experiment result to compare.
    """
    if method:
        keys = [resolve_method(method).key]
    else:
        # ###수정 실험 GUI 재실행은 기본적으로 CMSL method 한 개만 실행합니다.
        keys = ["cmsl_minimal"]
        # ###
    queued: list[dict[str, str]] = []
    specs: list[tuple[str, str]] = []
    for key in keys:
        run_method = METHODS[key]
        job_id = str(uuid.uuid4())
        create_job(job_id, paper_id, source_name, title, run_method.prompt_variant, run_method.key)
        specs.append((job_id, run_method.key))
        queued.append({"job_id": job_id, "method": run_method.key, "label": run_method.label})
    background_tasks.add_task(process_job_sequence, paper_id, specs)
    return queued


@app.get("/api/methods")
def api_list_methods() -> dict[str, Any]:
    return {"methods": available_methods()}


@app.post("/api/paperrag/papers/{paper_id}/extract")
def api_extract_existing_paper(
    paper_id: int,
    background_tasks: BackgroundTasks,
    method: str | None = Query(None, description="Extraction method key."),
) -> dict[str, Any]:
    snapshot = paper_snapshot(paper_id)
    paper_dir = Path(snapshot["paper_dir"])
    source_pdf = paper_dir / "source.pdf"
    if not source_pdf.exists():
        raise HTTPException(status_code=404, detail="PaperRAG source.pdf not found.")
    metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    title = str(metadata.get("title") or f"Paper {paper_id:04d}")
    latest = get_paper_extraction(paper_id)
    if latest and latest.get("status") in {"queued", "running"}:
        return {
            "job_id": latest.get("latest_job_id"),
            "paper_id": paper_id,
            "status": latest.get("status"),
            "paper": snapshot,
            "reused_running_job": True,
        }
    try:
        queued = _queue_methods(paper_id, source_pdf.name, title, method, background_tasks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": queued[0]["job_id"],
        "jobs": queued,
        "paper_id": paper_id,
        "status": "queued",
        "paper": snapshot,
        "reused_running_job": False,
    }


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job": job,
        "paper": paper_snapshot(int(job["paper_id"])),
        "events": list_events(job_id),
        "charts": [_chart_payload(chart) for chart in list_charts(job_id)],
    }


@app.get("/api/jobs/{job_id}/charts")
def api_list_charts(job_id: str) -> dict[str, Any]:
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"charts": [_chart_payload(chart) for chart in list_charts(job_id)]}


@app.get("/api/jobs/{job_id}/charts/{chart_id}")
def api_get_chart(job_id: str, chart_id: str) -> dict[str, Any]:
    chart = get_chart(job_id, chart_id)
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found.")
    return {
        "chart": _chart_payload(chart),
        "curve_series_data": _load_result_json(chart, "curve_series_data.json"),
        "validation_report": _load_result_json(chart, "validation_report.json"),
        "chart_extraction_strategy": _load_result_json(chart, "chart_extraction_strategy.json"),
        "axis_mapping": _load_result_json(chart, "axis_mapping.json"),
    }


@app.get("/api/jobs/{job_id}/charts/{chart_id}/debug")
def api_get_chart_debug(job_id: str, chart_id: str) -> dict[str, Any]:
    chart = get_chart(job_id, chart_id)
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found.")
    if not chart.get("result_dir"):
        raise HTTPException(status_code=404, detail="Extraction result not found.")
    inputs = _load_result_json(chart, "mvp_extraction_inputs.json")
    images = _debug_images(chart)
    if inputs and isinstance(inputs, dict) and inputs.get("source_figure_image"):
        images["source_figure"] = _file_url(inputs.get("source_figure_image"))
    payload = _chart_payload(chart)
    return {
        "chart": payload,
        "images": images,
        "debug_inputs": inputs,
        "raw_codex_vision": _raw_codex_vision(chart, inputs),
        "ocr_geometry": _ocr_geometry(chart, inputs),
        "crop_meta": _load_result_json(chart, "crop_meta.json"),
        "exclusion_regions": _load_result_json(chart, "exclusion_regions.json"),
        "axis_mapping": _load_result_json(chart, "axis_mapping.json"),
        "chart_extraction_strategy": _load_result_json(chart, "chart_extraction_strategy.json"),
        "image_quality_report": _load_result_json(chart, "image_quality_report.json"),
        "series_candidates": _load_result_json(chart, "series_candidates.json"),
        "curve_traces_crop_px": _load_result_json(chart, "curve_traces_crop_px.json"),
        "role_recolored_curve_traces_crop_px": _load_result_json(chart, "role_recolored_curve_traces_crop_px.json"),
        "curve_traces_panel_px": _load_result_json(chart, "curve_traces_panel_px.json"),
        "curve_series_data": _load_result_json(chart, "curve_series_data.json"),
        "mask_quality_qa": _load_result_json(chart, "mask_quality_qa.json"),
        "trace_overlay_qa": _load_result_json(chart, "trace_overlay_qa.json"),
        "visual_qa_summary": _load_result_json(chart, "visual_qa_summary.json"),
        "visual_qa_llm_input": _load_result_json(chart, "visual_qa_llm_input.json"),
        "validation_report": _load_result_json(chart, "validation_report.json"),
        "run_manifest": _load_result_json(chart, "run_manifest.json"),
    }


@app.get("/api/jobs/{job_id}/charts/{chart_id}/export.csv")
def api_export_csv(job_id: str, chart_id: str) -> FileResponse:
    chart = get_chart(job_id, chart_id)
    if not chart or not chart.get("csv_path"):
        raise HTTPException(status_code=404, detail="CSV not found.")
    csv_path = Path(chart["csv_path"])
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not found.")
    return FileResponse(csv_path, media_type="text/csv", filename=f"{chart_id}.csv")


@app.get("/api/files")
def api_file(path: str = Query(...)) -> FileResponse:
    target = Path(path).resolve()
    allowed_roots = [config.DATA_ROOT.resolve(), config.PAPERRAG_PAPERS_DIR.resolve()]
    if not any(target == root or root in target.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="File is outside allowed roots.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(target)

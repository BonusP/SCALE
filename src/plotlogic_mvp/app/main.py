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
)
from .services.ocr_backend import read_text_cached
from .services.ocr_geometry import build_ocr_text_geometry
from .services.run_methods import DEFAULT_METHOD
from .services.paperrag_bridge import list_canonical_papers, paper_snapshot, register_uploaded_pdf
from .services.pipeline import process_job_sequence


app = FastAPI(title="SCALE")
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


@app.post("/api/jobs")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
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
    queued = _queue_job(paper_id, file.filename, str(registered["title"]), background_tasks)
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


def _queue_job(
    paper_id: int,
    source_name: str,
    title: str,
    background_tasks: BackgroundTasks,
) -> list[dict[str, str]]:
    """Queue one extraction job for the paper with the default method."""
    job_id = str(uuid.uuid4())
    create_job(job_id, paper_id, source_name, title, DEFAULT_METHOD.prompt_variant, DEFAULT_METHOD.key)
    background_tasks.add_task(process_job_sequence, paper_id, [(job_id, DEFAULT_METHOD.key)])
    return [{"job_id": job_id, "method": DEFAULT_METHOD.key, "label": DEFAULT_METHOD.label}]


@app.post("/api/paperrag/papers/{paper_id}/extract")
def api_extract_existing_paper(
    paper_id: int,
    background_tasks: BackgroundTasks,
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
    queued = _queue_job(paper_id, source_pdf.name, title, background_tasks)
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

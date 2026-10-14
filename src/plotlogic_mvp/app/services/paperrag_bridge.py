from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import config
from ..db import utc_now


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_manifest() -> list[dict[str, Any]]:
    path = config.PAPERRAG_MANIFEST_PATH
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_manifest(rows: list[dict[str, Any]]) -> None:
    path = config.PAPERRAG_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in sorted(rows, key=lambda r: int(r["paper_id"]))]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _canonical_manifest_row(paper_dir: Path) -> dict[str, Any] | None:
    if not paper_dir.is_dir() or not paper_dir.name.isdigit():
        return None
    paper_id = int(paper_dir.name)
    metadata = _read_json(paper_dir / "metadata.json", {})
    source_pdf = paper_dir / "source.pdf"
    parser = metadata.get("parser") if isinstance(metadata.get("parser"), dict) else {}
    embedded_image_count = len([path for path in (paper_dir / "opendataloader").glob("*_images/*") if path.parent.name != "page_images"])
    actual_image_count = embedded_image_count + len(
        list((paper_dir / "opendataloader" / "page_images").glob("page-*.png"))
    )
    image_count = max(int(parser.get("image_count") or 0), actual_image_count)
    return {
        "paper_id": paper_id,
        "archive_dir": metadata.get("source_archive_dir") or "",
        "title": metadata.get("title") or f"Paper {paper_id:04d}",
        "authors": metadata.get("authors") or "",
        "year": metadata.get("year"),
        "doi": metadata.get("doi") or "",
        "journal": metadata.get("journal") or "",
        "pdf_path": str(source_pdf if source_pdf.exists() else metadata.get("source_pdf_path") or ""),
        "pdf_name": source_pdf.name if source_pdf.exists() else "",
        "pdf_size_bytes": source_pdf.stat().st_size if source_pdf.exists() else 0,
        "has_data_json": False,
        "has_source_summary": bool((paper_dir / "document.md").exists()),
        "has_figures": bool(image_count or (paper_dir / "opendataloader").exists()),
        "has_tables": False,
        "figures_count": image_count,
        "tables_count": 0,
        "sha256": metadata.get("source_pdf_sha256") or "",
    }


def _load_manifest_with_canonical() -> list[dict[str, Any]]:
    rows_by_id: dict[int, dict[str, Any]] = {}
    for row in _load_manifest():
        try:
            rows_by_id[int(row["paper_id"])] = row
        except Exception:
            continue
    for paper_dir in sorted(config.PAPERRAG_PAPERS_DIR.glob("*")):
        row = _canonical_manifest_row(paper_dir)
        if row:
            paper_id = int(row["paper_id"])
            merged = dict(rows_by_id.get(paper_id) or {})
            merged.update({key: value for key, value in row.items() if value not in (None, "")})
            rows_by_id[paper_id] = merged
    return [rows_by_id[paper_id] for paper_id in sorted(rows_by_id)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paperrag_python() -> str:
    candidate = config.PAPERRAG_ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else "python")


def _run_paperrag_command(stage: str, cmd: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(config.PAPERRAG_ROOT))
    if config.PAPERRAG_JAVA_HOME:
        java_bin = str(Path(config.PAPERRAG_JAVA_HOME) / "bin")
        env["JAVA_HOME"] = config.PAPERRAG_JAVA_HOME
        env["PATH"] = java_bin + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(config.PAPERRAG_ROOT),
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout or config.PAPERRAG_POSTPROCESS_TIMEOUT_SECONDS,
        )
        ok = result.returncode == 0
        return {
            "stage": stage,
            "ok": ok,
            "returncode": result.returncode,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "stage": stage,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": f"timed out after {exc.timeout}s",
        }
    except Exception as exc:
        return {
            "stage": stage,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": "",
            "stderr_tail": "",
            "error": str(exc),
        }


def _skipped_stage(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "ok": True,
        "skipped": True,
        "reason": reason,
        "returncode": None,
        "duration_seconds": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _has_korean_summary(metadata: dict[str, Any]) -> bool:
    summary = metadata.get("korean_summary")
    if not isinstance(summary, dict):
        return False
    return any(bool(value) for value in summary.values())


def _write_ingest_status(paper_id: int, status: dict[str, Any]) -> None:
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    status_path = paper_dir / "plotlogic_mvp_ingest.json"
    _write_json(status_path, status)
    metadata_path = paper_dir / "metadata.json"
    metadata = _read_json(metadata_path, {})
    plotlogic = metadata.get("plotlogic_mvp") if isinstance(metadata.get("plotlogic_mvp"), dict) else {}
    plotlogic["rag_ingest"] = {
        "status": status.get("status"),
        "updated_at_utc": status.get("updated_at_utc"),
        "stage_count": len(status.get("stages") or []),
        "ok_stage_count": sum(1 for stage in status.get("stages") or [] if stage.get("ok")),
        "status_path": str(status_path),
    }
    metadata["plotlogic_mvp"] = plotlogic
    _write_json(metadata_path, metadata)


def run_paperrag_postprocess(paper_id: int) -> dict[str, Any]:
    """Best-effort PaperRAG ingest after PDF parsing and PlotLogic extraction.

    Heavy LLM/vector stages can fail because a backend is offline or local Qdrant is locked.
    We record those failures instead of failing the PlotLogic extraction job.
    """
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    now = utc_now()
    status: dict[str, Any] = {
        "paper_id": paper_id,
        "status": "running",
        "started_at_utc": now,
        "updated_at_utc": now,
        "stages": [],
    }
    _write_ingest_status(paper_id, status)
    python = _paperrag_python()

    metadata = _read_json(paper_dir / "metadata.json", {})
    commands: list[tuple[str, list[str] | None, str | None]] = [
        (
            "detail_rag_files_pre_semantic",
            [
                python,
                "-c",
                (
                    "from paperrag.repository import PaperRepository; "
                    f"repo=PaperRepository(); repo.get_paper_detail({paper_id}); repo.get_rag_view({paper_id}); "
                    f"print('built detail/rag for {paper_id:04d}')"
                ),
            ],
            None,
        ),
        (
            "semantic_extraction",
            None
            if (paper_dir / "extraction" / "semantic.json").exists()
            else [
                python,
                "scripts/extract_paper_semantic.py",
                "--paper-id",
                str(paper_id),
                "--backend",
                config.PAPERRAG_SEMANTIC_BACKEND,
                "--model",
                config.PAPERRAG_SEMANTIC_MODEL,
                "--timeout",
                str(config.PAPERRAG_POSTPROCESS_TIMEOUT_SECONDS),
            ],
            "extraction/semantic.json already exists",
        ),
        (
            "korean_summary",
            None
            if _has_korean_summary(metadata)
            else [
                python,
                "scripts/generate_korean_summary.py",
                "--paper-id",
                str(paper_id),
                "--backend",
                config.PAPERRAG_SEMANTIC_BACKEND,
                "--model",
                config.PAPERRAG_KOREAN_SUMMARY_MODEL,
                "--timeout",
                str(config.PAPERRAG_POSTPROCESS_TIMEOUT_SECONDS),
            ],
            "metadata.korean_summary already exists",
        ),
        (
            "detail_rag_files_post_semantic",
            [
                python,
                "-c",
                (
                    "from paperrag.repository import PaperRepository; "
                    f"repo=PaperRepository(); repo.get_paper_detail({paper_id}); repo.get_rag_view({paper_id}); "
                    f"print('rebuilt detail/rag for {paper_id:04d}')"
                ),
            ],
            None,
        ),
        (
            "corpus_index",
            [python, "scripts/index_corpus.py", "--paper-id", str(paper_id)],
            None,
        ),
    ]

    for stage, cmd, skip_reason in commands:
        result = _skipped_stage(stage, skip_reason) if cmd is None else _run_paperrag_command(stage, cmd)
        status["stages"].append(result)
        status["updated_at_utc"] = utc_now()
        status["status"] = "running_with_errors" if any(not item.get("ok") for item in status["stages"]) else "running"
        _write_ingest_status(paper_id, status)

    failed = [stage for stage in status["stages"] if not stage.get("ok")]
    status["status"] = "completed_with_errors" if failed else "completed"
    status["completed_at_utc"] = utc_now()
    status["updated_at_utc"] = status["completed_at_utc"]
    status["failed_stages"] = [stage["stage"] for stage in failed]
    status["outputs"] = {
        "metadata": str(paper_dir / "metadata.json"),
        "semantic": str(paper_dir / "extraction" / "semantic.json"),
        "rag_info": str(paper_dir / "rag" / "info.json"),
        "document_chunks": str(paper_dir / "rag" / "document_chunks.jsonl"),
        "structured_units": str(paper_dir / "rag" / "structured_units.jsonl"),
        "qdrant_index": str(config.PAPERRAG_ROOT / "index" / "qdrant"),
    }
    _write_ingest_status(paper_id, status)
    return status


def next_paper_id() -> int:
    ids = set()
    for row in _load_manifest_with_canonical():
        try:
            ids.add(int(row["paper_id"]))
        except Exception:
            pass
    for folder in config.PAPERRAG_PAPERS_DIR.glob("*"):
        if folder.is_dir() and folder.name.isdigit():
            ids.add(int(folder.name))
    return (max(ids) + 1) if ids else 0


def find_duplicate_pdf_by_sha256(file_hash: str) -> dict[str, Any] | None:
    if not file_hash:
        return None
    for row in _load_manifest_with_canonical():
        if str(row.get("sha256") or "") == file_hash:
            paper_id = int(row["paper_id"])
            paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
            # A manifest row can outlive its paper folder (partially copied data roots).
            # Reusing that paper_id would parse a missing source.pdf, so re-register instead.
            if not (paper_dir / "source.pdf").exists():
                continue
            return {
                "paper_id": paper_id,
                "paper_dir": paper_dir,
                "pdf_path": paper_dir / "source.pdf",
                "title": row.get("title") or f"Paper {paper_id:04d}",
                "match_source": "manifest_sha256",
            }

    for paper_dir in sorted(config.PAPERRAG_PAPERS_DIR.glob("*")):
        if not paper_dir.is_dir() or not paper_dir.name.isdigit():
            continue
        metadata = _read_json(paper_dir / "metadata.json", {})
        if str(metadata.get("source_pdf_sha256") or "") == file_hash:
            if not (paper_dir / "source.pdf").exists():
                continue
            paper_id = int(paper_dir.name)
            return {
                "paper_id": paper_id,
                "paper_dir": paper_dir,
                "pdf_path": paper_dir / "source.pdf",
                "title": metadata.get("title") or f"Paper {paper_id:04d}",
                "match_source": "metadata_source_pdf_sha256",
            }

    for paper_dir in sorted(config.PAPERRAG_PAPERS_DIR.glob("*")):
        source_pdf = paper_dir / "source.pdf"
        if not paper_dir.is_dir() or not paper_dir.name.isdigit() or not source_pdf.exists():
            continue
        try:
            if _sha256(source_pdf) == file_hash:
                paper_id = int(paper_dir.name)
                metadata = _read_json(paper_dir / "metadata.json", {})
                return {
                    "paper_id": paper_id,
                    "paper_dir": paper_dir,
                    "pdf_path": source_pdf,
                    "title": metadata.get("title") or f"Paper {paper_id:04d}",
                    "match_source": "source_pdf_sha256_scan",
                }
        except OSError:
            continue
    return None


def register_uploaded_pdf(source_pdf: Path, original_name: str) -> dict[str, Any]:
    file_hash = _sha256(source_pdf)
    duplicate = find_duplicate_pdf_by_sha256(file_hash)
    if duplicate:
        return {
            **duplicate,
            "pdf_path": Path(duplicate["pdf_path"]),
            "duplicate": True,
            "source_pdf_sha256": file_hash,
        }

    paper_id = next_paper_id()
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    paper_dir.mkdir(parents=True, exist_ok=False)
    target_pdf = paper_dir / "source.pdf"
    shutil.copy2(source_pdf, target_pdf)
    title = Path(original_name).stem.replace("_", " ").replace("-", " ").strip() or f"Uploaded paper {paper_id:04d}"
    now = utc_now()

    metadata = {
        "paper_id": paper_id,
        "title": title,
        "title_source": "plotlogic_mvp_upload_filename",
        "authors": "",
        "authors_source": "unknown",
        "year": None,
        "year_source": "unknown",
        "doi": "",
        "doi_source": "unknown",
        "journal": "",
        "journal_source": "unknown",
        "source_archive_dir": "",
        "source_pdf_path": str(target_pdf),
        "source_pdf_sha256": file_hash,
        "parse_status": "pending",
        "canonical_source": "plotlogic_mvp_upload",
        "parser": {},
        "plotlogic_mvp": {
            "uploaded_at_utc": now,
            "original_filename": original_name,
        },
    }
    parse = {
        "paper_id": paper_id,
        "parser": config.PAPERRAG_OPENDATALOADER_CMD,
        "source_pdf": str(target_pdf),
        "status": "pending",
    }
    _write_json(paper_dir / "metadata.json", metadata)
    _write_json(paper_dir / "parse.json", parse)
    (paper_dir / "document.md").write_text(f"# {title}\n\nUploaded via PlotLogic MVP.\n", encoding="utf-8")

    rows = [row for row in _load_manifest_with_canonical() if int(row.get("paper_id", -1)) != paper_id]
    rows.append(
        {
            "paper_id": paper_id,
            "archive_dir": "",
            "title": title,
            "authors": "",
            "year": None,
            "doi": "",
            "pdf_path": str(target_pdf),
            "pdf_name": original_name,
            "pdf_size_bytes": target_pdf.stat().st_size,
            "has_data_json": False,
            "has_source_summary": False,
            "has_figures": False,
            "has_tables": False,
            "figures_count": 0,
            "tables_count": 0,
            "sha256": file_hash,
        }
    )
    _write_manifest(rows)
    return {
        "paper_id": paper_id,
        "paper_dir": paper_dir,
        "pdf_path": target_pdf,
        "title": title,
        "duplicate": False,
        "source_pdf_sha256": file_hash,
    }


# ###수정 Windows 실행 보완: opendataloader가 없을 때 embedded-image PDF adapter를 사용합니다.
def run_opendataloader_parse(paper_id: int) -> dict[str, Any]:
    if config.PDF_IMAGE_FALLBACK_ENABLED and not (shutil.which(config.PAPERRAG_OPENDATALOADER_CMD) or Path(config.PAPERRAG_OPENDATALOADER_CMD).is_file()):
        from .portable_pdf import extract_pdf_images
        try:
            return extract_pdf_images(paper_id)
        except Exception as error:
            return {"ok": False, "error": f"Portable PDF parser failed: {error}", "output_dir": config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}" / "opendataloader"}
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    source_pdf = paper_dir / "source.pdf"
    output_dir = paper_dir / "opendataloader"
    output_dir.mkdir(parents=True, exist_ok=True)
    parse_path = paper_dir / "parse.json"
    parse = _read_json(parse_path, {})
    parse.update(
        {
            "paper_id": paper_id,
            "parser": config.PAPERRAG_OPENDATALOADER_CMD,
            "source_pdf": str(source_pdf),
            "output_dir": str(output_dir),
            "status": "running",
        }
    )
    _write_json(parse_path, parse)
    cmd = [
        config.PAPERRAG_OPENDATALOADER_CMD,
        str(source_pdf),
        "--output-dir",
        str(output_dir),
        "--format",
        "markdown,json",
    ]
    env = os.environ.copy()
    if config.PAPERRAG_JAVA_HOME:
        java_bin = str(Path(config.PAPERRAG_JAVA_HOME) / "bin")
        env["JAVA_HOME"] = config.PAPERRAG_JAVA_HOME
        env["PATH"] = java_bin + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=900)
    except FileNotFoundError as exc:
        parse.update({"status": "error", "error": f"opendataloader-pdf not found: {exc}"})
        _write_json(parse_path, parse)
        return {"ok": False, "error": parse["error"], "output_dir": output_dir}
    except subprocess.TimeoutExpired as exc:
        parse.update({"status": "error", "error": f"opendataloader-pdf timed out after {exc.timeout}s"})
        _write_json(parse_path, parse)
        return {"ok": False, "error": parse["error"], "output_dir": output_dir}

    if result.returncode != 0:
        parse.update(
            {
                "status": "error",
                "error": result.stderr[-4000:] or f"opendataloader-pdf exited {result.returncode}",
            }
        )
        _write_json(parse_path, parse)
        return {"ok": False, "error": parse["error"], "output_dir": output_dir}

    markdown_candidates = sorted(output_dir.glob("*.md"))
    if markdown_candidates:
        (paper_dir / "document.md").write_text(markdown_candidates[0].read_text(encoding="utf-8"), encoding="utf-8")
    image_paths = collect_images(paper_id, render_fallback=True)
    metadata = _read_json(paper_dir / "metadata.json", {})
    metadata["parse_status"] = "success"
    metadata["canonical_source"] = "opendataloader"
    metadata["parser"] = {
        "output_dir": str(output_dir),
        "markdown_path": str(markdown_candidates[0]) if markdown_candidates else "",
        "json_path": str(sorted(output_dir.glob("*.json"))[0]) if sorted(output_dir.glob("*.json")) else "",
        "image_count": len(image_paths),
    }
    parse.update({"status": "success", "output_dir": str(output_dir)})
    parse.pop("error", None)
    _write_json(paper_dir / "metadata.json", metadata)
    _write_json(parse_path, parse)
    return {"ok": True, "output_dir": output_dir, "image_count": len(image_paths)}


def _figure_reference_count(paper_dir: Path) -> int:
    texts: list[str] = []
    for path in sorted((paper_dir / "opendataloader").glob("*.md")):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            pass
    for path in sorted((paper_dir / "opendataloader").glob("*.json")):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            pass
    figure_ids: set[int] = set()
    for text in texts:
        for match in re.finditer(r"\bFig(?:ure)?\.?\s*(\d+)", text, flags=re.IGNORECASE):
            try:
                figure_ids.add(int(match.group(1)))
            except ValueError:
                pass
    return len(figure_ids)


_CONTENT_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:fig(?:ure)?\.?\s*\d+|scheme\s*\d+)(?:\s*(?:[.:),;–-]|\be\b)|$)",
    re.IGNORECASE,
)
_BODY_FIGURE_REFERENCE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:fig(?:ure)?\.?\s*\d+|scheme\s*\d+)\s+"
    r"(?:shows?|represents?|illustrates?|depicts?|presents?|displays?|summari[sz]es?|compares?|indicates?)\b",
    re.IGNORECASE,
)


def _is_content_figure_caption(caption: str) -> bool:
    caption = re.sub(r"^#+\s*", "", str(caption or "")).lstrip("-* ").strip()
    return bool(_CONTENT_FIGURE_CAPTION_RE.search(caption)) and not _BODY_FIGURE_REFERENCE_RE.search(caption)


def _filtered_detail_figure_paths(paper_dir: Path) -> list[Path]:
    detail_info = _read_json(paper_dir / "detail" / "info.json", {})
    figures = detail_info.get("figures") if isinstance(detail_info, dict) else []
    if not isinstance(figures, list):
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    for fig in figures:
        if not isinstance(fig, dict):
            continue
        is_content_figure = _is_content_figure_caption(str(fig.get("caption") or ""))
        is_page_fallback = str(fig.get("source") or "") == "page_render_fallback"
        if not is_content_figure and not is_page_fallback:
            continue
        relpath = str(fig.get("path") or "").strip()
        if not relpath:
            continue
        path = (paper_dir / relpath).resolve()
        try:
            path.relative_to(paper_dir.resolve())
        except ValueError:
            continue
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _render_pdf_pages_for_fallback(paper_dir: Path) -> list[Path]:
    source_pdf = paper_dir / "source.pdf"
    output_dir = paper_dir / "opendataloader" / "page_images"
    existing = sorted(output_dir.glob("page-*.png"))
    if existing:
        return existing
    if not source_pdf.exists():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"
    result = subprocess.run(
        ["pdftoppm", "-png", "-r", "200", str(source_pdf), str(prefix)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        return []
    return sorted(output_dir.glob("page-*.png"))


def collect_images(paper_id: int, *, render_fallback: bool = True) -> list[Path]:
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    opdl_dir = paper_dir / "opendataloader"
    # PDF 입력 방향 보정: 기존 파싱 캐시도 렌더링된 이미지로 전달합니다.
    from .pdf_input import rendered_pdf_inputs
    rendered = rendered_pdf_inputs(paper_dir, create=render_fallback)
    if rendered:
        return rendered
    filtered_figures = _filtered_detail_figure_paths(paper_dir)
    if filtered_figures:
        return filtered_figures

    images: list[Path] = []
    for path in sorted(opdl_dir.glob("*_images/*")):
        if path.parent.name != "page_images" and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            images.append(path)
    page_images = sorted((opdl_dir / "page_images").glob("page-*.png"))
    if render_fallback and os.getenv("PLOTLOGIC_MVP_PAGE_RENDER_FALLBACK", "1").lower() not in {"0", "false", "no"}:
        figure_count = _figure_reference_count(paper_dir)
        if figure_count > len(images):
            page_images = _render_pdf_pages_for_fallback(paper_dir)
    images.extend(path for path in page_images if path.is_file())
    return images


def list_canonical_papers() -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    for paper_dir in sorted(config.PAPERRAG_PAPERS_DIR.glob("*")):
        if not paper_dir.is_dir() or not paper_dir.name.isdigit():
            continue
        paper_id = int(paper_dir.name)
        metadata = _read_json(paper_dir / "metadata.json", {})
        parse = _read_json(paper_dir / "parse.json", {})
        source_pdf = paper_dir / "source.pdf"
        title = metadata.get("title")
        if not title:
            title = source_pdf.stem if source_pdf.exists() else f"Paper {paper_id:04d}"
        papers.append(
            {
                "paper_id": paper_id,
                "title": title,
                "paper_dir": str(paper_dir),
                "source_pdf_exists": source_pdf.exists(),
                "source_pdf_name": source_pdf.name if source_pdf.exists() else "",
                "parse_status": parse.get("status") or metadata.get("parse_status") or "missing",
                "image_count": len(collect_images(paper_id, render_fallback=False)),
                "metadata_exists": (paper_dir / "metadata.json").exists(),
                "parse_exists": (paper_dir / "parse.json").exists(),
            }
        )
    return papers


def paper_snapshot(paper_id: int) -> dict[str, Any]:
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    return {
        "paper_id": paper_id,
        "paper_dir": str(paper_dir),
        "metadata": _read_json(paper_dir / "metadata.json", {}),
        "parse": _read_json(paper_dir / "parse.json", {}),
        "image_count": len(collect_images(paper_id, render_fallback=False)),
    }

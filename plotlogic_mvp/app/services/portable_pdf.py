from __future__ import annotations

# ###수정 Windows 실행 보완: opendataloader가 없을 때 PDF의 embedded chart images를 실험 저장소에 추출합니다.
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .. import config


def poppler_command() -> str | None:
    installed = shutil.which("pdftoppm")
    if installed:
        return installed
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe"
    return str(bundled) if bundled.exists() else None


def extract_pdf_images(paper_id: int) -> dict[str, Any]:
    paper_dir = config.PAPERRAG_PAPERS_DIR / f"{paper_id:04d}"
    output = paper_dir / "opendataloader"
    image_dir = output / "source_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(paper_dir / "source.pdf")
    figures = []
    text_parts = []
    rendered_pages = False
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
        images = list(page.images)
        for image_index, item in enumerate(images, start=1):
            suffix = Path(item.name).suffix.lower() or ".png"
            path = image_dir / f"page-{page_index:03d}-image-{image_index:03d}{suffix}"
            data = item.data
            # Keep the original JPEG stream for direct image XObjects instead of a decoded/reencoded JPEG.
            if item.indirect_reference is not None:
                obj = item.indirect_reference.get_object()
                filters = obj.get("/Filter", [])
                if filters == "/DCTDecode" or "/DCTDecode" in filters:
                    data = obj.get_data()
            path.write_bytes(data)
            caption = next((line.strip() for line in page_text.splitlines() if line.strip().lower().startswith(("fig.", "figure "))), f"Fig. {len(figures) + 1}. PDF embedded chart image.")
            figures.append({"path": path.relative_to(paper_dir).as_posix(), "caption": caption, "page": page_index, "source": "pypdf_embedded_image"})
        if not images:
            command = poppler_command()
            if not command:
                return {"ok": False, "error": "PDF page has no embedded images and pdftoppm is unavailable", "output_dir": output}
            pages = output / "page_images"
            pages.mkdir(parents=True, exist_ok=True)
            if not rendered_pages:
                result = subprocess.run([command, "-png", "-r", "200", str(paper_dir / "source.pdf"), str(pages / "page")], capture_output=True, timeout=300)
                if result.returncode:
                    return {"ok": False, "error": "PDF page rendering failed", "output_dir": output}
                rendered_pages = True
            candidates = [path for path in pages.glob("page-*.png") if int(path.stem.split("-")[-1]) == page_index]
            if candidates:
                path = candidates[0]
                figures.append({"path": path.relative_to(paper_dir).as_posix(), "caption": f"Rendered PDF page {page_index} (fallback image for vector figures)", "page": page_index, "source": "pdf_page_render_fallback"})
    text = "\n\n".join(text_parts)
    (paper_dir / "document.md").write_text(text, encoding="utf-8")
    (output / "source.md").write_text(text, encoding="utf-8")
    detail = {"figures": figures, "images": [item["path"] for item in figures], "parser": "pypdf_embedded_images"}
    (paper_dir / "detail_info.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    parse_path = paper_dir / "parse.json"
    parse = json.loads(parse_path.read_text(encoding="utf-8")) if parse_path.exists() else {}
    parse.update({"status": "success", "parser": "pypdf_embedded_images", "image_count": len(figures), "output_dir": str(output)})
    parse_path.write_text(json.dumps(parse, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = paper_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update({"parse_status": "success", "parser": {"name": "pypdf_embedded_images", "image_count": len(figures)}})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "output_dir": output, "image_count": len(figures), "parser": "pypdf_embedded_images"}
# ###

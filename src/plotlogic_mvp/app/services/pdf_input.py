from __future__ import annotations

# PDF 입력 방향 보정
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader

from .portable_pdf import poppler_command


def _image_nodes(value: Any):
    if isinstance(value, dict):
        if value.get("type") == "image":
            yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _image_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _image_nodes(child)


def _pixel_bbox(bbox, page_size, image_size):
    # PDF 입력 방향 보정: OpenDataLoader bbox는 회전된 CropBox 기준, 좌하단 원점입니다.
    left, bottom, right, top = map(float, bbox)
    if not all(math.isfinite(v) for v in (left, bottom, right, top)) or right <= left or top <= bottom:
        raise ValueError("Invalid PDF image bounding box")
    width, height = page_size
    px_width, px_height = image_size
    result = (
        max(0, min(px_width, math.floor(left / width * px_width))),
        max(0, min(px_height, math.floor((height - top) / height * px_height))),
        max(0, min(px_width, math.ceil(right / width * px_width))),
        max(0, min(px_height, math.ceil((height - bottom) / height * px_height))),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("PDF image bounding box is outside the rendered page")
    return result


def rendered_pdf_inputs(paper_dir: Path, *, create: bool = True) -> list[Path] | None:
    # PDF 입력 방향 보정: 원본 추출물은 보존하고 새 이름으로 캐시를 분리합니다.
    source = paper_dir / "source.pdf"
    layout_path = paper_dir / "opendataloader" / "source.json"
    parse_path = paper_dir / "parse.json"
    if not source.exists():
        return None
    layout_bytes = layout_path.read_bytes() if layout_path.exists() else b""
    layout = json.loads(layout_bytes) if layout_bytes else {}
    nodes = list(_image_nodes(layout))
    parse = json.loads(parse_path.read_text(encoding="utf-8")) if parse_path.exists() else {}
    portable = parse.get("parser") == "pypdf_embedded_images"
    if not nodes and not portable:
        return None

    digest = hashlib.sha256(b"pdf-render-v1\0" + source.read_bytes() + layout_bytes).hexdigest()
    output = paper_dir / "opendataloader" / "pdf_render_v1" / digest[:16]
    manifest = output / "inputs.json"
    if manifest.exists():
        records = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [output / item["file"] for item in records["images"]]
        if records.get("sha256") == digest and paths and all(p.is_file() for p in paths):
            return paths

    # PDF 입력 방향 보정: 목록 조회에서는 렌더링 작업을 시작하지 않습니다.
    if not create:
        return None

    command = poppler_command()
    if not command:
        raise RuntimeError("PDF input orientation correction requires pdftoppm (Poppler)")
    reader = PdfReader(source)
    # PDF 입력 방향 보정: 이전 pypdf 공유 이미지 캐시는 페이지당 한 번 렌더링합니다.
    if not nodes:
        nodes = [{"page number": i, "bounding box": None} for i in range(1, len(reader.pages) + 1)]
    output.mkdir(parents=True, exist_ok=True)
    pages = {}
    records = []
    for index, node in enumerate(nodes, start=1):
        number = int(node["page number"])
        if not 1 <= number <= len(reader.pages):
            raise ValueError(f"Invalid PDF image page number: {number}")
        page = reader.pages[number - 1]
        if number not in pages:
            prefix = output / f"page-{number:04d}"
            result = subprocess.run(
                [command, "-f", str(number), "-l", str(number), "-singlefile", "-cropbox",
                 "-png", "-r", "200", str(source), str(prefix)],
                capture_output=True, timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"PDF page rendering failed: {result.stderr.decode('utf-8', errors='replace')[-1000:]}")
            pages[number] = prefix.with_suffix(".png")
        width, height = float(page.cropbox.width), float(page.cropbox.height)
        if page.rotation % 180:
            width, height = height, width
        bbox = node.get("bounding box")
        if bbox is None and not portable:
            raise ValueError("OpenDataLoader image is missing its bounding box")
        filename = f"pdf_render_v1_{digest[:8]}_{index:04d}.png"
        with Image.open(pages[number]) as image:
            rect = _pixel_bbox(bbox, (width, height), image.size) if bbox is not None else (0, 0, *image.size)
            image.crop(rect).save(output / filename)
        records.append({"file": filename, "page": number, "bbox_pdf": bbox,
                        "bbox_render_px": list(rect), "original_source": node.get("source")})
    manifest.write_text(json.dumps({"sha256": digest, "method": "pdf_page_render_crop", "images": records},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return [output / item["file"] for item in records]

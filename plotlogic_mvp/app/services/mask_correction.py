from __future__ import annotations

import json
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

from .. import config


BBOX4 = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}

MASK_CORRECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string"},
        "overall_notes": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "filename": {"type": "string"},
                    "action": {"type": "string"},
                    "confidence": {"type": "number"},
                    "keep_bboxes_crop_px": {"type": "array", "items": BBOX4},
                    "remove_bboxes_crop_px": {"type": "array", "items": BBOX4},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": [
                    "filename",
                    "action",
                    "confidence",
                    "keep_bboxes_crop_px",
                    "remove_bboxes_crop_px",
                    "issues",
                    "notes",
                ],
            },
        },
    },
    "required": ["status", "overall_notes", "items"],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _bbox(raw: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in raw):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in raw]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _base_mask_paths(result_dir: Path) -> list[Path]:
    masks_dir = result_dir / "masks"
    if not masks_dir.exists():
        return []
    skip_tokens = (
        "_power_density_after_voltage_exclusion",
        "_after_power_density_exclusion",
        "_voltage_branch",
    )
    return [
        path
        for path in sorted(masks_dir.glob("*.png"))
        if not any(token in path.stem for token in skip_tokens)
    ]


def _thumb(image: Image.Image, *, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, "white")
    out.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return out


def _make_contact_sheet(result_dir: Path, mask_paths: list[Path]) -> Path:
    plot_path = result_dir / "plot_crop.png"
    color_path = result_dir / "color_filter_source.png"
    tile_w, tile_h = 260, 170
    label_h = 38
    rows = 2 + len(mask_paths)
    sheet = Image.new("RGB", (tile_w * 2 + 24, rows * (tile_h + label_h) + 16), "#f8fafc")
    draw = ImageDraw.Draw(sheet)

    def paste_row(row: int, title: str, left_path: Path | None, right_path: Path | None = None) -> None:
        y = 8 + row * (tile_h + label_h)
        draw.text((12, y), title, fill="#0f172a")
        for col, path in enumerate([left_path, right_path]):
            if not path or not path.exists():
                continue
            with Image.open(path) as image:
                tile = _thumb(image, size=(tile_w, tile_h))
            x = 12 + col * (tile_w + 12)
            sheet.paste(tile, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + tile_w, y + label_h + tile_h), outline="#cbd5e1")

    paste_row(0, "plot_crop.png | color_filter_source.png", plot_path, color_path)
    paste_row(1, "mask rows below use original crop pixel coordinates", None, None)
    for index, mask_path in enumerate(mask_paths, start=2):
        paste_row(index, f"{mask_path.name}", mask_path, plot_path)

    out_path = result_dir / "llm_mask_correction_contact_sheet.png"
    sheet.save(out_path)
    return out_path


def _mask_manifest(result_dir: Path, mask_paths: list[Path]) -> dict[str, Any]:
    series_candidates = _read_json(result_dir / "series_candidates.json", {})
    candidate_by_id = {
        item.get("series_id"): item
        for item in series_candidates.get("items", [])
        if isinstance(item, dict) and isinstance(item.get("series_id"), str)
    }
    masks: list[dict[str, Any]] = []
    for path in mask_paths:
        with Image.open(path) as image:
            width, height = image.size
        candidate = candidate_by_id.get(path.stem, {})
        masks.append(
            {
                "filename": path.name,
                "series_id": path.stem,
                "legend_label": candidate.get("legend_label"),
                "target_color_rgb": candidate.get("target_color_rgb"),
                "size_wh": [width, height],
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "coordinate_space": "crop_px",
        "prompt": str(config.PROMPTS_ROOT / "visual_qa" / "mask_correction.md"),
        "masks": masks,
        "visual_qa_summary": _read_json(result_dir / "visual_qa_summary.json", {}),
        "mask_quality_qa": _read_json(result_dir / "mask_quality_qa.json", {}),
    }


def _call_codex_for_plan(result_dir: Path, contact_sheet: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not config.USE_CODEX:
        return {
            "status": "skipped",
            "overall_notes": "Codex is disabled.",
            "items": [],
            "issues": ["CODEX_DISABLED"],
        }
    prompt_path = config.PROMPTS_ROOT / "visual_qa" / "mask_correction.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt += "\n\nMask manifest:\n"
    prompt += json.dumps(manifest, indent=2, ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix="plotlogic-mask-correction-") as tmpdir:
        tmp = Path(tmpdir)
        schema_path = tmp / "schema.json"
        output_path = tmp / "last_message.json"
        schema_path.write_text(json.dumps(MASK_CORRECTION_SCHEMA, indent=2, ensure_ascii=True), encoding="utf-8")
        cmd = [
            config.CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            str(config.PROJECT_ROOT),
            "--model",
            config.CODEX_MODEL,
            "-c",
            f'model_reasoning_effort="{config.CODEX_REASONING_EFFORT}"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--image",
            str(contact_sheet),
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=config.CODEX_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            return {"status": "failed", "overall_notes": str(exc), "items": [], "issues": ["CODEX_NOT_FOUND"]}
        except subprocess.TimeoutExpired:
            return {"status": "failed", "overall_notes": "Codex timed out.", "items": [], "issues": ["CODEX_TIMEOUT"]}
        if result.returncode != 0:
            return {
                "status": "failed",
                "overall_notes": result.stderr[-1200:],
                "items": [],
                "issues": ["CODEX_FAILED"],
            }
        if not output_path.exists():
            return {"status": "failed", "overall_notes": "Codex produced no output.", "items": [], "issues": ["CODEX_NO_OUTPUT"]}
        return json.loads(output_path.read_text(encoding="utf-8"))


def _blank_outside_keep_boxes(mask: Image.Image, keep_boxes: list[list[int]]) -> Image.Image:
    if not keep_boxes:
        return mask
    keep = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(keep)
    for bbox in keep_boxes:
        draw.rectangle(tuple(bbox), fill=255)
    src = mask.convert("L")
    out = Image.new("L", mask.size, 0)
    out.paste(src, mask=keep)
    return out


def _apply_item_plan(source_path: Path, item: dict[str, Any], out_path: Path) -> dict[str, Any]:
    with Image.open(source_path) as source:
        mask = source.convert("L")
    width, height = mask.size
    action = str(item.get("action") or "use_original")
    keep_boxes = [_bbox(raw, width, height) for raw in item.get("keep_bboxes_crop_px", []) if isinstance(raw, list)]
    remove_boxes = [_bbox(raw, width, height) for raw in item.get("remove_bboxes_crop_px", []) if isinstance(raw, list)]
    keep_boxes = [box for box in keep_boxes if box]
    remove_boxes = [box for box in remove_boxes if box]

    if action == "reject":
        corrected = Image.new("L", mask.size, 0)
    elif action == "apply_bboxes":
        corrected = _blank_outside_keep_boxes(mask, keep_boxes)
        draw = ImageDraw.Draw(corrected)
        for box in remove_boxes:
            draw.rectangle(tuple(box), fill=0)
    else:
        corrected = mask

    out_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.save(out_path)
    return {
        "filename": source_path.name,
        "corrected_filename": out_path.name,
        "action": action,
        "keep_bboxes_crop_px": keep_boxes,
        "remove_bboxes_crop_px": remove_boxes,
        "issues": item.get("issues") if isinstance(item.get("issues"), list) else [],
        "notes": item.get("notes") or "",
    }


def _apply_plan(result_dir: Path, mask_paths: list[Path], plan: dict[str, Any]) -> dict[str, Any]:
    corrected_dir = result_dir / "llm_corrected_masks"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    items_by_filename = {
        item.get("filename"): item
        for item in plan.get("items", [])
        if isinstance(item, dict) and isinstance(item.get("filename"), str)
    }
    applied: list[dict[str, Any]] = []
    for source_path in mask_paths:
        item = items_by_filename.get(source_path.name) or {
            "filename": source_path.name,
            "action": "use_original",
            "keep_bboxes_crop_px": [],
            "remove_bboxes_crop_px": [],
            "issues": ["NO_LLM_PLAN_FOR_MASK"],
            "notes": "",
        }
        applied.append(_apply_item_plan(source_path, item, corrected_dir / source_path.name))
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "source": "llm_mask_correction_plan",
        "corrected_mask_dir": str(corrected_dir),
        "items": applied,
    }


def _component_endpoint(points: np.ndarray, *, side: str) -> tuple[int, int]:
    x_values = points[:, 1]
    target_x = int(x_values.min() if side == "left" else x_values.max())
    near = points[np.abs(points[:, 1] - target_x) <= 1]
    return target_x, int(round(float(np.median(near[:, 0]))))


def _component_summary(mask: np.ndarray, label_id: int) -> dict[str, Any]:
    points = np.argwhere(mask == label_id)
    ys = points[:, 0]
    xs = points[:, 1]
    left = _component_endpoint(points, side="left")
    right = _component_endpoint(points, side="right")
    return {
        "label_id": int(label_id),
        "size": int(points.shape[0]),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "x_span": int(xs.max() - xs.min() + 1),
        "y_span": int(ys.max() - ys.min() + 1),
        "left_endpoint_xy": [left[0], left[1]],
        "right_endpoint_xy": [right[0], right[1]],
    }


def _clean_and_reconstruct_mask(source_path: Path, cleaned_path: Path, reconstructed_path: Path) -> dict[str, Any]:
    with Image.open(source_path) as source:
        binary = np.array(source.convert("L")) > 0
    height, width = binary.shape
    total_pixels = int(binary.sum())
    if total_pixels == 0:
        empty = Image.new("L", (width, height), 0)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        reconstructed_path.parent.mkdir(parents=True, exist_ok=True)
        empty.save(cleaned_path)
        empty.save(reconstructed_path)
        return {
            "filename": source_path.name,
            "pixel_count": 0,
            "component_count": 0,
            "kept_component_count": 0,
            "removed_component_count": 0,
            "bridge_count": 0,
            "issues": ["EMPTY_MASK"],
        }

    structure = np.ones((3, 3), dtype=bool)
    labels, component_count = ndimage.label(binary, structure=structure)
    component_rows = [_component_summary(labels, label_id) for label_id in range(1, component_count + 1)]
    min_pixels = max(6, int(round(total_pixels * 0.003)))
    min_x_span = max(3, int(round(width * 0.006)))
    kept = [
        item
        for item in component_rows
        if item["size"] >= min_pixels or (item["size"] >= 4 and item["x_span"] >= min_x_span)
    ]
    kept_ids = {item["label_id"] for item in kept}
    cleaned = np.isin(labels, list(kept_ids)) if kept_ids else np.zeros_like(binary)

    reconstructed_image = Image.fromarray((cleaned * 255).astype("uint8"), mode="L")
    draw = ImageDraw.Draw(reconstructed_image)
    bridge_count = 0
    max_gap = max(10, int(round(width * 0.045)))
    max_dy = max(10, int(round(height * 0.09)))
    by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in kept:
        by_start[item["bbox_xyxy"][0]].append(item)
    ordered = sorted(kept, key=lambda item: (item["bbox_xyxy"][0], -item["x_span"]))
    used_pairs: set[tuple[int, int]] = set()
    for left_item in ordered:
        lx, ly = left_item["right_endpoint_xy"]
        candidates: list[tuple[float, dict[str, Any]]] = []
        for right_item in ordered:
            if right_item["label_id"] == left_item["label_id"]:
                continue
            rx, ry = right_item["left_endpoint_xy"]
            gap = rx - lx
            if gap <= 0 or gap > max_gap:
                continue
            dy = abs(ry - ly)
            if dy > max_dy:
                continue
            candidates.append((gap + dy * 0.75, right_item))
        if not candidates:
            continue
        _score, right_item = min(candidates, key=lambda item: item[0])
        pair = (left_item["label_id"], right_item["label_id"])
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        rx, ry = right_item["left_endpoint_xy"]
        draw.line([(lx, ly), (rx, ry)], fill=255, width=2)
        bridge_count += 1

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((cleaned * 255).astype("uint8"), mode="L").save(cleaned_path)
    reconstructed_image.save(reconstructed_path)
    return {
        "filename": source_path.name,
        "pixel_count": total_pixels,
        "component_count": int(component_count),
        "kept_component_count": len(kept),
        "removed_component_count": int(component_count) - len(kept),
        "bridge_count": bridge_count,
        "min_pixels": min_pixels,
        "min_x_span": min_x_span,
        "max_gap_px": max_gap,
        "max_dy_px": max_dy,
        "issues": [] if kept else ["NO_COMPONENTS_AFTER_CLEANING"],
    }


def _reconstruct_masks(result_dir: Path, applied: dict[str, Any]) -> dict[str, Any]:
    cleaned_dir = result_dir / "cleaned_masks"
    reconstructed_dir = result_dir / "reconstructed_masks"
    rows = applied.get("items", []) if isinstance(applied.get("items"), list) else []
    items: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("corrected_filename") or item.get("filename") or "")
        if not filename:
            continue
        source_path = result_dir / "llm_corrected_masks" / filename
        if not source_path.exists():
            source_path = result_dir / "masks" / filename
        if not source_path.exists():
            continue
        items.append(
            _clean_and_reconstruct_mask(
                source_path,
                cleaned_dir / filename,
                reconstructed_dir / filename,
            )
        )
    payload = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "source": "programmatic_reconstruction_after_llm_mask_correction",
        "cleaned_mask_dir": str(cleaned_dir),
        "reconstructed_mask_dir": str(reconstructed_dir),
        "item_count": len(items),
        "bridge_count": sum(int(item.get("bridge_count") or 0) for item in items),
        "removed_component_count": sum(int(item.get("removed_component_count") or 0) for item in items),
        "items": items,
    }
    _write_json(result_dir / "mask_reconstruction_summary.json", payload)
    return payload


def _comparison_sheet(result_dir: Path, applied: dict[str, Any]) -> Path:
    rows = applied.get("items", []) if isinstance(applied.get("items"), list) else []
    tile_w, tile_h = 190, 125
    label_h = 34
    columns = [
        ("raw", "masks"),
        ("llm corrected", "llm_corrected_masks"),
        ("cleaned", "cleaned_masks"),
        ("reconstructed", "reconstructed_masks"),
    ]
    sheet = Image.new("RGB", (tile_w * len(columns) + 12 * (len(columns) + 1), max(1, len(rows)) * (tile_h + label_h) + 16), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(rows):
        y = 8 + index * (tile_h + label_h)
        filename = str(item.get("filename") or "")
        draw.text((12, y), f"{filename} | {item.get('action') or ''}", fill="#0f172a")
        for col, (label, folder) in enumerate(columns):
            path = result_dir / folder / filename
            if not path.exists():
                continue
            with Image.open(path) as image:
                tile = _thumb(image, size=(tile_w, tile_h))
            x = 12 + col * (tile_w + 12)
            draw.text((x, y + 15), label, fill="#475569")
            sheet.paste(tile, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + tile_w, y + label_h + tile_h), outline="#cbd5e1")
    out_path = result_dir / "mask_reconstruction_comparison.png"
    sheet.save(out_path)
    return out_path


def run_llm_mask_correction(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    mask_paths = _base_mask_paths(result_dir)
    if not mask_paths:
        payload = {
            "schema_version": 1,
            "generated_at_utc": _utc_now(),
            "status": "failed",
            "issues": ["NO_BASE_COLOR_MASKS"],
            "items": [],
        }
        _write_json(result_dir / "llm_mask_correction_summary.json", payload)
        return payload

    contact_sheet = _make_contact_sheet(result_dir, mask_paths)
    manifest = _mask_manifest(result_dir, mask_paths)
    _write_json(result_dir / "llm_mask_correction_input.json", manifest)
    plan = _call_codex_for_plan(result_dir, contact_sheet, manifest)
    plan["schema_version"] = 1
    plan["generated_at_utc"] = _utc_now()
    plan["contact_sheet"] = str(contact_sheet)
    _write_json(result_dir / "llm_mask_correction_plan.json", plan)
    applied = _apply_plan(result_dir, mask_paths, plan)
    _write_json(result_dir / "llm_mask_correction_applied.json", applied)
    reconstruction = _reconstruct_masks(result_dir, applied)
    comparison = _comparison_sheet(result_dir, applied)
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "status": plan.get("status") or "unknown",
        "overall_notes": plan.get("overall_notes") or "",
        "mask_count": len(mask_paths),
        "corrected_count": sum(1 for item in applied["items"] if item.get("action") == "apply_bboxes"),
        "rejected_count": sum(1 for item in applied["items"] if item.get("action") == "reject"),
        "use_original_count": sum(1 for item in applied["items"] if item.get("action") == "use_original"),
        "reconstructed_mask_count": reconstruction["item_count"],
        "bridge_count": reconstruction["bridge_count"],
        "removed_component_count": reconstruction["removed_component_count"],
        "artifacts": {
            "input": "llm_mask_correction_input.json",
            "contact_sheet": "llm_mask_correction_contact_sheet.png",
            "plan": "llm_mask_correction_plan.json",
            "applied": "llm_mask_correction_applied.json",
            "reconstruction_summary": "mask_reconstruction_summary.json",
            "comparison": "mask_reconstruction_comparison.png",
            "corrected_masks": "llm_corrected_masks/",
            "cleaned_masks": "cleaned_masks/",
            "reconstructed_masks": "reconstructed_masks/",
        },
        "comparison_path": str(comparison),
        "items": applied["items"],
    }
    _write_json(result_dir / "llm_mask_correction_summary.json", summary)
    return summary

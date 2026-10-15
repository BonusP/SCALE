from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .. import config


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _white_pixel_points(mask_path: Path) -> tuple[int, int, set[tuple[int, int]]]:
    with Image.open(mask_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        pixels = gray.load()
        points = {
            (x, y)
            for y in range(height)
            for x in range(width)
            if pixels[x, y] > 0
        }
    return width, height, points


def _component_sizes(points: set[tuple[int, int]], *, max_components: int = 400) -> list[int]:
    remaining = set(points)
    sizes: list[int] = []
    while remaining and len(sizes) < max_components:
        start = remaining.pop()
        count = 1
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    count += 1
        sizes.append(count)
    if remaining:
        sizes.append(len(remaining))
    return sorted(sizes, reverse=True)


def _mask_metrics(mask_path: Path) -> dict[str, Any]:
    width, height, points = _white_pixel_points(mask_path)
    point_count = len(points)
    issues: list[str] = []
    if not point_count:
        return {
            "filename": mask_path.name,
            "pixel_count": 0,
            "coverage_ratio": 0.0,
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "bbox_xyxy": None,
            "issues": ["EMPTY_MASK"],
            "severity": "fail",
        }

    xs = [x for x, _y in points]
    ys = [y for _x, y in points]
    bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
    component_sizes = _component_sizes(points)
    component_count = len(component_sizes)
    largest_ratio = component_sizes[0] / max(1, point_count)
    coverage_ratio = point_count / max(1, width * height)
    x_span_ratio = (bbox[2] - bbox[0]) / max(1, width)
    y_span_ratio = (bbox[3] - bbox[1]) / max(1, height)

    if component_count >= 12 and largest_ratio < 0.72:
        issues.append("FRAGMENTED_OR_MULTI_COMPONENT_MASK")
    if coverage_ratio > 0.18:
        issues.append("MASK_TOO_DENSE_POSSIBLE_TEXT_OR_AXIS_CONTAMINATION")
    if x_span_ratio < 0.18:
        issues.append("LOW_X_COVERAGE_POSSIBLE_TRUNCATED_CURVE")
    if y_span_ratio < 0.03 and x_span_ratio > 0.35:
        issues.append("NEAR_HORIZONTAL_MASK_BAND")

    if any(issue.startswith("MASK_TOO_DENSE") or issue.startswith("FRAGMENTED") for issue in issues):
        severity = "warn"
    else:
        severity = "ok"

    return {
        "filename": mask_path.name,
        "pixel_count": point_count,
        "coverage_ratio": round(coverage_ratio, 5),
        "component_count": component_count,
        "largest_component_ratio": round(largest_ratio, 4),
        "bbox_xyxy": bbox,
        "x_span_ratio": round(x_span_ratio, 4),
        "y_span_ratio": round(y_span_ratio, 4),
        "issues": issues,
        "severity": severity,
    }


def build_mask_quality_qa(result_dir: Path) -> dict[str, Any]:
    masks: list[dict[str, Any]] = []
    for folder_name in ("masks", "role_masks"):
        folder = result_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.png")):
            item = _mask_metrics(path)
            item["group"] = folder_name
            item["relative_path"] = str(path.relative_to(result_dir))
            masks.append(item)

    issue_count = sum(1 for item in masks if item.get("issues"))
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "source": "programmatic_mask_quality_prefilter",
        "llm_prompt_ref": str(config.PROMPTS_ROOT / "visual_qa" / "mask_quality.md"),
        "mask_count": len(masks),
        "issue_count": issue_count,
        "status": "needs_review" if issue_count else "ok",
        "items": masks,
    }


def _points(raw: Any) -> list[list[int]]:
    if not isinstance(raw, list):
        return []
    out: list[list[int]] = []
    for point in raw:
        if isinstance(point, list) and len(point) >= 2 and all(isinstance(value, (int, float)) for value in point[:2]):
            out.append([int(round(point[0])), int(round(point[1]))])
    return out


def _longest_same_y_run(points: list[list[int]]) -> int:
    longest = current = 1
    for left, right in zip(points, points[1:]):
        if right[1] == left[1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest if points else 0


def _direction_changes(points: list[list[int]]) -> int:
    signs: list[int] = []
    for left, right in zip(points, points[1:]):
        diff = right[1] - left[1]
        if diff:
            signs.append(1 if diff > 0 else -1)
    return sum(1 for left, right in zip(signs, signs[1:]) if left != right)


def _trace_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    points = _points(trace.get("points"))
    issues: list[str] = []
    if len(points) < 2:
        return {
            "trace_id": trace.get("trace_id"),
            "series_id": trace.get("series_id"),
            "curve_role": trace.get("curve_role"),
            "axis_binding_hint": trace.get("axis_binding_hint"),
            "point_count": len(points),
            "issues": ["TOO_FEW_TRACE_POINTS_FOR_QA"],
            "severity": "fail",
        }

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    plateau = _longest_same_y_run(points)
    changes = _direction_changes(points)
    normalized_plateau = plateau / max(1, len(points))
    normalized_changes = changes / max(1, len(points) - 2)

    if plateau >= max(8, int(round(len(points) * 0.08))) and x_span > 40:
        issues.append("LONG_HORIZONTAL_PLATEAU")
    if normalized_changes > 0.18 and len(points) >= 30:
        issues.append("EXCESSIVE_ZIGZAG")
    if x_span < 20:
        issues.append("LOW_X_SPAN_TRACE")
    if y_span <= 2 and x_span > 30:
        issues.append("NEAR_HORIZONTAL_TRACE")

    severity = "warn" if issues else "ok"
    return {
        "trace_id": trace.get("trace_id"),
        "series_id": trace.get("series_id"),
        "legend_label": trace.get("legend_label"),
        "curve_role": trace.get("curve_role"),
        "axis_binding_hint": trace.get("axis_binding_hint"),
        "extraction_method": trace.get("extraction_method"),
        "point_count": len(points),
        "x_span_px": x_span,
        "y_span_px": y_span,
        "longest_same_y_run": plateau,
        "direction_change_count": changes,
        "normalized_plateau": round(normalized_plateau, 4),
        "normalized_direction_changes": round(normalized_changes, 4),
        "issues": issues,
        "severity": severity,
    }


def build_trace_overlay_qa(result_dir: Path) -> dict[str, Any]:
    final_payload = _read_json(result_dir / "final_curve_traces_crop_px.json", {})
    raw_payload = _read_json(result_dir / "curve_traces_crop_px.json", {})
    final_traces = final_payload.get("traces") if isinstance(final_payload.get("traces"), list) else []
    raw_traces = raw_payload.get("traces") if isinstance(raw_payload.get("traces"), list) else []
    final_items = [_trace_metrics(trace) for trace in final_traces if isinstance(trace, dict)]
    raw_rejected_count = sum(1 for trace in raw_traces if isinstance(trace, dict) and trace.get("final_trace_status") == "rejected")
    issue_count = sum(1 for item in final_items if item.get("issues"))
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "source": "programmatic_trace_overlay_prefilter",
        "llm_prompt_ref": str(config.PROMPTS_ROOT / "visual_qa" / "trace_overlay.md"),
        "raw_trace_count": len(raw_traces),
        "raw_rejected_trace_count": raw_rejected_count,
        "final_trace_count": len(final_items),
        "issue_count": issue_count,
        "status": "needs_review" if issue_count else "ok",
        "items": final_items,
    }


def build_visual_qa_summary(result_dir: Path, mask_qa: dict[str, Any], trace_qa: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for item in mask_qa.get("items", []):
        for issue in item.get("issues", []) if isinstance(item, dict) else []:
            issues.append(f"MASK:{item.get('relative_path') or item.get('filename')}:{issue}")
    for item in trace_qa.get("items", []):
        for issue in item.get("issues", []) if isinstance(item, dict) else []:
            issues.append(f"TRACE:{item.get('trace_id') or item.get('series_id')}:{issue}")

    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "source": "programmatic_visual_qa_prefilter",
        "status": "needs_review" if issues else "ok",
        "issue_count": len(issues),
        "issues": issues,
        "artifacts": {
            "mask_quality_qa": "mask_quality_qa.json",
            "trace_overlay_qa": "trace_overlay_qa.json",
            "llm_input_bundle": "visual_qa_llm_input.json",
            "mask_quality_prompt": str(config.PROMPTS_ROOT / "visual_qa" / "mask_quality.md"),
            "trace_overlay_prompt": str(config.PROMPTS_ROOT / "visual_qa" / "trace_overlay.md"),
        },
        "human_review_hint": "Use these flags as QA signals only; extraction outputs are not changed by this artifact.",
    }


def build_visual_qa_llm_input(result_dir: Path, mask_qa: dict[str, Any], trace_qa: dict[str, Any]) -> dict[str, Any]:
    artifact_names = [
        "plot_crop.png",
        "color_filter_source.png",
        "role_recolored_crop.png",
        "final_trace_overlay.png",
        "raw_vs_final_trace_overlay.png",
        "overlay.png",
        "series_candidates.json",
        "axis_mapping.json",
        "curve_traces_crop_px.json",
        "final_curve_traces_crop_px.json",
        "curve_series_data.json",
    ]
    artifacts = [
        {"name": name, "path": str(result_dir / name)}
        for name in artifact_names
        if (result_dir / name).exists()
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "purpose": "Input bundle for optional LLM visual QA of C-V-P extraction artifacts.",
        "instructions": {
            "mask_quality_prompt": str(config.PROMPTS_ROOT / "visual_qa" / "mask_quality.md"),
            "trace_overlay_prompt": str(config.PROMPTS_ROOT / "visual_qa" / "trace_overlay.md"),
            "do_not_modify_extraction": True,
        },
        "artifacts": artifacts,
        "programmatic_mask_quality_prefilter": mask_qa,
        "programmatic_trace_overlay_prefilter": trace_qa,
        "expected_llm_output": {
            "status": "ok | needs_review | fail",
            "mask_findings": [],
            "trace_findings": [],
            "recommended_human_checks": [],
        },
    }


def ensure_visual_qa_artifacts(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    mask_qa = build_mask_quality_qa(result_dir)
    trace_qa = build_trace_overlay_qa(result_dir)
    summary = build_visual_qa_summary(result_dir, mask_qa, trace_qa)
    llm_input = build_visual_qa_llm_input(result_dir, mask_qa, trace_qa)
    _write_json(result_dir / "mask_quality_qa.json", mask_qa)
    _write_json(result_dir / "trace_overlay_qa.json", trace_qa)
    _write_json(result_dir / "visual_qa_llm_input.json", llm_input)
    _write_json(result_dir / "visual_qa_summary.json", summary)
    return summary

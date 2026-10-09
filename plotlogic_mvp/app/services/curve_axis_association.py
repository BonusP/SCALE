from __future__ import annotations

# ###수정 차이 18: 추출 curve ID가 표시된 overlay와 축 제목을 VLM에 제공해 y-axis를 대응시킵니다.
import colorsys
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .. import config


ASSOCIATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "assignments": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "trace_id": {"type": "string"},
                    "axis_binding": {"type": ["string", "null"], "enum": ["left_y", "right_y", None]},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                    # ###수정 추가 수정: curve별 올바른 범례와 실제 curve 여부를 판정합니다.
                    "legend_label": {"type": ["string", "null"]},
                    "is_curve": {"type": "boolean"},
                    # ###
                },
                "required": ["trace_id", "axis_binding", "confidence", "reason", "legend_label", "is_curve"],
            },
        },
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["assignments", "issues"],
}


def draw_curve_axis_overlay(panel_image: Image.Image, traces: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    scale = 2
    key_height = max(42, 24 * len(traces) + 20)
    width = max(panel_image.width * scale, 620)
    canvas = Image.new("RGB", (width, panel_image.height * scale + key_height), "white")
    canvas.paste(panel_image.convert("RGB").resize((panel_image.width * scale, panel_image.height * scale)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    inputs = []
    for index, trace in enumerate(traces):
        color = tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(index / max(1, len(traces)), 0.85, 0.75))
        points = [(int(x * scale), int(y * scale)) for x, y in trace.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        label = f"T{index + 1}"
        if points:
            x, y = points[len(points) // 2]
            box = draw.textbbox((x, y), label, font=font)
            draw.rectangle(box, fill="white")
            draw.text((x, y), label, font=font, fill=color)
        key_y = panel_image.height * scale + 12 + index * 24
        draw.line((12, key_y + 5, 38, key_y + 5), fill=color, width=3)
        draw.text((46, key_y), f"{label} = {trace['trace_id']} | {trace.get('legend_label') or '(no legend label)'}", font=font, fill=color)
        inputs.append({
            "trace_id": trace["trace_id"], "overlay_identifier": label,
            "overlay_color_rgb": list(color), "legend_label": trace.get("legend_label"),
            "original_series_color_rgb": trace.get("series_color_rgb"),
            # ###수정 추가 수정: 같은 색의 모든 범례와 support 정보를 VLM에 제공합니다.
            "legend_options": trace.get("legend_options", []),
            "point_count": len(trace.get("points", [])),
            "tracking_support": trace.get("features", {}).get("tracking_support_ref"),
            # ###
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return inputs


def validate_assignments(payload: dict[str, Any], trace_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {item["trace_id"] for item in trace_inputs}
    accepted: dict[str, dict[str, Any]] = {}
    issues = [str(value) for value in payload.get("issues", [])]
    for item in payload.get("assignments", []):
        if not isinstance(item, dict):
            issues.append("INVALID_AXIS_ASSIGNMENT")
            continue
        trace_id = item.get("trace_id")
        if trace_id not in expected:
            issues.append(f"UNKNOWN_TRACE_ID:{trace_id}")
            continue
        if trace_id in accepted:
            accepted[trace_id] = {"trace_id": trace_id, "axis_binding": None, "confidence": 0.0, "reason": "Duplicate assignment"}
            issues.append(f"DUPLICATE_TRACE_ASSIGNMENT:{trace_id}")
            continue
        axis = item.get("axis_binding")
        confidence = item.get("confidence")
        if axis not in {"left_y", "right_y", None} or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            accepted[trace_id] = {"trace_id": trace_id, "axis_binding": None, "confidence": 0.0, "reason": "Invalid assignment value"}
            issues.append(f"INVALID_AXIS_ASSIGNMENT:{trace_id}")
        else:
            # ###수정 추가 수정: curve validity와 label도 axis 판정과 함께 보존합니다.
            accepted[trace_id] = {"trace_id": trace_id, "axis_binding": axis, "confidence": float(confidence), "reason": str(item.get("reason") or ""),
                                  "legend_label": item.get("legend_label"), "is_curve": item.get("is_curve", True) is True}
            # ###
    assignments = []
    for item in trace_inputs:
        trace_id = item["trace_id"]
        assignment = accepted.get(trace_id, {"trace_id": trace_id, "axis_binding": None, "confidence": 0.0, "reason": "No VLM assignment"})
        if assignment["axis_binding"] is None:
            issues.append(f"UNRESOLVED_CURVE_AXIS:{trace_id}")
        assignments.append(assignment)
        # ###수정 추가 수정: VLM의 label은 제공된 실제 범례 후보에 포함된 값만 허용합니다.
        selected_label = assignment.get("legend_label")
        options = item.get("legend_options") or ([{"legend_label": item["legend_label"]}] if item.get("legend_label") else [])
        labels = {option.get("legend_label") for option in options if option.get("legend_label")}
        if selected_label is not None and selected_label not in labels:
            assignment["legend_label"] = None
            issues.append(f"INVALID_LEGEND_ASSIGNMENT:{trace_id}")
        # ###
    return {"assignments": assignments, "issues": list(dict.fromkeys(issues))}


def assign_curve_axes(*, panel_image: Image.Image, traces: list[dict[str, Any]], axis_mapping: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    overlay = out_dir / "curve_axis_overlay.png"
    inputs = draw_curve_axis_overlay(panel_image, traces, overlay)
    axes = {
        "left_y": axis_mapping.get("axes", {}).get("left", {}).get("axis_label") or "",
        "right_y": axis_mapping.get("axes", {}).get("right", {}).get("axis_label") or "",
    }
    base = {
        "schema_version": 1, "modification_note": "###수정 차이 18: extracted curve별 VLM axis 대응",
        "source": "codex_overlay_axis_association", "model": config.CODEX_MODEL,
        "axes": axes, "trace_inputs": inputs, "overlay_path": str(overlay),
    }
    if not traces:
        return {**base, "assignments": [], "issues": ["NO_EXTRACTED_CURVES"]}
    if not config.USE_CODEX:
        return {**base, **validate_assignments({"issues": ["CODEX_DISABLED"]}, inputs)}
    prompt = (config.PROMPTS_ROOT / "curve_axis_association_cmsl.md").read_text(encoding="utf-8")
    prompt += "\n\nInput metadata:\n" + json.dumps({"axes": axes, "curves": inputs}, ensure_ascii=False, indent=2)
    with tempfile.TemporaryDirectory(prefix="cmsl-axis-") as folder:
        schema = Path(folder) / "schema.json"
        output = Path(folder) / "output.json"
        schema.write_text(json.dumps(ASSOCIATION_SCHEMA), encoding="utf-8")
        command = [
            config.CODEX_BIN, "exec", "-c", f'service_tier="{config.CODEX_SERVICE_TIER}"', "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
            "--cd", str(config.PROJECT_ROOT), "--model", config.CODEX_MODEL,
            "-c", f'model_reasoning_effort="{config.CODEX_REASONING_EFFORT}"',
            "--output-schema", str(schema), "--output-last-message", str(output), "--image", str(overlay), "-",
        ]
        try:
            # ###수정 Windows CLI의 혼합 인코딩 로그를 안전하게 읽습니다.
            result = subprocess.run(command, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=config.CODEX_TIMEOUT_SECONDS)
            # ###
            if result.returncode or not output.exists():
                payload = {"issues": [f"AXIS_VLM_FAILED:{result.returncode}", result.stderr[-1200:]]}
            else:
                payload = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {"issues": ["AXIS_VLM_INVALID_JSON_OBJECT"]}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            payload = {"issues": [f"AXIS_VLM_ERROR:{type(error).__name__}:{error}"]}
    report = {**base, **validate_assignments(payload, inputs)}
    (out_dir / "curve_axis_association.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
# ###

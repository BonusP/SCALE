from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .. import config
from .bbox_validation import validate_figure_analysis


BBOX4 = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
POINT2 = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}

AXIS_LABELS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "x": {"type": "string"},
        "left_y": {"type": "string"},
        "right_y": {"type": "string"},
    },
    "required": ["title", "x", "left_y", "right_y"],
}

AXES_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "axis_role": {"type": "string"},
            "axis_line_bbox_panel_px": BBOX4,
            "axis_line_coordinate_px": {"type": "number"},
            "label_bbox_panel_px": BBOX4,
            "ticks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tick_position_px": {"type": "number"},
                        "tick_bbox_panel_px": BBOX4,
                        "label_text": {"type": "string"},
                        "label_numeric_value": {"type": ["number", "null"]},
                        "label_bbox_panel_px": BBOX4,
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "tick_position_px",
                        "tick_bbox_panel_px",
                        "label_text",
                        "label_numeric_value",
                        "label_bbox_panel_px",
                        "confidence",
                    ],
                },
            },
            "confidence": {"type": "number"},
        },
        "required": [
            "axis_role",
            "axis_line_bbox_panel_px",
            "axis_line_coordinate_px",
            "label_bbox_panel_px",
            "ticks",
            "confidence",
        ],
    },
}

LEGEND_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "legend_label": {"type": "string"},
            "label_bbox_panel_px": BBOX4,
            "swatch_color_rgb": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
            "swatch_bbox_panel_px": BBOX4,
            "marker_shape": {"type": ["string", "null"]},
            "confidence": {"type": "number"},
            "axis_binding_hint": {"type": ["string", "null"]},
        },
        "required": [
            "legend_label",
            "label_bbox_panel_px",
            "swatch_color_rgb",
            "swatch_bbox_panel_px",
            "marker_shape",
            "confidence",
            "axis_binding_hint",
        ],
    },
}

ANNOTATIONS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "annotation_text": {"type": "string"},
            "bbox_panel_px": BBOX4,
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["annotation_text", "bbox_panel_px", "reason", "confidence"],
    },
}

PANEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "panel_index": {"type": "integer"},
        "panel_label": {"type": "string"},
        "panel_type": {"type": "string"},
        "panel_bbox_figure_px": BBOX4,
        "is_chart": {"type": "boolean"},
        "is_extraction_target": {"type": "boolean"},
        "extraction_family": {"type": "string"},
        "is_target_dual_y": {"type": "boolean"},
        "chart_family": {"type": "string"},
        "confidence": {"type": "number"},
        "skip_reason": {"type": "string"},
        "plot_frame_bbox_panel_px": BBOX4,
        "origin_panel_px": POINT2,
        "axis_labels": AXIS_LABELS_SCHEMA,
        "axes": AXES_SCHEMA,
        "legend_region_bbox_panel_px": BBOX4,
        "legend_items": LEGEND_ITEMS_SCHEMA,
        "plot_annotation_regions": ANNOTATIONS_SCHEMA,
        "curve_role_rules": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "panel_index",
        "panel_label",
        "panel_type",
        "panel_bbox_figure_px",
        "is_chart",
        "is_extraction_target",
        "extraction_family",
        "is_target_dual_y",
        "chart_family",
        "confidence",
        "skip_reason",
        "plot_frame_bbox_panel_px",
        "origin_panel_px",
        "axis_labels",
        "axes",
        "legend_region_bbox_panel_px",
        "legend_items",
        "plot_annotation_regions",
        "curve_role_rules",
        "issues",
    ],
}

VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "figure_summary": {"type": "string"},
        "panel_count": {"type": "integer"},
        "panels": {"type": "array", "items": PANEL_SCHEMA},
    },
    "required": ["figure_summary", "panel_count", "panels"],
}


def resolve_prompt_variant(variant: str | None = None) -> str:
    """Prompt template for a run: explicit choice, else the configured default.

    Only names of existing templates in PROMPTS_ROOT are accepted, so a bad value
    fails here instead of silently falling back to a different prompt.
    """
    name = (variant or config.VISION_PROMPT_TEMPLATE).strip()
    if "/" in name or "\\" in name:
        raise ValueError(f"prompt variant must be a file name, got {name!r}")
    if not name.startswith("figure_panel_classifier") or not name.endswith(".md"):
        raise ValueError(f"unknown prompt variant: {name!r}")
    if not (config.PROMPTS_ROOT / name).is_file():
        raise ValueError(f"prompt variant not found: {name!r}")
    return name


def _prompt_template_path(variant: str | None = None) -> Path:
    return config.PROMPTS_ROOT / resolve_prompt_variant(variant)


# ###수정 차이 04: prompt별 registry와 hash를 구분해 기존 분류와 CMSL 실험을 분리합니다.
def _chart_families_path(variant: str | None = None) -> Path:
    # chart_fmailiy 다시 4개로 수정
    resolve_prompt_variant(variant)
    return config.PROMPTS_ROOT / "chart_families.json"
# ###


def _render_prompt(template: str, families: dict[str, Any]) -> str:
    return template.replace(
        "{chart_family_rules}",
        json.dumps(families, indent=2, ensure_ascii=False),
    )


def _digest(rendered: str, family_text: str) -> str:
    return hashlib.sha256((rendered + "\n" + family_text).encode("utf-8")).hexdigest()


# ###수정 차이 04: prompt별 registry와 hash를 구분해 기존 분류와 CMSL 실험을 분리합니다.
def _read_prompt_inputs(variant: str | None = None) -> tuple[str, dict[str, Any], str]:
    template = _prompt_template_path(variant).read_text(encoding="utf-8")
    family_text = _chart_families_path(variant).read_text(encoding="utf-8")
    families = json.loads(family_text)
    return _render_prompt(template, families), families, family_text
# ###


# ###수정 차이 04: prompt별 registry와 hash를 구분해 기존 분류와 CMSL 실험을 분리합니다.
def _prompt_metadata(
    prompt: str,
    family_text: str,
    families: dict[str, Any],
    variant: str | None = None,
) -> dict[str, Any]:
    digest = _digest(prompt, family_text)
    return {
        "prompt_template": str(_prompt_template_path(variant)),
        "prompt_variant": resolve_prompt_variant(variant),
        "chart_family_registry": str(_chart_families_path(variant)),
        "chart_family_registry_version": families.get("registry_version", ""),
        "prompt_sha256": digest,
    }
# ###


def _prompt(variant: str | None = None) -> str:
    prompt, _, _ = _read_prompt_inputs(variant)
    return prompt


def current_prompt_metadata(variant: str | None = None) -> dict[str, Any]:
    prompt, families, family_text = _read_prompt_inputs(variant)
    return _prompt_metadata(prompt, family_text, families, variant)


def fallback_panel(reason: str) -> dict[str, Any]:
    return {
        "panel_index": 1,
        "panel_label": "",
        "panel_type": "other",
        "panel_bbox_figure_px": [0, 0, 0, 0],
        "is_chart": False,
        "is_extraction_target": False,
        "extraction_family": "",
        "is_target_dual_y": False,
        "chart_family": "unknown",
        "confidence": 0.0,
        "skip_reason": reason,
        "plot_frame_bbox_panel_px": [0, 0, 0, 0],
        "origin_panel_px": [0, 0],
        "axis_labels": {"title": "", "x": "", "left_y": "", "right_y": ""},
        "axes": [],
        "legend_region_bbox_panel_px": [0, 0, 0, 0],
        "legend_items": [],
        "plot_annotation_regions": [],
        "curve_role_rules": [],
        "issues": [reason],
    }


def fallback_analysis(reason: str, variant: str | None = None) -> dict[str, Any]:
    panel = fallback_panel(reason)
    try:
        prompt, families, family_text = _read_prompt_inputs(variant)
        prompt_meta = _prompt_metadata(prompt, family_text, families, variant)
    except Exception:
        prompt_meta = {}
    return {"figure_summary": reason, "panel_count": 1, "panels": [panel], "prompt_metadata": prompt_meta}


def fallback_result(reason: str) -> dict[str, Any]:
    panel = fallback_panel(reason)
    return {key: value for key, value in panel.items() if key not in {"panel_index", "panel_label", "panel_type", "panel_bbox_figure_px"}}


def analyze_figure_panels(image_path: Path, variant: str | None = None) -> dict[str, Any]:
    if not config.USE_CODEX:
        return fallback_analysis("CODEX_DISABLED", variant)
    prompt, families, family_text = _read_prompt_inputs(variant)
    prompt_meta = _prompt_metadata(prompt, family_text, families, variant)
    with tempfile.TemporaryDirectory(prefix="plotlogic-mvp-vision-") as tmpdir:
        tmp = Path(tmpdir)
        schema_path = tmp / "schema.json"
        output_path = tmp / "last_message.json"
        schema_path.write_text(json.dumps(VISION_SCHEMA, indent=2, ensure_ascii=True), encoding="utf-8")
        cmd = [
            config.CODEX_BIN,
            "exec",
            # ###수정 global 설정을 바꾸지 않고 유효한 CLI service tier를 지정합니다.
            "-c", f'service_tier="{config.CODEX_SERVICE_TIER}"',
            # ###
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
            str(image_path),
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                # ###수정 Windows CLI 응답을 UTF-8로 읽어 한글 JSON의 decoding 오류를 방지합니다.
                encoding="utf-8",
                # ###수정 Windows CLI의 혼합 인코딩 로그가 결과 수신을 중단하지 않게 처리합니다.
                errors="replace",
                # ###
                # ###
                check=False,
                timeout=config.CODEX_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            return fallback_analysis(f"CODEX_NOT_FOUND:{exc}", variant)
        except subprocess.TimeoutExpired:
            return fallback_analysis("CODEX_TIMEOUT", variant)
        if result.returncode != 0:
            payload = fallback_analysis("CODEX_FAILED", variant)
            payload["panels"][0]["issues"].append(result.stderr[-1000:])
            return payload
        if not output_path.exists():
            return fallback_analysis("CODEX_NO_OUTPUT", variant)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["prompt_metadata"] = prompt_meta
        return validate_figure_analysis(image_path, payload)


def analyze_image(image_path: Path, variant: str | None = None) -> dict[str, Any]:
    analysis = analyze_figure_panels(image_path, variant)
    panels = [panel for panel in analysis.get("panels", []) if isinstance(panel, dict)]
    for panel in panels:
        if panel.get("is_extraction_target") or panel.get("is_target_dual_y"):
            return {key: value for key, value in panel.items() if key not in {"panel_index", "panel_label", "panel_type", "panel_bbox_figure_px"}}
    for panel in panels:
        if panel.get("is_chart"):
            return {key: value for key, value in panel.items() if key not in {"panel_index", "panel_label", "panel_type", "panel_bbox_figure_px"}}
    if panels:
        return {key: value for key, value in panels[0].items() if key not in {"panel_index", "panel_label", "panel_type", "panel_bbox_figure_px"}}
    return fallback_result("NO_PANEL_DETECTED")

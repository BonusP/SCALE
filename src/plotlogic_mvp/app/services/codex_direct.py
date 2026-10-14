"""Read a categorical chart's values straight from the panel image with codex.

Used as the fallback when pixel tracing of a categorical chart is incomplete. The
result is written in the same `curve_series_data.json` shape the plotlogic core
produces, so the results table and the CSV export work against it unchanged.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .. import config


CATEGORICAL_PROMPT_TEMPLATE = "categorical_chart_direct.md"

CATEGORY_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {"type": "string"},
        "group_label": {"type": ["string", "null"]},
        "value": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": ["category", "group_label", "value", "confidence"],
}

CATEGORICAL_TABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "panel_summary": {"type": "string"},
        "chart_type": {"type": "string"},
        "value_axis_label": {"type": "string"},
        "value_unit": {"type": "string"},
        "items": {"type": "array", "items": CATEGORY_ITEM_SCHEMA},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["panel_summary", "chart_type", "value_axis_label", "value_unit", "items", "issues"],
}


def _categorical_prompt_path() -> Path:
    return config.PROMPTS_ROOT / CATEGORICAL_PROMPT_TEMPLATE


def _empty_categorical_table(reason: str) -> dict[str, Any]:
    return {
        "panel_summary": reason,
        "chart_type": "unknown",
        "value_axis_label": "",
        "value_unit": "",
        "items": [],
        "issues": [reason],
    }


def read_categorical_table(panel_path: Path, *, overlay_path: Path | None = None, trace_context: dict | None = None) -> dict[str, Any]:
    """Ask codex to read a categorical-x line chart's (category, value) pairs directly, no axis geometry."""
    if not config.USE_CODEX:
        return _empty_categorical_table("CODEX_DISABLED")
    prompt = _categorical_prompt_path().read_text(encoding="utf-8")
    # categorical 처리를 위한 수정: give Codex the traced curves as supporting evidence.
    if trace_context is not None:
        prompt += "\nPixel tracing context (may be incomplete or contain artifacts):\n" + json.dumps(trace_context, ensure_ascii=False)
        prompt += "\nUse the original panel as the authority. The second image shows traced curves, not additional data. Do not fabricate values to fill trace gaps."
    with tempfile.TemporaryDirectory(prefix="plotlogic-mvp-categorical-") as tmpdir:
        tmp = Path(tmpdir)
        schema_path = tmp / "schema.json"
        output_path = tmp / "last_message.json"
        schema_path.write_text(json.dumps(CATEGORICAL_TABLE_SCHEMA, indent=2, ensure_ascii=True), encoding="utf-8")
        cmd = [
            config.CODEX_BIN,
            "exec",
            # chart_fmailiy 다시 4개로 수정: categorical 호출에도 service_tier를 지정합니다.
            "-c", f'service_tier="{config.CODEX_SERVICE_TIER}"',
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
            str(panel_path),
            "-",
        ]
        # categorical 처리를 위한 수정: include a trace overlay alongside the original panel.
        if overlay_path is not None and overlay_path.exists():
            cmd[-1:-1] = ["--image", str(overlay_path)]
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
            return _empty_categorical_table(f"CODEX_NOT_FOUND:{exc}")
        except subprocess.TimeoutExpired:
            return _empty_categorical_table("CODEX_TIMEOUT")
        if result.returncode != 0:
            table = _empty_categorical_table("CODEX_FAILED")
            table["issues"].append(result.stderr[-1000:])
            return table
        if not output_path.exists():
            return _empty_categorical_table("CODEX_NO_OUTPUT")
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _empty_categorical_table(f"CODEX_BAD_JSON:{exc}")


def to_categorical_series_data(panel_id: str, table: dict[str, Any]) -> dict[str, Any]:
    """Convert the codex categorical-line table into the same curve_series_data shape the other methods use.

    There is no x axis here, so points_data pairs a category label (as x) with its numeric value
    (as y) instead of two numbers. That is still a two-item list per point, so the unchanged CSV
    writer, debug page and API responses all keep working against it.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in table.get("items") or []:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        value = item.get("value")
        if not isinstance(category, str) or not category.strip() or not isinstance(value, (int, float)):
            continue
        group_label = item.get("group_label")
        group_key = str(group_label).strip() if isinstance(group_label, str) and group_label.strip() else "values"
        groups.setdefault(group_key, []).append(
            {
                "category": category.strip(),
                "value": float(value),
                "confidence": item.get("confidence"),
            }
        )

    series: list[dict[str, Any]] = []
    for index, (group_key, entries) in enumerate(groups.items(), start=1):
        confidences = [entry["confidence"] for entry in entries if isinstance(entry["confidence"], (int, float))]
        series.append(
            {
                "trace_id": f"codex_direct_categorical_{index:02d}",
                "series_id": f"codex_direct_categorical_{index:02d}",
                "legend_label": group_key,
                "curve_role": "category_value",
                "axis_binding": "left_y",
                "marker_shape": None,
                "confidence": (sum(confidences) / len(confidences)) if confidences else None,
                "points_data": [[entry["category"], entry["value"]] for entry in entries],
                "issues": [],
            }
        )
    return {
        "schema_version": 1,
        "panel_id": panel_id,
        "source": "codex_direct_categorical",
        "chart_type": table.get("chart_type"),
        "axis_labels": {
            "x": "category",
            "left_y": table.get("value_axis_label") or "",
            "right_y": "",
        },
        "series": series,
    }


def validation_report(curve_series: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    """Same report shape the other methods emit, so the UI badges keep working."""
    series = curve_series.get("series") or []
    empty = [item["legend_label"] for item in series if not item["points_data"]]
    issues = [str(issue) for issue in table.get("issues") or []]
    for item in series:
        issues.extend(f"{item['legend_label']}:{issue}" for issue in item["issues"])
    if empty:
        issues.append(f"SERIES_WITHOUT_POINTS:{','.join(empty)}")
    return {
        "schema_version": 1,
        "panel_id": curve_series.get("panel_id"),
        "checks": {
            "final_series_count": len(series),
            "point_count": sum(len(item["points_data"]) for item in series),
            "series_without_points": len(empty),
            "left_y_series": sum(1 for item in series if item["axis_binding"] == "left_y"),
            "right_y_series": sum(1 for item in series if item["axis_binding"] == "right_y"),
        },
        "issues": issues,
    }

"""Extraction methods that can be run side by side on the same paper.

A method bundles every choice that makes a run different: which vision prompt asks
codex for the geometry, and whether text boxes are replaced by OCR before extraction.

`baseline` is frozen. It must keep producing exactly what it produced before any of
the experiments existed, because it is the comparison point. Its chart ids carry no
suffix so its rows and result directories keep their historical identity; every other
method suffixes its chart ids so both results survive in the database at once instead
of one overwriting the other.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMethod:
    key: str
    label: str
    prompt_variant: str
    use_ocr_text: bool
    chart_id_suffix: str
    description: str
    # "plotlogic" traces curves from pixels; "codex_direct" asks codex for the values.
    extraction: str = "plotlogic"


BASELINE_METHOD = "baseline"

METHODS: dict[str, RunMethod] = {
    # ###수정 원본과 구분되는 CMSL 실험 method와 chart ID suffix를 추가합니다.
    "cmsl_minimal": RunMethod(
        key="cmsl_minimal", label="CMSL minimal experiment",
        prompt_variant="figure_panel_classifier_cmsl.md", use_ocr_text=True,
        chart_id_suffix="cmsl", description="3-bit colors, background/grid masks, greedy tracking and VLM axis association.",
        extraction="cmsl_minimal",
    ),
    # ###
    BASELINE_METHOD: RunMethod(
        key=BASELINE_METHOD,
        label="Baseline (codex only)",
        prompt_variant="figure_panel_classifier.md",
        use_ocr_text=False,
        chart_id_suffix="",
        description=(
            "Original pipeline: codex supplies every bbox, then the existing correction "
            "runs. Frozen as the comparison point."
        ),
    ),
    "ocr_v2": RunMethod(
        key="ocr_v2",
        label="OCR text + bbox-v2 prompt",
        prompt_variant="figure_panel_classifier_bbox_v2.md",
        use_ocr_text=True,
        chart_id_suffix="ocrv2",
        description=(
            "Geometry from codex using the bbox-precision prompt, with tick label, axis "
            "title and legend label boxes replaced by OCR before extraction."
        ),
    ),
    "codex_direct": RunMethod(
        key="codex_direct",
        label="Codex reads the values directly",
        prompt_variant="figure_panel_classifier.md",
        use_ocr_text=False,
        chart_id_suffix="direct",
        description=(
            "No bbox geometry and no curve tracing: codex is shown the panel crop and "
            "returns each curve as a table of values in the chart's own units."
        ),
        extraction="codex_direct",
    ),
}


def resolve_method(key: str | None) -> RunMethod:
    """Look up a method, defaulting to the baseline.

    An unknown key raises instead of silently falling back, so a typo cannot quietly
    produce baseline numbers under an experiment's name.
    """
    name = (key or BASELINE_METHOD).strip()
    if name not in METHODS:
        raise ValueError(f"unknown extraction method: {name!r} (known: {', '.join(METHODS)})")
    return METHODS[name]


def method_for_chart_id(chart_id: str) -> str:
    """Recover the method key from a chart id."""
    for method in METHODS.values():
        if method.chart_id_suffix and chart_id.endswith(f"__{method.chart_id_suffix}"):
            return method.key
    return BASELINE_METHOD


def apply_chart_id_suffix(chart_id: str, method: RunMethod) -> str:
    return f"{chart_id}__{method.chart_id_suffix}" if method.chart_id_suffix else chart_id


def available_methods() -> list[dict[str, object]]:
    return [
        {
            "key": method.key,
            "label": method.label,
            "description": method.description,
            "prompt_variant": method.prompt_variant,
            "uses_ocr_text": method.use_ocr_text,
            "is_baseline": method.key == BASELINE_METHOD,
        }
        for method in METHODS.values()
    ]

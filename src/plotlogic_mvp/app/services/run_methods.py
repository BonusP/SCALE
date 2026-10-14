"""The extraction method used for every run.

A method bundles the choices that define a run: which vision prompt asks codex for
the geometry, and whether text boxes are replaced by OCR before extraction. Its key
is recorded on each job, and its suffix is appended to every chart id.
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


DEFAULT_METHOD = RunMethod(
    key="cmsl_minimal",
    label="CMSL minimal experiment",
    prompt_variant="figure_panel_classifier_cmsl.md",
    use_ocr_text=True,
    chart_id_suffix="cmsl",
    description="3-bit colors, background/grid masks, greedy tracking and VLM axis association.",
)


def resolve_method(key: str | None) -> RunMethod:
    """Return the default method; any other key is rejected."""
    if key not in (None, DEFAULT_METHOD.key):
        raise ValueError(f"unknown extraction method: {key!r} (known: {DEFAULT_METHOD.key})")
    return DEFAULT_METHOD


def apply_chart_id_suffix(chart_id: str, method: RunMethod) -> str:
    return f"{chart_id}__{method.chart_id_suffix}"

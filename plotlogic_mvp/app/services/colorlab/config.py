"""Tunable parameters for the glyph-first colour extractor.

Every magic number in the package lives here so an experiment is one edit, and so
``params`` in the run manifest fully reproduces a result.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass(frozen=True)
class ColorLabParams:
    # -- S0 canvas -------------------------------------------------------
    ink_alpha_threshold: float = 0.35
    """Alpha above which a pixel counts as ink."""
    structural_kernel_fraction: float = 1 / 12
    """Long-run kernel length as a fraction of the crop side, for frame/tick/grid removal."""
    structural_kernel_min_px: int = 25
    structural_dilate_px: int = 1

    # -- S2 palette ------------------------------------------------------
    palette_refine_iterations: int = 6
    palette_min_separation_delta_e: float = 16.0
    """Refinement may not move a centre within this dE of another; two legend entries
    must stay two colours even when their swatches are nearly the same hue."""
    palette_max_k: int = 8
    soft_layer_delta_e: float = 12.0
    """A pixel joins every palette layer within this dE of its best match (soft membership)."""
    palette_max_match_delta_e: float = 45.0
    """Ink further than this from every palette entry belongs to no layer."""

    # -- S3 glyph detection ----------------------------------------------
    radius_search_min: int = 3
    radius_search_max: int = 14
    radius_min_stroke_factor: float = 2.5
    """Marker radius must exceed this multiple of the stroke half-width."""
    peak_score_threshold: float = 0.72
    nms_spacing_factor: float = 1.35
    """Minimum centre separation as a multiple of the marker radius."""
    dedupe_distance_factor: float = 0.80
    """Peaks from different colour layers closer than this * r are the same glyph."""
    min_glyph_ink_px: int = 8

    # -- S4 glyph typing --------------------------------------------------
    core_radius_factor: float = 0.60
    ring_inner_factor: float = 0.75
    ring_outer_factor: float = 1.10
    filled_min_fill_ratio: float = 0.75
    open_max_fill_ratio: float = 0.50
    open_min_ring_ratio: float = 0.55

    # -- S5 assembly ------------------------------------------------------
    colour_margin_delta_e: float = 6.0
    """Below this dE gap to the runner-up palette entry, a glyph is flagged ambiguous."""
    chain_max_link_factor: float = 3.5
    """Longest marker-to-marker link, as a multiple of the marker radius."""
    chain_colour_weight: float = 0.5
    colour_match_lightness_weight: float = 1.0
    """Weight on dL* when matching a chain to a legend swatch. Shaded 3-D markers vary
    far more in lightness than in hue, so a value below 1 tolerates that shading."""
    chain_direction_weight: float = 2.5
    chain_spacing_factor: float = 2.2
    """Link budget as a multiple of the observed median marker spacing."""
    chain_merge_gap_factor: float = 7.0
    """How far a fragment may reach to rejoin the chain it belongs to."""
    chain_merge_max_delta_e: float = 18.0
    min_points_per_series: int = 5
    outlier_mad_factor: float = 4.0
    merge_same_x_px: float = 1.5

    # -- line-mode fallback ------------------------------------------------
    line_mode_max_x_coverage: float = 0.30
    """Below this glyph x-coverage the panel is treated as a marker-less line chart."""
    line_curvature_penalty: float = 0.6

    # -- S8 QC gates -------------------------------------------------------
    qc_max_structural_ratio: float = 0.45
    qc_max_contested_ratio: float = 0.15
    qc_max_radius_cv: float = 0.35
    qc_min_x_coverage: float = 0.50
    qc_max_colour_ambiguous_ratio: float = 0.20

    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None) -> "ColorLabParams":
        if not overrides:
            return cls()
        known = {f for f in cls.__dataclass_fields__ if f != "extras"}
        kwargs: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in known:
                kwargs[key] = value
            else:
                extras[key] = value
        return cls(**kwargs, extras=extras)


DEFAULTS = ColorLabParams()

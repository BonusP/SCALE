"""S5 - group glyphs into series and give each series its axis and quantity.

The series key is ``(palette colour, glyph fill)``.  On a dual-y SOFC chart that key
is exactly ``(temperature, quantity)``: the filled and hollow markers of one colour are
the two curves that the legacy pipeline had to separate with mask subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canvas import Canvas
from .config import ColorLabParams
from .glyphs import AMBIGUOUS, FILLED, OPEN, Glyph, GlyphDetection
from .palette import Palette


@dataclass
class Series:
    series_id: str
    palette_index: int
    legend_label: str | None
    colour_rgb: list[int]
    glyph_type: str
    points_px: list[tuple[float, float]]
    point_flags: list[list[str]]
    glyphs: list[Glyph]
    shape: str = "unknown"
    shape_evidence: dict[str, Any] = field(default_factory=dict)
    y_axis: str = "left"
    quantity: str = "y"
    mode: str = "glyph"

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "palette_index": self.palette_index,
            "legend_label": self.legend_label,
            "colour_rgb": self.colour_rgb,
            "glyph_type": self.glyph_type,
            "mode": self.mode,
            "point_count": len(self.points_px),
            "shape": self.shape,
            "shape_evidence": self.shape_evidence,
            "y_axis": self.y_axis,
            "quantity": self.quantity,
            "x_span_px": (
                round(max(p[0] for p in self.points_px) - min(p[0] for p in self.points_px), 1)
                if self.points_px
                else 0.0
            ),
            "flagged_point_count": sum(1 for flags in self.point_flags if flags),
        }


def _classify_shape(points: list[tuple[float, float]]) -> tuple[str, dict[str, Any]]:
    """Shape is read in pixel space, where a peak is an interior *minimum* of y."""
    if len(points) < 4:
        return "unknown", {"reason": "too_few_points"}
    ys = np.array([p[1] for p in points], dtype=np.float64)
    span = float(ys.max() - ys.min())
    if span <= 0:
        return "flat", {"y_span_px": 0.0}
    index = int(ys.argmin())
    position = index / (len(ys) - 1)
    rise = float(ys[0] - ys.min())
    fall = float(ys[-1] - ys.min())
    peak_score = min(rise, fall) / span
    evidence = {
        "y_span_px": round(span, 1),
        "argmin_position": round(position, 3),
        "peak_score": round(peak_score, 3),
        "net_drop_px": round(float(ys[-1] - ys[0]), 1),
    }
    if 0.08 < position < 0.92 and peak_score > 0.15:
        return "rise_peak_fall", evidence
    if ys[-1] > ys[0]:
        return "monotonic_down", evidence
    return "monotonic_up", evidence


def _order_points(glyphs: list[Glyph], params: ColorLabParams) -> list[Glyph]:
    ordered = sorted(glyphs, key=lambda glyph: glyph.x)
    merged: list[Glyph] = []
    for glyph in ordered:
        if merged and abs(glyph.x - merged[-1].x) < params.merge_same_x_px and glyph.glyph_type == merged[-1].glyph_type:
            # Two detections of one marker: keep the stronger.
            if glyph.score > merged[-1].score:
                merged[-1] = glyph
            continue
        merged.append(glyph)
    return merged


def _flag_outliers(points: list[tuple[float, float]], params: ColorLabParams) -> list[list[str]]:
    flags: list[list[str]] = [[] for _ in points]
    if len(points) < 5:
        return flags
    ys = np.array([p[1] for p in points], dtype=np.float64)
    # residual against a local 3-point median: robust to a genuine peak, sharp on strays
    padded = np.pad(ys, 1, mode="edge")
    local = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    residual = np.abs(ys - local)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    threshold = max(2.0, params.outlier_mad_factor * (mad if mad > 0 else 1.0))
    for index, value in enumerate(residual):
        if value > threshold:
            flags[index].append("outlier")
    return flags


def _median_spacing(glyphs: list[Glyph]) -> float:
    """Median nearest-neighbour distance - how far apart this chart puts its markers."""
    if len(glyphs) < 2:
        return 0.0
    points = np.array([(glyph.x, glyph.y) for glyph in glyphs], dtype=np.float64)
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def _chain_glyphs(glyphs: list[Glyph], params: ColorLabParams) -> list[list[Glyph]]:
    """Link markers into curves by geometry, using colour only as a soft cue.

    Colour cannot be the partition key: two series whose CIELAB centres are close
    (a violet and a blue, say) get shuffled together pixel by pixel.  Direction
    continuity, on the other hand, is unambiguous - a curve does not double back on
    itself at a crossing.  Once a chain exists, its colour is settled by voting over
    every glyph on it, which is a far stronger signal than any single glyph.
    """
    if not glyphs:
        return []
    ordered = sorted(glyphs, key=lambda glyph: (glyph.x, glyph.y))
    radius = float(np.median([glyph.radius for glyph in ordered]))
    # Marker spacing is set by how densely the authors sampled, not by marker size, so
    # the link budget follows the observed spacing. A fixed multiple of the radius
    # silently shatters every sparsely sampled curve into singletons.
    max_link = max(params.chain_max_link_factor * radius, params.chain_spacing_factor * _median_spacing(ordered))
    used = [False] * len(ordered)
    chains: list[list[Glyph]] = []

    def cost(a: Glyph, b: Glyph, direction: tuple[float, float] | None) -> float | None:
        dx, dy = b.x - a.x, b.y - a.y
        distance = float(np.hypot(dx, dy))
        # Near-vertical stretches put consecutive markers in the same column, so the
        # forward test has to tolerate a pixel of backward drift.
        if dx < -1.0 or distance > max_link:
            return None
        colour = float(np.linalg.norm(np.subtract(a.lab, b.lab)))
        value = distance / radius + params.chain_colour_weight * colour / 10.0
        if direction is not None:
            norm = distance * float(np.hypot(*direction)) or 1.0
            cosine = (dx * direction[0] + dy * direction[1]) / norm
            value += params.chain_direction_weight * (1.0 - cosine)
        return value

    for start in range(len(ordered)):
        if used[start]:
            continue
        current = start
        used[current] = True
        chain = [ordered[current]]
        direction: tuple[float, float] | None = None
        while True:
            best_index, best_cost = None, float("inf")
            for index in range(len(ordered)):
                if used[index]:
                    continue
                value = cost(ordered[current], ordered[index], direction)
                if value is not None and value < best_cost:
                    best_index, best_cost = index, value
            if best_index is None:
                break
            previous = ordered[current]
            nxt = ordered[best_index]
            direction = (nxt.x - previous.x, nxt.y - previous.y)
            used[best_index] = True
            chain.append(nxt)
            current = best_index
        chains.append(chain)
    return _merge_chain_fragments(chains, radius, params)


def _merge_chain_fragments(
    chains: list[list[Glyph]],
    radius: float,
    params: ColorLabParams,
) -> list[list[Glyph]]:
    """Join fragments that are one curve: A ends where B starts, going the same way in
    the same colour.  Detection gaps in a dense marker run split chains otherwise."""
    chains = [chain for chain in chains if chain]
    chains.sort(key=lambda chain: chain[0].x)
    merged = True
    while merged:
        merged = False
        for i, left in enumerate(chains):
            if not left:
                continue
            tail = left[-1]
            tail_direction = (
                (tail.x - left[-2].x, tail.y - left[-2].y) if len(left) > 1 else None
            )
            best_index, best_cost = None, float("inf")
            for j, right in enumerate(chains):
                if i == j or not right:
                    continue
                head = right[0]
                dx, dy = head.x - tail.x, head.y - tail.y
                distance = float(np.hypot(dx, dy))
                if dx < -1.0 or distance > params.chain_merge_gap_factor * radius:
                    continue
                colour = float(np.linalg.norm(np.subtract(tail.lab, head.lab)))
                if colour > params.chain_merge_max_delta_e:
                    continue
                value = distance / radius + params.chain_colour_weight * colour / 10.0
                if tail_direction is not None:
                    norm = distance * float(np.hypot(*tail_direction)) or 1.0
                    cosine = (dx * tail_direction[0] + dy * tail_direction[1]) / norm
                    if cosine < 0.2:
                        continue
                    value += params.chain_direction_weight * (1.0 - cosine)
                if value < best_cost:
                    best_index, best_cost = j, value
            if best_index is not None:
                chains[i] = left + chains[best_index]
                chains[best_index] = []
                merged = True
                break
    return [chain for chain in chains if chain]


def _chain_colour_costs(chains: list[list[Glyph]], palette: Palette, params: ColorLabParams) -> np.ndarray:
    """(chains x palette) CIELAB distance from each chain's median ink to each entry."""
    weights = np.array([params.colour_match_lightness_weight, 1.0, 1.0], dtype=np.float32)
    medians = np.array(
        [np.median(np.array([glyph.lab for glyph in chain], dtype=np.float32), axis=0) for chain in chains],
        dtype=np.float32,
    )
    return np.linalg.norm((medians[:, None, :] - palette.match_centers_lab[None]) * weights, axis=2)


def _assign_chains_to_palette(
    chains: list[list[Glyph]],
    palette: Palette,
    params: ColorLabParams,
) -> list[tuple[int, float, float]]:
    """Label all chains of one marker type at once, with a capacity per legend entry.

    Independent nearest-colour votes collapse: whichever legend swatch sits nearest the
    muddy average of a thin outline attracts every uncertain chain, and one entry ends up
    owning most of the plot.  A balanced assignment spreads the chains the way a legend
    actually does - each entry draws roughly the same number of curves.
    """
    if not chains:
        return []
    costs = _chain_colour_costs(chains, palette, params)
    capacity = max(1, int(np.ceil(len(chains) / max(1, palette.size))))
    slots = np.repeat(np.arange(palette.size), capacity)
    assignment: dict[int, int] = {}
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(costs[:, slots])
        assignment = {int(row): int(slots[column]) for row, column in zip(rows, columns)}
    except Exception:
        assignment = {}

    out: list[tuple[int, float, float]] = []
    for index in range(len(chains)):
        # Chains beyond the slot budget keep their unconstrained best match.
        palette_index = assignment.get(index, int(costs[index].argmin()))
        order = np.argsort(costs[index])
        margin = float(costs[index][order[1]] - costs[index][order[0]]) if palette.size > 1 else 999.0
        out.append((palette_index, float(costs[index][palette_index]), margin))
    return out


def _assign_axes(series_list: list[Series], is_dual_y: bool, left_role: str, right_role: str | None) -> None:
    if not is_dual_y or not right_role:
        for series in series_list:
            series.y_axis = "left"
            series.quantity = left_role
        return

    by_colour: dict[int, list[Series]] = {}
    for series in series_list:
        by_colour.setdefault(series.palette_index, []).append(series)

    for members in by_colour.values():
        peaked = [s for s in members if s.shape == "rise_peak_fall"]
        if len(members) == 2 and len(peaked) != 1:
            # Both look alike; the stronger interior peak is the right-axis curve.
            peaked = [max(members, key=lambda s: s.shape_evidence.get("peak_score", 0.0))]
        for series in members:
            if series in peaked:
                series.y_axis = "right"
                series.quantity = right_role
            else:
                series.y_axis = "left"
                series.quantity = left_role


def assemble_series(
    detection: GlyphDetection,
    palette: Palette,
    params: ColorLabParams,
    *,
    is_dual_y: bool,
    left_role: str,
    right_role: str | None,
) -> list[Series]:
    # Ambiguous glyphs chain with whichever type they physically continue.
    by_type: dict[str, list[Glyph]] = {FILLED: [], OPEN: []}
    for glyph in detection.glyphs:
        if glyph.glyph_type in by_type:
            by_type[glyph.glyph_type].append(glyph)
    for glyph in detection.glyphs:
        if glyph.glyph_type != AMBIGUOUS:
            continue
        nearest = min(
            ((kind, members) for kind, members in by_type.items() if members),
            key=lambda item: min((glyph.x - m.x) ** 2 + (glyph.y - m.y) ** 2 for m in item[1]),
            default=None,
        )
        if nearest is not None:
            glyph.flags.append(f"chained_as_{nearest[0]}")
            nearest[1].append(glyph)

    series_list: list[Series] = []
    for glyph_type in (FILLED, OPEN):
        chains = [
            ordered
            for ordered in (
                _order_points(chain, params) for chain in _chain_glyphs(by_type[glyph_type], params)
            )
            if len(ordered) >= params.min_points_per_series
        ]
        for ordered, (palette_index, delta_e, margin) in zip(
            chains, _assign_chains_to_palette(chains, palette, params)
        ):
            points = [(glyph.x, glyph.y) for glyph in ordered]
            shape, evidence = _classify_shape(points)
            flags = _flag_outliers(points, params)
            for index, glyph in enumerate(ordered):
                flags[index] = list(dict.fromkeys(flags[index] + glyph.flags))
            evidence["chain_colour_delta_e"] = round(delta_e, 2)
            evidence["chain_colour_margin_delta_e"] = round(margin, 2)
            series_list.append(
                Series(
                    series_id=f"s{len(series_list) + 1:02d}",
                    palette_index=palette_index,
                    legend_label=palette.labels[palette_index],
                    colour_rgb=[int(v) for v in palette.display_rgb[palette_index]],
                    glyph_type=glyph_type,
                    points_px=points,
                    point_flags=flags,
                    glyphs=ordered,
                    shape=shape,
                    shape_evidence=evidence,
                )
            )

    series_list.sort(key=lambda series: (series.palette_index, series.glyph_type))
    for index, series in enumerate(series_list, start=1):
        series.series_id = f"s{index:02d}"
    _assign_axes(series_list, is_dual_y, left_role, right_role)
    return series_list


# ---------------------------------------------------------------------------
# line-mode fallback: charts drawn without markers
# ---------------------------------------------------------------------------


def _column_clusters(layer: np.ndarray, x: int, max_gap: int = 3) -> list[tuple[float, int]]:
    ys = np.nonzero(layer[:, x])[0]
    if ys.size == 0:
        return []
    clusters: list[list[int]] = [[int(ys[0])]]
    for y in ys[1:]:
        if y - clusters[-1][-1] <= max_gap:
            clusters[-1].append(int(y))
        else:
            clusters.append([int(y)])
    return [(float(np.mean(cluster)), len(cluster)) for cluster in clusters]


def trace_lines(
    canvas: Canvas,
    palette: Palette,
    params: ColorLabParams,
    *,
    is_dual_y: bool,
    left_role: str,
    right_role: str | None,
) -> list[Series]:
    """Velocity-predicting tracker: continuation follows the curve's direction, not the
    nearest y, which is what makes traces jump series at a crossing."""
    series_list: list[Series] = []
    for palette_index in range(palette.size):
        layer = palette.layers[palette_index] & canvas.curve_ink
        if layer.sum() < 50:
            continue
        tracks: list[dict[str, Any]] = []
        for x in range(canvas.width):
            clusters = _column_clusters(layer, x)
            if not clusters:
                continue
            used: set[int] = set()
            for y, _weight in clusters:
                best_index, best_cost = None, float("inf")
                for index, track in enumerate(tracks):
                    if index in used:
                        continue
                    gap = x - track["last_x"]
                    if gap <= 0 or gap > 12:
                        continue
                    predicted = track["last_y"] + track["velocity"] * gap
                    cost = abs(y - predicted) + params.line_curvature_penalty * abs(
                        (y - track["last_y"]) / gap - track["velocity"]
                    )
                    if cost < best_cost and abs(y - predicted) <= 6 + 1.5 * gap:
                        best_index, best_cost = index, cost
                if best_index is None:
                    tracks.append({"points": [(float(x), y)], "last_x": x, "last_y": y, "velocity": 0.0})
                    used.add(len(tracks) - 1)
                    continue
                track = tracks[best_index]
                gap = x - track["last_x"]
                velocity = (y - track["last_y"]) / gap
                track["velocity"] = 0.6 * track["velocity"] + 0.4 * velocity if track["points"] else velocity
                track["points"].append((float(x), y))
                track["last_x"], track["last_y"] = x, y
                used.add(best_index)

        for track in tracks:
            points = track["points"]
            if len(points) < max(params.min_points_per_series, int(canvas.width * 0.08)):
                continue
            shape, evidence = _classify_shape(points)
            series_list.append(
                Series(
                    series_id=f"l{len(series_list) + 1:02d}",
                    palette_index=palette_index,
                    legend_label=palette.labels[palette_index],
                    colour_rgb=[int(v) for v in palette.display_rgb[palette_index]],
                    glyph_type="line",
                    points_px=points,
                    point_flags=[[] for _ in points],
                    glyphs=[],
                    shape=shape,
                    shape_evidence=evidence,
                    mode="line",
                )
            )
    _assign_axes(series_list, is_dual_y, left_role, right_role)
    return series_list

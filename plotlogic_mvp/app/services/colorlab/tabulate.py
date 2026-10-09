"""S7 - write the tables: tidy points, a wide grid, and the machine-readable bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .axes import AxisFrame
from .assemble import Series

POINTS_COLUMNS = [
    "panel_id",
    "series_id",
    "series_label",
    "palette_rgb",
    "glyph_type",
    "y_axis_role",
    "quantity",
    "point_index",
    "x_px",
    "y_px",
    "x_value",
    "y_value",
    "glyph_radius_px",
    "fill_ratio",
    "colour_margin_dE",
    "flags",
]


def _round(value: float | None, digits: int = 6) -> float | str:
    if value is None or not np.isfinite(value):
        return ""
    return round(float(value), digits)


def build_point_rows(panel_id: str, series_list: list[Series], frame: AxisFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series in series_list:
        for index, (x_px, y_px) in enumerate(series.points_px):
            glyph = series.glyphs[index] if index < len(series.glyphs) else None
            rows.append(
                {
                    "panel_id": panel_id,
                    "series_id": series.series_id,
                    "series_label": series.legend_label or series.series_id,
                    "palette_rgb": "#%02x%02x%02x" % tuple(series.colour_rgb),
                    "glyph_type": series.glyph_type,
                    "y_axis_role": series.y_axis,
                    "quantity": series.quantity,
                    "point_index": index,
                    "x_px": round(x_px, 2),
                    "y_px": round(y_px, 2),
                    "x_value": _round(frame.x_value(x_px)),
                    "y_value": _round(frame.y_value(y_px, series.y_axis)),
                    "glyph_radius_px": round(glyph.radius, 2) if glyph else "",
                    "fill_ratio": round(glyph.fill_ratio, 3) if glyph else "",
                    "colour_margin_dE": round(glyph.colour_margin_delta_e, 2) if glyph else "",
                    "flags": "|".join(series.point_flags[index]),
                }
            )
    return rows


def write_points_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _pchip_interpolate(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson): no overshoot between measured points."""
    n = len(xs)
    out = np.full(len(grid), np.nan)
    if n < 2:
        return out
    h = np.diff(xs)
    delta = np.diff(ys) / h
    tangents = np.zeros(n)
    tangents[0] = delta[0]
    tangents[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            tangents[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            tangents[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    inside = (grid >= xs[0]) & (grid <= xs[-1])
    idx = np.clip(np.searchsorted(xs, grid[inside], side="right") - 1, 0, n - 2)
    t = (grid[inside] - xs[idx]) / h[idx]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    out[inside] = (
        h00 * ys[idx] + h10 * h[idx] * tangents[idx] + h01 * ys[idx + 1] + h11 * h[idx] * tangents[idx + 1]
    )
    return out


def write_wide_csv(
    path: Path,
    rows: list[dict[str, Any]],
    series_list: list[Series],
    x_label: str = "x_value",
) -> dict[str, Any]:
    """Resample every series onto the union x-grid. No extrapolation: outside a series'
    own range the cell is left blank."""
    by_series: dict[str, list[tuple[float, float]]] = {}
    header: dict[str, str] = {}
    for row in rows:
        if row["x_value"] == "" or row["y_value"] == "":
            continue
        by_series.setdefault(row["series_id"], []).append((float(row["x_value"]), float(row["y_value"])))
        # Fragmented curves can share a label and quantity, so the id stays in the header
        # to keep the columns distinguishable.
        header[row["series_id"]] = f"{row['series_label']} · {row['quantity']} [{row['series_id']}]"
    if not by_series:
        path.write_text("x_value\n", encoding="utf-8")
        return {"grid_points": 0, "series": 0, "issues": ["NO_CALIBRATED_POINTS"]}

    grid = np.unique(np.concatenate([np.array([p[0] for p in pts]) for pts in by_series.values()]))
    columns: dict[str, np.ndarray] = {}
    for series_id, points in by_series.items():
        points = sorted({round(x, 9): y for x, y in points}.items())
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        columns[series_id] = _pchip_interpolate(xs, ys, grid)

    order = [s.series_id for s in series_list if s.series_id in columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_label] + [header[series_id] for series_id in order])
        for index, x in enumerate(grid):
            line = [round(float(x), 6)]
            for series_id in order:
                value = columns[series_id][index]
                line.append("" if not np.isfinite(value) else round(float(value), 6))
            writer.writerow(line)
    return {"grid_points": int(len(grid)), "series": len(order), "issues": []}


def write_series_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

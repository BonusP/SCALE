"""End-to-end checks for the glyph-first extractor on a synthetic chart.

The fixture is drawn here rather than loaded, so the expected answer is known exactly:
two colours, each with a filled marker curve and a hollow marker curve.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.services.colorlab import ColorLabParams, run_colorlab  # noqa: E402
from app.services.colorlab.canvas import build_canvas  # noqa: E402

WIDTH, HEIGHT = 400, 300
OFFSET = (50, 40)
RED = (220, 30, 30)
BLUE = (30, 60, 220)
RADIUS = 6


def _draw_chart(path: Path) -> None:
    image = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    # axis frame + ticks: long runs that S0 must strip
    cv2.rectangle(image, (0, 0), (WIDTH - 1, HEIGHT - 1), (0, 0, 0), 2)
    for x in range(0, WIDTH, 50):
        cv2.line(image, (x, HEIGHT - 1), (x, HEIGHT - 12), (0, 0, 0), 1)

    for colour, base in ((RED, 40), (BLUE, 110)):
        bgr = colour[::-1]
        for step in range(14):
            x = 30 + step * 25
            # filled markers: a peak
            y_peak = int(base + 90 - 70 * np.sin(np.pi * step / 13))
            cv2.circle(image, (x, y_peak), RADIUS, bgr, -1)
            # hollow markers: a monotonic fall, deliberately crossing the peak curve
            y_fall = int(base + 20 + step * 9)
            cv2.circle(image, (x, y_fall), RADIUS, bgr, 2)
    cv2.imwrite(str(path), image)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    _draw_chart(tmp_path / "plot_crop.png")
    (tmp_path / "crop_meta.json").write_text(
        json.dumps(
            {
                "panel_id": "synthetic_panel",
                "crop_offset_xy": list(OFFSET),
                "crop_size_wh": [WIDTH, HEIGHT],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "exclusion_regions.json").write_text(json.dumps({"bboxes": []}), encoding="utf-8")
    (tmp_path / "chart_extraction_strategy.json").write_text(
        json.dumps(
            {
                "x_axis_role": "current_density",
                "left_y_role": "voltage",
                "right_y_role": "power_density",
            }
        ),
        encoding="utf-8",
    )
    # bottom: 1 px == 1 unit; left/right: linear in y
    (tmp_path / "axis_mapping.json").write_text(
        json.dumps(
            {
                "axes": {
                    "bottom": {
                        "axis_label": "x",
                        "pairs": [
                            {"pixel": OFFSET[0], "value": 0.0},
                            {"pixel": OFFSET[0] + 200, "value": 200.0},
                        ],
                    },
                    "left": {
                        "axis_label": "voltage",
                        "pairs": [
                            {"pixel": OFFSET[1], "value": 1.0},
                            {"pixel": OFFSET[1] + HEIGHT, "value": 0.0},
                        ],
                    },
                    "right": {
                        "axis_label": "power",
                        "pairs": [
                            {"pixel": OFFSET[1], "value": 100.0},
                            {"pixel": OFFSET[1] + HEIGHT, "value": 0.0},
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "mvp_extraction_inputs.json").write_text(
        json.dumps(
            {
                "panel_vision": {
                    "legend_items": [
                        {"legend_label": "red", "swatch_color_rgb": list(RED)},
                        {"legend_label": "blue", "swatch_color_rgb": list(BLUE)},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_structural_ink_removal_keeps_curves(run_dir: Path) -> None:
    canvas = build_canvas(run_dir / "plot_crop.png", [], ColorLabParams())
    assert canvas.structural.any(), "axis frame should be detected as structural ink"
    assert canvas.curve_ink.sum() > 0.4 * canvas.ink.sum(), "curves must survive frame removal"
    # the frame sits on the border; curve ink must not
    assert not canvas.curve_ink[0, :].any()


def test_run_produces_tables_and_report(run_dir: Path) -> None:
    summary = run_colorlab(run_dir)
    out = run_dir / "colorlab"

    assert summary["mode"] == "glyph"
    for name in ("points_long.csv", "series_wide.csv", "colorlab_summary.json", "comparison_report.html"):
        assert (out / name).exists(), name

    # nothing may be written outside colorlab/
    written = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert written == {
        "plot_crop.png",
        "crop_meta.json",
        "exclusion_regions.json",
        "chart_extraction_strategy.json",
        "axis_mapping.json",
        "mvp_extraction_inputs.json",
    }


def test_filled_and_hollow_markers_become_separate_series(run_dir: Path) -> None:
    summary = run_colorlab(run_dir, write_report=False)
    kinds = {(s["legend_label"], s["glyph_type"]) for s in summary["series"] if s["point_count"] >= 5}
    assert ("red", "filled") in kinds
    assert ("red", "open") in kinds
    assert ("blue", "filled") in kinds
    assert ("blue", "open") in kinds


def test_peaked_series_binds_to_the_right_axis(run_dir: Path) -> None:
    summary = run_colorlab(run_dir, write_report=False)
    peaked = [s for s in summary["series"] if s["shape"] == "rise_peak_fall"]
    assert peaked, "the sine-shaped marker curves should be recognised as peaked"
    assert all(s["y_axis"] == "right" and s["quantity"] == "power_density" for s in peaked)


def test_points_csv_carries_calibrated_values(run_dir: Path) -> None:
    run_colorlab(run_dir, write_report=False)
    with (run_dir / "colorlab" / "points_long.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    values = [(float(row["x_px"]), float(row["x_value"])) for row in rows if row["x_value"]]
    assert values, "axis mapping should have produced data values"
    # bottom axis is 1 unit per pixel with the crop offset removed
    # x_px is stored rounded to 2 decimals, so compare within that quantisation
    assert all(abs(x_value - x_px) < 0.01 for x_px, x_value in values)

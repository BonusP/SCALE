import pytest

from labeling_ui.lib.ids import (
    extract_figure_token,
    extract_panel_id_from_stem,
    panel_base_id,
)


@pytest.mark.parametrize(
    "panel_id,expected",
    [
        ("0001_F01-P01_chart", "0001_F01-P01"),
        ("0001_F01-P01_image", "0001_F01-P01"),
        ("0001_F01-P01_schematic", "0001_F01-P01"),
        ("0001_F01-P01_table", "0001_F01-P01"),
        ("0001_F01-P01_unknown", "0001_F01-P01"),
    ],
)
def test_panel_base_id_positive_cases(panel_id, expected):
    assert panel_base_id(panel_id) == expected


@pytest.mark.parametrize(
    "panel_id",
    [
        "0001_F01-P01",
        "0001_F01-P01_heatmap",
        "0001_F01-p01",
        "0001_F01-P01_chart_extra",
    ],
)
def test_panel_base_id_negative_cases(panel_id):
    assert panel_base_id(panel_id) == panel_id


@pytest.mark.parametrize(
    "filename,require_double_underscore,expected",
    [
        ("0001_F01__src.png", False, "01"),
        ("0001_F01__src.png", True, "01"),
        ("0001_F02.png", False, "02.png"),
        ("0001_F02.png", True, None),
        ("0001_no_figure.png", False, None),
    ],
)
def test_extract_figure_token(filename, require_double_underscore, expected):
    assert extract_figure_token(filename, require_double_underscore) == expected


@pytest.mark.parametrize(
    "panel_stem,paper_id,figure_id,expected",
    [
        ("0001_F01-P01_chart", "0001", "F01", "P01"),
        ("0001_F01-P01_chart_extra", "0001", "F01", "P01"),
        ("0001_F01-P01", "0001", "F01", "P01"),
        ("unmatched_panel", "0001", "F01", "unmatched_panel"),
    ],
)
def test_extract_panel_id_from_stem(panel_stem, paper_id, figure_id, expected):
    assert extract_panel_id_from_stem(panel_stem, paper_id, figure_id) == expected

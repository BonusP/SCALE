import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import labeling_ui.app as app_module
from labeling_ui.lib.storage import _panel_base_id


REQUIRED_KEYS = {
    "hierarchy_figure": {
        "paper_id": str,
        "figure_id": str,
        "figure_rel_path": str,
        "panels": list,
    },
    "hierarchy_panel": {
        "panel_id": str,
        "panel_rel_path": str,
    },
    "panels_item": {
        "panel_id": str,
        "panel_rel_path": str,
        "panel_type": str,
        "label": str,
    },
    "components_item": {
        "component_id": str,
        "component_type": str,
        "label": str,
    },
    "component_counts": {
        "paper_id": str,
        "counts": dict,
    },
}


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_png(path, size=(24, 24), color="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _assert_required_shape(item, required):
    assert required.keys() <= item.keys()
    for key, expected_type in required.items():
        assert isinstance(item[key], expected_type)


@pytest.fixture
def seeded_dataset(tmp_path, monkeypatch):
    paper_id = "0001"
    figure_rel = f"figures/{paper_id}/{paper_id}_F01__src.png"
    panel_rel = f"panels/{paper_id}/{paper_id}_F01-P01_chart.png"
    component_id = f"{paper_id}_F01-P01_chart-C01"

    _write_png(tmp_path / figure_rel, size=(64, 64), color="white")
    _write_png(tmp_path / panel_rel, size=(24, 24), color="black")
    _write_png(
        tmp_path / f"components/{paper_id}/crops/{component_id}_plot_frame.png",
        size=(12, 12),
        color="red",
    )

    _write_json(
        tmp_path / "labels" / "panels" / f"{paper_id}.json",
        {
            "panels": [
                {
                    "panel_id": f"{paper_id}_F01-P01",
                    "figure_id": f"{paper_id}_F01",
                    "panel_bbox_xyxy": [1, 1, 24, 24],
                    "panel_path": panel_rel,
                    "panel_type": "chart",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "labels" / "components" / f"{paper_id}.json",
        {
            "components": [
                {
                    "component_id": component_id,
                    "panel_id": f"{paper_id}_F01-P01_chart",
                    "component_type": "plot_frame",
                    "bbox_xyxy": [2, 2, 12, 12],
                    "panel_path": panel_rel,
                    "crop_path": f"components/{paper_id}/crops/{component_id}_plot_frame.png",
                    "overlay_path": f"components/{paper_id}/overlays/{paper_id}_F01-P01_chart__overlay.png",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "links" / "panel_index.json",
        {
            "panels": [
                {
                    "panel_id": f"{paper_id}_F01-P01",
                    "paper_id": paper_id,
                    "figure_id": f"{paper_id}_F01",
                    "panel_path": panel_rel,
                    "candidate_type": "chart",
                    "final_type": None,
                }
            ]
        },
    )
    _write_json(
        tmp_path / "links" / "component_index.json",
        {
            component_id: {
                "panel_id": f"{paper_id}_F01-P01_chart",
                "component_type": "plot_frame",
                "component_class": "chart",
                "bbox_xyxy": [2, 2, 12, 12],
                "panel_path": panel_rel,
                "crop_path": f"components/{paper_id}/crops/{component_id}_plot_frame.png",
                "overlay_path": f"components/{paper_id}/overlays/{paper_id}_F01-P01_chart__overlay.png",
            }
        },
    )

    monkeypatch.setattr(app_module, "DATASET_ROOT", tmp_path)
    app_module._PANEL_INDEX_CACHE = {"mtime": None, "by_figure": {}}
    app_module._PANEL_LABELS_CACHE = {}
    app_module._COMPONENT_INDEX_CACHE = {"mtime": None, "data": {}}
    app_module._COMPONENT_COUNTS_CACHE = {}
    return {
        "paper_id": paper_id,
        "figure_rel": figure_rel,
        "panel_rel": panel_rel,
    }


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
def test_panel_base_id_normalization_positive_cases(panel_id, expected):
    assert app_module.panel_base_id(panel_id) == expected
    assert _panel_base_id(panel_id) == expected


@pytest.mark.parametrize(
    "panel_id",
    [
        "0001_F01-P01",
        "0001_F01-P01_CHART",
        "0001_F01-P01chart",
        "0001_F01-P01_",
        "random-id",
        "",
    ],
)
def test_panel_base_id_normalization_negative_cases(panel_id):
    assert app_module.panel_base_id(panel_id) == panel_id
    assert _panel_base_id(panel_id) == panel_id


def test_hierarchy_contract_freeze(seeded_dataset):
    client = TestClient(app_module.app)

    response = client.get("/api/hierarchy")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload

    figure = next(
        item for item in payload if item["paper_id"] == seeded_dataset["paper_id"]
    )
    _assert_required_shape(figure, REQUIRED_KEYS["hierarchy_figure"])
    assert isinstance(figure["panels"], list)
    assert figure["panels"]

    panel = figure["panels"][0]
    _assert_required_shape(panel, REQUIRED_KEYS["hierarchy_panel"])


def test_panels_contract_freeze(seeded_dataset):
    client = TestClient(app_module.app)

    response = client.get(
        "/api/panels",
        params={"figure_rel": seeded_dataset["figure_rel"], "include_bbox": "1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload

    item = payload[0]
    _assert_required_shape(item, REQUIRED_KEYS["panels_item"])


def test_components_contract_freeze(seeded_dataset):
    client = TestClient(app_module.app)

    response = client.get(
        "/api/components",
        params={"panel_rel": seeded_dataset["panel_rel"], "include_bbox": "1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload

    item = payload[0]
    _assert_required_shape(item, REQUIRED_KEYS["components_item"])


def test_component_counts_contract_freeze(seeded_dataset):
    client = TestClient(app_module.app)

    response = client.get(
        "/api/component_counts",
        params={"paper_id": seeded_dataset["paper_id"]},
    )
    assert response.status_code == 200
    payload = response.json()

    _assert_required_shape(payload, REQUIRED_KEYS["component_counts"])
    assert isinstance(payload["counts"], dict)
    assert all(isinstance(key, str) for key in payload["counts"])
    assert all(isinstance(value, int) for value in payload["counts"].values())


@pytest.mark.parametrize(
    "rel,expected_status",
    [
        ("../x", 400),
        ("..\\x", 404),
        ("/etc/passwd", 400),
        ("C:\\Windows\\win.ini", 404),
        ("..%2Fx", 400),
        ("%2e%2e%2Fx", 400),
        ("%2e%2e%5Cx", 404),
    ],
)
def test_api_file_path_payload_matrix(rel, expected_status):
    client = TestClient(app_module.app)

    response = client.get(f"/api/file?rel={rel}")
    assert response.status_code == expected_status


def test_api_file_status_matrix_for_valid_and_missing(seeded_dataset):
    client = TestClient(app_module.app)

    valid = client.get("/api/file", params={"rel": seeded_dataset["figure_rel"]})
    assert valid.status_code == 200

    missing = client.get(
        "/api/file",
        params={"rel": f"figures/{seeded_dataset['paper_id']}/does_not_exist.png"},
    )
    assert missing.status_code == 404

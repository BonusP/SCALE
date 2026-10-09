import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import labeling_ui.app as app_module


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_png(path, size=(64, 64), color="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


@pytest.fixture
def seeded_dataset(tmp_path, monkeypatch):
    paper_id = "0001"
    figure_rel = f"figures/{paper_id}/{paper_id}_F01__src.png"
    panel_rel = f"panels/{paper_id}/{paper_id}_F01-P01_chart.png"
    panel_id = f"{paper_id}_F01-P01_chart"
    component_id = f"{panel_id}-C01"

    _write_png(tmp_path / figure_rel)
    _write_png(tmp_path / panel_rel, size=(32, 32), color="black")
    _write_png(
        tmp_path / f"components/{paper_id}/crops/{component_id}_plot_frame.png",
        size=(16, 16),
        color="red",
    )

    _write_json(
        tmp_path / "labels" / "panels" / f"{paper_id}.json",
        {
            "panels": [
                {
                    "panel_id": f"{paper_id}_F01-P01",
                    "figure_id": f"{paper_id}_F01",
                    "panel_bbox_xyxy": [1, 1, 12, 12],
                    "panel_path": panel_rel,
                    "panel_type": "chart",
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
        tmp_path / "labels" / "components" / f"{paper_id}.json",
        {
            "components": [
                {
                    "component_id": component_id,
                    "panel_id": panel_id,
                    "component_type": "plot_frame",
                    "component_class": "chart",
                    "bbox_xyxy": [1, 1, 10, 10],
                    "panel_path": panel_rel,
                    "crop_path": f"components/{paper_id}/crops/{component_id}_plot_frame.png",
                    "overlay_path": f"components/{paper_id}/overlays/{panel_id}__overlay.png",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "links" / "component_index.json",
        {
            component_id: {
                "panel_id": panel_id,
                "component_type": "plot_frame",
                "component_class": "chart",
                "bbox_xyxy": [1, 1, 10, 10],
                "panel_path": panel_rel,
                "crop_path": f"components/{paper_id}/crops/{component_id}_plot_frame.png",
                "overlay_path": f"components/{paper_id}/overlays/{panel_id}__overlay.png",
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


def test_cache_invalidated_after_save_panels(seeded_dataset):
    client = TestClient(app_module.app)

    before = client.get(
        "/api/panels",
        params={"figure_rel": seeded_dataset["figure_rel"], "include_bbox": "1"},
    )
    assert before.status_code == 200
    assert before.json()[0]["bbox_xyxy"] == [1, 1, 12, 12]

    saved = client.post(
        "/api/panels/save",
        json={
            "figure_rel_path": seeded_dataset["figure_rel"],
            "panel_boxes": [
                {"bbox_xyxy": [8, 8, 30, 30], "panel_type": "image"},
            ],
        },
    )
    assert saved.status_code == 200

    after = client.get(
        "/api/panels",
        params={"figure_rel": seeded_dataset["figure_rel"], "include_bbox": "1"},
    )
    assert after.status_code == 200
    payload = after.json()
    assert len(payload) == 1
    assert payload[0]["bbox_xyxy"] == [8, 8, 30, 30]
    assert payload[0]["panel_type"] == "image"


def test_cache_invalidated_after_save_components(seeded_dataset):
    client = TestClient(app_module.app)

    before = client.get(
        "/api/components",
        params={"panel_rel": seeded_dataset["panel_rel"], "include_bbox": "1"},
    )
    assert before.status_code == 200
    assert before.json()[0]["bbox_xyxy"] == [1, 1, 10, 10]

    saved = client.post(
        "/api/components/save",
        json={
            "panel_rel_path": seeded_dataset["panel_rel"],
            "components": [
                {
                    "bbox_xyxy": [4, 4, 14, 14],
                    "component_type": "legend",
                    "component_class": "chart",
                }
            ],
        },
    )
    assert saved.status_code == 200

    after = client.get(
        "/api/components",
        params={"panel_rel": seeded_dataset["panel_rel"], "include_bbox": "1"},
    )
    assert after.status_code == 200
    payload = after.json()
    assert len(payload) == 1
    assert payload[0]["bbox_xyxy"] == [4, 4, 14, 14]
    assert payload[0]["component_type"] == "legend"


def test_component_counts_reflects_updates(seeded_dataset):
    client = TestClient(app_module.app)

    before = client.get(
        "/api/component_counts",
        params={"paper_id": seeded_dataset["paper_id"]},
    )
    assert before.status_code == 200
    assert before.json()["counts"] == {"0001_F01-P01_chart": 1}

    saved = client.post(
        "/api/components/save",
        json={
            "panel_rel_path": seeded_dataset["panel_rel"],
            "components": [
                {
                    "bbox_xyxy": [2, 2, 12, 12],
                    "component_type": "legend",
                    "component_class": "chart",
                },
                {
                    "bbox_xyxy": [14, 14, 24, 24],
                    "component_type": "axis",
                    "component_class": "chart",
                },
            ],
        },
    )
    assert saved.status_code == 200

    after = client.get(
        "/api/component_counts",
        params={"paper_id": seeded_dataset["paper_id"]},
    )
    assert after.status_code == 200
    assert after.json()["counts"] == {"0001_F01-P01_chart": 2}

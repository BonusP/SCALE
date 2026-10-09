import json
from pathlib import Path

from labeling_ui.lib.storage import (
    save_panel_labels,
    update_component_index,
    update_panel_index,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_update_panel_index_replaces_typed_panel_id_for_same_slot(tmp_path: Path):
    dataset_root = tmp_path
    links_dir = dataset_root / "links"
    links_dir.mkdir(parents=True, exist_ok=True)
    panel_index_path = links_dir / "panel_index.json"
    panel_index_path.write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "0001_F01-P01_chart",
                        "paper_id": "0001",
                        "figure_id": "0001_F01",
                        "panel_path": "panels/0001/0001_F01-P01_chart.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    update_panel_index(
        dataset_root,
        [
            {
                "panel_id": "0001_F01-P01",
                "figure_id": "0001_F01",
                "panel_path": "panels/0001/0001_F01-P01_chart.png",
                "panel_type": "chart",
            }
        ],
        "0001",
    )

    data = _read_json(panel_index_path)
    panels = data["panels"]
    assert len(panels) == 1
    assert panels[0]["panel_id"] == "0001_F01-P01"


def test_save_panel_labels_replaces_typed_panel_id_for_same_slot(tmp_path: Path):
    labels_dir = tmp_path / "labels" / "panels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / "0001.json"
    labels_path.write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "panel_id": "0001_F01-P01_chart",
                        "panel_bbox_xyxy": [1, 1, 10, 10],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    save_panel_labels(
        str(labels_dir),
        "0001",
        [
            {
                "panel_id": "0001_F01-P01",
                "panel_bbox_xyxy": [2, 2, 20, 20],
            }
        ],
    )

    data = _read_json(labels_path)
    panels = data["panels"]
    assert len(panels) == 1
    assert panels[0]["panel_id"] == "0001_F01-P01"
    assert panels[0]["panel_bbox_xyxy"] == [2, 2, 20, 20]


def test_update_component_index_replaces_same_panel_components(tmp_path: Path):
    dataset_root = tmp_path
    links_dir = dataset_root / "links"
    links_dir.mkdir(parents=True, exist_ok=True)
    component_index_path = links_dir / "component_index.json"
    component_index_path.write_text(
        json.dumps(
            {
                "0001_F01-P01_chart-C01": {
                    "panel_id": "0001_F01-P01_chart",
                    "component_type": "axis",
                },
                "unrelated-C01": {
                    "panel_id": "0001_F01-P02_chart",
                    "component_type": "axis",
                },
            }
        ),
        encoding="utf-8",
    )

    update_component_index(
        dataset_root,
        [
            {
                "component_id": "0001_F01-P01_chart-C01",
                "panel_id": "0001_F01-P01_chart",
                "component_type": "plot_frame",
                "bbox_xyxy": [0, 0, 100, 100],
                "panel_path": "panels/0001/0001_F01-P01_chart.png",
                "crop_path": "components/0001/crops/0001_F01-P01_chart-C01_plot_frame.png",
                "overlay_path": "components/0001/overlays/0001_F01-P01_chart__overlay.png",
            }
        ],
    )

    data = _read_json(component_index_path)
    # same panel old entry is replaced, unrelated panel entry remains
    assert data["0001_F01-P01_chart-C01"]["component_type"] == "plot_frame"
    assert data["unrelated-C01"]["panel_id"] == "0001_F01-P02_chart"

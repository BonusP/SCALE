import json

from PIL import Image

import labeling_ui.app as app_module


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_save_panels_with_empty_boxes_clears_figure_panel_artifacts(tmp_path, monkeypatch):
    paper_id = "0006"
    figure_id = "0006_F05"
    figure_rel = f"figures/{paper_id}/{paper_id}_F05__src.png"
    panel_rel = f"panels/{paper_id}/{figure_id}-P01_chart.png"
    panel_id = f"{figure_id}-P01"
    component_panel_id = f"{figure_id}-P01_chart"

    figure_path = tmp_path / figure_rel
    panel_path = tmp_path / panel_rel
    crop_rel = f"components/{paper_id}/crops/{component_panel_id}-C01_plot_frame.png"
    overlay_rel = f"components/{paper_id}/overlays/{component_panel_id}__overlay.png"
    crop_path = tmp_path / crop_rel
    overlay_path = tmp_path / overlay_rel

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (64, 64), color="white").save(figure_path)
    Image.new("RGB", (32, 32), color="black").save(panel_path)
    Image.new("RGB", (16, 16), color="red").save(crop_path)
    Image.new("RGB", (32, 32), color="blue").save(overlay_path)

    _write_json(
        tmp_path / "labels" / "panels" / f"{paper_id}.json",
        {
            "panels": [
                {
                    "panel_id": panel_id,
                    "figure_id": figure_id,
                    "panel_bbox_xyxy": [1, 1, 30, 30],
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
                    "component_id": f"{component_panel_id}-C01",
                    "panel_id": component_panel_id,
                    "component_type": "plot_frame",
                    "bbox_xyxy": [2, 2, 10, 10],
                    "panel_path": panel_rel,
                    "crop_path": crop_rel,
                    "overlay_path": overlay_rel,
                }
            ]
        },
    )
    _write_json(
        tmp_path / "links" / "panel_index.json",
        {
            "panels": [
                {
                    "panel_id": panel_id,
                    "paper_id": paper_id,
                    "figure_id": figure_id,
                    "panel_path": panel_rel,
                    "candidate_type": "chart",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "links" / "component_index.json",
        {
            f"{component_panel_id}-C01": {
                "panel_id": component_panel_id,
                "component_type": "plot_frame",
                "bbox_xyxy": [2, 2, 10, 10],
                "panel_path": panel_rel,
                "crop_path": crop_rel,
                "overlay_path": overlay_rel,
            }
        },
    )

    monkeypatch.setattr(app_module, "DATASET_ROOT", tmp_path)
    app_module._PANEL_INDEX_CACHE = {"mtime": None, "by_figure": {}}
    app_module._PANEL_LABELS_CACHE = {}
    app_module._COMPONENT_INDEX_CACHE = {"mtime": None, "data": {}}
    app_module._COMPONENT_COUNTS_CACHE = {}

    res = app_module.save_panels(
        app_module.PanelSaveRequest(figure_rel_path=figure_rel, panel_boxes=[])
    )

    assert res == {"paper_id": paper_id, "panels": []}
    assert figure_path.exists()
    assert not panel_path.exists()
    assert not crop_path.exists()
    assert not overlay_path.exists()

    panel_labels = json.loads(
        (tmp_path / "labels" / "panels" / f"{paper_id}.json").read_text(encoding="utf-8")
    )
    component_labels = json.loads(
        (tmp_path / "labels" / "components" / f"{paper_id}.json").read_text(
            encoding="utf-8"
        )
    )
    panel_index = json.loads(
        (tmp_path / "links" / "panel_index.json").read_text(encoding="utf-8")
    )
    component_index = json.loads(
        (tmp_path / "links" / "component_index.json").read_text(encoding="utf-8")
    )

    assert panel_labels["panels"] == []
    assert component_labels["components"] == []
    assert panel_index["panels"] == []
    assert component_index == {}


def test_save_panels_replaces_old_panel_files_and_clears_stale_component_links(
    tmp_path, monkeypatch
):
    paper_id = "0008"
    figure_id = "0008_F02"
    figure_rel = f"figures/{paper_id}/{paper_id}_F02__src.png"
    old_panel_rel = f"panels/{paper_id}/{figure_id}-P03_chart.png"
    old_panel_id = f"{figure_id}-P03"
    old_typed_panel_id = f"{figure_id}-P03_chart"
    new_panel_rel = f"panels/{paper_id}/{figure_id}-P01_schematic.png"

    figure_path = tmp_path / figure_rel
    old_panel_path = tmp_path / old_panel_rel
    crop_rel = f"components/{paper_id}/crops/{old_typed_panel_id}-C01_plot_frame.png"
    overlay_rel = f"components/{paper_id}/overlays/{old_typed_panel_id}__overlay.png"
    crop_path = tmp_path / crop_rel
    overlay_path = tmp_path / overlay_rel

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    old_panel_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (80, 80), color="white").save(figure_path)
    Image.new("RGB", (20, 20), color="black").save(old_panel_path)
    Image.new("RGB", (10, 10), color="red").save(crop_path)
    Image.new("RGB", (20, 20), color="blue").save(overlay_path)

    _write_json(
        tmp_path / "labels" / "panels" / f"{paper_id}.json",
        {
            "panels": [
                {
                    "panel_id": old_panel_id,
                    "figure_id": figure_id,
                    "panel_bbox_xyxy": [5, 5, 30, 30],
                    "panel_path": old_panel_rel,
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
                    "component_id": f"{old_typed_panel_id}-C01",
                    "panel_id": old_typed_panel_id,
                    "component_type": "plot_frame",
                    "bbox_xyxy": [2, 2, 8, 8],
                    "panel_path": old_panel_rel,
                    "crop_path": crop_rel,
                    "overlay_path": overlay_rel,
                }
            ]
        },
    )
    _write_json(
        tmp_path / "links" / "panel_index.json",
        {
            "panels": [
                {
                    "panel_id": old_panel_id,
                    "paper_id": paper_id,
                    "figure_id": figure_id,
                    "panel_path": old_panel_rel,
                    "candidate_type": "chart",
                }
            ]
        },
    )
    _write_json(
        tmp_path / "links" / "component_index.json",
        {
            f"{old_typed_panel_id}-C01": {
                "panel_id": old_typed_panel_id,
                "component_type": "plot_frame",
                "bbox_xyxy": [2, 2, 8, 8],
                "panel_path": old_panel_rel,
                "crop_path": crop_rel,
                "overlay_path": overlay_rel,
            }
        },
    )

    monkeypatch.setattr(app_module, "DATASET_ROOT", tmp_path)
    app_module._PANEL_INDEX_CACHE = {"mtime": None, "by_figure": {}}
    app_module._PANEL_LABELS_CACHE = {}
    app_module._COMPONENT_INDEX_CACHE = {"mtime": None, "data": {}}
    app_module._COMPONENT_COUNTS_CACHE = {}

    app_module.save_panels(
        app_module.PanelSaveRequest(
            figure_rel_path=figure_rel,
            panel_boxes=[
                app_module.PanelBox(
                    bbox_xyxy=[10, 10, 60, 60],
                    panel_type="schematic",
                )
            ],
        )
    )

    assert figure_path.exists()
    assert not old_panel_path.exists()
    assert not crop_path.exists()
    assert not overlay_path.exists()
    assert (tmp_path / new_panel_rel).exists()

    panel_index = json.loads(
        (tmp_path / "links" / "panel_index.json").read_text(encoding="utf-8")
    )
    component_index = json.loads(
        (tmp_path / "links" / "component_index.json").read_text(encoding="utf-8")
    )
    component_labels = json.loads(
        (tmp_path / "labels" / "components" / f"{paper_id}.json").read_text(
            encoding="utf-8"
        )
    )

    assert [row["panel_path"] for row in panel_index["panels"]] == [new_panel_rel]
    assert component_index == {}
    assert component_labels["components"] == []

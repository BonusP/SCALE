import json

from PIL import Image

import labeling_ui.app as app_module


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_save_components_clears_old_crops_for_same_panel(tmp_path, monkeypatch):
    paper_id = "0040"
    panel_rel = f"panels/{paper_id}/0040_F01-P02_chart.png"
    panel_id = "0040_F01-P02_chart"
    panel_path = tmp_path / panel_rel
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color="white").save(panel_path)

    old_crop_c01_rel = (
        f"components/{paper_id}/crops/{panel_id}-C01_plot_frame.png"
    )
    old_crop_c02_rel = (
        f"components/{paper_id}/crops/{panel_id}-C02_axis.png"
    )
    overlay_rel = f"components/{paper_id}/overlays/{panel_id}__overlay.png"
    old_crop_c01_path = tmp_path / old_crop_c01_rel
    old_crop_c02_path = tmp_path / old_crop_c02_rel
    overlay_path = tmp_path / overlay_rel
    old_crop_c01_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color="red").save(old_crop_c01_path)
    Image.new("RGB", (20, 20), color="blue").save(old_crop_c02_path)
    Image.new("RGB", (20, 20), color="green").save(overlay_path)

    _write_json(
        tmp_path / "links" / "component_index.json",
        {
            f"{panel_id}-C01": {
                "panel_id": panel_id,
                "component_type": "plot_frame",
                "bbox_xyxy": [1, 1, 20, 20],
                "panel_path": panel_rel,
                "crop_path": old_crop_c01_rel,
                "overlay_path": overlay_rel,
            },
            f"{panel_id}-C02": {
                "panel_id": panel_id,
                "component_type": "axis",
                "bbox_xyxy": [21, 1, 40, 20],
                "panel_path": panel_rel,
                "crop_path": old_crop_c02_rel,
                "overlay_path": overlay_rel,
            },
        },
    )
    _write_json(
        tmp_path / "labels" / "components" / f"{paper_id}.json",
        {
            "components": [
                {
                    "component_id": f"{panel_id}-C01",
                    "panel_id": panel_id,
                    "component_type": "plot_frame",
                    "bbox_xyxy": [1, 1, 20, 20],
                    "panel_path": panel_rel,
                    "crop_path": old_crop_c01_rel,
                    "overlay_path": overlay_rel,
                },
                {
                    "component_id": f"{panel_id}-C02",
                    "panel_id": panel_id,
                    "component_type": "axis",
                    "bbox_xyxy": [21, 1, 40, 20],
                    "panel_path": panel_rel,
                    "crop_path": old_crop_c02_rel,
                    "overlay_path": overlay_rel,
                },
            ]
        },
    )

    monkeypatch.setattr(app_module, "DATASET_ROOT", tmp_path)
    app_module._COMPONENT_INDEX_CACHE = {"mtime": None, "data": {}}
    app_module._COMPONENT_COUNTS_CACHE = {}

    app_module.save_components(
        app_module.ComponentSaveRequest(
            panel_rel_path=panel_rel,
            components=[
                app_module.ComponentBox(
                    bbox_xyxy=[5, 5, 60, 60],
                    component_type="plot_frame",
                    component_class=None,
                )
            ],
        )
    )

    # old C02 should be removed when component set is replaced for same panel
    assert not old_crop_c02_path.exists()

from PIL import Image

import labeling_ui.app as app_module


def test_save_components_passes_class_labels_to_overlay(tmp_path, monkeypatch):
    paper_id = "0040"
    panel_rel = f"panels/{paper_id}/0040_F01-P02_chart.png"
    panel_path = tmp_path / panel_rel
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 120), color="white").save(panel_path)

    captured = {}

    def fake_draw_overlay(_img_path, _out_path, _boxes, labels=None):
        captured["labels"] = labels

    monkeypatch.setattr(app_module, "DATASET_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "draw_overlay", fake_draw_overlay)
    app_module._COMPONENT_INDEX_CACHE = {"mtime": None, "data": {}}
    app_module._COMPONENT_COUNTS_CACHE = {}

    app_module.save_components(
        app_module.ComponentSaveRequest(
            panel_rel_path=panel_rel,
            components=[
                app_module.ComponentBox(
                    bbox_xyxy=[10, 10, 50, 50],
                    component_type="plot_frame",
                    component_class=None,
                )
            ],
        )
    )

    assert captured["labels"] == ["chart"]

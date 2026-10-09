import pytest
from PIL import Image

import labeling_ui.app as app_module


def test_save_components_rejects_empty_bbox(tmp_path, monkeypatch):
    paper_id = "0038"
    panel_rel = f"panels/{paper_id}/0038_F02-P03_image.png"
    panel_path = tmp_path / panel_rel
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="white").save(panel_path)

    monkeypatch.setattr(app_module, "DATASET_ROOT", tmp_path)

    with pytest.raises(app_module.HTTPException) as exc:
        app_module.save_components(
            app_module.ComponentSaveRequest(
                panel_rel_path=panel_rel,
                components=[
                    app_module.ComponentBox(
                        bbox_xyxy=[],
                        component_type="plot_frame",
                        component_class=None,
                    )
                ],
            )
        )
    assert exc.value.status_code == 422
    assert "invalid component bbox" in str(exc.value.detail)

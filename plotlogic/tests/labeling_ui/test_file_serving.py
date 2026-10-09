from PIL import Image

import labeling_ui.app as app_module


def test_get_file_transcodes_mislabeled_tiff_png_to_png_response(tmp_path, monkeypatch):
    rel = "panels/0040/0040_F01-P02_chart.png"
    image_path = tmp_path / rel
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # Intentionally save TIFF bytes with .png extension (real-world dirty sample).
    Image.new("RGB", (12, 8), color="white").save(image_path, format="TIFF")

    monkeypatch.setattr(app_module, "DATASET_ROOT", tmp_path)

    response = app_module.get_file(rel=rel)

    assert response.media_type == "image/png"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.body.startswith(b"\x89PNG\r\n\x1a\n")

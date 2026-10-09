from PIL import Image

from labeling_ui.lib.overlay import draw_overlay


def test_overlay_created(tmp_path):
    img_path = tmp_path / "panel.png"
    Image.new("RGB", (64, 64), color="white").save(img_path)
    out = tmp_path / "overlay.png"
    draw_overlay(str(img_path), str(out), [[0, 0, 10, 10]])
    assert out.exists()


def test_overlay_created_with_labels(tmp_path):
    img_path = tmp_path / "panel.png"
    Image.new("RGB", (64, 64), color="white").save(img_path)
    out = tmp_path / "overlay_labeled.png"
    draw_overlay(
        str(img_path),
        str(out),
        [[5, 5, 40, 40], [20, 20, 50, 50]],
        labels=["C01", "C02"],
    )
    assert out.exists()

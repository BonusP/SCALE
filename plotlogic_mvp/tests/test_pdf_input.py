import json
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject, RectangleObject

from app.services.pdf_input import _pixel_bbox, rendered_pdf_inputs
from app.services.portable_pdf import poppler_command


def test_bbox_bottom_left_to_pixel_top_left():
    assert _pixel_bbox([10, 20, 40, 50], (100, 200), (200, 400)) == (20, 300, 80, 360)
    with pytest.raises(ValueError):
        _pixel_bbox([0, 0, 0, 10], (100, 200), (200, 400))


@pytest.mark.parametrize('rotation,bbox', [(0, [5, 35, 85, 75]), (90, [35, 95, 75, 175])])
def test_render_crop_matches_rotated_cropbox(tmp_path: Path, rotation, bbox):
    if not poppler_command():
        pytest.skip('Poppler unavailable')
    # PDF 입력 방향 보정: CropBox 원점·페이지 회전·상하 반전 이미지 배치를 함께 검증합니다.
    writer = PdfWriter()
    page = writer.add_blank_page(200, 300)
    page.mediabox = RectangleObject([10, 20, 210, 320])
    page.cropbox = RectangleObject([15, 25, 195, 305])
    image = DecodedStreamObject()
    image.set_data(bytes([255, 0, 0]) * 50 + bytes([0, 0, 255]) * 50)
    image.update({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Image'),
                  NameObject('/Width'): NumberObject(10), NameObject('/Height'): NumberObject(10),
                  NameObject('/ColorSpace'): NameObject('/DeviceRGB'), NameObject('/BitsPerComponent'): NumberObject(8)})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/XObject'): DictionaryObject({NameObject('/Im'): writer._add_object(image)})})
    contents = DecodedStreamObject()
    contents.set_data(b'q 80 0 0 -40 20 100 cm /Im Do Q')
    page[NameObject('/Contents')] = writer._add_object(contents)
    page.rotate(rotation)
    writer.write(tmp_path / 'source.pdf')
    output = tmp_path / 'opendataloader'
    output.mkdir()
    (output / 'source.json').write_text(json.dumps({'kids': [{'type': 'image', 'page number': 1,
                                                          'bounding box': bbox}]}), encoding='utf-8')
    paths = rendered_pdf_inputs(tmp_path)
    assert len(paths) == 1
    with Image.open(paths[0]) as crop:
        if rotation == 0:
            assert crop.getpixel((crop.width // 2, crop.height // 4))[:3] == (0, 0, 255)
            assert crop.getpixel((crop.width // 2, crop.height * 3 // 4))[:3] == (255, 0, 0)
        # Correct area, rather than a white part of the page.
        colored = sum(1 for r, g, b in crop.convert('RGB').get_flattened_data() if g < 20 and (r > 240 or b > 240))
        assert colored / (crop.width * crop.height) > 0.95
    stamp = paths[0].stat().st_mtime_ns
    assert rendered_pdf_inputs(tmp_path) == paths
    assert paths[0].stat().st_mtime_ns == stamp


def test_portable_cache_uses_pages_instead_of_shared_images(tmp_path: Path):
    if not poppler_command():
        pytest.skip('Poppler unavailable')
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_blank_page(100, 100)
    writer.write(tmp_path / 'source.pdf')
    (tmp_path / 'parse.json').write_text(json.dumps({'parser': 'pypdf_embedded_images', 'image_count': 441}))
    assert len(rendered_pdf_inputs(tmp_path)) == 2

import json
import os
from pathlib import Path

import pytest

from labeling_ui.lib.jsonio import (
    read_json_missing_default_strict_malformed,
    read_json_tolerant,
    write_json_atomic,
)


def test_read_json_returns_default_on_missing(tmp_path):
    path = tmp_path / "missing.json"
    default = {"panels": []}

    result = read_json_tolerant(path, default)

    assert result == default


def test_read_json_returns_default_on_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"panels": [}', encoding="utf-8")
    default = {"panels": []}

    result = read_json_tolerant(path, default)

    assert result == default


def test_read_json_does_not_modify_source_on_malformed(tmp_path):
    path = tmp_path / "bad.json"
    original = '{"components": [}'
    path.write_text(original, encoding="utf-8")

    _ = read_json_tolerant(path, {"components": []})

    assert path.read_text(encoding="utf-8") == original


def test_strict_reader_raises_on_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"components": [}', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_json_missing_default_strict_malformed(path, {"components": []})


def test_write_json_atomic_formats_with_indent_ascii_and_preserves_order(tmp_path):
    path = tmp_path / "output.json"
    payload = {"z": "한글", "a": 1}

    write_json_atomic(path, payload)

    expected = json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False)
    assert path.read_text(encoding="utf-8") == expected


def test_write_json_atomic_uses_same_directory_temp_file_and_replace(
    tmp_path, monkeypatch
):
    path = tmp_path / "output.json"
    payload = {"k": "v"}

    calls = []
    original_replace = os.replace

    def spy_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    write_json_atomic(path, payload)

    assert len(calls) == 1
    src, dst = calls[0]
    assert src.parent == path.parent
    assert dst == path

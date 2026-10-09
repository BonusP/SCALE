"""Pluggable text detection/recognition backend.

Kept deliberately thin: everything downstream only needs `(text, bbox, confidence)`
triples, so a different engine can be dropped in without touching callers.

The engine is loaded lazily and once, because model init costs seconds. If no
engine is installed the module reports that and callers degrade to their
heuristic path rather than failing.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .. import config


_LOCK = threading.Lock()
_ENGINE: Any = None
_ENGINE_ERROR: str | None = None
_ENGINE_NAME: str = ""

# Tick labels are numbers: digits, sign, decimal point, exponent, and the
# occasional unicode minus or OCR confusion for it.
_NUMERIC_TEXT = re.compile(r"^[\s]*[-+−–—]?\s*\d+(?:[.,]\d+)?\s*(?:[eE][-+]?\d+)?[\s]*$")


def _load_engine() -> tuple[Any, str | None, str]:
    global _ENGINE, _ENGINE_ERROR, _ENGINE_NAME
    with _LOCK:
        if _ENGINE is not None or _ENGINE_ERROR is not None:
            return _ENGINE, _ENGINE_ERROR, _ENGINE_NAME
        if not config.OCR_ENABLED:
            _ENGINE_ERROR = "OCR_DISABLED"
            return None, _ENGINE_ERROR, ""
        if config.OCR_BACKEND != "rapidocr":
            _ENGINE_ERROR = f"OCR_BACKEND_NOT_SUPPORTED:{config.OCR_BACKEND}"
            return None, _ENGINE_ERROR, ""
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception as exc:
            _ENGINE_ERROR = f"OCR_ENGINE_NOT_INSTALLED:{type(exc).__name__}"
            return None, _ENGINE_ERROR, ""
        try:
            _ENGINE = RapidOCR()
            _ENGINE_NAME = "rapidocr_onnxruntime"
        except Exception as exc:
            _ENGINE_ERROR = f"OCR_ENGINE_INIT_FAILED:{type(exc).__name__}: {exc}"
            return None, _ENGINE_ERROR, ""
        return _ENGINE, None, _ENGINE_NAME


def engine_status() -> dict[str, Any]:
    _, error, name = _load_engine()
    return {"available": error is None, "backend": name or config.OCR_BACKEND, "error": error}


def numeric_value(text: str) -> float | None:
    """Parse a tick label, tolerating the unicode minus signs OCR likes to emit."""
    if not _NUMERIC_TEXT.match(text or ""):
        return None
    cleaned = (
        str(text)
        .strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace(",", ".")
        .replace(" ", "")
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def read_text(image: Image.Image) -> dict[str, Any]:
    """Detect and recognise text, returning axis-aligned boxes in image pixels."""
    engine, error, name = _load_engine()
    if error or engine is None:
        return {"backend": name or config.OCR_BACKEND, "error": error, "items": []}
    try:
        raw, _ = engine(np.asarray(image.convert("RGB")))
    except Exception as exc:
        return {"backend": name, "error": f"OCR_RUN_FAILED:{type(exc).__name__}: {exc}", "items": []}

    items: list[dict[str, Any]] = []
    for entry in raw or []:
        try:
            polygon, text, confidence = entry[0], entry[1], float(entry[2])
        except (IndexError, TypeError, ValueError):
            continue
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        bbox = [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        if confidence < config.OCR_MIN_CONFIDENCE:
            continue
        items.append(
            {
                "text": str(text),
                "bbox": bbox,
                "confidence": round(confidence, 3),
                "numeric_value": numeric_value(str(text)),
            }
        )
    return {"backend": name, "error": None, "items": items}


def read_text_cached(image_path: Path, cache_path: Path) -> dict[str, Any]:
    """Run OCR once per panel; model init and inference cost seconds."""
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and isinstance(cached.get("items"), list):
                cached["cached"] = True
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    with Image.open(image_path) as handle:
        result = read_text(handle.convert("RGB"))
    result["source_image"] = str(image_path)
    if not result.get("error"):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        except OSError:
            pass
    result["cached"] = False
    return result

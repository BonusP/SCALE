"""Panel geometry with text boxes taken from OCR instead of from codex.

Everything that is not text — plot frame, axis lines, tick positions, legend
swatches, annotation regions — stays exactly as the shipped pipeline produced it.
Only the boxes around glyphs are replaced, because a text detector locates glyphs
more reliably than codex or an ink-extent heuristic can.

OCR annotation bbox 보정을 위한 수정: CMSL extraction opts into splitting in-plot
annotation search regions into individual OCR text boxes. Broad unconfirmed boxes
are recorded for review without erasing their entire area. Other callers retain
the original annotation behavior.

Runs at debug-request time and writes nothing back into the extraction, so the
shipped pipeline is unaffected and finished jobs can be inspected without a re-run.
"""

from __future__ import annotations

import json
from typing import Any

from .bbox_refinement import _bbox, _normalize_axis_role


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _label_center(role: str, box: list[int]) -> float:
    """Position of a label along its axis: x for the bottom axis, y otherwise."""
    if role == "bottom":
        return (box[0] + box[2]) / 2
    return (box[1] + box[3]) / 2


def _axis_offset(role: str, box: list[int], frame: list[int]) -> float:
    """Perpendicular distance from the axis line to a box centre."""
    if role == "bottom":
        return (box[1] + box[3]) / 2 - frame[3]
    if role == "left":
        return frame[0] - (box[0] + box[2]) / 2
    return (box[0] + box[2]) / 2 - frame[2]


def _axis_of_ocr_item(item: dict[str, Any], frame: list[int]) -> str | None:
    """Place a recognised number against an axis by where it sits around the frame."""
    box = item["bbox"]
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    frame_w = max(1, frame[2] - frame[0])
    frame_h = max(1, frame[3] - frame[1])
    in_x_span = frame[0] - frame_w * 0.08 <= center_x <= frame[2] + frame_w * 0.08
    in_y_span = frame[1] - frame_h * 0.08 <= center_y <= frame[3] + frame_h * 0.08
    if in_x_span and frame[3] < center_y <= frame[3] + frame_h * 0.22:
        return "bottom"
    if in_y_span and frame[0] - frame_w * 0.30 <= center_x < frame[0]:
        return "left"
    if in_y_span and frame[2] < center_x <= frame[2] + frame_w * 0.30:
        return "right"
    return None


def _ocr_items_by_axis(items: list[dict[str, Any]], frame: list[int]) -> dict[str, list[dict[str, Any]]]:
    by_axis: dict[str, list[dict[str, Any]]] = {"bottom": [], "left": [], "right": []}
    for item in items:
        if item.get("numeric_value") is None or not isinstance(item.get("bbox"), list):
            continue
        role = _axis_of_ocr_item(item, frame)
        if role:
            by_axis[role].append({**item, "center": _label_center(role, item["bbox"])})
    return by_axis


def _match_ocr_to_ticks(
    role: str,
    ticks: list[dict[str, Any]],
    ocr_items: list[dict[str, Any]],
    frame: list[int],
) -> dict[str, Any]:
    """Pair OCR readings with codex ticks and record where the two disagree."""
    unused = list(ocr_items)
    matched = 0
    value_mismatches: list[str] = []

    # Prefer pairing on the value itself, then fall back to position proximity.
    for match_on_value in (True, False):
        for tick in ticks:
            if tick.get("ocr_text") is not None or not unused:
                continue
            tick_value = tick.get("label_numeric_value")
            anchor = None
            label_bbox = _bbox(tick.get("label_bbox_panel_px"))
            if label_bbox:
                anchor = _label_center(role, label_bbox)
            if match_on_value:
                if not isinstance(tick_value, (int, float)):
                    continue
                candidates = [
                    item for item in unused
                    if abs(float(item["numeric_value"]) - float(tick_value)) <= abs(float(tick_value)) * 1e-6 + 1e-9
                ]
                if not candidates:
                    continue
                if anchor is not None and len(candidates) > 1:
                    candidates.sort(key=lambda item: abs(item["center"] - anchor))
                chosen = candidates[0]
            else:
                if anchor is None:
                    continue
                chosen = min(unused, key=lambda item: abs(item["center"] - anchor))
                box = chosen["bbox"]
                extent = (box[2] - box[0]) if role == "bottom" else (box[3] - box[1])
                if abs(chosen["center"] - anchor) > max(12.0, extent * 1.5):
                    continue
            unused.remove(chosen)
            matched += 1
            tick["ocr_text"] = chosen["text"]
            tick["ocr_numeric_value"] = chosen["numeric_value"]
            tick["ocr_confidence"] = chosen["confidence"]
            tick["ocr_label_bbox_panel_px"] = chosen["bbox"]
            if isinstance(tick_value, (int, float)) and chosen["numeric_value"] != tick_value:
                tick["ocr_value_agrees"] = False
                value_mismatches.append(f"codex={tick_value}/ocr={chosen['numeric_value']}")
            else:
                tick["ocr_value_agrees"] = True

    # Leftovers only count as a missed tick if they sit in the same band as the
    # real labels. Superscripts in the axis title read as bare numbers too.
    matched_offsets = [
        _axis_offset(role, _bbox(tick["ocr_label_bbox_panel_px"]) or [0, 0, 1, 1], frame)
        for tick in ticks
        if tick.get("ocr_label_bbox_panel_px")
    ]
    off_band = 0
    if matched_offsets:
        typical = _median(matched_offsets)
        in_band = []
        for item in unused:
            box = item["bbox"]
            size = (box[3] - box[1]) if role == "bottom" else (box[2] - box[0])
            if abs(_axis_offset(role, box, frame) - typical) <= max(12.0, size):
                in_band.append(item)
            else:
                off_band += 1
        unused = in_band

    return {
        "matched": matched,
        "codex_ticks": len(ticks),
        "ocr_numbers": len(ocr_items),
        "unmatched_ocr": [
            {"text": item["text"], "value": item["numeric_value"], "center": round(item["center"], 1)}
            for item in unused
        ],
        "off_band_numbers": off_band,
        "value_mismatches": value_mismatches,
    }


def _ocr_union_for_region(
    items: list[dict[str, Any]],
    target: list[int],
    consumed: set[tuple[int, ...]],
    *,
    expand: float = 0.25,
    max_growth: float = 3.0,
) -> list[int] | None:
    """Union of the OCR boxes sitting inside a codex text region.

    Long titles come back as several detections ("Power" + "density(m"), so they
    have to be merged. The growth cap stops a neighbouring tick label from being
    swallowed into an axis title.
    """
    width = max(1, target[2] - target[0])
    height = max(1, target[3] - target[1])
    expanded = [
        target[0] - width * expand,
        target[1] - height * expand,
        target[2] + width * expand,
        target[3] + height * expand,
    ]
    hits = []
    for item in items:
        box = item["bbox"]
        if tuple(box) in consumed:
            continue
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        if expanded[0] <= center_x <= expanded[2] and expanded[1] <= center_y <= expanded[3]:
            hits.append(item)
    if not hits:
        return None
    union = [
        min(item["bbox"][0] for item in hits),
        min(item["bbox"][1] for item in hits),
        max(item["bbox"][2] for item in hits),
        max(item["bbox"][3] for item in hits),
    ]
    if (union[2] - union[0]) * (union[3] - union[1]) > max(1, width * height) * max_growth:
        return None
    for item in hits:
        consumed.add(tuple(item["bbox"]))
    return union


def build_ocr_text_geometry(
    panel_vision: dict[str, Any],
    ocr_items: list[dict[str, Any]] | None = None,
    *,
    refine_annotations: bool = False,
) -> dict[str, Any]:
    """Return the pipeline's geometry with every text box replaced by an OCR box."""
    vision = json.loads(json.dumps(panel_vision))
    ocr_items = ocr_items or []
    consumed: set[tuple[int, ...]] = set()
    issues: list[str] = []

    frame = _bbox(vision.get("plot_frame_bbox_panel_px"))
    if not frame:
        issues.append("NO_PLOT_FRAME_IN_PIPELINE_OUTPUT")
    ocr_by_axis = _ocr_items_by_axis(ocr_items, frame) if (frame and ocr_items) else {}

    axis_reports: list[dict[str, Any]] = []
    replaced = {"tick_label": 0, "axis_label": 0, "legend_label": 0}
    for axis in vision.get("axes", []):
        if not isinstance(axis, dict):
            continue
        role = _normalize_axis_role(axis.get("axis_role"))
        axis["axis_role"] = role
        ticks = [tick for tick in axis.get("ticks", []) if isinstance(tick, dict)]

        report = None
        if frame and role in {"bottom", "left", "right"} and ocr_by_axis:
            report = _match_ocr_to_ticks(role, ticks, ocr_items=ocr_by_axis.get(role, []), frame=frame)

        from_ocr = 0
        for tick in ticks:
            ocr_bbox = _bbox(tick.get("ocr_label_bbox_panel_px"))
            if not ocr_bbox:
                continue
            tick["codex_label_bbox_panel_px"] = tick.get("label_bbox_panel_px")
            tick["label_bbox_panel_px"] = ocr_bbox
            tick["label_bbox_source"] = "ocr"
            consumed.add(tuple(ocr_bbox))
            from_ocr += 1
        replaced["tick_label"] += from_ocr

        axis_reports.append(
            {
                "axis_role": role,
                "tick_count": len(ticks),
                "tick_labels_from_ocr": from_ocr,
                "ocr": report,
            }
        )
        if report:
            if report["value_mismatches"]:
                issues.append(f"{role}:OCR_VALUE_MISMATCH:{','.join(report['value_mismatches'])}")
            if report["unmatched_ocr"]:
                labels = ",".join(str(item["text"]) for item in report["unmatched_ocr"])
                issues.append(f"{role}:OCR_LABEL_MISSING_FROM_CODEX:{labels}")
            unread = report["codex_ticks"] - report["matched"]
            if unread:
                issues.append(f"{role}:CODEX_TICK_NOT_CONFIRMED_BY_OCR:{unread}")

    # Axis titles and legend labels: the tick pass has already claimed the numbers,
    # so whatever OCR text is left can be matched by position alone.
    for axis in vision.get("axes", []):
        if not isinstance(axis, dict):
            continue
        target = _bbox(axis.get("label_bbox_panel_px"))
        if not target:
            continue
        union = _ocr_union_for_region(ocr_items, target, consumed)
        if union:
            axis["codex_label_bbox_panel_px"] = axis.get("label_bbox_panel_px")
            axis["label_bbox_panel_px"] = union
            axis["label_bbox_source"] = "ocr"
            replaced["axis_label"] += 1
    for item in vision.get("legend_items", []):
        if not isinstance(item, dict):
            continue
        target = _bbox(item.get("label_bbox_panel_px"))
        if not target:
            continue
        union = _ocr_union_for_region(ocr_items, target, consumed)
        if union:
            item["codex_label_bbox_panel_px"] = item.get("label_bbox_panel_px")
            item["label_bbox_panel_px"] = union
            item["label_bbox_source"] = "ocr"
            replaced["legend_label"] += 1

    # OCR annotation bbox 보정을 위한 수정: split in-plot text regions into individual OCR boxes.
    # Only CMSL opts in; axis geometry and the other methods retain their existing behavior.
    annotation_report = None
    if refine_annotations and frame:
        annotation_report = _refine_annotation_regions(vision, ocr_items, frame, consumed)
        replaced["annotation_text"] = annotation_report["ocr_box_count"]
        issues.extend(annotation_report["issues"])

    confirmed = 0
    disputed = 0
    for axis in vision.get("axes", []):
        if not isinstance(axis, dict):
            continue
        for tick in axis.get("ticks", []):
            if not isinstance(tick, dict):
                continue
            if tick.get("ocr_value_agrees") is True:
                confirmed += 1
            elif tick.get("ocr_value_agrees") is False:
                disputed += 1

    return {
        "coordinate_space": "panel_px",
        "geometry_source": "codex_pipeline",
        "text_source": "ocr",
        "plot_frame_bbox_panel_px": frame,
        "panel_vision": vision,
        "axes_report": axis_reports,
        "issues": issues,
        "summary": {
            "text_boxes_replaced": replaced,
            "ocr_confirmed_values": confirmed,
            "ocr_conflicting_values": disputed,
            "annotation_refinement": annotation_report,
        },
    }


# OCR annotation bbox 보정을 위한 수정: a broad Codex box is a search region, not an eraser.
def _refine_annotation_regions(vision, items, frame, consumed):
    from .. import config
    frame_w, frame_h = frame[2] - frame[0], frame[3] - frame[1]
    refined, issues = [], []
    seen = set(consumed)
    ocr_count = skipped = 0
    for index, region in enumerate(vision.get("plot_annotation_regions", [])):
        if not isinstance(region, dict):
            continue
        target = _bbox(region.get("bbox_panel_px"))
        text = str(region.get("annotation_text") or "").lower()
        if not target or "legend" in text or "toolbar" in text or "footer" in text or "ui" in text:
            refined.append(region)
            continue
        if target[2] <= frame[0] or target[0] >= frame[2] or target[3] <= frame[1] or target[1] >= frame[3]:
            refined.append(region)
            continue
        hits = []
        matched = False
        for item in items:
            box = _bbox(item.get("bbox"))
            confidence = item.get("confidence")
            if not box or not str(item.get("text") or "").strip():
                continue
            if not isinstance(confidence, (int, float)) or confidence < config.OCR_MIN_CONFIDENCE:
                continue
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (target[0] <= cx <= target[2] and target[1] <= cy <= target[3]
                    and frame[0] <= cx <= frame[2] and frame[1] <= cy <= frame[3]):
                continue
            # Reject OCR detections that themselves span an implausibly large part of the plot.
            if box[2] - box[0] > frame_w * 0.25 or box[3] - box[1] > frame_h * 0.15:
                continue
            matched = True
            if tuple(box) not in seen:
                seen.add(tuple(box))
                hits.append({**region, "annotation_text": item["text"], "bbox_panel_px": box,
                             "bbox_source": "ocr_annotation", "confidence": confidence,
                             "codex_bbox_panel_px": target, "annotation_region_index": index})
        if matched:
            refined.extend(hits)
            ocr_count += len(hits)
        elif ((target[2] - target[0]) <= frame_w * 0.20
              and (target[3] - target[1]) <= frame_h * 0.12):
            # Preserve a small unconfirmed text box, but flag it for review.
            refined.append({**region, "bbox_source": "codex_annotation_unconfirmed"})
            issues.append(f"ANNOTATION_NOT_CONFIRMED_BY_OCR:{index}")
        else:
            # Keep provenance in debug output without deleting an unconfirmed broad rectangle.
            refined.append({**region, "bbox_panel_px": None, "codex_bbox_panel_px": target,
                            "bbox_source": "ocr_annotation_unconfirmed_skipped"})
            skipped += 1
            issues.append(f"BROAD_ANNOTATION_EXCLUSION_SKIPPED:{index}")
    vision["plot_annotation_regions"] = refined
    return {"ocr_box_count": ocr_count, "broad_regions_skipped": skipped, "issues": issues}

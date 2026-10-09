# C-V-P Color Mask Correction

You are reviewing color masks for a current-voltage-power dual-y chart extraction.

You will see a contact sheet with the plot crop, color-filter source, and one row per candidate color mask.
Each mask is binary: white pixels are currently selected as belonging to that color/series, black pixels are rejected.

Goal:
- Keep only pixels that plausibly belong to the visible plotted curve for that mask.
- Remove legend text, axis lines, tick labels, panel labels, annotations, isolated dust, and unrelated curves.
- If two same-color traces are mixed, keep the trace that matches the mask label/role evidence when it is visually clear.
- If the mask is already acceptable, use the original mask.
- If the mask is too ambiguous to correct with rectangular regions, mark it for human review instead of guessing.

Return JSON only.

Coordinate rules:
- All bboxes must use crop pixel coordinates from the original plot crop, not the resized contact-sheet thumbnail.
- Use `[x1, y1, x2, y2]` with integer values.
- `keep_bboxes_crop_px` means pixels outside the union of those boxes will be removed.
- `remove_bboxes_crop_px` means pixels inside those boxes will be removed.
- Prefer a small number of broad, safe bboxes over many tiny boxes.

Do not invent data points. Do not change axis mapping. Do not estimate curve values.

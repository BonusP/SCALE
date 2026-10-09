from __future__ import annotations

# ###수정 추가 수정: 점선·marker의 짧은 빈 구간을 별도 support mask로 기록하며 tracker의 10/10 제한은 유지합니다.
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage


def build_tracking_support(mask: Image.Image, *, tracker: Callable, forbidden_mask: Image.Image | None = None,
                           max_gap_px: int = 24, alignment_tolerance_px: float = 5.0) -> tuple[Image.Image, Image.Image, dict]:
    observed = np.asarray(mask.convert("L")) > 0
    support = observed.copy()
    anchors = observed.copy()
    blocked = np.asarray(forbidden_mask.convert("L")) > 0 if forbidden_mask is not None else np.zeros_like(observed)
    labels, _count = ndimage.label(observed, structure=np.ones((3, 3)))
    marker_centers = set()
    marker_limit = 14
    for component, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        width, height = x_slice.stop - x_slice.start, y_slice.stop - y_slice.start
        local = labels[slices] == component
        if width <= marker_limit and height <= marker_limit and local.sum() >= 5 and .4 <= width / height <= 2.5:
            ys, xs = np.nonzero(local)
            cx, cy = int(round(float(xs.mean()))) + x_slice.start, int(round(float(ys.mean()))) + y_slice.start
            marker_centers.add((cx, cy))
            anchors[slices] &= ~local
            anchors[cy, cx] = True
            # Filled support makes hollow marker columns yield one representative y rather than two edge tracks.
            for x in range(width):
                column = np.flatnonzero(local[:, x])
                if len(column):
                    support[y_slice.start + column[0]:y_slice.start + column[-1] + 1, x_slice.start + x] = True
            support[cy, cx] = True
    points = {(int(x), int(y)) for y, x in np.argwhere(anchors)}
    fragments = tracker(points, image_width=mask.width, min_width_px=1, retain_fragments=True)

    def tangent(track, start):
        if len(track) < 4 or (track[0] if start else track[-1]) in marker_centers:
            return None
        # Rasterized stroke endpoints have asymmetric column thickness; estimate direction from interior columns.
        interior = track[1:-1]
        values = interior[:8] if start else interior[-8:]
        xs, ys = np.asarray(values, dtype=float).T
        variance = float(np.sum((xs - xs.mean()) ** 2))
        return float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / variance) if variance else None

    endpoints = [(track[0], track[-1], tangent(track, True), tangent(track, False)) for track in fragments]
    edges = []
    for i, (_start, end, _first_slope, last_slope) in enumerate(endpoints):
        for j, (start, _end, first_slope, _last_slope) in enumerate(endpoints):
            dx, dy = start[0] - end[0], start[1] - end[1]
            if i == j or not 1 < dx <= max_gap_px or abs(dy) > max_gap_px * 4:
                continue
            residuals = []
            if last_slope is not None:
                residuals.append(abs(dy - last_slope * dx))
            if first_slope is not None:
                residuals.append(abs(dy - first_slope * dx))
            if not residuals:
                # Three consecutive anchors must support the proposed direction of an isolated marker pair.
                slope = dy / dx
                evidence = []
                for k, (next_start, prev_end, _a, _b) in enumerate(endpoints):
                    if k in (i, j):
                        continue
                    before = end[0] - prev_end[0]
                    after = next_start[0] - start[0]
                    if 1 < before <= max_gap_px:
                        evidence.append(abs(end[1] - prev_end[1] - slope * before))
                    if 1 < after <= max_gap_px:
                        evidence.append(abs(next_start[1] - start[1] - slope * after))
                if not evidence:
                    continue
                residuals.append(min(evidence))
            if max(residuals) > alignment_tolerance_px:
                continue
            segment = [(end[0] + step, int(round(end[1] + dy * step / dx))) for step in range(1, dx)]
            if any(blocked[y, x] for x, y in segment):
                continue
            edges.append((max(residuals) * 4 + dx + abs(dy) * .1, i, j, segment))
    outgoing, incoming = set(), set()
    accepted_edges = []
    for _score, i, j, segment in sorted(edges):
        if i in outgoing or j in incoming:
            continue
        outgoing.add(i)
        incoming.add(j)
        for x, y in segment:
            support[y, x] = True
        accepted_edges.append({"from": list(endpoints[i][1]), "to": list(endpoints[j][0]), "inserted_positions": len(segment)})
    support[blocked] = False
    inferred = support & ~observed
    report = {"modification_note": "###수정 추가 수정: 별도 marker/gap support 보간; 본문 밖의 실험 보완 규칙",
              "max_support_gap_px": max_gap_px, "alignment_tolerance_px": alignment_tolerance_px,
              "marker_component_limit_px": marker_limit, "marker_center_count": len(marker_centers),
              "observed_pixel_count": int(observed.sum()), "inferred_pixel_count": int(inferred.sum()),
              "tracker_max_x_gap_px": 10, "tracker_max_y_gap_px": 10, "bridges": accepted_edges}
    return Image.fromarray(support.astype(np.uint8) * 255), Image.fromarray(inferred.astype(np.uint8) * 255), report
# ###수정 추가 수정: 같은 색의 교차점에서 방향이 급변한 두 track의 tail만 교환하고 좌표·10/10 간격은 보존합니다.
def repair_crossing_tracks(tracks: list[list[tuple[int, int]]], *, max_gap_px: int = 10) -> tuple[list[list[tuple[int, int]]], int]:
    import math
    repaired = [sorted(track) for track in tracks]
    swaps = 0

    def direction(points):
        xs, ys = np.asarray(points, dtype=float).T
        variance = float(np.sum((xs - xs.mean()) ** 2))
        return math.atan(float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / variance)) if variance else None

    for i in range(len(repaired)):
        for j in range(i + 1, len(repaired)):
            a, b = repaired[i], repaired[j]
            by_x_a, by_x_b = dict(a), dict(b)
            near = [x for x in by_x_a.keys() & by_x_b.keys() if abs(by_x_a[x] - by_x_b[x]) <= 6]
            best = None
            for x in near:
                al, ar = [p for p in a if p[0] <= x], [p for p in a if p[0] > x]
                bl, br = [p for p in b if p[0] <= x], [p for p in b if p[0] > x]
                if min(map(len, (al, ar, bl, br))) < 8:
                    continue
                joins = [(al[-1], br[0]), (bl[-1], ar[0])]
                if any(not 0 < end[0] - start[0] <= max_gap_px or abs(end[1] - start[1]) > max_gap_px for start, end in joins):
                    continue
                angles = [direction(points) for points in (al[-12:], ar[:12], bl[-12:], br[:12])]
                if any(angle is None for angle in angles):
                    continue
                left_a, right_a, left_b, right_b = angles
                improvement = abs(left_a - right_a) + abs(left_b - right_b) - abs(left_a - right_b) - abs(left_b - right_a)
                if improvement > .25 and (best is None or improvement > best[0]):
                    best = (improvement, al + br, bl + ar)
            if best:
                repaired[i], repaired[j] = best[1], best[2]
                swaps += 1
    return repaired, swaps
# ###

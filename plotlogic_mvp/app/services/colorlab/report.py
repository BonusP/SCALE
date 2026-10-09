"""Self-contained side-by-side comparison report.

Everything is inlined (images as base64, CSVs as data URIs) so the file can be copied
anywhere and still work.  Open it as a local file in a browser and the CSV links
download; the page needs no server.
"""

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable

from .assemble import Series
from .axes import AxisFrame

PLOT_W, PLOT_H = 430, 300
PAD_L, PAD_R, PAD_T, PAD_B = 62, 16, 26, 46


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _img(path: Path) -> str | None:
    return _data_uri(path, "image/png") if path.exists() else None


def _csv_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:text/csv;charset=utf-8;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _read_legacy_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    series: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row.get("trace_id") or row.get("series_id") or "?"
            entry = series.setdefault(
                key,
                {
                    "label": row.get("legend_label") or row.get("series_id") or key,
                    "axis": "right" if (row.get("axis_binding") or "").startswith("right") else "left",
                    "role": row.get("curve_role") or "",
                    "points": [],
                },
            )
            try:
                entry["points"].append((float(row["x"]), float(row["y"])))
            except (KeyError, TypeError, ValueError):
                continue
    for entry in series.values():
        entry["points"].sort(key=lambda point: point[0])
    return series


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not (high > low):
        return [low]
    raw = (high - low) / max(1, count)
    magnitude = 10 ** int(f"{raw:e}".split("e")[1])
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=magnitude * 10)
    start = step * int(low / step)
    ticks = []
    value = start
    while value <= high + step * 0.5:
        if value >= low - step * 0.001:
            ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.3g}"
    return f"{value:g}"


def _svg_chart(
    title: str,
    entries: list[dict[str, Any]],
    x_label: str,
    y_label: str,
    *,
    marker: bool,
) -> str:
    points_all = [point for entry in entries for point in entry["points"]]
    if not points_all:
        return (
            f'<figure class="chart empty"><figcaption>{html.escape(title)}</figcaption>'
            f'<div class="empty-box">no data</div></figure>'
        )
    xs = [p[0] for p in points_all]
    ys = [p[1] for p in points_all]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    pad_y = (y1 - y0) * 0.06
    y0, y1 = y0 - pad_y, y1 + pad_y

    def sx(value: float) -> float:
        return PAD_L + (value - x0) / (x1 - x0) * (PLOT_W - PAD_L - PAD_R)

    def sy(value: float) -> float:
        return PLOT_H - PAD_B - (value - y0) / (y1 - y0) * (PLOT_H - PAD_T - PAD_B)

    parts: list[str] = [f'<svg viewBox="0 0 {PLOT_W} {PLOT_H}" role="img" aria-label="{html.escape(title)}">']
    parts.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{PLOT_W - PAD_L - PAD_R}" '
        f'height="{PLOT_H - PAD_T - PAD_B}" class="plot-bg"/>'
    )
    for tick in _nice_ticks(x0, x1):
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PLOT_H - PAD_B}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{PLOT_H - PAD_B + 15}" class="tick" text-anchor="middle">{_fmt(tick)}</text>')
    for tick in _nice_ticks(y0, y1):
        y = sy(tick)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PLOT_W - PAD_R}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD_L - 7}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">{_fmt(tick)}</text>')

    for entry in entries:
        colour = entry.get("colour", "#64748b")
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{sx(px):.1f},{sy(py):.1f}"
            for index, (px, py) in enumerate(entry["points"])
        )
        dash = ' stroke-dasharray="5 3"' if entry.get("axis") == "right" else ""
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.6"{dash}/>')
        if marker:
            for px, py in entry["points"]:
                parts.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="2.1" fill="{colour}"/>')

    parts.append(
        f'<text x="{(PAD_L + PLOT_W - PAD_R) / 2:.0f}" y="{PLOT_H - 8}" class="axis-label" '
        f'text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="14" y="{(PAD_T + PLOT_H - PAD_B) / 2:.0f}" class="axis-label" text-anchor="middle" '
        f'transform="rotate(-90 14 {(PAD_T + PLOT_H - PAD_B) / 2:.0f})">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>")
    return f'<figure class="chart"><figcaption>{html.escape(title)}</figcaption>{"".join(parts)}</figure>'


def _colorlab_entries(series_list: list[Series], rows: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    by_series: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row["y_axis_role"] != axis or row["x_value"] == "" or row["y_value"] == "":
            continue
        by_series.setdefault(row["series_id"], []).append((float(row["x_value"]), float(row["y_value"])))
    entries = []
    for series in series_list:
        points = sorted(by_series.get(series.series_id, []))
        if not points:
            continue
        entries.append(
            {
                "label": f"{series.legend_label or series.series_id} ({series.glyph_type})",
                "colour": "#%02x%02x%02x" % tuple(series.colour_rgb),
                "points": points,
                "axis": axis,
            }
        )
    return entries


def _legacy_entries(legacy: dict[str, dict[str, Any]], axis: str, palette: dict[str, str]) -> list[dict[str, Any]]:
    entries = []
    for key, entry in legacy.items():
        if entry["axis"] != axis or not entry["points"]:
            continue
        entries.append(
            {
                "label": f"{entry['label']} ({entry['role']})",
                "colour": palette.get(str(entry["label"]), "#64748b"),
                "points": entry["points"],
                "axis": axis,
            }
        )
    return entries


def _legend_html(entries: Iterable[dict[str, Any]]) -> str:
    seen: dict[str, str] = {}
    for entry in entries:
        seen.setdefault(entry["label"], entry["colour"])
    if not seen:
        return ""
    chips = "".join(
        f'<span class="chip"><i style="background:{colour}"></i>{html.escape(label)}</span>'
        for label, colour in seen.items()
    )
    return f'<div class="legend">{chips}</div>'


def write_comparison_report(
    *,
    run_dir: Path,
    out_path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    series_list: list[Series],
    frame: AxisFrame,
    legacy_traces: list[dict[str, Any]],
) -> Path:
    out_dir = out_path.parent
    legacy = _read_legacy_csv(run_dir / "curve_series_data.csv")
    palette = {
        (series.legend_label or series.series_id): "#%02x%02x%02x" % tuple(series.colour_rgb)
        for series in series_list
    }

    x_label = frame.models["bottom"].label or frame.x_role
    left_label = frame.models["left"].label or frame.left_role
    right_label = frame.models["right"].label or (frame.right_role or "right")

    qc = summary["qc"]
    metrics = qc["metrics"]
    verdict_class = "warn" if qc["low_confidence"] else "ok"
    verdict_text = (
        "저신뢰 — " + ", ".join(qc["failed_gates"]) if qc["low_confidence"] else "QC 게이트 전부 통과"
    )

    charts_left = [
        _svg_chart(f"기존 파이프라인 · {left_label}", _legacy_entries(legacy, "left", palette), x_label, left_label, marker=False),
        _svg_chart(f"colorlab · {left_label}", _colorlab_entries(series_list, rows, "left"), x_label, left_label, marker=True),
    ]
    charts_right = [
        _svg_chart(f"기존 파이프라인 · {right_label}", _legacy_entries(legacy, "right", palette), x_label, right_label, marker=False),
        _svg_chart(f"colorlab · {right_label}", _colorlab_entries(series_list, rows, "right"), x_label, right_label, marker=True),
    ]

    legacy_overlay = _img(out_dir / "legacy_overlay.png")
    colorlab_overlay = _img(out_dir / "colorlab_overlay.png")
    curve_only = _img(out_dir / "curve_only.png")
    layer_map = _img(out_dir / "layer_map.png")
    crop = _img(run_dir / "plot_crop.png")

    downloads = [
        ("colorlab · points_long.csv", "마커 단위 원본 측정점 (권장)", _csv_uri(out_dir / "points_long.csv"), "points_long.csv"),
        ("colorlab · series_wide.csv", "공통 x 격자 리샘플 (엑셀 붙여넣기용)", _csv_uri(out_dir / "series_wide.csv"), "series_wide.csv"),
        ("기존 · curve_series_data.csv", "기존 파이프라인 결과 (비교용)", _csv_uri(run_dir / "curve_series_data.csv"), "legacy_curve_series_data.csv"),
    ]
    download_html = "".join(
        f'<a class="dl" download="{name}" href="{uri}"><strong>{html.escape(title)}</strong>'
        f'<span>{html.escape(note)}</span></a>'
        for title, note, uri, name in downloads
        if uri
    )

    series_rows = "".join(
        f"<tr><td><code>{s['series_id']}</code></td>"
        f'<td><span class="swatch" style="background:{"#%02x%02x%02x" % tuple(s["colour_rgb"])}"></span>'
        f"{html.escape(str(s['legend_label'] or '—'))}</td>"
        f"<td>{s['glyph_type']}</td><td>{s['point_count']}</td>"
        f"<td>{s['shape']}</td><td>{s['y_axis']} · {html.escape(str(s['quantity']))}</td>"
        f"<td>{s['flagged_point_count']}</td></tr>"
        for s in summary["series"]
    )

    gate_rows = "".join(
        f'<tr class="{"gate-fail" if not g["passed"] else ""}"><td>{g["gate"]}</td>'
        f'<td>{g["value"]}</td><td>{g["comparison"]} {g["limit"]}</td>'
        f'<td>{"통과" if g["passed"] else "실패 — " + html.escape(g["message"])}</td></tr>'
        for g in qc["gates"]
    )

    palette_rows = "".join(
        f'<tr><td>{e["palette_index"]}</td><td>{html.escape(str(e["legend_label"] or "—"))}</td>'
        f'<td><span class="swatch" style="background:rgb({",".join(str(v) for v in (e["seed_rgb"] or [200,200,200]))})"></span>'
        f'{e["seed_rgb"]}</td>'
        f'<td><span class="swatch" style="background:rgb({",".join(str(v) for v in e["refined_rgb"])})"></span>'
        f'{e["refined_rgb"]}</td><td>{e["nearest_other_delta_e"]}</td><td>{e["layer_px"]}</td></tr>'
        for e in summary["palette"]["entries"]
    )

    def image_card(title: str, note: str, uri: str | None) -> str:
        if not uri:
            return ""
        return (
            f'<figure class="shot"><figcaption><strong>{html.escape(title)}</strong>'
            f'<span>{html.escape(note)}</span></figcaption>'
            f'<img src="{uri}" alt="{html.escape(title)}"></figure>'
        )

    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>곡선 추출 비교 · {html.escape(summary['panel_id'])}</title>
<style>
  :root {{
    --bg:#f6f7f9; --card:#ffffff; --ink:#111827; --muted:#6b7280; --line:#e5e7eb;
    --ok:#047857; --ok-bg:#ecfdf5; --warn:#b45309; --warn-bg:#fffbeb; --accent:#1d4ed8;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI","Noto Sans KR",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:32px 22px 72px; }}
  h1 {{ font-size:24px; margin:0 0 4px; letter-spacing:-.01em; }}
  h2 {{ font-size:17px; margin:36px 0 12px; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); margin:0 0 20px; font-size:14px; }}
  .verdict {{ display:inline-flex; gap:8px; align-items:center; padding:8px 14px; border-radius:999px;
    font-size:13px; font-weight:600; margin-bottom:8px; }}
  .verdict.ok {{ background:var(--ok-bg); color:var(--ok); }}
  .verdict.warn {{ background:var(--warn-bg); color:var(--warn); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }}
  .stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .stat b {{ display:block; font-size:21px; font-variant-numeric:tabular-nums; }}
  .stat span {{ color:var(--muted); font-size:12.5px; }}
  figure {{ margin:0; }}
  .shot figcaption {{ display:flex; flex-direction:column; gap:2px; margin-bottom:8px; }}
  .shot figcaption span {{ color:var(--muted); font-size:12.5px; }}
  .shot img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:8px; background:#fff; }}
  .chart {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 12px 4px; }}
  .chart figcaption {{ font-size:13px; font-weight:600; margin-bottom:4px; }}
  .chart svg {{ width:100%; height:auto; }}
  .chart .plot-bg {{ fill:#fcfcfd; }}
  .chart .grid {{ stroke:var(--line); stroke-width:1; }}
  .chart .tick {{ font-size:9.5px; fill:var(--muted); }}
  .chart .axis-label {{ font-size:10.5px; fill:var(--muted); }}
  .empty-box {{ height:230px; display:grid; place-items:center; color:var(--muted);
    border:1px dashed var(--line); border-radius:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  td code {{ font-size:12.5px; }}
  tr.gate-fail td {{ background:var(--warn-bg); }}
  .swatch {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:7px;
    border:1px solid rgba(0,0,0,.15); vertical-align:-1px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 2px; }}
  .chip {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }}
  .chip i {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}
  .dl {{ display:flex; flex-direction:column; gap:2px; padding:14px 16px; border-radius:10px;
    border:1px solid var(--line); background:var(--card); text-decoration:none; color:var(--ink); }}
  .dl:hover {{ border-color:var(--accent); }}
  .dl strong {{ font-size:14px; }}
  .dl span {{ color:var(--muted); font-size:12.5px; }}
  .note {{ color:var(--muted); font-size:13px; margin:8px 0 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>곡선 추출 비교 리포트</h1>
  <p class="sub"><code>{html.escape(summary['panel_id'])}</code> · 생성 {html.escape(summary['generated_at_utc'])}
     · 모드 <b>{summary['mode']}</b></p>
  <div class="verdict {verdict_class}">{html.escape(verdict_text)}</div>

  <div class="grid4" style="margin-top:14px">
    <div class="stat"><b>{metrics['glyph_count']}</b><span>검출 글리프</span></div>
    <div class="stat"><b>{metrics['series_count']}</b><span>계열 (색 × 채움)</span></div>
    <div class="stat"><b>{len(rows)}</b><span>colorlab 데이터점</span></div>
    <div class="stat"><b>{sum(len(e['points']) for e in legacy.values())}</b><span>기존 파이프라인 점 (리샘플)</span></div>
  </div>

  <h2>1. 곡선 따온 결과 — 이미지 위 비교</h2>
  <div class="grid2">
    {image_card("기존 파이프라인", "final_curve_traces_crop_px.json 의 트레이스", legacy_overlay)}
    {image_card("colorlab (신규)", "글리프 중심 = 데이터점, 십자=채움 / 기울인 십자=빈 마커", colorlab_overlay)}
  </div>
  <div class="grid2" style="margin-top:16px">
    {image_card("bbox 마스킹 후 곡선만 남은 이미지", "범례·주석 bbox 제거 + 축/눈금/격자 구조 잉크 제거", curve_only)}
    {image_card("색 레이어 맵", "언믹싱된 잉크의 팔레트 최근접 배정", layer_map)}
  </div>
  {image_card("원본 plot_crop", "비교 기준", crop) if crop else ""}

  <h2>2. 데이터 공간 비교 — {html.escape(left_label)}</h2>
  {_legend_html(_colorlab_entries(series_list, rows, "left"))}
  <div class="grid2">{"".join(charts_left)}</div>

  <h2>3. 데이터 공간 비교 — {html.escape(right_label)}</h2>
  {_legend_html(_colorlab_entries(series_list, rows, "right"))}
  <div class="grid2">{"".join(charts_right)}</div>

  <h2>4. CSV 다운로드</h2>
  <div class="grid2">{download_html}</div>
  <p class="note">링크는 파일에 그대로 들어있는 데이터 URI입니다. 이 HTML을 로컬 브라우저로 열면 바로 저장됩니다.</p>

  <h2>5. 검출된 계열</h2>
  <div class="card"><table>
    <thead><tr><th>id</th><th>범례</th><th>마커</th><th>점 수</th><th>형태</th><th>축 · 물리량</th><th>플래그</th></tr></thead>
    <tbody>{series_rows}</tbody>
  </table></div>

  <h2>6. 팔레트 (범례 시드 → 실제 잉크색 정련)</h2>
  <div class="card"><table>
    <thead><tr><th>#</th><th>범례</th><th>시드 RGB</th><th>정련 RGB</th><th>최근접 타 색 ΔE</th><th>레이어 px</th></tr></thead>
    <tbody>{palette_rows}</tbody>
  </table></div>

  <h2>7. QC 게이트</h2>
  <div class="card"><table>
    <thead><tr><th>게이트</th><th>값</th><th>임계</th><th>판정</th></tr></thead>
    <tbody>{gate_rows}</tbody>
  </table></div>

  <h2>8. 실행 파라미터</h2>
  <div class="card"><pre style="margin:0;overflow-x:auto;font-size:12px">{html.escape(json.dumps(summary['params'], indent=2, ensure_ascii=False))}</pre></div>
</div>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")
    return out_path

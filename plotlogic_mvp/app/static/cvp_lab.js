const $ = (id) => document.getElementById(id);

let state = { fixtures: [], runs: [], near_duplicate_groups: [] };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function issueList(items) {
  const issues = (items || []).filter(Boolean);
  if (!issues.length) return '<span class="muted">none</span>';
  return `<ul class="issue-list">${issues.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function bbox(raw) {
  if (!Array.isArray(raw) || raw.length !== 4) return null;
  const out = raw.map((value) => Number(value));
  if (out.some((value) => !Number.isFinite(value))) return null;
  if (out[2] <= out[0] || out[3] <= out[1]) return null;
  return out;
}

function pointList(raw) {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((point) => Array.isArray(point) && point.length >= 2 ? [Number(point[0]), Number(point[1])] : null)
    .filter((point) => point && Number.isFinite(point[0]) && Number.isFinite(point[1]));
}

function tracePoints(trace) {
  if (!trace || typeof trace !== "object") return [];
  for (const key of ["points_crop_px", "points_panel_px", "points_px", "polyline_crop_px", "polyline_panel_px", "skeleton_points_crop_px"]) {
    const points = pointList(trace[key]);
    if (points.length) return points;
  }
  return pointList(trace.points);
}

function offsetBox(raw, offset) {
  const box = bbox(raw);
  if (!box) return null;
  return [box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1]];
}

const GEOMETRY_COLORS = {
  plot: "#1f6feb",
  axis: "#d9480f",
  tick: "#7048e8",
  legend: "#0ca678",
  exclusion: "#e03131",
};

function makeGeometryBox(kind, raw, label) {
  const box = bbox(raw);
  return box ? { kind, bbox: box, label } : null;
}

function buildGeometryBoxes(result) {
  const inputs = result.lab_inputs || {};
  const plot = bbox(inputs.plot_panel_override?.gt_bbox_xyxy);
  const boxes = [
    makeGeometryBox("plot", plot, "plot frame"),
    makeGeometryBox("axis", inputs.axes_panel_override?.axis_left_bbox_xyxy, "left axis"),
    makeGeometryBox("axis", inputs.axes_panel_override?.axis_bottom_bbox_xyxy, "bottom axis"),
    makeGeometryBox("axis", inputs.axes_panel_override?.axis_right_bbox_xyxy, "right axis"),
  ].filter(Boolean);

  for (const axis of inputs.ticks_panel_override?.axes || []) {
    const role = axis.axis_role || "axis";
    for (const candidate of axis.tick_label_candidates || []) {
      boxes.push(makeGeometryBox("tick", candidate.bbox_xyxy, `${role} ${candidate.text || ""}`));
    }
  }

  const offset = plot ? [plot[0], plot[1]] : [0, 0];
  for (const item of inputs.legend_items_crop_px || []) {
    boxes.push(makeGeometryBox("legend", offsetBox(item.label_bbox_crop_px, offset), item.legend_label || "legend"));
    boxes.push(makeGeometryBox("legend", offsetBox(item.swatch_bbox_crop_px, offset), "swatch"));
  }
  for (const raw of inputs.exclusion_bboxes_crop_px || []) {
    boxes.push(makeGeometryBox("exclusion", offsetBox(raw, offset), "exclusion"));
  }
  return boxes.filter(Boolean);
}

function geometrySection(result) {
  const inputs = result.lab_inputs || {};
  if (!Object.keys(inputs).length) return "";
  return `
    <details class="mask-details geometry-details">
      <summary>Panel Geometry</summary>
      <div class="lab-geometry-stage">
        <canvas data-geometry-fixture="${escapeHtml(result.fixture_id)}"></canvas>
      </div>
      <div class="lab-geometry-legend">
        ${Object.entries(GEOMETRY_COLORS).map(([kind, color]) => `<span><i style="background:${color}"></i>${escapeHtml(kind)}</span>`).join("")}
      </div>
    </details>
  `;
}

function formatChartValue(value) {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs >= 1000) return Math.round(value).toLocaleString("en-US");
  if (abs >= 10) return Number(value.toFixed(1)).toLocaleString("en-US");
  if (abs >= 1) return Number(value.toFixed(2)).toLocaleString("en-US");
  return Number(value.toFixed(3)).toLocaleString("en-US");
}

function objectArray(value) {
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item)) : [];
}

function buildTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const step = (max - min) / Math.max(1, count - 1);
  return Array.from({ length: count }, (_, index) => min + step * index);
}

function rangeForPoints(points) {
  const ys = points.map((point) => point[1]).filter(Number.isFinite);
  const min = Math.min(...ys);
  const max = Math.max(...ys);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
  if (min === max) return { min: min - 0.5, max: max + 0.5 };
  return { min, max };
}

function linePath(points, scaleX, scaleY) {
  return points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${scaleX(x).toFixed(2)} ${scaleY(y).toFixed(2)}`)
    .join(" ");
}

function rgbToCss(rgb, fallback = "#2563eb") {
  if (!Array.isArray(rgb) || rgb.length < 3) return fallback;
  const values = rgb.slice(0, 3).map((value) => Math.max(0, Math.min(255, Number(value) || 0)));
  return `rgb(${values[0]}, ${values[1]}, ${values[2]})`;
}

function buildAxisTicksFromMapping(axisMapping) {
  const axes = axisMapping?.axes || {};
  const ticksForRole = (role) => objectArray(axes?.[role]?.pairs)
    .map((pair) => {
      const value = Number(pair.value);
      return Number.isFinite(value) ? { value, label: formatChartValue(value) } : null;
    })
    .filter(Boolean)
    .filter((tick, index, arr) => arr.findIndex((candidate) => candidate.value === tick.value) === index)
    .sort((a, b) => a.value - b.value);
  return {
    x: ticksForRole("bottom"),
    leftY: ticksForRole("left"),
    rightY: ticksForRole("right"),
  };
}

function buildAxisLabelsFromMapping(axisMapping) {
  const axes = axisMapping?.axes || {};
  const labelForRole = (role, fallback) => {
    const label = axes?.[role]?.axis_label;
    return typeof label === "string" && label.trim() ? label.trim() : fallback;
  };
  return {
    x: labelForRole("bottom", "x_data"),
    leftY: labelForRole("left", "left_y"),
    rightY: labelForRole("right", "right_y"),
  };
}

function buildInteractiveSeriesRows(result) {
  const seriesRows = objectArray(result.curve_series_data?.series)
    .map((row) => {
      // 시리즈·조각 구분: 좌표 유지, 빈 구간 연결 방지
      const segments = (row.segments || [row]).map((segment) => pointList(segment.points_data));
      const points = segments.flat();
      if (points.length < 2) return null;
      const traceId = typeof row.trace_id === "string" ? row.trace_id : "trace";
      return {
        id: traceId,
        seriesId: row.series_id,
        label: [row.legend_label, row.curve_role].filter(Boolean).join(" / ") || traceId,
        axisBinding: typeof row.axis_binding === "string" ? row.axis_binding : "left_y",
        color: "#2563eb",
        points,
        segments,
      };
    })
    .filter(Boolean);

  const traceRows = objectArray(result.curve_traces_crop_px?.traces);
  const colorByTrace = new Map();
  for (const trace of traceRows) {
    if (typeof trace.trace_id === "string") colorByTrace.set(trace.trace_id, rgbToCss(trace.series_color_rgb));
    if (typeof trace.series_id === "string") colorByTrace.set(trace.series_id, rgbToCss(trace.series_color_rgb));
  }
  for (const series of seriesRows) {
    series.color = colorByTrace.get(series.id) || colorByTrace.get(series.seriesId) || series.color;
  }
  return seriesRows;
}

function buildInteractiveChartModel(result) {
  const seriesRows = buildInteractiveSeriesRows(result);
  if (!seriesRows.length) return null;

  const axisTicks = buildAxisTicksFromMapping(result.axis_mapping);
  const labels = buildAxisLabelsFromMapping(result.axis_mapping);
  const allPoints = seriesRows.flatMap((series) => series.points);
  const xs = [...allPoints.map((point) => point[0]), ...(axisTicks.x || []).map((tick) => tick.value)].filter(Number.isFinite);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null;

  const width = 980;
  const height = 520;
  const hasRightAxis = seriesRows.some((series) => series.axisBinding === "right_y");
  const pad = { left: 70, right: hasRightAxis ? 76 : 30, top: 26, bottom: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xRange = maxX - minX || 1;
  const leftSeries = seriesRows.filter((series) => series.axisBinding !== "right_y");
  const rightSeries = seriesRows.filter((series) => series.axisBinding === "right_y");
  const leftRange = rangeForPoints([...(leftSeries.length ? leftSeries : seriesRows).flatMap((series) => series.points), ...(axisTicks.leftY || []).map((tick) => [minX, tick.value])]);
  const rightRange = rangeForPoints([...(rightSeries.length ? rightSeries : seriesRows).flatMap((series) => series.points), ...(axisTicks.rightY || []).map((tick) => [minX, tick.value])]);
  const scaleX = (value) => pad.left + ((value - minX) / xRange) * plotW;
  const scaleYLeft = (value) => pad.top + ((leftRange.max - value) / (leftRange.max - leftRange.min || 1)) * plotH;
  const scaleYRight = (value) => pad.top + ((rightRange.max - value) / (rightRange.max - rightRange.min || 1)) * plotH;
  const scaleYForSeries = (series) => hasRightAxis && series.axisBinding === "right_y" ? scaleYRight : scaleYLeft;
  const xTicks = axisTicks.x?.length ? axisTicks.x : buildTicks(minX, maxX).map((value) => ({ value, label: formatChartValue(value) }));
  const leftTicks = axisTicks.leftY?.length ? axisTicks.leftY : buildTicks(leftRange.min, leftRange.max).map((value) => ({ value, label: formatChartValue(value) }));
  const rightTicks = axisTicks.rightY?.length ? axisTicks.rightY : buildTicks(rightRange.min, rightRange.max).map((value) => ({ value, label: formatChartValue(value) }));
  return {
    seriesRows,
    labels,
    width,
    height,
    hasRightAxis,
    pad,
    plotW,
    plotH,
    scaleX,
    scaleYLeft,
    scaleYRight,
    scaleYForSeries,
    xTicks,
    leftTicks,
    rightTicks,
  };
}

function interactiveChartSection(result) {
  const model = buildInteractiveChartModel(result);
  if (!model) return "";
  const {
    seriesRows,
    labels,
    width,
    height,
    hasRightAxis,
    pad,
    plotW,
    plotH,
    scaleX,
    scaleYLeft,
    scaleYRight,
    scaleYForSeries,
    xTicks,
    leftTicks,
    rightTicks,
  } = model;

  const xTickSvg = xTicks.map((tick) => `<g><line x1="${scaleX(tick.value).toFixed(2)}" x2="${scaleX(tick.value).toFixed(2)}" y1="${pad.top}" y2="${pad.top + plotH}" stroke="#e2e8f0"></line><text x="${scaleX(tick.value).toFixed(2)}" y="${height - 28}" text-anchor="middle" font-size="11" fill="#64748b">${escapeHtml(tick.label)}</text></g>`).join("");
  const leftTickSvg = leftTicks.map((tick) => `<g><line x1="${pad.left}" x2="${pad.left + plotW}" y1="${scaleYLeft(tick.value).toFixed(2)}" y2="${scaleYLeft(tick.value).toFixed(2)}" stroke="#e2e8f0"></line><text x="${pad.left - 10}" y="${(scaleYLeft(tick.value) + 4).toFixed(2)}" text-anchor="end" font-size="11" fill="#64748b">${escapeHtml(tick.label)}</text></g>`).join("");
  const rightTickSvg = hasRightAxis ? rightTicks.map((tick) => `<text x="${pad.left + plotW + 10}" y="${(scaleYRight(tick.value) + 4).toFixed(2)}" text-anchor="start" font-size="11" fill="#b45309">${escapeHtml(tick.label)}</text>`).join("") : "";
  const seriesSvg = seriesRows.map((series) => (series.segments || [series.points]).map((points) => `<path data-series-id="${escapeHtml(series.id)}" d="${linePath(points, scaleX, scaleYForSeries(series))}" fill="none" stroke="${escapeHtml(series.color)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"></path>`).join("")).join("");
  const legend = seriesRows.map((series) => `
    <label class="lab-chart-series">
      <input type="checkbox" data-series-id="${escapeHtml(series.id)}" checked>
      <i style="background:${escapeHtml(series.color)}"></i>
      ${escapeHtml(series.label)}
      <small>${escapeHtml(series.axisBinding)} · ${series.points.length}</small>
    </label>
  `).join("");

  return `
    <details class="mask-details interactive-details">
      <summary>Interactive Extracted Chart (${seriesRows.length})</summary>
      <div class="lab-chart-toolbar">
        <button type="button" data-chart-action="all">Select all</button>
        <button type="button" data-chart-action="none">Clear all</button>
        <span class="muted">Hover the chart to inspect nearest extracted point.</span>
      </div>
      <div class="lab-chart-frame">
        <svg data-lab-chart-fixture="${escapeHtml(result.fixture_id)}" viewBox="0 0 ${width} ${height}" role="img" aria-label="interactive extracted chart">
          <rect width="${width}" height="${height}" fill="#ffffff"></rect>
          <rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" fill="#f8fafc" stroke="#cbd5e1"></rect>
          ${xTickSvg}${leftTickSvg}${rightTickSvg}${seriesSvg}
          <text x="${pad.left + plotW / 2}" y="${height - 10}" text-anchor="middle" font-size="12" fill="#334155">${escapeHtml(labels.x)}</text>
          <text x="20" y="${pad.top + plotH / 2}" text-anchor="middle" font-size="12" fill="#334155" transform="rotate(-90 20 ${pad.top + plotH / 2})">${escapeHtml(labels.leftY)}</text>
          ${hasRightAxis ? `<text x="${width - 20}" y="${pad.top + plotH / 2}" text-anchor="middle" font-size="12" fill="#b45309" transform="rotate(90 ${width - 20} ${pad.top + plotH / 2})">${escapeHtml(labels.rightY)}</text>` : ""}
          <g data-chart-hover class="hidden"></g>
        </svg>
      </div>
      <div class="lab-chart-legend">${legend}</div>
    </details>
  `;
}

function hydrateGeometryCanvases(run) {
  const resultsById = new Map((run?.results || []).map((result) => [result.fixture_id, result]));
  document.querySelectorAll("canvas[data-geometry-fixture]").forEach((canvas) => {
    const result = resultsById.get(canvas.dataset.geometryFixture);
    if (!result) return;
    const fixture = fixtureById(result.fixture_id);
    const imageUrl = fixture.image_url;
    if (!imageUrl) return;
    const boxes = buildGeometryBoxes(result);
    const img = new Image();
    const draw = () => {
      const container = canvas.parentElement;
      if (canvas.closest("details") && !canvas.closest("details").open) return;
      const stageWidth = Math.max(260, (container?.clientWidth || 720) - 2);
      let displayWidth = Math.min(img.naturalWidth, stageWidth);
      let displayHeight = displayWidth * (img.naturalHeight / img.naturalWidth);
      const maxHeight = 560;
      if (displayHeight > maxHeight) {
        displayHeight = maxHeight;
        displayWidth = displayHeight * (img.naturalWidth / img.naturalHeight);
      }
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(displayWidth * dpr);
      canvas.height = Math.round(displayHeight * dpr);
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
      const sx = displayWidth / img.naturalWidth;
      const sy = displayHeight / img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, displayWidth, displayHeight);
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
      ctx.font = "11px Inter, sans-serif";
      ctx.lineWidth = 2;
      for (const item of boxes) {
        const [x1, y1, x2, y2] = item.bbox;
        const color = GEOMETRY_COLORS[item.kind] || "#111827";
        const x = x1 * sx;
        const y = y1 * sy;
        const w = (x2 - x1) * sx;
        const h = (y2 - y1) * sy;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x, y, w, h);
        if (item.label) {
          const label = String(item.label).slice(0, 42);
          const labelX = Math.max(0, Math.min(x, displayWidth - 20));
          const labelY = Math.max(0, y - 16);
          const tw = Math.min(ctx.measureText(label).width + 8, displayWidth - labelX);
          ctx.globalAlpha = 0.88;
          ctx.fillRect(labelX, labelY, tw, 15);
          ctx.globalAlpha = 1;
          ctx.fillStyle = "#fff";
          ctx.fillText(label, labelX + 4, labelY + 11);
        }
      }
    };
    img.addEventListener("load", draw);
    canvas.closest("details")?.addEventListener("toggle", draw);
    window.addEventListener("resize", draw);
    img.src = imageUrl;
  });
}

function hydrateInteractiveCharts(run) {
  const resultsById = new Map((run?.results || []).map((result) => [result.fixture_id, result]));
  document.querySelectorAll("svg[data-lab-chart-fixture]").forEach((svg) => {
    const result = resultsById.get(svg.dataset.labChartFixture);
    const model = result ? buildInteractiveChartModel(result) : null;
    if (!model) return;
    const details = svg.closest("details");
    const hoverGroup = svg.querySelector("[data-chart-hover]");
    const checkboxes = Array.from(details?.querySelectorAll("input[data-series-id]") || []);
    const paths = Array.from(svg.querySelectorAll("path[data-series-id]"));

    const visibleIds = () => new Set(checkboxes.filter((input) => input.checked).map((input) => input.dataset.seriesId));
    const syncVisibility = () => {
      const visible = visibleIds();
      paths.forEach((path) => {
        path.classList.toggle("hidden", !visible.has(path.dataset.seriesId));
      });
    };

    checkboxes.forEach((input) => input.addEventListener("change", syncVisibility));
    details?.querySelector("[data-chart-action='all']")?.addEventListener("click", () => {
      checkboxes.forEach((input) => { input.checked = true; });
      syncVisibility();
    });
    details?.querySelector("[data-chart-action='none']")?.addEventListener("click", () => {
      checkboxes.forEach((input) => { input.checked = false; });
      syncVisibility();
      if (hoverGroup) {
        hoverGroup.classList.add("hidden");
        hoverGroup.innerHTML = "";
      }
    });

    svg.addEventListener("pointermove", (event) => {
      if (!hoverGroup) return;
      const rect = svg.getBoundingClientRect();
      const sx = ((event.clientX - rect.left) / rect.width) * model.width;
      const sy = ((event.clientY - rect.top) / rect.height) * model.height;
      const visible = visibleIds();
      let best = null;
      let bestDist = Infinity;
      for (const series of model.seriesRows) {
        if (!visible.has(series.id)) continue;
        const stride = Math.max(1, Math.floor(series.points.length / 250));
        const scaleY = model.scaleYForSeries(series);
        for (let index = 0; index < series.points.length; index += stride) {
          const point = series.points[index];
          const px = model.scaleX(point[0]);
          const py = scaleY(point[1]);
          const dist = (px - sx) ** 2 + (py - sy) ** 2;
          if (dist < bestDist) {
            bestDist = dist;
            best = { series, point, sx: px, sy: py };
          }
        }
      }
      if (!best || bestDist >= 700) {
        hoverGroup.classList.add("hidden");
        hoverGroup.innerHTML = "";
        return;
      }
      const boxX = Math.min(model.width - 246, best.sx + 12);
      const boxY = Math.max(10, best.sy - 42);
      hoverGroup.classList.remove("hidden");
      hoverGroup.innerHTML = `
        <circle cx="${best.sx.toFixed(2)}" cy="${best.sy.toFixed(2)}" r="5" fill="${escapeHtml(best.series.color)}" stroke="#0f172a"></circle>
        <rect x="${boxX.toFixed(2)}" y="${boxY.toFixed(2)}" width="234" height="56" rx="8" fill="#0f172a" opacity="0.92"></rect>
        <text x="${(boxX + 10).toFixed(2)}" y="${(boxY + 22).toFixed(2)}" font-size="11" fill="#fff">${escapeHtml(best.series.label).slice(0, 44)}</text>
        <text x="${(boxX + 10).toFixed(2)}" y="${(boxY + 41).toFixed(2)}" font-size="11" fill="#cbd5e1">x=${formatChartValue(best.point[0])}, y=${formatChartValue(best.point[1])}</text>
      `;
    });
    svg.addEventListener("pointerleave", () => {
      if (!hoverGroup) return;
      hoverGroup.classList.add("hidden");
      hoverGroup.innerHTML = "";
    });
    syncVisibility();
  });
}

function duplicateGroupFor(fixtureId) {
  return (state.near_duplicate_groups || []).find((group) => (
    (group.fixture_ids || []).includes(fixtureId)
  ));
}

function duplicateBadge(fixtureId) {
  const group = duplicateGroupFor(fixtureId);
  if (!group) return "";
  const others = (group.fixture_ids || []).filter((id) => id !== fixtureId);
  const pairText = (group.pairs || [])
    .map((pair) => `${pair.left_fixture_id} <-> ${pair.right_fixture_id}: d=${pair.hash_distance}`)
    .join("; ");
  return `
    <div class="duplicate-badge" title="${escapeHtml(pairText)}">
      near duplicate: ${escapeHtml(others.join(", "))}
    </div>
  `;
}

function renderFixtures() {
  $("fixtureCount").textContent = `${state.fixtures.length}`;
  if (!state.fixtures.length) {
    $("fixtureList").innerHTML = '<div class="empty-state">Import existing C-V-P targets to create fixtures.</div>';
    return;
  }
  $("fixtureList").innerHTML = state.fixtures.map((fixture) => {
    const source = fixture.source || {};
    const imageInfo = fixture.image_info || {};
    return `
      <article class="fixture-card">
        <img src="${escapeHtml(fixture.image_url)}" alt="">
        <div>
          <div class="fixture-title">${escapeHtml(fixture.fixture_id)}</div>
          <div class="muted">${escapeHtml(imageInfo.size_label || "unknown size")}</div>
          <div class="muted">paper ${escapeHtml(source.paper_id ?? "")} · job ${escapeHtml(source.job_id ?? "")}</div>
          <div class="muted">${escapeHtml(source.panel_id || "")}</div>
          ${duplicateBadge(fixture.fixture_id)}
        </div>
      </article>
    `;
  }).join("");
}

function renderRuns() {
  $("runCount").textContent = `${state.runs.length}`;
  if (!state.runs.length) {
    $("runList").innerHTML = '<div class="empty-state">No lab runs yet.</div>';
    renderResults(null);
    return;
  }
  $("runList").innerHTML = state.runs.map((run, index) => `
    <button class="run-item ${index === 0 ? "selected" : ""}" type="button" data-run-id="${escapeHtml(run.run_id)}">
      <span>${escapeHtml(run.run_id)}</span>
      <small>${escapeHtml(run.ok_count)} ok · ${escapeHtml(run.failed_count)} failed · ${escapeHtml(run.fixture_count)} fixtures</small>
    </button>
  `).join("");
  for (const button of $("runList").querySelectorAll("[data-run-id]")) {
    button.addEventListener("click", async () => {
      for (const item of $("runList").querySelectorAll(".run-item")) item.classList.remove("selected");
      button.classList.add("selected");
      const payload = await api(`/api/cvp-lab/runs/${encodeURIComponent(button.dataset.runId)}`);
      renderResults(payload.run);
    });
  }
  renderResults(state.runs[0]);
}

function fixtureById(id) {
  return state.fixtures.find((fixture) => fixture.fixture_id === id) || {};
}

function imageBlock(label, url) {
  if (!url) return "";
  return `
    <figure class="lab-image-block">
      <figcaption>${escapeHtml(label)}</figcaption>
      <img src="${escapeHtml(url)}" alt="">
    </figure>
  `;
}

function maskSection(title, masks) {
  const items = masks || [];
  if (!items.length) return "";
  return `
    <details class="mask-details">
      <summary>${escapeHtml(title)} (${items.length})</summary>
      <div class="mask-grid">
        ${items.map((mask) => `
          <figure class="mask-tile">
            <img src="${escapeHtml(mask.url)}" alt="">
            <figcaption>
              <strong>${escapeHtml(mask.series_id)}</strong>
              <span>${escapeHtml(mask.label || "")}</span>
              <small>${escapeHtml(mask.filename)}</small>
            </figcaption>
          </figure>
        `).join("")}
      </div>
    </details>
  `;
}

function llmMaskCorrectionSection(result) {
  const summary = result.llm_mask_correction_summary || {};
  const correctedMasks = result.llm_corrected_mask_artifacts || [];
  const cleanedMasks = result.cleaned_mask_artifacts || [];
  const reconstructedMasks = result.reconstructed_mask_artifacts || [];
  const comparisonUrl = result.mask_reconstruction_comparison_url || result.llm_mask_correction_comparison_url;
  const hasArtifacts = Object.keys(summary).length || correctedMasks.length || comparisonUrl;
  const status = summary.status || result.mask_correction_status || "not run";
  return `
    <details class="mask-details llm-mask-correction-details" ${hasArtifacts ? "open" : ""}>
      <summary>LLM Mask Correction + Reconstruction (${escapeHtml(status)})</summary>
      <div class="visual-qa-note">
        LLM writes a bbox correction plan. The program then removes tiny components and bridges short breaks into reconstructed masks. These do not overwrite extraction inputs yet.
      </div>
      <div class="lab-inline-actions">
        <button class="icon-button run-mask-correction-btn" type="button" data-run-id="${escapeHtml(result.run_id)}" data-fixture-id="${escapeHtml(result.fixture_id)}">
          Run LLM mask correction
        </button>
        <span class="muted">
          corrected ${escapeHtml(summary.corrected_count ?? result.mask_correction_corrected_count ?? 0)}
          · rejected ${escapeHtml(summary.rejected_count ?? result.mask_correction_rejected_count ?? 0)}
          · removed dots ${escapeHtml(summary.removed_component_count ?? result.mask_reconstruction_removed_component_count ?? 0)}
          · bridges ${escapeHtml(summary.bridge_count ?? result.mask_reconstruction_bridge_count ?? 0)}
        </span>
      </div>
      <div class="lab-image-row compact">
        ${imageBlock("LLM contact sheet", result.llm_mask_correction_contact_sheet_url)}
        ${imageBlock("raw / corrected / cleaned / reconstructed", comparisonUrl)}
      </div>
      ${maskSection("LLM corrected masks", correctedMasks)}
      ${maskSection("Cleaned masks", cleanedMasks)}
      ${maskSection("Reconstructed masks", reconstructedMasks)}
    </details>
  `;
}

function rgbText(rgb) {
  if (!Array.isArray(rgb) || rgb.length < 3) return "";
  return rgb.slice(0, 3).map((value) => Math.round(Number(value) || 0)).join(", ");
}

function colorChip(rgb) {
  const css = rgbToCss(rgb, "#d1d5db");
  return `<i class="legend-color-chip" style="background:${escapeHtml(css)}"></i>`;
}

function legendColorSeedsSection(result) {
  const legendItems = result.lab_inputs?.legend_items_crop_px || [];
  const candidates = objectArray(result.series_candidates?.items);
  if (!legendItems.length && !candidates.length) return "";

  const candidateByLegendIndex = new Map();
  const candidateByLabel = new Map();
  for (const candidate of candidates) {
    const match = candidate.legend_match;
    if (match && Number.isInteger(match.legend_item_index)) {
      candidateByLegendIndex.set(match.legend_item_index, candidate);
    }
    if (typeof candidate.legend_label === "string") {
      candidateByLabel.set(candidate.legend_label, candidate);
    }
  }

  const rows = legendItems.map((item, index) => {
    const candidate = candidateByLegendIndex.get(index) || candidateByLabel.get(item.legend_label) || null;
    const match = candidate?.legend_match || {};
    const seed = item.swatch_color_rgb;
    const target = candidate?.target_color_rgb;
    return `
      <tr>
        <td>${escapeHtml(item.legend_label || `legend ${index + 1}`)}</td>
        <td><span class="legend-color-cell">${colorChip(seed)}${escapeHtml(rgbText(seed))}</span></td>
        <td><span class="legend-color-cell">${colorChip(target)}${escapeHtml(rgbText(target) || "unmatched")}</span></td>
        <td>${escapeHtml(candidate?.series_id || "")}</td>
        <td>${escapeHtml(match.color_distance ?? candidate?.color_distance ?? "")}</td>
        <td>${escapeHtml(JSON.stringify(item.swatch_bbox_crop_px || []))}</td>
        <td>${escapeHtml(JSON.stringify(item.label_bbox_crop_px || []))}</td>
      </tr>
    `;
  }).join("");

  const unmatched = candidates
    .filter((candidate) => candidate.legend_label && !legendItems.some((item) => item.legend_label === candidate.legend_label))
    .map((candidate) => `
      <tr>
        <td>${escapeHtml(candidate.legend_label || candidate.series_id || "")}</td>
        <td></td>
        <td><span class="legend-color-cell">${colorChip(candidate.target_color_rgb)}${escapeHtml(rgbText(candidate.target_color_rgb))}</span></td>
        <td>${escapeHtml(candidate.series_id || "")}</td>
        <td>${escapeHtml(candidate.color_distance ?? "")}</td>
        <td></td>
        <td></td>
      </tr>
    `).join("");

  return `
    <details class="mask-details legend-seeds-details">
      <summary>Legend Color Seeds (${legendItems.length})</summary>
      <div class="legend-seed-table-wrap">
        <table class="legend-seed-table">
          <thead>
            <tr>
              <th>Legend</th>
              <th>Swatch seed RGB</th>
              <th>Matched target RGB</th>
              <th>Series</th>
              <th>Distance</th>
              <th>Swatch bbox</th>
              <th>Label bbox</th>
            </tr>
          </thead>
          <tbody>${rows}${unmatched}</tbody>
        </table>
      </div>
    </details>
  `;
}

function visualQASection(result) {
  const summary = result.visual_qa_summary || {};
  const maskItems = objectArray(result.mask_quality_qa?.items)
    .filter((item) => item.severity !== "ok" || (item.issues || []).length)
    .slice(0, 12);
  const traceItems = objectArray(result.trace_overlay_qa?.items)
    .filter((item) => item.severity !== "ok" || (item.issues || []).length)
    .slice(0, 12);
  if (!Object.keys(summary).length && !maskItems.length && !traceItems.length) return "";
  const rows = [
    ...maskItems.map((item) => `
      <tr>
        <td>mask</td>
        <td>${escapeHtml(item.relative_path || item.filename || "")}</td>
        <td>${escapeHtml(item.severity || "")}</td>
        <td>${escapeHtml((item.issues || []).join(", "))}</td>
        <td>${escapeHtml(`components=${item.component_count ?? ""}, largest=${item.largest_component_ratio ?? ""}`)}</td>
      </tr>
    `),
    ...traceItems.map((item) => `
      <tr>
        <td>trace</td>
        <td>${escapeHtml(item.trace_id || item.series_id || "")}</td>
        <td>${escapeHtml(item.severity || "")}</td>
        <td>${escapeHtml((item.issues || []).join(", "))}</td>
        <td>${escapeHtml(`plateau=${item.longest_same_y_run ?? ""}, zigzag=${item.direction_change_count ?? ""}`)}</td>
      </tr>
    `),
  ].join("");
  return `
    <details class="mask-details visual-qa-details">
      <summary>Visual QA (${escapeHtml(summary.status || result.visual_qa_status || "not generated")} · ${escapeHtml(summary.issue_count ?? result.visual_qa_issue_count ?? 0)})</summary>
      <div class="visual-qa-note">${escapeHtml(summary.human_review_hint || "QA flags do not change extraction outputs.")}</div>
      <div class="legend-seed-table-wrap">
        <table class="legend-seed-table">
          <thead>
            <tr>
              <th>Kind</th>
              <th>Item</th>
              <th>Severity</th>
              <th>Issues</th>
              <th>Metrics</th>
            </tr>
          </thead>
          <tbody>${rows || '<tr><td colspan="5">No QA flags.</td></tr>'}</tbody>
        </table>
      </div>
    </details>
  `;
}

function renderResults(run) {
  if (!run) {
    $("resultTitle").textContent = "Latest Results";
    $("resultMeta").textContent = "";
    $("resultGrid").innerHTML = '<div class="empty-state">Run the lab to compare extraction artifacts.</div>';
    return;
  }
  $("resultTitle").textContent = `Run ${run.run_id}`;
  $("resultMeta").textContent = `${run.ok_count} ok · ${run.failed_count} failed · ${run.fixture_count} fixtures · ${run.run_dir}`;
  $("resultGrid").innerHTML = (run.results || []).map((result) => {
    const fixture = fixtureById(result.fixture_id);
    const checks = result.checks || {};
    const imageInfo = fixture.image_info || {};
    const artifactLinks = (result.debug_artifacts || []).map((artifact) => (
      `<a href="${escapeHtml(artifact.url)}" target="_blank" rel="noreferrer">${escapeHtml(artifact.label)}</a>`
    )).join("");
    return `
      <article class="lab-result-card">
        <div class="lab-result-header">
          <div>
            <h2>${escapeHtml(result.fixture_id)}</h2>
            <p class="muted">${escapeHtml(result.status)} · ${escapeHtml(result.series_count)} series · ${escapeHtml(imageInfo.size_label || "unknown size")} · palette ${escapeHtml(checks.min_palette_distance ?? "n/a")} · sharp ${escapeHtml(checks.image_laplacian_variance ?? "n/a")}</p>
            ${duplicateBadge(result.fixture_id)}
          </div>
          ${result.csv_url ? `<a class="download-button" href="${escapeHtml(result.csv_url)}">CSV</a>` : ""}
        </div>
        <div class="lab-image-row">
          ${imageBlock("fixture", fixture.image_url)}
          ${imageBlock("overlay", result.overlay_url)}
          ${imageBlock("final trace", result.final_trace_overlay_url)}
          ${imageBlock("raw vs final", result.raw_vs_final_trace_overlay_url)}
          ${imageBlock("color filter", result.color_filter_source_url)}
          ${imageBlock("role recolor", result.role_recolored_crop_url)}
        </div>
        ${geometrySection(result)}
        ${legendColorSeedsSection(result)}
        ${visualQASection(result)}
        ${llmMaskCorrectionSection(result)}
        ${interactiveChartSection(result)}
        ${maskSection("Color masks", result.mask_artifacts)}
        ${maskSection("Role masks", result.role_mask_artifacts)}
        ${result.error ? `<div class="error-box">${escapeHtml(result.error)}</div>` : ""}
        <div class="lab-issues">
          <div><strong>Quality</strong>${issueList(result.quality_issues)}</div>
          <div><strong>Validation</strong>${issueList(result.validation_issues)}</div>
        </div>
        <div class="artifact-links">${artifactLinks}</div>
      </article>
    `;
  }).join("");
  hydrateGeometryCanvases(run);
  hydrateInteractiveCharts(run);
  hydrateMaskCorrectionButtons(run);
}

function hydrateMaskCorrectionButtons(run) {
  for (const button of document.querySelectorAll(".run-mask-correction-btn")) {
    button.addEventListener("click", async () => {
      const runId = button.dataset.runId;
      const fixtureId = button.dataset.fixtureId;
      button.disabled = true;
      $("labStatus").textContent = `Running LLM mask correction for ${fixtureId}...`;
      try {
        const payload = await api(`/api/cvp-lab/runs/${encodeURIComponent(runId)}/results/${encodeURIComponent(fixtureId)}/mask-correction`, {
          method: "POST",
        });
        $("labStatus").textContent = `Mask correction ${payload.summary.status || "complete"} for ${fixtureId}.`;
        renderResults(payload.run || run);
        await refresh();
      } catch (err) {
        $("labStatus").textContent = err.message;
        button.disabled = false;
      }
    });
  }
}

async function refresh() {
  state = await api("/api/cvp-lab");
  renderFixtures();
  renderRuns();
}

async function doImport() {
  $("labStatus").textContent = "Importing fixtures...";
  $("importBtn").disabled = true;
  try {
    const payload = await api("/api/cvp-lab/import-existing", { method: "POST" });
    $("labStatus").textContent = `Imported ${payload.imported_count} fixtures.`;
    await refresh();
  } finally {
    $("importBtn").disabled = false;
  }
}

async function doImportPlotlogic() {
  $("labStatus").textContent = "Importing PlotLogic phase1 dual-y fixtures...";
  $("importPlotlogicBtn").disabled = true;
  try {
    const payload = await api("/api/cvp-lab/import-plotlogic-phase1", { method: "POST" });
    $("labStatus").textContent = `Imported ${payload.imported_count} PlotLogic fixtures.`;
    await refresh();
  } finally {
    $("importPlotlogicBtn").disabled = false;
  }
}

async function doUploadFixtures(event) {
  event.preventDefault();
  const input = $("fixtureImageInput");
  if (!input.files.length) {
    $("labStatus").textContent = "Choose one or more chart images.";
    return;
  }
  const body = new FormData();
  for (const file of input.files) body.append("files", file);
  $("labStatus").textContent = "Adding fixture images...";
  $("uploadFixtureBtn").disabled = true;
  try {
    const payload = await api("/api/cvp-lab/fixtures", { method: "POST", body });
    input.value = "";
    $("labStatus").textContent = `Added ${payload.imported_count} images.`;
    await refresh();
  } finally {
    $("uploadFixtureBtn").disabled = false;
  }
}

async function doRun() {
  $("labStatus").textContent = "Running lab...";
  $("runBtn").disabled = true;
  try {
    const payload = await api("/api/cvp-lab/run", { method: "POST" });
    $("labStatus").textContent = `Run ${payload.run.run_id} complete.`;
    await refresh();
  } finally {
    $("runBtn").disabled = false;
  }
}

$("importBtn").addEventListener("click", () => doImport().catch((err) => {
  $("labStatus").textContent = err.message;
  $("importBtn").disabled = false;
}));

$("importPlotlogicBtn").addEventListener("click", () => doImportPlotlogic().catch((err) => {
  $("labStatus").textContent = err.message;
  $("importPlotlogicBtn").disabled = false;
}));

$("runBtn").addEventListener("click", () => doRun().catch((err) => {
  $("labStatus").textContent = err.message;
  $("runBtn").disabled = false;
}));

$("fixtureUploadForm").addEventListener("submit", (event) => doUploadFixtures(event).catch((err) => {
  $("labStatus").textContent = err.message;
  $("uploadFixtureBtn").disabled = false;
}));

refresh().catch((err) => {
  $("labStatus").textContent = err.message;
});

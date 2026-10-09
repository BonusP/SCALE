const COLORS = {
  plot: "#1f6feb",
  axis: "#d9480f",
  axisLabel: "#f59f00",
  tick: "#7048e8",
  tickLabel: "#ae3ec9",
  legend: "#0ca678",
  annotation: "#e03131",
  exclusion: "#e03131",
  trace: "#0b7285",
  corrected: "#868e96",
  ocr: "#c2255c",
};

const LABELS = {
  plot: "plot frame",
  axis: "axis line",
  axisLabel: "axis label",
  tick: "tick",
  tickLabel: "tick label",
  legend: "legend",
  annotation: "annotation",
  exclusion: "exclusion",
  trace: "trace",
  corrected: "current pipeline (dashed)",
  ocr: "OCR text",
};

const el = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function show(id, visible) {
  el(id).classList.toggle("hidden", !visible);
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function routeParts() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return { jobId: parts[1], chartId: parts.slice(2).join("/") };
}

function bbox(raw) {
  if (!Array.isArray(raw) || raw.length !== 4) return null;
  const out = raw.map((v) => Number(v));
  if (out.some((v) => !Number.isFinite(v))) return null;
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
  if (Array.isArray(trace.points)) {
    const points = pointList(trace.points);
    if (points.length) return points;
  }
  return [];
}

function makeBox(kind, raw, label, meta = "") {
  const box = bbox(raw);
  if (!box) return null;
  return { kind, bbox: box, label, meta };
}

function buildPanelBoxes(panelVision) {
  const boxes = [];
  if (!panelVision) return boxes;
  boxes.push(makeBox("plot", panelVision.plot_frame_bbox_panel_px, "plot frame"));
  boxes.push(makeBox("legend", panelVision.legend_region_bbox_panel_px, "legend region"));
  for (const axis of panelVision.axes || []) {
    const role = axis.axis_role || "axis";
    boxes.push(makeBox("axis", axis.axis_line_bbox_panel_px, `${role} axis`));
    boxes.push(makeBox("axisLabel", axis.label_bbox_panel_px, `${role} label`));
    for (const tick of axis.ticks || []) {
      boxes.push(makeBox("tick", tick.tick_bbox_panel_px, `${role} tick ${tick.label_text || ""}`));
      boxes.push(makeBox("tickLabel", tick.label_bbox_panel_px, `${role} label ${tick.label_text || ""}`, tick.label_numeric_value));
    }
  }
  for (const item of panelVision.legend_items || []) {
    boxes.push(makeBox("legend", item.label_bbox_panel_px, `legend ${item.legend_label || ""}`));
    boxes.push(makeBox("legend", item.swatch_bbox_panel_px, `swatch ${item.legend_label || ""}`));
  }
  for (const region of panelVision.plot_annotation_regions || []) {
    boxes.push(makeBox("annotation", region.bbox_panel_px, region.annotation_text || "annotation", region.reason || ""));
  }
  return boxes.filter(Boolean);
}

function buildCropBoxes(data) {
  const boxes = [];
  for (const raw of data.exclusion_regions?.bboxes || []) {
    boxes.push(makeBox("exclusion", raw, "exclusion region"));
  }
  for (const item of data.debug_inputs?.legend_items_crop_px || []) {
    boxes.push(makeBox("legend", item.label_bbox_crop_px, `legend ${item.legend_label || ""}`));
    boxes.push(makeBox("legend", item.swatch_bbox_crop_px, `swatch ${item.legend_label || ""}`));
  }
  return boxes.filter(Boolean);
}

function buildCropTraces(data) {
  return (data.curve_traces_crop_px?.traces || []).map((trace, index) => ({
    kind: "trace",
    label: trace.trace_id || trace.series_id || `trace ${index + 1}`,
    points: tracePoints(trace),
  })).filter((trace) => trace.points.length);
}

function renderLegend(container, kinds) {
  container.innerHTML = "";
  for (const kind of kinds) {
    const item = document.createElement("span");
    item.className = "debug-legend-item";
    item.innerHTML = `<i style="background:${COLORS[kind] || "#111"}"></i>${escapeHtml(LABELS[kind] || kind)}`;
    container.appendChild(item);
  }
}

function renderStage(stage, imageUrl, boxes = [], traces = []) {
  stage.innerHTML = "";
  if (!imageUrl) {
    stage.innerHTML = '<div class="empty-state">No image artifact available.</div>';
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.className = "debug-canvas";
  stage.appendChild(canvas);
  const img = new Image();

  const draw = () => {
    if (!img.naturalWidth || !img.naturalHeight) return;
    const stageWidth = Math.max(260, stage.clientWidth - 2 || img.naturalWidth);
    let displayWidth = Math.min(img.naturalWidth, stageWidth);
    let displayHeight = displayWidth * (img.naturalHeight / img.naturalWidth);
    // Fit inside the stage when it has a bounded height (quad layout).
    const maxHeight = Math.min(720, Math.max(120, stage.clientHeight - 22 || 720));
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
      const color = COLORS[item.kind] || "#111";
      const x = x1 * sx;
      const y = y1 * sy;
      const w = (x2 - x1) * sx;
      const h = (y2 - y1) * sy;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.setLineDash(item.dashed ? [5, 4] : []);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      if (item.label) {
        const label = String(item.label).slice(0, 48);
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

    for (const trace of traces) {
      ctx.strokeStyle = COLORS.trace;
      ctx.lineWidth = 2;
      ctx.beginPath();
      trace.points.forEach(([px, py], index) => {
        const x = px * sx;
        const y = py * sy;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  };

  img.addEventListener("load", draw);
  img.src = imageUrl;
  window.addEventListener("resize", draw);
}

function renderOcrGeomTable(geom) {
  const rows = [];
  for (const report of geom.axes_report || []) {
    const ocr = report.ocr || {};
    rows.push([
      report.axis_role,
      report.tick_count,
      `${report.tick_labels_from_ocr} / ${report.tick_count}`,
      ocr.matched != null ? `${ocr.matched} matched, ${(ocr.unmatched_ocr || []).length} left over` : "no ocr",
      (ocr.value_mismatches || []).join(", ") || "—",
    ]);
  }
  const tickRows = [];
  for (const axis of geom.panel_vision?.axes || []) {
    for (const tick of axis.ticks || []) {
      const codexBox = tick.codex_label_bbox_panel_px;
      if (!codexBox && tick.ocr_value_agrees !== false) continue;
      tickRows.push([
        axis.axis_role,
        tick.label_text ?? "",
        tick.ocr_text ?? "—",
        tick.ocr_value_agrees === true ? "ok" : tick.ocr_value_agrees === false ? "MISMATCH" : "—",
        JSON.stringify(codexBox || []),
        JSON.stringify(tick.label_bbox_panel_px || []),
      ]);
    }
  }
  el("ocrGeomTable").innerHTML =
    rowsTable(["Axis", "Codex ticks", "Labels from OCR", "OCR match", "Value mismatch"], rows)
    + `<div class="debug-issues"><strong>Tick label boxes replaced by OCR</strong></div>`
    + rowsTable(["Axis", "Codex label", "OCR text", "Value", "Codex bbox", "OCR bbox"], tickRows.slice(0, 40));
}

// Text boxes come from OCR; the frame, axis lines, tick positions, swatches and
// annotation regions are whatever codex plus the shipped correction produced.
function renderOcrGeom(data) {
  const stage = el("ocrGeomStage");
  const controls = el("ocrGeomControls");
  const note = el("ocrGeomNote");
  const legend = el("ocrGeomLegend");
  controls.innerHTML = "";
  legend.innerHTML = "";
  note.innerHTML = "";
  el("ocrGeomTable").innerHTML = "";

  const geom = data.ocr_geometry;
  if (!geom || geom.error) {
    stage.innerHTML = `<div class="empty-state">${escapeHtml(geom?.error || "OCR geometry unavailable for this chart.")}</div>`;
    return;
  }

  const boxes = buildPanelBoxes(geom.panel_vision);
  const codexBoxes = buildPanelBoxes(data.debug_inputs?.panel_vision)
    .map((item) => ({ ...item, kind: "corrected", label: "", dashed: true }));
  const ocr = geom.ocr || {};
  const ocrBoxes = (ocr.items || [])
    .map((item) => makeBox("ocr", item.bbox, item.text, `conf ${item.confidence}`))
    .filter(Boolean);

  let showCodex = true;
  let showOcr = false;
  const draw = () => {
    renderStage(stage, data.images.panel_image, [
      ...boxes,
      ...(showCodex ? codexBoxes : []),
      ...(showOcr ? ocrBoxes : []),
    ]);
  };

  const summary = geom.summary || {};
  const replaced = summary.text_boxes_replaced || {};
  controls.innerHTML = `
    <label class="debug-control"><input type="checkbox" id="ocrGeomShowCodex" checked> overlay codex boxes (dashed)</label>
    <label class="debug-control"><input type="checkbox" id="ocrGeomShowOcr"> show all OCR text boxes (${ocrBoxes.length})</label>
    <span class="muted">${replaced.tick_label ?? 0} tick labels · ${replaced.axis_label ?? 0} axis titles · ${replaced.legend_label ?? 0} legend labels replaced</span>
  `;
  renderLegend(legend, ["plot", "axis", "axisLabel", "tick", "tickLabel", "legend", "annotation", "corrected", "ocr"]);
  const ocrState = ocr.error
    ? `unavailable (${escapeHtml(ocr.error)})`
    : `${escapeHtml(ocr.backend || "")} · ${ocr.item_count ?? 0} text items · <strong>${summary.ocr_confirmed_values ?? 0} values confirmed</strong>, ${summary.ocr_conflicting_values ?? 0} disputed`;
  note.innerHTML = `
    <strong>Geometry</strong>: ${escapeHtml(geom.geometry_source)} (unchanged) · frame ${escapeHtml(JSON.stringify(geom.plot_frame_bbox_panel_px))}
    <br><strong>Text</strong>: ${escapeHtml(geom.text_source)} · ${ocrState}
    <br><strong>Issues</strong>: ${escapeHtml((geom.issues || []).join(" · ") || "none")}
  `;

  el("ocrGeomShowCodex").addEventListener("change", (event) => {
    showCodex = event.target.checked;
    draw();
  });
  el("ocrGeomShowOcr").addEventListener("change", (event) => {
    showOcr = event.target.checked;
    draw();
  });
  draw();
  renderOcrGeomTable(geom);
}

let maskSource = "current";

const MASK_SOURCES = {
  current: "current pipeline",
  ocr_text: "OCR text + codex geometry",
};

function maskSourceBoxes(data, key) {
  const vision = key === "ocr_text"
    ? data.ocr_geometry?.panel_vision
    : data.debug_inputs?.panel_vision;
  return { boxes: buildPanelBoxes(vision), frame: bbox(vision?.plot_frame_bbox_panel_px) };
}

// The mask erases identified elements so they cannot be mistaken for curve ink.
function renderMaskSection(data) {
  const options = [{ key: "current", label: MASK_SOURCES.current }];
  if (data.ocr_geometry && !data.ocr_geometry.error) {
    options.push({ key: "ocr_text", label: MASK_SOURCES.ocr_text });
  }
  if (!options.some((option) => option.key === maskSource)) maskSource = "current";
  const { boxes, frame } = maskSourceBoxes(data, maskSource);
  renderMaskView(
    el("maskStage"),
    el("maskControls"),
    data.images.panel_image,
    boxes,
    `${data.chart.chart_id}_element_mask_${maskSource}`,
    {
      options,
      selected: maskSource,
      onChange: (key) => {
        maskSource = key;
        renderMaskSection(data);
      },
    },
    frame,
  );
}
// The plot frame is a container region, not ink: filling it would erase the curves too.
const MASK_CONTAINER_KINDS = ["plot"];
const MASK_KINDS = ["plot", "axis", "axisLabel", "tick", "tickLabel", "legend", "annotation"];

// Kept across source switches so changing the bbox source does not reset the
// kind checkboxes or the padding the user dialled in.
let maskEnabledKinds = null;
let maskPadding = 1;
// Everything outside the plot frame is furniture, never curve ink, so clearing
// it wholesale is the useful starting point rather than box-by-box masking.
let maskOutsideFrame = true;

function renderMaskView(stage, controls, imageUrl, boxes, fileName, sourceControl = null, frame = null) {
  stage.innerHTML = "";
  controls.innerHTML = "";
  const sourceSelect = sourceControl
    ? `<label class="debug-control">bbox source
        <select id="maskSource">
          ${sourceControl.options.map((option) => `
            <option value="${escapeHtml(option.key)}"${option.key === sourceControl.selected ? " selected" : ""}>${escapeHtml(option.label)}</option>
          `).join("")}
        </select>
      </label>`
    : "";
  const maskable = boxes.filter((item) => MASK_KINDS.includes(item.kind));
  if (!imageUrl || (!maskable.length && !frame)) {
    controls.innerHTML = sourceSelect;
    if (sourceControl) {
      el("maskSource")?.addEventListener("change", (event) => sourceControl.onChange(event.target.value));
    }
    stage.innerHTML = '<div class="empty-state">No panel image or element bbox available.</div>';
    return;
  }

  const kinds = MASK_KINDS.filter((kind) => maskable.some((item) => item.kind === kind));
  if (!maskEnabledKinds) {
    maskEnabledKinds = new Set(kinds.filter((kind) => !MASK_CONTAINER_KINDS.includes(kind)));
  }
  const enabled = maskEnabledKinds;
  let padding = maskPadding;

  const canvas = document.createElement("canvas");
  canvas.className = "debug-canvas";
  stage.appendChild(canvas);
  const img = new Image();

  const download = document.createElement("a");
  download.className = "secondary-link";
  download.textContent = "Download PNG";
  download.download = `${fileName || "element_mask"}.png`;

  const draw = () => {
    if (!img.naturalWidth || !img.naturalHeight) return;
    // Draw at natural resolution so the mask is pixel-exact and the PNG export is full size.
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    ctx.fillStyle = "#ffffff";
    // Clear everything beyond the frame first, keeping the frame border itself.
    if (maskOutsideFrame && frame) {
      const [fx1, fy1, fx2, fy2] = frame;
      const rows = fy2 - fy1 + 1;
      ctx.fillRect(0, 0, canvas.width, fy1);
      ctx.fillRect(0, fy2 + 1, canvas.width, canvas.height - fy2 - 1);
      ctx.fillRect(0, fy1, fx1, rows);
      ctx.fillRect(fx2 + 1, fy1, canvas.width - fx2 - 1, rows);
    }
    let masked = 0;
    for (const item of maskable) {
      if (!enabled.has(item.kind)) continue;
      const [x1, y1, x2, y2] = item.bbox;
      ctx.fillRect(x1 - padding, y1 - padding, x2 - x1 + padding * 2, y2 - y1 + padding * 2);
      masked += 1;
    }
    // The stage is a flex container, so give the canvas an explicit display size
    // instead of relying on max-width/height:auto, matching renderStage.
    const stageWidth = Math.max(260, stage.clientWidth - 22 || img.naturalWidth);
    let displayWidth = Math.min(img.naturalWidth, stageWidth);
    let displayHeight = displayWidth * (img.naturalHeight / img.naturalWidth);
    const maxHeight = 720;
    if (displayHeight > maxHeight) {
      displayHeight = maxHeight;
      displayWidth = displayHeight * (img.naturalWidth / img.naturalHeight);
    }
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;
    download.href = canvas.toDataURL("image/png");
    const count = el("maskCount");
    if (count) {
      const outside = maskOutsideFrame && frame ? "outside frame cleared · " : "";
      const inside = insideKinds.size ? ` (${insideKinds.size} kind(s) inside the frame)` : "";
      count.textContent = `${outside}${masked} / ${maskable.length} boxes masked${inside}`;
    }
  };

  const counts = new Map(kinds.map((kind) => [kind, maskable.filter((item) => item.kind === kind).length]));
  const insideKinds = new Set(
    maskable
      .filter((item) => frame && item.bbox[0] >= frame[0] && item.bbox[1] >= frame[1] && item.bbox[2] <= frame[2] && item.bbox[3] <= frame[3])
      .map((item) => item.kind),
  );
  controls.innerHTML = `
    ${sourceSelect}
    <label class="debug-control"${frame ? "" : ' title="no plot frame available"'}>
      <input type="checkbox" id="maskOutside"${maskOutsideFrame ? " checked" : ""}${frame ? "" : " disabled"}>
      mask everything outside the plot frame
    </label>
    ${kinds.map((kind) => `
      <label class="debug-control">
        <input type="checkbox" data-mask-kind="${escapeHtml(kind)}"${enabled.has(kind) ? " checked" : ""}>
        <i style="background:${COLORS[kind] || "#111"}"></i>
        ${escapeHtml(LABELS[kind] || kind)}
        <span class="muted">${counts.get(kind)}${MASK_CONTAINER_KINDS.includes(kind) ? ", container" : ""}</span>
      </label>
    `).join("")}
    <span class="debug-controls-gap"></span>
    <label class="debug-control">expand <input type="number" id="maskPadding" min="0" max="12" step="1" value="${padding}"> px</label>
    <button type="button" class="debug-control-button" id="maskAll">All</button>
    <button type="button" class="debug-control-button" id="maskNone">None</button>
    <span class="muted" id="maskCount"></span>
  `;
  controls.appendChild(download);

  const checkboxes = () => controls.querySelectorAll("input[data-mask-kind]");
  controls.addEventListener("change", (event) => {
    const target = event.target;
    if (target.dataset.maskKind) {
      if (target.checked) enabled.add(target.dataset.maskKind);
      else enabled.delete(target.dataset.maskKind);
    } else if (target.id === "maskPadding") {
      padding = Math.max(0, Math.min(12, Number(target.value) || 0));
      maskPadding = padding;
      target.value = padding;
    } else if (target.id === "maskOutside") {
      maskOutsideFrame = target.checked;
    } else if (target.id === "maskSource" && sourceControl) {
      sourceControl.onChange(target.value);
      return;
    } else {
      return;
    }
    draw();
  });
  const setAll = (checked) => {
    for (const box of checkboxes()) {
      box.checked = checked;
      if (checked) enabled.add(box.dataset.maskKind);
      else enabled.delete(box.dataset.maskKind);
    }
    draw();
  };
  el("maskAll").addEventListener("click", () => setAll(true));
  el("maskNone").addEventListener("click", () => setAll(false));

  img.addEventListener("load", draw);
  img.addEventListener("error", () => {
    stage.innerHTML = `<div class="empty-state">Failed to load panel image: ${escapeHtml(imageUrl)}</div>`;
  });
  img.src = imageUrl;
  window.addEventListener("resize", draw);
}

function rowsTable(headers, rows) {
  if (!rows.length) return '<div class="empty-state">No data.</div>';
  return `
    <table>
      <thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;
}

function rgbToCss(rgb, fallback = "#2563eb") {
  if (!Array.isArray(rgb) || rgb.length < 3) return fallback;
  const values = rgb.slice(0, 3).map((value) => Math.max(0, Math.min(255, Number(value) || 0)));
  return `rgb(${values[0]}, ${values[1]}, ${values[2]})`;
}

function cropOverlayColor(role, axisBinding, fallback) {
  if (axisBinding === "right_y" || String(role || "").includes("power_density")) return "#d97706";
  if (axisBinding === "left_y" || String(role || "").includes("voltage")) return "#0369a1";
  if (String(role || "").includes("unknown")) return "#64748b";
  return fallback;
}

function seriesLabel(legendLabel, role, id, includeTraceId = false) {
  const parts = [legendLabel, role, includeTraceId ? id : null]
    .filter((part) => typeof part === "string" && part.trim().length);
  return parts.join(" / ") || id;
}

function formatChartValue(value) {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs >= 1000) return Math.round(value).toLocaleString("en-US");
  if (abs >= 10) return Number(value.toFixed(1)).toLocaleString("en-US");
  if (abs >= 1) return Number(value.toFixed(2)).toLocaleString("en-US");
  return Number(value.toFixed(3)).toLocaleString("en-US");
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

function objectArray(value) {
  return Array.isArray(value)
    ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item))
    : [];
}

function normalizePoints(value) {
  return pointList(value);
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

function buildInteractivePlotData(seriesPayload, tracePayload, axisMappingPayload) {
  // categorical 처리를 위한 수정: plot category labels at ordered discrete x positions.
  const categoryOrder = [...(seriesPayload?.category_order || [])];
  for (const row of objectArray(seriesPayload?.series)) {
    for (const point of row.points_data || []) {
      if (typeof point[0] === "string" && !categoryOrder.includes(point[0])) categoryOrder.push(point[0]);
    }
  }
  const traceRows = objectArray(tracePayload?.traces);
  const colorByTrace = new Map();
  for (const trace of traceRows) {
    const color = rgbToCss(trace.series_color_rgb);
    if (typeof trace.trace_id === "string") colorByTrace.set(trace.trace_id, color);
    if (typeof trace.series_id === "string" && !colorByTrace.has(trace.series_id)) {
      colorByTrace.set(trace.series_id, color);
    }
  }

  const dataSeries = objectArray(seriesPayload?.series)
    .map((row) => {
      // 시리즈·조각 구분: 좌표 유지, 빈 구간 연결 방지
      const segments = (row.segments || [row]).map((segment) => categoryOrder.length
        ? (segment.points_data || []).filter((p) => typeof p[0] === "string" && Number.isFinite(p[1])).map((p) => [categoryOrder.indexOf(p[0]), p[1]])
        : normalizePoints(segment.points_data));
      const points = segments.flat();
      if (points.length < (categoryOrder.length ? 1 : 2)) return null;
      const traceId = typeof row.trace_id === "string" ? row.trace_id : "trace";
      const seriesId = typeof row.series_id === "string" ? row.series_id : traceId;
      const role = typeof row.curve_role === "string" ? row.curve_role : "curve";
      const axisBinding = typeof row.axis_binding === "string" ? row.axis_binding : "left_y";
      const legendLabel = typeof row.legend_label === "string" ? row.legend_label : seriesId;
      return {
        id: traceId,
        label: seriesLabel(legendLabel, role, traceId),
        color: colorByTrace.get(traceId) || colorByTrace.get(seriesId) || (row.series_color_rgb ? rgbToCss(row.series_color_rgb) : "#2563eb"),
        role,
        axisBinding,
        points,
        segments,
      };
    })
    .filter(Boolean);

  if (dataSeries.length) {
    const labels = buildAxisLabelsFromMapping(axisMappingPayload);
    // categorical 처리를 위한 수정: retain y ticks and display original category text.
    const axisTicks = buildAxisTicksFromMapping(axisMappingPayload);
    if (categoryOrder.length) axisTicks.x = categoryOrder.map((label, value) => ({ value, label }));
    return {
      mode: "data",
      xLabel: categoryOrder.length ? (seriesPayload?.axis_labels?.x || "category") : labels.x,
      yLabel: labels.leftY,
      rightYLabel: labels.rightY,
      yIncreasesDown: false,
      axisTicks,
      categoryOrder, // categorical 처리를 위한 수정: hover uses labels rather than indices.
      series: dataSeries,
    };
  }

  const pixelSeries = traceRows
    .map((trace) => {
      const points = tracePoints(trace);
      if (points.length < 2) return null;
      const traceId = typeof trace.trace_id === "string" ? trace.trace_id : "trace";
      const seriesId = typeof trace.series_id === "string" ? trace.series_id : traceId;
      const role = typeof trace.curve_role === "string" ? trace.curve_role : "pixel_curve";
      const axisBinding = typeof trace.axis_binding_hint === "string" ? trace.axis_binding_hint : "unknown";
      const legendLabel = typeof trace.legend_label === "string" ? trace.legend_label : seriesId;
      return {
        id: traceId,
        label: seriesLabel(legendLabel, role, traceId, true),
        color: cropOverlayColor(role, axisBinding, rgbToCss(trace.series_color_rgb)),
        role,
        axisBinding,
        points,
      };
    })
    .filter(Boolean);

  if (!pixelSeries.length) return null;
  return {
    mode: "pixel",
    xLabel: "x_pixel",
    yLabel: "y_pixel",
    yIncreasesDown: true,
    series: pixelSeries,
  };
}

let interactiveHidden = new Set();

function renderInteractiveChart(seriesPayload, tracePayload, axisMappingPayload) {
  const container = el("interactiveChartBox");
  const data = buildInteractivePlotData(seriesPayload, tracePayload, axisMappingPayload);
  interactiveHidden = new Set([...interactiveHidden].filter((id) => data?.series.some((series) => series.id === id)));

  if (!data || !data.series.length) {
    container.innerHTML = '<div class="empty-state">Run extraction to load interactive extracted curves.</div>';
    return;
  }

  const visibleSeries = data.series.filter((series) => !interactiveHidden.has(series.id));
  const plottedSeries = visibleSeries.length ? visibleSeries : data.series;
  const allPoints = plottedSeries.flatMap((series) => series.points);
  const xAxisTicks = data.mode === "data" ? data.axisTicks?.x || [] : [];
  const leftAxisTicks = data.mode === "data" ? data.axisTicks?.leftY || [] : [];
  const rightAxisTicks = data.mode === "data" ? data.axisTicks?.rightY || [] : [];
  const xs = [...allPoints.map((point) => point[0]), ...xAxisTicks.map((tick) => tick.value)].filter(Number.isFinite);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    container.innerHTML = '<div class="empty-state">No plottable extracted points found.</div>';
    return;
  }

  const width = 980;
  const height = 560;
  const hasRightAxis = data.mode === "data" && data.series.some((series) => series.axisBinding === "right_y");
  const pad = { left: 70, right: hasRightAxis ? 76 : 30, top: 30, bottom: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xRange = maxX - minX || 1;
  const leftScaleSeries = plottedSeries.filter((series) => series.axisBinding !== "right_y");
  const rightScaleSeries = plottedSeries.filter((series) => series.axisBinding === "right_y");
  const leftRange = rangeForPoints([
    ...(leftScaleSeries.length ? leftScaleSeries : plottedSeries).flatMap((series) => series.points),
    ...leftAxisTicks.map((tick) => [minX, tick.value]),
  ]);
  const rightRange = rangeForPoints([
    ...(rightScaleSeries.length ? rightScaleSeries : plottedSeries).flatMap((series) => series.points),
    ...rightAxisTicks.map((tick) => [minX, tick.value]),
  ]);
  const leftYRange = leftRange.max - leftRange.min || 1;
  const rightYRange = rightRange.max - rightRange.min || 1;
  const scaleX = (value) => pad.left + ((value - minX) / xRange) * plotW;
  const scaleYLeft = data.yIncreasesDown
    ? (value) => pad.top + ((value - leftRange.min) / leftYRange) * plotH
    : (value) => pad.top + ((leftRange.max - value) / leftYRange) * plotH;
  const scaleYRight = data.yIncreasesDown
    ? (value) => pad.top + ((value - rightRange.min) / rightYRange) * plotH
    : (value) => pad.top + ((rightRange.max - value) / rightYRange) * plotH;
  const scaleYForSeries = (series) => hasRightAxis && series.axisBinding === "right_y" ? scaleYRight : scaleYLeft;
  const xTicks = xAxisTicks.length ? xAxisTicks : buildTicks(minX, maxX).map((value) => ({ value, label: formatChartValue(value) }));
  const leftTicks = leftAxisTicks.length ? leftAxisTicks : buildTicks(leftRange.min, leftRange.max).map((value) => ({ value, label: formatChartValue(value) }));
  const rightTicks = rightAxisTicks.length ? rightAxisTicks : buildTicks(rightRange.min, rightRange.max).map((value) => ({ value, label: formatChartValue(value) }));

  const xTickSvg = xTicks.map((tick) => `
    <g>
      <line x1="${scaleX(tick.value).toFixed(2)}" x2="${scaleX(tick.value).toFixed(2)}" y1="${pad.top}" y2="${pad.top + plotH}" stroke="#e2e8f0"></line>
      <text x="${scaleX(tick.value).toFixed(2)}" y="${height - 28}" text-anchor="middle" font-size="11" fill="#64748b">${escapeHtml(tick.label)}</text>
    </g>
  `).join("");
  const leftTickSvg = leftTicks.map((tick) => `
    <g>
      <line x1="${pad.left}" x2="${pad.left + plotW}" y1="${scaleYLeft(tick.value).toFixed(2)}" y2="${scaleYLeft(tick.value).toFixed(2)}" stroke="#e2e8f0"></line>
      <text x="${pad.left - 10}" y="${(scaleYLeft(tick.value) + 4).toFixed(2)}" text-anchor="end" font-size="11" fill="#64748b">${escapeHtml(tick.label)}</text>
    </g>
  `).join("");
  const rightTickSvg = hasRightAxis ? rightTicks.map((tick) => `
    <g>
      <text x="${pad.left + plotW + 10}" y="${(scaleYRight(tick.value) + 4).toFixed(2)}" text-anchor="start" font-size="11" fill="#b45309">${escapeHtml(tick.label)}</text>
    </g>
  `).join("") : "";
  const zeroLines = [
    data.mode === "data" && minX <= 0 && maxX >= 0
      ? `<line x1="${scaleX(0).toFixed(2)}" x2="${scaleX(0).toFixed(2)}" y1="${pad.top}" y2="${pad.top + plotH}" stroke="#0f172a" stroke-dasharray="5 5" opacity="0.5"></line>`
      : "",
    data.mode === "data" && leftRange.min <= 0 && leftRange.max >= 0
      ? `<line x1="${pad.left}" x2="${pad.left + plotW}" y1="${scaleYLeft(0).toFixed(2)}" y2="${scaleYLeft(0).toFixed(2)}" stroke="#0f172a" stroke-dasharray="5 5" opacity="0.5"></line>`
      : "",
  ].join("");
  const seriesSvg = visibleSeries.map((series) => `
    ${(series.segments || [series.points]).map((points) => `<path d="${linePath(points, scaleX, scaleYForSeries(series))}" fill="none" stroke="${escapeHtml(series.color)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"></path>`).join("")}
    ${data.categoryOrder?.length ? series.points.map((point) => `<circle cx="${scaleX(point[0]).toFixed(2)}" cy="${scaleYForSeries(series)(point[1]).toFixed(2)}" r="3" fill="${escapeHtml(series.color)}"></circle>`).join("") : ""}
  `).join("");

  container.innerHTML = `
    <div class="interactive-chart-toolbar">
      <span class="interactive-pill">mode: ${escapeHtml(data.mode)}</span>
      <span class="muted">Hover points to inspect coordinates; use checkboxes to show or hide series.</span>
      <button type="button" id="interactiveSelectAll">Select all</button>
      <button type="button" id="interactiveClearAll">Clear all</button>
    </div>
    <div class="interactive-chart-frame">
      <svg id="interactiveChartSvg" viewBox="0 0 ${width} ${height}" role="img" aria-label="interactive extracted curve plot">
        <rect width="${width}" height="${height}" fill="#ffffff"></rect>
        <rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" fill="#f8fafc" stroke="#cbd5e1"></rect>
        ${xTickSvg}
        ${leftTickSvg}
        ${rightTickSvg}
        ${zeroLines}
        ${seriesSvg}
        <text x="${pad.left + plotW / 2}" y="${height - 10}" text-anchor="middle" font-size="12" fill="#334155">${escapeHtml(data.xLabel)}</text>
        <text x="20" y="${pad.top + plotH / 2}" text-anchor="middle" font-size="12" fill="#334155" transform="rotate(-90 20 ${pad.top + plotH / 2})">${escapeHtml(data.yLabel)}</text>
        ${hasRightAxis ? `<text x="${width - 20}" y="${pad.top + plotH / 2}" text-anchor="middle" font-size="12" fill="#b45309" transform="rotate(90 ${width - 20} ${pad.top + plotH / 2})">${escapeHtml(data.rightYLabel || "right_y")}</text>` : ""}
        <g id="interactiveHover" class="hidden"></g>
      </svg>
    </div>
    <div class="interactive-chart-legend">
      ${data.series.map((series) => `
        <label class="interactive-series">
          <input type="checkbox" data-series-id="${escapeHtml(series.id)}" ${interactiveHidden.has(series.id) ? "" : "checked"}>
          <span class="interactive-swatch" style="background:${escapeHtml(series.color)}"></span>
          <span class="interactive-series-label">${escapeHtml(series.label)}</span>
          <span class="interactive-axis">${escapeHtml(series.axisBinding)}</span>
          <span class="interactive-count">${series.points.length}</span>
        </label>
      `).join("")}
    </div>
  `;

  const rerender = () => renderInteractiveChart(seriesPayload, tracePayload, axisMappingPayload);
  el("interactiveSelectAll")?.addEventListener("click", () => {
    interactiveHidden = new Set();
    rerender();
  });
  el("interactiveClearAll")?.addEventListener("click", () => {
    interactiveHidden = new Set(data.series.map((series) => series.id));
    rerender();
  });
  container.querySelectorAll("input[data-series-id]").forEach((input) => {
    input.addEventListener("change", () => {
      const id = input.dataset.seriesId;
      if (!id) return;
      if (input.checked) interactiveHidden.delete(id);
      else interactiveHidden.add(id);
      rerender();
    });
  });

  const svg = el("interactiveChartSvg");
  const hoverGroup = el("interactiveHover");
  svg?.addEventListener("pointermove", (event) => {
    // Map through the CTM so hover stays accurate when the viewBox is letterboxed.
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const local = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const sx = local.x;
    const sy = local.y;
    let best = null;
    let bestDist = Infinity;
    for (const series of visibleSeries) {
      const stride = Math.max(1, Math.floor(series.points.length / 250));
      const scaleY = scaleYForSeries(series);
      for (let index = 0; index < series.points.length; index += stride) {
        const point = series.points[index];
        const px = scaleX(point[0]);
        const py = scaleY(point[1]);
        const dist = (px - sx) ** 2 + (py - sy) ** 2;
        if (dist < bestDist) {
          bestDist = dist;
          best = { series, point, sx: px, sy: py };
        }
      }
    }
    if (!hoverGroup) return;
    if (!best || bestDist >= 700) {
      hoverGroup.classList.add("hidden");
      hoverGroup.innerHTML = "";
      return;
    }
    const boxX = Math.min(width - 240, best.sx + 12);
    const boxY = Math.max(10, best.sy - 42);
    hoverGroup.classList.remove("hidden");
    hoverGroup.innerHTML = `
      <circle cx="${best.sx.toFixed(2)}" cy="${best.sy.toFixed(2)}" r="5" fill="${escapeHtml(best.series.color)}" stroke="#0f172a"></circle>
      <rect x="${boxX.toFixed(2)}" y="${boxY.toFixed(2)}" width="228" height="54" rx="8" fill="#0f172a" opacity="0.92"></rect>
      <text x="${(boxX + 10).toFixed(2)}" y="${(boxY + 22).toFixed(2)}" font-size="11" fill="#fff">${escapeHtml(best.series.label).slice(0, 42)}</text>
      <text x="${(boxX + 10).toFixed(2)}" y="${(boxY + 40).toFixed(2)}" font-size="11" fill="#cbd5e1">x=${escapeHtml(data.categoryOrder?.[best.point[0]] ?? formatChartValue(best.point[0]))}, y=${formatChartValue(best.point[1])}</text>
    `;
  });
  svg?.addEventListener("pointerleave", () => {
    hoverGroup?.classList.add("hidden");
    if (hoverGroup) hoverGroup.innerHTML = "";
  });
}

function renderAxisMapping(axisMapping) {
  const axes = axisMapping?.axes || {};
  const rows = Object.entries(axes).map(([role, axis]) => [
    role,
    axis?.model ? "ready" : "missing",
    axis?.source || "",
    JSON.stringify(axis?.model || null),
    (axis?.issues || []).join(", "),
  ]);
  el("axisMappingBox").innerHTML = rowsTable(["Axis", "Model", "Source", "Mapping", "Issues"], rows);
}

function renderTicks(panelVision) {
  const rows = [];
  for (const axis of panelVision?.axes || []) {
    for (const tick of axis.ticks || []) {
      rows.push([
        axis.axis_role || "",
        tick.label_text || "",
        tick.label_numeric_value ?? "",
        tick.tick_position_px ?? "",
        JSON.stringify(tick.label_bbox_panel_px || []),
        tick.confidence ?? "",
      ]);
    }
  }
  el("tickBox").innerHTML = rowsTable(["Axis", "Text", "Numeric", "Position px", "Label bbox", "Confidence"], rows);
}

function renderLegendTables(data) {
  const visionItems = data.debug_inputs?.panel_vision?.legend_items || [];
  const exclusions = data.exclusion_regions?.bboxes || data.debug_inputs?.exclusion_bboxes_crop_px || [];
  const rows = [
    ...visionItems.map((item) => [
      "legend",
      item.legend_label || "",
      JSON.stringify(item.label_bbox_panel_px || []),
      JSON.stringify(item.swatch_bbox_panel_px || []),
      item.confidence ?? "",
    ]),
    ...exclusions.map((box, index) => ["exclusion", `region ${index + 1}`, JSON.stringify(box), "", ""]),
  ];
  el("legendBox").innerHTML = rowsTable(["Type", "Label", "Label bbox", "Swatch/region bbox", "Confidence"], rows);
}

function renderValidation(validation) {
  const checks = validation?.checks || {};
  const rows = Object.entries(checks).map(([key, value]) => [key, value]);
  const issues = validation?.issues || [];
  el("validationBox").innerHTML = rowsTable(["Check", "Value"], rows)
    + `<div class="debug-issues"><strong>Issues</strong><br>${escapeHtml(issues.length ? issues.join(" · ") : "none")}</div>`;
  const badges = el("validationBadges");
  badges.innerHTML = "";
  // 시리즈·조각 구분: 시리즈 수와 조각 수를 따로 표시합니다.
  for (const key of ["x_axis_mapping_ready", "left_y_axis_mapping_ready", "right_y_axis_mapping_ready", "final_series_count", "segment_count"]) {
    const badge = document.createElement("span");
    badge.className = checks[key] ? "debug-badge ok" : "debug-badge warn";
    badge.textContent = `${key}: ${checks[key] ?? "n/a"}`;
    badges.appendChild(badge);
  }
}

function renderVisualQa(data) {
  const summary = data.visual_qa_summary || {};
  const maskItems = objectArray(data.mask_quality_qa?.items)
    .filter((item) => item.severity !== "ok" || (item.issues || []).length)
    .slice(0, 16)
    .map((item) => [
      "mask",
      item.relative_path || item.filename || "",
      item.severity || "",
      (item.issues || []).join(", "),
      `components=${item.component_count ?? ""}, largest=${item.largest_component_ratio ?? ""}`,
    ]);
  const traceItems = objectArray(data.trace_overlay_qa?.items)
    .filter((item) => item.severity !== "ok" || (item.issues || []).length)
    .slice(0, 16)
    .map((item) => [
      "trace",
      item.trace_id || item.series_id || "",
      item.severity || "",
      (item.issues || []).join(", "),
      `plateau=${item.longest_same_y_run ?? ""}, zigzag=${item.direction_change_count ?? ""}`,
    ]);
  const rows = [...maskItems, ...traceItems];
  const header = `
    <div class="debug-issues">
      <strong>Status</strong>: ${escapeHtml(summary.status || "not generated")}
      · <strong>Issues</strong>: ${escapeHtml(summary.issue_count ?? 0)}
      <br>${escapeHtml(summary.human_review_hint || "")}
    </div>
  `;
  el("visualQaBox").innerHTML = header + rowsTable(["Kind", "Item", "Severity", "Issues", "Metrics"], rows);
}

function renderSeries(seriesData) {
  const rows = (seriesData?.series || []).map((series) => [
    series.legend_label || series.series_id || series.trace_id || "",
    series.curve_role || "",
    series.axis_binding || "",
    (series.points_data || []).length,
    JSON.stringify((series.points_data || [])[0] || []),
  ]);
  el("seriesBox").innerHTML = rowsTable(["Series", "Role", "Axis", "Points", "First point"], rows);
}

function renderRaw(data) {
  const keys = [
    "raw_codex_vision",
    "ocr_geometry",
    "debug_inputs",
    "crop_meta",
    "axis_mapping",
    "chart_extraction_strategy",
    "image_quality_report",
    "series_candidates",
    "curve_traces_crop_px",
    "mask_quality_qa",
    "trace_overlay_qa",
    "visual_qa_summary",
    "visual_qa_llm_input",
    "curve_series_data",
    "validation_report",
    "run_manifest",
  ];
  el("rawBox").innerHTML = keys.map((key) => `
    <details>
      <summary>${escapeHtml(key)}</summary>
      <pre>${escapeHtml(JSON.stringify(data[key] ?? null, null, 2))}</pre>
    </details>
  `).join("");
}

async function main() {
  const { jobId, chartId } = routeParts();
  el("backLink").href = `/#${encodeURIComponent(jobId || "")}`;
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/charts/${encodeURIComponent(chartId)}/debug`);
  const chart = data.chart;
  const panelVision = data.debug_inputs?.panel_vision;
  el("debugTitle").textContent = chart.image_name || chart.chart_id;
  el("debugSubtitle").textContent = chart.chart_id;
  el("debugMeta").textContent = `· ${chart.panel_type || "panel"} · ${chart.chart_family || "unknown"} · ${chart.extraction_family || "no extraction family"} · ${chart.status}`;
  if (chart.csv_url) {
    el("csvDebugButton").href = chart.csv_url;
    show("csvDebugButton", true);
  }

  // Quad layout: only these four panels exist in debug.html.
  const panelBoxes = buildPanelBoxes(panelVision);
  renderStage(el("panelStage"), data.images.panel_image, panelBoxes);
  renderLegend(el("panelLegend"), ["plot", "axis", "axisLabel", "tick", "tickLabel", "legend", "annotation"]);
  renderStage(el("colorStage"), data.images.color_filter_source);
  renderTicks(panelVision);
  renderInteractiveChart(data.curve_series_data, data.curve_traces_crop_px, data.axis_mapping);
}

main().catch((err) => {
  const box = el("debugError");
  box.textContent = err.message;
  show("debugError", true);
});

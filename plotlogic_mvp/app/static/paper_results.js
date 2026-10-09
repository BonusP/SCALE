
const state = {
  paperId: null,
  charts: [],
  selectedChartId: null,
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

function formatNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return Math.abs(n) >= 100 ? n.toFixed(2) : n.toPrecision(5);
}

function routePaperId() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return Number(parts[1]);
}

function chartFamilyLabel(chart) {
  const panelType = chart.panel_type || (chart.is_chart ? "chart" : "panel");
  const chartFamily = chart.chart_family || "unknown";
  const extractionFamily = chart.extraction_family ? ` · extraction ${chart.extraction_family}` : "";
  return `${panelType} · ${chartFamily}${extractionFamily}`;
}

function chartBadge(chart) {
  if (chart.status === "extracted") return `extracted · ${chart.series_count} series`;
  if (chart.status === "extraction_failed") return "extraction failed";
  return chart.skip_reason || chart.status;
}

function extractionLabel(extraction) {
  if (!extraction) return "not extracted";
  const counts = `${extraction.target_count || 0} target · ${extraction.skipped_count || 0} skipped`;
  return `${extraction.status} · ${extraction.phase || ""} · ${counts}`;
}

function renderCharts(charts) {
  const list = el("paperChartsList");
  list.innerHTML = "";
  if (!charts.length) {
    list.innerHTML = '<div class="empty-state">No PlotLogic results for this paper.</div>';
    return;
  }
  for (const chart of charts) {
    const item = document.createElement("div");
    item.className = `chart-item ${chart.chart_id === state.selectedChartId ? "selected" : ""}`;
    item.innerHTML = `
      <div class="item-title">${escapeHtml(chart.image_name)}</div>
      <div class="item-meta">${escapeHtml(chartFamilyLabel(chart))} · confidence ${Number(chart.confidence || 0).toFixed(2)}</div>
      <div class="item-meta"><span class="method-tag">${escapeHtml(chart.method || "baseline")}</span></div>
      <div class="item-meta">${escapeHtml(chartBadge(chart))}</div>
      <div class="item-meta">Job ${escapeHtml(String(chart.job_id).slice(0, 8))}</div>
    `;
    item.addEventListener("click", () => loadChart(chart.job_id, chart.chart_id));
    list.appendChild(item);
  }
}

function renderJobs(jobs) {
  const list = el("paperJobsList");
  list.innerHTML = "";
  if (!jobs.length) {
    list.innerHTML = '<div class="empty-state">No extraction runs.</div>';
    return;
  }
  for (const job of jobs) {
    const item = document.createElement("a");
    item.className = "paper-run";
    item.href = `/#${encodeURIComponent(job.job_id)}`;
    item.innerHTML = `
      <span>${escapeHtml(job.created_at || "")}</span>
      <strong>${escapeHtml(job.status)} · ${escapeHtml(job.phase)}</strong>
      <span>${job.target_count || 0} target · ${job.skipped_count || 0} skipped · ${escapeHtml(String(job.job_id).slice(0, 8))}</span>
    `;
    list.appendChild(item);
  }
}

function renderArtifacts(artifacts) {
  const box = el("paperArtifactBox");
  box.innerHTML = "";
  if (!artifacts.length) {
    show("paperArtifactBox", false);
    return;
  }
  const heading = document.createElement("div");
  heading.className = "artifact-heading";
  heading.textContent = "Stored artifacts";
  box.appendChild(heading);
  const links = document.createElement("div");
  links.className = "artifact-links";
  for (const artifact of artifacts) {
    const link = document.createElement("a");
    link.href = artifact.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = `${artifact.label} · ${artifact.kind}`;
    links.appendChild(link);
  }
  box.appendChild(links);
  show("paperArtifactBox", true);
}

function renderSeries(curveSeriesData) {
  const tbody = el("paperSeriesTable").querySelector("tbody");
  tbody.innerHTML = "";
  const series = curveSeriesData?.series || [];
  if (!series.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5">No data series extracted.</td>';
    tbody.appendChild(row);
    return;
  }
  for (const item of series) {
    const points = item.points_data || [];
    const first = points.length ? `[${formatNumber(points[0][0])}, ${formatNumber(points[0][1])}]` : "-";
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(item.legend_label || item.series_id || item.trace_id || "-")}</td>
      <td>${escapeHtml(item.curve_role || "-")}</td>
      <td>${escapeHtml(item.axis_binding || "-")}</td>
      <td>${points.length}</td>
      <td>${escapeHtml(first)}</td>
    `;
    tbody.appendChild(row);
  }
}

async function loadChart(jobId, chartId) {
  state.selectedChartId = chartId;
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/charts/${encodeURIComponent(chartId)}`);
  const chart = data.chart;
  show("paperChartEmpty", false);
  show("paperChartDetail", true);
  el("paperChartTitle").textContent = chart.image_name;
  el("paperChartSubtitle").textContent = `${chartFamilyLabel(chart)} · ${chart.status}`;
  el("paperSourceImage").src = chart.image_url || "";
  el("paperOverlayImage").src = chart.overlay_url || chart.image_url || "";

  const csvButton = el("paperCsvButton");
  if (chart.csv_url) {
    csvButton.href = chart.csv_url;
    show("paperCsvButton", true);
  } else {
    show("paperCsvButton", false);
  }

  const debugButton = el("paperDebugButton");
  if (chart.debug_url) {
    debugButton.href = chart.debug_url;
    show("paperDebugButton", true);
  } else {
    show("paperDebugButton", false);
  }

  const jobButton = el("paperJobButton");
  jobButton.href = `/#${encodeURIComponent(chart.job_id)}`;
  show("paperJobButton", true);

  const issues = chart.issues || [];
  const issuesBox = el("paperIssuesBox");
  if (issues.length || chart.skip_reason) {
    issuesBox.style.display = "block";
    issuesBox.textContent = [chart.skip_reason, ...issues].filter(Boolean).join(" · ");
  } else {
    issuesBox.style.display = "none";
    issuesBox.textContent = "";
  }

  renderArtifacts(chart.artifacts || []);
  renderSeries(data.curve_series_data);
  renderCharts(state.charts);
}

async function loadPaperResults() {
  const data = await api(`/api/papers/${encodeURIComponent(state.paperId)}/results`);
  const metadata = data.paper?.metadata || {};
  const title = metadata.title || `Paper ${state.paperId}`;
  state.charts = data.charts || [];
  el("paperTitle").textContent = title;
  el("paperMeta").textContent = `Paper ${state.paperId} · parse ${(data.paper?.parse?.status || metadata.parse_status || "missing")} · ${(data.paper?.images || []).length} images`;
  el("paperResultStatus").textContent = extractionLabel(data.extraction);
  el("metricPanels").textContent = `${data.summary?.chart_count || 0} panels`;
  el("metricExtracted").textContent = `${data.summary?.extracted_count || 0} extracted`;
  el("metricCsv").textContent = `${data.summary?.csv_count || 0} CSV`;
  el("metricTargets").textContent = `${data.summary?.target_count || 0} targets`;
  renderCharts(state.charts);
  renderJobs(data.jobs || []);
  if (!state.selectedChartId && state.charts.length) {
    const firstExtracted = state.charts.find((chart) => chart.status === "extracted") || state.charts[0];
    await loadChart(firstExtracted.job_id, firstExtracted.chart_id);
  }
}

async function main() {
  state.paperId = routePaperId();
  if (!Number.isFinite(state.paperId)) throw new Error("Invalid paper id.");
  await loadPaperResults();
}

el("refreshPaperResultsBtn").addEventListener("click", () => {
  loadPaperResults().catch((err) => {
    el("paperMeta").textContent = err.message;
  });
});

main().catch((err) => {
  el("paperMeta").textContent = err.message;
});

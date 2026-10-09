const state = {
  currentJobId: null,
  currentChartId: null,
  pollTimer: null,
};

const el = (id) => document.getElementById(id);

function setText(id, text) {
  el(id).textContent = text ?? "";
}

function show(id, visible) {
  el(id).classList.toggle("hidden", !visible);
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

function statusLabel(job) {
  return `${job.status} · ${job.progress}%`;
}

function renderJobs(jobs) {
  const list = el("jobsList");
  list.innerHTML = "";
  if (!jobs.length) {
    list.innerHTML = '<div class="empty-state">No jobs yet.</div>';
    return;
  }
  for (const job of jobs) {
    const item = document.createElement("div");
    item.className = `job-item ${job.job_id === state.currentJobId ? "selected" : ""}`;
    item.innerHTML = `
      <div class="item-title">${escapeHtml(job.title || job.source_pdf_name)}</div>
      <div class="item-meta">Paper ${job.paper_id} · ${escapeHtml(statusLabel(job))}</div>
      <div class="item-meta">${job.target_count} target · ${job.skipped_count} skipped</div>
      <button type="button" class="job-delete-btn" title="Delete job">Delete</button>
    `;
    item.addEventListener("click", () => loadJob(job.job_id));
    item.querySelector(".job-delete-btn").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteJob(job.job_id, job.title || job.source_pdf_name);
    });
    list.appendChild(item);
  }
}

async function deleteJob(jobId, label) {
  const ok = confirm(
    `Delete job "${label}"? This removes its extracted images, CSVs, and debug artifacts. This cannot be undone.`
  );
  if (!ok) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    if (state.currentJobId === jobId) {
      state.currentJobId = null;
      state.currentChartId = null;
      stopPolling();
      show("jobView", false);
      show("emptyState", true);
    }
    await loadJobs();
  } catch (err) {
    alert(`Could not delete job: ${err.message}`);
  }
}

async function loadJobs() {
  const data = await api("/api/jobs");
  renderJobs(data.jobs);
}

function queuedMethodsLabel(data) {
  const jobs = data.jobs || [];
  if (!jobs.length) return "1 job";
  return jobs.map((job) => job.method).join(" + ");
}

function renderEvents(events) {
  const box = el("eventsList");
  box.innerHTML = "";
  for (const event of events) {
    const row = document.createElement("div");
    row.className = "event";
    row.textContent = `${event.created_at} · ${event.level} · ${event.message}`;
    box.appendChild(row);
  }
}

function chartBadge(chart) {
  if (chart.status === "extracted") return `extracted · ${chart.series_count} series`;
  if (chart.status === "extraction_failed") return "extraction failed";
  return chart.skip_reason || chart.status;
}

function chartFamilyLabel(chart) {
  const panelType = chart.panel_type || (chart.is_chart ? "chart" : "panel");
  const chartFamily = chart.chart_family || "unknown";
  const extractionFamily = chart.extraction_family ? ` · extraction ${chart.extraction_family}` : "";
  return `${panelType} · ${chartFamily}${extractionFamily}`;
}

function renderCharts(charts) {
  const list = el("chartsList");
  list.innerHTML = "";
  if (!charts.length) {
    list.innerHTML = '<div class="empty-state">No panels classified yet.</div>';
    return;
  }
  for (const chart of charts) {
    const item = document.createElement("div");
    item.className = `chart-item ${chart.chart_id === state.currentChartId ? "selected" : ""}`;
    item.innerHTML = `
      <div class="item-title">${escapeHtml(chart.image_name)}</div>
      <div class="item-meta">${escapeHtml(chartFamilyLabel(chart))} · confidence ${Number(chart.confidence || 0).toFixed(2)}</div>
      <div class="item-meta"><span class="method-tag">${escapeHtml(chart.method || "baseline")}</span></div>
      <div class="item-meta">${escapeHtml(chartBadge(chart))}</div>
    `;
    item.addEventListener("click", () => loadChart(chart.job_id, chart.chart_id));
    list.appendChild(item);
  }
}

async function loadJob(jobId) {
  state.currentJobId = jobId;
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  show("emptyState", false);
  show("jobView", true);
  const job = data.job;
  setText("jobTitle", job.title || job.source_pdf_name);
  setText(
    "jobMeta",
    `PaperRAG paper ${job.paper_id} · ${job.source_pdf_name} · method ${job.method || "baseline"}`
  );
  setText("jobStatus", job.status);
  setText("jobPhase", job.phase);
  el("progressBar").style.width = `${job.progress || 0}%`;
  renderEvents(data.events || []);
  renderCharts(data.charts || []);
  await loadJobs();
  if (["completed", "completed_with_skips", "failed"].includes(job.status)) {
    stopPolling();
  } else {
    startPolling();
  }
}

async function loadChart(jobId, chartId) {
  state.currentChartId = chartId;
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/charts/${encodeURIComponent(chartId)}`);
  const chart = data.chart;
  show("chartEmpty", false);
  show("chartDetail", true);
  setText("chartTitle", chart.image_name);
  setText("chartSubtitle", `${chartFamilyLabel(chart)} · ${chart.status}`);
  el("sourceImage").src = chart.image_url || "";
  el("overlayImage").src = chart.overlay_url || chart.image_url || "";
  const csvButton = el("csvButton");
  if (chart.csv_url) {
    csvButton.href = chart.csv_url;
    show("csvButton", true);
  } else {
    show("csvButton", false);
  }
  const debugButton = el("debugButton");
  if (chart.debug_url) {
    debugButton.href = chart.debug_url;
    show("debugButton", true);
  } else {
    show("debugButton", false);
  }
  const issues = chart.issues || [];
  const issuesBox = el("issuesBox");
  if (issues.length || chart.skip_reason) {
    issuesBox.style.display = "block";
    issuesBox.textContent = [chart.skip_reason, ...issues].filter(Boolean).join(" · ");
  } else {
    issuesBox.style.display = "none";
    issuesBox.textContent = "";
  }
  renderSeries(data.curve_series_data);
  renderCharts((await api(`/api/jobs/${encodeURIComponent(jobId)}/charts`)).charts);
}

function renderSeries(curveSeriesData) {
  const tbody = el("seriesTable").querySelector("tbody");
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

function formatNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return Math.abs(n) >= 100 ? n.toFixed(2) : n.toPrecision(5);
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(() => {
    if (state.currentJobId) loadJob(state.currentJobId).catch(console.error);
  }, 3000);
}

function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// 파일 선택 영문 표시: 선택한 파일 이름을 보여 줍니다.
el("pdfInput").addEventListener("change", () => {
  const file = el("pdfInput").files[0];
  setText("pdfFileName", file ? file.name : "No file chosen");
});

el("uploadForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = el("pdfInput").files[0];
  if (!file) {
    setText("uploadStatus", "Choose a PDF first.");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  el("uploadBtn").disabled = true;
  setText("uploadStatus", "Uploading and registering in PaperRAG...");
  try {
    const data = await api("/api/jobs", { method: "POST", body: form });
    if (data.duplicate) {
      setText(
        "uploadStatus",
        `Duplicate PDF detected (${data.duplicate_match_source || "sha256"}). Queued extraction for existing PaperRAG paper ${data.paper_id}.`
      );
    } else {
      setText("uploadStatus", `Queued ${queuedMethodsLabel(data)} for PaperRAG paper ${data.paper_id}. They run one after another.`);
    }
    if (data.job_id) await loadJob(data.job_id);
  } catch (err) {
    setText("uploadStatus", err.message);
  } finally {
    el("uploadBtn").disabled = false;
  }
});

el("refreshJobsBtn").addEventListener("click", () => {
  loadJobs().catch((err) => setText("uploadStatus", err.message));
  if (state.currentJobId) loadJob(state.currentJobId).catch(console.error);
});

loadJobs().catch((err) => setText("uploadStatus", err.message));

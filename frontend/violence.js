"use strict";

let selectedScan = null;
let scanListKey = "";
let displayedScan = null;
let loadingScan = null;
let scanReport = null;
let uploadingVideo = false;
let viewedScan = null;

function timeLabel(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function renderViolenceControls() {
  const ready = connected && Boolean(state?.violence?.ready);
  $("scan-submit").disabled = !ready || uploadingVideo || pending;
  $("scan-file").disabled = uploadingVideo;
  $("scan-submit").textContent = uploadingVideo ? "Uploading video…" : "Upload & analyse";
  document.querySelectorAll(".scan-recording").forEach((button) => { button.disabled = !ready || pending; });
  document.querySelectorAll(".cancel-scan").forEach((button) => { button.disabled = !connected || pending; });
}

function renderViolence() {
  if (!state?.violence) return;
  const scans = state.violence;
  const missing = $("scan-model-status");
  missing.textContent = scans.ready ? "Model files ready · CPU processing" : `Missing model files: ${scans.missing.join(", ")}`;
  missing.classList.toggle("danger-text", !scans.ready);
  const jobs = scans.jobs;
  $("scan-nav-count").textContent = jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.status)).length;
  const flagged = jobs.filter((job) => job.summary?.prediction === "possible_violence").length;
  $("scan-overview-summary").textContent = `${jobs.length} scans this session · ${flagged} with review candidates`;
  if (!selectedScan || !jobs.some((job) => job.id === selectedScan)) selectedScan = jobs[0]?.id || null;
  const key = JSON.stringify([jobs, selectedScan]);
  if (key !== scanListKey) {
    scanListKey = key;
    $("scan-jobs").replaceChildren();
    if (!jobs.length) $("scan-jobs").append(empty("No analyses yet", "Upload a video or analyse a clip from Recordings."));
    jobs.forEach((job) => {
      const row = element("div", `scan-job ${job.id === selectedScan ? "selected" : ""}`);
      const select = element("button", "scan-job-select");
      select.type = "button";
      select.append(element("strong", "", job.name));
      let label = job.status;
      if (job.status === "running") label = job.total ? `Scanning ${job.done} / ${job.total} windows` : "Loading model…";
      if (job.status === "complete") label = job.summary.prediction === "possible_violence" ? "Review suggested" : "No windows flagged";
      select.append(element("span", job.summary?.prediction === "possible_violence" ? "danger-text" : "", label));
      select.addEventListener("click", () => { selectedScan = job.id; renderViolence(); });
      row.append(select);
      if (["queued", "running", "cancelling"].includes(job.status)) {
        const cancel = element("button", "button cancel-scan", job.status === "cancelling" ? "Cancelling…" : "Cancel");
        cancel.addEventListener("click", () => action("/api/violence/cancel", {id: job.id}));
        row.append(cancel);
      }
      $("scan-jobs").append(row);
    });
  }
  const job = jobs.find((item) => item.id === selectedScan);
  if (viewedScan !== selectedScan) {
    $("scan-player").pause();
    viewedScan = selectedScan;
  }
  $("scan-review").hidden = !job;
  if (job) {
    $("scan-selected-name").textContent = job.name;
    $("scan-job-message").textContent = job.error || (job.status === "queued" ? "Waiting for the current scan to finish." :
      job.status === "running" ? (job.total ? `${job.done} of ${job.total} windows checked` : "Loading the video model. This can take a moment.") :
      job.status === "cancelled" ? "Scan cancelled. No conclusion was produced." :
      job.status === "cancelling" ? "Stopping the scan process…" : "Scan complete. Review the results below.");
    $("scan-progress").hidden = !["running", "queued", "cancelling"].includes(job.status);
    $("scan-progress").max = job.total || 1;
    if (job.total) $("scan-progress").value = job.done;
    else $("scan-progress").removeAttribute("value");
    $("scan-results").hidden = job.status !== "complete" || displayedScan !== job.id;
    if (job.status === "complete" && displayedScan !== job.id && loadingScan !== job.id) loadScanReport(job);
  }
  renderViolenceControls();
}

async function loadScanReport(job) {
  loadingScan = job.id;
  try {
    const response = await fetch(`/api/violence/${job.id}/report`, {signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error("Could not load the scan report.");
    const report = await response.json();
    if (selectedScan !== job.id) return;
    scanReport = report;
    displayedScan = job.id;
    $("scan-outcome").textContent = report.prediction === "possible_violence" ? "Review suggested" : "No windows flagged";
    $("scan-outcome").classList.toggle("danger-text", report.prediction === "possible_violence");
    $("scan-score").textContent = report.violence_score.toFixed(3);
    $("scan-threshold").textContent = report.threshold.toFixed(2);
    $("scan-window-count").textContent = report.windows_scanned;
    $("scan-download").href = `/api/violence/${job.id}/report`;
    $("scan-video-download").href = `/api/violence/${job.id}/video`;
    $("scan-video-download").download = job.name;
    $("scan-player-error").hidden = true;
    const player = $("scan-player");
    player.src = `/api/violence/${job.id}/video`;
    player.load();
    $("scan-segments").replaceChildren();
    if (!report.review_segments.length) {
      $("scan-segments").append(element("p", "scan-muted", "No windows exceeded the threshold. This does not confirm that the video is free of violence."));
    }
    report.review_segments.forEach((segment) => {
      const button = element("button", "segment-button", `${timeLabel(segment.start_seconds)} – ${timeLabel(segment.end_seconds)} · score ${segment.peak_score.toFixed(3)}`);
      button.addEventListener("click", () => seekScan(segment.start_seconds));
      $("scan-segments").append(button);
    });
    $("scan-results").hidden = false;
    drawScanChart();
  } catch (error) { notice(error.message, true); }
  finally { if (loadingScan === job.id) loadingScan = null; }
}

function seekScan(seconds) {
  const player = $("scan-player");
  const seek = () => { player.currentTime = seconds; player.play().catch(() => {}); };
  if (player.readyState >= 1) seek();
  else player.addEventListener("loadedmetadata", seek, {once: true});
}

function drawScanChart() {
  if (!scanReport) return;
  const canvas = $("scan-chart"), ctx = canvas.getContext("2d");
  const {width, height} = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111110"; ctx.fillRect(0, 0, width, height);
  const left = 44, top = 12, right = width - 16, bottom = height - 28;
  const x = (seconds) => left + seconds / scanReport.duration_seconds * (right - left);
  const y = (score) => bottom - score * (bottom - top);
  ctx.strokeStyle = "#34332f"; ctx.lineWidth = 1;
  for (const score of [0, .5, 1]) {
    ctx.beginPath(); ctx.moveTo(left, y(score)); ctx.lineTo(right, y(score)); ctx.stroke();
    ctx.fillStyle = "#aaa69d"; ctx.font = "12px sans-serif"; ctx.fillText(score.toFixed(1), 8, y(score) + 4);
  }
  ctx.strokeStyle = "#e5a66c"; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(left, y(scanReport.threshold)); ctx.lineTo(right, y(scanReport.threshold)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "#8bc48a"; ctx.lineWidth = 2;
  ctx.beginPath();
  scanReport.windows.forEach((window, index) => ctx[index ? "lineTo" : "moveTo"](x((window.start_seconds + window.end_seconds) / 2), y(window.violence_score)));
  ctx.stroke();
  for (const window of scanReport.windows) {
    ctx.fillStyle = window.flagged ? "#ff8a82" : "#8bc48a";
    ctx.beginPath(); ctx.arc(x((window.start_seconds + window.end_seconds) / 2), y(window.violence_score), 2.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = "#aaa69d"; ctx.fillText("0:00", left, height - 7); ctx.fillText(timeLabel(scanReport.duration_seconds), right - 48, height - 7);
}

$("scan-upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("scan-file").files[0];
  if (!file || !connected || uploadingVideo) return;
  if (file.size === 0 || file.size > state.violence.max_upload_bytes) return notice("Choose a non-empty video up to 256 MB.", true);
  uploadingVideo = true; renderViolenceControls();
  try {
    const response = await fetch("/api/violence/upload", {method: "POST", body: file,
      headers: {"X-Trace-Token": state.token, "X-File-Name": encodeURIComponent(file.name), "Content-Type": "application/octet-stream"},
      signal: AbortSignal.timeout(120000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Upload failed.");
    state = result;
    selectedScan = result.violence.jobs[0]?.id || null;
    $("scan-file").value = "";
    notice("Video uploaded. Analysis will run in the background.");
    render();
  } catch (error) { notice(error.message, true); }
  finally { uploadingVideo = false; renderViolenceControls(); }
});
$("scan-player").addEventListener("error", () => { $("scan-player-error").hidden = false; });
$("scan-chart").addEventListener("click", (event) => {
  if (!scanReport) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width * event.currentTarget.width;
  seekScan(Math.max(0, Math.min(1, (x - 44) / (event.currentTarget.width - 60))) * scanReport.duration_seconds);
});
renderViolenceControls();

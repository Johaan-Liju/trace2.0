"use strict";

const $ = (id) => document.getElementById(id);
let state = null;
let connected = false;
let pending = false;
let settingsDirty = false;
let frameURL = null;
let page = "overview";
let points = [];
let editingId = null;
let zoneImage = null;
let lastEvents = "";
let lastClips = "";
let lastZones = "";
const labels = {
  overview: ["Overview", "A clear view of what’s happening, as it happens."],
  zones: ["Zones", "Draw the boundaries. TRACE will watch for entries."],
  activity: ["Activity", "Every restricted entry, in one place."],
  recordings: ["Recordings", "The moments that matter, saved on your device."],
  settings: ["Settings", "Make this view your own."],
};
const statuses = {idle: "Standby", starting: "Starting", monitoring: "Monitoring", preview: "Preview", stopping: "Stopping", ended: "Video ended", error: "Needs attention"};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = !message;
}

function showPage(next) {
  page = next;
  document.querySelectorAll(".page").forEach((node) => { node.hidden = node.id !== `page-${next}`; });
  document.querySelectorAll(".nav-item").forEach((node) => node.classList.toggle("active", node.dataset.page === next));
  $("page-title").textContent = labels[next][0];
  $("breadcrumb").textContent = labels[next][0];
  $("page-description").textContent = labels[next][1];
  location.hash = next;
}

function empty(title, text, symbol = "✓") {
  const node = element("div", "empty-state");
  node.append(element("span", "empty-symbol", symbol), element("h3", "", title), element("p", "", text));
  return node;
}

async function action(endpoint, payload = {}) {
  if (pending || !state || !connected) return null;
  pending = true;
  renderControls();
  try {
    const response = await fetch(endpoint, {
      method: "POST", headers: {"Content-Type": "application/json", "X-Trace-Token": state.token},
      body: JSON.stringify(payload), signal: AbortSignal.timeout(10000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The request could not be completed.");
    state = result;
    notice("");
    render();
    return result;
  } catch (error) {
    notice(error.message, true);
    return null;
  } finally {
    pending = false;
    renderControls();
  }
}

function renderControls() {
  const busy = Boolean(state?.busy);
  $("start-stop").textContent = busy ? "■  Stop monitoring" : "▶  Start monitoring";
  $("start-stop").classList.toggle("stop", busy);
  $("start-stop").disabled = pending || !connected || state?.status === "stopping";
  $("mode").disabled = pending || busy || !connected;
  $("settings-fields").disabled = busy || pending || !connected;
  $("save-settings").disabled = busy || pending || !connected;
  $("add-zone").disabled = busy || pending || !connected || !state?.has_frame;
  $("settings-hint").textContent = busy ? "Stop monitoring to edit these settings." : "Changes apply the next time you start.";
  document.querySelectorAll(".zone-edit, .zone-delete").forEach((button) => { button.disabled = busy || pending || !connected; });
}

function fillSettings() {
  if (settingsDirty || !state) return;
  const c = state.config;
  $("setting-name").value = c.name;
  $("setting-source").value = c.source;
  $("setting-source").placeholder = c.source_type === "rtsp" ? "Private stream configured · leave blank to keep" : "0, a video path, or an RTSP URL";
  $("setting-confidence").value = c.confidence * 100;
  $("confidence-label").textContent = `${Math.round(c.confidence * 100)}%`;
  $("setting-alerts").checked = c.alerts;
  $("setting-clips").checked = c.clips;
  $("setting-sound").checked = c.sound;
}

function renderEvents() {
  const key = JSON.stringify(state.events);
  if (key === lastEvents) return;
  lastEvents = key;
  for (const [id, limit] of [["recent-events", 5], ["all-events", 500]]) {
    const parent = $(id);
    parent.replaceChildren();
    if (!state.events.length) {
      parent.append(empty("Nothing to report", "New zone entries will appear here when monitoring is active."));
    }
    state.events.slice(0, limit).forEach((event) => {
      const row = element("div", "event");
      const body = element("div", "event-body");
      body.append(element("strong", "", "Restricted area entry"), element("p", "", `Person #${event.track_id} · ${event.zone_name}`));
      const date = new Date(event.timestamp);
      const stamp = element("time", "", date.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"}));
      stamp.dateTime = event.timestamp;
      stamp.title = date.toLocaleString();
      row.append(element("span", "event-icon", "↳"), body, stamp);
      parent.append(row);
    });
  }
}

function renderZones() {
  const zones = state.config.zones;
  const key = JSON.stringify([zones, state.zone_counts]);
  if (key === lastZones) return;
  lastZones = key;
  $("zone-summary").replaceChildren();
  $("zones-list").replaceChildren();
  if (!zones.length) {
    $("zone-summary").append(element("p", "empty-copy", "No zones yet. Capture a camera preview, then draw your first restricted area in Zones."));
    $("zones-list").append(empty("Give TRACE an area to watch", "Start Camera preview, stop it, then choose Draw a zone to mark an area.", "⌗"));
  }
  zones.forEach((zone) => {
    const count = state.zone_counts[zone.id] || 0;
    $("zone-summary").append(element("span", `zone-chip ${zone.type === "ignore" ? "ignore" : ""}`, `${zone.type === "ignore" ? "◇" : "⌗"}  ${zone.name} · ${count}`));
    const row = element("div", "zone-row");
    const body = element("div");
    body.append(element("h3", "", zone.name), element("p", "", `${zone.type} · ${zone.points.length} corners · ${count} people`));
    const edit = element("button", "button zone-edit", "Edit");
    edit.addEventListener("click", () => openZone(zone));
    const remove = element("button", "button zone-delete danger-text", "Remove");
    remove.addEventListener("click", async () => {
      const result = await action("/api/config", {zones: state.config.zones.filter((item) => item.id !== zone.id)});
      if (result) notice(`Removed “${zone.name}”.`);
    });
    row.append(element("span", "empty-symbol", "⌗"), body, edit, remove);
    $("zones-list").append(row);
  });
}

function renderClips() {
  const key = JSON.stringify(state.clips);
  if (key === lastClips) return;
  lastClips = key;
  $("clips-list").replaceChildren();
  if (!state.clips.length) $("clips-list").append(empty("No recordings yet", "Enable Record entry clips in Settings. New people appearing in view will trigger a clip.", "▻"));
  state.clips.forEach((clip, index) => {
    const row = element("div", "clip-row");
    const body = element("div");
    body.append(element("h3", "", `Entry recording ${state.clips.length - index}`), element("p", "", clip.name));
    const link = element("a", "button", "↓ Download MP4");
    link.href = clip.url;
    link.download = clip.name;
    row.append(element("span", "empty-symbol", "▻"), body, link);
    $("clips-list").append(row);
  });
}

function render() {
  if (!state) return;
  const live = ["monitoring", "preview"].includes(state.status);
  $("camera-status").textContent = statuses[state.status] || state.status;
  $("camera-detail").textContent = state.busy ? `${state.config.source_type === "file" ? "Video file" : "Camera"} · ${state.config.camera_id}` : "Ready when you are";
  $("camera-name").textContent = state.config.name;
  $("people-count").textContent = state.people;
  $("alert-count").textContent = state.total_alerts;
  $("activity-nav-count").textContent = state.total_alerts;
  $("clip-count").textContent = state.clips.length;
  $("zone-count").textContent = state.config.zones.length;
  $("zone-nav-count").textContent = state.config.zones.length;
  $("feed-badge").textContent = live ? (state.status === "preview" ? "PREVIEW" : "LIVE") : state.has_frame ? "LAST FRAME" : "OFFLINE";
  $("feed-badge").classList.toggle("online", live);
  $("feed-dot").classList.toggle("live", live);
  $("feed-message").textContent = state.status === "monitoring" ? "Detection & tracking active" : state.status === "preview" ? "Camera preview · detection off" : state.status === "starting" ? "Opening source and loading model…" : state.status === "stopping" ? "Releasing camera & finishing clips…" : state.status === "ended" ? "Video ended or decoder stopped" : state.has_frame ? "Camera stopped · showing last frame" : "Waiting for camera";
  $("feed-label").textContent = `${state.config.camera_id.toUpperCase()} · ${live ? "LIVE" : "STANDBY"}`;
  $("fps").textContent = state.fps.toFixed(1);
  $("frames").textContent = state.frames.toLocaleString();
  $("model-name").textContent = state.config.model;
  $("model-device").textContent = `${state.config.device.toUpperCase()} · LOCAL INFERENCE`;
  $("feed-empty").hidden = state.has_frame;
  $("camera-feed").hidden = !state.has_frame;
  if (!state.has_frame) $("feed-resolution").textContent = "NO SIGNAL";
  if (state.error) notice(state.error, true);
  fillSettings();
  renderEvents();
  renderClips();
  renderZones();
  renderControls();
}

async function pollState() {
  try {
    const response = await fetch("/api/state", {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("Dashboard unavailable");
    const next = await response.json();
    if (!pending) {
      const wasConnected = connected;
      connected = true;
      state = next;
      if (!wasConnected) notice("");
      render();
    }
  } catch {
    connected = false;
    notice("Cannot reach TRACE. Keep dashboard.py running; this page will reconnect automatically.", true);
    $("camera-status").textContent = "Disconnected";
    $("feed-badge").textContent = "DISCONNECTED";
    $("feed-badge").classList.remove("online");
    $("feed-dot").classList.remove("live");
    $("feed-message").textContent = "Server unavailable · last image may be stale";
    renderControls();
  } finally {
    setTimeout(pollState, 900);
  }
}

async function pollFrame() {
  try {
    if (connected && state?.has_frame && page === "overview" && !document.hidden) {
      const response = await fetch("/api/frame.jpg", {signal: AbortSignal.timeout(5000)});
      if (response.ok) {
        const url = URL.createObjectURL(await response.blob());
        const previous = frameURL;
        frameURL = url;
        $("camera-feed").src = url;
        if (previous) URL.revokeObjectURL(previous);
      }
    }
  } catch { /* The state request reports connection errors. */ }
  finally { setTimeout(pollFrame, 160); }
}

async function openZone(zone = null) {
  if (state.busy) return notice("Stop the camera before editing zones.", true);
  if (!state.has_frame) return notice("Start Camera preview and stop it to capture a frame for the zone editor.", true);
  try {
    const response = await fetch("/api/raw.jpg", {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("A camera frame is not available yet.");
    if (zoneImage) zoneImage.close();
    zoneImage = await createImageBitmap(await response.blob());
    points = zone ? zone.points.map((point) => [...point]) : [];
    editingId = zone?.id || null;
    $("zone-name").value = zone?.name || "";
    // Existing legacy zones retain their type unless deliberately changed.
    $("zone-type").querySelectorAll("[data-legacy]").forEach((node) => node.remove());
    if (zone && !["restricted", "ignore"].includes(zone.type)) {
      const option = element("option", "", zone.type);
      option.value = zone.type;
      option.dataset.legacy = "true";
      $("zone-type").append(option);
    }
    $("zone-type").value = zone?.type || "restricted";
    $("zone-error").textContent = "";
    $("zone-canvas").width = zoneImage.width;
    $("zone-canvas").height = zoneImage.height;
    drawZone();
    $("zone-dialog").showModal();
  } catch (error) { notice(error.message, true); }
}

function drawZone() {
  const canvas = $("zone-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (zoneImage) ctx.drawImage(zoneImage, 0, 0);
  if (points.length) {
    ctx.beginPath();
    points.forEach(([x, y], index) => { ctx[index ? "lineTo" : "moveTo"](x * canvas.width, y * canvas.height); });
    if (points.length >= 3) ctx.closePath();
    ctx.fillStyle = $("zone-type").value === "ignore" ? "#9aabb63b" : "#f69c373b";
    ctx.strokeStyle = $("zone-type").value === "ignore" ? "#c7d7e0" : "#ffc071";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
    points.forEach(([x, y], index) => {
      ctx.beginPath(); ctx.arc(x * canvas.width, y * canvas.height, 5, 0, Math.PI * 2);
      ctx.fillStyle = "white"; ctx.fill();
      ctx.fillStyle = "#ffffff"; ctx.font = "bold 15px sans-serif";
      ctx.fillText(String(index + 1), x * canvas.width + 10, y * canvas.height - 8);
    });
  }
  $("point-count").textContent = `${points.length} corners`;
  $("save-zone").disabled = points.length < 3;
}

document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => showPage(button.dataset.page)));
window.addEventListener("hashchange", () => { const next = location.hash.slice(1); if (labels[next] && page !== next) showPage(next); });
$("start-stop").addEventListener("click", async () => {
  if (!state.busy && settingsDirty) return notice("Save your settings before starting the camera.", true);
  await action(state.busy ? "/api/stop" : "/api/start", {mode: $("mode").value});
});
$("fullscreen").addEventListener("click", async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("feed-container").requestFullscreen(); }
  catch { notice("Fullscreen is unavailable in this browser.", true); }
});
$("camera-feed").addEventListener("load", () => { $("feed-resolution").textContent = `${$("camera-feed").naturalWidth} × ${$("camera-feed").naturalHeight}`; });
$("settings-form").addEventListener("input", () => { settingsDirty = true; $("confidence-label").textContent = `${$("setting-confidence").value}%`; });
$("settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {name: $("setting-name").value, confidence: Number($("setting-confidence").value) / 100,
    alerts: $("setting-alerts").checked, sound: $("setting-sound").checked, clips: $("setting-clips").checked};
  if ($("setting-source").value.trim()) payload.source = $("setting-source").value;
  const result = await action("/api/config", payload);
  if (result) { settingsDirty = false; fillSettings(); notice("Settings saved. They’ll apply the next time you start monitoring."); }
});
$("add-zone").addEventListener("click", () => openZone());
$("close-zone").addEventListener("click", () => $("zone-dialog").close());
$("zone-canvas").addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  const rect = event.currentTarget.getBoundingClientRect();
  points.push([Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))]);
  drawZone();
});
$("undo-point").addEventListener("click", () => { points.pop(); drawZone(); });
$("clear-points").addEventListener("click", () => { points = []; drawZone(); });
$("zone-type").addEventListener("change", drawZone);
$("zone-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const zone = {id: editingId || `zone_${crypto.randomUUID().slice(0, 8)}`, name: $("zone-name").value.trim(), type: $("zone-type").value, points};
  const zones = state.config.zones.filter((item) => item.id !== editingId).concat([zone]);
  $("save-zone").disabled = true;
  const result = await action("/api/config", {zones});
  if (result) { $("zone-dialog").close(); notice(`Saved “${zone.name}”.`); }
  else $("zone-error").textContent = $("notice").textContent;
  $("save-zone").disabled = points.length < 3;
});
function updateClock() { $("clock").textContent = new Date().toLocaleDateString([], {month: "short", day: "numeric", year: "numeric"}); }
updateClock();
setInterval(updateClock, 60000);
showPage(labels[location.hash.slice(1)] ? location.hash.slice(1) : "overview");
renderControls();
pollState();
pollFrame();

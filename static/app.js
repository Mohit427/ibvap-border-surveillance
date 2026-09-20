(() => {
  "use strict";

  const TOGGLE_META = [
    ["detection", "Human & Vehicle Detection", "YOLOv8 detection + ByteTrack IDs with motion trails"],
    ["anpr", "ANPR (Plate Reading)", "Plate detector + OCR, sampled periodically"],
    ["face", "Face Detection", "Bounding boxes only — no identity matching"],
    ["fence", "Virtual Fence", "Alerts when a tracked object enters the drawn zone"],
    ["activity", "Suspicious Activity", "Loitering & fast-movement heuristics from real tracks"],
    ["night_enhance", "Night Enhancement", "Auto contrast boost when frame brightness is low"],
  ];

  const EVENT_LABELS = {
    INTRUSION: "INTRUSION",
    LOITERING: "LOITERING",
    FAST_MOVEMENT: "FAST MOVEMENT",
    PLATE_READ: "PLATE READ",
  };

  const el = (id) => document.getElementById(id);
  const videoFeed = el("videoFeed");
  const fenceCanvas = el("fenceCanvas");
  const ctx = fenceCanvas.getContext("2d");
  const videoWrap = document.querySelector(".video-wrap");

  let currentJpegUrl = null;
  let knownFrameSize = [0, 0];

  // Cloud deploys are served over https, where browsers require wss://.
  const WS_PROTO = location.protocol === "https:" ? "wss:" : "ws:";

  // ---------------- Video WebSocket ----------------

  function connectVideoSocket() {
    const ws = new WebSocket(`${WS_PROTO}//${location.host}/ws/video`);
    ws.binaryType = "blob";
    ws.onmessage = (evt) => {
      const url = URL.createObjectURL(evt.data);
      const old = currentJpegUrl;
      videoFeed.src = url;
      currentJpegUrl = url;
      if (old) URL.revokeObjectURL(old);
    };
    ws.onclose = () => setTimeout(connectVideoSocket, 1000);
    ws.onerror = () => ws.close();
  }

  videoFeed.onload = () => {
    const w = videoFeed.naturalWidth, h = videoFeed.naturalHeight;
    if (w && h && (w !== knownFrameSize[0] || h !== knownFrameSize[1])) {
      knownFrameSize = [w, h];
      videoWrap.style.aspectRatio = `${w} / ${h}`;
      el("feedRes").textContent = `${w}×${h}`;
      resizeCanvas();
    }
  };

  // ---------------- State WebSocket ----------------

  let connOk = { video: false, state: false };

  function connectStateSocket() {
    const ws = new WebSocket(`${WS_PROTO}//${location.host}/ws/state`);
    ws.onopen = () => { connOk.state = true; updateConnBadge(); };
    ws.onmessage = (evt) => applyState(JSON.parse(evt.data));
    ws.onclose = () => { connOk.state = false; updateConnBadge(); setTimeout(connectStateSocket, 1000); };
    ws.onerror = () => ws.close();
  }

  function updateConnBadge() {
    const badge = el("connBadge");
    const ok = connOk.state;
    badge.className = "badge" + (ok ? " live" : "");
    badge.innerHTML = `<span class="dot"></span>${ok ? "LIVE" : "RECONNECTING"}`;
  }

  let togglesBuilt = false;
  let lastFenceSignature = "";

  function applyState(snap) {
    const stats = snap.stats || {};
    el("statPeople").textContent = stats.people_count ?? 0;
    el("statVehicles").textContent = stats.vehicle_count ?? 0;
    el("statPlates").textContent = stats.plates_read ?? 0;
    el("statAlerts").textContent = stats.active_alerts ?? 0;
    el("statFps").textContent = (stats.fps ?? 0).toFixed(1);

    const alertTile = el("alertTile");
    alertTile.classList.toggle("active", (stats.active_alerts ?? 0) > 0);

    const sourceBadge = el("sourceBadge");
    sourceBadge.innerHTML = `<span class="dot"></span>SOURCE: ${(stats.source || "--").toUpperCase()}`;

    const modeBadge = el("modeBadge");
    const isNight = stats.mode === "NIGHT";
    modeBadge.className = "badge " + (isNight ? "night" : "day");
    modeBadge.innerHTML = `<span class="dot"></span>${isNight ? "NIGHT" : "DAY"}`;

    const alertBadge = el("alertBadge");
    if ((stats.active_alerts ?? 0) > 0) {
      alertBadge.style.display = "";
      alertBadge.className = "badge alert";
      alertBadge.innerHTML = `<span class="dot"></span>${stats.active_alerts} ACTIVE ALERT${stats.active_alerts > 1 ? "S" : ""}`;
    } else {
      alertBadge.style.display = "none";
    }

    if (!togglesBuilt && snap.toggles) {
      buildToggles(snap.toggles);
      togglesBuilt = true;
    }

    if (!fenceEditing) {
      const sig = JSON.stringify(snap.fence || []);
      if (sig !== lastFenceSignature) {
        lastFenceSignature = sig;
        fencePoints = (snap.fence || []).map((p) => p.slice());
        redrawFence();
      }
    }

    if (snap.new_events && snap.new_events.length) {
      snap.new_events.forEach(addEventToLog);
    }
  }

  // ---------------- Module toggles ----------------

  function buildToggles(toggles) {
    const container = el("toggleList");
    container.innerHTML = "";
    TOGGLE_META.forEach(([key, name, desc]) => {
      const row = document.createElement("div");
      row.className = "toggle-row";
      row.innerHTML = `
        <div>
          <div class="name">${name}</div>
          <div class="desc">${desc}</div>
        </div>
        <label class="switch">
          <input type="checkbox" data-key="${key}" ${toggles[key] ? "checked" : ""}>
          <span class="track"></span>
          <span class="thumb"></span>
        </label>`;
      container.appendChild(row);
    });
    container.querySelectorAll("input[type=checkbox]").forEach((input) => {
      input.addEventListener("change", () => {
        fetch("/api/toggles", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ [input.dataset.key]: input.checked }),
        });
      });
    });
  }

  // ---------------- Event log ----------------

  function addEventToLog(evt) {
    const log = el("eventLog");
    const empty = log.querySelector(".empty-state");
    if (empty) empty.remove();

    const row = document.createElement("div");
    row.className = `event ${evt.type}`;
    const time = new Date(evt.timestamp * 1000).toLocaleTimeString();
    const thumb = evt.snapshot
      ? `<img class="thumb" src="data:image/jpeg;base64,${evt.snapshot}">`
      : `<div class="thumb"></div>`;
    row.innerHTML = `
      <span class="stripe"></span>
      ${thumb}
      <div class="body">
        <div class="type">${EVENT_LABELS[evt.type] || evt.type}</div>
        <div class="msg">${evt.message}</div>
        <div class="time">${time}</div>
      </div>`;
    log.insertBefore(row, log.firstChild);
    while (log.children.length > 120) log.removeChild(log.lastChild);
  }

  el("btnClearLog").onclick = () => {
    el("eventLog").innerHTML = '<div class="empty-state">No events yet. Alerts will appear here as they\'re detected.</div>';
  };

  // ---------------- Video source controls ----------------

  el("srcWebcam").onclick = () => fetch("/api/source", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ type: "webcam" }) });

  el("srcSample").onclick = async () => {
    const list = el("sampleList");
    if (list.style.display === "none") {
      const res = await fetch("/api/sources");
      const data = await res.json();
      list.innerHTML = "";
      (data.samples || []).forEach((name) => {
        const b = document.createElement("button");
        b.className = "small";
        b.textContent = name;
        b.style.marginRight = "6px";
        b.style.marginTop = "6px";
        b.onclick = () => fetch("/api/source", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ type: "sample", name }) });
        list.appendChild(b);
      });
      if (!data.samples || !data.samples.length) {
        list.innerHTML = '<span class="hint">No sample clips found. Run scripts/generate_sample_clips.py</span>';
      }
      list.style.display = "block";
    } else {
      list.style.display = "none";
    }
  };

  el("srcUploadBtn").onclick = () => el("uploadInput").click();
  el("uploadInput").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    await fetch("/api/upload", { method: "POST", body: form });
  };

  // ---------------- Fence drawing ----------------

  let fencePoints = [];
  let fenceEditing = false;
  let dragIndex = -1;

  function resizeCanvas() {
    const rect = videoWrap.getBoundingClientRect();
    fenceCanvas.width = rect.width;
    fenceCanvas.height = rect.height;
    redrawFence();
  }
  window.addEventListener("resize", resizeCanvas);
  new ResizeObserver(resizeCanvas).observe(videoWrap);

  function redrawFence() {
    ctx.clearRect(0, 0, fenceCanvas.width, fenceCanvas.height);
    if (!fencePoints.length) return;
    const pts = fencePoints.map(([nx, ny]) => [nx * fenceCanvas.width, ny * fenceCanvas.height]);

    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    if (pts.length > 2) ctx.closePath();
    ctx.strokeStyle = fenceEditing ? "#2dd4bf" : "#ffd23c";
    ctx.lineWidth = 2;
    ctx.setLineDash(fenceEditing ? [6, 4] : []);
    ctx.stroke();
    ctx.setLineDash([]);

    pts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#2dd4bf";
      ctx.fill();
    });
  }

  function toNorm(evt) {
    const rect = fenceCanvas.getBoundingClientRect();
    return [(evt.clientX - rect.left) / rect.width, (evt.clientY - rect.top) / rect.height];
  }

  function nearestPointIndex(norm, thresholdPx = 10) {
    const rect = fenceCanvas.getBoundingClientRect();
    let best = -1, bestDist = Infinity;
    fencePoints.forEach(([nx, ny], i) => {
      const dx = (nx - norm[0]) * rect.width;
      const dy = (ny - norm[1]) * rect.height;
      const d = Math.hypot(dx, dy);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return bestDist <= thresholdPx ? best : -1;
  }

  fenceCanvas.addEventListener("mousedown", (e) => {
    if (!fenceEditing) return;
    const norm = toNorm(e);
    const idx = nearestPointIndex(norm);
    if (idx >= 0) {
      dragIndex = idx;
    } else {
      fencePoints.push(norm);
      redrawFence();
    }
  });

  fenceCanvas.addEventListener("mousemove", (e) => {
    if (!fenceEditing || dragIndex < 0) return;
    fencePoints[dragIndex] = toNorm(e);
    redrawFence();
  });

  window.addEventListener("mouseup", () => { dragIndex = -1; });

  fenceCanvas.addEventListener("dblclick", (e) => {
    if (!fenceEditing) return;
    const idx = nearestPointIndex(toNorm(e));
    if (idx >= 0) {
      fencePoints.splice(idx, 1);
      redrawFence();
    }
  });

  el("btnDrawFence").onclick = () => {
    fenceEditing = true;
    fencePoints = [];
    el("fenceHint").textContent = "Click to add points, drag to adjust, double-click to remove. Save when done.";
    el("btnSaveFence").disabled = false;
    redrawFence();
  };

  el("btnClearFence").onclick = () => {
    fenceEditing = false;
    fencePoints = [];
    fetch("/api/fence", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ points: [] }) });
    el("fenceHint").textContent = "Click points on the video to outline a restricted zone, then Save.";
    redrawFence();
  };

  el("btnSaveFence").onclick = () => {
    if (fencePoints.length < 3) {
      el("fenceHint").textContent = "Need at least 3 points to save a zone.";
      return;
    }
    fetch("/api/fence", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ points: fencePoints }) });
    fenceEditing = false;
    el("fenceHint").textContent = "Fence saved. Click “Draw Fence” to redraw.";
    redrawFence();
  };

  // ---------------- Clock ----------------

  setInterval(() => { el("clock").textContent = new Date().toLocaleTimeString(); }, 1000);

  // ---------------- Boot ----------------

  connectVideoSocket();
  connectStateSocket();
  resizeCanvas();
})();

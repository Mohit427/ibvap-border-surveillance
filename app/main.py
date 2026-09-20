from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config, video_source
from app.camera_worker import CameraWorker
from app.state import state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ibvap.main")

app = FastAPI(title="IBVAP - Intelligent Border Video Analytics Platform")

config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

_worker: CameraWorker | None = None


@app.on_event("startup")
def startup():
    global _worker
    _worker = CameraWorker(state)
    _worker.start()


@app.on_event("shutdown")
def shutdown():
    if _worker:
        _worker.stop()


# ---------------- REST API ----------------

@app.get("/api/state")
def get_state():
    return state.snapshot()


@app.post("/api/toggles")
async def set_toggles(payload: dict):
    for key, value in payload.items():
        state.set_toggle(key, bool(value))
    return {"toggles": state.get_toggles()}


@app.post("/api/fence")
async def set_fence(payload: dict):
    points = payload.get("points", [])
    cleaned = [[float(p[0]), float(p[1])] for p in points if len(p) == 2]
    state.set_fence(cleaned)
    return {"fence": cleaned}


@app.get("/api/sources")
def get_sources():
    return {"samples": video_source.list_sample_videos(), "current": state.current_source}


@app.post("/api/source")
async def set_source(payload: dict):
    kind = payload.get("type")
    if kind == "webcam":
        state.request_source({"type": "webcam"})
    elif kind == "sample":
        name = payload.get("name")
        req = {"type": "sample"}
        if name:
            req["path"] = str(config.SAMPLE_VIDEOS_DIR / name)
        state.request_source(req)
    else:
        return JSONResponse({"error": "unknown source type"}, status_code=400)
    return {"ok": True}


@app.post("/api/upload")
async def upload_video(file: UploadFile):
    ext = Path(file.filename or "upload.mp4").suffix or ".mp4"
    dest = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    state.request_source({"type": "upload", "path": str(dest)})
    return {"ok": True, "path": dest.name}


@app.get("/api/events")
def get_events(limit: int = 50):
    return {"events": state.get_recent_events(limit)}


# ---------------- WebSockets ----------------

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            jpeg = state.get_frame()
            if jpeg is not None:
                await websocket.send_bytes(jpeg)
            await asyncio.sleep(1.0 / config.TARGET_FPS)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    last_seq = 0
    try:
        while True:
            snap = state.snapshot()
            new_events = state.get_events_since(last_seq, limit=20)
            if new_events:
                last_seq = new_events[-1]["seq"]
            snap["new_events"] = new_events
            snap["server_time"] = time.time()
            await websocket.send_json(snap)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ---------------- Static frontend ----------------

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(config.STATIC_DIR / "index.html"))

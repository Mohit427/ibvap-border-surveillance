"""Per-frame orchestration: runs the enabled modules over a frame, updates
shared state (stats, track history, events) and draws the annotated
overlay that gets streamed to the dashboard.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import deque

import cv2

from app import config
from app.modules import activity, anpr, detection, face, fence, night
from app.state import AppState

logger = logging.getLogger("ibvap.pipeline")

# ANPR (EasyOCR) is the slowest module by far and the one most likely to
# stall on a weak/throttled CPU. Running it through a single-worker executor
# with a bounded wait means a stuck OCR call can never freeze the main video
# loop - worst case we just skip that frame's plate read and move on; if a
# previous call is still in flight we skip resubmitting rather than piling
# up background work.
_anpr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ibvap-anpr")
_anpr_inflight: concurrent.futures.Future | None = None
_ANPR_TIMEOUT_S = 6.0


def _run_anpr_bounded(frame):
    global _anpr_inflight
    if _anpr_inflight is not None and not _anpr_inflight.done():
        return []
    _anpr_inflight = _anpr_executor.submit(anpr.detect_plates, frame)
    try:
        return _anpr_inflight.result(timeout=_ANPR_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        logger.warning("ANPR took longer than %.0fs, skipping this frame's result", _ANPR_TIMEOUT_S)
        return []
    except Exception:
        logger.exception("ANPR failed")
        return []


# Detection/face inference is the main per-frame CPU cost. On a weak/shared
# CPU we run it only every DETECT_FRAME_INTERVAL / FACE_FRAME_INTERVAL
# frames and reuse the last result in between, so the raw video keeps
# streaming smoothly while box positions just refresh less often.
_last_detections: list = []
_last_faces: list = []


COLOR_PERSON = (60, 220, 80)
COLOR_VEHICLE = (255, 170, 40)
COLOR_FACE = (200, 200, 255)
COLOR_FENCE_OK = (255, 210, 60)
COLOR_FENCE_BREACH = (40, 40, 255)
COLOR_PLATE = (60, 200, 255)
COLOR_ALERT = (40, 40, 255)


def _draw_box(frame, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)


def _draw_trail(frame, history, color):
    pts = [(int(x), int(y)) for _, x, y in history]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)


def _update_track_history(state: AppState, track_id, centroid, now):
    hist = state.track_history.get(track_id)
    if hist is None:
        hist = deque()
        state.track_history[track_id] = hist
    hist.append((now, centroid[0], centroid[1]))
    while hist and now - hist[0][0] > config.TRAIL_HISTORY_SECONDS:
        hist.popleft()


def _prune_stale_tracks(state: AppState, seen_ids, now):
    stale = [tid for tid, hist in state.track_history.items()
             if tid not in seen_ids and (not hist or now - hist[-1][0] > config.TRAIL_HISTORY_SECONDS)]
    for tid in stale:
        del state.track_history[tid]


def process_frame(frame, state: AppState, frame_idx: int):
    global _last_detections, _last_faces
    now = time.time()
    h, w = frame.shape[:2]
    toggles = state.get_toggles()

    is_night_frame = night.is_night(frame)
    state.stats["mode"] = "NIGHT" if is_night_frame else "DAY"
    proc_frame = night.enhance(frame) if (is_night_frame and toggles["night_enhance"]) else frame
    annotated = proc_frame.copy()

    detections = []
    detected_this_frame = toggles["detection"] and (frame_idx % config.DETECT_FRAME_INTERVAL == 0)
    if toggles["detection"]:
        if detected_this_frame:
            detections = detection.run(proc_frame)
            _last_detections = detections
        else:
            detections = _last_detections
    else:
        _last_detections = []

    seen_ids = set()
    person_count = 0
    vehicle_count = 0
    vehicle_boxes = []

    fence_polygon = state.get_fence()
    fence_active = bool(toggles["fence"] and len(fence_polygon) >= 3)
    intruding_ids = set()
    if fence_active:
        intruders = fence.check_intrusions(detections, fence_polygon, w, h)
        intruding_ids = {d["id"] for d in intruders}
        for det in intruders:
            if state.should_alert(det["id"], "INTRUSION", config.ALERT_COOLDOWN_SECONDS):
                snap = _encode_snapshot(annotated, det["box"])
                state.add_event(
                    "INTRUSION",
                    f"{det['label'].capitalize()} #{det['id']} entered the fenced zone",
                    snapshot_b64=snap,
                )

    if toggles["detection"]:
        for det in detections:
            seen_ids.add(det["id"])
            if detected_this_frame:
                _update_track_history(state, det["id"], det["centroid"], now)

            if det["kind"] == "person":
                person_count += 1
                color = COLOR_PERSON
            else:
                vehicle_count += 1
                vehicle_boxes.append(det["box"])
                color = COLOR_VEHICLE

            breach = det["id"] in intruding_ids
            box_color = COLOR_FENCE_BREACH if breach else color
            label = f"{det['label']} #{det['id']}"
            _draw_box(annotated, det["box"], box_color, label)
            _draw_trail(annotated, state.track_history.get(det["id"], []), color)

            if toggles["activity"] and detected_this_frame:
                flag = activity.analyze(state.track_history[det["id"]])
                if flag and state.should_alert(det["id"], flag, config.ALERT_COOLDOWN_SECONDS):
                    snap = _encode_snapshot(annotated, det["box"])
                    verb = "is loitering" if flag == "LOITERING" else "is moving unusually fast"
                    state.add_event(
                        flag,
                        f"{det['label'].capitalize()} #{det['id']} {verb}",
                        snapshot_b64=snap,
                    )

        if detected_this_frame:
            _prune_stale_tracks(state, seen_ids, now)

    state.stats["people_count"] = person_count
    state.stats["vehicle_count"] = vehicle_count

    if fence_active:
        color = COLOR_FENCE_BREACH if intruding_ids else COLOR_FENCE_OK
        pts = fence.polygon_to_px(fence_polygon, w, h)
        cv2.polylines(annotated, [pts], True, color, 2, cv2.LINE_AA)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)

    if toggles["face"]:
        if frame_idx % config.FACE_FRAME_INTERVAL == 0:
            _last_faces = face.detect(proc_frame)
        for f in _last_faces:
            _draw_box(annotated, f["box"], COLOR_FACE, f"face {f['conf']:.2f}", thickness=1)
    else:
        _last_faces = []

    if toggles["anpr"] and frame_idx % config.ANPR_FRAME_INTERVAL == 0:
        for plate in _run_anpr_bounded(proc_frame):
            label = plate["text"] or "?"
            _draw_box(annotated, plate["box"], COLOR_PLATE, f"PLATE {label}")
            if plate["text"]:
                state.stats["plates_read"] += 1
                snap = _encode_snapshot(annotated, plate["box"])
                state.add_event(
                    "PLATE_READ",
                    f"Plate read: {plate['text']} (ocr {plate['ocr_conf']:.2f})",
                    snapshot_b64=snap,
                )

    if is_night_frame:
        cv2.putText(annotated, "NIGHT MODE", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2, cv2.LINE_AA)

    state.stats["active_alerts"] = len(intruding_ids)
    return annotated


def _encode_snapshot(frame, box, pad=30) -> str | None:
    import base64
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")

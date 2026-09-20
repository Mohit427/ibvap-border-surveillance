"""Per-frame orchestration: runs the enabled modules over a frame, updates
shared state (stats, track history, events) and draws the annotated
overlay that gets streamed to the dashboard.
"""
from __future__ import annotations

import time
from collections import deque

import cv2

from app import config
from app.modules import activity, anpr, detection, face, fence, night
from app.state import AppState

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
    now = time.time()
    h, w = frame.shape[:2]
    toggles = state.get_toggles()

    is_night_frame = night.is_night(frame)
    state.stats["mode"] = "NIGHT" if is_night_frame else "DAY"
    proc_frame = night.enhance(frame) if (is_night_frame and toggles["night_enhance"]) else frame
    annotated = proc_frame.copy()

    detections = []
    if toggles["detection"]:
        detections = detection.run(proc_frame)

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
            _draw_trail(annotated, state.track_history[det["id"]], color)

            if toggles["activity"]:
                flag = activity.analyze(state.track_history[det["id"]])
                if flag and state.should_alert(det["id"], flag, config.ALERT_COOLDOWN_SECONDS):
                    snap = _encode_snapshot(annotated, det["box"])
                    verb = "is loitering" if flag == "LOITERING" else "is moving unusually fast"
                    state.add_event(
                        flag,
                        f"{det['label'].capitalize()} #{det['id']} {verb}",
                        snapshot_b64=snap,
                    )

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
        for f in face.detect(proc_frame):
            _draw_box(annotated, f["box"], COLOR_FACE, f"face {f['conf']:.2f}", thickness=1)

    if toggles["anpr"] and frame_idx % config.ANPR_FRAME_INTERVAL == 0:
        for plate in anpr.detect_plates(proc_frame):
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

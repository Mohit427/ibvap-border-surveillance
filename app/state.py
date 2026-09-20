"""Shared, thread-safe application state.

The camera worker thread writes to this object every processed frame; the
FastAPI event loop (WebSocket/REST handlers) reads and occasionally writes
to it (toggles, fence polygon, source changes). All access goes through
`state.lock` to keep the two sides consistent.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Optional


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val == "1"


DEFAULT_TOGGLES = {
    "detection": True,
    # ANPR (EasyOCR) is by far the heaviest module and the one most likely
    # to stall on a constrained shared-CPU host (e.g. Render's free tier) -
    # default it off there via IBVAP_ANPR_DEFAULT=0 (see render.yaml).
    # Still toggleable live from the dashboard.
    "anpr": _env_bool("IBVAP_ANPR_DEFAULT", True),
    "face": True,
    "fence": True,
    "activity": True,
    "night_enhance": True,
}

MAX_EVENTS = 300


class AppState:
    def __init__(self) -> None:
        self.lock = threading.RLock()

        self.toggles: dict[str, bool] = dict(DEFAULT_TOGGLES)

        # Fence polygon as normalized (0-1) (x, y) points, drawn by the operator.
        self.fence_polygon: list[list[float]] = []

        # Requested source change, consumed by the camera worker.
        # One of: None, {"type": "webcam"}, {"type": "sample"}, {"type": "upload", "path": str}
        self.pending_source_change: Optional[dict] = None
        self.current_source: str = "starting"

        self.latest_jpeg: Optional[bytes] = None
        self.frame_w = 0
        self.frame_h = 0

        self.stats = {
            "people_count": 0,
            "vehicle_count": 0,
            "plates_read": 0,
            "active_alerts": 0,
            "mode": "DAY",
            "fps": 0.0,
            "source": "starting",
        }

        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self._event_seq = 0

        # track_id -> deque[(timestamp, cx, cy)]
        self.track_history: dict[int, deque] = {}

        # (track_id, alert_type) -> last emitted timestamp, for cooldown
        self._alert_cooldown: dict[tuple, float] = {}

    # ---- toggles ----
    def set_toggle(self, name: str, value: bool) -> None:
        with self.lock:
            if name in self.toggles:
                self.toggles[name] = bool(value)

    def get_toggles(self) -> dict:
        with self.lock:
            return dict(self.toggles)

    # ---- fence ----
    def set_fence(self, points: list[list[float]]) -> None:
        with self.lock:
            self.fence_polygon = points

    def get_fence(self) -> list[list[float]]:
        with self.lock:
            return list(self.fence_polygon)

    # ---- source ----
    def request_source(self, change: dict) -> None:
        with self.lock:
            self.pending_source_change = change

    # ---- frame ----
    def set_frame(self, jpeg_bytes: bytes, w: int, h: int) -> None:
        with self.lock:
            self.latest_jpeg = jpeg_bytes
            self.frame_w = w
            self.frame_h = h

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    # ---- events ----
    def add_event(self, event_type: str, message: str, snapshot_b64: Optional[str] = None,
                  extra: Optional[dict] = None) -> None:
        with self.lock:
            self._event_seq += 1
            event = {
                "seq": self._event_seq,
                "type": event_type,
                "message": message,
                "timestamp": time.time(),
                "snapshot": snapshot_b64,
            }
            if extra:
                event.update(extra)
            self.events.append(event)

    def should_alert(self, track_id: Any, alert_type: str, cooldown_s: float) -> bool:
        """Rate-limit repeat alerts for the same track/type pair."""
        key = (track_id, alert_type)
        now = time.time()
        with self.lock:
            last = self._alert_cooldown.get(key, 0.0)
            if now - last >= cooldown_s:
                self._alert_cooldown[key] = now
                return True
            return False

    def get_events_since(self, last_seq: int, limit: int = 50) -> list[dict]:
        with self.lock:
            items = [e for e in self.events if e["seq"] > last_seq]
            return items[-limit:]

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        with self.lock:
            return list(self.events)[-limit:]

    # ---- stats snapshot ----
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "toggles": dict(self.toggles),
                "stats": dict(self.stats),
                "fence": list(self.fence_polygon),
                "frame_size": [self.frame_w, self.frame_h],
            }


state = AppState()

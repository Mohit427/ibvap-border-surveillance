"""Suspicious activity heuristics (part of module 5).

Both heuristics are derived purely from the real tracked centroid history
kept in `state.track_history` (populated by the pipeline from YOLO+ByteTrack
output) - nothing here is randomly triggered.

- LOITERING: a tracked person's centroid has stayed within a small radius
  for at least LOITER_WINDOW_SECONDS.
- FAST_MOVEMENT: a tracked object's instantaneous speed between the last two
  observed centroids exceeds FAST_MOVEMENT_PX_PER_SEC (in processed-frame
  pixel space).
"""
from __future__ import annotations

import math

from app import config


def analyze(history) -> str | None:
    """history: deque[(timestamp, cx, cy)] for one track, oldest first."""
    if len(history) < 2:
        return None

    now_t, now_x, now_y = history[-1]

    # Fast movement: speed over the most recent step.
    prev_t, prev_x, prev_y = history[-2]
    dt = max(now_t - prev_t, 1e-3)
    speed = math.hypot(now_x - prev_x, now_y - prev_y) / dt
    if speed > config.FAST_MOVEMENT_PX_PER_SEC:
        return "FAST_MOVEMENT"

    # Loitering: bounding spread of all points within the trailing window
    # stays inside a small radius for the whole window.
    window = [p for p in history if now_t - p[0] <= config.LOITER_WINDOW_SECONDS]
    if len(window) >= 2 and (window[-1][0] - window[0][0]) >= config.LOITER_WINDOW_SECONDS:
        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        if spread < config.LOITER_RADIUS_PX:
            return "LOITERING"

    return None

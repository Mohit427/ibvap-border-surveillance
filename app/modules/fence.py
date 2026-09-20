"""Virtual fence intrusion detection (module 4).

The operator draws a polygon (normalized 0-1 coordinates) on the video in
the browser. Any tracked person/vehicle whose centroid falls inside that
polygon is flagged as an intrusion.
"""
from __future__ import annotations

import cv2
import numpy as np


def polygon_to_px(polygon_norm: list[list[float]], w: int, h: int) -> np.ndarray:
    return np.array([[x * w, y * h] for x, y in polygon_norm], dtype=np.int32)


def point_inside(polygon_px: np.ndarray, point: tuple[float, float]) -> bool:
    if polygon_px is None or len(polygon_px) < 3:
        return False
    return cv2.pointPolygonTest(polygon_px, point, False) >= 0


def check_intrusions(detections: list[dict], polygon_norm: list[list[float]], w: int, h: int) -> list[dict]:
    if not polygon_norm or len(polygon_norm) < 3:
        return []
    polygon_px = polygon_to_px(polygon_norm, w, h)
    intruders = []
    for det in detections:
        if point_inside(polygon_px, det["centroid"]):
            intruders.append(det)
    return intruders

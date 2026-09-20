"""Night-time detection & enhancement (part of module 5).

A simple, explainable brightness heuristic: if the mean pixel intensity of
the frame drops below a threshold we tag the frame as NIGHT and apply CLAHE
contrast enhancement (on the luminance channel) before running the other
detectors, which noticeably helps YOLO/face/plate detection on dim footage.
"""
from __future__ import annotations

import cv2

from app import config

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def mean_brightness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_night(frame, threshold: float = config.NIGHT_BRIGHTNESS_THRESHOLD) -> bool:
    return mean_brightness(frame) < threshold


def enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Mild additional gain so the stream is visibly brighter for the operator.
    return cv2.convertScaleAbs(enhanced, alpha=1.25, beta=15)

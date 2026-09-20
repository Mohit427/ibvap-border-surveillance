"""Face detection (module 3) - detection only, no identity matching.

Uses OpenCV's bundled YuNet detector (a small ONNX model, downloaded once
by scripts/download_models.py). Falls back to OpenCV's built-in Haar
cascade if the YuNet weights are missing, so the module still works even
if the download step was skipped.
"""
from __future__ import annotations

import cv2

from app import config

_yunet = None
_haar = None
_backend = None


def _load():
    global _yunet, _haar, _backend
    if _backend is not None:
        return
    if config.FACE_DETECTOR_WEIGHTS.exists():
        _yunet = cv2.FaceDetectorYN.create(
            str(config.FACE_DETECTOR_WEIGHTS), "", (320, 320),
            score_threshold=config.FACE_SCORE_THRESHOLD,
        )
        _backend = "yunet"
    else:
        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _backend = "haar"


def detect(frame):
    _load()
    h, w = frame.shape[:2]
    results = []

    if _backend == "yunet":
        _yunet.setInputSize((w, h))
        _, faces = _yunet.detect(frame)
        if faces is not None:
            for f in faces:
                x, y, fw, fh = f[0:4]
                conf = float(f[-1])
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + fw), min(h, y + fh)
                results.append({"box": (x1, y1, x2, y2), "conf": conf})
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, fw, fh) in faces:
            results.append({"box": (float(x), float(y), float(x + fw), float(y + fh)), "conf": 1.0})

    return results

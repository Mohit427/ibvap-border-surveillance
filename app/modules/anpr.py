"""Automatic Number Plate Recognition (module 2).

Two stages:
1. A YOLOv8 model fine-tuned for license-plate detection (open pretrained
   weights, downloaded once by scripts/download_models.py) finds plate
   bounding boxes anywhere in the frame.
2. EasyOCR reads the text out of each cropped plate.

This is the most expensive module (a second YOLO forward pass + OCR), so
the pipeline only calls `detect_plates` every config.ANPR_FRAME_INTERVAL
frames rather than on every frame.
"""
from __future__ import annotations

import logging

import cv2

from app import config

logger = logging.getLogger("ibvap.anpr")

_plate_model = None
_ocr_reader = None


def _get_plate_model():
    global _plate_model
    if _plate_model is None and config.PLATE_DETECTOR_WEIGHTS.exists():
        from ultralytics import YOLO
        _plate_model = YOLO(str(config.PLATE_DETECTOR_WEIGHTS))
    return _plate_model


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def _ocr_plate(crop) -> tuple[str | None, float]:
    if crop is None or crop.size == 0:
        return None, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    reader = _get_ocr_reader()
    results = reader.readtext(gray)
    if not results:
        return None, 0.0
    text = "".join(r[1] for r in results).upper().replace(" ", "")
    text = "".join(ch for ch in text if ch.isalnum())
    if not text:
        return None, 0.0
    conf = sum(r[2] for r in results) / len(results)
    return text, float(conf)


def detect_plates(frame) -> list[dict]:
    """Returns [{box, text, ocr_conf, det_conf}] for every plausible plate."""
    model = _get_plate_model()
    if model is None:
        return []

    results = model.predict(frame, conf=config.PLATE_DETECT_CONF, verbose=False)
    plates = []
    if not results:
        return plates

    boxes = results[0].boxes
    if boxes is None:
        return plates

    h, w = frame.shape[:2]
    for box, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        crop = frame[y1:y2, x1:x2]
        text, ocr_conf = _ocr_plate(crop)
        plates.append({
            "box": (x1, y1, x2, y2),
            "text": text,
            "ocr_conf": ocr_conf,
            "det_conf": float(conf),
        })
    return plates

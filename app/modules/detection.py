"""Human & vehicle detection + tracking (module 1).

Uses a pretrained YOLOv8n (COCO) model via Ultralytics' built-in ByteTrack
integration so every detected person/vehicle keeps a persistent ID across
frames, which the fence and activity modules build on.
"""
from __future__ import annotations

import logging

from app import config

logger = logging.getLogger("ibvap.detection")

_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        weights = config.YOLO_DETECTION_WEIGHTS
        _model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")
    return _model


def run(frame):
    """Run detection+tracking on a frame. Returns a list of detections:
    {id, cls_id, label, kind ("person"|"vehicle"), conf, box (x1,y1,x2,y2), centroid (cx,cy)}
    """
    model = get_model()
    track_kwargs = dict(
        persist=True,
        classes=config.TRACKED_CLASS_IDS,
        conf=config.DETECTION_CONF,
        verbose=False,
        tracker="bytetrack.yaml",
    )
    if config.YOLO_IMGSZ:
        track_kwargs["imgsz"] = config.YOLO_IMGSZ
    results = model.track(frame, **track_kwargs)

    detections = []
    if not results:
        return detections

    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return detections

    ids = boxes.id.int().tolist()
    clss = boxes.cls.int().tolist()
    confs = boxes.conf.tolist()
    xyxy = boxes.xyxy.tolist()

    for track_id, cls_id, conf, box in zip(ids, clss, confs, xyxy):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cls_id == config.PERSON_CLASS_ID:
            kind, label = "person", "person"
        else:
            kind, label = "vehicle", config.VEHICLE_CLASS_IDS.get(cls_id, "vehicle")
        detections.append({
            "id": track_id,
            "cls_id": cls_id,
            "label": label,
            "kind": kind,
            "conf": float(conf),
            "box": (float(x1), float(y1), float(x2), float(y2)),
            "centroid": (cx, cy),
        })
    return detections

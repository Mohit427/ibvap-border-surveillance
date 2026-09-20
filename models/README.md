# models/

Pretrained weights downloaded by `python scripts/download_models.py`
(git-ignored, ~50-100MB total):

- `yolov8n.pt` — COCO person/vehicle detector (Ultralytics)
- `plate_detector.pt` — YOLOv8 license-plate detector (community weights, see README known limitations)
- `face_detection_yunet_2023mar.onnx` — OpenCV YuNet face detector

If a download fails (e.g. no internet at setup time), the related module
falls back to a degraded mode (face detection uses OpenCV's built-in Haar
cascade) or stays disabled (ANPR) until you place a compatible file here.

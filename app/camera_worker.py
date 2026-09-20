"""Background thread that owns the OpenCV VideoCapture, runs the analytics
pipeline on every frame, and publishes the annotated JPEG + stats into the
shared AppState. Runs outside the asyncio event loop so heavy CV work never
blocks the FastAPI server.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2

from app import config, pipeline, video_source
from app.state import AppState

logger = logging.getLogger("ibvap.camera_worker")


def _resize_keep_width(frame, target_w: int):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / w
    return cv2.resize(frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)


class CameraWorker(threading.Thread):
    def __init__(self, state: AppState):
        super().__init__(daemon=True, name="camera-worker")
        self.state = state
        self._stop_event = threading.Event()
        self.cap = None
        self.source_label = "starting"
        self.frame_idx = 0

    def stop(self):
        self._stop_event.set()

    def _open(self, request=None):
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap, self.source_label = video_source.open_source(request)
        except RuntimeError as exc:
            logger.error(str(exc))
            self.cap, self.source_label = None, "unavailable"
        self.state.current_source = self.source_label
        self.state.stats["source"] = self.source_label
        logger.info("Video source: %s", self.source_label)

    def run(self):
        self._open({"type": "auto"})
        target_dt = 1.0 / config.TARGET_FPS
        fps_ema = 0.0

        while not self._stop_event.is_set():
            t0 = time.time()

            with self.state.lock:
                pending = self.state.pending_source_change
                self.state.pending_source_change = None
            if pending:
                self._open(pending)

            if self.cap is None:
                time.sleep(1.0)
                self._open({"type": "auto"})
                continue

            ok, frame = self.cap.read()
            if not ok:
                if self.source_label.startswith(("sample", "upload")):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Lost source %s, reopening", self.source_label)
                time.sleep(0.5)
                self._open({"type": "auto"})
                continue

            frame = _resize_keep_width(frame, config.PROCESS_WIDTH)

            try:
                annotated = pipeline.process_frame(frame, self.state, self.frame_idx)
            except Exception:
                logger.exception("Pipeline error, showing raw frame")
                annotated = frame

            ok2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            if ok2:
                h, w = annotated.shape[:2]
                self.state.set_frame(buf.tobytes(), w, h)

            self.frame_idx += 1
            elapsed = time.time() - t0
            fps_ema = (0.9 * fps_ema + 0.1 * (1.0 / max(elapsed, 1e-6)))
            self.state.stats["fps"] = round(fps_ema, 1)

            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

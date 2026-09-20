"""Video input handling: webcam auto-detect with graceful fallback.

Priority when no explicit request is made: try the default webcam; if it
can't be opened (or can't deliver a frame), fall back to the bundled
sample video. The operator can also explicitly pick a source from the UI.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

from app import config

logger = logging.getLogger("ibvap.video_source")


def _try_webcam(index: int = 0):
    # CAP_DSHOW is a Windows-only backend and noticeably faster to open there
    # than the default; on other platforms (e.g. a Linux container with no
    # camera device at all) we let OpenCV pick, which just fails to open
    # quickly and falls back to the sample video below.
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def _open_file(path: Path):
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def open_source(request: dict | None) -> tuple[cv2.VideoCapture, str]:
    """Returns (capture, source_label). Raises RuntimeError only if even the
    bundled sample video is unavailable."""
    req_type = (request or {}).get("type", "auto")

    if req_type in ("webcam", "auto"):
        cap = _try_webcam()
        if cap is not None:
            return cap, "webcam"
        if req_type == "webcam":
            logger.warning("Webcam requested but unavailable, falling back to sample video")

    if req_type == "upload":
        upload_path = Path(request["path"])
        cap = _open_file(upload_path)
        if cap is not None:
            return cap, f"upload:{upload_path.name}"
        logger.warning("Upload %s unavailable, falling back to sample video", upload_path)

    # "sample" or fallback from any of the above.
    sample_path = Path(request["path"]) if req_type == "sample" and "path" in (request or {}) else config.DEFAULT_SAMPLE_VIDEO
    cap = _open_file(sample_path)
    if cap is not None:
        return cap, f"sample:{sample_path.name}"

    raise RuntimeError(
        f"No video source available: webcam failed and sample video missing at {sample_path}. "
        "Run `python scripts/generate_sample_clips.py` or place a video there."
    )


def list_sample_videos() -> list[str]:
    if not config.SAMPLE_VIDEOS_DIR.exists():
        return []
    return sorted(p.name for p in config.SAMPLE_VIDEOS_DIR.glob("*.mp4"))

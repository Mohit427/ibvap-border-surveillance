"""One-time setup: populate data/sample_videos/ so the app has a guaranteed
fallback feed even with no webcam.

Two real, freely-licensed clips (Mixkit Stock Video license: free for
commercial & personal use) are downloaded because pretrained YOLOv8 needs
real pedestrians/vehicles to actually detect anything - synthetic cartoon
shapes would not trigger real detections. A third clip (a plate close-up,
the way a dedicated ANPR camera at a checkpoint is framed) and a darkened
night variant of the pedestrian clip are generated locally with OpenCV so
the ANPR and night-mode demos are guaranteed to work offline.

Run: python scripts/generate_sample_clips.py
"""
from __future__ import annotations

import random
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "sample_videos"

# Mixkit Stock Video License - free for commercial & personal use, no
# attribution required. Credited here anyway: mixkit.co
REAL_CLIPS = [
    ("pedestrian_crossing.mp4", "https://assets.mixkit.co/videos/4832/4832-720.mp4",
     "Man dressed in suit crossing the street - mixkit.co"),
    ("busy_street.mp4", "https://assets.mixkit.co/videos/4000/4000-720.mp4",
     "Busy street in the city - mixkit.co"),
]


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = resp.read(1 << 16)
            while chunk:
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {downloaded * 100 // total}%", end="", flush=True)
                chunk = resp.read(1 << 16)
        print()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  failed: {exc}")
        return False


def generate_plate_clip(dest: Path, width=640, height=400, fps=15):
    """Simulates a fixed ANPR checkpoint camera framed tightly on the rear
    bumper/plate of successive vehicles - a realistic camera placement for
    border/toll ANPR, and one that guarantees a legible plate for OCR."""
    plates = ["KA01AB1234", "DL9CAF7788"]
    seconds_each = 6
    frames_each = seconds_each * fps

    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for plate_text in plates:
        for i in range(frames_each):
            frame = np.full((height, width, 3), (35, 35, 38), dtype=np.uint8)
            # bumper gradient
            for y in range(height):
                shade = 45 + int(25 * (y / height))
                frame[y, :] = (shade, shade, shade + 3)

            plate_w, plate_h = int(width * 0.42), int(width * 0.42 / 4.7)
            jitter_x = random.randint(-2, 2)
            jitter_y = random.randint(-1, 1)
            px = (width - plate_w) // 2 + jitter_x
            py = (height - plate_h) // 2 + jitter_y

            cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (235, 235, 235), -1)
            cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (20, 20, 20), 3)

            font_scale = plate_w / 260.0
            (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 3)
            tx = px + (plate_w - tw) // 2
            ty = py + (plate_h + th) // 2
            cv2.putText(frame, plate_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, (15, 15, 15), 3, cv2.LINE_AA)

            noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            writer.write(frame)
    writer.release()
    print(f"  wrote {dest.name}")


def generate_night_variant(src: Path, dest: Path, darkness=0.22):
    if not src.exists():
        print(f"  skip night variant: {src.name} missing")
        return
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dark = cv2.convertScaleAbs(frame, alpha=darkness, beta=0)
        noise = np.random.normal(0, 6, dark.shape).astype(np.int16)
        dark = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(dark)

    cap.release()
    writer.release()
    print(f"  wrote {dest.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url, credit in REAL_CLIPS:
        dest = OUT_DIR / filename
        if dest.exists() and dest.stat().st_size > 500_000:
            print(f"[skip] {filename} already present")
            continue
        print(f"[get]  {filename} ({credit})")
        if not download(url, dest):
            print(f"  WARNING: could not fetch {filename}. Fallback demo footage for "
                  f"people/vehicle detection will be missing; place a video at {dest} manually.")

    plate_dest = OUT_DIR / "anpr_plate_closeup.mp4"
    if not plate_dest.exists():
        print("[gen]  anpr_plate_closeup.mp4 (synthetic ANPR checkpoint close-up)")
        generate_plate_clip(plate_dest)
    else:
        print("[skip] anpr_plate_closeup.mp4 already present")

    night_dest = OUT_DIR / "night_intrusion.mp4"
    if not night_dest.exists():
        print("[gen]  night_intrusion.mp4 (darkened pedestrian clip, for night-mode demo)")
        generate_night_variant(OUT_DIR / "pedestrian_crossing.mp4", night_dest)
    else:
        print("[skip] night_intrusion.mp4 already present")

    print("\nSample clips ready in data/sample_videos/. The app defaults to "
          "pedestrian_crossing.mp4 when no webcam is available; switch clips from the dashboard.")


if __name__ == "__main__":
    sys.exit(main())

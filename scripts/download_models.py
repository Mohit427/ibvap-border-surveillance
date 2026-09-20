"""One-time setup: download the pretrained open-source model weights the
prototype needs, into models/. This is the only step that needs internet
access - once downloaded, the app runs fully offline.

Run: python scripts/download_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# (destination filename, list of candidate URLs tried in order, min expected bytes)
DOWNLOADS = [
    (
        "plate_detector.pt",
        [
            "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt",
            "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
        ],
        1_000_000,
    ),
    (
        "face_detection_yunet_2023mar.onnx",
        [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        100_000,
    ),
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
                    pct = downloaded * 100 // total
                    print(f"\r  {dest.name}: {pct}% ({downloaded // 1024} KB)", end="", flush=True)
                chunk = resp.read(1 << 16)
        print()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  failed: {exc}")
        return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, urls, min_bytes in DOWNLOADS:
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"[skip] {filename} already present")
            continue
        print(f"[get]  {filename}")
        ok = False
        for url in urls:
            print(f"  trying {url}")
            if download(url, dest) and dest.stat().st_size >= min_bytes:
                ok = True
                break
        if not ok:
            print(f"  WARNING: could not download {filename}. The related module will "
                  f"fall back to a degraded mode or stay disabled. You can manually place "
                  f"a compatible file at {dest}.")

    print("\nyolov8n.pt (person/vehicle detector) is fetched automatically by ultralytics "
          "on first run if not already cached.")
    try:
        from ultralytics import YOLO
        yolo_dest = MODELS_DIR / "yolov8n.pt"
        if not yolo_dest.exists():
            print("[get]  yolov8n.pt")
            model = YOLO("yolov8n.pt")
            src = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else None
            if src and src.exists() and src != yolo_dest:
                yolo_dest.write_bytes(src.read_bytes())
        else:
            print("[skip] yolov8n.pt already present")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not pre-fetch yolov8n.pt now ({exc}). "
              f"It will be downloaded automatically the first time detection runs.")

    print("\nDone. Run `python scripts/generate_sample_clips.py` next if you don't "
          "already have data/sample_videos populated, then `python run.py`.")


if __name__ == "__main__":
    sys.exit(main())

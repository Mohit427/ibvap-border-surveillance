# IBVAP Prototype
**Intelligent Border Video Analytics Platform** — a local, offline-capable
CCTV analytics dashboard built for SIH 2026 (MHA problem statement:
*AI-Based Intelligent Video Analytics Platform for Border Surveillance
using existing CCTV Infrastructure*).

It ingests a live webcam or a video file, runs five real CV pipelines on
every frame (not mocked), and streams the annotated feed plus a live event
log to a dark, command-center style dashboard in your browser.

**Live demo (sample clips):** _add your Render URL here once deployed —
see [Live demo deployment](#live-demo-deployment) below._
> A hosted container has no camera, so the demo link only runs against the
> bundled sample clips / an uploaded file. The live-webcam experience below
> is what you run locally for the actual pitch.

## Setup

Requires Python 3.10+ (tested on 3.14) on Windows.

```powershell
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_models.py        # one-time: pretrained weights (needs internet)
python scripts/generate_sample_clips.py   # one-time: bundled fallback demo footage
```

## Run

```powershell
python run.py
```

Opens the dashboard at **http://localhost:8000**. Stop with Ctrl+C.

The app auto-detects a webcam on startup; if none is found (or it's
unplugged), it falls back to the bundled sample clip automatically. You can
switch source (webcam / bundled sample / your own uploaded file) from the
sidebar at any time, live.

<details>
<summary>Or run it in Docker (no camera access, sample clips only)</summary>

```bash
docker build -t ibvap .
docker run -p 8000:8000 -e PORT=8000 ibvap
```

Open http://localhost:8000. This is the same image the hosted demo uses —
see [Live demo deployment](#live-demo-deployment).
</details>

## What each module does

| # | Module | How it works |
|---|--------|---------------|
| 1 | **Human & vehicle detection/tracking** | YOLOv8n (pretrained COCO) + ByteTrack gives every person/vehicle a persistent ID and a drawn motion trail. Vehicles are further labelled by COCO class (car/truck/bus/motorcycle/bicycle). |
| 2 | **ANPR** | A YOLOv8 model fine-tuned for license-plate detection finds plate boxes; EasyOCR reads the text out of the crop. Logged to the event feed with timestamp. Runs every ~1s (not every frame) since it's the most CPU-expensive module. |
| 3 | **Face detection** | OpenCV's YuNet ONNX detector (falls back to a Haar cascade if the weight wasn't downloaded). Detection only — no identity matching, by design. |
| 4 | **Virtual fence** | Draw a polygon directly on the video in the dashboard. Any tracked centroid that falls inside it fires an `INTRUSION` alert with a snapshot. |
| 5 | **Night mode + suspicious activity** | *Night*: mean-brightness threshold flags dim frames, applies CLAHE contrast enhancement, and tags the UI `NIGHT`. *Suspicious activity*: two explainable heuristics computed from the real ByteTrack history — `LOITERING` (a track's centroid stays within a small radius for 8+ seconds) and `FAST_MOVEMENT` (instantaneous speed between frames exceeds a threshold). Neither is randomly triggered. |

Every module has an on/off toggle in the sidebar, live.

## Architecture

- **Backend**: FastAPI. A dedicated background thread (`app/camera_worker.py`)
  owns the `cv2.VideoCapture`, runs `app/pipeline.py` on each frame, and
  publishes the annotated JPEG + stats into a thread-safe `AppState`
  (`app/state.py`). This keeps the heavy CV work off the asyncio event loop.
- **Streaming**: `/ws/video` pushes the latest annotated JPEG as a binary
  WebSocket message; `/ws/state` pushes stats + toggle state + new events as
  JSON, both at a few Hz.
- **Frontend**: `static/index.html` + `app.js`, no framework/build step —
  a `<canvas>` overlay on the video for drawing the fence polygon, a stats
  strip, module toggles, and a scrolling event log with snapshot thumbnails.
- **Config**: all tunables (resolution, target FPS, thresholds, alert
  cooldowns) live in `app/config.py`.

## Performance trade-off (read this before a demo)

Running YOLO detection + ANPR + face detection + tracking together is heavy
for a laptop CPU. To keep it real-time-ish:
- Every frame is downscaled to `PROCESS_WIDTH` (640px wide) before any model runs.
- The pipeline targets `TARGET_FPS` (10) rather than running flat-out.
- ANPR (plate detection + OCR) only runs every `ANPR_FRAME_INTERVAL` (15) frames — it's
  the slowest module by far.

If your laptop is slow, lower `PROCESS_WIDTH` in `app/config.py` or disable
ANPR/face during the parts of the demo that don't need them.

## Known limitations

- **ANPR accuracy** depends heavily on plate size/clarity in frame; it will
  struggle on small, blurry, or angled plates, and the bundled plate
  detector was fine-tuned mostly on non-Indian plate datasets. The bundled
  `anpr_plate_closeup.mp4` clip simulates a dedicated ANPR checkpoint camera
  (tight, front-on framing) to guarantee a clean read for the demo.
- **Face detection is detection-only** — no recognition/identity matching,
  as scoped for this prototype.
- **Tracking IDs can occasionally re-assign** after long occlusion, which is
  a known ByteTrack limitation, not a bug in the fence/activity logic.
- **Night enhancement** is a straightforward CLAHE + gain boost, not a
  learned low-light model — it helps but won't recover a pitch-black frame.
- The bundled sample clips (`pedestrian_crossing.mp4`, `busy_street.mp4`)
  are real footage from Mixkit (Mixkit Stock Video License — free for
  commercial & personal use); `anpr_plate_closeup.mp4` and
  `night_intrusion.mp4` are generated locally by
  `scripts/generate_sample_clips.py`.
- Model weight downloads (`scripts/download_models.py`) need internet once,
  at setup time — after that the whole pipeline runs fully offline with no
  cloud API keys.

## Live demo deployment

The repo ships a `Dockerfile` and a `render.yaml` blueprint for
[Render](https://render.com)'s free web service tier — no credit card
required. Render builds the `Dockerfile` directly and auto-redeploys on
every push to `main`, no GitHub Actions/secrets needed.

**Deploy it:** go to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints),
connect this GitHub repo, and Render will read `render.yaml` and set the
service up on the **Free** plan automatically. (Or: New → Web Service →
connect the repo → Render auto-detects the Dockerfile.)

Two things worth knowing about the free plan:
- **512MB RAM.** torch + YOLOv8 + EasyOCR + OpenCV loaded together can run
  close to that ceiling — it may run fine, or the service may crash/restart
  under load. If it does, the fix is trimming a module (ANPR/EasyOCR is the
  single biggest consumer at ~300-500MB) out of the hosted build rather than
  switching provider again; ask if you want that done.
- **Sleeps after 15 min idle**, waking on the next visit with a cold start
  (image pull + model load) that can take a minute or so.

Note the hosted demo is a single shared session: everyone visiting the link
sees the same feed and can change the same toggles/fence, which is fine for
a demo link but isn't a multi-tenant setup.

## Mapping to the problem statement

| MHA requirement | Where it's implemented |
|---|---|
| Human & vehicle detection/tracking on existing CCTV feeds | `app/modules/detection.py`, works on any `cv2.VideoCapture` source (webcam, file, and — with a URL/RTSP capture string — an existing IP camera) |
| Automatic Number Plate Recognition | `app/modules/anpr.py` |
| Face detection | `app/modules/face.py` |
| Virtual fencing / intrusion alerts | `app/modules/fence.py` + fence canvas in the dashboard |
| Night-time / suspicious activity detection | `app/modules/night.py`, `app/modules/activity.py` |
| Real-time alerting to an operator console | `/ws/state` event feed + dashboard alert badges |
| Runs on existing infrastructure, no specialized hardware | Pure CPU pipeline, standard webcam/CCTV video input, no cloud dependency at runtime |

## Live demo script

1. **Open the dashboard** (`python run.py`, browser opens automatically).
   Point out the stats strip, module toggles, and event log — this is the
   "operator console."
2. **Human & vehicle tracking**: stand in front of the webcam (or start the
   `busy_street` sample). Show the persistent ID + trail line following you
   as you move.
3. **Virtual fence**: click "Draw Fence," outline a zone on the feed, hit
   Save. Walk into it — call out the `INTRUSION` alert appearing in the
   event log in real time, with the fence border turning red.
4. **Suspicious activity**: stand still inside/near the fence for ~8s to
   trigger `LOITERING`; then walk briskly across frame to trigger
   `FAST_MOVEMENT`. Explain both are computed from the same real tracking
   data, not scripted.
5. **Face detection**: turn to face the camera — call out the live
   confidence score on the box.
6. **ANPR**: switch source to the `anpr_plate_closeup` sample clip — point
   out the plate box and the read text appearing in the event log with a
   timestamp.
7. **Night mode**: switch to the `night_intrusion` sample clip — call out
   the `NIGHT` badge flipping and the visibly brighter enhanced feed versus
   the raw dark input.
8. **Toggle a module off live** (e.g. face detection) to show the boxes
   disappearing instantly — demonstrates the modular on/off architecture
   an operator could use to manage load.

# Raspberry Pi NCNN driveway benchmark

This test uses Qengineering's YOLOv8-nano NCNN model and detector code with:

- the existing isolated OpenCV 5 build for RTSP decoding;
- an isolated Tencent NCNN build for ARM inference;
- separate capture and inference work so detection never blocks the camera;
- no desktop or display requirement.

The Qengineering source/model files `yoloV8.cpp`, `yoloV8.h`,
`yolov8n.param`, and `yolov8n.bin` are copied from the BSD-3-Clause licensed
`Qengineering/YoloV8-ncnn-Raspberry-Pi-4` repository when deploying the test.

Configure with both package locations:

```sh
cmake -S . -B build \
  -DOpenCV_DIR="$HOME/.local/opencv-5/lib/cmake/opencv5" \
  -Dncnn_DIR="$HOME/.local/ncnn/lib/cmake/ncnn"
cmake --build build --parallel 4
```

Run from the directory containing the two model files:

```sh
CAMERA_URL='rtsps://camera-address/stream' ./build/driveway_ncnn_benchmark
```

For a temporary Pi-hosted dashboard (five detections per second):

```sh
CAMERA_URL='rtsps://camera-address/stream' ./build/driveway_ncnn_server
```

Open `http://PI_ADDRESS:8000/` in a browser. This preview is intentionally
started manually; the permanent-service step adds system startup, event
tracking, snapshots, and notifications.

## Permanent service

The permanent server runs five general NCNN detections per second while camera
capture continues independently. It can also run Microsoft's compact
MegaDetector V6 YOLOv10 wildlife model once per second. If the camera or
network disconnects, every failed
RTSP session waits for `RECONNECT_DELAY_SECONDS` before reopening, and
detection pauses until a genuinely new frame arrives. It records the first
appearance of each person or vehicle class inside `DRIVEWAY_ZONE`, waits until
the class has cleared, and uses a 60-second per-class cooldown to suppress
duplicate alerts. It writes:

- annotated JPEG snapshots to `runtime/events/`;
- a durable `runtime/events.csv` log;
- the newest events to the dashboard at `http://PI_ADDRESS:8000/`.
- optional text-only ntfy phone alerts for accepted events, using the same
  cooldown as the event log.
- wildlife alerts from any configured boundary, grouped under one cooldown.
- an optional amber mailbox zone that only records a vehicle after it remains
  nearly stationary for a configured dwell period.
- full-screen event viewing and confirmed deletion of both the snapshot and its
  history row from the dashboard.
- manual snapshots and 30-second native-stream video recordings on the
  separate recording SSD, with a local gallery, download links, deletion, and
  free-space/retention protection.
- a separate no-alert road-traffic archive that tracks passing vehicles and
  saves one snapshot per track, with its own retention limit and gallery tab.

Snapshot retention defaults to the newest 200 JPEGs. The system service is
defined in `driveway-watch.service`, restarts after failures, and starts after
the network on every headless boot. Camera credentials live only in the Pi's
private `.env` file and are not embedded in the executable or service unit.
The ntfy topic is also stored only in that private file. Set `NTFY_SERVER` and
`NTFY_TOPIC` to enable alerts; snapshots remain local to the Pi.

Person detections use a 0.25 confidence threshold, MegaDetector wildlife
detections default to 0.20, and vehicles retain the 0.40 threshold. Zone entry
is evaluated at the
bottom-center of each box (the person's feet or object's contact point), which
is more appropriate for a driveway boundary than the box center.

### Optional MegaDetector V6 wildlife model

The wildlife pass uses `MDV6-yolov10-c`, a 2.3-million-parameter camera-trap
model with animal, person, and vehicle categories. The service consumes only
its animal output; the existing YOLOv8n detector remains responsible for
people and detailed vehicle classes.

The model is not committed to this repository. Export the official
`MDV6-yolov10-c.pt` weight with Ultralytics 8.4.104:

```sh
yolo export model=MDV6-yolov10-c.pt format=ncnn imgsz=640 device=cpu
mkdir -p wildlife-model
cp MDV6-yolov10-c_ncnn_model/model.ncnn.param wildlife-model/
cp MDV6-yolov10-c_ncnn_model/model.ncnn.bin wildlife-model/
```

Then set `WILDLIFE_MODEL_ENABLED=true`. `WILDLIFE_DETECTION_FPS=1` is the
recommended Pi 5 starting point. The status API and dashboard show whether the
model loaded and its most recent inference time. The weight and Ultralytics
export are AGPL-3.0 licensed; the project-authored integration remains MIT.

MegaDetector detects wildlife but does not name species. The optional DFNE
classifier provides a second stage tuned for northeastern North America. The
C++ service sends only an in-memory JPEG crop to a localhost-only classifier;
the crop is not written to disk or uploaded. Low-confidence, `no-species`, and
human classifications are rejected before event creation.

Install the classifier in an isolated environment. Install the CPU-only
PyTorch packages first so pip does not download unusable NVIDIA libraries:

```sh
python3 -m venv .wildlife-venv
.wildlife-venv/bin/python -m pip install --upgrade pip
.wildlife-venv/bin/python -m pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
.wildlife-venv/bin/python -m pip install \
  -r cpp_pi_ncnn/wildlife-classifier-requirements.txt
```

Run `wildlife_classifier.py` as a separate service and keep it bound to
`127.0.0.1`. The first model load downloads the DFNE weights; subsequent
classification stays local. Configure the C++ service with:

```dotenv
SPECIES_CLASSIFIER_ENABLED=true
SPECIES_CLASSIFIER_REQUIRED=true
SPECIES_CLASSIFIER_PORT=8765
SPECIES_CONFIDENCE_THRESHOLD=0.65
SPECIES_CLASSIFIER_TIMEOUT_SECONDS=8
SPECIES_CACHE_SECONDS=8
```

`wildlife-classifier.service` is the `/opt/ai-driveway-watch` deployment
template. Adjust its account and paths to match the installation. Start with
`ANIMAL_ALERTS_ENABLED=false`, verify the live overlay and `/api/status`, then
enable animal events only after local false-positive testing.

For unattended field testing, enable the separate local candidate archive:

```dotenv
WILDLIFE_TEST_CAPTURES_ENABLED=true
WILDLIFE_TEST_COOLDOWN_SECONDS=30
WILDLIFE_TEST_SNAPSHOT_RETENTION=200
GENERAL_EVENTS_ENABLED=false
```

The **Wildlife test** dashboard tab shows labeled snapshots for both confirmed
species and rejected candidates such as `no-species`. These test records never
enter the alert archive or notification queue. Each archive image, manual
snapshot, and manual video includes preview, download, and delete controls.
Set `GENERAL_EVENTS_ENABLED=false` only on a dedicated wildlife Pi to avoid
duplicating the primary Pi's driveway, mailbox, and road archives.

Planned dashboard improvement: editable zone handles over the live image with
Preview, Save, and Cancel controls, so the homeowner can redraw the boundary
without editing coordinates or restarting from a terminal.

## Raspberry Pi 5 result

Measured on the stock-clocked 16 GB Raspberry Pi 5 on July 18, 2026, using
the 1280x720, 30 FPS driveway stream:

- camera decoding: 29.7 FPS;
- YOLOv8-nano inference: 55.1 ms average, or 18.2 FPS capacity;
- full-load temperature: 58.7 C before and 68.6 C after 30 seconds;
- throttling: none (`0x0`).

This deliberately ran inference as fast as possible. A driveway service only
needs roughly 2-5 detections per second, leaving substantial capacity for event
tracking, snapshots, and notifications.

## Headless startup

The Pi now boots to `multi-user.target`; SSH and wired networking remain active,
and the installed desktop does not start. The post-reboot test decoded the
camera at 29.3 FPS and sustained 19.2 inference FPS without throttling.

To restore desktop startup later:

```sh
sudo systemctl set-default graphical.target
sudo reboot
```

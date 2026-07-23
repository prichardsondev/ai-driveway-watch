# Raspberry Pi service installation

Build and test the server manually before installing the permanent service.
The example below assumes this repository will live at
`/opt/ai-driveway-watch`.

## 1. Create the service account and directories

```bash
sudo useradd --system --home /opt/ai-driveway-watch \
  --shell /usr/sbin/nologin driveway-watch
sudo mkdir -p /opt/ai-driveway-watch
sudo chown driveway-watch:driveway-watch /opt/ai-driveway-watch
```

Copy the repository to that directory using your preferred deployment method.
Do not copy a development `.env`, runtime archive, or private event images into
the public repository.

## 2. Fetch assets and build

```bash
cd /opt/ai-driveway-watch/cpp_pi_ncnn
sudo -u driveway-watch ./fetch_qengineering_assets.sh
sudo -u driveway-watch cmake -S . -B build \
  -DOpenCV_DIR=/opt/opencv-5/lib/cmake/opencv5 \
  -Dncnn_DIR=/opt/ncnn/lib/cmake/ncnn
sudo -u driveway-watch cmake --build build --parallel 2
```

Adjust the two package paths to match the isolated OpenCV 5 and NCNN installs.

## 3. Configure privately

```bash
sudo -u driveway-watch cp ../.env.example .env
sudo chmod 600 .env
sudo -u driveway-watch mkdir -p runtime/events runtime/road_events
```

Edit `.env` locally on the Pi. Never paste the completed file into an issue or
agent transcript. Set at least `CAMERA_URL` and the three zone polygons.

## 4. Install and verify

```bash
sudo cp driveway-watch.service /etc/systemd/system/driveway-watch.service
sudo systemctl daemon-reload
sudo systemctl enable --now driveway-watch.service
systemctl status driveway-watch.service
curl --fail http://127.0.0.1:8000/api/status
```

Open `http://PI_ADDRESS:8000/` from the trusted LAN and verify all overlays.

## Optional regional wildlife classifier

The two-stage wildlife path uses the compact NCNN MegaDetector as a fast gate
and a persistent DFNE classifier for northeastern North American species.
Follow the classifier installation in
[`cpp_pi_ncnn/README.md`](../cpp_pi_ncnn/README.md), install
`wildlife-classifier.service`, and keep it listening only on
`127.0.0.1:8765`.

Leave `ANIMAL_ALERTS_ENABLED=false` during initial testing. Confirm
`species_classifier_ready`, the last species label and confidence, and a zero
event count through `/api/status` before enabling animal events.

## Rollback

Keep the previously working binary or source before updating. To stop and
disable the service without deleting configuration or event history:

```bash
sudo systemctl disable --now driveway-watch.service
```

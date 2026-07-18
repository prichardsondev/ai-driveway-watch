# Agent guide

## Mission

Keep AI Driveway Watch local-first, understandable, and safe for household use.
The production path is the C++ service in `cpp_pi_ncnn/`; the Python code is an
earlier prototype.

## Non-negotiable privacy rules

- Never print, commit, upload, or repeat `.env` values.
- Treat camera URLs, ntfy topics, LAN addresses, event images, and household
  names as private data.
- Never commit `runtime/`, snapshots, event CSV files, model binaries, or local
  build output.
- Never add internet exposure or router port forwarding as a default.
- ntfy road-traffic events must remain disabled; only accepted driveway and
  mailbox events may enter the notification queue.

## Architecture invariants

- Camera capture, inference, notification delivery, and HTTP serving remain
  independent so network delays cannot stall video processing.
- Zone tests use the bottom-center of the detection box.
- Driveway events are debounced by presence and cooldown.
- Mailbox events require dwell time plus limited movement.
- Road events use short-lived position tracks and save once per track.
- Event deletion removes both the JPEG and its CSV history row and requires a
  UI confirmation.

## Working locally

```bash
pytest -q
```

The production C++ build requires OpenCV 5, NCNN, and the pinned Qengineering
assets:

```bash
cd cpp_pi_ncnn
./fetch_qengineering_assets.sh
cmake -S . -B build \
  -DOpenCV_DIR="$HOME/.local/opencv-5/lib/cmake/opencv5" \
  -Dncnn_DIR="$HOME/.local/ncnn/lib/cmake/ncnn"
cmake --build build --parallel 2
```

## Deployment safety

Before restarting a live service:

1. back up the current binary or source;
2. compile successfully on the target;
3. preserve its private `.env` and runtime archive;
4. restart only `driveway-watch.service`;
5. verify `/api/status`, both event APIs, the live overlay, and ntfy error state;
6. never create or delete a real event merely to test the UI.

## Definition of done

- no secret or private image appears in the diff;
- Python tests pass when affected;
- the C++ target builds on the Pi when affected;
- the dashboard remains usable on a narrow phone screen;
- the service survives a restart with its event history intact;
- README, `.env.example`, and agent prompts reflect new configuration.

# Raspberry Pi 5 C++ benchmark

This benchmark compares the same OpenCV 5.0.0 INT8 YOLOX model and 1280x720 RTSPS camera used by the laptop prototype.

Pi tested:

- Raspberry Pi 5 Model B Rev 1.1
- 16 GB RAM and NVMe storage
- Debian 13 arm64
- Stock 2.4 GHz Cortex-A76, four cores
- OpenCV 5.0.0 built locally with NEON, FP16/dot-product dispatch, ARM64 MLAS assembly, KleidiCV, and FFmpeg

OpenCV is isolated at `$HOME/.local/opencv-5`; no system OpenCV installation is replaced.

## Results

Same 640x640 YOLOX INT8 model, four OpenCV threads:

| OpenCV DNN engine | Inference latency | Capacity | Capture rate in single-thread benchmark |
| --- | ---: | ---: | ---: |
| Classic | 651 ms | 1.5 FPS | 21.2 FPS |
| New graph engine | 1,283 ms | 0.8 FPS | 11.2 FPS |

The 30-second default-engine run reached 71 C with no throttling. A production service should use independent latest-frame capture and inference threads so slow inference never blocks camera ingest.

## Build

```bash
cmake -S . -B build \
  -D CMAKE_BUILD_TYPE=Release \
  -D OpenCV_DIR="$HOME/.local/opencv-5/lib/cmake/opencv5"
cmake --build build --parallel 4
```

Keep the private camera URL in an ignored `.env` file. Force the faster classic engine for this model:

```bash
export CAMERA_URL="..."
export LD_LIBRARY_PATH="$HOME/.local/opencv-5/lib"
OPENCV_FORCE_DNN_ENGINE=1 ./build/driveway_benchmark \
  ./models/object_detection_yolox_2022nov_int8.onnx 30
```

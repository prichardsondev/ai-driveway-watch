#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  bootstrap_file="$(mktemp /tmp/get-pip.XXXXXX.py)"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$bootstrap_file"
  .venv/bin/python "$bootstrap_file"
  rm -f "$bootstrap_file"
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
mkdir -p models
if [ ! -f models/object_detection_yolox_2022nov_int8.onnx ]; then
  curl -fL --retry 3 -o models/object_detection_yolox_2022nov_int8.onnx \
    https://huggingface.co/opencv/opencv_zoo/resolve/main/models/object_detection_yolox/object_detection_yolox_2022nov_int8.onnx
fi
echo "Setup complete. OpenCV version: $(.venv/bin/python -c 'import cv2; print(cv2.__version__)')"

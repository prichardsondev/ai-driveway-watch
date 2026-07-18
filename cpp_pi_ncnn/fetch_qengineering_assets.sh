#!/usr/bin/env bash
set -euo pipefail

upstream_commit="3342965252320bb4aca1835efbd115bef4653034"
base_url="https://raw.githubusercontent.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4/${upstream_commit}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
asset_tmp_dir="$(mktemp -d)"
trap 'rm -rf "${asset_tmp_dir}"' EXIT

files=(yoloV8.cpp yoloV8.h yolov8n.bin yolov8n.param LICENSE)
for file in "${files[@]}"; do
    curl --fail --location --show-error --silent \
        "${base_url}/${file}" --output "${asset_tmp_dir}/${file}"
done

(
    cd "${asset_tmp_dir}"
    sha256sum --check <<'CHECKSUMS'
3e9e9a8c6998b06d38ecb6d150937801e904078b03f3781f63303878886a1e13  yoloV8.cpp
d41681415b10633bc1fb3397170f23cb35b8234aa456c30f32095eab8ac55770  yoloV8.h
ae8ea22b22c9fec92c7197f3b293e4badb459ee31a08dacad3331f660dc81cbb  yolov8n.bin
08243613c6519a892b933810ad4d377d56ecd4c112ab8a7f637e82d91f3cdff7  yolov8n.param
ac24c9686451ed6cf5e0a362e387bebf243eb21b367a05c91a1df265a34d184c  LICENSE
CHECKSUMS
)

install -m 0644 "${asset_tmp_dir}/yoloV8.cpp" "${script_dir}/yoloV8.cpp"
install -m 0644 "${asset_tmp_dir}/yoloV8.h" "${script_dir}/yoloV8.h"
install -m 0644 "${asset_tmp_dir}/yolov8n.bin" "${script_dir}/yolov8n.bin"
install -m 0644 "${asset_tmp_dir}/yolov8n.param" "${script_dir}/yolov8n.param"
install -m 0644 "${asset_tmp_dir}/LICENSE" "${script_dir}/LICENSE.qengineering"

echo "Installed pinned Qengineering detector assets in ${script_dir}"

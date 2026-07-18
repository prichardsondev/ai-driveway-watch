from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def _number(name: str, default: float, cast):
    value = os.getenv(name, "").strip()
    return cast(value) if value else cast(default)


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _classes(name: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in os.getenv(name, "").split(",") if item.strip())


def parse_zone(value: str) -> tuple[tuple[float, float], ...]:
    try:
        points = tuple(
            (float(pair.split(",")[0]), float(pair.split(",")[1]))
            for pair in value.split(";")
        )
    except (ValueError, IndexError) as exc:
        raise ValueError("DRIVEWAY_ZONE must contain x,y pairs separated by semicolons") from exc
    if len(points) < 3 or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
        raise ValueError("DRIVEWAY_ZONE needs at least 3 normalized points between 0 and 1")
    return points


@dataclass(frozen=True)
class Settings:
    camera_url: str
    source_mode: str
    model_path: Path
    confidence_threshold: float
    nms_threshold: float
    class_filter: tuple[str, ...]
    inference_interval: int
    opencv_threads: int
    driveway_zone: tuple[tuple[float, float], ...]
    event_cooldown_seconds: float
    event_clear_seconds: float
    snapshots_enabled: bool
    stream_fps: int
    capture_buffer_size: int
    reconnect_delay_seconds: float
    output_dir: Path
    web_host: str
    web_port: int


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    output = Path(os.getenv("OUTPUT_DIR", "runtime"))
    model = Path(os.getenv("MODEL_PATH", "models/object_detection_yolox_2022nov_int8.onnx"))
    if not output.is_absolute():
        output = ROOT / output
    if not model.is_absolute():
        model = ROOT / model
    return Settings(
        camera_url=os.getenv("CAMERA_URL", "").strip(),
        source_mode=os.getenv("SOURCE_MODE", "sample").strip().lower(),
        model_path=model,
        confidence_threshold=_number("CONFIDENCE_THRESHOLD", .40, float),
        nms_threshold=_number("NMS_THRESHOLD", .50, float),
        class_filter=_classes("CLASS_FILTER"),
        inference_interval=max(1, _number("INFERENCE_INTERVAL", 5, int)),
        opencv_threads=max(1, _number("OPENCV_THREADS", 4, int)),
        driveway_zone=parse_zone(os.getenv("DRIVEWAY_ZONE", "0,0;1,0;1,1;0,1")),
        event_cooldown_seconds=max(0, _number("EVENT_COOLDOWN_SECONDS", 60, float)),
        event_clear_seconds=max(0, _number("EVENT_CLEAR_SECONDS", 5, float)),
        snapshots_enabled=_flag("SNAPSHOTS_ENABLED", True),
        stream_fps=max(1, _number("STREAM_FPS", 10, int)),
        capture_buffer_size=max(1, _number("CAPTURE_BUFFER_SIZE", 1, int)),
        reconnect_delay_seconds=max(.1, _number("RECONNECT_DELAY_SECONDS", 2, float)),
        output_dir=output,
        web_host=os.getenv("WEB_HOST", "127.0.0.1"),
        web_port=_number("WEB_PORT", 8000, int),
    )

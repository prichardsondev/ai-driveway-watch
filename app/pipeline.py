from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
import time

import cv2
import numpy as np

from app.config import Settings
from app.detector import Detection, YoloXDetector
from app.events import DrivewayEvent, EventEngine


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        cv2.setNumThreads(settings.opencv_threads)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.jpeg: bytes | None = None
        self.connected = False
        self.error = ""
        self.source = "starting"
        self.frames = 0
        self.capture_fps = 0.0
        self.inference_ms: deque[float] = deque(maxlen=100)
        self.events: deque[DrivewayEvent] = deque(maxlen=50)
        self.last_detections: list[Detection] = []
        self.detector = YoloXDetector(
            settings.model_path, settings.confidence_threshold, settings.nms_threshold, settings.class_filter
        )
        self.event_engine = EventEngine(
            settings.output_dir, settings.driveway_zone, settings.event_cooldown_seconds,
            settings.event_clear_seconds, settings.snapshots_enabled,
        )

    def start(self):
        self.thread = threading.Thread(target=self._run, name="driveway-pipeline", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)

    def get_jpeg(self):
        with self.lock:
            return self.jpeg

    def status(self):
        with self.lock:
            latencies = list(self.inference_ms)
            return {
                "connected": self.connected,
                "source": self.source,
                "error": self.error,
                "opencv_version": cv2.__version__,
                "frames": self.frames,
                "capture_fps": round(self.capture_fps, 1),
                "inference_ms_last": round(latencies[-1], 1) if latencies else 0,
                "inference_ms_avg": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                "inference_interval": self.settings.inference_interval,
                "opencv_threads": cv2.getNumThreads(),
                "publish_fps": self.settings.stream_fps,
                "events": [event.__dict__ for event in self.events],
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

    def _source(self):
        if self.settings.source_mode == "sample":
            return None
        if self.settings.source_mode != "camera" or not self.settings.camera_url:
            raise ValueError("SOURCE_MODE=camera requires CAMERA_URL")
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ]
        capture = cv2.VideoCapture(self.settings.camera_url, cv2.CAP_FFMPEG, params)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.settings.capture_buffer_size)
        return capture

    def _sample(self, index: int):
        frame = np.full((540, 960, 3), (29, 34, 38), np.uint8)
        cv2.putText(frame, "OpenCV 5 sample mode", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 2)
        x = 20 + index * 7 % 700
        cv2.rectangle(frame, (x, 260), (x + 180, 390), (60, 160, 240), -1)
        time.sleep(1 / 20)
        return frame

    def _run(self):
        while not self.stop_event.is_set():
            capture = None
            try:
                capture = self._source()
                if capture is not None and not capture.isOpened():
                    raise RuntimeError("camera could not be opened")
                with self.lock:
                    self.connected = True
                    self.source = "camera" if capture is not None else "sample"
                    self.error = ""
                started = time.monotonic()
                last_publish = 0.0
                counted = 0
                while not self.stop_event.is_set():
                    if capture is None:
                        frame = self._sample(self.frames)
                    else:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError("camera stopped returning frames")
                    self.frames += 1
                    counted += 1
                    elapsed = time.monotonic() - started
                    if elapsed >= 2:
                        self.capture_fps = counted / elapsed
                        started, counted = time.monotonic(), 0
                    if self.frames % self.settings.inference_interval == 0:
                        self.last_detections, latency = self.detector.detect(frame)
                        new_events = self.event_engine.update(self.last_detections, frame)
                        with self.lock:
                            self.inference_ms.append(latency)
                            self.events.extendleft(reversed(new_events))
                    now = time.monotonic()
                    if now - last_publish >= 1 / self.settings.stream_fps:
                        annotated = self._annotate(frame, self.last_detections)
                        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if ok:
                            with self.lock:
                                self.jpeg = jpeg.tobytes()
                        last_publish = now
            except Exception as exc:
                with self.lock:
                    self.connected = False
                    self.error = str(exc)
                time.sleep(self.settings.reconnect_delay_seconds)
            finally:
                if capture is not None:
                    capture.release()

    def _annotate(self, frame, detections):
        result = frame.copy()
        h, w = result.shape[:2]
        zone = np.asarray([(int(x * w), int(y * h)) for x, y in self.settings.driveway_zone], np.int32)
        overlay = result.copy()
        cv2.fillPoly(overlay, [zone], (35, 120, 70))
        cv2.addWeighted(overlay, .16, result, .84, 0, result)
        cv2.polylines(result, [zone], True, (70, 220, 130), 2)
        for item in detections:
            x1, y1, x2, y2 = item.box
            cv2.rectangle(result, (x1, y1), (x2, y2), (40, 200, 255), 2)
            cv2.putText(result, f"{item.class_name} {item.confidence:.0%}", (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 200, 255), 2)
        return result

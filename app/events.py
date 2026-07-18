from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import time
import uuid

import cv2
import numpy as np

from app.detector import Detection


@dataclass(frozen=True)
class DrivewayEvent:
    event_id: str
    timestamp: str
    class_name: str
    confidence: float
    snapshot_path: str


def point_in_zone(detection: Detection, frame_shape: tuple[int, ...], zone: tuple[tuple[float, float], ...]) -> bool:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = detection.box
    center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
    polygon = np.asarray(zone, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, center, False) >= 0


class EventEngine:
    def __init__(self, output_dir: Path, zone, cooldown: float, clear_after: float, snapshots: bool):
        self.zone = zone
        self.cooldown = cooldown
        self.clear_after = clear_after
        self.snapshots = snapshots
        self.active: dict[str, float] = {}
        self.last_event: dict[str, float] = {}
        self.events_dir = output_dir / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "events.csv"
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(DrivewayEvent.__annotations__.keys())

    def update(self, detections: list[Detection], frame: np.ndarray, now: float | None = None) -> list[DrivewayEvent]:
        now = time.monotonic() if now is None else now
        in_zone = [item for item in detections if point_in_zone(item, frame.shape, self.zone)]
        present = {item.class_name for item in in_zone}
        for name in list(self.active):
            if name not in present and now - self.active[name] >= self.clear_after:
                del self.active[name]
        created: list[DrivewayEvent] = []
        for detection in sorted(in_zone, key=lambda item: item.confidence, reverse=True):
            name = detection.class_name
            was_active = name in self.active
            self.active[name] = now
            if was_active or now - self.last_event.get(name, -1e12) < self.cooldown:
                continue
            self.last_event[name] = now
            event_id = uuid.uuid4().hex[:12]
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            snapshot_path = ""
            if self.snapshots:
                snapshot = self.events_dir / f"{timestamp.replace(':', '-')}_{name}_{event_id}.jpg"
                cv2.imwrite(str(snapshot), frame)
                snapshot_path = snapshot.name
            event = DrivewayEvent(event_id, timestamp, name, detection.confidence, snapshot_path)
            with self.log_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=DrivewayEvent.__annotations__.keys()).writerow(asdict(event))
            created.append(event)
        return created


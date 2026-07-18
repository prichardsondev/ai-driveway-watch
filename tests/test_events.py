import numpy as np

from app.detector import Detection
from app.events import EventEngine, point_in_zone


ZONE = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))


def test_detection_center_in_zone():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert point_in_zone(Detection("car", .9, (40, 40, 60, 60)), frame.shape, ZONE)
    assert not point_in_zone(Detection("car", .9, (0, 0, 10, 10)), frame.shape, ZONE)


def test_event_debounces_until_object_clears(tmp_path):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    engine = EventEngine(tmp_path, ZONE, cooldown=0, clear_after=2, snapshots=False)
    car = Detection("car", .9, (40, 40, 60, 60))
    assert len(engine.update([car], frame, now=0)) == 1
    assert engine.update([car], frame, now=1) == []
    engine.update([], frame, now=4)
    assert len(engine.update([car], frame, now=5)) == 1


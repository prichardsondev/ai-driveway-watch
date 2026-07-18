from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


class YoloXDetector:
    input_size = (640, 640)
    strides = (8, 16, 32)

    def __init__(self, model_path: Path, confidence: float, nms: float, allowed: tuple[str, ...]):
        if not model_path.exists():
            raise FileNotFoundError(f"model missing: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.confidence = confidence
        self.nms = nms
        self.allowed = set(allowed)
        grids, expanded = [], []
        for stride in self.strides:
            size = self.input_size[0] // stride
            xv, yv = np.meshgrid(np.arange(size), np.arange(size))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded.append(np.full((*grid.shape[:2], 1), stride))
        self.grids = np.concatenate(grids, 1)
        self.expanded_strides = np.concatenate(expanded, 1)

    @staticmethod
    def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        ratio = min(640 / height, 640 / width)
        resized = cv2.resize(frame, (int(width * ratio), int(height * ratio))).astype(np.float32)
        padded = np.full((640, 640, 3), 114, dtype=np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return padded, ratio

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        image, ratio = self._letterbox(frame)
        blob = np.transpose(image, (2, 0, 1))[None]
        started = perf_counter()
        self.net.setInput(blob)
        output = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        latency_ms = (perf_counter() - started) * 1000
        output[:, :2] = (output[:, :2] + self.grids[0]) * self.expanded_strides[0]
        output[:, 2:4] = np.exp(output[:, 2:4]) * self.expanded_strides[0]
        scores = output[:, 4:5] * output[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        boxes = np.empty_like(output[:, :4])
        boxes[:, 0] = output[:, 0] - output[:, 2] / 2
        boxes[:, 1] = output[:, 1] - output[:, 3] / 2
        boxes[:, 2:] = output[:, 2:4]
        keep = cv2.dnn.NMSBoxesBatched(
            boxes.tolist(), confidences.tolist(), class_ids.tolist(), self.confidence, self.nms
        )
        detections: list[Detection] = []
        for index in keep:
            class_name = COCO_CLASSES[int(class_ids[index])]
            if self.allowed and class_name not in self.allowed:
                continue
            x, y, w, h = (boxes[index] / ratio).astype(int)
            detections.append(Detection(class_name, float(confidences[index]), (x, y, x + w, y + h)))
        return detections, latency_ms

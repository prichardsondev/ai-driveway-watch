#!/usr/bin/env python3
"""Local-only DFNE wildlife classifier for AI Driveway Watch.

The server accepts an animal crop as a JPEG over localhost and returns one
tab-separated label and confidence. Images are decoded in memory and are
never written to disk or sent to another host.
"""

from __future__ import annotations

import argparse
import io
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
from PytorchWildlife.models import classification as wildlife_classification


MAX_IMAGE_BYTES = 8 * 1024 * 1024
SAFE_LABEL = re.compile(r"[^A-Za-z0-9 ._-]+")


class Classifier:
    def __init__(self) -> None:
        self._model = wildlife_classification.DFNE(device="cpu")
        self._lock = threading.Lock()

    def classify(self, jpeg: bytes) -> tuple[str, float]:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        pixels = np.asarray(image)
        with self._lock:
            result = self._model.single_image_classification(pixels)
        label = SAFE_LABEL.sub("", str(result["prediction"])).strip()
        confidence = float(result["confidence"])
        if not label:
            raise ValueError("classifier returned an empty label")
        return label, confidence


class Handler(BaseHTTPRequestHandler):
    server: "ClassifierServer"

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        body = b"ready\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/classify":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "invalid content length")
            return
        if length <= 0 or length > MAX_IMAGE_BYTES:
            self.send_error(413, "invalid image size")
            return
        try:
            label, confidence = self.server.classifier.classify(
                self.rfile.read(length)
            )
            body = f"{label}\t{confidence:.6f}\n".encode("utf-8")
        except Exception as error:  # Keep malformed crops from killing service.
            self.log_error("classification failed: %s", error)
            self.send_error(422, "classification failed")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, message: str, *args: object) -> None:
        print(f"classifier: {message % args}", flush=True)


class ClassifierServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], classifier: Classifier) -> None:
        super().__init__(address, Handler)
        self.classifier = classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("wildlife classifier must listen on localhost")

    print("Loading DFNE regional wildlife classifier...", flush=True)
    classifier = Classifier()
    server = ClassifierServer((args.host, args.port), classifier)
    print(f"DFNE classifier ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

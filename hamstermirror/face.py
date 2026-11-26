"""Face detection and cropping.

Backends, tried in order with backend="auto":
  1. mediapipe  - MediaPipe Tasks BlazeFace (needs models/blaze_face_short_range.tflite,
                  fetched by scripts/download_models.py)
  2. mp_legacy  - old mediapipe.solutions API (mediapipe < 0.10.30)
  3. haar       - OpenCV Haar cascade, always available, used as an offline fallback
"""
import os

import cv2
import numpy as np

from . import config


class FaceDetector:
    def __init__(self, backend="auto", min_conf=0.5, model_path=config.MP_FACE_MODEL):
        self.min_conf = min_conf
        self.backend = None
        order = ["mediapipe", "mp_legacy", "haar"] if backend == "auto" else [backend]
        errors = []
        for name in order:
            try:
                getattr(self, f"_init_{name}")(model_path)
                self.backend = name
                break
            except Exception as e:  # try the next backend
                errors.append(f"{name}: {e}")
        if self.backend is None:
            raise RuntimeError("No face detector available. " + " | ".join(errors))

    # backend setup -------------------------------------------------------
    def _init_mediapipe(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run scripts/download_models.py)")
        opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=self.min_conf,
        )
        self._mp = mp
        self._det = vision.FaceDetector.create_from_options(opts)

    def _init_mp_legacy(self, _):
        import mediapipe as mp

        self._det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=self.min_conf
        )

    def _init_haar(self, _):
        name = "haarcascade_frontalface_default.xml"
        candidates = [config.MODEL_DIR / name]  # bundled copy (some OpenCV builds omit it)
        if hasattr(cv2, "data"):
            candidates.insert(0, cv2.data.haarcascades + name)
        for path in candidates:
            if os.path.exists(path):
                self._det = cv2.CascadeClassifier(str(path))
                if not self._det.empty():
                    return
        raise RuntimeError("Could not load Haar cascade")

    # detection -----------------------------------------------------------
    def detect(self, frame_bgr):
        """Return list of (x, y, w, h, score) in pixel coords, largest first."""
        h, w = frame_bgr.shape[:2]
        boxes = []
        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            for d in self._det.detect(img).detections:
                b = d.bounding_box
                boxes.append((b.origin_x, b.origin_y, b.width, b.height, d.categories[0].score))
        elif self.backend == "mp_legacy":
            res = self._det.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                r = d.location_data.relative_bounding_box
                boxes.append((int(r.xmin * w), int(r.ymin * h), int(r.width * w),
                              int(r.height * h), d.score[0]))
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            found = self._det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            boxes = [(int(x), int(y), int(bw), int(bh), 1.0) for x, y, bw, bh in found]
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes

    def largest(self, frame_bgr):
        boxes = self.detect(frame_bgr)
        return boxes[0] if boxes else None


def crop_face(frame_bgr, box, pad=0.25, size=224):
    """Padded square crop around box, returned as RGB uint8 (size x size)."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = box[:4]
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * (1 + 2 * pad)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    # pad with edge pixels so crops near the border stay square
    padl, padt = max(0, -x0), max(0, -y0)
    padr, padb = max(0, x1 - w), max(0, y1 - h)
    if padl or padt or padr or padb:
        frame_bgr = cv2.copyMakeBorder(frame_bgr, padt, padb, padl, padr, cv2.BORDER_REPLICATE)
        x0, x1, y0, y1 = x0 + padl, x1 + padl, y0 + padt, y1 + padt
    crop = frame_bgr[y0:y1, x0:x1]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

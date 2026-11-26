"""Hand landmark tracking (MediaPipe Hand Landmarker, 21 points per hand).

Backends: "mediapipe" (Tasks API, needs models/hand_landmarker.task from
scripts/download_models.py) or "mp_legacy" (mediapipe.solutions, older
versions). If neither is available, HandTracker.available is False and the
app keeps working on faces only.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 2) pixel coords
    handedness: str = ""
    score: float = 1.0

    @property
    def bbox(self):
        x0, y0 = self.landmarks.min(0)
        x1, y1 = self.landmarks.max(0)
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    @property
    def palm_center(self):
        return self.landmarks[[0, 5, 9, 13, 17]].mean(0)


class HandTracker:
    def __init__(self, backend="auto", max_hands=2, min_conf=0.5, model_path=config.MP_HAND_MODEL):
        self.backend = None
        self.error = ""
        order = ["mediapipe", "mp_legacy"] if backend == "auto" else [backend]
        errs = []
        for name in order:
            try:
                getattr(self, f"_init_{name}")(model_path, max_hands, min_conf)
                self.backend = name
                break
            except Exception as e:
                errs.append(f"{name}: {e}")
        self.error = " | ".join(errs)

    @property
    def available(self):
        return self.backend is not None

    def _init_mediapipe(self, model_path, max_hands, min_conf):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} (run scripts/download_models.py)")
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
        )
        self._mp = mp
        self._det = vision.HandLandmarker.create_from_options(opts)

    def _init_mp_legacy(self, _, max_hands, min_conf):
        import mediapipe as mp

        self._det = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                                             min_detection_confidence=min_conf)

    def detect(self, frame_bgr):
        if not self.available:
            return []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands = []
        if self.backend == "mediapipe":
            res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                pts = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
                hands.append(Hand(pts, hd[0].category_name, hd[0].score))
        else:
            res = self._det.process(rgb)
            for lms, hd in zip(res.multi_hand_landmarks or [], res.multi_handedness or []):
                pts = np.array([[p.x * w, p.y * h] for p in lms.landmark], np.float32)
                c = hd.classification[0]
                hands.append(Hand(pts, c.label, c.score))
        return hands


# MediaPipe hand skeleton, for drawing
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]


def draw_hands(img, hands, color=(80, 255, 120)):
    for hand in hands:
        pts = hand.landmarks.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return img

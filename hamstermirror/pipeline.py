"""Frame -> face -> matcher -> smoothing -> meme. Shared by the app and eval scripts."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from .face import FaceDetector, crop_face
from .gestures import HEART_HAND_REACH, GestureSmoother, context_box, gesture_offsets
from .gestures import classify as classify_gesture
from .memes import load_memes
from .smoothing import Smoother


@dataclass
class Result:
    meme_idx: int | None
    topk: list = field(default_factory=list)  # [(meme_idx, score)]
    box: tuple | None = None
    face: np.ndarray | None = None
    timings: dict = field(default_factory=dict)  # ms per stage
    hands: list = field(default_factory=list)
    gesture: str | None = None
    crop_box: tuple | None = None


class HamsterMirror:
    """hands: optional HandTracker. When given (and gestures enabled), the detected
    gesture boosts memes tagged with it, and CLIP sees a crop that includes the hands."""

    def __init__(self, detector, matcher, smoothing=True, alpha=0.3, margin=None,
                 min_frames=3, k=3, hands=None, gestures=True, gesture_bonus=None,
                 context_crop=True):
        self.detector = detector
        self.hands = hands if (hands is not None and hands.available) else None
        self.gestures_enabled = gestures and self.hands is not None
        self.context_crop = context_crop
        self._gesture_bonus = gesture_bonus
        self.gesture_smoother = GestureSmoother()
        self.k = k
        self.smoothing_kwargs = dict(alpha=alpha, min_frames=min_frames)
        # cosine gaps are small for CLIP, probability gaps are bigger for the baseline
        self._margin = margin
        self.set_matcher(matcher)
        self.set_smoothing(smoothing)

    def set_matcher(self, matcher):
        self.matcher = matcher
        clip_like = matcher.normalize_vector
        margin = self._margin if self._margin is not None else getattr(
            matcher, "margin", 0.01 if clip_like else 0.05)
        self.gesture_bonus = self._gesture_bonus if self._gesture_bonus is not None else getattr(
            matcher, "gesture_bonus", 0.1 if clip_like else 0.5)
        enabled = getattr(getattr(self, "smoother", None), "enabled", True)
        self.smoother = Smoother(enabled, normalize=matcher.normalize_vector, margin=margin,
                                 **self.smoothing_kwargs)
        self.last = Result(None)

    def set_smoothing(self, enabled):
        self.smoother.enabled = enabled
        self.smoother.reset()

    def set_gestures(self, enabled):
        self.gestures_enabled = enabled and self.hands is not None
        self.gesture_smoother.reset()
        self.smoother.reset()

    def reset(self):
        self.smoother.reset()
        self.gesture_smoother.reset()
        self.last = Result(None)

    def process(self, frame_bgr, box=None, hands=None):
        t0 = time.perf_counter()
        if box is None:
            box = self.detector.largest(frame_bgr)
        t1 = time.perf_counter()
        if hands is None:
            hands = self.hands.detect(frame_bgr) if self.gestures_enabled else []
        t_h = time.perf_counter()
        if box is None:  # no face: keep showing the last meme
            self.last = Result(self.last.meme_idx, self.last.topk, None, None,
                               {"detect": (t1 - t0) * 1e3, "hands": (t_h - t1) * 1e3,
                                "total": (t_h - t0) * 1e3}, hands)
            return self.last
        gesture = None
        crop_box = box
        if self.gestures_enabled:
            gesture = self.gesture_smoother.update(classify_gesture(hands, box))
            if self.context_crop and getattr(self.matcher, "uses_context", False) and hands:
                reach = HEART_HAND_REACH if gesture == "heart_hands" else 2.5
                crop_box = context_box(box, hands, reach=reach)
        face = crop_face(frame_bgr, crop_box, pad=0.25 if crop_box is box else 0.1)
        vec = self.matcher.vector(face)
        t2 = time.perf_counter()
        vec = self.smoother.smooth_vector(vec)
        scores = self.matcher.scores(vec)
        scores = scores + gesture_offsets(self.matcher.memes, gesture, self.gesture_bonus)
        idx = self.smoother.choose(scores)
        order = np.argsort(-scores)[: self.k]
        t3 = time.perf_counter()
        self.last = Result(
            idx, [(int(i), float(scores[i])) for i in order], box, face,
            {"detect": (t1 - t0) * 1e3, "hands": (t_h - t1) * 1e3, "embed": (t2 - t_h) * 1e3,
             "match": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3},
            hands, gesture, crop_box,
        )
        return self.last


def build_matchers(memes=None, clip_model=config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED,
                   text_weight=0.6, use_projection=True, want=("expr", "clip", "baseline"),
                   index_path=config.MEME_INDEX, baseline_ckpt=config.BASELINE_CKPT,
                   projection_ckpt=config.PROJECTION_CKPT, verbose=True):
    """Build whichever matchers have their artifacts available. Returns dict name -> matcher."""
    memes = memes or load_memes()
    out = {}
    if "clip" in want:
        from .clip_retrieval import ClipEncoder, ClipMatcher, MemeIndex

        enc = ClipEncoder(clip_model, pretrained)
        if index_path.exists():
            index = MemeIndex.load(index_path)
            if index.files != [m.file for m in memes]:
                if verbose:
                    print("[pipeline] meme index stale, rebuilding in memory "
                          "(run scripts/build_index.py to save it)")
                index = MemeIndex.build(enc, memes)
        else:
            index = MemeIndex.build(enc, memes)
        out["clip"] = ClipMatcher(memes, enc, index, text_weight)
        if use_projection and projection_ckpt.exists():
            from .projection import ProjectionHead

            out["clip+proj"] = ClipMatcher(memes, enc, index, text_weight,
                                           ProjectionHead.load(projection_ckpt))
    if "expr" in want:
        try:
            from .expression import BlendshapeExtractor, Calibration, ExprClipMatcher, ExprMatcher

            ex = ExprMatcher(memes, BlendshapeExtractor(), Calibration.load(), verbose=verbose)
            first = {"expr": ex}
            if "clip" in out:
                first["expr+clip"] = ExprClipMatcher(ex, out["clip"])
            out = {**first, **out}  # expression matcher first = default in the app
        except Exception as e:
            if verbose:
                print(f"[pipeline] expression matcher unavailable ({e})")
    if "baseline" in want:
        if baseline_ckpt.exists():
            from .baseline import BaselineClassifier, BaselineMatcher

            out["baseline"] = BaselineMatcher(memes, BaselineClassifier(baseline_ckpt))
        elif verbose:
            print(f"[pipeline] no baseline checkpoint at {baseline_ckpt}, "
                  "run scripts/train_baseline.py to enable it")
    return memes, out

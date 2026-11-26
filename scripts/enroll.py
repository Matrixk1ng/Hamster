"""Teach the expression matcher YOUR face.

  python scripts/enroll.py                 # neutral face, then every meme
  python scripts/enroll.py --neutral-only  # just recalibrate your resting face
  python scripts/enroll.py --only h11_smug_grin.jpg h13_simple_happy_smile.jpg

Step 1: relax your face and look at the camera. Your resting expression is
subtracted from everything afterwards, so a neutral face reads as neutral.
Step 2: each meme is shown next to the webcam. Copy the face and press SPACE,
hold it for about a second while it records. S skips a meme, B goes back
one, Q saves and quits. Memes you skip keep their hand-written profile
from memes/expressions.csv.

Saved to checkpoints/expr_calib.npz (it is per person, re-run for a new user).
"""
import argparse
import time

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.expression import FEATURES, BlendshapeExtractor, Calibration
from hamstermirror.face import FaceDetector, crop_face
from hamstermirror.memes import load_memes

H = 420


def fit(img, h=H):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def put(img, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


class Cam:
    def __init__(self, idx, det, ext):
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise SystemExit("Could not open camera")
        self.det, self.ext = det, ext

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            return None, None
        f = cv2.flip(f, 1)
        box = self.det.largest(f)
        feat = self.ext.features(crop_face(f, box)) if box is not None else None
        return f, feat

    def record(self, seconds, show):
        """Median feature vector over `seconds` of frames."""
        feats, end = [], time.time() + seconds
        while time.time() < end:
            f, feat = self.read()
            if f is None:
                continue
            if feat is not None:
                feats.append(feat)
            show(f, f"recording... {end - time.time():.1f}s", (0, 0, 255))
        return np.median(feats, 0) if len(feats) >= 5 else None


def top_features(v, k=4):
    order = np.argsort(-v)[:k]
    return ", ".join(f"{FEATURES[i]} {v[i]:.2f}" for i in order if v[i] > 0.05) or "nothing strong"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--neutral-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="enroll just these meme files")
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--out", default=str(config.EXPR_CALIB))
    args = ap.parse_args()

    memes = load_memes()
    calib = Calibration.load(args.out)
    cam = Cam(args.camera, FaceDetector(args.detector), BlendshapeExtractor())

    def show(frame, msg, color=(255, 255, 255), meme=None, sub=""):
        view = fit(frame)
        if meme is not None:
            view = np.hstack([view, fit(meme)])
        put(view, msg, 32, color)
        if sub:
            put(view, sub, H - 20, (180, 230, 255), 0.55)
        cv2.imshow("enroll", view)
        return cv2.waitKey(1) & 0xFF

    # ---- step 1: neutral face
    while True:
        f, feat = cam.read()
        if f is None:
            continue
        live = top_features(feat) if feat is not None else "no face found"
        k = show(f, "Relax your face, look at the camera, press SPACE", sub="raw: " + live)
        if k == ord("q"):
            raise SystemExit
        if k == ord(" "):
            n = cam.record(2.0, lambda fr, m, c: show(fr, m, c))
            if n is not None:
                calib.neutral = n.astype(np.float32)
                calib.save(args.out)
                print(f"neutral saved ({top_features(n)})")
                break
    if args.neutral_only:
        return

    # ---- step 2: enroll memes
    todo = [m for m in memes if not args.only or m.file in args.only]
    i = 0
    while i < len(todo):
        m = todo[i]
        img = cv2.cvtColor(np.array(m.load()), cv2.COLOR_RGB2BGR)
        f, feat = cam.read()
        if f is None:
            continue
        live = top_features(calib.apply(feat)) if feat is not None else "no face found"
        done = "  (enrolled)" if m.file in calib.protos else ""
        k = show(f, f"[{i + 1}/{len(todo)}] copy this face, SPACE to record{done}",
                 meme=img, sub=f"{m.name}: you now = {live}")
        if k == ord("q"):
            break
        if k == ord("s"):
            i += 1
        elif k == ord("b"):
            i = max(0, i - 1)
        elif k == ord(" "):
            v = cam.record(args.seconds, lambda fr, msg, c: show(fr, msg, c, meme=img))
            if v is None:
                print(f"{m.name}: face not found, try again")
                continue
            calib.protos[m.file] = calib.apply(v).astype(np.float32)
            calib.save(args.out)
            print(f"{m.name}: {top_features(calib.protos[m.file])}")
            i += 1
    cv2.destroyAllWindows()
    print(f"saved {len(calib.protos)} enrolled memes -> {args.out}")


if __name__ == "__main__":
    main()

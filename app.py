"""HamsterMirror live demo: webcam on the left, best-matching hamster meme on the right.

Keys:  q quit | m switch method (expr / expr+clip / clip / clip+proj / baseline) | s smoothing
       g toggle hand gestures | t cycle CLIP text weight | f mirror on/off
       space save snapshot to results/
"""
import argparse
import time

import cv2
import numpy as np

from hamstermirror import config
from hamstermirror.face import FaceDetector
from hamstermirror.hands import HandTracker, draw_hands
from hamstermirror.pipeline import HamsterMirror, build_matchers

PANEL_H = 480
TEXT_WEIGHTS = [0.0, 0.3, 0.6, 1.0]


def fit(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h))


def label(img, text, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(frame, res, meme_imgs, memes, info):
    left = frame.copy()
    draw_hands(left, res.hands)
    if res.crop_box is not None and res.crop_box is not res.box:
        x, y, w, h = map(int, res.crop_box[:4])
        cv2.rectangle(left, (x, y), (x + w, y + h), (255, 160, 60), 1)
    if res.box is not None:
        x, y, w, h = map(int, res.box[:4])
        cv2.rectangle(left, (x, y), (x + w, y + h), (0, 220, 255), 2)
    if res.gesture:
        cv2.putText(left, res.gesture.replace("_", " "), (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (80, 255, 120), 2, cv2.LINE_AA)
    left = fit(left, PANEL_H)
    if res.meme_idx is not None:
        right = fit(meme_imgs[res.meme_idx], PANEL_H)
    else:
        right = np.full((PANEL_H, PANEL_H, 3), 40, np.uint8)
        label(right, "show me your face", (20, PANEL_H // 2), 0.8)
    canvas = np.hstack([left, right])
    bar = np.full((70, canvas.shape[1], 3), 25, np.uint8)
    label(bar, info, (10, 25))
    if res.topk:
        top = "  ".join(f"{memes[i].name} {s:.3f}" for i, s in res.topk)
        label(bar, "top-3: " + top, (10, 55), 0.5, (180, 230, 255))
    return np.vstack([canvas, bar])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="play a video file instead of the webcam")
    ap.add_argument("--method", default="expr",
                    help="expr | expr+clip | clip | clip+proj | baseline")
    ap.add_argument("--detector", default="auto", help="auto | mediapipe | mp_legacy | haar")
    ap.add_argument("--pretrained", default=config.CLIP_PRETRAINED)
    ap.add_argument("--text-weight", type=float, default=0.6)
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--no-gestures", action="store_true", help="faces only, skip hand tracking")
    ap.add_argument("--gesture-bonus", type=float, default=None,
                    help="score boost for memes tagged with the detected gesture")
    ap.add_argument("--no-window", action="store_true", help="headless: print stats only")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    memes, matchers = build_matchers(pretrained=args.pretrained, text_weight=args.text_weight)
    names = list(matchers)
    if args.method not in matchers:
        print(f"[app] method '{args.method}' unavailable, using '{names[0]}'")
        args.method = names[0]
    detector = FaceDetector(args.detector)
    hands = None if args.no_gestures else HandTracker()
    if hands is not None and not hands.available:
        print(f"[app] hand tracking off ({hands.error})")
    n_g = sum(bool(m.gesture) for m in memes)
    print(f"[app] detector={detector.backend} hands={hands.backend if hands else None} "
          f"methods={names} memes={len(memes)} ({n_g} gesture-tagged)")
    hm = HamsterMirror(detector, matchers[args.method], smoothing=not args.no_smoothing,
                       hands=hands, gesture_bonus=args.gesture_bonus)
    meme_imgs = [cv2.cvtColor(np.array(m.load()), cv2.COLOR_RGB2BGR) for m in memes]

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video. Try --camera 1 or check permissions.")
    mirror = not args.video
    tw_i = TEXT_WEIGHTS.index(args.text_weight) if args.text_weight in TEXT_WEIGHTS else 2
    fps, n, t_last = 0.0, 0, time.perf_counter()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        res = hm.process(frame)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_last, 1e-6)) if n else 0.0
        t_last, n = now, n + 1
        tw = getattr(hm.matcher, "text_weight", None)
        info = (f"{hm.matcher.name} | smoothing {'on' if hm.smoother.enabled else 'off'}"
                + f" | gestures {'on' if hm.gestures_enabled else 'off'}"
                + (f" | text_w {tw:.1f}" if tw is not None else "")
                + f" | {fps:4.1f} FPS | {res.timings.get('total', 0):5.1f} ms")
        if args.no_window:
            if n % 30 == 0:
                print(info)
        else:
            cv2.imshow("HamsterMirror", render(frame, res, meme_imgs, memes, info))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("m"):
                args.method = names[(names.index(args.method) + 1) % len(names)]
                hm.set_matcher(matchers[args.method])
            elif key == ord("s"):
                hm.set_smoothing(not hm.smoother.enabled)
            elif key == ord("g"):
                hm.set_gestures(not hm.gestures_enabled)
            elif key == ord("t") and tw is not None:
                tw_i = (tw_i + 1) % len(TEXT_WEIGHTS)
                hm.matcher.set_text_weight(TEXT_WEIGHTS[tw_i])
                hm.reset()
            elif key == ord("f"):
                mirror = not mirror
            elif key == ord(" "):
                p = config.RESULTS_DIR / f"snapshot_{int(time.time())}.png"
                cv2.imwrite(str(p), render(frame, res, meme_imgs, memes, info))
                print(f"[app] saved {p}")
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    cv2.destroyAllWindows()
    if hm.last.meme_idx is not None:
        print(f"[app] {n} frames | last meme: {memes[hm.last.meme_idx].name} | {info}")


if __name__ == "__main__":
    main()

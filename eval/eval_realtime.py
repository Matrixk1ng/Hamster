"""Real-time performance: FPS, per-stage latency, and flicker rate
(meme switches per second while the expression is held still).

  python eval/eval_realtime.py --videos data/clips/p01_neutral_*.mp4
  python eval/eval_realtime.py --webcam 10      # 10 s live per configuration, hold still

With no arguments, uses every clip whose prompt is 'neutral' or 'deadpan stare'.
"""
import argparse
import glob
import time

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.data import read_csv
from hamstermirror.face import FaceDetector
from hamstermirror.hands import HandTracker
from hamstermirror.metrics import count_switches, md_table
from hamstermirror.pipeline import HamsterMirror, build_matchers


def frames_from(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def run(hm, frame_source):
    """frame_source yields frames. Returns stats dict."""
    hm.reset()
    choices, timings, n = [], [], 0
    t0 = time.perf_counter()
    for f in frame_source:
        r = hm.process(f)
        n += 1
        if r.meme_idx is not None:
            choices.append(r.meme_idx)
        timings.append(r.timings)
    wall = time.perf_counter() - t0
    return n, wall, choices, timings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--webcam", type=float, default=0, help="seconds of live capture per config")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--pretrained", default=config.CLIP_PRETRAINED)
    ap.add_argument("--no-gestures", action="store_true", help="time without hand tracking")
    args = ap.parse_args()

    memes, matchers = build_matchers(pretrained=args.pretrained)
    det = FaceDetector(args.detector)
    hands = None if args.no_gestures else HandTracker()
    videos = []
    if not args.webcam:
        paths = []
        for v in args.videos or []:
            paths += glob.glob(v)
        if not paths:
            paths = [str(config.CLIP_DIR / c["clip"]) for c in read_csv(config.CLIP_META_CSV)
                     if c["prompt"] in ("neutral", "deadpan stare")]
        if not paths:
            raise SystemExit("No videos. Pass --videos or --webcam SECONDS.")
        videos = [frames_from(p) for p in paths]
        print(f"{len(videos)} static-expression videos, {sum(len(v[0]) for v in videos)} frames")

    table = []
    for name, m in matchers.items():
        for smooth in (False, True):
            hm = HamsterMirror(det, m, smoothing=smooth, hands=hands)
            frames_total, wall_total, sw, dur, tims = 0, 0.0, 0, 0.0, []
            if args.webcam:
                input(f"\n[{name}, smoothing={smooth}] hold a still expression, press Enter")
                cap = cv2.VideoCapture(args.camera)
                end = time.time() + args.webcam

                def live():
                    while time.time() < end:
                        ok, f = cap.read()
                        if ok:
                            yield cv2.flip(f, 1)
                n, wall, ch, t = run(hm, live())
                cap.release()
                frames_total, wall_total, sw, dur, tims = n, wall, count_switches(ch), wall, t
            else:
                for frames, fps in videos:
                    n, wall, ch, t = run(hm, frames)
                    frames_total += n
                    wall_total += wall
                    sw += count_switches(ch)
                    dur += n / fps
                    tims += t
            tot = np.array([t.get("total", 0) for t in tims])
            stage = lambda k: float(np.mean([t[k] for t in tims if k in t] or [np.nan]))
            table.append([name, "on" if smooth else "off", frames_total / wall_total,
                          float(np.median(tot)), float(np.percentile(tot, 95)),
                          stage("detect"), stage("hands"), stage("embed"), sw / max(dur, 1e-6)])
            print(table[-1])

    lines = ["# Real-time performance", "",
             f"detector: {det.backend}, hands: {getattr(hands, 'backend', None)}, device: "
             f"{config.best_device()}", "",
             md_table(["method", "smoothing", "FPS", "latency p50 ms", "latency p95 ms",
                       "detect ms", "hands ms", "embed ms", "flicker (switches/s)"], table)]
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "realtime.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

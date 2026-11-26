"""Blind A/B user study: baseline vs the proposed method (expr by default), rated 1-5.

Run a session per participant:
  python eval/user_study.py --participant u01 --trials 10
Participant makes an expression and presses SPACE. Two memes appear as A and B
(random order, method hidden). They press 1-5 for A, then 1-5 for B.
Results append to results/user_study.csv.

Summarise everyone:
  python eval/user_study.py --analyze
"""
import argparse
import csv
import random
import time
from collections import defaultdict

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.data import append_csv, read_csv
from hamstermirror.face import FaceDetector
from hamstermirror.hands import HandTracker
from hamstermirror.metrics import md_table
from hamstermirror.pipeline import HamsterMirror, build_matchers

OUT = config.RESULTS_DIR / "user_study.csv"
FIELDS = ["participant", "trial", "order", "proposed_method", "meme_baseline", "meme_proposed",
          "rating_baseline", "rating_proposed", "same_meme", "time"]


def put(img, text, y, scale=0.8):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def fit(img, h=360):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def wait_rating():
    while True:
        k = cv2.waitKey(0) & 0xFF
        if ord("1") <= k <= ord("5"):
            return k - ord("0")
        if k == ord("q"):
            raise SystemExit


def session(args):
    memes, matchers = build_matchers(pretrained=args.pretrained, use_projection=False)
    if "baseline" not in matchers:
        raise SystemExit("User study needs checkpoints/baseline.pt (scripts/train_baseline.py)")
    if args.proposed not in matchers:
        raise SystemExit(f"method {args.proposed} unavailable, have {list(matchers)}")
    matchers = {"baseline": matchers["baseline"], "proposed": matchers[args.proposed]}
    det = FaceDetector(args.detector)
    ht = HandTracker()
    hms = {k: HamsterMirror(det, matchers[k], hands=ht) for k in ("baseline", "proposed")}
    imgs = [cv2.cvtColor(np.array(m.load()), cv2.COLOR_RGB2BGR) for m in memes]
    cap = cv2.VideoCapture(args.camera)
    for trial in range(1, args.trials + 1):
        while True:  # live preview, both pipelines warm up their smoothing
            ok, f = cap.read()
            if not ok:
                continue
            f = cv2.flip(f, 1)
            box = det.largest(f)
            hs = ht.detect(f) if ht.available else []
            res = {k: hm.process(f, box=box, hands=hs) if box else hm.last
                   for k, hm in hms.items()}
            view = fit(f.copy())
            put(view, f"trial {trial}/{args.trials}: make a face, SPACE to lock", 30, 0.6)
            cv2.imshow("study", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                raise SystemExit
            if k == ord(" ") and box is not None:
                break
        order = random.sample(["baseline", "proposed"], 2)
        panels = []
        for lab, meth in zip("AB", order):
            p = fit(imgs[res[meth].meme_idx])
            put(p, lab, 40, 1.2)
            panels.append(p)
        view = np.hstack([fit(f)] + panels)
        put(view, "Rate A (1-5)", view.shape[0] - 20)
        cv2.imshow("study", view)
        ra = wait_rating()
        put(view, f"A={ra}. Now rate B (1-5)", view.shape[0] - 55)
        cv2.imshow("study", view)
        rb = wait_rating()
        rating = dict(zip(order, (ra, rb)))
        append_csv(OUT, FIELDS, {
            "participant": args.participant, "trial": trial, "order": "-".join(order),
            "proposed_method": args.proposed,
            "meme_baseline": memes[res["baseline"].meme_idx].file,
            "meme_proposed": memes[res["proposed"].meme_idx].file,
            "rating_baseline": rating["baseline"], "rating_proposed": rating["proposed"],
            "same_meme": res["baseline"].meme_idx == res["proposed"].meme_idx,
            "time": int(time.time())})
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved -> {OUT}")


def analyze():
    rows = read_csv(OUT)
    if not rows:
        raise SystemExit(f"No data in {OUT}")
    b = np.array([int(r["rating_baseline"]) for r in rows])
    c = np.array([int(r["rating_proposed"]) for r in rows])
    per_p = defaultdict(list)
    for r in rows:
        per_p[r["participant"]].append(int(r["rating_proposed"]) - int(r["rating_baseline"]))
    lines = ["# User study", "",
             f"{len(per_p)} participants, {len(rows)} trials "
             f"({sum(r['same_meme'] == 'True' for r in rows)} where both methods showed the same meme)", "",
             md_table(["method", "mean rating", "std"],
                      [["baseline", float(b.mean()), float(b.std())],
                       [rows[0].get("proposed_method", "proposed"), float(c.mean()), float(c.std())]]),
             "", f"Proposed method rated higher in {np.mean(c > b):.0%} of trials, tied {np.mean(c == b):.0%}."]
    try:
        from scipy.stats import wilcoxon

        pm = np.array([np.mean(v) for v in per_p.values()])
        if len(pm) >= 5 and np.any(pm != 0):
            stat, p = wilcoxon(pm)
            lines.append(f"Wilcoxon signed-rank on per-participant mean difference: "
                         f"W={stat:.1f}, p={p:.4f}")
    except ImportError:
        lines.append("(pip install scipy for a Wilcoxon signed-rank test)")
    (config.RESULTS_DIR / "user_study.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--participant")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--pretrained", default=config.CLIP_PRETRAINED)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--proposed", default="expr", help="method compared against the baseline")
    args = ap.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    if args.analyze:
        analyze()
    elif args.participant:
        session(args)
    else:
        ap.error("pass --participant ID or --analyze")


if __name__ == "__main__":
    main()

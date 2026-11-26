"""Turn recorded clips into labeled face crops.

Step 1 (automatic): sample frames from each clip in data/clips.csv, detect and
crop the face, save crops to data/frames/.
Step 2 (you): for each clip, look at its crops next to the numbered meme sheet
and type the FER emotion plus every meme that is an acceptable match.
Labels apply to all sampled frames of the clip (one held expression per clip).

  python scripts/annotate.py --annotator obinna
  python scripts/annotate.py --extract-only        # just crop frames
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.data import ANNOT_FIELDS, append_csv, read_csv
from hamstermirror.face import FaceDetector, crop_face
from hamstermirror.gestures import classify, context_box
from hamstermirror.hands import HandTracker
from hamstermirror.memes import load_memes
from hamstermirror.sheet import contact_sheet


META_FIELDS = ["frame", "context", "gesture"]


def extract(clip_name, detector, every, max_frames, hands=None):
    """Returns list of (frame_idx, crop_path) for a clip, extracting if needed.
    Also saves a context crop (face + nearby hands) and the detected gesture per
    frame to data/frame_meta.csv, used by the gesture-aware retrieval eval."""
    out_dir = config.FRAME_DIR / Path(clip_name).stem
    existing = sorted(out_dir.glob("f_*.png"))
    if existing:
        return [(int(p.stem.split("_")[-1]), p) for p in existing]
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(config.CLIP_DIR / clip_name))
    res, i = [], 0
    while len(res) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            box = detector.largest(frame)
            if box is not None:
                p = out_dir / f"f_{i:05d}.png"
                cv2.imwrite(str(p), cv2.cvtColor(crop_face(frame, box), cv2.COLOR_RGB2BGR))
                res.append((i, p))
                if hands is not None and hands.available:
                    hs = hands.detect(frame)
                    cp = out_dir / f"c_{i:05d}.png"
                    cb = context_box(box, hs) if hs else box
                    ctx = crop_face(frame, cb, pad=0.25 if cb is box else 0.1)
                    cv2.imwrite(str(cp), cv2.cvtColor(ctx, cv2.COLOR_RGB2BGR))
                    append_csv(config.FRAME_META_CSV, META_FIELDS, {
                        "frame": str(p.relative_to(config.ROOT)),
                        "context": str(cp.relative_to(config.ROOT)),
                        "gesture": classify(hs, box) or ""})
        i += 1
    cap.release()
    return res


def ask_emotion():
    opts = " ".join(f"{i}={c}" for i, c in enumerate(config.FER_CLASSES))
    while True:
        a = input(f"  emotion ({opts}, s=skip): ").strip().lower()
        if a == "s":
            return None
        if a.isdigit() and int(a) < len(config.FER_CLASSES):
            return config.FER_CLASSES[int(a)]
        if a in config.FER_CLASSES:
            return a


def ask_memes(n):
    while True:
        a = input("  acceptable meme numbers, best first (e.g. 3,12): ").strip()
        try:
            ids = [int(x) for x in a.replace(" ", "").split(",") if x != ""]
        except ValueError:
            continue
        if ids and all(0 <= i < n for i in ids):
            return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--annotator", default="anon")
    ap.add_argument("--every", type=int, default=3, help="sample every Nth frame")
    ap.add_argument("--max-frames", type=int, default=20, help="max crops per clip")
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    memes = load_memes()
    clips = read_csv(config.CLIP_META_CSV)
    if not clips:
        raise SystemExit("No clips yet. Run scripts/record_clips.py first.")
    detector = FaceDetector(args.detector)
    hands = HandTracker()
    if not hands.available:
        print(f"[annotate] hand tracking off, no gesture metadata ({hands.error})")
    done = {(r["clip"], r["annotator"]) for r in read_csv(config.ANNOT_CSV)}

    sheet = contact_sheet(memes)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(config.RESULTS_DIR / "meme_sheet.png"), sheet)

    for c in clips:
        frames = extract(c["clip"], detector, args.every, args.max_frames, hands)
        if args.extract_only:
            print(f"{c['clip']}: {len(frames)} face crops")
            continue
        if (c["clip"], args.annotator) in done or not frames:
            continue
        strip = np.hstack([cv2.resize(cv2.imread(str(p)), (160, 160))
                           for _, p in frames[:: max(1, len(frames) // 6)][:6]])
        cv2.imshow("clip frames", strip)
        cv2.imshow("memes", sheet)
        cv2.waitKey(300)
        print(f"\n{c['clip']}  (prompt: {c['prompt']}, {len(frames)} frames)")
        emo = ask_emotion()
        if emo is None:
            continue
        ids = ask_memes(len(memes))
        gt = "|".join(memes[i].file for i in ids)
        for idx, p in frames:
            append_csv(config.ANNOT_CSV, ANNOT_FIELDS, {
                "frame": str(Path(p).relative_to(config.ROOT)), "clip": c["clip"],
                "frame_idx": idx, "annotator": args.annotator, "emotion": emo, "gt_memes": gt})
    cv2.destroyAllWindows()
    print(f"\nannotations -> {config.ANNOT_CSV}")


if __name__ == "__main__":
    main()

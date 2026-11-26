"""Record short webcam clips of one participant making prompted expressions.

  python scripts/record_clips.py --participant p01 --lighting bright --pose frontal

Each prompt gets a 3 s countdown then SECONDS of recording. Clips go to
data/clips/, metadata to data/clips.csv. Only record people who consented.
Skin tone is optional and self-reported (e.g. Monk scale 1-10), used only for
the failure analysis.
"""
import argparse
import time

import cv2

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.data import CLIP_FIELDS, append_csv

POSED = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust",
         "smug", "betrayed", "panicked", "laughing hysterically", "deadpan stare"]
GESTURE_PROMPTS = ["heart hands", "hand on forehead", "hands on cheeks screaming",
                   "steepled fingers scheming", "hand over face", "thumbs up", "peace sign"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--lighting", default="normal", help="bright | normal | dim | backlit")
    ap.add_argument("--pose", default="frontal", help="frontal | turned | tilted")
    ap.add_argument("--glasses", default="no", help="yes | no")
    ap.add_argument("--skin-tone", default="", help="optional self-reported, e.g. monk-4")
    ap.add_argument("--spontaneous", type=int, default=0,
                    help="also record N unprompted clips (talk, react to a video, etc.)")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--no-gestures", action="store_true", help="skip the hand-gesture prompts")
    args = ap.parse_args()

    if input(f"Has {args.participant} consented to being recorded for this class project? [y/N] "
             ).strip().lower() != "y":
        raise SystemExit("No consent, nothing recorded.")

    config.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prompts = args.prompts or (POSED + ([] if args.no_gestures else GESTURE_PROMPTS))
    jobs = [(p, "posed") for p in prompts] + \
           [(f"spontaneous{i + 1}", "spontaneous") for i in range(args.spontaneous)]

    def show(frame, text, color=(255, 255, 255)):
        f = frame.copy()
        cv2.putText(f, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(f, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("record", f)
        return cv2.waitKey(1) & 0xFF

    for prompt, kind in jobs:
        t_end = time.time() + 3
        while time.time() < t_end:  # countdown with live preview
            ok, frame = cap.read()
            if not ok:
                break
            if show(cv2.flip(frame, 1), f"{prompt}  in {int(t_end - time.time()) + 1}") == ord("q"):
                raise SystemExit
        name = f"{args.participant}_{prompt.replace(' ', '-')}_{args.lighting}_{int(time.time())}.mp4"
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(str(config.CLIP_DIR / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            out.write(frame)
            show(frame, f"REC {prompt}", (0, 0, 255))
        out.release()
        append_csv(config.CLIP_META_CSV, CLIP_FIELDS, {
            "clip": name, "participant": args.participant, "prompt": prompt,
            "posed": kind == "posed", "lighting": args.lighting, "pose": args.pose,
            "glasses": args.glasses, "skin_tone": args.skin_tone, "consent": "yes"})
        print(f"saved {name}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

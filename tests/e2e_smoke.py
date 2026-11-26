"""End-to-end smoke test: runs every script on synthetic data in a temp folder.

No webcam, no downloads (random CLIP weights, Haar detector), so it checks the
plumbing, not accuracy.   python tests/e2e_smoke.py
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage import data

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from conftest import make_memes  # noqa: E402

FER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def fake_fer(path):
    rng = np.random.default_rng(0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "pixels", "Usage"])
        for usage, n in [("Training", 30), ("PublicTest", 6), ("PrivateTest", 6)]:
            for c in range(7):
                for _ in range(n):
                    img = rng.integers(0, 60, (48, 48)) + c * 25
                    w.writerow([c, " ".join(map(str, img.ravel())), usage])


def fake_clips(root):
    clips = root / "data" / "clips"
    clips.mkdir(parents=True)
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    rows = []
    rng = np.random.default_rng(1)
    for p, light, tone in [("p01", "bright", "monk-3"), ("p02", "dim", "monk-8")]:
        for prompt, gain in [("happy", 1.0), ("sad", 0.8), ("neutral", 1.1)]:
            name = f"{p}_{prompt}.mp4"
            out = cv2.VideoWriter(str(clips / name), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                  (face.shape[1], face.shape[0]))
            for _ in range(24):
                f = np.clip(face * gain + rng.normal(0, 6, face.shape), 0, 255).astype(np.uint8)
                out.write(f)
            out.release()
            rows.append(dict(clip=name, participant=p, prompt=prompt, posed=True, lighting=light,
                             pose="frontal", glasses="no", skin_tone=tone, consent="yes"))
    with open(root / "data" / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def run(cmd, env):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run([sys.executable] + cmd, cwd=REPO, env=env, text=True,
                       capture_output=True, timeout=900)
    print(r.stdout[-1500:])
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {cmd}")


def main():
    root = Path(tempfile.mkdtemp(prefix="hm_e2e_"))
    env = {**os.environ, "HM_ROOT": str(root)}
    make_memes(root / "memes", n=6)
    shutil.copy(REPO / "memes" / "hamster_laughing.png", root / "memes")
    with open(root / "memes" / "memes.csv", "a") as f:
        f.write('hamster_laughing.png,happy,"laughing hysterically"\n')
    csv_txt = (root / "memes" / "memes.csv").read_text().splitlines()
    csv_txt[0] += ",gesture"
    csv_txt = [csv_txt[0]] + [l + (",heart_hands" if "hamster" in l else ",") for l in csv_txt[1:]]
    (root / "memes" / "memes.csv").write_text("\n".join(csv_txt) + "\n")
    fake_fer(root / "fer.csv")
    clips = fake_clips(root)
    pn = ["--pretrained", "none"]

    run(["scripts/train_baseline.py", "--fer", str(root / "fer.csv"), "--epochs", "1",
         "--batch", "32", "--workers", "0", "--no-imagenet"], env)
    run(["scripts/build_index.py"] + pn, env)
    run(["scripts/annotate.py", "--extract-only", "--detector", "haar"], env)

    # write annotations as a human would (every clip's frames get one label set)
    import importlib
    os.environ["HM_ROOT"] = str(root)
    from hamstermirror import config as cfg
    importlib.reload(cfg)
    gt = {"happy": "hamster_laughing.png|m0.png", "sad": "m1.png", "neutral": "m4.png"}
    with open(root / "data" / "annotations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "clip", "frame_idx", "annotator", "emotion", "gt_memes"])
        n = 0
        for c in clips:
            for p in sorted((root / "data" / "frames" / Path(c["clip"]).stem).glob("*.png")):
                w.writerow([p.relative_to(root), c["clip"], int(p.stem.split("_")[1]),
                            "tester", c["prompt"], gt[c["prompt"]]])
                n += 1
    assert n >= 12, f"only {n} face crops extracted"
    # fake gesture metadata (the hand model can't be downloaded in CI)
    with open(root / "data" / "frame_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "context", "gesture"])
        for c in clips:
            for p in sorted((root / "data" / "frames" / Path(c["clip"]).stem).glob("f_*.png")):
                rel = p.relative_to(root)
                w.writerow([rel, rel, "heart_hands" if c["prompt"] == "happy" else ""])

    run(["scripts/train_projection.py", "--epochs", "50"] + pn, env)
    run(["eval/eval_expression.py", "--fer", str(root / "fer.csv")], env)
    run(["eval/eval_retrieval.py", "--proj-cv"] + pn, env)
    run(["eval/eval_realtime.py", "--detector", "haar"] + pn, env)
    run(["app.py", "--video", str(root / "data" / "clips" / clips[0]["clip"]), "--no-window",
         "--max-frames", "40", "--detector", "haar", "--method", "clip"] + pn, env)

    (root / "results").mkdir(exist_ok=True)
    rng = np.random.default_rng(2)
    with open(root / "results" / "user_study.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "trial", "order", "proposed_method", "meme_baseline",
                    "meme_proposed", "rating_baseline", "rating_proposed", "same_meme", "time"])
        for p in range(6):
            for t in range(5):
                w.writerow([f"u{p}", t, "proposed-baseline", "expr", "m0.png", "m1.png",
                            rng.integers(1, 4), rng.integers(2, 6), False, 0])
    run(["eval/user_study.py", "--analyze"], env)

    for name in ["expression.md", "retrieval.md", "realtime.md", "user_study.md",
                 "retrieval_per_frame.csv", "meme_sheet.png"]:
        assert (root / "results" / name).exists(), name
    print(f"\nE2E SMOKE TEST PASSED (artifacts in {root})")


if __name__ == "__main__":
    main()

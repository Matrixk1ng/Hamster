"""Fine-tune the optional projection head on annotated (face crop, meme) pairs.
Uses the first-listed (best) meme of each annotation as the target.
For an honest number, use eval/eval_retrieval.py --proj-cv (leave-one-participant-out)."""
import argparse

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.clip_retrieval import ClipEncoder, MemeIndex
from hamstermirror.data import load_annotations
from hamstermirror.memes import load_memes
from hamstermirror.projection import train_projection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained", default=config.CLIP_PRETRAINED)
    ap.add_argument("--text-weight", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    memes = load_memes()
    rows = [r for r in load_annotations(memes=memes) if r["gt_idx"]]
    if len(rows) < 10:
        raise SystemExit(f"Only {len(rows)} annotated frames, annotate more first.")
    enc = ClipEncoder(config.CLIP_MODEL, args.pretrained)
    index = MemeIndex.build(enc, memes)
    X = enc.encode_images([np.array(Image.open(r["frame"]).convert("RGB")) for r in rows])
    y = np.array([r["gt_idx"][0] for r in rows])
    head = train_projection(X, y, index.embeddings(args.text_weight), epochs=args.epochs)
    head.save(config.PROJECTION_CKPT, text_weight=args.text_weight)
    print(f"saved {config.PROJECTION_CKPT} (trained on {len(rows)} frames)")


if __name__ == "__main__":
    main()

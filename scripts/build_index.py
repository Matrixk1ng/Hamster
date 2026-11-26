"""Encode every meme in memes/memes.csv with CLIP (image + description) and save the index.
Re-run whenever you add or edit memes."""
import argparse

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.clip_retrieval import ClipEncoder, MemeIndex
from hamstermirror.memes import load_memes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memes", default=str(config.MEME_CSV))
    ap.add_argument("--out", default=str(config.MEME_INDEX))
    ap.add_argument("--model", default=config.CLIP_MODEL)
    ap.add_argument("--pretrained", default=config.CLIP_PRETRAINED)
    args = ap.parse_args()

    memes = load_memes(args.memes)
    enc = ClipEncoder(args.model, args.pretrained)
    index = MemeIndex.build(enc, memes)
    index.save(args.out)
    print(f"indexed {len(memes)} memes -> {args.out}")

    # sanity check: which meme description does each meme image match best?
    sims = index.image_emb @ index.text_emb.T
    agree = sum(int(sims[i].argmax() == i) for i in range(len(memes)))
    print(f"image->own-description top-1: {agree}/{len(memes)} "
          "(low values mean descriptions are vague or too similar)")


if __name__ == "__main__":
    main()

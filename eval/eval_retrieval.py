"""Retrieval quality: top-1 / top-3 hit rate against human-annotated meme matches.

Compares the baseline (emotion -> meme) and CLIP retrieval, ablates the CLIP
text weight and temporal smoothing, optionally evaluates the projection head
with leave-one-participant-out training, and breaks results down by lighting,
head pose, glasses, skin tone and posed/spontaneous.

  python eval/eval_retrieval.py --proj-cv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.data import load_annotations, read_csv
from hamstermirror.gestures import gesture_offsets
from hamstermirror.memes import load_memes
from hamstermirror.metrics import count_switches, group_mean, md_table, mean, topk_hit
from hamstermirror.pipeline import build_matchers
from hamstermirror.smoothing import Smoother

TEXT_WEIGHTS = [0.0, 0.3, 0.6, 1.0]
BREAKDOWN = ["gt_gesture", "lighting", "pose", "glasses", "skin_tone", "posed"]


def run_sequences(rows, vecs, score_fn, normalize, smoothing, margin):
    """Per-frame scores and choices, smoothing applied within each clip in frame order."""
    by_clip = defaultdict(list)
    for i, r in enumerate(rows):
        by_clip[(r["clip"], r["annotator"])].append(i)
    scores, choice, switches, steps = [None] * len(rows), [None] * len(rows), 0, 0
    for idxs in by_clip.values():
        idxs.sort(key=lambda i: int(rows[i]["frame_idx"]))
        sm = Smoother(smoothing, normalize=normalize, margin=margin)
        seq = []
        for i in idxs:
            s = score_fn(sm.smooth_vector(vecs[i]), i)
            scores[i], choice[i] = s, sm.choose(s)
            seq.append(choice[i])
        switches += count_switches(seq)
        steps += max(len(seq) - 1, 0)
    return scores, choice, switches / max(steps, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pretrained", default=config.CLIP_PRETRAINED)
    ap.add_argument("--proj-cv", action="store_true",
                    help="leave-one-participant-out projection head evaluation")
    ap.add_argument("--main-text-weight", type=float, default=0.6)
    args = ap.parse_args()

    memes = load_memes()
    rows = [r for r in load_annotations(memes=memes) if r["gt_idx"]]
    if not rows:
        raise SystemExit("No annotations. Run scripts/record_clips.py then scripts/annotate.py.")
    faces = [np.array(Image.open(r["frame"]).convert("RGB")) for r in rows]
    for r in rows:
        r["gt_gesture"] = memes[r["gt_idx"][0]].gesture or "none"
    _, matchers = build_matchers(memes, pretrained=args.pretrained, use_projection=False)
    print(f"{len(rows)} annotated frames, {len(memes)} memes, methods: {list(matchers)}")

    # vectors computed once per model
    methods = {}  # name -> (vecs, score_fn, normalize, margin)
    if "baseline" in matchers:
        b = matchers["baseline"]
        methods["baseline"] = (b.vectors(faces), lambda v, i: b.scores(v), False, 0.05)
    for name in ("expr", "expr+clip"):
        if name in matchers:
            m = matchers[name]
            methods[name] = (m.vectors(faces), lambda v, i, m=m: m.scores(v), False, m.margin)
    clip = matchers["clip"]
    clip_vecs = clip.vectors(faces)
    for w in TEXT_WEIGHTS:
        emb = clip.index.embeddings(w)
        methods[f"clip (text_w={w})"] = (clip_vecs, lambda v, i, e=emb: e @ v, True, 0.01)

    if args.proj_cv:
        from hamstermirror.projection import train_projection

        parts = sorted({r.get("participant", "?") for r in rows})
        if len(parts) < 2:
            print("projection CV needs at least 2 participants, skipping")
        else:
            emb = clip.index.embeddings(args.main_text_weight)
            proj = np.zeros_like(clip_vecs)
            for p in parts:
                tr = [i for i, r in enumerate(rows) if r.get("participant") != p]
                te = [i for i, r in enumerate(rows) if r.get("participant") == p]
                head = train_projection(clip_vecs[tr], [rows[i]["gt_idx"][0] for i in tr], emb,
                                        verbose=False)
                proj[te] = head.apply(clip_vecs[te])
            methods[f"clip+proj LOPO (text_w={args.main_text_weight})"] = (
                proj, lambda v, i, e=emb: e @ v, True, 0.01)

    # gesture-aware variants: CLIP on face+hands context crops, plus the gesture bonus
    meta = {}
    for m in read_csv(config.FRAME_META_CSV):
        meta[str(config.ROOT / m["frame"]) if not Path(m["frame"]).is_absolute() else m["frame"]] = m
    if meta:
        gest = [meta.get(r["frame"], {}).get("gesture") or None for r in rows]
        ctx = []
        for r, f in zip(rows, faces):
            c = meta.get(r["frame"], {}).get("context")
            ctx.append(np.array(Image.open(config.ROOT / c).convert("RGB")) if c else f)
        print(f"gesture metadata for {sum(r['frame'] in meta for r in rows)}/{len(rows)} frames, "
              f"{sum(g is not None for g in gest)} with a gesture")
        emb = clip.index.embeddings(args.main_text_weight)
        off = lambda i, bonus: gesture_offsets(memes, gest[i], bonus, bonus / 2)
        methods[f"clip+gestures (text_w={args.main_text_weight})"] = (
            clip.vectors(ctx), lambda v, i, e=emb: e @ v + off(i, 0.1), True, 0.01)
        if "expr" in methods:
            ex = matchers["expr"]
            methods["expr+gestures"] = (methods["expr"][0], lambda v, i: ex.scores(v) + off(
                i, ex.gesture_bonus), False, ex.margin)
        if "baseline" in matchers:
            b = matchers["baseline"]
            methods["baseline+gestures"] = (methods["baseline"][0],
                                            lambda v, i: b.scores(v) + off(i, 0.5), False, 0.05)
    else:
        print("no data/frame_meta.csv (re-run scripts/annotate.py with the hand model "
              "downloaded to evaluate gestures)")

    chance1 = mean(len(r["gt_idx"]) / len(memes) for r in rows)
    chance3 = mean(min(1.0, 3 * len(r["gt_idx"]) / len(memes)) for r in rows)
    lines = ["# Retrieval quality", "",
             f"{len(rows)} annotated frames from {len({r['clip'] for r in rows})} clips, "
             f"{len(memes)} memes. Chance: top-1 {chance1:.3f}, top-3 about {chance3:.3f}.", ""]
    table, per_frame = [], {}
    for name, (vecs, fn, norm, margin) in methods.items():
        res = {}
        for sm in (False, True):
            scores, choice, flick = run_sequences(rows, vecs, fn, norm, sm, margin)
            t1 = mean(float(c in r["gt_idx"]) for c, r in zip(choice, rows))
            t3 = mean(topk_hit(s, r["gt_idx"], 3) for s, r in zip(scores, rows))
            res[sm] = (t1, t3, flick, choice)
        table.append([name, res[False][0], res[False][1], res[True][0], res[True][1],
                      res[False][2], res[True][2]])
        per_frame[name] = res[False][3]
    lines += ["## Main results", "",
              md_table(["method", "top-1", "top-3", "top-1 smoothed", "top-3 smoothed",
                        "switch rate", "switch rate smoothed"], table), "",
              "_switch rate = fraction of consecutive sampled frames where the shown meme changed._", ""]

    focus = [n for n in per_frame if n.startswith(("baseline", "expr"))
             or f"text_w={args.main_text_weight}" in n]
    for key in BREAKDOWN:
        hdr, body, groups = [key], [], None
        for n in focus:
            for r, c in zip(rows, per_frame[n]):
                r["_hit"] = float(c in r["gt_idx"])
            g = group_mean(rows, key, "_hit")
            groups = groups or {k: [k] for k in g}
            for k, (v, cnt) in g.items():
                groups[k].append(v)
            hdr.append(f"{n} top-1")
        counts = {k: cnt for k, (_, cnt) in group_mean(rows, key, "_hit").items()}
        body = [v + [counts[k]] for k, v in groups.items()]
        lines += [f"## Failure analysis by {key}", "", md_table(hdr + ["n"], body), ""]

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "retrieval.md").write_text("\n".join(lines))
    with open(config.RESULTS_DIR / "retrieval_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "clip", "participant", "gt"] + list(per_frame))
        for i, r in enumerate(rows):
            w.writerow([r["frame"], r["clip"], r.get("participant", ""), r["gt_memes"]]
                       + [memes[per_frame[n][i]].file for n in per_frame])
    print("\n".join(lines))
    print(f"\n-> {config.RESULTS_DIR / 'retrieval.md'}")


if __name__ == "__main__":
    main()

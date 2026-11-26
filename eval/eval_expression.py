"""Expression accuracy of the baseline: accuracy and macro-F1 on the FER2013
test split and on our own annotated webcam frames.

  python eval/eval_expression.py --fer data/fer2013.csv
"""
import argparse

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.baseline import BaselineClassifier
from hamstermirror.data import load_annotations, load_fer2013
from hamstermirror.metrics import group_mean, md_table


def report(name, y, p):
    acc, f1 = accuracy_score(y, p), f1_score(y, p, average="macro",
                                              labels=range(len(config.FER_CLASSES)))
    cm = confusion_matrix(y, p, labels=range(len(config.FER_CLASSES)))
    out = [f"## {name}", f"n={len(y)}  accuracy **{acc:.3f}**  macro-F1 **{f1:.3f}**", "",
           "Confusion matrix (rows = true, cols = predicted):", "",
           md_table(["true \\ pred"] + config.FER_CLASSES,
                    [[c] + list(map(int, cm[i])) for i, c in enumerate(config.FER_CLASSES)]), ""]
    print("\n".join(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fer", help="fer2013.csv or folder (skip to only evaluate own clips)")
    ap.add_argument("--ckpt", default=str(config.BASELINE_CKPT))
    args = ap.parse_args()

    clf = BaselineClassifier(args.ckpt)
    lines = ["# Expression accuracy (baseline CNN)", ""]
    if args.fer:
        x, y = load_fer2013(args.fer, "test")
        probs = np.concatenate([clf.predict_batch([np.stack([im] * 3, -1) for im in x[i:i + 256]])
                                for i in range(0, len(x), 256)])
        lines += report("FER2013 test", y, probs.argmax(1))

    rows = load_annotations()
    if rows:
        cls = {c: i for i, c in enumerate(config.FER_CLASSES)}
        faces = [np.array(Image.open(r["frame"]).convert("RGB")) for r in rows]
        pred = np.concatenate([clf.predict_batch(faces[i:i + 64]).argmax(1)
                               for i in range(0, len(faces), 64)])
        y = np.array([cls[r["emotion"]] for r in rows])
        lines += report("Own webcam clips", y, pred)
        for r, p, t in zip(rows, pred, y):
            r["correct"] = float(p == t)
        for key in ["lighting", "pose", "glasses", "skin_tone", "posed", "participant"]:
            g = group_mean(rows, key, "correct")
            lines += [f"Accuracy by {key}:", "",
                      md_table([key, "accuracy", "n"], [[k, v, n] for k, (v, n) in g.items()]), ""]
    else:
        lines.append("_No own-clip annotations yet (data/annotations.csv)._")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "expression.md").write_text("\n".join(lines))
    print(f"\n-> {config.RESULTS_DIR / 'expression.md'}")


if __name__ == "__main__":
    main()

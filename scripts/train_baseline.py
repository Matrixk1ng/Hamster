"""Fine-tune the baseline ResNet18 expression classifier on FER2013.

  python scripts/train_baseline.py --fer data/fer2013.csv --epochs 15
  python scripts/train_baseline.py --fer data/fer2013 --epochs 15      # folder layout

Saves the best checkpoint (by validation macro-F1) to checkpoints/baseline.pt.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from hamstermirror import config
from hamstermirror.baseline import build_model, eval_transform, train_transform
from hamstermirror.data import ArrayDataset, load_fer2013


def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fer", required=True, help="fer2013.csv or folder with train/ and test/")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only N training images (smoke test)")
    ap.add_argument("--no-imagenet", action="store_true")
    ap.add_argument("--out", default=str(config.BASELINE_CKPT))
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = config.best_device()

    tr_x, tr_y = load_fer2013(args.fer, "train")
    try:
        va_x, va_y = load_fer2013(args.fer, "val")
    except Exception:
        va_x, va_y = [], np.array([])
    if len(va_x) == 0:  # folder layout: hold out 10% of train
        perm = rng.permutation(len(tr_x))
        cut = len(perm) // 10
        va_x, va_y = [tr_x[i] for i in perm[:cut]], tr_y[perm[:cut]]
        tr_x, tr_y = [tr_x[i] for i in perm[cut:]], tr_y[perm[cut:]]
    te_x, te_y = load_fer2013(args.fer, "test")
    if args.limit:
        tr_x, tr_y = tr_x[: args.limit], tr_y[: args.limit]
    print(f"train {len(tr_x)}  val {len(va_x)}  test {len(te_x)}  device {device}")

    # class-balanced loss: FER2013 has very few 'disgust' images
    counts = np.bincount(tr_y, minlength=len(config.FER_CLASSES)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(counts), device=device)

    mk = lambda x, y, tf, sh: DataLoader(ArrayDataset(x, y, tf), batch_size=args.batch,
                                        shuffle=sh, num_workers=args.workers)
    tr = mk(tr_x, tr_y, train_transform(), True)
    va = mk(va_x, va_y, eval_transform(), False)
    te = mk(te_x, te_y, eval_transform(), False)

    model = build_model(imagenet=not args.no_imagenet).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tr))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    config.CKPT_DIR.mkdir(exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        acc, f1 = evaluate(model, va, device)
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / n:.4f}  val acc {acc:.4f}  "
              f"val macro-F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best:
            best = f1
            torch.save({"model": model.state_dict(), "classes": config.FER_CLASSES}, args.out)

    model.load_state_dict(torch.load(args.out, map_location=device)["model"])
    acc, f1 = evaluate(model, te, device)
    print(f"TEST  acc {acc:.4f}  macro-F1 {f1:.4f}  (best checkpoint -> {args.out})")


if __name__ == "__main__":
    main()

"""Dataset helpers: FER2013 loading, self-collected clip metadata, annotations."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import config

CLIP_FIELDS = ["clip", "participant", "prompt", "posed", "lighting", "pose", "glasses",
               "skin_tone", "consent"]
ANNOT_FIELDS = ["frame", "clip", "frame_idx", "annotator", "emotion", "gt_memes"]


# ---------------------------------------------------------------- FER2013
def load_fer2013(path, split="train"):
    """Returns (images uint8 HxW list, labels int array).

    Supports the Kaggle CSV (emotion,pixels,Usage) and the folder layout
    path/{train,test}/{angry,disgust,...}/*.png. CSV splits: train=Training,
    val=PublicTest, test=PrivateTest. Folder splits: train, test (val is
    carved from train by the training script).
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".csv":
        usage = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}[split]
        imgs, labels = [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("Usage", "Training") != usage:
                    continue
                px = np.array(row["pixels"].split(), dtype=np.uint8)
                side = int(np.sqrt(px.size))
                imgs.append(px.reshape(side, side))
                labels.append(int(row["emotion"]))
        return imgs, np.array(labels)
    sub = path / ("train" if split in ("train", "val") else "test")
    if not sub.exists():
        raise FileNotFoundError(f"Expected {sub} or a fer2013.csv file")
    imgs, labels = [], []
    for ci, cls in enumerate(config.FER_CLASSES):
        for p in sorted((sub / cls).glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                imgs.append(np.array(Image.open(p).convert("L")))
                labels.append(ci)
    return imgs, np.array(labels)


class ArrayDataset(Dataset):
    def __init__(self, imgs, labels, transform):
        self.imgs, self.labels, self.tf = imgs, labels, transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = Image.fromarray(self.imgs[i]).convert("RGB")
        return self.tf(im), int(self.labels[i])


# ---------------------------------------------------------------- CSV helpers
def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def load_annotations(annot_csv=config.ANNOT_CSV, clips_csv=config.CLIP_META_CSV, memes=None):
    """Annotation rows joined with clip metadata. Adds 'gt_idx' (list of meme
    indices) when memes are given, and resolves 'frame' to an absolute path."""
    clips = {r["clip"]: r for r in read_csv(clips_csv)}
    file_to_idx = {m.file: i for i, m in enumerate(memes)} if memes else {}
    rows = []
    for r in read_csv(annot_csv):
        r = {**clips.get(r["clip"], {}), **r}
        p = Path(r["frame"])
        r["frame"] = str(p if p.is_absolute() else config.ROOT / p)
        files = [f for f in r["gt_memes"].split("|") if f]
        r["gt_files"] = files
        if memes:
            r["gt_idx"] = [file_to_idx[f] for f in files if f in file_to_idx]
        rows.append(r)
    return rows

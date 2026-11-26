"""Meme gallery: memes/memes.csv with columns file, emotion, description, gesture.
gesture is optional (blank = no hand gesture), one of config.GESTURES."""
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import config


@dataclass
class Meme:
    file: str
    emotion: str
    description: str
    path: Path
    gesture: str = ""

    @property
    def name(self) -> str:
        return Path(self.file).stem

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


def load_memes(csv_path=config.MEME_CSV, meme_dir=None) -> list:
    csv_path = Path(csv_path)
    meme_dir = Path(meme_dir) if meme_dir else csv_path.parent
    memes = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in f if r.strip() and not r.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        emotion = row["emotion"].strip().lower()
        if emotion not in config.FER_CLASSES:
            raise ValueError(
                f"{row['file']}: emotion '{emotion}' must be one of {config.FER_CLASSES}"
            )
        gesture = (row.get("gesture") or "").strip().lower()
        if gesture and gesture not in config.GESTURES:
            raise ValueError(f"{row['file']}: gesture '{gesture}' must be blank or one of "
                             f"{config.GESTURES}")
        path = meme_dir / row["file"].strip()
        if not path.exists():
            raise FileNotFoundError(f"Meme image missing: {path}")
        memes.append(Meme(row["file"].strip(), emotion, row["description"].strip(), path, gesture))
    if not memes:
        raise ValueError(f"No memes listed in {csv_path}")
    return memes

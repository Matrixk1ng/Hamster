import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EMOS = ["happy", "sad", "angry", "surprise", "neutral"]


def make_memes(folder: Path, n=5):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = ["file,emotion,description"]
    for i in range(n):
        im = Image.new("RGB", (200, 220), tuple(int(c) for c in rng.integers(150, 255, 3)))
        d = ImageDraw.Draw(im)
        d.ellipse((30, 30, 170, 200), outline=(0, 0, 0), width=3)
        d.arc((70, 110 + 10 * i, 130, 160), 0, 180 * (1 + i % 2), fill=(0, 0, 0), width=4)
        im.save(folder / f"m{i}.png")
        lines.append(f'm{i}.png,{EMOS[i % len(EMOS)]},"test meme number {i}"')
    (folder / "memes.csv").write_text("\n".join(lines) + "\n")
    return folder / "memes.csv"


@pytest.fixture
def meme_csv(tmp_path):
    return make_memes(tmp_path / "memes")


@pytest.fixture(scope="session")
def clip_encoder():
    from hamstermirror.clip_retrieval import ClipEncoder

    return ClipEncoder("ViT-B-32", pretrained="none", device="cpu")

import numpy as np
from skimage import data
import cv2

from hamstermirror.clip_retrieval import ClipMatcher, MemeIndex
from hamstermirror.face import FaceDetector
from hamstermirror.memes import load_memes
from hamstermirror.pipeline import HamsterMirror


def test_index_roundtrip_and_matcher(meme_csv, clip_encoder, tmp_path):
    memes = load_memes(meme_csv)
    idx = MemeIndex.build(clip_encoder, memes)
    assert idx.image_emb.shape == (5, 512) and idx.text_emb.shape == (5, 512)
    idx.save(tmp_path / "i.npz")
    idx2 = MemeIndex.load(tmp_path / "i.npz")
    assert idx2.files == idx.files and np.allclose(idx2.image_emb, idx.image_emb)
    m = ClipMatcher(memes, clip_encoder, idx2, text_weight=0.0)
    # a meme image must retrieve itself with pure image embeddings
    v = m.vector(np.array(memes[3].load().resize((224, 224))))
    assert int(np.argmax(m.scores(v))) == 3


def test_live_pipeline_on_real_face(meme_csv, clip_encoder):
    memes = load_memes(meme_csv)
    m = ClipMatcher(memes, clip_encoder, MemeIndex.build(clip_encoder, memes))
    hm = HamsterMirror(FaceDetector("haar"), m, smoothing=True)
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    rng = np.random.default_rng(0)
    choices = []
    for _ in range(8):
        f = np.clip(img.astype(int) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)
        r = hm.process(f)
        assert r.box is not None and len(r.topk) == 3
        assert set(r.timings) >= {"detect", "embed", "match", "total"}
        choices.append(r.meme_idx)
    assert len(set(choices)) == 1  # smoothing keeps a static face stable
    r = hm.process(np.zeros_like(img))  # no face: keeps last meme
    assert r.box is None and r.meme_idx == choices[-1]

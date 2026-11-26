import numpy as np
import pytest

from hamstermirror import config
from hamstermirror.face import FaceDetector, crop_face
from hamstermirror.memes import load_memes
from hamstermirror.metrics import count_switches, topk_hit
from hamstermirror.smoothing import EMA, Hysteresis, Smoother


def test_ema_normalizes_and_lags():
    e = EMA(alpha=0.5)
    a = e.update([1, 0])
    b = e.update([0, 1])
    assert np.allclose(a, [1, 0])
    assert np.isclose(np.linalg.norm(b), 1) and np.allclose(b[0], b[1])


def test_hysteresis_needs_margin_and_persistence():
    h = Hysteresis(margin=0.1, min_frames=3)
    assert h.update([0.9, 0.1]) == 0
    # challenger only slightly ahead: never switches
    for _ in range(10):
        assert h.update([0.50, 0.55]) == 0
    # clearly ahead but only for 2 frames, then a blip back: no switch
    h.update([0.1, 0.9]); h.update([0.1, 0.9]); h.update([0.9, 0.1])
    assert h.current == 0
    for _ in range(3):
        out = h.update([0.1, 0.9])
    assert out == 1


def test_smoother_reduces_flicker():
    rng = np.random.default_rng(0)
    base = np.array([0.6, 0.58, 0.1])
    noisy = [base + rng.normal(0, 0.05, 3) for _ in range(200)]
    raw = [int(np.argmax(s)) for s in noisy]
    sm = Smoother(True, alpha=0.3, margin=0.02, min_frames=3, normalize=False)
    smoothed = [sm.choose(sm.smooth_vector(s)) for s in noisy]
    assert count_switches(smoothed) < count_switches(raw) / 4


def test_crop_face_square_and_border_safe():
    frame = np.zeros((480, 640, 3), np.uint8)
    for box in [(300, 200, 100, 120), (0, 0, 80, 80), (600, 440, 60, 60)]:
        c = crop_face(frame, box, size=224)
        assert c.shape == (224, 224, 3)


def test_haar_detector_runs():
    d = FaceDetector("haar")
    assert d.detect(np.zeros((240, 320, 3), np.uint8)) == []


def test_haar_finds_real_face():
    from skimage import data
    import cv2

    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    box = FaceDetector("haar").largest(img)
    assert box is not None and box[2] > 50


def test_load_memes(meme_csv, tmp_path):
    memes = load_memes(meme_csv)
    assert len(memes) == 5 and memes[0].emotion == "happy"
    bad = tmp_path / "memes" / "bad.csv"
    bad.write_text("file,emotion,description\nm0.png,smug,x\n")
    with pytest.raises(ValueError):
        load_memes(bad)


def test_repo_meme_gallery_valid():
    memes = load_memes()
    assert any("hamster" in m.file for m in memes)


def test_topk():
    assert topk_hit([0.1, 0.9, 0.5], [2], 1) == 0
    assert topk_hit([0.1, 0.9, 0.5], [2], 2) == 1


def test_baseline_matcher_maps_emotion(meme_csv):
    from hamstermirror.baseline import BaselineClassifier, BaselineMatcher, build_model

    memes = load_memes(meme_csv)
    clf = BaselineClassifier(model=build_model(imagenet=False), device="cpu")
    m = BaselineMatcher(memes, clf)
    probs = np.zeros(len(config.FER_CLASSES))
    probs[config.FER_CLASSES.index("sad")] = 1
    assert memes[int(np.argmax(m.scores(probs)))].emotion == "sad"
    v = m.vector(np.zeros((224, 224, 3), np.uint8))
    assert v.shape == (7,) and np.isclose(v.sum(), 1, atol=1e-4)


def test_projection_learns():
    from hamstermirror.projection import train_projection

    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 16)); M /= np.linalg.norm(M, axis=1, keepdims=True)
    y = rng.integers(0, 4, 200)
    # faces live in a rotated copy of meme space, so raw cosine is poor
    R = np.linalg.qr(rng.normal(size=(16, 16)))[0]
    X = (M[y] + rng.normal(0, 0.1, (200, 16))) @ R
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    head = train_projection(X, y, M, epochs=300, lr=1e-2, verbose=False)
    acc = (np.argmax(head.apply(X) @ M.T, 1) == y).mean()
    assert acc > 0.9

"""Expression matcher tests with synthetic MediaPipe blendshapes against the real
meme profiles in memes/expressions.csv. These encode the failure cases seen live:
a neutral face must not match 'deeply unimpressed', a normal smile must not match
'forced toothy grin', and a smirk should match the closest remaining grin."""
import numpy as np
import pytest

from hamstermirror.expression import (FEATURES, Calibration, ExprMatcher,
                                      features_from_blendshapes, load_profiles)
from hamstermirror.memes import load_memes

FACES = {
    "neutral": {},
    "smirk": dict(mouthSmileLeft=.7, mouthSmileRight=.2, browDownLeft=.45, browDownRight=.45,
                  eyeSquintLeft=.45, eyeSquintRight=.45),
    "big_toothy_grin": dict(mouthSmileLeft=.85, mouthSmileRight=.85, mouthUpperUpLeft=.6,
                            mouthUpperUpRight=.6, mouthLowerDownLeft=.5, mouthLowerDownRight=.5,
                            eyeWideLeft=.35, eyeWideRight=.35, mouthStretchLeft=.3,
                            mouthStretchRight=.3),
    "normal_smile": dict(mouthSmileLeft=.7, mouthSmileRight=.7, mouthUpperUpLeft=.2,
                         mouthUpperUpRight=.2, mouthLowerDownLeft=.2, mouthLowerDownRight=.2,
                         eyeSquintLeft=.35, eyeSquintRight=.35),
    "jaw_drop": dict(jawOpen=.6, eyeWideLeft=.45, eyeWideRight=.45, browInnerUp=.7),
    "unimpressed": dict(browDownLeft=.4, browDownRight=.4, eyeSquintLeft=.4, eyeSquintRight=.4,
                        mouthFrownLeft=.3, mouthFrownRight=.3, mouthPressLeft=.2,
                        mouthPressRight=.2),
    "pout": dict(mouthPucker=.8, eyeSquintLeft=.2, eyeSquintRight=.2),
}
EXPECTED = {
    "neutral": "h14_tiny_blank_stare.jpg",
    "smirk": "h11_smug_grin.jpg",
    "big_toothy_grin": "h15_forced_toothy_grin.jpg",
    "normal_smile": "h17_sweet_content_smile.jpg",
    "jaw_drop": "h09_jaw_drop_shock.jpg",
    "unimpressed": "h05_deeply_unimpressed.jpg",
    "pout": "h06_duckface_pout.jpg",
}


class FakeExtractor:
    def __init__(self):
        self.b = {}

    def features(self, rgb):
        return features_from_blendshapes(self.b)


@pytest.fixture(scope="module")
def matcher():
    memes = load_memes()
    return memes, FakeExtractor(), lambda ex, calib=None: ExprMatcher(memes, ex, calib, verbose=False)


def test_all_repo_memes_have_profiles():
    memes = load_memes()
    P = load_profiles(memes)
    assert not np.isnan(P).any(), [m.file for m, r in zip(memes, P) if np.isnan(r).any()]


@pytest.mark.parametrize("face", list(FACES))
def test_face_maps_to_expected_meme(matcher, face):
    memes, ex, make = matcher
    m = make(ex)
    ex.b = FACES[face]
    s = m.scores(m.vector(np.zeros((224, 224, 3), np.uint8)))
    top = [memes[i].file for i in np.argsort(-s)[:3]]
    assert top[0] == EXPECTED[face], f"{face}: top-3 {top}"


def test_neutral_calibration_removes_resting_face(matcher):
    """Someone whose resting face looks slightly frowny should read as neutral after calibration."""
    memes, ex, make = matcher
    resting = dict(browDownLeft=.25, browDownRight=.25, mouthFrownLeft=.2, mouthFrownRight=.2,
                   eyeSquintLeft=.2, eyeSquintRight=.2)
    ex.b = resting
    uncal = make(ex)
    s = uncal.scores(uncal.vector(None))
    assert memes[int(np.argmax(s))].file != "h14_tiny_blank_stare.jpg"  # the bug
    cal = make(ex, Calibration(neutral=features_from_blendshapes(resting)))
    s = cal.scores(cal.vector(None))
    assert memes[int(np.argmax(s))].file == "h14_tiny_blank_stare.jpg"


def test_enrolled_prototype_overrides(matcher, tmp_path):
    memes, ex, make = matcher
    weird = np.zeros(len(FEATURES), np.float32)
    weird[FEATURES.index("cheek_puff")] = 1.0
    calib = Calibration(protos={"h12_stoic_bat_mask.jpg": weird})
    calib.save(tmp_path / "c.npz")
    calib = Calibration.load(tmp_path / "c.npz")
    m = make(ex, calib)
    ex.b = dict(cheekPuff=.5)
    assert memes[int(np.argmax(m.scores(m.vector(None))))].file == "h12_stoic_bat_mask.jpg"


def test_no_face_keeps_last_vector(matcher):
    memes, ex, make = matcher
    m = make(ex)
    ex.b = FACES["jaw_drop"]
    v1 = m.vector(None)

    class Nothing:
        def features(self, rgb):
            return None
    m.extractor = Nothing()
    assert np.allclose(m.vector(None), v1)


def test_memes_without_profile_never_win(meme_csv):
    memes = load_memes(meme_csv)
    m = ExprMatcher(memes, FakeExtractor(), verbose=False)  # synthetic memes: no profiles
    assert not m.valid.any()
    assert (m.scores(np.zeros(len(FEATURES))) <= -1e3).all()


def test_pipeline_and_hybrid(matcher, clip_encoder):
    import cv2
    from skimage import data
    from hamstermirror.clip_retrieval import ClipMatcher, MemeIndex
    from hamstermirror.expression import ExprClipMatcher
    from hamstermirror.face import FaceDetector
    from hamstermirror.pipeline import HamsterMirror

    memes, ex, make = matcher
    em = make(ex)
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    hm = HamsterMirror(FaceDetector("haar"), em)
    assert hm.smoother.hyst.margin == em.margin and hm.gesture_bonus == em.gesture_bonus
    ex.b = FACES["jaw_drop"]
    for _ in range(6):
        r = hm.process(img)
    assert memes[r.meme_idx].file == EXPECTED["jaw_drop"]
    ex.b = FACES["unimpressed"]  # switches after hysteresis
    for _ in range(10):
        r = hm.process(img)
    assert memes[r.meme_idx].file == EXPECTED["unimpressed"]

    hy = ExprClipMatcher(em, ClipMatcher(memes, clip_encoder, MemeIndex.build(clip_encoder, memes)),
                         clip_weight=0.0)
    ex.b = FACES["pout"]
    s = hy.scores(hy.vector(np.zeros((224, 224, 3), np.uint8)))
    assert memes[int(np.argmax(s))].file == EXPECTED["pout"]

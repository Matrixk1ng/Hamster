import numpy as np
import pytest

from hamstermirror.gestures import (HEART_HAND_REACH, GestureSmoother, classify,
                                    context_box, fingers_extended, gesture_offsets)
from hamstermirror.hands import Hand

FACE = (200, 100, 200, 200)  # x, y, w, h  -> center (300, 200), face width 200
P = 80  # palm length in px


def make_hand(fingers=(True, True, True, True, True), x=0.0, y=0.0):
    """Upright synthetic hand, wrist at (x, y). fingers = [thumb, index, middle, ring, pinky]."""
    lm = np.zeros((21, 2), np.float32)
    lm[0] = (x, y)
    # thumb
    lm[1] = (x - 0.3 * P, y - 0.2 * P)
    lm[2] = (x - 0.5 * P, y - 0.4 * P)
    lm[3] = (x - 0.65 * P, y - 0.55 * P)
    lm[4] = (x - 1.0 * P, y - 0.5 * P) if fingers[0] else (x - 0.25 * P, y - 0.6 * P)
    for f, dx in zip(range(1, 5), (-0.3, -0.1, 0.1, 0.3)):
        mcp = 1 + 4 * f
        fx, my = x + dx * P, y - P
        lm[mcp] = (fx, my)
        if fingers[f]:
            lm[mcp + 1:mcp + 4] = [(fx, my - 0.4 * P), (fx, my - 0.7 * P), (fx, my - P)]
        else:
            lm[mcp + 1:mcp + 4] = [(fx, my - 0.3 * P), (fx, my - 0.15 * P), (fx, my + 0.1 * P)]
    return Hand(lm)


def at_palm(hand, px, py):
    """Translate a hand so its palm center lands at (px, py)."""
    return Hand(hand.landmarks + (np.array([px, py]) - hand.palm_center))


def thumbs_up_hand(px, py):
    h = make_hand((False, False, False, False, False))
    lm = h.landmarks.copy()
    lm[3] = (-0.5 * P, -1.1 * P)
    lm[4] = (-0.5 * P, -1.6 * P)
    return at_palm(Hand(lm), px, py)


FAR = (650, 250)  # near the face (within reach) but not touching it


@pytest.mark.parametrize("fingers,expected", [
    ((True, True, True, True, True), "open_palm"),
    ((False, True, True, False, False), "peace"),
    ((False, True, False, False, False), "pointing"),
    ((False, False, False, False, False), "fist"),
])
def test_single_hand_shapes(fingers, expected):
    h = at_palm(make_hand(fingers), *FAR)
    assert fingers_extended(h.landmarks) == list(fingers)
    assert classify([h], FACE) == expected


def test_thumbs_up_and_down():
    up = thumbs_up_hand(*FAR)
    assert classify([up], FACE) == "thumbs_up"
    lm = up.landmarks.copy()
    lm[:, 1] = 2 * FAR[1] - lm[:, 1]  # mirror vertically
    assert classify([Hand(lm)], FACE) == "thumbs_down"


def test_hand_on_forehead_vs_over_face():
    assert classify([at_palm(make_hand(), 300, 110)], FACE) == "hand_on_forehead"
    assert classify([at_palm(make_hand(), 300, 230)], FACE) == "hand_over_face"


def test_hands_on_cheeks():
    hs = [at_palm(make_hand(), 180, 210), at_palm(make_hand(), 420, 210)]
    assert classify(hs, FACE) == "hands_on_cheeks"


@pytest.mark.parametrize("y_offset", [0, 500])
def test_heart_hands(y_offset):
    a, b = make_hand(x=220, y=470), make_hand(x=380, y=470)
    a.landmarks[8], b.landmarks[8] = (295, 330), (305, 330)
    a.landmarks[4], b.landmarks[4] = (295, 420), (305, 420)
    a.landmarks[:, 1] += y_offset
    b.landmarks[:, 1] += y_offset
    assert classify([a, b], FACE) == "heart_hands"
    if y_offset:
        x, y, w, h = context_box(FACE, [a, b], reach=HEART_HAND_REACH)
        assert y + h >= b.bbox[1] + b.bbox[3]


def test_steepled_fingers():
    a, b = make_hand(x=220, y=480), make_hand(x=380, y=480)
    for tip, ty in ((8, 360), (12, 350), (16, 365)):
        a.landmarks[tip], b.landmarks[tip] = (295, ty), (305, ty)
    assert classify([a, b], FACE) == "steepled_fingers"


def test_no_face_or_no_hands():
    assert classify([], FACE) is None
    assert classify([make_hand()], None) is None
    # hand far away from the face is ignored
    assert classify([at_palm(make_hand(), 1500, 900)], FACE) is None


def test_smoother_needs_persistence():
    g = GestureSmoother(alpha=0.4, threshold=0.5)
    assert g.update("peace") is None  # one frame is not enough
    assert g.update("peace") == "peace"
    assert g.update(None) == "peace"  # one dropout frame does not cancel
    for _ in range(3):
        out = g.update(None)
    assert out is None


def test_context_box_includes_hand():
    h = at_palm(make_hand(), 300, 420)
    x, y, w, hgt = context_box(FACE, [h])
    assert y + hgt > 420 and x <= 200 and w >= 200


def test_gesture_offsets():
    class M:
        def __init__(self, g):
            self.gesture = g
    memes = [M(""), M("heart_hands"), M("peace")]
    matching = gesture_offsets(memes, "heart_hands", 0.1)
    assert matching[:2] == pytest.approx([0, 0.1])
    assert np.isneginf(matching[2])
    assert gesture_offsets(memes, "heart_hands", 0.1, -0.05)[2] == pytest.approx(-0.05)
    assert np.isneginf(gesture_offsets(memes, None, 0.1)[1:]).all()


def test_gesture_meme_is_gated_when_tracking_is_disabled():
    from types import SimpleNamespace
    from hamstermirror.pipeline import HamsterMirror

    class Detector:
        def largest(self, frame):
            return (10, 10, 20, 20)

    class Matcher:
        normalize_vector = False
        uses_context = False
        memes = [SimpleNamespace(gesture=""), SimpleNamespace(gesture="peace")]

        def vector(self, face):
            return np.zeros(1, np.float32)

        def scores(self, vector):
            return np.array([0.0, 10.0], np.float32)

    mirror = HamsterMirror(Detector(), Matcher(), smoothing=False, gestures=False)
    result = mirror.process(np.zeros((40, 40, 3), np.uint8))
    assert result.meme_idx == 0
    assert result.topk[0][0] == 0


def test_pipeline_gesture_switches_meme(tmp_path, clip_encoder):
    """With a fake hand tracker, making the tagged gesture should switch to that meme."""
    import cv2
    from skimage import data
    from conftest import make_memes
    from hamstermirror.clip_retrieval import ClipMatcher, MemeIndex
    from hamstermirror.face import FaceDetector
    from hamstermirror.memes import load_memes
    from hamstermirror.pipeline import HamsterMirror

    csv_path = make_memes(tmp_path / "m", n=4)
    lines = csv_path.read_text().splitlines()
    lines[0] += ",gesture"
    lines = [lines[0]] + [l + ("," + ("peace" if i == 2 else "")) for i, l in enumerate(lines[1:])]
    csv_path.write_text("\n".join(lines) + "\n")
    memes = load_memes(csv_path)
    assert memes[2].gesture == "peace"

    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    face = FaceDetector("haar").largest(img)
    fx, fy, fw, fh = face[:4]

    class FakeHands:
        available = True
        backend = "fake"
        hands = []

        def detect(self, frame):
            return self.hands

    fake = FakeHands()
    m = ClipMatcher(memes, clip_encoder, MemeIndex.build(clip_encoder, memes))
    hm = HamsterMirror(FaceDetector("haar"), m, hands=fake, gesture_bonus=1.0)
    for _ in range(5):
        r = hm.process(img)
    assert r.gesture is None and r.meme_idx != 2
    assert all(i != 2 for i, _ in r.topk)
    peace = make_hand((False, True, True, False, False))
    fake.hands = [at_palm(peace, fx + fw * 2.0, fy + fh * 0.8)]
    for _ in range(8):
        r = hm.process(img)
    assert r.gesture == "peace" and r.meme_idx == 2
    assert r.crop_box != r.box  # CLIP saw the wider face + hand crop

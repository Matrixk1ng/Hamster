"""Rule-based hand gesture classifier.

Works on MediaPipe hand landmarks plus the face box, so it can tell "hand on
forehead" from "hand over face" and "hands on cheeks". All distances are
measured relative to the face width, so rules hold at any distance from the camera.

Landmark ids: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
(MCP, PIP, DIP, TIP for each finger).
"""
import numpy as np

from . import config

TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]
HEART_HAND_REACH = 4.0


def _d(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def fingers_extended(lm):
    """[thumb, index, middle, ring, pinky] booleans, orientation independent."""
    wrist = lm[0]
    palm = _d(lm[0], lm[9])
    out = []
    # thumb: tip clearly farther from the index MCP than its IP joint is
    out.append(_d(lm[4], lm[5]) > 1.2 * _d(lm[3], lm[5]) and _d(lm[4], lm[5]) > 0.5 * palm)
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        out.append(_d(lm[tip], wrist) > 1.1 * _d(lm[pip], wrist))
    return out


def _near_face(hand, face_box, reach=2.5):
    fx, fy, fw, fh = face_box[:4]
    c = np.array([fx + fw / 2, fy + fh / 2])
    return _d(hand.palm_center, c) < reach * fw


def _two_hand(a, b, face_box):
    fx, fy, fw, fh = face_box[:4]
    s = fw
    A, B = a.landmarks, b.landmarks
    cx, cy = fx + fw / 2, fy + fh / 2
    # steepled: index, middle and ring fingertips of both hands touching
    if _d(A[8], B[8]) < 0.3 * s and _d(A[12], B[12]) < 0.3 * s and _d(A[16], B[16]) < 0.35 * s:
        return "steepled_fingers"
    # heart: index tips touch at the top, thumb tips touch at the bottom, gap between
    if _d(A[8], B[8]) < 0.35 * s and _d(A[4], B[4]) < 0.35 * s:
        if (A[8][1] + B[8][1]) / 2 < (A[4][1] + B[4][1]) / 2 - 0.15 * s:
            return "heart_hands"
    # hands on cheeks: one palm each side of the face, at face height
    pa, pb = a.palm_center, b.palm_center
    if (pa[0] - cx) * (pb[0] - cx) < 0:  # opposite sides
        ok = all(0.25 * s < abs(p[0] - cx) < 1.1 * s and abs(p[1] - cy) < 0.75 * fh
                 for p in (pa, pb))
        if ok:
            return "hands_on_cheeks"
    return None


def _on_face(hand, face_box):
    fx, fy, fw, fh = face_box[:4]
    px, py = hand.palm_center
    if not (fx - 0.2 * fw < px < fx + 1.2 * fw):
        return None
    if fy - 0.6 * fh < py < fy + 0.35 * fh:
        return "hand_on_forehead"
    if fy + 0.35 * fh <= py < fy + fh:
        return "hand_over_face"
    return None


def _shape(hand):
    lm = hand.landmarks
    t, i, m, r, p = fingers_extended(lm)
    palm = _d(lm[0], lm[9])
    if t and not (i or m or r or p):
        if lm[4][1] < lm[0][1] - 0.3 * palm and lm[4][1] < lm[5][1]:
            return "thumbs_up"
        if lm[4][1] > lm[0][1] + 0.3 * palm and lm[4][1] > lm[5][1]:
            return "thumbs_down"
    if i and m and not r and not p:
        return "peace"
    if i and not (m or r or p):
        return "pointing"
    if i and m and r and p:
        return "open_palm"
    if not (t or i or m or r or p):
        return "fist"
    return None


def classify(hands, face_box):
    """Return one gesture name from config.GESTURES, or None.
    Two-hand gestures first, then hand-on-face, then single-hand shapes."""
    if face_box is None or not hands:
        return None
    nearby = [h for h in hands if _near_face(h, face_box)]
    if len(nearby) >= 2:
        g = _two_hand(nearby[0], nearby[1], face_box)
        if g:
            return g
    heart_hands = [h for h in hands if _near_face(h, face_box, HEART_HAND_REACH)]
    if len(heart_hands) >= 2 and _two_hand(heart_hands[0], heart_hands[1], face_box) == "heart_hands":
        return "heart_hands"
    for h in nearby:
        g = _on_face(h, face_box)
        if g:
            return g
    for h in nearby:
        g = _shape(h)
        if g:
            return g
    return None


class GestureSmoother:
    """EMA over one-hot gesture votes with two thresholds: a gesture turns on
    once its score passes `threshold` and stays on until it drops below
    `release`, so it must persist a few frames to count and a one-frame
    tracking dropout does not cancel it."""

    def __init__(self, alpha=0.4, threshold=0.5, release=0.3):
        self.names = config.GESTURES
        self.alpha, self.threshold, self.release = alpha, threshold, release
        self.reset()

    def reset(self):
        self.state = np.zeros(len(self.names), np.float32)
        self.current = None

    def update(self, gesture):
        v = np.zeros_like(self.state)
        if gesture:
            v[self.names.index(gesture)] = 1
        self.state = self.alpha * v + (1 - self.alpha) * self.state
        k = int(np.argmax(self.state))
        if self.current is not None:
            c = self.names.index(self.current)
            challenger = k != c and self.state[k] >= self.threshold and self.state[k] > self.state[c]
            if self.state[c] >= self.release and not challenger:
                return self.current
        self.current = self.names[k] if self.state[k] >= self.threshold else None
        return self.current


def context_box(face_box, hands, reach=2.5):
    """Union of the face box and every hand near the face, so CLIP sees the gesture."""
    fx, fy, fw, fh = face_box[:4]
    x0, y0, x1, y1 = fx, fy, fx + fw, fy + fh
    for h in hands:
        if _near_face(h, face_box, reach):
            hx, hy, hw, hh = h.bbox
            x0, y0 = min(x0, hx), min(y0, hy)
            x1, y1 = max(x1, hx + hw), max(y1, hy + hh)
    return (x0, y0, x1 - x0, y1 - y0)


def gesture_offsets(memes, gesture, bonus, unmatched_bonus=None):
    """Boost a matching gesture meme and exclude memes requiring another gesture."""
    v = np.zeros(len(memes), np.float32)
    unmatched_bonus = -np.inf if unmatched_bonus is None else unmatched_bonus
    for i, m in enumerate(memes):
        if m.gesture:
            v[i] = bonus if m.gesture == gesture else unmatched_bonus
    return v

"""Expression matcher based on MediaPipe Face Landmarker blendshapes.

The Face Landmarker outputs 52 ARKit-style blendshape scores per face
(mouthSmileLeft, browDownRight, jawOpen, ...). We compress them into a
small set of interpretable features (smile, smile asymmetry, brow down,
teeth showing, ...). Each meme has a target profile on those features,
written by hand in memes/expressions.csv or learned per person by
scripts/enroll.py. The closest profile wins.

Why not CLIP alone: CLIP compares a human photo with a cartoon drawing as a
whole and does not measure mouth corners or eyebrows, so a few memes act as
"hubs" that win for almost any face.
"""
import csv
from pathlib import Path

import numpy as np

from . import config

FEATURES = ["smile", "smile_asym", "teeth", "jaw_open", "stretch", "frown", "pucker",
            "press", "sneer", "cheek_puff", "brow_down", "brow_up", "brow_asym",
            "eye_wide", "eye_squint", "eye_closed", "side_eye"]

# Typical maximum raw value of each feature, so 1.0 means "about as strong as
# people usually make it". Enrollment makes this per person.
SCALE = {"smile": 0.8, "smile_asym": 0.35, "teeth": 0.6, "jaw_open": 0.6, "stretch": 0.5,
         "frown": 0.5, "pucker": 0.8, "press": 0.6, "sneer": 0.5, "cheek_puff": 0.5,
         "brow_down": 0.6, "brow_up": 0.7, "brow_asym": 0.45, "eye_wide": 0.5,
         "eye_squint": 0.7, "eye_closed": 0.8, "side_eye": 0.6}

# Asymmetry features are what separate a smirk from a smile and a raised
# eyebrow from surprise, so they count a bit more.
WEIGHTS = np.array([1.5 if f in ("smile_asym", "brow_asym") else 1.0 for f in FEATURES],
                   np.float32)


def features_from_blendshapes(b):
    """b: dict blendshape name -> score. Returns normalized feature vector (F,)."""
    g = lambda k: float(b.get(k, 0.0))
    avg = lambda a, c: (g(a) + g(c)) / 2
    raw = {
        "smile": avg("mouthSmileLeft", "mouthSmileRight"),
        # dead zones: every face is a little lopsided, only clear asymmetry counts
        "smile_asym": max(0.0, abs(g("mouthSmileLeft") - g("mouthSmileRight"))
                          + 0.5 * abs(g("mouthLeft") - g("mouthRight")) - 0.08),
        "teeth": (g("mouthUpperUpLeft") + g("mouthUpperUpRight")
                  + g("mouthLowerDownLeft") + g("mouthLowerDownRight")) / 4,
        "jaw_open": g("jawOpen"),
        "stretch": avg("mouthStretchLeft", "mouthStretchRight"),
        "frown": avg("mouthFrownLeft", "mouthFrownRight"),
        "pucker": max(g("mouthPucker"), g("mouthFunnel")),
        "press": max(avg("mouthPressLeft", "mouthPressRight"),
                     avg("mouthRollLower", "mouthRollUpper")),
        "sneer": avg("noseSneerLeft", "noseSneerRight"),
        "cheek_puff": g("cheekPuff"),
        "brow_down": avg("browDownLeft", "browDownRight"),
        "brow_up": max(g("browInnerUp"), avg("browOuterUpLeft", "browOuterUpRight")),
        "brow_asym": max(0.0, abs(g("browOuterUpLeft") - g("browOuterUpRight"))
                         + abs(g("browDownLeft") - g("browDownRight")) - 0.12),
        "eye_wide": avg("eyeWideLeft", "eyeWideRight"),
        "eye_squint": max(avg("eyeSquintLeft", "eyeSquintRight"),
                          avg("cheekSquintLeft", "cheekSquintRight")),
        "eye_closed": avg("eyeBlinkLeft", "eyeBlinkRight"),
        # both eyes looking the same way sideways
        "side_eye": abs(avg("eyeLookOutLeft", "eyeLookInRight")
                        - avg("eyeLookInLeft", "eyeLookOutRight")),
    }
    return np.array([min(raw[f] / SCALE[f], 1.5) for f in FEATURES], np.float32)


class BlendshapeExtractor:
    """MediaPipe Face Landmarker with blendshapes (Tasks API only)."""

    def __init__(self, model_path=config.MP_FACE_LANDMARKER):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} (run scripts/download_models.py)")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=True)
        self._mp = mp
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def blendshapes(self, rgb):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect(img)
        if not res.face_blendshapes:
            return None
        return {c.category_name: c.score for c in res.face_blendshapes[0]}

    def features(self, rgb):
        b = self.blendshapes(rgb)
        return None if b is None else features_from_blendshapes(b)


# ---------------------------------------------------------------- profiles
def load_profiles(memes, csv_path=config.EXPR_CSV):
    """Hand-written profiles. Returns (M, F) array, NaN rows for memes with no profile."""
    P = np.full((len(memes), len(FEATURES)), np.nan, np.float32)
    if not Path(csv_path).exists():
        return P
    idx = {m.file: i for i, m in enumerate(memes)}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in f if r.strip() and not r.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        i = idx.get(row["file"].strip())
        if i is None:
            continue
        P[i] = [float(row.get(k) or 0) for k in FEATURES]
    return P


class Calibration:
    """Per-person data from scripts/enroll.py: resting face + enrolled meme prototypes."""

    def __init__(self, neutral=None, protos=None):
        self.neutral = neutral if neutral is not None else np.zeros(len(FEATURES), np.float32)
        self.protos = protos or {}  # meme file -> (F,)

    def save(self, path=config.EXPR_CALIB):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        files = list(self.protos)
        np.savez(path, neutral=self.neutral, files=np.array(files, dtype=str),
                 protos=np.array([self.protos[f] for f in files], np.float32).reshape(
                     len(files), len(FEATURES)), features=np.array(FEATURES))

    @classmethod
    def load(cls, path=config.EXPR_CALIB):
        if not Path(path).exists():
            return cls()
        d = np.load(path, allow_pickle=False)
        if list(d["features"]) != FEATURES:
            print(f"[expr] {path} was made with a different feature set, ignoring it")
            return cls()
        return cls(d["neutral"], {f: p for f, p in zip(d["files"].tolist(), d["protos"])})

    def apply(self, feat):
        return np.clip(feat - self.neutral, 0, None)


class ExprMatcher:
    name = "expr"
    normalize_vector = False
    uses_context = False   # needs the tight face crop
    margin = 0.03          # hysteresis margin, in feature-distance units
    gesture_bonus = 0.25

    def __init__(self, memes, extractor, calib=None, profiles=None, verbose=True):
        self.memes, self.extractor = memes, extractor
        self.calib = calib or Calibration()
        P = load_profiles(memes) if profiles is None else profiles.copy()
        for i, m in enumerate(memes):  # enrolled prototypes override hand-written ones
            if m.file in self.calib.protos:
                P[i] = self.calib.protos[m.file]
        self.valid = ~np.isnan(P).any(1)
        self.P = np.nan_to_num(P)
        self._last = np.zeros(len(FEATURES), np.float32)
        if verbose:
            missing = [m.file for m, v in zip(memes, self.valid) if not v]
            n_en = sum(m.file in self.calib.protos for m in memes)
            print(f"[expr] {self.valid.sum()}/{len(memes)} memes have a profile "
                  f"({n_en} enrolled)" + (f", no profile: {missing[:5]}" if missing else ""))

    def vector(self, face_rgb):
        f = self.extractor.features(face_rgb)
        if f is not None:
            self._last = self.calib.apply(f)
        return self._last

    def vectors(self, faces):
        return np.stack([self.vector(f) for f in faces])

    def scores(self, v):
        d = np.sqrt((((self.P - np.asarray(v)) ** 2) * WEIGHTS).sum(1))
        return np.where(self.valid, -d, -1e3).astype(np.float32)


def _z(s, valid):
    mu, sd = s[valid].mean(), s[valid].std() + 1e-6
    return (s - mu) / sd


class ExprClipMatcher:
    """Hybrid: z-scored expression score + clip_weight * z-scored CLIP score."""

    name = "expr+clip"
    normalize_vector = False
    uses_context = False
    margin = 0.15
    gesture_bonus = 1.0

    def __init__(self, expr, clip, clip_weight=0.3):
        self.expr, self.clip, self.w = expr, clip, clip_weight
        self.memes = expr.memes
        self.nf = len(FEATURES)

    def vector(self, face_rgb):
        return np.concatenate([self.expr.vector(face_rgb), self.clip.vector(face_rgb)])

    def vectors(self, faces):
        return np.stack([self.vector(f) for f in faces])

    def scores(self, v):
        v = np.asarray(v)
        se, sc = self.expr.scores(v[: self.nf]), self.clip.scores(v[self.nf:])
        ok = self.expr.valid
        return np.where(ok, _z(se, ok) + self.w * _z(sc, ok), -1e3).astype(np.float32)

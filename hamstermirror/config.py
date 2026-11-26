"""Shared paths and constants."""
import os
from pathlib import Path

# HM_ROOT lets tests point all data/checkpoint paths at a scratch folder.
PKG_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("HM_ROOT", PKG_ROOT))
MEME_DIR = ROOT / "memes"
MEME_CSV = MEME_DIR / "memes.csv"
MODEL_DIR = PKG_ROOT / "models"  # weights are not per-experiment data
CKPT_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "data"
CLIP_DIR = DATA_DIR / "clips"
FRAME_DIR = DATA_DIR / "frames"
CLIP_META_CSV = DATA_DIR / "clips.csv"
ANNOT_CSV = DATA_DIR / "annotations.csv"
RESULTS_DIR = ROOT / "results"

MEME_INDEX = CKPT_DIR / "meme_index.npz"
BASELINE_CKPT = CKPT_DIR / "baseline.pt"
PROJECTION_CKPT = CKPT_DIR / "projection.pt"
MP_FACE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
MP_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

MP_HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
MP_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
FRAME_META_CSV = DATA_DIR / "frame_meta.csv"
MP_FACE_LANDMARKER = MODEL_DIR / "face_landmarker.task"
MP_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
EXPR_CSV = MEME_DIR / "expressions.csv"      # hand-written expression profile per meme
EXPR_CALIB = CKPT_DIR / "expr_calib.npz"     # per-person neutral face + enrolled profiles

# FER2013 label order (matches the Kaggle CSV "emotion" column).
FER_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Hand gestures the rule-based classifier can output (see gestures.py).
GESTURES = ["heart_hands", "steepled_fingers", "hands_on_cheeks", "hand_on_forehead",
            "hand_over_face", "thumbs_up", "thumbs_down", "peace", "pointing",
            "open_palm", "fist"]

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
# Template that turns a meme's free-text description into a CLIP text prompt.
# Text bridges the human-face / cartoon-hamster domain gap.
TEXT_TEMPLATE = "a photo of a face that looks {}"


def best_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

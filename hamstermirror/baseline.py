"""Baseline: CNN expression classifier (FER2013 classes) + emotion -> meme lookup."""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from . import config

INPUT_SIZE = 112


def build_model(num_classes=len(config.FER_CLASSES), imagenet=True):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
    try:
        model = torchvision.models.resnet18(weights=weights)
    except Exception as e:  # offline: fall back to random init
        print(f"[baseline] could not load ImageNet weights ({e}), using random init")
        model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def eval_transform():
    T = torchvision.transforms
    return T.Compose([
        T.Grayscale(num_output_channels=3),  # FER2013 is grayscale, keep webcam consistent
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def train_transform():
    T = torchvision.transforms
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.3, 0.3),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


class BaselineClassifier:
    def __init__(self, ckpt=config.BASELINE_CKPT, device=None, model=None):
        self.device = device or config.best_device()
        if model is None:
            model = build_model(imagenet=False)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state["model"] if "model" in state else state)
        self.model = model.to(self.device).eval()
        self.tf = eval_transform()

    @torch.no_grad()
    def predict_batch(self, faces_rgb):
        x = torch.stack([self.tf(Image.fromarray(f)) for f in faces_rgb]).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()

    def predict(self, face_rgb):
        return self.predict_batch([face_rgb])[0]


class BaselineMatcher:
    """Scores each meme by the predicted probability of its emotion tag."""

    name = "baseline"
    normalize_vector = False  # probabilities, not unit embeddings
    uses_context = False  # FER was trained on tight face crops

    def __init__(self, memes, classifier):
        self.memes = memes
        self.clf = classifier
        cls_idx = {c: i for i, c in enumerate(config.FER_CLASSES)}
        self.meme_cls = np.array([cls_idx[m.emotion] for m in memes])
        # tiny tie-break so the first listed meme of an emotion wins
        self.tiebreak = -1e-6 * np.arange(len(memes))

    def vector(self, face_rgb):
        return self.clf.predict(face_rgb)

    def vectors(self, faces_rgb):
        return self.clf.predict_batch(faces_rgb)

    def scores(self, probs):
        return np.asarray(probs)[self.meme_cls] + self.tiebreak

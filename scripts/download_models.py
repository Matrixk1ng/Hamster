"""Fetch model weights once so the demo runs offline afterwards:
MediaPipe BlazeFace + Hand Landmarker + Face Landmarker, CLIP ViT-B/32 (OpenAI weights via open_clip), ResNet18 ImageNet."""
import shutil
import ssl
import urllib.request
import os

import _bootstrap  # noqa: F401
from hamstermirror import config


def download(url, destination):
    context = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def main():
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    config.MODEL_DIR.mkdir(exist_ok=True)
    if not config.MP_FACE_MODEL.exists():
        print(f"downloading {config.MP_FACE_MODEL.name} ...")
        download(config.MP_FACE_MODEL_URL, config.MP_FACE_MODEL)
    print("MediaPipe face model ok")
    if not config.MP_HAND_MODEL.exists():
        print(f"downloading {config.MP_HAND_MODEL.name} ...")
        download(config.MP_HAND_MODEL_URL, config.MP_HAND_MODEL)
    print("MediaPipe hand model ok")
    if not config.MP_FACE_LANDMARKER.exists():
        print(f"downloading {config.MP_FACE_LANDMARKER.name} ...")
        download(config.MP_FACE_LANDMARKER_URL, config.MP_FACE_LANDMARKER)
    print("MediaPipe face landmarker (blendshapes) ok")

    import open_clip

    open_clip.create_model_and_transforms(config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
    print(f"CLIP {config.CLIP_MODEL}/{config.CLIP_PRETRAINED} ok")

    import torchvision

    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    print("ResNet18 ImageNet weights ok")


if __name__ == "__main__":
    main()

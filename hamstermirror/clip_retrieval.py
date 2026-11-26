"""Proposed method: CLIP embedding retrieval of the nearest meme.

Each meme is represented by a blend of its CLIP image embedding and the CLIP
text embedding of its description. Faces are embedded with the image tower
(optionally passed through a fine-tuned projection head), then matched by
cosine similarity.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config


def _unit(x, axis=-1):
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8, None)


class ClipEncoder:
    def __init__(self, model_name=config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, device=None):
        import open_clip

        self.device = device or config.best_device()
        if pretrained in (None, "none", ""):
            pretrained = None  # random weights, for offline tests only
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name, self.pretrained = model_name, str(pretrained)

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        """images: list of PIL images or RGB uint8 arrays. Returns unit vectors (N, D)."""
        out = []
        for i in range(0, len(images), batch_size):
            chunk = [Image.fromarray(im) if isinstance(im, np.ndarray) else im
                     for im in images[i:i + batch_size]]
            x = torch.stack([self.preprocess(im) for im in chunk]).to(self.device)
            out.append(self.model.encode_image(x).float().cpu().numpy())
        return _unit(np.concatenate(out))

    @torch.no_grad()
    def encode_texts(self, texts):
        tok = self.tokenizer(texts).to(self.device)
        return _unit(self.model.encode_text(tok).float().cpu().numpy())


class MemeIndex:
    def __init__(self, files, image_emb, text_emb, model_name, pretrained):
        self.files = list(files)
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.model_name, self.pretrained = model_name, pretrained

    @classmethod
    def build(cls, encoder, memes, template=config.TEXT_TEMPLATE):
        img = encoder.encode_images([m.load() for m in memes])
        txt = encoder.encode_texts([template.format(m.description) for m in memes])
        return cls([m.file for m in memes], img, txt, encoder.model_name, encoder.pretrained)

    def save(self, path=config.MEME_INDEX):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, files=np.array(self.files), image_emb=self.image_emb,
                 text_emb=self.text_emb, model_name=self.model_name, pretrained=self.pretrained)

    @classmethod
    def load(cls, path=config.MEME_INDEX):
        d = np.load(path, allow_pickle=False)
        return cls(d["files"].tolist(), d["image_emb"], d["text_emb"],
                   str(d["model_name"]), str(d["pretrained"]))

    def embeddings(self, text_weight=0.6):
        """Blend image and text embeddings. text_weight=0 is pure image-to-image."""
        return _unit((1 - text_weight) * self.image_emb + text_weight * self.text_emb)


class ClipMatcher:
    name = "clip"
    normalize_vector = True
    uses_context = True  # CLIP sees hands in the crop

    def __init__(self, memes, encoder, index, text_weight=0.6, projection=None):
        files = [m.file for m in memes]
        if files != index.files:
            raise ValueError("Meme index is stale. Re-run scripts/build_index.py")
        if index.model_name != encoder.model_name:
            raise ValueError("Meme index was built with a different CLIP model")
        self.memes, self.encoder, self.index = memes, encoder, index
        self.projection = projection
        self.set_text_weight(text_weight)
        if projection is not None:
            self.name = "clip+proj"

    def set_text_weight(self, w):
        self.text_weight = w
        self.meme_emb = self.index.embeddings(w)

    def _project(self, emb):
        return self.projection.apply(emb) if self.projection is not None else emb

    def vector(self, face_rgb):
        return self.vectors([face_rgb])[0]

    def vectors(self, faces_rgb):
        return self._project(self.encoder.encode_images(faces_rgb))

    def scores(self, emb):
        return self.meme_emb @ _unit(np.asarray(emb, dtype=np.float32))

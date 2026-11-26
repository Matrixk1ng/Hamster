"""Optional lightweight projection head fine-tuned on (face, meme) pairs.

f(x) = normalize(x + W x), W zero-initialised, so an untrained head is the
identity and fine-tuning can only nudge CLIP's face embedding toward the
memes people actually picked for it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return F.normalize(x + self.lin(x), dim=-1)

    @torch.no_grad()
    def apply(self, emb):
        x = torch.as_tensor(np.asarray(emb, dtype=np.float32))
        return self.eval()(x).numpy()

    def save(self, path, **meta):
        torch.save({"state": self.state_dict(), "dim": self.lin.in_features, **meta}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")
        head = cls(d["dim"])
        head.load_state_dict(d["state"])
        return head.eval()


def train_projection(face_emb, meme_idx, meme_emb, epochs=200, lr=1e-3,
                     weight_decay=1e-3, temperature=0.05, seed=0, verbose=True):
    """face_emb (N,D) unit CLIP face embeddings, meme_idx (N,) ground-truth meme,
    meme_emb (M,D) fixed meme embeddings. Cross-entropy over cosine/temperature."""
    torch.manual_seed(seed)
    X = torch.as_tensor(face_emb, dtype=torch.float32)
    y = torch.as_tensor(meme_idx, dtype=torch.long)
    M = torch.as_tensor(meme_emb, dtype=torch.float32)
    head = ProjectionHead(X.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        head.train()
        logits = head(X) @ M.T / temperature
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}  train top1 {acc:.3f}")
    return head.eval()

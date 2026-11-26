"""Temporal smoothing: EMA over vectors plus a hysteresis switch rule."""
import numpy as np


class EMA:
    """Exponential moving average. alpha = weight of the newest frame."""

    def __init__(self, alpha=0.3, normalize=True):
        self.alpha = alpha
        self.normalize = normalize
        self.state = None

    def reset(self):
        self.state = None

    def update(self, vec):
        vec = np.asarray(vec, dtype=np.float32)
        if self.state is None:
            self.state = vec.copy()
        else:
            self.state = self.alpha * vec + (1 - self.alpha) * self.state
        if self.normalize:
            n = np.linalg.norm(self.state)
            if n > 0:
                self.state = self.state / n
        return self.state


class Hysteresis:
    """Only switch away from the current meme when a challenger beats it
    by `margin` for `min_frames` consecutive frames."""

    def __init__(self, margin=0.02, min_frames=3):
        self.margin = margin
        self.min_frames = min_frames
        self.reset()

    def reset(self):
        self.current = None
        self._challenger = None
        self._count = 0

    def update(self, scores):
        scores = np.asarray(scores)
        best = int(np.argmax(scores))
        if self.current is None:
            self.current = best
            return best
        if best != self.current and scores[best] - scores[self.current] > self.margin:
            if best == self._challenger:
                self._count += 1
            else:
                self._challenger, self._count = best, 1
            if self._count >= self.min_frames:
                self.current, self._challenger, self._count = best, None, 0
        else:
            self._challenger, self._count = None, 0
        return self.current


class Smoother:
    """EMA on the matcher's vector, hysteresis on the meme scores."""

    def __init__(self, enabled=True, alpha=0.3, margin=0.02, min_frames=3, normalize=True):
        self.enabled = enabled
        self.ema = EMA(alpha, normalize)
        self.hyst = Hysteresis(margin, min_frames)

    def reset(self):
        self.ema.reset()
        self.hyst.reset()

    def smooth_vector(self, vec):
        return self.ema.update(vec) if self.enabled else vec

    def choose(self, scores):
        return self.hyst.update(scores) if self.enabled else int(np.argmax(scores))

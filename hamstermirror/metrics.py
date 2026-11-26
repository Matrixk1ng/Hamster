"""Small metric and reporting helpers."""
from collections import defaultdict

import numpy as np


def topk_hit(scores, gt_idx, k):
    top = np.argsort(-np.asarray(scores))[:k]
    return float(any(int(t) in gt_idx for t in top))


def count_switches(choices):
    return int(sum(a != b for a, b in zip(choices, choices[1:])))


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def group_mean(rows, key, value):
    g = defaultdict(list)
    for r in rows:
        g[r.get(key) or "unknown"].append(r[value])
    return {k: (mean(v), len(v)) for k, v in sorted(g.items())}


def md_table(header, rows):
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)

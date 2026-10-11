"""Within-stratum selection, reweighting, and a streaming reservoir."""
import numpy as np


def _group_rank(labels: np.ndarray, sort_key: np.ndarray):
    """Return (order, rank_in_group) with tokens sorted by (label, sort_key)."""
    order = np.lexsort((sort_key, labels))
    sl = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sl)) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(sl)]))
    return order, np.arange(len(sl)) - group_start


def select_within(labels: np.ndarray, n: np.ndarray, how: str, positions: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Pick n[h] tokens from each stratum h. Returns sorted token indices (into labels)."""
    labels = np.asarray(labels)
    if how == "random":
        key = rng.random(len(labels))
    elif how == "oldest_first_evict":  # SimBin: keep the newest
        key = -positions.astype(np.float64)
    elif how == "systematic":
        key = positions.astype(np.float64)
    else:
        raise ValueError(how)
    order, rank = _group_rank(labels, key)
    lab = labels[order]
    nh = n[lab]
    if how == "systematic":
        Nh = np.bincount(labels, minlength=len(n))[lab]
        step = Nh / np.maximum(nh, 1)
        u = rng.random(len(n))[lab] * step  # random start per stratum, in [0, step)
        j = np.ceil((rank - u) / step)
        keep = (nh > 0) & (j >= 0) & (j < nh) & (j * step + u < rank + 1)
    else:
        keep = rank < nh
    return np.sort(order[keep])


def reweight_bias(labels: np.ndarray, n: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Logit bias log(N_h / n_h) for the selected tokens."""
    N = np.bincount(labels, minlength=len(n))
    lab = labels[idx]
    return np.log(N[lab] / n[lab])


class Reservoir:
    """Algorithm R: uniform sample of size k from a stream."""

    def __init__(self, k: int, rng: np.random.Generator):
        self.k, self.rng, self.seen, self.items = k, rng, 0, []

    def add(self, x):
        if len(self.items) < self.k:
            self.items.append(x)
        else:
            j = self.rng.integers(0, self.seen + 1)
            if j < self.k:
                self.items[j] = x
        self.seen += 1

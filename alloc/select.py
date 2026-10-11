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


def inclusion_probs(log_w: np.ndarray, B: int) -> np.ndarray:
    """pi_i = min(1, exp(log_w_i + log_c)) with sum(pi) = B; log_c by bisection in log space
    (weights can span far more than float64's exp range)."""
    log_w = np.asarray(log_w, dtype=np.float64)
    f = lambda lc: np.exp(np.minimum(log_w + lc, 0.0)).sum()
    lo, hi = -log_w.max() - 50.0, -log_w.min() + 50.0      # f(lo) ~ 0 < B <= len = f(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < B:
            lo = mid
        else:
            hi = mid
    pi = np.exp(np.minimum(log_w + hi, 0.0))
    capped = pi >= 1.0
    free = B - capped.sum()                                 # rescale the uncapped part to remove residue
    pi[~capped] *= free / pi[~capped].sum()
    return pi


def systematic_pps(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fixed-size sampling with exact inclusion probabilities pi (sum(pi) integer), random order."""
    perm = rng.permutation(len(pi))
    cum = np.cumsum(pi[perm])
    B = int(round(cum[-1]))
    pts = rng.random() + np.arange(B)
    pos = np.searchsorted(cum, pts, side="right")
    return np.sort(perm[np.minimum(pos, len(pi) - 1)])


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

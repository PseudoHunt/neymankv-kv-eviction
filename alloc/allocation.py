"""Budget allocation across strata: proportional, Neyman, water-filling, Webster rounding."""
import numpy as np


def waterfill(a: np.ndarray, N: np.ndarray, B: float, floor: float = 1.0) -> np.ndarray:
    """Continuous shares n_h = clip(lam * a_h, lo_h, N_h) with sum = B.

    lo_h = min(floor, N_h). If the floors alone exceed B they are returned as-is
    (webster() then resolves the infeasibility)."""
    a = np.asarray(a, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    lo = np.minimum(floor, N)
    if B >= N.sum():
        return N.copy()
    if lo.sum() >= B:
        return lo.copy()
    if a.sum() <= 0:
        a = N.copy()  # no signal: fall back to proportional
    f = lambda lam: np.clip(lam * a, lo, N).sum()
    lo_l, hi_l = 0.0, 1.0
    while f(hi_l) < B:
        hi_l *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo_l + hi_l)
        if f(mid) < B:
            lo_l = mid
        else:
            hi_l = mid
    return np.clip(hi_l * a, lo, N)


def webster(shares: np.ndarray, N: np.ndarray, B: int, floor: int = 1) -> np.ndarray:
    """Webster / Sainte-Lague rounding with bounds lo_h <= n_h <= N_h and sum(n) == B.

    Divisor method: n_h(D) = clip(round_half_up(shares_h / D), lo_h, N_h), D chosen by
    bisection; remaining ties go to the largest priority shares_h / (n_h + 0.5)."""
    shares = np.asarray(shares, dtype=np.float64)
    N = np.asarray(N, dtype=np.int64)
    B = int(B)
    if B >= N.sum():
        return N.copy()
    lo = np.minimum(floor, N)
    if lo.sum() > B:
        # Infeasible floors: one slot to the B strata with the largest share.
        n = np.zeros_like(N)
        top = np.lexsort((-N, -shares))[:B]
        n[top] = 1
        return n
    f = lambda D: np.clip(np.floor(shares / D + 0.5), lo, N).astype(np.int64)
    pos = shares[shares > 0]
    if len(pos) == 0:
        shares = N.astype(np.float64)
        pos = shares
    # Bracket: large D -> floors (sum <= B); small D -> caps (sum >= B).
    d_hi, d_lo = pos.max() * 4 + 1.0, pos.min() / (N.max() + 1.0) / 4
    for _ in range(200):
        mid = np.sqrt(d_lo * d_hi)
        if f(mid).sum() > B:
            d_lo = mid
        else:
            d_hi = mid
    n = f(d_hi)
    r = B - n.sum()
    if r > 0:
        pri = np.where(n < N, shares / (n + 0.5), -np.inf)
        n[np.argsort(-pri, kind="stable")[:r]] += 1
    elif r < 0:  # cannot happen with the bracket above, kept as a guard
        pri = np.where(n > lo, shares / (n - 0.5 + 1e-12), np.inf)
        n[np.argsort(pri, kind="stable")[:-r]] -= 1
    assert n.sum() == B and np.all(n <= N) and np.all(n >= 0)
    return n


def proportional(N: np.ndarray, B: int) -> np.ndarray:
    """SimBin rule: share_h = B * N_h / N, then Webster rounding."""
    N = np.asarray(N)
    return webster(B * N / N.sum(), N, B)


def neyman(N: np.ndarray, S: np.ndarray, B: int) -> np.ndarray:
    """share_h ∝ N_h * S_h, floor 1 and cap N_h via water-filling, then Webster rounding."""
    N = np.asarray(N)
    shares = waterfill(N * np.asarray(S, dtype=np.float64), N, B)
    return webster(shares, N, B)


def cv_and_gain(N: np.ndarray, S: np.ndarray):
    """Size-weighted CV of S_h and the predicted Neyman gain CV^2/(1+CV^2) (no fpc)."""
    W = np.asarray(N, dtype=np.float64)
    W = W / W.sum()
    S = np.asarray(S, dtype=np.float64)
    m = (W * S).sum()
    if m <= 0:
        return 0.0, 0.0
    cv2 = (W * (S - m) ** 2).sum() / m ** 2
    return float(np.sqrt(cv2)), float(cv2 / (1 + cv2))

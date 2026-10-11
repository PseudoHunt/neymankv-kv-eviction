"""Row-batched (torch) allocation and per-stratum selection, for decode-time compression.

Rows are independent (layer, head) pairs; strata are columns. Matches alloc.allocation / alloc.select."""
import torch


def waterfill(a: torch.Tensor, caps: torch.Tensor, B: torch.Tensor, floor: int = 1, iters: int = 60) -> torch.Tensor:
    """Continuous n = clip(lam * a, lo, caps) with row sums B. a > 0 wherever caps > 0."""
    a, caps = a.double(), caps.double()
    B = B.double().view(-1, 1)
    lo = torch.minimum(torch.full_like(caps, float(floor)), caps)
    ratio = torch.where(caps > 0, caps / a.clamp_min(1e-300), torch.zeros_like(caps))
    hi = ratio.max(-1, keepdim=True).values.clamp_min(1e-300)
    lo_l = torch.zeros_like(hi)
    for _ in range(iters):
        mid = 0.5 * (lo_l + hi)
        under = torch.clip(mid * a, lo, caps).sum(-1, keepdim=True) < B
        lo_l, hi = torch.where(under, mid, lo_l), torch.where(under, hi, mid)
    out = torch.clip(hi * a, lo, caps)
    full = caps.sum(-1, keepdim=True) <= B
    return torch.where(full, caps, out)


def webster(shares: torch.Tensor, caps: torch.Tensor, B: torch.Tensor, floor: int = 1, iters: int = 80) -> torch.Tensor:
    """Integer n with lo <= n <= caps and row sums == min(B, caps.sum()); Webster divisor method.
    If the floors alone exceed B, one slot goes to the B strata with the largest share."""
    shares, capd = shares.double(), caps.double()
    R, H = shares.shape
    Bv = torch.minimum(B.view(-1).long(), caps.sum(-1).long())
    lo = torch.minimum(torch.full_like(capd, float(floor)), capd)
    pos = torch.where(shares > 0, shares, torch.full_like(shares, float("inf")))
    d_hi = shares.max(-1, keepdim=True).values * 4 + 1.0
    d_lo = (pos.min(-1, keepdim=True).values / (capd.max(-1, keepdim=True).values + 1) / 4).clamp_min(1e-300)
    f = lambda D: torch.clip(torch.floor(shares / D + 0.5), lo, capd)
    for _ in range(iters):
        mid = torch.sqrt(d_lo * d_hi)
        over = f(mid).sum(-1) > Bv
        d_lo = torch.where(over[:, None], mid, d_lo)
        d_hi = torch.where(over[:, None], d_hi, mid)
    n = f(d_hi)
    r = (Bv - n.sum(-1).long()).clamp_min(0)
    pri = torch.where(n < capd, shares / (n + 0.5), torch.full_like(n, -float("inf")))
    rank = torch.argsort(torch.argsort(-pri, dim=-1, stable=True), dim=-1)
    n = n + (rank < r[:, None]).double()
    # infeasible floors
    infeas = lo.sum(-1) > Bv
    if infeas.any():
        srank = torch.argsort(torch.argsort(-torch.where(capd > 0, shares, -torch.ones_like(shares)), dim=-1, stable=True), dim=-1)
        n1 = ((srank < Bv[:, None]) & (capd > 0)).double()
        n = torch.where(infeas[:, None], n1, n)
    n = n.long()
    assert torch.all(n.sum(-1) == Bv) and torch.all(n <= caps) and torch.all(n >= 0)
    return n


def select_bottom(labels: torch.Tensor, order_key: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """labels [R, M] in [0, H) (or -1 = not a candidate); keep the n[r, h] entries of each stratum with the
    smallest order_key. Returns a bool mask [R, M]."""
    R, M = labels.shape
    H = n.shape[1]
    lab = torch.where(labels < 0, torch.full_like(labels, H), labels)
    n_ext = torch.cat([n, torch.zeros(R, 1, dtype=n.dtype, device=n.device)], 1)
    o1 = torch.argsort(order_key, dim=-1, stable=True)
    lab1 = lab.gather(1, o1)
    o2 = torch.argsort(lab1, dim=-1, stable=True)
    order = o1.gather(1, o2)
    sl = lab.gather(1, order)
    counts = torch.zeros(R, H + 1, dtype=torch.long, device=lab.device).scatter_add_(1, lab, torch.ones_like(lab))
    start = torch.cumsum(counts, 1) - counts
    rank = torch.arange(M, device=lab.device)[None] - start.gather(1, sl)
    keep_sorted = rank < n_ext.gather(1, sl)
    mask = torch.zeros(R, M, dtype=torch.bool, device=lab.device)
    mask.scatter_(1, order, keep_sorted)
    return mask

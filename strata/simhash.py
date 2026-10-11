"""SimHash codes and strata construction (per KV head)."""
import torch

POS_BINS = 4
NORM_BANDS = 3


def projection(d: int, bits: int, seed: int = 1234, dtype=torch.float32) -> torch.Tensor:
    """Fixed-seed Gaussian projection shared across heads and layers."""
    g = torch.Generator().manual_seed(seed + bits)
    return torch.randn(d, bits, generator=g, dtype=torch.float32).to(dtype)


def simhash(keys: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """keys [N, d] -> integer codes [N] from sign bits of the L2-normalized key."""
    k = keys.float()
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    bits = (k @ proj.float().to(k.device)) > 0
    pw = (2 ** torch.arange(bits.shape[-1], device=k.device)).long()
    return (bits.long() * pw).sum(-1)


def equal_count_bins(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Assign each element to one of n_bins equal-count bins by rank."""
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(x), device=x.device)
    return (ranks * n_bins) // len(x)


def build_strata(variant: str, bits: int, k_post: torch.Tensor, k_pre: torch.Tensor,
                 positions: torch.Tensor) -> torch.Tensor:
    """Return dense stratum labels 0..H-1 for each token. Empty strata never appear."""
    if variant == "post":
        raw = simhash(k_post, projection(k_post.shape[-1], bits))
    elif variant == "pre":
        raw = simhash(k_pre, projection(k_pre.shape[-1], bits))
    elif variant == "pre_pos_norm":
        h = simhash(k_pre, projection(k_pre.shape[-1], bits))
        pb = equal_count_bins(positions.to(h.device).float(), POS_BINS)
        nb = equal_count_bins(k_pre.float().norm(dim=-1), NORM_BANDS)
        raw = (h * POS_BINS + pb) * NORM_BANDS + nb
    else:
        raise ValueError(variant)
    _, labels = torch.unique(raw, return_inverse=True)
    return labels


def n_cells(variant: str, bits: int) -> int:
    return 2 ** bits * (POS_BINS * NORM_BANDS if variant == "pre_pos_norm" else 1)


def choose_bits(variant: str, budget: int, choices=(4, 6, 8)) -> int:
    """Pick b so the number of cells is closest (in log2) to budget/4."""
    import math
    target = math.log2(max(budget / 4, 1))
    return min(choices, key=lambda b: (abs(math.log2(n_cells(variant, b)) - target), b))

"""R-KVHash scorer, replicated from minghui-liu/kvpress@dixi/rkv_lsh, kvpress/presses/rkv_press_lsh.py.

Faithful details kept on purpose (they matter for T0):
  * keys L2-normalized in the cache dtype (bf16), one shared torch.randn(head_dim, b) projection in bf16;
  * bucket counts accumulated with scatter_add_ in bf16 (counts saturate at 256);
  * avg_cosine = (counts @ cos(Ham/b) - counts) / (total + 1e-8), no pi inside the cosine;
  * softmax over tokens with dtype=torch.bfloat16;
  * lam=0: final = redundancy and topk keeps the highest scores (as in the code);
    lam in (0,1): final = lam * attn - (1 - lam) * redundancy, attn from the last 8 queries, max over the
    GQA group, max_pool1d(kernel 5)."""
import math

import torch
import torch.nn.functional as F

_PROJ = {}


def _proj(d, b, device, seed=4321):
    key = (d, b, str(device))
    if key not in _PROJ:
        g = torch.Generator(device="cpu").manual_seed(seed + b)
        _PROJ[key] = torch.randn(d, b, generator=g, dtype=torch.bfloat16).to(device)
    return _PROJ[key]


def bucket_codes(keys: torch.Tensor, b: int) -> torch.Tensor:
    k = keys.to(torch.bfloat16)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    hb = k @ _proj(k.shape[-1], b, k.device)
    powers = (torch.tensor(1, device=k.device, dtype=torch.int64) << torch.arange(b, device=k.device)).to(torch.bfloat16)
    return torch.sum((hb > 0).int() * powers, dim=-1).long()


def redundancy(keys: torch.Tensor, b: int):
    """keys [N, d] (post-RoPE, scored tokens only). Returns (softmaxed bf16 redundancy [N], codes [N])."""
    codes = bucket_codes(keys, b)
    nb = 2 ** b
    buckets = torch.arange(nb, device=keys.device)
    ham = torch.bitwise_xor(buckets[:, None], buckets[None, :])
    ham = sum(((ham >> i) & 1) for i in range(b)).to(torch.int64)
    cosb = torch.cos(ham / b).to(torch.bfloat16)
    c = codes[None]
    counts = torch.zeros(1, nb, device=keys.device, dtype=torch.bfloat16)
    counts.scatter_add_(1, c, torch.ones_like(c, dtype=torch.bfloat16))
    total = counts.sum(dim=1, keepdim=True)
    avg = (counts @ cosb - counts) / (total + 1e-8)
    red = avg.gather(1, c)
    red = F.softmax(red, dim=-1, dtype=torch.bfloat16)
    return red[0], codes


def window_attention(q_win: torch.Tensor, keys: torch.Tensor, group: int, kernel_size: int = 5):
    """q_win [G*W, d] (last W observed queries per grouped head, post-RoPE, ordered head-major);
    keys [N, d] = the whole cache up to the cut. Returns attention importance [N - 0] over cache tokens."""
    d = keys.shape[-1]
    W = q_win.shape[0] // group
    aw = (q_win.float() @ keys.float().T) / math.sqrt(d)
    N = keys.shape[0]
    # query j of the window sits at cache position N - W + j and may not see later window tokens
    mask = torch.ones(W, N, device=keys.device).triu(diagonal=N - W + 1) * -1e9
    aw = aw.view(group, W, N) + mask
    aw = F.softmax(aw, dim=-1, dtype=torch.bfloat16).float()
    s = aw.mean(dim=1).max(dim=0).values
    return F.max_pool1d(s[None, None], kernel_size=kernel_size, padding=kernel_size // 2, stride=1)[0, 0]


def rkv_scores(keys_cand: torch.Tensor, b: int, lam: float = 0.0, attn_cand: torch.Tensor | None = None):
    red, codes = redundancy(keys_cand, b)
    if lam == 0:
        final = red
    else:
        final = lam * attn_cand.to(torch.bfloat16) - (1 - lam) * red
    return final, red, codes

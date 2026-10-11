"""Compression policies. Each returns, for one slot, kept slot indices [nL, Hkv, L'] and their logit bias.

Protected set (all but rkvhash_released): the whole prompt (contains the first 4 tokens) + last `recent`
generated tokens. Candidates = generated tokens older than the recent window; B of them are kept."""
import math

import torch
import torch.nn.functional as F

from alloc import batched as AB


def _proj(shape, seed, dtype=torch.float32, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype).to(device)


def _codes(x, proj):
    """x [..., d] · proj [d, b] (or broadcastable) -> int codes from sign bits."""
    bits = (x.float() @ proj.float()) > 0
    pw = (2 ** torch.arange(bits.shape[-1], device=x.device)).long()
    return (bits.long() * pw).sum(-1)


class Policy:
    evicts = True
    meta_fields: dict = {}

    def __init__(self, e):
        self.e = e

    def on_prefill(self, l, s, k_pre, k_post):
        pass

    def on_insert(self, l, ar, idx, k_pre, k_post):
        pass

    def select(self, s, L, P, n_gen):
        raise NotImplementedError


class Full(Policy):
    evicts = False

    def select(self, s, L, P, n_gen):
        return None, None


class Protected(Policy):
    """Shared candidate logic for methods with the standard protected set."""

    def select(self, s, L, P, n_gen):
        e = self.e
        c0, c1 = P, L - e.recent
        M = c1 - c0
        if M <= e.budget:
            return None, None
        cand = torch.arange(c0, c1, device=e.dev)
        kept_c, bias_c = self.choose(s, cand, P)                       # [R, B] slot indices, [R, B] bias
        R = e.nL * e.Hkv
        head = torch.arange(P, device=e.dev).expand(R, P)
        tail = torch.arange(c1, L, device=e.dev).expand(R, L - c1)
        keep = torch.cat([head, kept_c, tail], 1)
        bias = torch.cat([torch.zeros(R, P, device=e.dev), bias_c.float(), torch.zeros(R, L - c1, device=e.dev)], 1)
        return keep.view(e.nL, e.Hkv, -1), bias.view(e.nL, e.Hkv, -1)

    def _rows(self, a, s, cand):
        """a [nL, S, Hkv, cap] -> [R, M] for slot s and candidate slots."""
        return a[:, s][:, :, cand].reshape(self.e.nL * self.e.Hkv, -1)

    @staticmethod
    def _take(mask, cand_rows, vals=None):
        R = mask.shape[0]
        kept = cand_rows[mask].view(R, -1)
        return kept, (vals[mask].view(R, -1) if vals is not None else torch.zeros_like(kept, dtype=torch.float32))


class Random(Protected):
    """Uniform random eviction per head at each compression (fresh randomness), no reweighting."""

    def choose(self, s, cand, P):
        e = self.e
        R, M = e.nL * e.Hkv, len(cand)
        r = torch.rand(R, M, generator=e.gen_rng, device=e.dev)
        sel = torch.topk(r, e.budget, dim=-1, largest=False).indices.sort(-1).values
        return cand[sel], torch.zeros(R, e.budget, device=e.dev)


class SimBin(Protected):
    """Post-RoPE SimHash (8 bits, per-head projection), proportional + Webster, keep newest, no reweighting."""
    meta_fields = {"code": torch.long}
    BITS = 8

    def __init__(self, e):
        super().__init__(e)
        self.proj = _proj((e.nL, e.Hkv, e.d, self.BITS), 1717, device=e.dev)

    def on_insert(self, l, ar, idx, k_pre, k_post):
        code = _codes(k_post[:, :, None, :], self.proj[l][None]).squeeze(-1)   # [S, Hkv]
        self.e.meta["code"][l, ar, :, idx] = code

    def choose(self, s, cand, P):
        e = self.e
        R = e.nL * e.Hkv
        lab = self._rows(e.meta["code"], s, cand)
        H = 2 ** self.BITS
        present = torch.zeros(R, H, dtype=torch.long, device=e.dev).scatter_add_(1, lab, torch.ones_like(lab))
        Bv = torch.full((R,), e.budget, device=e.dev)
        shares = present.double() * e.budget / present.sum(-1, keepdim=True)
        n = AB.webster(shares, present, Bv, floor=0)
        mask = AB.select_bottom(lab, -self._rows(e.pos, s, cand).double(), n)
        return self._take(mask, cand.expand(R, -1))


class NeymanKV(Protected):
    """Centered pre-RoPE SimHash (8 bits, hashed once on arrival) x 4 position bins; proportional allocation
    on seen counts with a floor of 1 per non-empty stratum, Webster rounding; uniform reservoir within strata
    (bottom-k of a per-token priority drawn on arrival); logit bias log(N_h / n_h)."""
    meta_fields = {"code": torch.long, "prio": torch.float32, "seen": torch.bool}
    BITS, POS_BINS = 8, 4

    def __init__(self, e, weight=True, floor=1, one_stratum=False):
        super().__init__(e)
        self.weight, self.floor, self.one = weight, floor, one_stratum
        self.H = 1 if one_stratum else 2 ** self.BITS * self.POS_BINS
        self.proj = _proj((e.nL, e.Hkv, e.d, self.BITS), 2929, device=e.dev)
        self.center = torch.zeros(e.nL, e.S, e.Hkv, e.d, device=e.dev)
        self.seen = torch.zeros(e.S, e.nL * e.Hkv, self.H, dtype=torch.long, device=e.dev)

    def on_prefill(self, l, s, k_pre, k_post):
        self.center[l, s] = k_pre.float().mean(1)          # centre = mean prompt pre-RoPE key per head
        if l == 0:
            self.seen[s].zero_()

    def on_insert(self, l, ar, idx, k_pre, k_post):
        e = self.e
        code = _codes((k_pre.float() - self.center[l])[:, :, None, :], self.proj[l][None]).squeeze(-1)
        e.meta["code"][l, ar, :, idx] = code
        e.meta["prio"][l, ar, :, idx] = torch.rand(e.S, e.Hkv, generator=e.gen_rng, device=e.dev)
        e.meta["seen"][l, ar, :, idx] = False

    def choose(self, s, cand, P):
        e = self.e
        R = e.nL * e.Hkv
        if self.one:
            lab = torch.zeros(R, len(cand), dtype=torch.long, device=e.dev)
        else:
            gpos = self._rows(e.pos, s, cand) - P
            pbin = torch.clamp(gpos * self.POS_BINS // e.max_new, max=self.POS_BINS - 1)
            lab = self._rows(e.meta["code"], s, cand) * self.POS_BINS + pbin
        new = ~self._rows(e.meta["seen"], s, cand)
        self.seen[s].scatter_add_(1, lab, new.long())
        e.meta["seen"][:, s][:, :, cand] = True
        present = torch.zeros(R, self.H, dtype=torch.long, device=e.dev).scatter_add_(1, lab, torch.ones_like(lab))
        Bv = torch.full((R,), e.budget, device=e.dev)
        Nseen = torch.maximum(self.seen[s], present)
        shares = AB.waterfill(Nseen.double(), present, Bv, floor=self.floor)
        n = AB.webster(shares, present, Bv, floor=self.floor)
        mask = AB.select_bottom(lab, self._rows(e.meta["prio"], s, cand).double(), n)
        bias = torch.log(Nseen.gather(1, lab).double() / n.gather(1, lab).clamp_min(1).double()).float()
        if not self.weight:
            bias = torch.zeros_like(bias)
        return self._take(mask, cand.expand(R, -1), bias)


class RKVFixed(Protected):
    """R-KVHash scorer (lam=0.1: 0.1*attention - 0.9*redundancy) in fp32 with int32 bucket counts,
    over the standard candidate set; keeps the top-B candidates."""
    BITS, LAM, KERNEL = 6, 0.1, 5

    def __init__(self, e):
        super().__init__(e)
        self.proj = _proj((e.d, self.BITS), 4321 + self.BITS, dtype=torch.bfloat16, device=e.dev).float()
        nb = 2 ** self.BITS
        b = torch.arange(nb, device=e.dev)
        x = b[:, None] ^ b[None, :]
        ham = sum(((x >> i) & 1) for i in range(self.BITS)).float()
        self.cos = torch.cos(ham / self.BITS)

    def window_attn(self, s, L, dtype):
        e = self.e
        q = e.qwin[:, s].to(dtype)                                     # [nL, Hq, W, d]
        k = e.K[:, s, :, :L].to(dtype).repeat_interleave(e.G, dim=1)   # [nL, Hq, L, d]
        W = q.shape[2]
        aw = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(e.d)
        mask = (torch.ones(W, L, device=e.dev) * -1e9).triu(diagonal=L - W + 1).to(dtype)
        aw = torch.softmax(aw + mask, dim=-1, dtype=dtype)
        return aw                                                      # [nL, Hq, W, L]

    def redundancy(self, keys):
        """keys [R, M, d] fp32 -> softmaxed redundancy [R, M]."""
        k = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
        codes = _codes(k, self.proj)
        R, M = codes.shape
        counts = torch.zeros(R, 2 ** self.BITS, dtype=torch.int32, device=keys.device)
        counts.scatter_add_(1, codes, torch.ones_like(codes, dtype=torch.int32))
        cf = counts.float()
        avg = (cf @ self.cos - cf) / (M + 1e-8)
        return torch.softmax(avg.gather(1, codes), dim=-1)

    def choose(self, s, cand, P):
        e = self.e
        L = int(e.L[s])
        R = e.nL * e.Hkv
        keys = e.K[:, s][:, :, cand].float().reshape(R, len(cand), e.d)
        red = self.redundancy(keys)
        aw = self.window_attn(s, L, torch.float32).mean(2)                         # [nL, Hq, L]
        sc = aw.view(e.nL, e.Hkv, e.G, L).max(2).values.reshape(R, 1, L)
        sc = F.max_pool1d(sc, kernel_size=self.KERNEL, padding=self.KERNEL // 2, stride=1)[:, 0, cand]
        final = self.LAM * sc - (1 - self.LAM) * red
        sel = torch.topk(final, e.budget, dim=-1).indices.sort(-1).values
        return cand[sel], torch.zeros(R, e.budget, device=e.dev)


class RKVReleased(RKVFixed):
    """Exact port of kvpress rkv_press_lsh.py (branch dixi/rkv_lsh) decode compression, default lam=0.1:
    bf16 throughout, shared randn(d, 6) bf16 projection, bf16 scatter_add counts, bf16 softmax,
    only the last 8 tokens protected (prompt not protected), topk(cache_budget) gathered in topk order.
    cache_budget = prompt_len + B + recent, so the post-compression cache size equals the other methods'."""
    WINDOW = 8

    def __init__(self, e):
        super().__init__(e)
        self.proj_bf16 = _proj((e.d, self.BITS), 4321 + self.BITS, dtype=torch.bfloat16, device=e.dev)
        self.powers = (torch.tensor(1, device=e.dev) << torch.arange(self.BITS, device=e.dev)).to(torch.bfloat16)
        self.cos_bf16 = self.cos.to(torch.bfloat16)

    def select(self, s, L, P, n_gen):
        e = self.e
        budget = P + e.budget + e.recent
        if budget >= L:
            return None, None
        W, R = self.WINDOW, e.nL * e.Hkv
        bf = torch.bfloat16
        # attention importance (compute_window_attention + mean/max/maxpool)
        aw = self.window_attn(s, L, bf)[..., :-W]                                   # [nL, Hq, W, L-W]
        scores = aw.mean(dim=-2).view(e.nL, e.Hkv, e.G, L - W).max(dim=-2).values   # [nL, Hkv, L-W]
        scores = F.max_pool1d(scores, kernel_size=self.KERNEL, padding=self.KERNEL // 2, stride=1)
        # redundancy via LSH
        keys = e.K[:, s, :, :L - W]                                                  # bf16
        keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
        hb = torch.matmul(keys.reshape(-1, e.d), self.proj_bf16)
        codes = torch.sum((hb > 0).int() * self.powers, dim=-1).view(R, L - W).long()
        counts = torch.zeros(R, 2 ** self.BITS, device=e.dev, dtype=bf)
        counts.scatter_add_(1, codes, torch.ones_like(codes, dtype=bf))
        total = counts.sum(dim=1, keepdim=True)
        avg = (counts @ self.cos_bf16 - counts) / (total + 1e-8)
        red = F.softmax(avg.gather(1, codes), dim=-1, dtype=bf).view(e.nL, e.Hkv, L - W)
        final = self.LAM * scores - (1 - self.LAM) * red
        mx = final.amax(dim=(1, 2), keepdim=True).expand(-1, e.Hkv, W)              # per-layer max, as per call
        final = torch.cat([final, mx], dim=-1)
        keep = final.topk(budget, dim=-1).indices                                   # topk order, not sorted
        return keep, torch.zeros(e.nL, e.Hkv, budget, device=e.dev)


POLICIES = {
    "full": Full,
    "random": Random,
    "simbin": SimBin,
    "rkvhash_released": RKVReleased,
    "rkvhash_fixed": RKVFixed,
    "neymankv_v1": NeymanKV,
    "neymankv_noweight": lambda e: NeymanKV(e, weight=False),
    "neymankv_nofloor": lambda e: NeymanKV(e, floor=0),
    "neymankv_onestratum": lambda e: NeymanKV(e, floor=0, one_stratum=True),
}
for _k, _v in list(POLICIES.items()):
    if not isinstance(_v, type):
        POLICIES[_k] = (lambda f: (lambda e, **kw: f(e)))(_v)

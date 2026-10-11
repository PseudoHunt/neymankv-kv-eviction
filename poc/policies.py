"""Compression policies. Each returns, for one slot, kept slot indices [nL, Hkv, L'] and their logit bias.

Protected set (all but rkvhash_released): the whole prompt (contains the first 4 tokens) + last `recent`
generated tokens. Candidates = generated tokens older than the recent window; B of them are kept."""
import math
import os

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
        self.n_gen = n_gen                                              # true generated count (cache is compacted)
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

    def choose(self, s, cand, P, exclude=None, brow=None):
        e = self.e
        R, M = e.nL * e.Hkv, len(cand)
        r = torch.rand(R, M, generator=e.gen_rng, device=e.dev)
        if exclude is None:
            sel = torch.topk(r, e.budget, dim=-1, largest=False).indices.sort(-1).values
            return cand[sel], torch.zeros(R, e.budget, device=e.dev)
        r = torch.where(exclude, torch.full_like(r, 2.0), r)
        rank = torch.argsort(torch.argsort(r, dim=-1), dim=-1)
        return rank < brow[:, None], torch.zeros(R, M, device=e.dev)               # mask form (reserve path)


class RandomHT(Random):
    """random + Hajek correction: each survivor's bias accumulates log(M/B) per eviction round (M candidates, B kept),
    i.e. -log of its exact inclusion probability; = floor(a/64) * log((B+64)/B) in steady state. Optional clip."""
    meta_fields = {"logw": torch.float32}

    def __init__(self, e, clip=None):
        super().__init__(e)
        self.clip = clip
        self.max_bias = [0.0] * e.S

    def on_prefill(self, l, s, k_pre, k_post):
        if l == 0:
            self.max_bias[s] = 0.0

    def on_insert(self, l, ar, idx, k_pre, k_post):
        self.e.meta["logw"][l, ar, :, idx] = 0.0

    def choose(self, s, cand, P):
        e = self.e
        R, M = e.nL * e.Hkv, len(cand)
        r = torch.rand(R, M, generator=e.gen_rng, device=e.dev)
        sel = torch.topk(r, e.budget, dim=-1, largest=False).indices.sort(-1).values
        logw = self._rows(e.meta["logw"], s, cand).gather(1, sel) + math.log(M / e.budget)
        kept = cand[sel]
        # persist the accumulated weight on the kept tokens; the compaction gather then carries it along
        tmp = e.meta["logw"][:, s].reshape(R, -1).clone()
        tmp.scatter_(1, kept, logw)
        e.meta["logw"][:, s] = tmp.view(e.nL, e.Hkv, -1)
        bias = logw if self.clip is None else logw.clamp(max=self.clip)
        self.max_bias[s] = max(self.max_bias[s], float(bias.max()))
        return kept, bias


class AgeKV(Protected):
    """Phase E2. Target marginal inclusion pi(a) = clip(c * profile[l, h, a // 64], 1/50, 1), c solved per head so the
    expected kept count is B. Survivors are thinned with kappa = pi_now / pi_cum (new candidates: kappa = pi_now),
    so a token's realised inclusion probability is its current target; systematic PPS (random order) keeps exactly B
    per head. Bias = -log pi_cum (Horvitz-Thompson), <= log 50."""
    meta_fields = {"picum": torch.float32}
    PI_MIN = 1 / 50

    def __init__(self, e, profile_path=None):
        super().__init__(e)
        import numpy as np
        path = profile_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            "runs", "followup", "profiles", "R1-Distill-Qwen-1.5B.npz")
        prof = np.load(path)["mean"]                                       # [nL, Hkv, bins]
        prof = np.maximum(prof, prof[prof > 0].min())
        self.prof = torch.tensor(prof, dtype=torch.float64, device=e.dev).reshape(e.nL * e.Hkv, -1)
        self.max_bias = [0.0] * e.S

    def on_prefill(self, l, s, k_pre, k_post):
        if l == 0:
            self.max_bias[s] = 0.0

    def on_insert(self, l, ar, idx, k_pre, k_post):
        self.e.meta["picum"][l, ar, :, idx] = 1.0

    def choose(self, s, cand, P):
        e = self.e
        R, M, B = e.nL * e.Hkv, len(cand), e.budget
        age = self.n_gen - (self._rows(e.pos, s, cand) - P)
        f = self.prof.gather(1, torch.clamp(age // 64, max=self.prof.shape[1] - 1))
        picum = self._rows(e.meta["picum"], s, cand).double()
        kappa = lambda lc: torch.minimum(torch.ones_like(f), torch.clamp(torch.exp(lc) * f, self.PI_MIN, 1.0) / picum)
        lo = torch.full((R, 1), -60.0, device=e.dev, dtype=torch.float64)
        hi = torch.full((R, 1), 60.0, device=e.dev, dtype=torch.float64)
        for _ in range(80):
            mid = (lo + hi) / 2
            under = kappa(mid).sum(-1, keepdim=True) < B
            lo, hi = torch.where(under, mid, lo), torch.where(under, hi, mid)
        k = kappa(hi)
        k = k * (B / k.sum(-1, keepdim=True)).clamp(max=1.0)                   # exact sum B (only scales down)
        k = torch.minimum(k * B / k.sum(-1, keepdim=True), torch.ones_like(k))
        # systematic PPS in a random order, exactly B per row
        order = torch.argsort(torch.rand(R, M, generator=e.gen_rng, device=e.dev), dim=-1)
        cum = torch.cumsum(k.gather(1, order), dim=-1)
        cum = cum / cum[:, -1:] * B
        pts = torch.rand(R, 1, generator=e.gen_rng, device=e.dev, dtype=torch.float64) + torch.arange(B, device=e.dev)
        pick = torch.searchsorted(cum, pts, right=True).clamp(max=M - 1)
        sel = order.gather(1, pick).sort(-1).values
        # guard against duplicate picks from float residue: fall back to top-kappa fill
        dup = (sel[:, 1:] == sel[:, :-1]).any(-1)
        if dup.any():
            alt = torch.topk(k + 1e-9 * torch.rand_like(k), B, dim=-1).indices.sort(-1).values
            sel = torch.where(dup[:, None], alt, sel)
        newpi = (picum * k).gather(1, sel)
        kept = cand[sel]
        tmp = e.meta["picum"][:, s].reshape(R, -1).clone()
        tmp.scatter_(1, kept, newpi.float())
        e.meta["picum"][:, s] = tmp.view(e.nL, e.Hkv, -1)
        bias = (-torch.log(newpi)).float()
        self.max_bias[s] = max(self.max_bias[s], float(bias.max()))
        return kept, bias


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


class NeymanKV2(NeymanKV):
    """v2: Random Attention's implicit recency decay made explicit. Position bins are the 64-token arrival chunks.
    Chunk budget ∝ N_chunk * (B/(B+64))^(age/64) (Webster), then inside each chunk hash buckets get a proportional
    share with a 1-slot floor (when feasible) and a uniform reservoir; bias log(N_h/n_h) per (chunk, bucket)."""

    def __init__(self, e, weight=True, floor=1, strata=True, shared=False):
        Protected.__init__(self, e)
        self.weight, self.floor, self.one = weight, floor, False
        self.strata, self.shared = strata, shared   # Phase B ablations: one hash bucket; keep-set shared across KV heads
        self.CH = e.interval
        self.nb = e.max_new // self.CH + 1
        self.NB = 2 ** self.BITS if strata else 1
        self.H = self.nb * self.NB
        self.r = e.budget / (e.budget + e.interval)
        self.proj = _proj((e.nL, e.Hkv, e.d, self.BITS), 2929, device=e.dev)
        self.center = torch.zeros(e.nL, e.S, e.Hkv, e.d, device=e.dev)
        self.seen = torch.zeros(e.S, e.nL * e.Hkv, self.H, dtype=torch.long, device=e.dev)

    def choose(self, s, cand, P, exclude=None, brow=None):
        e = self.e
        R, nb, NB = e.nL * e.Hkv, self.nb, self.NB
        gpos = self._rows(e.pos, s, cand) - P
        chunk = gpos // self.CH
        lab = chunk * NB + (self._rows(e.meta["code"], s, cand) if self.strata else 0)
        if exclude is not None:   # reserved tokens leave the sampling frame (and are not counted as seen yet)
            lab = torch.where(exclude, torch.full_like(lab, -1), lab)
        labc = lab.clamp_min(0)
        new = ~self._rows(e.meta["seen"], s, cand) & (lab >= 0)
        self.seen[s].scatter_add_(1, labc, new.long())
        if exclude is None:
            e.meta["seen"][:, s][:, :, cand] = True
        else:
            sf = self._rows(e.meta["seen"], s, cand) | (lab >= 0)
            e.meta["seen"][:, s][:, :, cand] = sf.view(e.nL, e.Hkv, -1)
        present = torch.zeros(R, self.H, dtype=torch.long, device=e.dev).scatter_add_(1, labc, (lab >= 0).long())
        Nseen = torch.maximum(self.seen[s], present)
        # level 1: chunk budgets with the explicit recency prior
        n_gen = self.n_gen
        c_present = present.view(R, nb, NB).sum(-1)
        c_seen = Nseen.view(R, nb, NB).sum(-1)
        age = (n_gen - (torch.arange(nb, device=e.dev) * self.CH + self.CH / 2)).clamp_min(0)
        prior = c_seen.double() * self.r ** (age / self.CH)
        Bv = torch.full((R,), e.budget, device=e.dev) if brow is None else brow
        nc = AB.webster(AB.waterfill(prior + 1e-300 * (c_present > 0), c_present, Bv, floor=0), c_present, Bv, floor=0)
        # level 2: hash buckets inside each chunk, proportional on seen counts, 1-slot floor, Webster
        pr, sr = present.view(R * nb, NB), Nseen.view(R * nb, NB)
        rows = (nc.view(-1) > 0).nonzero()[:, 0]
        n = torch.zeros(R * nb, NB, dtype=torch.long, device=e.dev)
        if len(rows):
            sh = AB.waterfill(sr[rows].double(), pr[rows], nc.view(-1)[rows], floor=self.floor)
            n[rows] = AB.webster(sh, pr[rows], nc.view(-1)[rows], floor=self.floor)
        n = n.view(R, self.H)
        prio = self._rows(e.meta["prio"], s, cand).double()
        if self.shared:   # same priorities in every KV head of a layer -> identical keep-sets (needs strata=False)
            prio = prio.view(e.nL, e.Hkv, -1)[:, :1].expand(-1, e.Hkv, -1).reshape(R, -1)
        mask = AB.select_bottom(lab, prio, n)
        bias = torch.log(Nseen.gather(1, labc).double() / n.gather(1, labc).clamp_min(1).double()).float()
        if not self.weight:
            bias = torch.zeros_like(bias)
        if exclude is not None:
            return mask, bias                                                     # mask form (reserve path)
        return self._take(mask, cand.expand(R, -1), bias)


def make_reserve(base_cls, kind, k=8, **base_kw):
    """Reserve up to k slots per head with pi=1, bias 0, outside the base sampler (which gets B - r per head).
    kind='singleton': centred pre-RoPE 8-bit bucket holds no other candidate in that head (oldest first).
    kind='oracle': generated tokens inside the forced prefix (first k generated tokens): retention upper bound."""

    class Reserved(base_cls):
        meta_fields = dict(getattr(base_cls, "meta_fields", {}), rcode=torch.long)

        def __init__(self, e):
            base_cls.__init__(self, e, **base_kw)
            self.rproj = _proj((e.nL, e.Hkv, e.d, 8), 2929, device=e.dev)
            self.rcenter = torch.zeros(e.nL, e.S, e.Hkv, e.d, device=e.dev)

        def on_prefill(self, l, s, k_pre, k_post):
            base_cls.on_prefill(self, l, s, k_pre, k_post)
            self.rcenter[l, s] = k_pre.float().mean(1)

        def on_insert(self, l, ar, idx, k_pre, k_post):
            base_cls.on_insert(self, l, ar, idx, k_pre, k_post)
            code = _codes((k_pre.float() - self.rcenter[l])[:, :, None, :], self.rproj[l][None]).squeeze(-1)
            self.e.meta["rcode"][l, ar, :, idx] = code

        def choose(self, s, cand, P):
            e = self.e
            R, M = e.nL * e.Hkv, len(cand)
            pos = self._rows(e.pos, s, cand)
            if kind == "singleton":
                code = self._rows(e.meta["rcode"], s, cand)
                occ = torch.zeros(R, 256, dtype=torch.long, device=e.dev).scatter_add_(1, code, torch.ones_like(code))
                elig = occ.gather(1, code) == 1
            elif kind == "oracle":
                elig = (pos - P) < k
            else:
                raise ValueError(kind)
            key = torch.where(elig, pos.double(), torch.full_like(pos, 1 << 40).double())   # oldest first
            rank = torch.argsort(torch.argsort(key, dim=-1), dim=-1)
            res = elig & (rank < k)
            brow = e.budget - res.sum(-1)
            mask, bias = base_cls.choose(self, s, cand, P, exclude=res, brow=brow)
            mask = mask & ~res
            total = mask | res
            assert torch.all(total.sum(-1) == e.budget), total.sum(-1)
            bias = torch.where(res, torch.zeros_like(bias), bias)
            self.n_reserved = getattr(self, "n_reserved", []) + [float(res.sum(-1).float().mean())]
            return self._take(total, cand.expand(R, -1), bias)

    return Reserved


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


class RKV(Protected):
    """R-KV (Zefan-Cai/R-KV, HuggingFace/rkv/compression/r1_kv.py + utils.py), correct sign:
    final = mix_lambda * attn - (1 - mix_lambda) * similarity, mix_lambda=0.07, kernel 7, window 8 queries,
    attention = softmax over non-window keys of max-over-GQA-group logits, mean over window queries, max_pool;
    similarity = cosine-similarity matrix (diag 0, per row the last key with sim > 0.5 zeroed), mean over rows, softmax.
    Ported inside our protected set: scored over the candidates; keeps the top-B candidates."""
    LAM, KERNEL, THRESH = 0.07, 7, 0.5

    def choose(self, s, cand, P):
        e = self.e
        L = int(e.L[s])
        R, M = e.nL * e.Hkv, len(cand)
        W = e.qwin.shape[3]
        q = e.qwin[:, s].float().view(e.nL, e.Hkv, e.G, W, e.d)
        k = e.K[:, s, :, :L].float()                                                    # [nL, Hkv, L, d]
        aw = torch.einsum("lhgwd,lhnd->lhgwn", q, k) / math.sqrt(e.d)
        aw = aw.max(dim=2).values[..., :L - W]                                           # max-pool over the group
        aw = torch.softmax(aw, dim=-1).mean(dim=-2)                                      # [nL, Hkv, L-W]
        aw = F.max_pool1d(aw.reshape(R, 1, -1), kernel_size=self.KERNEL, padding=self.KERNEL // 2, stride=1)[:, 0]
        attn = aw[:, cand]
        kc = k.reshape(R, L, e.d)[:, cand]
        kn = kc / (kc.norm(dim=-1, keepdim=True) + 1e-8)
        sim = kn @ kn.transpose(-1, -2)
        sim.diagonal(dim1=-2, dim2=-1).zero_()
        idx = torch.where(sim > self.THRESH, torch.arange(M, device=e.dev).view(1, 1, M), torch.zeros_like(sim, dtype=torch.long))
        sim.scatter_(-1, idx.max(dim=-1).values.unsqueeze(-1), 0)
        simscore = torch.softmax(sim.mean(dim=-2), dim=-1)
        final = self.LAM * attn - (1 - self.LAM) * simscore
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
    "neymankv_v2": NeymanKV2,
    "v2_nostrata": lambda e: NeymanKV2(e, strata=False),
    "v2_noweight": lambda e: NeymanKV2(e, weight=False),
    "v2_nofloor": lambda e: NeymanKV2(e, floor=0),
    "v2_decay_only": lambda e: NeymanKV2(e, strata=False, shared=True),
    "rkv": RKV,
    "agekv": AgeKV,
    "random_reserve": make_reserve(Random, "singleton"),
    "v2_reserve": make_reserve(NeymanKV2, "singleton"),
    "random_oracle": make_reserve(Random, "oracle", k=20),
    "v2_oracle": make_reserve(NeymanKV2, "oracle", k=20),
    "random_ht": RandomHT,
    "random_ht_clip": lambda e: RandomHT(e, clip=math.log(50)),
    "neymankv_v2_noweight": lambda e: NeymanKV2(e, weight=False),
    "neymankv_onestratum": lambda e: NeymanKV(e, floor=0, one_stratum=True),
}
for _k, _v in list(POLICIES.items()):
    if not isinstance(_v, type):
        POLICIES[_k] = (lambda f: (lambda e, **kw: f(e)))(_v)

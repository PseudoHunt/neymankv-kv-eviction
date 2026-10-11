"""Per-head evaluation engine: always-kept set, method selection, and metrics."""
from dataclasses import dataclass, field

import numpy as np
import torch

from alloc import allocation as A
from alloc import select as Sel
from alloc import stats as St
from estimate import estimator as E
from estimate import rkvhash as R
from strata import simhash as SH

N_SINK, N_RECENT = 4, 64


@dataclass
class Head:
    k_post: torch.Tensor  # [N, d]
    k_pre: torch.Tensor   # [N, d]
    v: torch.Tensor       # [N, d]
    pos: torch.Tensor     # [N]
    q_fut: torch.Tensor   # [M, d] held-out future queries (post-RoPE), all grouped heads pooled
    q_obs: torch.Tensor   # [Mo, d] observed queries (post-RoPE), all grouped heads pooled
    group: int = 1
    q_win: torch.Tensor | None = None  # last 8 observed queries per grouped head, for rkvhash lam>0
    q_obs_pos: torch.Tensor | None = None  # positions of q_obs rows (for the causal mask of source="observed")
    cache: dict = field(default_factory=dict)

    def __post_init__(self):
        N = self.k_post.shape[0]
        dev = self.k_post.device
        self.always = torch.cat([torch.arange(min(N_SINK, N)), torch.arange(max(N - N_RECENT, N_SINK), N)]).unique().to(dev)
        m = torch.ones(N, dtype=torch.bool, device=dev)
        m[self.always] = False
        self.cand = torch.nonzero(m)[:, 0]
        self.L = E.logits(self.q_fut, self.k_post)
        self.P = torch.softmax(self.L, dim=-1)
        self.o = self.P @ self.v.float()

    @property
    def n_cand(self):
        return len(self.cand)

    def labels(self, variant, bits):
        key = ("lab", variant, bits)
        if key not in self.cache:
            c = self.cand
            self.cache[key] = SH.build_strata(variant, bits, self.k_post[c], self.k_pre[c], self.pos[c])
        return self.cache[key]

    def spreads(self, variant, bits, source):
        key = ("S", variant, bits, source)
        if key not in self.cache:
            lab = self.labels(variant, bits)
            H = int(lab.max()) + 1
            if source == "oracle":
                P = self.P
            elif source == "proxy":
                P = St.proxy_weights(self.q_obs, self.k_post)
            elif source == "observed":  # extra: empirical attention of the observed queries (causal)
                Lo = E.logits(self.q_obs, self.k_post)
                Lo = Lo.masked_fill(self.pos[None, :] > self.q_obs_pos[:, None], float("-inf"))
                P = torch.softmax(Lo, dim=-1)
            else:
                raise ValueError(source)
            self.cache[key] = {k: v.cpu().numpy() for k, v in St.spreads(P, self.v, self.cand, lab, H).items()}
        return self.cache[key]

    def evaluate(self, idx_c: torch.Tensor, bias_c: torch.Tensor | None):
        """idx_c: kept positions within cand; bias_c: logit bias for them (None = 0)."""
        idx = torch.cat([self.always, self.cand[idx_c]])
        bias = None
        if bias_c is not None:
            bias = torch.cat([torch.zeros(len(self.always), device=bias_c.device), bias_c.float()])
        o_hat = E.est_output(self.L, self.v, idx, bias)
        err = E.rel_err(o_hat, self.o)
        mean, p95 = E.summarize(err)
        return mean, p95, float(E.mass_recall(self.P, idx).mean())


def select(head: Head, m: dict, B: int, rng: np.random.Generator):
    """Return (idx within cand [torch long], bias or None) for method spec m."""
    dev = head.k_post.device
    Nc = head.n_cand
    name = m["name"]
    if B >= Nc:
        return torch.arange(Nc, device=dev), None
    if name == "random_uniform":
        idx = np.sort(rng.choice(Nc, B, replace=False))
        return torch.as_tensor(idx, device=dev), torch.full((B,), float(np.log(Nc / B)), device=dev)
    if name == "random_pnorm":
        d = head.k_post.shape[-1]
        s = (head.k_post[head.cand].double().norm(dim=-1) ** 2 / (2 * d ** 0.5)).cpu().numpy()
        g = s + rng.gumbel(size=Nc)                       # Gumbel top-B = weighted sampling w/o replacement
        idx = np.sort(np.argpartition(-g, B)[:B])
        lse = np.logaddexp.reduce(s)
        bias = -s[idx] + lse - np.log(B)                  # 1/pi_i with pi_i ~= B e^{s_i} / sum_j e^{s_j}
        return torch.as_tensor(idx, device=dev), torch.as_tensor(bias, device=dev)
    if name == "topk_oracle":
        mass = head.P[:, head.cand].sum(0)
        return torch.topk(mass, B).indices.sort().values, None
    if name.startswith("rkvhash"):
        attn = None
        if m.get("lam", 0.0) > 0:
            attn = R.window_attention(head.q_win, head.k_post, head.group)[head.cand]
        final, _, _ = R.rkv_scores(head.k_post[head.cand], m["bits"], m.get("lam", 0.0), attn)
        return torch.topk(final.float(), B).indices.sort().values, None
    # stratified family: simbin, strat_prop, neyman_*
    lab_t = head.labels(m["strata"], m["bits"])
    lab = lab_t.cpu().numpy()
    Nh = np.bincount(lab)
    if m["alloc"] == "proportional":
        n = A.proportional(Nh, B)
    else:
        S = head.spreads(m["strata"], m["bits"], m["source"])[m["stat"]]
        n = A.neyman(Nh, S, B)
    pos = head.pos[head.cand].cpu().numpy()
    idx = Sel.select_within(lab, n, m["within"], pos, rng)
    bias = torch.as_tensor(Sel.reweight_bias(lab, n, idx), device=dev) if m["reweight"] else None
    return torch.as_tensor(idx, device=dev), bias

"""Stratum spreads S_h for Neyman allocation (oracle from held-out queries, proxy from observed ones)."""
import torch


def _per_stratum_std(X: torch.Tensor, labels: torch.Tensor, H: int) -> torch.Tensor:
    """X [Q, Nc] -> sample std over tokens in each stratum, per query: [Q, H] (0 for singletons)."""
    Qn = X.shape[0]
    s1 = torch.zeros(Qn, H, device=X.device, dtype=torch.float64).index_add_(1, labels, X.double())
    s2 = torch.zeros(Qn, H, device=X.device, dtype=torch.float64).index_add_(1, labels, X.double() ** 2)
    Nh = torch.bincount(labels, minlength=H).double()
    var = (s2 - s1 ** 2 / Nh) / (Nh - 1).clamp_min(1)
    return var.clamp_min(0).sqrt()


def spreads(P: torch.Tensor, V: torch.Tensor, cand: torch.Tensor, labels: torch.Tensor, H: int,
            with_vec: bool = False) -> dict:
    """P [Q, N]: attention weights over the full cache (normalized per query); V [N, d].

    key:    S_h = mean_q std_{i in h} p_i(q)
    val:    S_h = mean_q std_{i in h} ||p_i(q) (v_i - o(q))||
    valvec: S_h = mean_q sqrt(trace Cov_{i in h}[p_i(q)(v_i - o(q))])   (diagnostic only)
    """
    V = V.float()
    o = P @ V                                                    # [Q, d]
    Pc, Vc = P[:, cand], V[cand]
    dist2 = (Vc ** 2).sum(-1)[None] - 2 * o @ Vc.T + (o ** 2).sum(-1)[:, None]
    X = Pc * dist2.clamp_min(0).sqrt()
    out = {"key": _per_stratum_std(Pc, labels, H).mean(0), "val": _per_stratum_std(X, labels, H).mean(0)}
    if with_vec:
        Nh = torch.bincount(labels, minlength=H).double()
        sq = torch.zeros(P.shape[0], H, device=P.device, dtype=torch.float64).index_add_(1, labels, (X ** 2).double())
        # sum_{i in h} p_i (v_i - o) = A_h - (sum_{i in h} p_i) o with A_h = sum p_i v_i
        onehot = torch.zeros(len(cand), H, device=P.device, dtype=torch.float32)
        onehot[torch.arange(len(cand), device=P.device), labels] = 1.0
        psum = Pc @ onehot                                       # [Q, H]
        tr = sq.clone()
        for h0 in range(0, H, 64):                               # chunk to bound memory
            h1 = min(H, h0 + 64)
            A = torch.einsum("qn,nh,nd->qhd", Pc, onehot[:, h0:h1], Vc)
            m = A - psum[:, h0:h1, None] * o[:, None, :]
            tr[:, h0:h1] -= (m.double() ** 2).sum(-1) / Nh[h0:h1]
        out["valvec"] = (tr / (Nh - 1).clamp_min(1)).clamp_min(0).sqrt().mean(0)
    return {k: v.float() for k, v in out.items()}


def proxy_weights(Q_obs: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Gaussian query model from observed queries: log E[w_i] = mu.k/sqrt(d) + k^T Sigma k / (2d).

    Returns normalized expected weights over the full cache as a single 'query' row [1, N]."""
    d = K.shape[-1]
    Qo, Kf = Q_obs.double(), K.double()
    mu = Qo.mean(0)
    Sig = torch.cov(Qo.T)
    logE = Kf @ mu / d ** 0.5 + ((Kf @ Sig) * Kf).sum(-1) / (2 * d)
    return torch.softmax(logE, dim=-1)[None].float()

"""Six synthetic checks (CPU). Run: python -m pytest tests -q"""
import numpy as np
import pytest
import torch

from alloc import allocation as A
from alloc import select as Sel
from alloc import stats as St
from common import seed_all
from estimate import estimator as E
from estimate import methods as M
from strata import rope

N, D, HEADS = 2000, 128, 4


def rand_head(seed, n=N, d=D, q=256):
    g = torch.Generator().manual_seed(seed)
    kpre = torch.randn(n, d, generator=g)
    pos = torch.arange(n)
    kpost = rope.apply_rope(kpre, pos, 10000.0)
    v = torch.randn(n, d, generator=g)
    qf = torch.randn(q, d, generator=g)
    qo = torch.randn(q, d, generator=g)
    return M.Head(kpost, kpre, v, pos, qf, qo)


STRAT_METHODS = [
    dict(name="strat_prop", strata="pre", bits=6, alloc="proportional", within="random", reweight=True),
    dict(name="neyman", strata="pre_pos_norm", bits=4, alloc="neyman", stat="val", source="oracle", within="random", reweight=True),
    dict(name="neyman", strata="pre", bits=6, alloc="neyman", stat="key", source="proxy", within="systematic", reweight=True),
    dict(name="simbin", strata="post", bits=6, alloc="proportional", within="oldest_first_evict", reweight=False),
]


# 1. Full-budget identity
@pytest.mark.parametrize("m", STRAT_METHODS + [dict(name="random_uniform"), dict(name="random_pnorm")])
def test_full_budget_identity(m):
    for h in range(HEADS):
        head = rand_head(h)
        idx, bias = M.select(head, m, head.n_cand, np.random.default_rng(0))
        assert len(idx) == head.n_cand
        keep = torch.cat([head.always, head.cand[idx]])
        b = None if bias is None else torch.cat([torch.zeros(len(head.always)), bias.float()])
        o_hat = E.est_output(head.L, head.v, keep, b)
        assert (o_hat - head.o).abs().max() < 1e-5
    # the allocation path itself, with budget == N through Webster, also yields n_h == N_h and zero bias
    lab = head.labels("pre", 6).numpy()
    Nh = np.bincount(lab)
    n = A.neyman(Nh, np.random.rand(len(Nh)), int(Nh.sum()))
    assert np.array_equal(n, Nh)
    assert np.allclose(Sel.reweight_bias(lab, n, np.arange(len(lab))), 0)


# 2. Reweighting identity
def test_reweighting_identity():
    for h in range(HEADS):
        head = rand_head(100 + h)
        K4, V4 = head.k_post.repeat_interleave(4, 0), head.v.repeat_interleave(4, 0)
        L4 = E.logits(head.q_fut, K4)
        full4 = E.full_output(L4, V4)
        keep = torch.arange(0, 4 * N, 4)
        est = E.est_output(L4, V4, keep, torch.full((N,), float(np.log(4))))
        assert (est - full4).abs().max() < 1e-5


# 3. Webster rounding
def test_webster():
    rng = np.random.default_rng(0)
    for trial in range(500):
        H = rng.integers(1, 300)
        Nh = rng.integers(1, 60, size=H)
        B = int(rng.integers(1, Nh.sum() + 1))
        shares = rng.dirichlet(np.ones(H) * rng.uniform(0.1, 3)) * B
        n = A.webster(shares, Nh, B)
        assert n.sum() == B
        assert np.all(n <= Nh) and np.all(n >= 0)
        if B >= H:
            assert np.all(n >= 1)  # floor holds whenever it is feasible
        # Proportional and Neyman paths obey the same invariants
        for n2 in (A.proportional(Nh, B), A.neyman(Nh, rng.random(H), B)):
            assert n2.sum() == B and np.all(n2 <= Nh)
    # exactness on an easy case: shares already integers
    assert np.array_equal(A.webster(np.array([3., 5., 2.]), np.array([10, 10, 10]), 10), [3, 5, 2])


# 4. Reservoir
def test_reservoir():
    T, K, TRIALS = 10_000, 50, 2_000
    counts = np.zeros(T)
    rng = np.random.default_rng(0)
    for _ in range(TRIALS):
        r = Sel.Reservoir(K, rng)
        for x in range(T):
            r.add(x)
        counts[r.items] += 1
    p = counts / TRIALS
    target = K / T
    # Per-token frequencies over 2,000 trials have ~32% binomial noise (10 expected hits), so the +/-20%
    # check is applied to blocks of 100 consecutive stream positions (1,000 expected hits per block).
    blocks = p.reshape(100, 100).mean(1)
    assert np.all(np.abs(blocks / target - 1) < 0.20), (blocks.min() / target, blocks.max() / target)
    assert abs(p.mean() / target - 1) < 1e-9
    # early vs late stream positions are not favoured
    assert abs(p[:1000].mean() / p[-1000:].mean() - 1) < 0.1


# 5. Inverse RoPE (+ hook-captured pre-RoPE key vs inverse-rotated post-RoPE key)
def test_inverse_rope_math():
    x = torch.randn(3, 500, D, dtype=torch.float32)
    pos = torch.arange(500) * 7 + 3
    for theta in (10000.0, 1e6):
        y = rope.apply_rope(x, pos, theta)
        assert (rope.apply_rope(y, pos, theta, inverse=True) - x).abs().max() < 1e-5


@pytest.mark.parametrize("arch", ["qwen3", "qwen2"])
def test_inverse_rope_hooks(arch):
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen3Config
    from capture.hooks import Capturer, rope_theta
    seed_all(0)
    Cfg = Qwen3Config if arch == "qwen3" else Qwen2Config
    cfg = Cfg(vocab_size=500, hidden_size=256, intermediate_size=512, num_hidden_layers=3, num_attention_heads=8,
              num_key_value_heads=2, head_dim=32, max_position_embeddings=4096, rope_theta=1e6 if arch == "qwen3" else 1e4,
              attn_implementation="sdpa")
    model = AutoModelForCausalLM.from_config(cfg).float().eval()
    T, cut = 300, 200
    ids = torch.randint(0, 500, (1, T))
    with torch.no_grad(), Capturer(model, [0, 2], cut=cut, q_lo=cut - 50, q_hi=T, dtype=torch.float32) as cap:
        out = model(ids, position_ids=torch.arange(T)[None])
    ref = model(ids, position_ids=torch.arange(T)[None]).logits
    assert torch.allclose(out.logits, ref, atol=1e-5)  # capture does not change the forward pass
    th = rope_theta(model.config)
    for l in (0, 2):
        d = cap.data[l]
        assert d["k_pre"].shape == (2, cut, 32) and d["q_pre"].shape == (8, T - cut + 50, 32)
        kinv = rope.apply_rope(d["k_post"], torch.arange(cut), th, inverse=True)
        qinv = rope.apply_rope(d["q_post"], torch.arange(cut - 50, T), th, inverse=True)
        assert (kinv - d["k_pre"]).abs().max() < 1e-5
        assert (qinv - d["q_pre"]).abs().max() < 1e-5


# 6. Neyman sanity
def test_neyman_sanity():
    g = torch.Generator().manual_seed(7)
    H, per = 8, 250
    lab = np.repeat(np.arange(H), per)
    K = 0.05 * torch.randn(N, D, generator=g)            # near-uniform attention
    V = torch.randn(N, D, generator=g)
    V[lab == 0] *= 10.0                                  # stratum 0: 10x within-bucket value spread
    Q = torch.randn(256, D, generator=g)
    L = E.logits(Q, K)
    P = torch.softmax(L, -1)
    o = P @ V
    lab_t = torch.as_tensor(lab)
    S = St.spreads(P, V, torch.arange(N), lab_t, H)["val"].numpy()
    B = 200
    n_prop, n_ney = A.proportional(np.bincount(lab), B), A.neyman(np.bincount(lab), S, B)
    assert n_ney[0] > n_prop[0]
    errs = {"prop": [], "ney": []}
    for seed in range(40):
        for name, n in (("prop", n_prop), ("ney", n_ney)):
            rng = np.random.default_rng(seed)
            idx = Sel.select_within(lab, n, "random", np.arange(N), rng)
            bias = torch.as_tensor(Sel.reweight_bias(lab, n, idx))
            errs[name].append(float(E.rel_err(E.est_output(L, V, torch.as_tensor(idx), bias), o).mean()))
    assert np.mean(errs["ney"]) < np.mean(errs["prop"]), {k: np.mean(v) for k, v in errs.items()}
    print(f"\nneyman sanity: n0 prop={n_prop[0]} ney={n_ney[0]}; err prop={np.mean(errs['prop']):.4f} ney={np.mean(errs['ney']):.4f}")


def test_systematic_selection_counts():
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 20, 3000)
    Nh = np.bincount(lab)
    n = A.proportional(Nh, 300)
    for how in ("random", "systematic", "oldest_first_evict"):
        idx = Sel.select_within(lab, n, how, np.arange(3000), rng)
        assert np.array_equal(np.bincount(lab[idx], minlength=20), n), how


def test_pps_inclusion_probabilities():
    rng = np.random.default_rng(3)
    logw = rng.normal(0, 2.5, 300)            # heavy-tailed: several tokens capped at pi = 1
    pi = Sel.inclusion_probs(logw, 40)
    assert abs(pi.sum() - 40) < 1e-9 and pi.max() <= 1 + 1e-12 and (pi == 1).sum() > 0
    hits = np.zeros(300)
    for _ in range(20000):
        idx = Sel.systematic_pps(pi, rng)
        assert len(idx) == 40 and len(np.unique(idx)) == 40
        hits[idx] += 1
    big = pi > 0.05
    assert np.all(np.abs(hits[big] / 20000 / pi[big] - 1) < 0.1)


def test_neyman_zero_spread_strata():
    Nh = np.array([500, 400, 300, 200, 100])
    for S in (np.array([0.07, 0, 0, 0, 0]), np.zeros(5), np.array([1e-30, 0, 0, 2.0, 0]), np.array([np.nan, 1, 0, 0, 0])):
        for B in (5, 37, 251, 600, 1499):
            n = A.neyman(Nh, S, B)
            assert n.sum() == B and np.all(n <= Nh) and np.all(n >= 1)

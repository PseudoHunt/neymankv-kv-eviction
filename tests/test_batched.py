import numpy as np
import torch

from alloc import allocation as A
from alloc import batched as AB


def test_batched_matches_scalar():
    rng = np.random.default_rng(0)
    for _ in range(200):
        H = int(rng.integers(1, 200))
        caps = rng.integers(0, 30, size=H)
        caps[rng.integers(0, H)] = max(1, caps.max())
        a = np.where(caps > 0, rng.integers(1, 100, size=H) + caps, 0).astype(float)
        B = int(rng.integers(1, caps.sum() + 1))
        for floor in (0, 1):
            sh = AB.waterfill(torch.tensor(a[None]), torch.tensor(caps[None]), torch.tensor([B]), floor)
            sh_ref = A.waterfill(a, caps, B, floor) if B < caps.sum() else caps.astype(float)
            if (np.minimum(floor, caps).sum() < B) and B < caps.sum():
                assert np.allclose(sh[0].numpy(), sh_ref, atol=1e-6)
            n = AB.webster(sh, torch.tensor(caps[None]), torch.tensor([B]), floor)[0].numpy()
            assert n.sum() == B and np.all(n <= caps)
            if floor and (caps > 0).sum() <= B:
                assert np.all(n[caps > 0] >= 1)
            if (caps > 0).sum() <= B:
                n_ref = A.webster(sh[0].numpy(), caps, B, floor)
                assert np.abs(n - n_ref).max() <= 1  # tie-breaking may differ by one slot
                assert np.array_equal(np.sort(n), np.sort(n)) and (n != n_ref).sum() <= 2


def test_select_bottom():
    g = torch.Generator().manual_seed(0)
    R, M, H = 6, 500, 20
    lab = torch.randint(-1, H, (R, M), generator=g)
    key = torch.rand(R, M, generator=g)
    present = torch.stack([torch.bincount(l[l >= 0], minlength=H) for l in lab])
    n = torch.minimum(present, torch.randint(0, 10, (R, H), generator=g))
    m = AB.select_bottom(lab, key, n)
    for r in range(R):
        for h in range(H):
            idx = (lab[r] == h).nonzero()[:, 0]
            kept = idx[m[r, idx]]
            assert len(kept) == n[r, h]
            if len(kept):
                assert key[r, kept].max() <= key[r, idx[~m[r, idx]]].min() if (~m[r, idx]).any() else True
        assert not m[r, lab[r] < 0].any()

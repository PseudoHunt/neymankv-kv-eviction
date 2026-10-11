"""Phase E1: attention mass received by a generated token vs its age (64-token bins), per layer and KV head,
from 15 full-cache MATH traces; geometric fit per head. Writes runs/followup/profiles/R1-Distill-Qwen-1.5B.npz."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from poc import clock, data  # noqa: E402
from poc.run import generate, load  # noqa: E402

OUT = os.path.join(ROOT, "runs", "followup", "profiles")
NB, EVERY = 49, 8


def main():
    clock.use("followup")
    print(f"[followup clock {clock.fmt()}] E1", flush=True)
    seed_all(0)
    tok, model = load()
    probs = [p for p in data.problems() if p["set"] == "math"][:15]
    acc = {}
    step = [0]

    def on_step(e, st, positions):
        step[0] += 1
        if step[0] % EVERY:
            return
        act = [s for s, x in enumerate(st) if x is not None]
        if not act:
            return
        if "mass" not in acc:
            acc["mass"] = torch.zeros(e.nL, e.Hkv, NB, dtype=torch.float64, device=e.dev)
            acc["cnt"] = torch.zeros(e.nL, e.Hkv, NB, dtype=torch.float64, device=e.dev)
            acc["prompt"] = torch.zeros(e.nL, e.Hkv, dtype=torch.float64, device=e.dev)
            acc["nq"] = 0
        a = torch.tensor(act, device=e.dev)
        P = torch.tensor([st[s]["P"] for s in act], device=e.dev)
        Lm = int(e.L[a].max())
        for l in range(e.nL):
            q = e.qwin[l, a, :, -1].float().view(len(act), e.Hkv, e.G, e.d)
            k = e.K[l, a, :, :Lm].float()
            sc = torch.einsum("shgd,shnd->shgn", q, k) * e.scale + e.bias[l, a, :, None, :Lm]
            p = torch.softmax(sc, dim=-1).mean(2)                                    # [A, Hkv, Lm] avg over group
            pos = e.pos[l, a, :, :Lm]
            age = positions[a][:, None, None] - pos
            gen = (pos >= P[:, None, None]) & (pos >= 0)
            b = torch.clamp(age // 64, max=NB - 1)
            b = torch.where(gen, b, torch.zeros_like(b))
            acc["mass"][l].scatter_add_(1, b.permute(1, 0, 2).reshape(e.Hkv, -1), (p * gen).permute(1, 0, 2).reshape(e.Hkv, -1).double())
            acc["cnt"][l].scatter_add_(1, b.permute(1, 0, 2).reshape(e.Hkv, -1), gen.permute(1, 0, 2).reshape(e.Hkv, -1).double())
            acc["prompt"][l] += (p * ((pos < P[:, None, None]) & (pos >= 0))).sum((0, 2)).double()
        acc["nq"] += len(act)

    res, _, _ = generate(model, tok, probs, "full", 256, slots=15, max_new=3072, verbose=False, on_step=on_step)
    mass, cnt = acc["mass"].cpu().numpy(), acc["cnt"].cpu().numpy()
    mean = mass / np.maximum(cnt, 1)                                                   # mean mass per token at age bin
    # geometric fit per head: log mean = a + b * bin over bins with enough samples (bin >= 1: past the recent window)
    nL, H = mean.shape[:2]
    slope, rmse, r2 = np.zeros((nL, H)), np.zeros((nL, H)), np.zeros((nL, H))
    for l in range(nL):
        for h in range(H):
            ok = (cnt[l, h] > 200) & (mean[l, h] > 0)
            ok[0] = False
            x, y = np.nonzero(ok)[0], np.log(mean[l, h][ok])
            if len(x) < 3:
                continue
            A = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            slope[l, h], rmse[l, h] = coef[1], np.sqrt(np.mean((y - pred) ** 2))
            r2[l, h] = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "R1-Distill-Qwen-1.5B.npz"), mean=mean, mass=mass, cnt=cnt, slope=slope, rmse=rmse, r2=r2,
             prompt_mass=acc["prompt"].cpu().numpy() / acc["nq"], bin_width=64, every=EVERY)
    json.dump(dict(n_traces=len(res), lengths=[r["n_gen"] for r in res], acc=sum(r["correct"] for r in res),
                   random_implied_slope_per_bin=float(np.log(256 / 320)), median_slope=float(np.median(slope)),
                   median_rmse=float(np.median(rmse)), median_r2=float(np.median(r2)), gpu_clock=clock.fmt()),
              open(os.path.join(OUT, "e1_summary.json"), "w"), indent=1)
    print(f"[followup clock {clock.fmt()}] E1 done", flush=True)


if __name__ == "__main__":
    main()

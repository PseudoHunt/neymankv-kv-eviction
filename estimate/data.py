"""Load captured prompts into per-(layer, KV head) Head objects."""
import glob
import json
import os

import torch
from safetensors.torch import load_file

from common import ROOT
from estimate.methods import Head

CAP = os.path.join(ROOT, "data", "captures")


def prompts():
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(CAP, "*.json")))]


def heads(meta: dict, device="cuda", layers=None):
    """Yield (layer, kv_head, Head)."""
    t = load_file(os.path.join(CAP, f"{meta['id']}.safetensors"))
    G = meta["n_q_heads"] // meta["n_kv_heads"]
    n_obs = meta["n_observed"]
    pos = t["positions"].long().to(device)
    for l in meta["layers"]:
        if layers is not None and l not in layers:
            continue
        kpre, kpost, v, q = (t[f"L{l}.{n}"].to(device).float() for n in ("k_pre", "k_post", "v", "q_post"))
        for j in range(meta["n_kv_heads"]):
            qg = q[j * G:(j + 1) * G]                     # HF repeat_kv: query head h uses KV head h // G
            yield l, j, Head(k_post=kpost[j], k_pre=kpre[j], v=v[j], pos=pos,
                             q_fut=qg[:, n_obs:].reshape(-1, q.shape[-1]),
                             q_obs=qg[:, :n_obs].reshape(-1, q.shape[-1]),
                             group=G, q_win=qg[:, n_obs - 8:n_obs].reshape(-1, q.shape[-1]),
                             q_obs_pos=(meta["q_lo"] + torch.arange(n_obs, device=device)).repeat(G))

"""Gate A: every capture reloads; inverse-RoPE check on real captures.

Real tensors are bf16 in the model (stored fp16, which holds bf16 values exactly in this range), so the
1e-5 synthetic tolerance cannot apply. Two checks per layer:
  (a) HF-replica: rotating the captured pre-RoPE key with HF's bf16 op path reproduces the captured
      post-RoPE key; report the fraction of bit-identical elements.
  (b) inverse: fp64 inverse rotation of the post-RoPE key vs the pre-RoPE key, relative error per token;
      must sit at bf16 rounding level (median < 2^-8)."""
import glob
import json
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402
from strata import rope  # noqa: E402

rows, bad = [], []
for js in sorted(glob.glob(os.path.join(ROOT, "data/captures/*.json"))):
    meta = json.load(open(js))
    try:
        t = load_file(js.replace(".json", ".safetensors"))
    except Exception as e:
        bad.append((meta["id"], repr(e)))
        continue
    pos, qpos = t["positions"].long(), t["q_positions"].long()
    G = meta["n_q_heads"] // meta["n_kv_heads"]
    for l in meta["layers"]:
        kp, kq = t[f"L{l}.k_pre"], t[f"L{l}.k_post"]
        shapes_ok = (kp.shape == kq.shape == t[f"L{l}.v"].shape == (meta["n_kv_heads"], meta["cut"], meta["head_dim"])
                     and t[f"L{l}.q_pre"].shape == (meta["n_q_heads"], meta["q_hi"] - meta["q_lo"], meta["head_dim"]))
        finite = all(torch.isfinite(t[f"L{l}.{n}"]).all() for n in ("k_pre", "k_post", "v", "q_pre", "q_post"))
        rep = rope.apply_rope_like_hf(kp.to(torch.bfloat16), pos, meta["rope_theta"])
        exact = float((rep == kq.to(torch.bfloat16)).float().mean())
        inv = rope.apply_rope(kq.double(), pos, meta["rope_theta"], inverse=True)
        rel = (inv - kp.double()).norm(dim=-1) / kp.double().norm(dim=-1).clamp_min(1e-12)
        qinv = rope.apply_rope(t[f"L{l}.q_post"].double(), qpos, meta["rope_theta"], inverse=True)
        qrel = (qinv - t[f"L{l}.q_pre"].double()).norm(dim=-1) / t[f"L{l}.q_pre"].double().norm(dim=-1).clamp_min(1e-12)
        rows.append(dict(id=meta["id"], layer=l, shapes_ok=shapes_ok, finite=finite, hf_replica_exact=exact,
                         k_rel_med=float(rel.median()), k_rel_max=float(rel.max()), q_rel_med=float(qrel.median()),
                         passed=shapes_ok and finite and float(rel.median()) < 2 ** -8 and exact > 0.99))
import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "runs/gate_a.csv"), index=False)
per_prompt = df.groupby("id").passed.sum()
print(df.describe().T[["mean", "min", "max"]].to_string())
print("prompts:", df.id.nunique(), "unreadable:", bad)
print("prompts with >=3 passing layers:", int((per_prompt >= 3).sum()), "/", len(per_prompt))
print("GATE A:", "PASS" if not bad and (per_prompt >= 3).all() and df.shapes_ok.all() and df.finite.all() else "FAIL")

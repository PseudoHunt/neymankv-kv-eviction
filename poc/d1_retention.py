"""Phase D1: why does v2 lose the forced-prefix fact?
 (a) v2 on the 20 forced-prefix prompts, instrumented: for each fact token, layer, head, compression: present/kept,
     8-bit bucket occupancy, (chunk, bucket) stratum occupancy, key-norm percentile, distance-from-bucket-mean percentile.
 (b) full cache on the same prompts: attention mass the recall-cue query (the one predicting the first code token)
     puts on the fact tokens, per layer and query head."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from poc import clock, data, policies  # noqa: E402
from poc.engine import POLICIES  # noqa: E402
from poc.run import generate, load  # noqa: E402

OUT = os.path.join(ROOT, "runs", "followup")
FACT = "the access code is 7391-KQ"


def fact_indices(tok):
    ids = tok(data.FORCED_PREFIX, add_special_tokens=False)["input_ids"]
    text, spans = "", []
    for i, t in enumerate(ids):
        a = len(text)
        text += tok.decode([t])
        spans.append((a, len(text)))
    lo = text.index(FACT)
    hi = lo + len(FACT)
    return [i for i, (a, b) in enumerate(spans) if b > lo and a < hi], [tok.decode([ids[i]]) for i in range(len(ids))]


class Instrumented(policies.NeymanKV2):
    meta_fields = {"code": torch.long, "prio": torch.float32, "seen": torch.bool, "kpre": (torch.float32, 128)}
    records, fidx, pid_of_slot = [], [], {}

    def on_insert(self, l, ar, idx, k_pre, k_post):
        super().on_insert(l, ar, idx, k_pre, k_post)
        self.e.meta["kpre"][l, ar, :, idx] = k_pre.float() - self.center[l]

    def choose(self, s, cand, P):
        e = self.e
        R, M = e.nL * e.Hkv, len(cand)
        pos = self._rows(e.pos, s, cand)
        code = self._rows(e.meta["code"], s, cand)
        u = e.meta["kpre"][:, s][:, :, cand].reshape(R, M, -1)
        un = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
        knorm = u.norm(dim=-1)  # centred pre-RoPE norm
        NB = 2 ** self.BITS
        occ = torch.zeros(R, NB, device=e.dev).scatter_add_(1, code, torch.ones_like(code, dtype=torch.float))
        sums = torch.zeros(R, NB, un.shape[-1], device=e.dev).scatter_add_(1, code[..., None].expand(-1, -1, un.shape[-1]), un)
        mean = sums / occ.clamp_min(1)[..., None]
        dist = (un - mean.gather(1, code[..., None].expand(-1, -1, un.shape[-1]))).norm(dim=-1)
        chunk = (pos - P) // self.CH
        strat = chunk * NB + code
        kept, _ = super().choose(s, cand, P)
        kept_mask = torch.zeros(R, int(e.cap), dtype=torch.bool, device=e.dev)
        kept_mask.scatter_(1, kept, True)
        n_gen = self.n_gen
        for fi in self.fidx:
            hit = (pos == P + fi)                                          # [R, M]
            rows = hit.any(1).nonzero()[:, 0]
            for r in rows.tolist():
                j = int(hit[r].nonzero()[0])
                self.records.append(dict(
                    pid=self.pid_of_slot.get(s), n_gen=n_gen, layer=r // e.Hkv, head=r % e.Hkv, fact_idx=fi,
                    kept=bool(kept_mask[r, cand[j]]), bucket_occ=int(occ[r, code[r, j]]),
                    stratum_occ=int((strat[r] == strat[r, j]).sum()),
                    norm_pct=float((knorm[r] < knorm[r, j]).float().mean()),
                    dist_pct=float((dist[r] < dist[r, j]).float().mean()), n_cand=M))
        return kept, _


def main():
    clock.use("followup")
    print(f"[followup clock {clock.fmt()}] D1", flush=True)
    seed_all(0)
    tok, model = load()
    fidx, toks = fact_indices(tok)
    print("fact token indices in the forced prefix:", fidx, [toks[i] for i in fidx], flush=True)
    probs = [p for p in data.problems() if p["set"] == "forced"]
    Instrumented.fidx = fidx
    POLICIES["v2_instr"] = Instrumented

    # map slot -> pid via on_step (state dicts carry the problem)
    def on_step(e, st, positions):
        for s, x in enumerate(st):
            if x is not None:
                Instrumented.pid_of_slot[s] = x["p"]["pid"]

    res, stats, _ = generate(model, tok, probs, "v2_instr", 256, slots=20, max_new=3072, verbose=False, on_step=on_step)
    df = pd.DataFrame(Instrumented.records)
    df.to_csv(os.path.join(OUT, "d1_v2_fact_tokens.csv"), index=False)
    rec = {r["pid"]: r["passcode_recall"] for r in res}

    # (b) recall attention under full
    att = []

    def on_recall(e, s, x):
        P = x["P"]
        L = int(e.L[s]) + 1
        for l in range(e.nL):
            q = e.qwin[l, s, :, -1].float()                                     # [Hq, d]
            k = e.K[l, s, :, :L].float().repeat_interleave(e.G, 0)             # [Hq, L, d]
            b = e.bias[l, s, :, :L].repeat_interleave(e.G, 0)
            p = torch.softmax(torch.einsum("hd,hld->hl", q, k) * e.scale + b, dim=-1)
            pos = e.pos[l, s, :, :L].repeat_interleave(e.G, 0)
            fm = torch.zeros_like(pos, dtype=torch.bool)
            for fi in fidx:
                fm |= pos == P + fi
            att.append(dict(pid=x["p"]["pid"], layer=l, **{f"qh{h}": float((p[h] * fm[h]).sum()) for h in range(e.Hq)}))

    res_full, _, _ = generate(model, tok, probs, "full", 256, slots=20, max_new=3072, verbose=False, on_recall=on_recall)
    pd.DataFrame(att).to_csv(os.path.join(OUT, "d1_full_recall_attention.csv"), index=False)
    json.dump(dict(fact_idx=fidx, fact_tokens=[toks[i] for i in fidx], v2_recall=rec,
                   full_recall={r["pid"]: r["passcode_recall"] for r in res_full}, gpu_clock=clock.fmt()),
              open(os.path.join(OUT, "d1_meta.json"), "w"), indent=1)
    print(f"[followup clock {clock.fmt()}] D1 done", flush=True)


if __name__ == "__main__":
    main()

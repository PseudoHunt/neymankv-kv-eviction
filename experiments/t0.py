"""Phase D (T0): R-KVHash scorer on real keys."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, provenance_md, seed_all  # noqa: E402
from estimate import data  # noqa: E402
from estimate import rkvhash as R  # noqa: E402

FRACS = (0.10, 0.25)


def main():
    seed_all(0)
    rows, t0 = [], time.time()
    for meta in data.prompts():
        for l, j, head in data.heads(meta):
            kc = head.k_post[head.cand]
            Nc = len(kc)
            attn = R.window_attention(head.q_win, head.k_post, head.group)[head.cand]
            for b in (6, 8):
                red, codes = R.redundancy(kc, b)
                occ_true = torch.bincount(codes, minlength=2 ** b)[codes]
                occ_bf16 = torch.zeros(2 ** b, device=kc.device, dtype=torch.bfloat16).scatter_add_(
                    0, codes, torch.ones_like(codes, dtype=torch.bfloat16))
                r = dict(prompt=meta["id"], prompt_type=meta["prompt_type"], layer=l, head=j, bits=b, n_cand=Nc,
                         n_buckets=int((torch.bincount(codes) > 0).sum()),
                         max_bucket_true=int(occ_true.max()), max_bucket_bf16=float(occ_bf16.max()),
                         n_distinct_scores=int(torch.unique(red).numel()),
                         spearman_R_occupancy=float(spearmanr(red.float().cpu(), occ_true.cpu()).statistic))
                for f in FRACS:
                    B = int(round(f * Nc))
                    sc = red.float()
                    cutoff = torch.topk(sc, B).values[-1]
                    tied = int((sc == cutoff).sum())
                    above = int((sc > cutoff).sum())
                    idx = torch.topk(sc, B).indices.sort().values
                    kept = torch.zeros(Nc, dtype=torch.bool, device=kc.device)
                    kept[idx] = True
                    ev = ~kept
                    tag = f"_{int(f * 100)}"
                    r["tied_at_cutoff" + tag] = tied
                    r["tie_slots" + tag] = B - above          # slots filled by arbitrary tie-breaking
                    r["evicted_singleton_frac" + tag] = float((occ_true[ev] == 1).float().mean())
                    r["singleton_frac_all"] = float((occ_true == 1).float().mean())
                    r["mean_err" + tag], r["p95_err" + tag], r["mass_recall" + tag] = head.evaluate(idx, None)
                    fin, _, _ = R.rkv_scores(kc, b, 0.1, attn)
                    idx2 = torch.topk(fin.float(), B).indices.sort().values
                    r["mean_err_lam0.1" + tag], _, r["mass_recall_lam0.1" + tag] = head.evaluate(idx2, None)
                rows.append(r)
        print(meta["id"], f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "runs", "t0_rkvhash.csv"), index=False)
    cols = [c for c in df.columns if c not in ("prompt", "prompt_type", "layer", "head", "bits")]
    md = ["# T0: R-KVHash scorer on real keys\n", provenance_md(),
          "Scores replicate kvpress@dixi/rkv_lsh (bf16 normalize/projection/counts/softmax, lam=0 keeps top scores).",
          "`max_bucket_bf16` is the bucket count after bf16 scatter_add (the code's own counts).",
          "`tie_slots_x` = kept slots decided by torch.topk tie-breaking among equal bf16 scores.\n",
          "## Medians per prompt type and bits\n",
          df.groupby(["prompt_type", "bits"])[cols].median().round(4).T.to_markdown()]
    open(os.path.join(ROOT, "runs", "t0_rkvhash.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()

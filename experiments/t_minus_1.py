"""Phase C (T-1): predicted Neyman gain from oracle S_h on held-out queries."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alloc import allocation as A  # noqa: E402
from alloc import stats as St  # noqa: E402
from common import ROOT, provenance_md, seed_all  # noqa: E402
from estimate import data  # noqa: E402

VARIANTS, BITS = ("post", "pre", "pre_pos_norm"), (4, 6, 8)


PARTS = os.path.join(ROOT, "runs", "t_minus_1_parts")


def main():
    seed_all(0)
    os.makedirs(PARTS, exist_ok=True)
    t0 = time.time()
    for meta in data.prompts():
        out = os.path.join(PARTS, f"{meta['id']}.csv")
        if os.path.exists(out):
            continue
        rows = []
        for l, j, head in data.heads(meta):
            for var in VARIANTS:
                for b in BITS:
                    lab = head.labels(var, b)
                    H = int(lab.max()) + 1
                    S = St.spreads(head.P, head.v, head.cand, lab, H, with_vec=True)
                    Nh = torch.bincount(lab, minlength=H).cpu().numpy()
                    r = dict(prompt=meta["id"], prompt_type=meta["prompt_type"], layer=l, head=j, strata_variant=var,
                             bits=b, n_strata=H, n_cand=head.n_cand)
                    for k in ("key", "val", "valvec"):
                        cv, g = A.cv_and_gain(Nh, S[k].cpu().numpy())
                        r[f"cv_{k}"], r[f"gain_{k}"] = cv, g
                    rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(meta["id"], f"{time.time() - t0:.0f}s", flush=True)
    import glob
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(PARTS, "*.csv")))])
    df.to_csv(os.path.join(ROOT, "runs", "t_minus_1.csv"), index=False)
    write_summary(df)


def write_summary(df):
    cols = ["gain_key", "gain_val", "gain_valvec", "cv_key", "cv_val"]
    md = ["# T-1 diagnostic: predicted Neyman gain (oracle S_h, held-out queries)\n", provenance_md(),
          "gain = CV^2/(1+CV^2) of the size-weighted CV of S_h across strata; the predicted fractional variance",
          "reduction of Neyman over proportional allocation (no finite-population correction).",
          "`valvec` (extra, not in the brief) uses sqrt(trace Cov) of the vector p_i(v_i - o) instead of std of its norm.\n",
          "## Medians by prompt type, strata variant, bits\n",
          df.groupby(["prompt_type", "strata_variant", "bits"])[cols + ["n_strata"]].median().round(3).to_markdown(),
          "\n## Medians by prompt type and layer (all strata variants and bits pooled)\n",
          df.groupby(["prompt_type", "layer"])[cols].median().round(3).to_markdown(),
          "\n## Medians by prompt type (pooled)\n",
          df.groupby("prompt_type")[cols].median().round(3).to_markdown(),
          "\n## Gate C\n", gate_c(df)]
    open(os.path.join(ROOT, "runs", "t_minus_1.md"), "w").write("\n".join(md) + "\n")
    print(gate_c(df))


def gate_c(df):
    m = df[df.strata_variant.isin(["pre", "pre_pos_norm"])].groupby("prompt_type")[["gain_key", "gain_val"]].median()
    lines = ["Medians over pre / pre_pos_norm strata, all bits (the strata the Neyman methods use):\n",
             m.round(3).to_markdown(), ""]
    if (m.gain_val >= 0.15).any():
        if ((m.gain_key >= 0.15) & (m.gain_val - m.gain_key < 0.03)).all():
            v = ("CONTINUE. Both rules fire: gain_val >= 0.15, and gain_val does not exceed gain_key by >= 0.03 "
                 "(values add nothing to the predicted gain). The two gains measure different variances "
                 "(attention weights vs output), so T1 runs both key and val S_h and lets measured error decide.")
        else:
            v = "CONTINUE to T1 with values (median gain_val >= 0.15 on at least one prompt type)."
    elif (m.gain_key >= 0.15).any():
        v = "CONTINUE with key-only S_h."
    elif (m[["gain_key", "gain_val"]] < 0.10).all().all():
        v = "STOP: proportional allocation is near-optimal on these captures."
    else:
        v = "AMBIGUOUS: gains between 0.10 and 0.15; not covered by the gate rules."
    lines.append(f"**Verdict: {v}**")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

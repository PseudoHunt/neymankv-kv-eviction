"""Phase E (T1): all methods, budgets 10% / 25%, 5 seeds per random method.

  python experiments/t1.py [shard k of n]   -> runs/t1_parts/<prompt>.csv ; then  python experiments/t1.py --merge
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from estimate import data  # noqa: E402
from estimate import methods as M  # noqa: E402
from strata.simhash import choose_bits  # noqa: E402

FRACS = (0.10, 0.25)
SEEDS = range(5)
PARTS = os.path.join(ROOT, "runs", "t1_parts")


def method_specs(B):
    specs = [dict(name="random_uniform", strata="-", alloc="-", reweight=True, seeds=SEEDS),
             dict(name="random_pnorm", strata="-", alloc="-", reweight=True, seeds=SEEDS),
             dict(name="random_pnorm_naiveHT", strata="-", alloc="-", reweight=True, seeds=SEEDS),
             dict(name="simbin", strata="post", alloc="proportional", within="oldest_first_evict", reweight=False, seeds=[0]),
             dict(name="topk_oracle", strata="-", alloc="-", reweight=False, seeds=[0])]
    for b in (6, 8):
        specs.append(dict(name="rkvhash", strata=f"lsh{b}", alloc="-", bits=b, lam=0.0, reweight=False, seeds=[0]))
    specs.append(dict(name="rkvhash_lam0.1", strata="lsh8", alloc="-", bits=8, lam=0.1, reweight=False, seeds=[0]))
    for st in ("post", "pre", "pre_pos_norm"):
        for rw in (True, False):
            specs.append(dict(name="strat_prop", strata=st, alloc="proportional", within="random", reweight=rw, seeds=SEEDS))
    for src in ("oracle", "proxy"):
        for stat in ("key", "val"):
            for st in ("pre", "pre_pos_norm"):
                for rw in (True, False):
                    specs.append(dict(name=f"neyman_{src}_{stat}", strata=st, alloc=f"neyman_{stat}", source=src,
                                      stat=stat, within="random", reweight=rw, seeds=SEEDS))
    # extra (not in the brief, excluded from Gate E): S_h from the observed queries' own attention
    for stat in ("key", "val"):
        for st in ("pre", "pre_pos_norm"):
            specs.append(dict(name=f"neyman_observed_{stat}", strata=st, alloc=f"neyman_{stat}", source="observed",
                              stat=stat, within="random", reweight=True, seeds=SEEDS))
    # within-stratum ablation (extra): systematic instead of random
    specs.append(dict(name="neyman_proxy_val", strata="pre_pos_norm", alloc="neyman_val", source="proxy", stat="val",
                      within="systematic", reweight=True, seeds=SEEDS))
    for s in specs:
        if s["strata"] in ("post", "pre", "pre_pos_norm"):
            s["bits"] = choose_bits(s["strata"], B)
        s.setdefault("within", "-")
    return specs


def run_prompt(meta):
    rows = []
    for l, j, head in data.heads(meta):
        for f in FRACS:
            B = int(round(f * head.n_cand))
            for spec in method_specs(B):
                for seed in spec["seeds"]:
                    rng = np.random.default_rng([seed, l, j, int(f * 100)])
                    idx, bias = M.select(head, spec, B, rng)
                    mean, p95, mr = head.evaluate(idx, bias)
                    rows.append(dict(method=spec["name"], strata=spec["strata"], allocation=spec["alloc"],
                                     within=spec["within"], bits=spec.get("bits", -1), reweight=spec["reweight"],
                                     budget=f, B=B, prompt=meta["id"], prompt_type=meta["prompt_type"], layer=l,
                                     head=j, mean_err=mean, p95_err=p95, mass_recall=mr, seed=seed))
    return pd.DataFrame(rows)


def main():
    seed_all(0)
    os.makedirs(PARTS, exist_ok=True)
    k, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)
    for i, meta in enumerate(data.prompts()):
        out = os.path.join(PARTS, f"{meta['id']}.csv")
        if i % n != k or os.path.exists(out):
            continue
        t0 = time.time()
        df = run_prompt(meta)
        df.to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        print(meta["id"], len(df), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(PARTS, "*.csv")))])
        df.to_csv(os.path.join(ROOT, "runs", "t1_results.csv"), index=False)
        print(len(df), "rows")
    else:
        main()

"""Paired per-problem analysis for the follow-up phases.

Per problem, correctness is averaged over the seeds a method was run with; differences are paired by problem;
95% intervals are percentile bootstraps over problems (10,000 resamples, seed 0)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402
from poc import data, grade  # noqa: E402

D = os.path.join(ROOT, "runs", "followup")
RNG = np.random.default_rng(0)
NBOOT = 10_000


def load(phase=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "gen", "*.jsonl"))):
        st = json.load(open(f.replace(".jsonl", ".stats.json")))
        name = st["name"]
        ph = name.split("_")[-2] if name.endswith(tuple(f"_s{i}" for i in range(10))) else name.split("_")[-1]
        if phase and ph != phase:
            continue
        for line in open(f):
            r = json.loads(line)
            ans = grade.answer_part(r["text"])
            r["pred"] = grade.last_boxed(ans)
            r["correct"] = grade.correct(r["pred"], r["gold"])
            r.update(method=st["method"], budget=st["budget"], seed=st.get("seed", 0), phase=ph)
            r["dataset"] = "math" if r["set"].startswith("math") else ("gsm" if r["set"].startswith("gsm") else r["set"])
            r["orig40"] = r["set"] in ("math", "gsm")
            r.pop("text")
            rows.append(r)
    return pd.DataFrame(rows)


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(NBOOT, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def per_problem(df, method, metric="correct", budget=None):
    d = df[df.method == method]
    if budget is not None:
        d = d[d.budget == budget]
    return d.groupby("pid")[metric].mean()


def paired(df, a, b, metric="correct", budget_a=None, budget_b=None, subset=None):
    pa, pb = per_problem(df, a, metric, budget_a), per_problem(df, b, metric, budget_b)
    common = pa.index.intersection(pb.index)
    if subset is not None:
        common = common.intersection(subset)
    diff = (pa[common] - pb[common]).astype(float)
    m, lo, hi = boot_ci(diff)
    return dict(comparison=f"{a} - {b}", n_problems=len(common), mean_a=float(pa[common].mean()), mean_b=float(pb[common].mean()),
                diff=m, ci_lo=lo, ci_hi=hi, wins=int((diff > 0).sum()), losses=int((diff < 0).sum()), ties=int((diff == 0).sum()),
                excludes_0=bool(lo > 0 or hi < 0))


def method_table(df, methods, budget=None):
    out = []
    for m in methods:
        d = df[df.method == m]
        if budget is not None:
            d = d[d.budget == budget]
        if d.empty:
            continue
        for ds in sorted(d.dataset.unique()):
            x = d[d.dataset == ds]
            row = dict(method=m, dataset=ds, n_traces=len(x), n_problems=x.pid.nunique(), seeds=sorted(x.seed.unique().tolist()))
            for metric in ("correct", "n_gen", "hit_cap"):
                pp = x.groupby("pid")[metric].mean()
                mean, lo, hi = boot_ci(pp)
                row[f"{metric}"], row[f"{metric}_lo"], row[f"{metric}_hi"] = mean, lo, hi
            out.append(row)
    return pd.DataFrame(out)


def phase_a(df):
    rows = []
    for ds in ("math", "gsm", "all"):
        sub = df if ds == "all" else df[df.dataset == ds]
        for orig in (None, True, False):
            s2 = sub if orig is None else sub[sub.orig40 == orig]
            ids = pd.Index(s2.pid.unique())
            for a, b in (("neymankv_v2", "random"), ("neymankv_v2", "full"), ("random", "full")):
                r = paired(s2, a, b)
                r.update(dataset=ds, problems={None: "all", True: "PoC original 40", False: "new"}[orig])
                rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "A"
    df = load(ph)
    pd.set_option("display.width", 220)
    print(df.groupby(["method", "budget", "seed", "dataset"]).size())
    if ph == "A":
        t = phase_a(df)
        t.to_csv(os.path.join(D, "phaseA_paired.csv"), index=False)
        print(t.round(3).to_string())
        mt = method_table(df, ["full", "random", "neymankv_v2"])
        mt.to_csv(os.path.join(D, "phaseA_methods.csv"), index=False)
        print(mt.round(3).to_string())

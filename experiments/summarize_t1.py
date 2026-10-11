"""Summaries, Gate E, ablations and the plot for T1.  python experiments/summarize_t1.py"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, provenance_md  # noqa: E402

KEY = ["method", "strata", "within", "reweight"]
RUNS = os.path.join(ROOT, "runs")


def load():
    df = pd.read_csv(os.path.join(RUNS, "t1_results.csv"))
    df["strata"] = df["strata"].fillna("-")
    df["within"] = df["within"].fillna("-")
    return df


def table(df):
    """Primary stat: median over heads of the seed-averaged mean_err. Std: across seeds of per-seed medians."""
    g = KEY + ["prompt_type", "budget"]
    per_head = df.groupby(g + ["prompt", "layer", "head"])[["mean_err", "p95_err", "mass_recall"]].mean()
    med = per_head.groupby(g).median()
    seed_med = df.groupby(g + ["seed"]).mean_err.median().groupby(g).std().fillna(0).rename("seed_std")
    return med.join(seed_med).reset_index()


def label(r):
    s = r.method
    if r.strata != "-":
        s += f" [{r.strata}]"
    if r.within not in ("-", "random"):
        s += f" ({r.within})"
    if r.method.startswith(("strat_prop", "neyman")):
        s += " rw" if r.reweight else " no-rw"
    return s


def gate_e(t):
    out = ["## Gate E\n",
           "Criterion: `neyman_proxy_*` with reweighting on `pre`/`pre_pos_norm` cuts the median relative output error by",
           ">= 15% vs `random_pnorm` at both budgets on long_doc, and is not worse than `random_pnorm` on reasoning.\n"]
    base = t[t.method == "random_pnorm"].set_index(["prompt_type", "budget"]).mean_err
    rows, any_pass = [], {}
    for fam in ("neyman_proxy", "neyman_oracle"):
        for stat in ("key", "val"):
            for st in ("pre", "pre_pos_norm"):
                sel = t[(t.method == f"{fam}_{stat}") & (t.strata == st) & t.reweight & (t.within == "random")]
                s = sel.set_index(["prompt_type", "budget"]).mean_err
                red = 1 - s / base
                ok_doc = all(red.get(("long_doc", b), -1) >= 0.15 for b in (0.10, 0.25))
                ok_rea = all(red.get(("reasoning", b), -1) >= 0.0 for b in (0.10, 0.25))
                rows.append(dict(method=f"{fam}_{stat}", strata=st,
                                 **{f"{pt[:4]}@{int(b*100)}%": f"{s.get((pt, b), np.nan):.4f} ({red.get((pt, b), np.nan):+.0%})"
                                    for pt in ("long_doc", "reasoning") for b in (0.10, 0.25)},
                                 passes=ok_doc and ok_rea))
                any_pass.setdefault(fam, []).append(ok_doc and ok_rea)
    out.append("Cells: median error (reduction vs random_pnorm; positive = better).\n")
    out.append(f"random_pnorm baseline: " + ", ".join(f"{pt}@{int(b*100)}% = {v:.4f}" for (pt, b), v in base.items()) + "\n")
    out.append(pd.DataFrame(rows).to_markdown(index=False))
    p, o = any(any_pass["neyman_proxy"]), any(any_pass["neyman_oracle"])
    verdict = "PASS" if p else "FAIL"
    note = ""
    if not p and o:
        note = " The oracle passes but the proxy fails: the problem is the S_h proxy."
    elif not p and not o:
        note = " Neither the oracle nor the proxy passes."
    out.append(f"\n**Gate E: {verdict}** (proxy passes: {p}; oracle passes: {o}).{note}\n")
    return "\n".join(out), verdict


def ablations(t):
    md = ["## Ablations (median error; long_doc / reasoning; 10% / 25%)\n"]
    piv = lambda d: d.pivot_table(index="config", columns=["prompt_type", "budget"], values="mean_err").round(4)

    def mk(d):
        d = d.copy()
        d["config"] = d.apply(label, axis=1)
        return piv(d).to_markdown()
    r = t[t.within.isin(["random", "-"])]
    md += ["### Reweighting on vs off (fixed allocation)\n",
           mk(r[r.method.isin(["strat_prop", "neyman_oracle_val", "neyman_proxy_val"])]), ""]
    md += ["### Proportional vs Neyman (fixed strata, reweighting on)\n",
           mk(r[r.reweight & r.strata.isin(["pre", "pre_pos_norm"]) & r.method.str.match("strat_prop|neyman")]), ""]
    md += ["### post vs pre strata (proportional, reweighting on)\n",
           mk(r[(r.method == "strat_prop") & r.reweight]), ""]
    md += ["### key vs val S_h (reweighting on)\n",
           mk(r[r.reweight & r.method.str.startswith("neyman")]), ""]
    md += ["### Within-stratum selection: random vs systematic (neyman_proxy_val, pre_pos_norm, rw)\n",
           mk(t[(t.method == "neyman_proxy_val") & (t.strata == "pre_pos_norm") & t.reweight]), ""]
    return "\n".join(md)


MAIN = [  # (label, selector)
    ("random_uniform", lambda t: t.method == "random_uniform"),
    ("random_pnorm", lambda t: t.method == "random_pnorm"),
    ("simbin", lambda t: t.method == "simbin"),
    ("rkvhash (b=8)", lambda t: (t.method == "rkvhash") & (t.strata == "lsh8")),
    ("strat_prop [pre_pos_norm] rw", lambda t: (t.method == "strat_prop") & (t.strata == "pre_pos_norm") & t.reweight),
    ("neyman_proxy_val [pre] rw", lambda t: (t.method == "neyman_proxy_val") & (t.strata == "pre") & t.reweight & (t.within == "random")),
]
REF = [("neyman_oracle_val [pre_pos_norm] rw", lambda t: (t.method == "neyman_oracle_val") & (t.strata == "pre_pos_norm") & t.reweight),
       ("topk_oracle (reference)", lambda t: t.method == "topk_oracle")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot(t):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, pt in zip(axes, ("long_doc", "reasoning")):
        for i, (name, sel) in enumerate(MAIN + REF):
            d = t[sel(t) & (t.prompt_type == pt)].sort_values("budget")
            ax.plot(d.budget * 100, d.mean_err, color=COLORS[i], marker=MARKERS[i], markersize=8, linewidth=2,
                    linestyle="--" if i >= len(MAIN) else "-", label=name)
        ax.set_yscale("log")
        ax.set_xticks([10, 25])
        ax.set_xlabel("budget (% of evictable cache)")
        ax.set_title({"long_doc": "Long documents (Qwen3-4B)", "reasoning": "Reasoning traces (R1-Distill-7B)"}[pt],
                     loc="left", fontsize=11)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("median relative output error (log)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "t1_error_vs_budget.png"), dpi=150, bbox_inches="tight")


def main():
    df = load()
    t = table(df)
    t["config"] = t.apply(label, axis=1)
    t.to_csv(os.path.join(RUNS, "t1_summary.csv"), index=False)
    gate, verdict = gate_e(t)
    full = t.pivot_table(index="config", columns=["prompt_type", "budget"], values=["mean_err", "p95_err", "mass_recall"])
    md = ["# T1 results\n", provenance_md(),
          "Statistic: per head, average mean_err over seeds; then median over (prompt, layer, KV head).",
          "seed_std = std over seeds of the per-seed median (random methods only).\n",
          gate, ablations(t), "## Full table: median mean_err\n",
          full["mean_err"].round(4).to_markdown(), "\n## Median p95 error over queries\n",
          full["p95_err"].round(4).to_markdown(), "\n## Median attention-mass recall\n",
          full["mass_recall"].round(3).to_markdown(), "\n## Seed std of the median (mean_err)\n",
          t.pivot_table(index="config", columns=["prompt_type", "budget"], values="seed_std").round(4).to_markdown(),
          "\n![error vs budget](t1_error_vs_budget.png)\n"]
    open(os.path.join(RUNS, "t1_summary.md"), "w").write("\n".join(md) + "\n")
    plot(t)
    print(gate)


if __name__ == "__main__":
    main()

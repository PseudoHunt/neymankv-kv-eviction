"""Tables, verdict inputs and the plot for the PoC.  python -m poc.summarize"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402
from poc import data, grade  # noqa: E402

D = os.path.join(ROOT, "runs", "poc")
ORDER = ["full", "random", "simbin", "rkvhash_released", "rkvhash_fixed", "neymankv_v1", "neymankv_noweight",
         "neymankv_v2", "random_ht", "random_ht_clip"]


def load():
    rows, stats = [], []
    for f in sorted(glob.glob(os.path.join(D, "gen", "*.jsonl"))):
        st = json.load(open(f.replace(".jsonl", ".stats.json")))
        stats.append(st)
        for line in open(f):
            r = json.loads(line)
            ans = grade.answer_part(r["text"])
            r["pred"] = grade.last_boxed(ans)                     # regrade with the final grader
            r["correct"] = grade.correct(r["pred"], r["gold"])
            r["passcode_answer"] = data.CODE in ans and "</think>" in r["text"]
            r["method"], r["budget"] = st["method"], st["budget"]
            r.setdefault("passcode_recall", None)
            r.setdefault("max_bias", None)
            del r["text"]
            rows.append(r)
    return pd.DataFrame(rows), stats


def table(df, stats):
    g = df.groupby(["method", "budget", "set"])
    t = g.agg(n=("correct", "size"), acc=("correct", "mean"), len_mean=("n_gen", "mean"),
              len_p95=("n_gen", lambda x: np.percentile(x, 95)), loop_rate=("loop", "mean"), cap_rate=("hit_cap", "mean"),
              passcode=("passcode_answer", "mean"), passcode_any=("passcode_any", "mean"),
              passcode_recall=("passcode_recall", "mean"), max_bias_mean=("max_bias", "mean"),
              max_bias_max=("max_bias", "max")).reset_index()
    s = pd.DataFrame([dict(method=x["method"], budget=x["budget"], set=x["name"].split("_B")[-1].split("_", 1)[1], tok_s=x["tok_per_s"],
                           peak_mem_gb=x["peak_mem_gb"], gpu_clock_end=x["gpu_clock_end"]) for x in stats])
    s["set"] = s["set"].str.replace("math-gsm", "math,gsm")
    return t, s


def plot(t):
    """Only B=256 was run (S3 dropped), so: accuracy per method at B=256, one panel per dataset, full as reference."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    methods = [m for m in ORDER if m != "full" and m in set(t.method)]
    for ax, st in zip(axes, ("math", "gsm")):
        d = t[(t.set == st)].set_index("method")
        if "full" in d.index:
            ax.axvline(d.loc["full", "acc"] * 100, color="#52514e", linestyle="--", linewidth=1.5, label="full (no eviction)")
        for i, m in enumerate(methods):
            if m in d.index:
                acc, n = d.loc[m, "acc"] * 100, d.loc[m, "n"]
                se = 100 * np.sqrt(d.loc[m, "acc"] * (1 - d.loc[m, "acc"]) / n)
                c = "#6250d6" if m.startswith("neymankv_v2") else ("#eb6834" if m.startswith("neymankv") else "#2a78d6")
                ax.errorbar(acc, i, xerr=se, fmt="o", color=c, markersize=8, capsize=3, linewidth=1.5)
                ax.text(acc, i + 0.28, f"{acc:.1f}", ha="center", fontsize=8, color="#52514e")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.invert_yaxis()
        ax.set_xlim(20, 90)
        ax.set_xlabel("accuracy (%) at B=256, ±1 s.e.")
        ax.set_title({"math": "MATH500 (n=40)", "gsm": "GSM8K (n=40; v2/random_ht not run)"}[st], loc="left", fontsize=11)
        ax.grid(axis="x", color="#e6e6e3", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "accuracy_vs_budget.png"), dpi=150, bbox_inches="tight")


def main():
    df, stats = load()
    t, s = table(df, stats)
    df.to_csv(os.path.join(D, "per_problem.csv"), index=False)
    t.to_csv(os.path.join(D, "results.csv"), index=False)
    s.to_csv(os.path.join(D, "run_stats.csv"), index=False)
    kept = [dict(method=x["method"], budget=x["budget"], set=x["name"].split("_B")[-1].split("_", 1)[1], pid=x["kept_log_pid"],
                 n_compressions=len(x["kept_log"]), cache_len=sorted({k["cache_len"] for k in x["kept_log"]}),
                 gen_kept_min=min([k["gen_kept_min"] for k in x["kept_log"]], default=None),
                 gen_kept_max=max([k["gen_kept_max"] for k in x["kept_log"]], default=None)) for x in stats]
    pd.DataFrame(kept).to_csv(os.path.join(D, "kept_set_log.csv"), index=False)
    if not t[t.set.isin(["math", "gsm"])].empty:
        plot(t)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print(s.round(1).to_string())
    print(pd.DataFrame(kept).to_string())


if __name__ == "__main__":
    main()

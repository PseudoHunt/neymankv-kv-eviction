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
         "neymankv_nofloor"]


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
            del r["text"]
            rows.append(r)
    return pd.DataFrame(rows), stats


def table(df, stats):
    g = df.groupby(["method", "budget", "set"])
    t = g.agg(n=("correct", "size"), acc=("correct", "mean"), len_mean=("n_gen", "mean"),
              len_p95=("n_gen", lambda x: np.percentile(x, 95)), loop_rate=("loop", "mean"), cap_rate=("hit_cap", "mean"),
              passcode=("passcode_answer", "mean"), passcode_any=("passcode_any", "mean")).reset_index()
    s = pd.DataFrame([dict(method=x["method"], budget=x["budget"], set=x["name"].split("_")[-1], tok_s=x["tok_per_s"],
                           peak_mem_gb=x["peak_mem_gb"], gpu_clock_end=x["gpu_clock_end"]) for x in stats])
    s["set"] = s["set"].str.replace("math-gsm", "math,gsm")
    return t, s


def plot(t):
    from poc.summarize import ORDER
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, st in zip(axes, ("math", "gsm")):
        d = t[t.set == st]
        for i, m in enumerate(ORDER):
            dm = d[d.method == m].sort_values("budget")
            if dm.empty:
                continue
            if m == "full":
                ax.axhline(dm.acc.iloc[0] * 100, color=colors[i], linestyle="--", linewidth=2, label="full (no eviction)")
                continue
            ax.plot(dm.budget, dm.acc * 100, color=colors[i], marker=markers[i], markersize=8, linewidth=2, label=m)
        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 1024])
        ax.set_xticklabels(["256", "1024"])
        ax.set_xlabel("generated-token budget B")
        ax.set_title({"math": "MATH500 (n=40)", "gsm": "GSM8K (n=40)"}[st], loc="left", fontsize=11)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("accuracy (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "accuracy_vs_budget.png"), dpi=150, bbox_inches="tight")


def main():
    df, stats = load()
    t, s = table(df, stats)
    df.to_csv(os.path.join(D, "per_problem.csv"), index=False)
    t.to_csv(os.path.join(D, "results.csv"), index=False)
    s.to_csv(os.path.join(D, "run_stats.csv"), index=False)
    kept = [dict(method=x["method"], budget=x["budget"], set=x["name"].split("_")[-1], pid=x["kept_log_pid"],
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

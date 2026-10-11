"""P1.1 figure: kept generated tokens per head by age, one MATH trace (math_134) per method, layers/heads pooled."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402

d = json.load(open(os.path.join(ROOT, "runs", "poc", "age_hist.json")))
edges = np.array(d["edges"])
mid = (edges[:-1] + edges[1:]) / 2
colors = {"random": "#2a78d6", "simbin": "#1baf7a", "neymankv_v1": "#eb6834", "neymankv_v2": "#6250d6"}
markers = {"random": "o", "simbin": "^", "neymankv_v1": "s", "neymankv_v2": "D"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, at in zip(axes, ("1024", "1536", "2048")):
    for m in ("random", "simbin", "neymankv_v1", "neymankv_v2"):
        h = d["hist"].get(m, {}).get(at)
        if h is None:
            continue
        n = int(at) // 128
        ax.plot(mid[:n], h[:n], color=colors[m], marker=markers[m], markersize=6, linewidth=2,
                label=m + (" (age bug: prior not applied)" if m == "neymankv_v2" else ""))
    ax.set_title(f"after {at} generated tokens", loc="left", fontsize=11)
    ax.set_xlabel("token age (generated tokens since it was produced)")
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("kept tokens per head, per 128-token age bin")
axes[2].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("Kept generated tokens by age, B=256 (first bin includes the 64 protected recent tokens)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
for out in ("runs/poc/age_histogram.png", "runs/agekv/age_histogram.png"):
    fig.savefig(os.path.join(ROOT, out), dpi=150, bbox_inches="tight")

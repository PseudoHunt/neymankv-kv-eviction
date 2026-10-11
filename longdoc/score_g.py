"""Per-sample scoring for Phase G and paired bootstrap vs a reference method."""
import json, sys, glob, os
import numpy as np

G = "runs/followup/G"

def ruler_score(r):
    pred = r["predicted_answer"].lower(); ans = [a.lower() for a in r["answer"]]
    if "qa" in r["task"]:  # string_match_part
        return float(max(a in pred for a in ans))
    return sum(a in pred for a in ans) / len(ans)  # string_match_all

def rouge_l(pred, ref):
    a, b = pred.lower().split(), ref.lower().split()
    if not a or not b: return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    l = dp[-1]
    if l == 0: return 0.0
    p, rc = l / len(a), l / len(b)
    return 2 * p * rc / (p + rc)

def load(d, m, ratio):
    f = f"{G}/{d}_{m}_r{ratio}.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f)]
    seen = {}
    for r in rows:
        ref = r["answer"] if isinstance(r["answer"], str) else (r["answer"][0] if d == "loogle" else None)
        r["score"] = rouge_l(r["predicted_answer"], ref) if d == "loogle" else ruler_score(r)
        # context_hash is Python's salted hash(), not stable across processes: key on content + occurrence
        base = (r["task"], r["question"], json.dumps(r["answer"]))
        seen[base] = seen.get(base, -1) + 1
        r["key"] = base + (seen[base],)
    return rows

def paired(a, b, n_boot=10000, seed=0):
    ka = {r["key"]: r["score"] for r in a}; kb = {r["key"]: r["score"] for r in b}
    keys = [k for k in ka if k in kb]
    d = np.array([ka[k] - kb[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return dict(n=len(d), mean_a=np.mean([ka[k] for k in keys]), mean_b=np.mean([kb[k] for k in keys]),
                diff=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()))

if __name__ == "__main__":
    import pandas as pd
    comps = [("v1_prop", "random"), ("v1_prop", "v1_prop_noweight"), ("random_rw", "random"), ("v1_prop_noweight", "random"), ("v1_prop_noweight", "random_rw"),
             ("v1_prop", "random_rw"), ("v1_prop", "expected_attention"), ("expected_attention", "random"),
             ("kvzip", "random"), ("random", "full"), ("v1_prop", "full")]
    out = []
    for d in ["ruler4k", "ruler8k", "loogle"]:
        for ratio in [0.75, 0.9]:
            for a, b in comps:
                ra, rb = load(d, a, ratio), load(d, b, ratio)
                if ra is None or rb is None: continue
                out.append(dict(dataset=d, ratio=ratio, comparison=f"{a} - {b}", **paired(ra, rb)))
    df = pd.DataFrame(out).round(4)
    df.to_csv(f"{G}/paired.csv", index=False)
    print(df.to_string())

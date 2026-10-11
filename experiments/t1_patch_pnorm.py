"""Re-run random_pnorm with exact capped inclusion probabilities; keep the first version as random_pnorm_naiveHT."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from estimate import data  # noqa: E402
from estimate import methods as M  # noqa: E402
from t1 import FRACS, PARTS, SEEDS  # noqa: E402

k, n = int(sys.argv[1]), int(sys.argv[2])
for i, meta in enumerate(data.prompts()):
    if i % n != k or meta["prompt_type"] != "long_doc":
        continue
    path = os.path.join(PARTS, f"{meta['id']}.csv")
    df = pd.read_csv(path)
    if (df.method == "random_pnorm_naiveHT").any():
        continue
    df.loc[df.method == "random_pnorm", "method"] = "random_pnorm_naiveHT"
    rows = []
    for l, j, head in data.heads(meta):
        for f in FRACS:
            B = int(round(f * head.n_cand))
            for seed in SEEDS:
                rng = np.random.default_rng([seed, l, j, int(f * 100)])
                idx, bias = M.select(head, dict(name="random_pnorm"), B, rng)
                assert len(idx) == B
                mean, p95, mr = head.evaluate(idx, bias)
                rows.append(dict(method="random_pnorm", strata="-", allocation="-", within="-", bits=-1, reweight=True,
                                 budget=f, B=B, prompt=meta["id"], prompt_type=meta["prompt_type"], layer=l, head=j,
                                 mean_err=mean, p95_err=p95, mass_recall=mr, seed=seed))
    pd.concat([df, pd.DataFrame(rows)]).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(meta["id"], flush=True)

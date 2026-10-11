"""Bug check: age histogram of kept generated tokens (random vs neymankv_v1 vs simbin), one trace each."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from poc import clock, data  # noqa: E402
from poc import engine as E  # noqa: E402
from poc.run import generate, load  # noqa: E402

AT = (1024, 1536, 2048)
EDGES = list(range(0, 2049, 128))
out = {}
orig = E.Engine.compress


def patched(self, s, P, n_gen):
    Ln = orig(self, s, P, n_gen)
    if Ln is not None and n_gen in AT:
        pos = self.pos[:, s, :, :Ln].reshape(-1, Ln)
        age = (n_gen - (pos - P)).float()
        age = age[pos >= P]
        h = np.histogram(age.cpu().numpy(), bins=EDGES)[0] / (self.nL * self.Hkv)   # kept tokens per head per age bin
        out.setdefault(self.method, {})[n_gen] = h.round(1).tolist()
    return Ln


E.Engine.compress = patched
print(f"[GPU clock {clock.fmt()}] age histogram check", flush=True)
seed_all(0)
tok, model = load()
p = [x for x in data.problems() if x["pid"] == "math_134"]
for m in ("random", "neymankv_v1", "neymankv_v2", "simbin"):
    generate(model, tok, p, m, 256, slots=1, max_new=2100, verbose=False)
    print(m, {k: v for k, v in out.get(m, {}).items()}, flush=True)
json.dump(dict(edges=EDGES, hist=out, note="kept generated tokens per head, by age (tokens since generation); "
               "last 64 are the protected recent window"), open(os.path.join(ROOT, "runs", "poc", "age_hist.json"), "w"), indent=1)

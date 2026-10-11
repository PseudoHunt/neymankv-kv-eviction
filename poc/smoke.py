"""S0: 2 prompts x 7 methods, 512 tokens, B=256; then a batch-8 throughput probe."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from poc import clock, data  # noqa: E402
from poc.equiv import METHODS  # noqa: E402
from poc.run import generate, load  # noqa: E402

clock.start()
print(f"[GPU clock {clock.fmt()}] S0 smoke", flush=True)
seed_all(0)
tok, model = load()
ps = data.problems()
two = [ps[44], ps[45]]
rows = []
for m in METHODS:
    res, st, kl = generate(model, tok, two, m, 256, slots=2, max_new=512, log_pid=two[0]["pid"], verbose=False)
    rows.append(dict(method=m, n_gen=[r["n_gen"] for r in res], correct=[r["correct"] for r in res],
                     tok_s=round(st["tok_per_s"]), n_compress=st["n_compress"], kept=kl[:3]))
    print(rows[-1], flush=True)
for m in ("full", "neymankv_v1", "rkvhash_released"):
    probe = [p for p in ps if p["set"] == "math"][:8]
    res, st, _ = generate(model, tok, probe, m, 256, slots=8, max_new=512, verbose=False)
    print(f"throughput {m}: batch 8, {st['gen_tokens']} tok in {st['wall_s']:.0f}s = {st['tok_per_s']:.0f} tok/s, "
          f"compress {st['compress_s']:.1f}s, peak {st['peak_mem_gb']:.1f} GB", flush=True)
    rows.append(dict(method=m, throughput=st))
json.dump(rows, open(os.path.join(ROOT, "runs", "poc", "smoke.json"), "w"), indent=1, default=str)
print(f"[GPU clock {clock.fmt()}] S0 done", flush=True)

"""Run S1-S4 with up to W concurrent method processes (each batch 8). Enforces the 4h15m launch cutoff and the
300 tok/s rule (S1 aggregate) that cuts S3 to 25 problems per dataset."""
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402
from poc import clock  # noqa: E402

W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
GEN = os.path.join(ROOT, "runs", "poc", "gen")
LOG = os.path.join(ROOT, "runs", "poc", "schedule.log")
ALL7 = ["full", "random", "simbin", "rkvhash_released", "rkvhash_fixed", "neymankv_v1", "neymankv_noweight"]


def log(msg):
    line = f"[GPU clock {clock.fmt()}] {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def run_stage(name, jobs):
    log(f"stage {name}: {len(jobs)} jobs")
    t0 = time.time()
    pending, running = list(jobs), []
    while pending or running:
        running = [(j, p) for j, p in running if p.poll() is None]
        while pending and len(running) < W:
            if clock.elapsed() > clock.LIMIT_NEW_RUNS:
                log(f"past 4h15m: skipping {len(pending)} jobs: {pending}")
                pending = []
                break
            j = pending.pop(0)
            m, b, sets, n = j
            tag = f"{m}_B{b}_{sets}"
            p = subprocess.Popen([sys.executable, "-m", "poc.run", "--method", m, "--budget", str(b), "--sets", sets,
                                  "--n", str(n)], cwd=ROOT, stdout=open(os.path.join(ROOT, "runs", "poc", "logs", tag + ".log"), "w"),
                                 stderr=subprocess.STDOUT)
            running.append((j, p))
            log(f"launched {tag} n={n}")
        time.sleep(10)
    wall = time.time() - t0
    toks = 0
    for m, b, sets, n in jobs:
        f = os.path.join(GEN, f"{m}_B{b}_{sets.replace(',', '-')}.stats.json")
        if os.path.exists(f):
            toks += json.load(open(f))["gen_tokens"]
        else:
            log(f"MISSING result for {m} B={b} {sets}")
    agg = toks / wall
    log(f"stage {name} done: {toks} tokens in {wall:.0f}s = {agg:.0f} tok/s aggregate")
    return agg


def main():
    os.makedirs(os.path.join(ROOT, "runs", "poc", "logs"), exist_ok=True)
    clock.start()
    agg = run_stage("S1 B=256", [(m, 256, "math,gsm", 40) for m in ALL7])
    n3 = 40
    if agg < 300:
        n3 = 25
        log(f"S1 aggregate {agg:.0f} tok/s < 300: S3 cut to 25 problems per dataset")
    run_stage("S2 probe B=256", [(m, 256, "probe", 20) for m in
                                  ["random", "simbin", "rkvhash_fixed", "neymankv_v1", "neymankv_noweight"]])
    run_stage("S3 B=1024", [(m, 1024, "math,gsm", n3) for m in ALL7 if m != "full"])
    run_stage("S4 probe extras", [("neymankv_nofloor", 256, "probe", 20), ("rkvhash_released", 256, "probe", 20),
                                  ("full", 256, "probe", 20)])
    log("all stages done")


if __name__ == "__main__":
    main()

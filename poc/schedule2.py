"""Scheduler v2: adopts already-running poc.run jobs. Stages: S1 rest, S2 (forced-prefix probe + original probe
control, all 7 methods), S3 (B=1024), S4 (neymankv_nofloor forced probe). Up to W concurrent; 4h15m cutoff;
S1 aggregate < 300 tok/s cuts S3 to 25 problems per dataset."""
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
LOGD = os.path.join(ROOT, "runs", "poc", "logs")
LOG = os.path.join(ROOT, "runs", "poc", "schedule.log")
ALL7 = ["full", "random", "simbin", "rkvhash_released", "rkvhash_fixed", "neymankv_v1", "neymankv_noweight"]
S1_START = 1 * 3600 + 11 * 60 + 42  # GPU clock when S1 (valid tokenizer) was launched


def log(msg):
    line = f"[GPU clock {clock.fmt()}] {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def stats_path(m, b, sets):
    return os.path.join(GEN, f"{m}_B{b}_{sets.replace(',', '-')}.stats.json")


def running_jobs():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    jobs = set()
    for line in out:
        if "poc.run" in line and "--method" in line:
            a = line.split()
            jobs.add((a[a.index("--method") + 1], int(a[a.index("--budget") + 1]), a[a.index("--sets") + 1]))
    return jobs


def run_stage(name, jobs):
    log(f"stage {name}: {len(jobs)} jobs")
    pending = [j for j in jobs if not os.path.exists(stats_path(*j[:3]))]
    while True:
        run = running_jobs()
        pending = [j for j in pending if j[:3] not in run and not os.path.exists(stats_path(*j[:3]))]
        mine = [j for j in jobs if j[:3] in run]
        if not pending and not mine:
            break
        while pending and len(run) < W:
            if clock.elapsed() > clock.LIMIT_NEW_RUNS:
                log(f"past 4h15m: skipping {pending}")
                pending = []
                break
            m, b, sets, n = pending.pop(0)
            tag = f"{m}_B{b}_{sets}"
            subprocess.Popen([sys.executable, "-m", "poc.run", "--method", m, "--budget", str(b), "--sets", sets,
                              "--n", str(n)], cwd=ROOT, stdout=open(os.path.join(LOGD, tag + ".log"), "w"),
                             stderr=subprocess.STDOUT)
            run.add((m, b, sets))
            log(f"launched {tag} n={n}")
        time.sleep(10)
    toks = sum(json.load(open(stats_path(*j[:3])))["gen_tokens"] for j in jobs if os.path.exists(stats_path(*j[:3])))
    missing = [j for j in jobs if not os.path.exists(stats_path(*j[:3]))]
    log(f"stage {name} done: {toks} generated tokens; missing: {missing}")
    return toks


def main():
    os.makedirs(LOGD, exist_ok=True)
    toks = run_stage("S1 B=256", [(m, 256, "math,gsm", 40) for m in ALL7])
    agg = toks / max(1.0, clock.elapsed() - S1_START)
    n3 = 40 if agg >= 300 else 25
    log(f"S1 aggregate {agg:.0f} tok/s (4 concurrent processes x batch 8) -> S3 uses {n3} problems per dataset")
    run_stage("S2 probes B=256", [(m, 256, "forced", 20) for m in ALL7] + [(m, 256, "probe", 20) for m in ALL7])
    run_stage("S3 B=1024", [(m, 1024, "math,gsm", n3) for m in ALL7 if m != "full"])
    run_stage("S4 nofloor probe", [("neymankv_nofloor", 256, "forced", 20)])
    log("all stages done")


if __name__ == "__main__":
    main()

"""Follow-up job queue: W concurrent poc.run jobs (32 slots each), per-job wall time logged by phase.

  python -m poc.queue W phase:method:budget:sets:seed [...]
"""
import csv
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT  # noqa: E402
from poc import clock  # noqa: E402

clock.use("followup")
clock.start()
D = os.path.join(ROOT, "runs", "followup")
LOG, JOBS = os.path.join(D, "queue.log"), os.path.join(D, "jobs.csv")


def log(msg):
    line = f"[followup clock {clock.fmt()}] {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def main():
    W = int(sys.argv[1])
    jobs = [j.split(":") for j in sys.argv[2:]]
    pending, running = list(jobs), []
    while pending or running:
        for j, p, t0 in list(running):
            if p.poll() is not None:
                running.remove((j, p, t0))
                wall = time.time() - t0
                new = not os.path.exists(JOBS)
                with open(JOBS, "a") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["phase", "method", "budget", "sets", "seed", "exit", "wall_s", "end_clock"])
                    w.writerow(j + [p.returncode, round(wall), clock.fmt()])
                log(f"finished {':'.join(j)} exit={p.returncode} in {wall / 60:.1f} min")
        while pending and len(running) < W:
            j = pending.pop(0)
            phase, m, b, sets, seed = j
            tag = f"{phase}_{m}_B{b}_{sets.replace(',', '-')}_s{seed}"
            p = subprocess.Popen([sys.executable, "-m", "poc.run", "--method", m, "--budget", b, "--sets", sets, "--seed", seed,
                                  "--slots", "32", "--out", "followup", "--clock", "followup", "--tag", f"_{phase}"],
                                 cwd=ROOT, stdout=open(os.path.join(D, "logs", tag + ".log"), "w"), stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            running.append((j, p, time.time()))
            log(f"launched {tag}")
        time.sleep(10)
    log("queue done")


if __name__ == "__main__":
    main()

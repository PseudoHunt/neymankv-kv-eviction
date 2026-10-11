"""GPU clock: wall time since the first PoC GPU job (the GPU is dedicated, concurrent jobs count once)."""
import os
import time

from common import ROOT

START = os.path.join(ROOT, "runs", "poc", "clock_start")


def use(name: str):
    """Select runs/<name>/clock_start (the PoC and the follow-up keep separate GPU clocks)."""
    global START
    os.makedirs(os.path.join(ROOT, "runs", name), exist_ok=True)
    START = os.path.join(ROOT, "runs", name, "clock_start")
LIMIT_NEW_RUNS = 4 * 3600 + 15 * 60


def start():
    if not os.path.exists(START):
        open(START, "w").write(str(time.time()))


def elapsed() -> float:
    return time.time() - float(open(START).read()) if os.path.exists(START) else 0.0


def fmt(s=None):
    s = elapsed() if s is None else s
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m{int(s % 60):02d}s"

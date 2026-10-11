"""Shared utilities: seeding, determinism, provenance."""
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """Deterministic algorithms where they do not break SDPA. Returns a note for the report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return "torch.use_deterministic_algorithms(True, warn_only=True)"
    except Exception as e:  # pragma: no cover
        return f"deterministic algorithms unavailable: {e}"


def provenance() -> dict:
    def run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unavailable"

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": run(["git", "status", "--porcelain"]) != "",
        "gpu": gpu,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }


def write_provenance(path: str, extra: dict | None = None):
    p = provenance()
    if extra:
        p.update(extra)
    with open(path, "w") as f:
        json.dump(p, f, indent=1)


def provenance_md() -> str:
    p = provenance()
    return (f"<!-- provenance -->\n- git commit: `{p['git_commit']}` (dirty={p['git_dirty']})\n"
            f"- GPU: {p['gpu']}; python {p['python']}; torch {p['torch']}\n"
            f"- pip freeze: {len(p['pip_freeze'])} packages, full list in `runs/pip_freeze.txt`\n")

"""Phase G runner (kvpress venv).  /home/venv_kvpress/bin/python longdoc/run_g.py METHOD RATIO DATASET
DATASET in {ruler4k, ruler8k, loogle}. Query-agnostic: each context is compressed once without its questions, then all
of its questions are answered from the same compressed cache (kvpress pipeline protocol)."""
import hashlib
import json
import os
import sys
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kvpress  # noqa: E402,F401  (registers the pipeline, patches attention)
from longdoc.presses import install_bias_patch, make  # noqa: E402

MODEL = "Qwen/Qwen3-4B"
OUT = "/home/neymankv/runs/followup/G"
N_PER_TASK, N_LOOGLE_CTX, MAX_CTX = 20, 12, 30000


def data(name, tok, n_per_task=N_PER_TASK):
    if name.startswith("ruler"):
        df = load_dataset("simonjegou/ruler", {"ruler4k": "4096", "ruler8k": "8192"}[name], split="test").to_pandas()
        # 20 per task (random_state 0); smaller n takes the first n of those 20, so budgets share samples
        df = df.groupby("task", group_keys=False).apply(lambda x: x.sample(n=min(N_PER_TASK, len(x)), random_state=0).head(n_per_task))
    else:
        df = load_dataset("simonjegou/loogle", "shortdep_qa", split="test").to_pandas()
        ctx = df.drop_duplicates("context")[["context"]]
        ctx["n_tok"] = [len(tok(c)["input_ids"]) for c in ctx.context]
        keep = ctx[ctx.n_tok <= MAX_CTX].sample(n=N_LOOGLE_CTX, random_state=0).context
        df = df[df.context.isin(set(keep))]
    return df.reset_index(drop=True)


def main():
    method, ratio, dname = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    n_per_task = int(sys.argv[4]) if len(sys.argv) > 4 else N_PER_TASK
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{dname}_{method}_r{ratio}.jsonl")
    torch.manual_seed(0)
    pipe = pipeline("kv-press-text-generation", model=MODEL, device="cuda",
                    model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
    install_bias_patch()
    df = data(dname, pipe.tokenizer, n_per_task)
    press = make(method, ratio)
    t0, rows = time.time(), []
    for ctx, g in df.groupby("context", sort=False):
        res = pipe(ctx, questions=g.question.tolist(), answer_prefix=g.answer_prefix.iloc[0], press=press,
                   max_new_tokens=int(g.max_new_tokens.iloc[0]))
        for (_, r), a in zip(g.iterrows(), res["answers"]):
            ans = r.answer.tolist() if hasattr(r.answer, "tolist") else r.answer
            rows.append(dict(task=r.task, question=r.question, answer=ans, predicted_answer=a, context_hash=hashlib.sha1(ctx.encode()).hexdigest()[:12]))
    wall = time.time() - t0
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    json.dump(dict(method=method, ratio=ratio, dataset=dname, n=len(rows), n_contexts=df.context.nunique(), wall_s=wall,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30), open(out.replace(".jsonl", ".stats.json"), "w"))
    print(f"done {out}: {len(rows)} answers in {wall/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

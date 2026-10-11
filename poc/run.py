"""Run one method at one budget over a problem set with continuous batching.

  python -m poc.run --method neymankv_v1 --budget 256 --sets math,gsm [--n 40] [--max_new 3072]
"""
import argparse
import json
import os
import sys
import time
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from poc import clock, data, grade  # noqa: E402
from poc.engine import Engine, sample  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
THINK_END = 151649
OUT = os.path.join(ROOT, "runs", "poc")



def load_tokenizer(name):
    """transformers 5.x maps these repos' LlamaTokenizerFast to a rebuilt LlamaTokenizer that drops the byte-level
    pre-tokenizer (spaces vanish). Load tokenizer.json directly and refuse to run unless a round trip is exact."""
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(name)
    s = "Hello world, the answer is 42.\nNext line"
    assert tok.decode(tok(s, add_special_tokens=False)["input_ids"]) == s, f"tokenizer round trip failed for {name}"
    return tok

def load():
    tok = load_tokenizer(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    return tok, model


def generate(model, tok, probs, method, budget, slots=8, max_new=3072, seed=0, log_pid=None, policy_kw=None,
             verbose=True):
    prompts = {p["pid"]: tok(data.prompt(tok, p), add_special_tokens=False)["input_ids"] for p in probs}
    cap_prompt = max(len(v) for v in prompts.values()) + 8
    e = Engine(model, tok, method, budget, n_slots=slots, max_new=max_new, cap_prompt=cap_prompt, seed=seed,
               policy_kw=policy_kw)
    queue = list(probs)
    st = [None] * slots
    results, kept_log = [], []
    dev = e.dev
    tokens = torch.zeros(slots, dtype=torch.long, device=dev)
    positions = torch.zeros(slots, dtype=torch.long, device=dev)
    gens = [torch.Generator(device=dev) for _ in range(slots)]
    n_tok, t0 = 0, time.time()
    torch.cuda.reset_peak_memory_stats()

    def finish(s):
        x = st[s]
        text = tok.decode(x["gen"], skip_special_tokens=True)
        ans = grade.answer_part(text)
        pred = grade.last_boxed(ans)
        p = x["p"]
        results.append(dict(pid=p["pid"], set=p["set"], gold=p["gold"], pred=pred, correct=grade.correct(pred, p["gold"]),
                            n_gen=len(x["gen"]), hit_cap=len(x["gen"]) >= max_new, think_closed=THINK_END in x["gen"],
                            loop=grade.has_loop(x["gen"]), passcode_answer=data.CODE in ans and "</think>" in text,
                            passcode_any=data.CODE in text[len(x["p"].get("forced_prefix", "")):], prompt_len=x["P"], text=text))
        if verbose:
            r = results[-1]
            print(f"[{clock.fmt()}] {method} B={budget} {r['pid']}: n={r['n_gen']} ok={r['correct']} pred={str(r['pred'])[:30]!r}", flush=True)
        st[s] = None

    def refill(s):
        p = queue.pop(0)
        ids = prompts[p["pid"]]
        gens[s].manual_seed(seed * 1000003 + zlib.crc32(p["pid"].encode()))
        logits = e.prefill(s, ids)
        t = sample(logits[None], [gens[s]])
        forced = tok(p["forced_prefix"], add_special_tokens=False)["input_ids"] if p.get("forced_prefix") else []
        if forced:
            t = torch.tensor([forced[0]], device=dev)
        st[s] = dict(p=p, P=len(ids), gen=[int(t)], boxed_after=False, forced=forced[1:])
        tokens[s] = t[0]
        positions[s] = len(ids)

    for s in range(slots):
        if queue:
            refill(s)
    while any(x is not None for x in st):
        active = torch.tensor([x is not None for x in st], device=dev)
        logits = e.step(tokens, positions, active)
        nxt = sample(logits, gens)
        nl = nxt.tolist()
        for s in range(slots):
            x = st[s]
            if x is None:
                continue
            n_tok += 1
            positions[s] += 1
            n_in = len(x["gen"])                        # generated tokens now in the cache
            if n_in % e.interval == 0:
                Ln = e.compress(s, x["P"], n_in)
                if Ln is not None and x["p"]["pid"] == log_pid:
                    P = x["P"]
                    g = (e.pos[:, s, :, :Ln] >= P).sum(-1)
                    kept_log.append(dict(n_gen=n_in, cache_len=Ln, gen_kept_min=int(g.min()), gen_kept_max=int(g.max())))
            t = nl[s]
            if x["forced"]:                              # teacher-force the remaining prefix tokens
                t = x["forced"].pop(0)
            stop = t == tok.eos_token_id
            if not stop:
                x["gen"].append(t)
                tokens[s] = t
                if THINK_END in x["gen"][-1:] or (x.get("closed") and len(x["gen"]) % 8 == 0):
                    x["closed"] = True
                    tail = tok.decode(x["gen"][x["gen"].index(THINK_END):], skip_special_tokens=True)
                    if grade.last_boxed(tail) is not None:
                        stop = True
                stop = stop or len(x["gen"]) >= max_new
            if stop:
                finish(s)
                if queue:
                    refill(s)
    wall = time.time() - t0
    stats = dict(method=method, budget=budget, n_problems=len(results), gen_tokens=n_tok, wall_s=wall,
                 tok_per_s=n_tok / wall, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                 n_compress=e.n_compress, compress_s=e.compress_time, slots=slots, max_new=max_new)
    return results, stats, kept_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--sets", default="math,gsm")
    ap.add_argument("--n", type=int, default=40, help="problems per set")
    ap.add_argument("--max_new", type=int, default=3072)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    clock.start()
    print(f"[GPU clock {clock.fmt()}] start {a.method} B={a.budget} sets={a.sets} n={a.n}", flush=True)
    if clock.elapsed() > clock.LIMIT_NEW_RUNS and not a.force:
        print("past 4h15m: not launching new runs", flush=True)
        return
    seed_all(0)
    tok, model = load()
    sets = a.sets.split(",")
    probs = []
    for s in sets:
        probs += [p for p in data.problems() if p["set"] == s][:a.n]
    res, stats, kept = generate(model, tok, probs, a.method, a.budget, a.slots, a.max_new, log_pid=probs[0]["pid"])
    name = f"{a.method}_B{a.budget}_{'-'.join(sets)}{a.tag}"
    os.makedirs(os.path.join(OUT, "gen"), exist_ok=True)
    with open(os.path.join(OUT, "gen", name + ".jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    stats.update(name=name, gpu_clock_end=clock.fmt(), kept_log=kept, kept_log_pid=probs[0]["pid"])
    json.dump(stats, open(os.path.join(OUT, "gen", name + ".stats.json"), "w"), indent=1)
    print(f"[GPU clock {clock.fmt()}] done {name}: {stats['tok_per_s']:.0f} tok/s, acc "
          f"{sum(r['correct'] for r in res)}/{len(res)}", flush=True)


if __name__ == "__main__":
    main()

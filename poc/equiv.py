"""Equivalence checks before the PoC runs (brief section 3) + engine vs HF logits check."""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from poc import clock, data  # noqa: E402
from poc.engine import Engine  # noqa: E402
from poc.run import generate, load  # noqa: E402

METHODS = ["full", "random", "simbin", "rkvhash_released", "rkvhash_fixed", "neymankv_v1", "neymankv_noweight"]


def main():
    clock.start()
    print(f"[GPU clock {clock.fmt()}] equivalence checks", flush=True)
    seed_all(0)
    tok, model = load()
    probs = data.problems()
    out = {}

    # 0. engine logits vs HF eager forward, 50 teacher-forced tokens
    p = probs[40]
    ids = tok(data.prompt(tok, p), add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        ref_full = model.generate(torch.tensor([ids]).cuda(), do_sample=False, max_new_tokens=50)[0].tolist()
        out_ = model(torch.tensor([ids]).cuda(), use_cache=True)   # HF eager, incremental with its own cache
        pkv, inc = out_.past_key_values, [out_.logits[0, -1].float()]
        for t in range(len(ids), len(ref_full) - 1):
            out_ = model(torch.tensor([[ref_full[t]]]).cuda(), past_key_values=pkv, use_cache=True)
            pkv = out_.past_key_values
            inc.append(out_.logits[0, -1].float())
        ref = torch.stack(inc)
    e = Engine(model, tok, "full", 4096, n_slots=1, cap_prompt=len(ids) + 8, max_new=64)
    lg = [e.prefill(0, ids)]
    for t in range(len(ids), len(ref_full) - 1):
        lg.append(e.step(torch.tensor([ref_full[t]]).cuda(), torch.tensor([t]).cuda(), torch.tensor([True]).cuda())[0])
    lg = torch.stack(lg)
    r = ref
    out["engine_vs_hf"] = dict(max_abs_logit_diff=float((lg - r).abs().max()), argmax_agree=float((lg.argmax(-1) == r.argmax(-1)).float().mean()),
                               max_prob_diff=float((lg.softmax(-1) - r.softmax(-1)).abs().max()))
    print("engine vs HF eager:", out["engine_vs_hf"], flush=True)

    # 1. budget >= trace length: identical tokens to full
    three = [probs[0], probs[40], probs[41]]
    toks = {}
    for m in METHODS:
        res, _, _ = generate(model, tok, three, m, budget=4096, slots=3, max_new=300, verbose=False)
        toks[m] = {r["pid"]: r["text"] for r in res}
    same = {m: all(toks[m][k] == toks["full"][k] for k in toks["full"]) for m in METHODS}
    out["check1_same_tokens_as_full"] = same
    print("check 1:", same, flush=True)

    # 2. neymankv_v1 with one stratum, no floor == random in kept-set size per head at every compression
    logs = {}
    for m in ("random", "neymankv_onestratum"):
        _, _, kl = generate(model, tok, [probs[0]], m, budget=64, slots=1, max_new=640, log_pid=probs[0]["pid"], verbose=False)
        logs[m] = [(k["n_gen"], k["gen_kept_min"], k["gen_kept_max"]) for k in kl]
    n = min(len(logs["random"]), len(logs["neymankv_onestratum"]))
    # logged count = kept candidates (B=64) + the 64 protected recent tokens
    ok2 = n > 0 and logs["random"][:n] == logs["neymankv_onestratum"][:n] and all(a == b == 128 for _, a, b in logs["random"][:n])
    out["check2"] = dict(passed=ok2, compressions_compared=n, random=logs["random"][:n], onestratum=logs["neymankv_onestratum"][:n])
    print("check 2:", ok2, logs["random"][:3], flush=True)

    # 3. bias path: 4x duplicated cache (bias 0) vs one copy with bias log 4
    ids = ids[:40]
    T = len(ids)
    out["check3"] = {}
    model32 = None
    for mode in ("bf16", "fp32_attention", "fp32_model"):
        if mode == "fp32_model":
            from transformers import AutoModelForCausalLM
            from poc.run import MODEL
            model32 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
        mdl = model32 if mode == "fp32_model" else model
        kw = dict(n_slots=1, cap_prompt=4 * T + 8, max_new=8, attn_fp32=(mode != "bf16"))
        ea, eb = Engine(mdl, tok, "full", 4096, **kw), Engine(mdl, tok, "full", 4096, **kw)
        ea.prefill(0, ids)
        eb.prefill(0, ids)
        for a in ("K", "V", "pos"):
            x = getattr(ea, a)[:, 0, :, :T].clone()
            getattr(ea, a)[:, 0, :, :4 * T] = x.repeat_interleave(4, dim=2)
        ea.bias[:, 0, :, :4 * T] = 0.0
        ea.L[0] = 4 * T
        eb.bias[:, 0, :, :T] = math.log(4)
        nxt, pos = torch.tensor([ids[-1]]).cuda(), torch.tensor([T]).cuda()
        la = ea.step(nxt, pos, torch.tensor([True]).cuda())[0]
        lb = eb.step(nxt, pos, torch.tensor([True]).cuda())[0]
        rel = float((la - lb).abs().max() / la.abs().max())
        out["check3"][mode] = dict(max_abs_logit_diff=float((la - lb).abs().max()), rel_diff=rel, passed=rel < 1e-3,
                                   prob_max_diff=float((la.softmax(-1) - lb.softmax(-1)).abs().max()))
    print("check 3:", out["check3"], flush=True)
    out["gpu_clock"] = clock.fmt()
    json.dump(out, open(os.path.join(ROOT, "runs", "poc", "equivalence.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

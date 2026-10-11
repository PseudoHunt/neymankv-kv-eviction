"""Phase A: generate continuations/traces and capture Q/K/V with hooks.

  python -m capture.run_capture docs     # Qwen3-4B: 20 GovReport + 5 needle prompts
  python -m capture.run_capture reason   # R1-Distill-Qwen-7B: 15 MATH500 traces
"""
import json
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture.hooks import Capturer, rope_theta  # noqa: E402
from common import ROOT, seed_all, set_deterministic, write_provenance  # noqa: E402

OUT = os.path.join(ROOT, "data", "captures")
N_FUT, N_OBS = 512, 256
CAPTURE_MODEL = "Qwen/Qwen3-4B"
REASON_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SUMMARY_Q = "Question: Summarize the main findings and recommendations of the report above."
NEEDLE_Q = "Question: What is the secret passcode stated in the document above? Reply with the passcode only."



def load_tokenizer(name):
    """transformers 5.x maps these repos' LlamaTokenizerFast to a rebuilt LlamaTokenizer that drops the byte-level
    pre-tokenizer (spaces vanish). Load tokenizer.json directly and refuse to run unless a round trip is exact."""
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(name)
    s = "Hello world, the answer is 42.\nNext line"
    assert tok.decode(tok(s, add_special_tokens=False)["input_ids"]) == s, f"tokenizer round trip failed for {name}"
    return tok

def load(name):
    tok = load_tokenizer(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    return tok, model


def kept_layers(model):
    return list(range(0, model.config.num_hidden_layers, 4))


def capture_and_save(model, tok, ids: torch.Tensor, cut: int, meta: dict, pid: str):
    T = ids.shape[1]
    q_hi = min(cut + N_FUT, T)
    layers = kept_layers(model)
    with torch.no_grad(), Capturer(model, layers, cut=cut, q_lo=cut - N_OBS, q_hi=q_hi) as cap:
        model.model(input_ids=ids.cuda(), position_ids=torch.arange(T, device="cuda")[None])
    tensors = {}
    for l in layers:
        for name, x in cap.data[l].items():
            assert torch.isfinite(x).all(), (pid, l, name)
            tensors[f"L{l}.{name}"] = x
    tensors["positions"] = torch.arange(cut, dtype=torch.int32)
    tensors["q_positions"] = torch.arange(cut - N_OBS, q_hi, dtype=torch.int32)
    cfg = model.config
    meta.update(dict(id=pid, model=cfg._name_or_path, cut=cut, q_lo=cut - N_OBS, q_hi=q_hi, n_future=q_hi - cut,
                     n_observed=N_OBS, seq_len=T, layers=layers, n_q_heads=cfg.num_attention_heads,
                     n_kv_heads=cfg.num_key_value_heads, head_dim=tensors[f"L{layers[0]}.k_pre"].shape[-1],
                     rope_theta=rope_theta(cfg), token_ids=ids[0].tolist()))
    save_file(tensors, os.path.join(OUT, f"{pid}.safetensors"))
    with open(os.path.join(OUT, f"{pid}.json"), "w") as f:
        json.dump(meta, f)


def cut_index(tok, text: str, marker: str) -> tuple[torch.Tensor, int]:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    qpos = text.rindex(marker)
    cut = next(i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= qpos)
    return torch.tensor([enc["input_ids"]]), cut


def chat(tok, user: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def extend_greedy(model, ids, cut):
    """Greedy-generate so that at least N_FUT positions follow the cut."""
    n_gen = max(1, cut + N_FUT - ids.shape[1])
    with torch.no_grad():
        out = model.generate(ids.cuda(), do_sample=False, max_new_tokens=n_gen, min_new_tokens=n_gen,
                             temperature=None, top_p=None, top_k=None)
    return out.cpu(), n_gen


def run_docs():
    tok, model = load(CAPTURE_MODEL)
    ds = load_dataset("ccdv/govreport-summarization", split="test")
    lens = [len(tok(r, add_special_tokens=False)["input_ids"]) for r in ds["report"]]
    good = [i for i, n in enumerate(lens) if 4000 <= n <= 7600]
    print(f"{len(good)} GovReport test docs in [4000, 7600] tokens", flush=True)
    for j, i in enumerate(good[:20]):
        pid = f"doc_{j:02d}"
        if os.path.exists(os.path.join(OUT, f"{pid}.json")):
            continue
        seed_all(1000 + j)
        text = chat(tok, ds[i]["report"].strip() + "\n\n" + SUMMARY_Q)
        ids, cut = cut_index(tok, text, SUMMARY_Q)
        full, n_gen = extend_greedy(model, ids, cut)
        meta = dict(prompt_type="long_doc", source=f"ccdv/govreport-summarization:test[{i}]", n_generated=n_gen,
                    continuation=tok.decode(full[0, ids.shape[1]:ids.shape[1] + 120]))
        capture_and_save(model, tok, full, cut, meta, pid)
        print(pid, "len", full.shape[1], "cut", cut, flush=True)
    # needles: filler from other GovReport docs
    rng = np.random.default_rng(7)
    filler_docs = [ds[i]["report"] for i in good[20:60]]
    for j in range(5):
        pid = f"needle_{j:02d}"
        if os.path.exists(os.path.join(OUT, f"{pid}.json")):
            continue
        seed_all(2000 + j)
        target = int(rng.integers(4000, 7500))
        sents = " ".join(filler_docs[2 * j:2 * j + 2]).split(". ")
        filler, n = [], 0
        for s in sents:
            filler.append(s)
            n += len(tok(s, add_special_tokens=False)["input_ids"]) + 1
            if n >= target:
                break
        code = f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 4))}-{rng.integers(1000, 9999)}"
        depth = float(rng.uniform(0.1, 0.9))
        filler.insert(int(depth * len(filler)), f"The secret passcode is {code}. Remember this passcode")
        text = chat(tok, ". ".join(filler) + ".\n\n" + NEEDLE_Q)
        ids, cut = cut_index(tok, text, NEEDLE_Q)
        full, n_gen = extend_greedy(model, ids, cut)
        ans = tok.decode(full[0, ids.shape[1]:ids.shape[1] + 30], skip_special_tokens=True)
        meta = dict(prompt_type="long_doc", subtype="needle", source="synthetic needle in GovReport filler",
                    passcode=code, depth=depth, answer=ans, answer_correct=code in ans, n_generated=n_gen)
        capture_and_save(model, tok, full, cut, meta, pid)
        print(pid, "len", full.shape[1], "cut", cut, "correct", code in ans, flush=True)


def run_reason():
    tok, model = load(REASON_MODEL)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    cands = [i for i in range(len(ds)) if ds[i]["level"] >= 4]
    j = 0
    log = []
    for i in cands:
        if j >= 15:
            break
        pid = f"reason_{j:02d}"
        if os.path.exists(os.path.join(OUT, f"{pid}.json")):
            j += 1
            continue
        seed_all(3000 + i)
        user = ds[i]["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids.cuda(), do_sample=True, temperature=0.6, top_p=0.95, top_k=None,
                                 max_new_tokens=4096).cpu()
        gen = out[0, ids.shape[1]:]
        if gen[-1].item() == tok.eos_token_id:
            gen = gen[:-1]
        trace_len = len(gen)
        rate = trace_len / (time.time() - t0)
        cut = ids.shape[1] + int(0.6 * trace_len)
        full = torch.cat([ids[0], gen])[None]
        ok = full.shape[1] - cut >= N_FUT
        log.append(dict(problem=i, trace_len=trace_len, used=ok))
        print(f"problem {i}: trace {trace_len} tok at {rate:.1f} tok/s, used={ok}", flush=True)
        if not ok:
            continue  # too short to give 512 future queries after a 60% cut
        meta = dict(prompt_type="reasoning", source=f"HuggingFaceH4/MATH-500:test[{i}]", unique_id=ds[i]["unique_id"],
                    level=ds[i]["level"], trace_len=trace_len, prompt_len=ids.shape[1], hit_cap=trace_len >= 4095,
                    gold=ds[i]["answer"], trace_tail=tok.decode(gen[-200:]))
        capture_and_save(model, tok, full, cut, meta, pid)
        j += 1
    with open(os.path.join(ROOT, "runs", "reason_generation_log.json"), "a") as f:
        json.dump(log, f)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    note = set_deterministic()
    print(note, flush=True)
    t0 = time.time()
    {"docs": run_docs, "reason": run_reason}[sys.argv[1]]()
    write_provenance(os.path.join(ROOT, "runs", f"provenance_capture_{sys.argv[1]}.json"),
                     {"determinism": note, "gpu_seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s", flush=True)

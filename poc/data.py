"""Fixed problem lists (seed 0) and prompts."""
import numpy as np
from datasets import load_dataset

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE = "7391-KQ"


def problems():
    rng = np.random.default_rng(0)
    m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for lvl in range(1, 6):
        idx = [i for i in range(len(m)) if m[i]["level"] == lvl]
        for i in sorted(rng.choice(idx, 8, replace=False).tolist()):
            out.append(dict(pid=f"math_{i}", set="math", question=m[i]["problem"], gold=m[i]["answer"], level=lvl))
    g = load_dataset("openai/gsm8k", "main", split="test")
    pick = rng.choice(len(g), 60, replace=False).tolist()
    for i in pick[:40]:
        out.append(dict(pid=f"gsm_{i}", set="gsm", question=g[i]["question"],
                        gold=g[i]["answer"].split("####")[-1].strip().replace(",", "")))
    for i in pick[40:60]:
        q = f"The access code is {CODE}. Remember it.\n" + g[i]["question"]
        out.append(dict(pid=f"probe_{i}", set="probe", question=q, gold=g[i]["answer"].split("####")[-1].strip().replace(",", ""),
                        suffix_extra=" After the answer, restate the access code."))
    return out


def prompt(tok, p):
    user = p["question"] + SUFFIX + p.get("suffix_extra", "")
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

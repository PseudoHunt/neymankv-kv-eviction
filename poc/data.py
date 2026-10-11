"""Fixed problem lists (seed 0) and prompts."""
import numpy as np
from datasets import load_dataset

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE = "7391-KQ"
# The chat template already ends with "<think>\n" (prompt, protected), so only this note is forced as generated text.
# "Okay," keeps the model reasoning: without it the model closes </think> right after the note (6/8 problems),
# traces stay too short for B=256 to evict anything, and even full-cache restates the code 0-1/8 (variant test).
FORCED_PREFIX = f"Note to self: the access code is {CODE}.\n\nOkay,"


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
    # forced-prefix probe: the code is stated only in teacher-forced *generated* tokens (evictable region);
    # the recall instruction stays in the (protected) prompt.
    for i in pick[40:60]:
        out.append(dict(pid=f"forced_{i}", set="forced", question=g[i]["question"],
                        gold=g[i]["answer"].split("####")[-1].strip().replace(",", ""),
                        suffix_extra=" After the answer, restate the access code.",
                        forced_prefix=FORCED_PREFIX))
    # Follow-up (Phase A) extensions, disjoint from everything above: 80 more MATH500 (16 per level, seed 1) as set
    # "math_x", and 40 more GSM8K as set "gsm_x". The PoC's original lists keep their set names "math"/"gsm".
    rng1 = np.random.default_rng(1)
    used_m = {int(p["pid"].split("_")[1]) for p in out if p["set"] == "math"}
    for lvl in range(1, 6):
        idx = [i for i in range(len(m)) if m[i]["level"] == lvl and i not in used_m]
        for i in sorted(rng1.choice(idx, 16, replace=False).tolist()):
            out.append(dict(pid=f"math_{i}", set="math_x", question=m[i]["problem"], gold=m[i]["answer"], level=lvl))
    rest = [i for i in range(len(g)) if i not in set(pick)]
    for i in rng1.choice(rest, 40, replace=False).tolist():
        out.append(dict(pid=f"gsm_{i}", set="gsm_x", question=g[i]["question"],
                        gold=g[i]["answer"].split("####")[-1].strip().replace(",", "")))
    return out


def prompt(tok, p):
    user = p["question"] + SUFFIX + p.get("suffix_extra", "")
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

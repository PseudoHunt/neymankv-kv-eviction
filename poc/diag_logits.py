"""Where does the engine-vs-HF logit gap come from? Per-layer residual-stream diffs over 50 teacher-forced steps."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, seed_all  # noqa: E402
from poc import clock, data  # noqa: E402
from poc.engine import Engine  # noqa: E402
from poc.run import load  # noqa: E402


def main():
    print(f"[GPU clock {clock.fmt()}] logit-gap diagnostic", flush=True)
    seed_all(0)
    tok, model = load()
    p = data.problems()[40]
    ids = tok(data.prompt(tok, p), add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        full = model.generate(torch.tensor([ids]).cuda(), do_sample=False, max_new_tokens=50)[0].tolist()
        hf_h, hooks = [], []
        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(lambda m, i, o: hf_h.append((o[0] if isinstance(o, tuple) else o)[:, -1].float().clone())))
        out = model(torch.tensor([ids]).cuda(), use_cache=True)
        pkv, hf_logits = out.past_key_values, [out.logits[0, -1].float()]
        for t in range(len(ids), len(full) - 1):
            out = model(torch.tensor([[full[t]]]).cuda(), past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            hf_logits.append(out.logits[0, -1].float())
        for h in hooks:
            h.remove()
    nL = model.config.num_hidden_layers
    hf_h = torch.stack(hf_h).view(len(full) - len(ids), nL, -1)                 # [steps, layers, D]
    hf_logits = torch.stack(hf_logits)
    res = {}
    for variant in ("engine_grouped_gemm", "engine_hf_shape_gemm"):
        e = Engine(model, tok, "full", 4096, n_slots=1, cap_prompt=len(ids) + 8, max_new=64)
        e.hf_shape_attn = variant == "engine_hf_shape_gemm"
        lg, hs = [e.prefill(0, ids)], []
        for t in range(len(ids), len(full) - 1):
            e.trace = []
            lg.append(e.step(torch.tensor([full[t]]).cuda(), torch.tensor([t]).cuda(), torch.tensor([True]).cuda())[0])
            hs.append(torch.stack([x[0, 0].float() for x in e.trace]))
        lg = torch.stack(lg)
        hs = torch.stack(hs)                                                        # [steps-1, layers, D] (decode steps)
        dh = (hs - hf_h[1:]).abs().amax(-1)                                         # per decode step, per layer
        rel = dh / hf_h[1:].abs().amax(-1)
        dl = (lg - hf_logits).abs().amax(-1)
        worst = int(dl.argmax())
        first_layer = [int((row > 0).nonzero()[0]) if (row > 0).any() else None for row in dh]
        res[variant] = dict(max_logit_diff=float(dl.max()), worst_step=worst, prefill_logit_diff=float(dl[0]),
                            argmax_agree=float((lg.argmax(-1) == hf_logits.argmax(-1)).float().mean()),
                            first_differing_layer_per_step=first_layer[:10],
                            frac_steps_with_any_hidden_diff=float(((dh > 0).any(-1)).float().mean()),
                            max_rel_hidden_diff_by_layer=[round(float(x), 5) for x in rel.amax(0)],
                            worst_step_top_logit=float(hf_logits[worst].max()))
        print(variant, {k: v for k, v in res[variant].items() if k != "max_rel_hidden_diff_by_layer"}, flush=True)
        print("  max rel hidden diff by layer:", res[variant]["max_rel_hidden_diff_by_layer"], flush=True)
    json.dump(res, open(os.path.join(ROOT, "runs", "poc", "logit_gap_diagnostic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

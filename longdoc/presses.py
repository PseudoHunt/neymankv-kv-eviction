"""Phase G presses for kvpress (query-agnostic context compression, reused across questions).

v1_prop        : pre-RoPE SimHash strata per KV head (b bits, fixed-seed projection), proportional allocation with a
                 1-slot floor + Webster rounding, uniform random within strata, logit bias log(N_h / n_h).
v1_prop_noweight: same selection, no bias.
random_rw      : uniform random (as kvpress RandomPress) + constant bias log(N / n) (control: reweighting without strata).
The bias is stored on the attention module and added by a wrapper around the attention function for every query that
comes after the context prefill (question prefill and decoding)."""
import math
import os
import sys
from dataclasses import dataclass, field

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alloc import batched as AB  # noqa: E402
from kvpress.presses.base_press import BasePress  # noqa: E402


def _bias_wrap(func):
    def wrapper(module, query, key, value, attention_mask, dropout=0.0, **kwargs):
        if query.shape[2] == key.shape[2]:                      # context prefill: a new context starts
            module.kv_logit_bias = None
            return func(module, query, key, value, attention_mask, dropout, **kwargs)
        b = getattr(module, "kv_logit_bias", None)
        if b is None:
            return func(module, query, key, value, attention_mask, dropout, **kwargs)
        bsz, hq, ql, kl = query.shape[0], query.shape[1], query.shape[2], key.shape[2]
        n = b.shape[-1]
        full = torch.zeros(bsz, b.shape[1], kl, device=query.device, dtype=torch.float32)
        full[..., :n] = b
        full = full.repeat_interleave(hq // b.shape[1], dim=1)[:, :, None, :]            # [b, Hq, 1, kl]
        i = torch.arange(ql, device=query.device)[:, None]
        j = torch.arange(kl, device=query.device)[None, :]
        causal = torch.where(j <= kl - ql + i, 0.0, float("-inf"))                       # [ql, kl]
        mask = (full + causal).to(query.dtype)
        kwargs.pop("is_causal", None)
        return func(module, query, key, value, mask, dropout, **kwargs)
    return wrapper


def install_bias_patch():
    for name in list(ALL_ATTENTION_FUNCTIONS.keys()):
        if not getattr(ALL_ATTENTION_FUNCTIONS[name], "_bias_patched", False):
            w = _bias_wrap(ALL_ATTENTION_FUNCTIONS[name])
            w._bias_patched = True
            ALL_ATTENTION_FUNCTIONS[name] = w


def _proj(d, bits, seed=1234):
    g = torch.Generator().manual_seed(seed + bits)
    return torch.randn(d, bits, generator=g)


def choose_bits(B):
    target = math.log2(max(B / 4, 1))
    return min((4, 6, 8), key=lambda b: (abs(b - target), b))


@dataclass
class StratPress(BasePress):
    compression_ratio: float = 0.0
    reweight: bool = True
    strata: bool = True
    seed: int = 0
    _gen: dict = field(default_factory=dict, repr=False)

    def _rand(self, shape, device):
        g = self._gen.get(device)
        if g is None:
            g = self._gen[device] = torch.Generator(device=device).manual_seed(self.seed)
        return torch.rand(*shape, generator=g, device=device)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        bsz, H, L, d = keys.shape
        n_kept = int(L * (1 - self.compression_ratio))
        R = bsz * H
        if self.strata:
            kp = module.k_proj(hidden_states).view(bsz, L, H, d)
            if hasattr(module, "k_norm"):
                kp = module.k_norm(kp)
            kp = kp.transpose(1, 2).float()                                              # pre-RoPE [b, H, L, d]
            bits = choose_bits(n_kept)
            bitsv = (kp @ _proj(d, bits).to(kp.device)) > 0
            lab = (bitsv.long() * (2 ** torch.arange(bits, device=kp.device))).sum(-1).reshape(R, L)
            Hs = 2 ** bits
        else:
            lab = torch.zeros(R, L, dtype=torch.long, device=keys.device)
            Hs = 1
        present = torch.zeros(R, Hs, dtype=torch.long, device=keys.device).scatter_add_(1, lab, torch.ones_like(lab))
        Bv = torch.full((R,), n_kept, device=keys.device)
        floor = 1 if self.strata else 0
        shares = AB.waterfill(present.double(), present, Bv, floor=floor)
        n = AB.webster(shares, present, Bv, floor=floor)
        mask = AB.select_bottom(lab, self._rand((R, L), keys.device).double(), n)
        idx = torch.arange(L, device=keys.device).expand(R, L)[mask].view(R, n_kept)
        if self.reweight:
            bias = torch.log(present.gather(1, lab).double() / n.gather(1, lab).clamp_min(1).double()).float()
            bias = bias[mask].view(R, n_kept)
            module.kv_logit_bias = bias.view(bsz, H, n_kept)
        else:
            module.kv_logit_bias = None
        g = idx.view(bsz, H, n_kept, 1).expand(-1, -1, -1, d)
        return keys.gather(2, g).contiguous(), values.gather(2, g).contiguous()


def make(name, ratio):
    from kvpress import AdaKVPress, ExpectedAttentionPress, KVzipPress, RandomPress
    return {
        "full": lambda: None,
        "random": lambda: RandomPress(compression_ratio=ratio, seed=0),
        "expected_attention": lambda: AdaKVPress(ExpectedAttentionPress(compression_ratio=ratio, epsilon=1e-2)),
        "kvzip": lambda: KVzipPress(compression_ratio=ratio),
        "v1_prop": lambda: StratPress(compression_ratio=ratio, reweight=True),
        "v1_prop_noweight": lambda: StratPress(compression_ratio=ratio, reweight=False),
        "random_rw": lambda: StratPress(compression_ratio=ratio, reweight=True, strata=False),
    }[name]()

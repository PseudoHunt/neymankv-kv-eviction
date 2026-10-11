"""Forward hooks that capture pre/post-RoPE Q/K and V from HF Qwen2/Qwen3 attention layers.

Pre-RoPE: output of q_norm/k_norm (Qwen3) or q_proj/k_proj reshaped (Qwen2).
Post-RoPE + values: a registered attention function that sees exactly what attention sees."""
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

_ACTIVE = []  # stack of active Capturer objects


def _capture_attention(module, query, key, value, attention_mask, **kw):
    for cap in _ACTIVE:
        cap._on_attention(module, query, key, value)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register("capture_sdpa", _capture_attention)


def rope_theta(config) -> float:
    rp = getattr(config, "rope_parameters", None) or {}
    if isinstance(rp, dict) and "rope_theta" in rp:
        assert rp.get("rope_type", "default") == "default", f"unsupported rope scaling {rp}"
        return float(rp["rope_theta"])
    return float(config.rope_theta)


class Capturer:
    """Context manager. Keys/values kept for positions [0, cut); queries for [q_lo, q_hi)."""

    def __init__(self, model, layers, cut: int, q_lo: int, q_hi: int, dtype=torch.float16):
        self.model, self.layers, self.cut, self.q_lo, self.q_hi, self.dtype = model, set(layers), cut, q_lo, q_hi, dtype
        self.data = {l: {} for l in self.layers}
        self.handles = []
        cfg = model.config
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    def _store(self, layer, name, x):  # x: [B, H, T, D]
        x = x[0]
        if name.startswith("q"):
            x = x[:, self.q_lo:self.q_hi]
        else:
            x = x[:, :self.cut]
        self.data[layer][name] = x.detach().to(self.dtype).cpu().contiguous()

    def _pre_hook(self, layer, name):
        def fn(mod, inp, out):
            x = out
            if x.dim() == 3:  # q_proj/k_proj output [B, T, H*D]
                x = x.view(x.shape[0], x.shape[1], -1, self.hd)
            self._store(layer, name, x.transpose(1, 2))
        return fn

    def _on_attention(self, module, query, key, value):
        l = module.layer_idx
        if l in self.layers:
            self._store(l, "q_post", query)
            self._store(l, "k_post", key)
            self._store(l, "v", value)

    def __enter__(self):
        for l, layer in enumerate(self.model.model.layers):
            if l not in self.layers:
                continue
            a = layer.self_attn
            qn, kn = (a.q_norm, a.k_norm) if hasattr(a, "q_norm") else (a.q_proj, a.k_proj)
            self.handles.append(qn.register_forward_hook(self._pre_hook(l, "q_pre")))
            self.handles.append(kn.register_forward_hook(self._pre_hook(l, "k_pre")))
        self._prev_impl = self.model.config._attn_implementation
        self.model.config._attn_implementation = "capture_sdpa"
        for m in self.model.modules():  # sub-configs share the object in Qwen2/3, but be safe
            if hasattr(m, "config") and m.config is not self.model.config and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "capture_sdpa"
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.remove(self)
        for h in self.handles:
            h.remove()
        self.model.config._attn_implementation = self._prev_impl
        missing = [(l, n) for l in self.layers for n in ("q_pre", "k_pre", "q_post", "k_post", "v")
                   if n not in self.data[l]]
        assert not missing or exc[0] is not None, f"capture missed {missing[:5]}"

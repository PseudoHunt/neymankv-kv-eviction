"""Decode-time KV eviction engine for Qwen2-family models (R1-Distill-Qwen-1.5B).

Own decode loop over the HF modules, own KV cache with a per-token additive logit bias.
Cache arrays are [layer, slot, kv_head, capacity(, d)]. Keys are stored post-RoPE at their original
position, so eviction never renumbers positions. Attention is eager: bf16 QK^T, fp32 softmax with the
bias added, bf16 PV (HF eager semantics). Slots are refilled as soon as a sequence finishes."""
import math
import time

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from poc.policies import POLICIES

NEG = float("-inf")
WINDOW_Q = 8  # last queries kept for the R-KVHash attention term


class Engine:
    def __init__(self, model, tok, method: str, budget: int, n_slots: int = 8, max_new: int = 3072,
                 interval: int = 64, recent: int = 64, cap_prompt: int = 512, seed: int = 0, policy_kw=None,
                 attn_fp32: bool = False):
        self.attn_fp32 = attn_fp32
        self.hf_shape_attn = False   # diagnostic: HF eager GEMM shapes (repeat_kv to all Q heads)
        self.trace = None            # diagnostic: list collecting the residual stream after each layer  # fp32 attention (scores + PV), used only to isolate the bias path in check 3
        self.m, self.tok = model, tok
        cfg = model.config
        self.nL, self.Hq, self.Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = cfg.hidden_size // cfg.num_attention_heads
        self.G = self.Hq // self.Hkv
        self.S, self.max_new, self.interval, self.recent, self.budget = n_slots, max_new, interval, recent, budget
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.policy = POLICIES[method](self, **(policy_kw or {}))
        self.method = method
        if self.policy.evicts:
            cap = cap_prompt + budget + interval + recent + 8
        else:
            cap = cap_prompt + max_new + 8
        self.cap = cap
        z = dict(device=self.dev)
        self.K = torch.zeros(self.nL, self.S, self.Hkv, cap, self.d, dtype=self.dtype, **z)
        self.V = torch.zeros_like(self.K)
        self.bias = torch.full((self.nL, self.S, self.Hkv, cap), NEG, dtype=torch.float32, **z)
        self.pos = torch.full((self.nL, self.S, self.Hkv, cap), -1, dtype=torch.long, **z)
        self.meta = {k: torch.zeros(self.nL, self.S, self.Hkv, cap, dtype=dt, **z) for k, dt in self.policy.meta_fields.items()}
        self.qwin = torch.zeros(self.nL, self.S, self.Hq, WINDOW_Q, self.d, dtype=self.dtype, **z)
        self.L = torch.zeros(self.S, dtype=torch.long, **z)
        self.gen_rng = torch.Generator(device=self.dev).manual_seed(seed + 99991)  # eviction randomness
        self.seed = seed
        self.layers = model.model.layers
        self.scale = 1.0 / math.sqrt(self.d)
        self.n_compress = 0
        self.compress_time = 0.0

    # ---------------------------------------------------------------- model pieces
    def _qkv(self, attn, x, B, T):
        q = attn.q_proj(x).view(B, T, self.Hq, self.d).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, self.Hkv, self.d).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, self.Hkv, self.d).transpose(1, 2)
        return q, k, v

    @torch.no_grad()
    def prefill(self, s: int, ids: list[int]):
        """Run one prompt into slot s. Returns last-position logits [V]."""
        T = len(ids)
        assert T + 8 < self.cap
        x = self.m.model.embed_tokens(torch.tensor([ids], device=self.dev))
        pos = torch.arange(T, device=self.dev)[None]
        cos, sin = self.m.model.rotary_emb(x, pos)
        for l, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q, k_pre, v = self._qkv(attn, h, 1, T)
            q, k = apply_rotary_pos_emb(q, k_pre, cos, sin)
            # SDPA computes QK^T in fp32 internally. HF "eager" rounds the scores to bf16 before the softmax, which on
            # this model's large attention logits corrupts copying (e.g. "80 / 2 = 44"); see runs/poc/eager_bug.md.
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + attn.o_proj(o.transpose(1, 2).reshape(1, T, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            self.K[l, s, :, :T] = k[0]
            self.V[l, s, :, :T] = v[0]
            self.bias[l, s, :, :] = NEG
            self.bias[l, s, :, :T] = 0.0
            self.pos[l, s, :, :T] = torch.arange(T, device=self.dev)
            w = min(T, WINDOW_Q)
            self.qwin[l, s].zero_()
            self.qwin[l, s, :, WINDOW_Q - w:] = q[0, :, T - w:]
            self.policy.on_prefill(l, s, k_pre[0], k[0])
        self.L[s] = T
        x = self.m.model.norm(x[:, -1])
        return self.m.lm_head(x)[0].float()

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, positions: torch.Tensor, active: torch.Tensor):
        """One decode step for all slots. tokens/positions [S]. Inactive slots compute but do not advance."""
        S = self.S
        x = self.m.model.embed_tokens(tokens[:, None])
        cos, sin = self.m.model.rotary_emb(x, positions[:, None])
        ar = torch.arange(S, device=self.dev)
        idx = self.L
        Lm = int(self.L.max()) + 1
        for l, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q, k_pre, v = self._qkv(attn, h, S, 1)
            q, k = apply_rotary_pos_emb(q, k_pre, cos, sin)
            self.K[l, ar, :, idx] = k[:, :, 0]
            self.V[l, ar, :, idx] = v[:, :, 0]
            self.bias[l, ar, :, idx] = 0.0
            self.pos[l, ar, :, idx] = positions[:, None]
            self.qwin[l] = torch.roll(self.qwin[l], -1, dims=2)
            self.qwin[l, :, :, -1] = q[:, :, 0]
            self.policy.on_insert(l, ar, idx, k_pre[:, :, 0], k[:, :, 0])
            Kl, Vl = self.K[l, :, :, :Lm], self.V[l, :, :, :Lm]
            if self.hf_shape_attn:
                kr, vr = Kl.repeat_interleave(self.G, 1), Vl.repeat_interleave(self.G, 1)       # [S, Hq, Lm, d]
                bq = self.bias[l, :, :, :Lm].repeat_interleave(self.G, 1)[:, :, None, :]
                sc = (torch.matmul(q, kr.transpose(-1, -2)) * self.scale).float() + bq
                o = torch.matmul(torch.softmax(sc, dim=-1).to(self.dtype), vr).transpose(1, 2).reshape(S, 1, -1)
                x = x + attn.o_proj(o)
                x = x + layer.mlp(layer.post_attention_layernorm(x))
                if self.trace is not None:
                    self.trace.append(x.detach().clone())
                continue
            if self.attn_fp32:
                qg = q.view(S, self.Hkv, self.G, self.d)
                sc = torch.matmul(qg.float(), Kl.float().transpose(-1, -2)) * self.scale + self.bias[l, :, :, None, :Lm]
                o = torch.matmul(torch.softmax(sc, dim=-1), Vl.float()).to(self.dtype)
            else:
                # SDPA with the per-token logit bias as an additive float mask (per KV head, shared by its Q heads)
                mask = self.bias[l, :, :, None, :Lm].repeat_interleave(self.G, dim=1).to(self.dtype)
                o = F.scaled_dot_product_attention(q, Kl, Vl, attn_mask=mask, enable_gqa=True)   # [S, Hq, 1, d]
                o = o.transpose(1, 2)
            o = o.reshape(S, 1, self.Hq * self.d)
            x = x + attn.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            if self.trace is not None:
                self.trace.append(x.detach().clone())
        self.L += active.long()
        x = self.m.model.norm(x[:, 0])
        return self.m.lm_head(x).float()

    # ---------------------------------------------------------------- compression
    def compress(self, s: int, prompt_len: int, n_gen: int):
        t0 = time.time()
        L = int(self.L[s])
        keep, newbias = self.policy.select(s, L, prompt_len, n_gen)  # keep [nL, Hkv, L'] slot indices
        if keep is None:
            return None
        Ln = keep.shape[-1]
        g = lambda a: a[:, s].gather(2, keep if a.dim() == 4 else keep[..., None].expand(-1, -1, -1, a.shape[-1]))
        self.K[:, s, :, :Ln] = g(self.K)
        self.V[:, s, :, :Ln] = g(self.V)
        self.pos[:, s, :, :Ln] = g(self.pos)
        for k in self.meta:
            self.meta[k][:, s, :, :Ln] = g(self.meta[k])
        self.bias[:, s] = NEG
        self.bias[:, s, :, :Ln] = newbias
        self.pos[:, s, :, Ln:] = -1
        self.L[s] = Ln
        self.n_compress += 1
        self.compress_time += time.time() - t0
        return Ln


def sample(logits: torch.Tensor, gens: list, temperature=0.6, top_p=0.95) -> torch.Tensor:
    """Per-slot top-p sampling with per-slot generators (so a slot's stream is independent of the batch)."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    sp = torch.where(cum - sp < top_p, sp, torch.zeros_like(sp))
    out = []
    for r in range(logits.shape[0]):
        j = torch.multinomial(sp[r], 1, generator=gens[r])
        out.append(si[r, j])
    return torch.cat(out)

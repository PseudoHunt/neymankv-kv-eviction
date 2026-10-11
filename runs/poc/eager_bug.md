# bf16 "eager" attention corrupts R1-Distill-Qwen-1.5B (found at GPU clock ~2h10m)

Greedy decoding of the same GSM8K prompt, transformers 5.7.0 / torch 2.11:

| dtype | attention | text |
|---|---|---|
| bf16 | eager | "80 divided by 2 is 44. So, each class has 4…" |
| bf16 | sdpa | "80 divided by 2, which is 40." |
| fp32 | sdpa | "80 divided by 2, which is 40." |
| fp16 | sdpa | "80 divided by 2, which is 40." |

HF eager computes `matmul(q, k^T) * scaling` in bf16, rounding attention scores before the fp32 softmax. This model's
attention logits are large (bf16 spacing is 1.0 near 128), so the rounding corrupts copying of digits. All weights load
exactly (339/339 tensors bit-identical to the safetensors files). The engine originally replicated eager semantics
(and matched HF eager); it now uses SDPA with the logit bias as an additive float mask. All results produced on the eager
path (S1 attempt 2, GPU clock 1h11m–2h10m) are quarantined in `runs/poc/invalid_eager_bf16/` and not used.

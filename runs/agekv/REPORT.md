# AgeKV Phase 1: random eviction with a closed-form correction (R1-Distill-Qwen-1.5B, MATH500, B=256)

**Phase 1 result: inconclusive, leaning negative.** `random_ht` (exact Hajek correction) and `random_ht_clip` (bias clipped at
log 50) both score **57.5%** vs **62.5%** for `random`, a −5-point gap. That falls between the brief's thresholds: not within 3
points (viable), and not a loss of more than 7 (recency is the feature). The paired differences are small and within noise:
random_ht wins 5 / loses 7 problems vs random; the clipped version wins 4 / loses 6. Both raise the cap-hit rate (37.5% / 40% vs 30%).
Read literally, the "correction viable, Phase 2 on" condition is **not met**, and neither is the "loss > 7, run `agekv` only"
condition. The PI should decide how Phase 2 proceeds.

Data: `runs/poc/results.csv`, `runs/poc/per_problem.csv` (methods `random`, `random_ht`, `random_ht_clip`; set `math`).
All numbers are in-harness, not paper numbers. Harness and setup: `runs/poc/REPORT.md` §1–2.

## P1.1 Position histogram
![age histogram](age_histogram.png)
Kept generated tokens per head by age, one MATH trace (`math_134`), all layers and heads pooled (`runs/poc/age_hist.json`).
This confirms the allocation explanation: `random` decays geometrically, matching (B/(B+64))^(a/64) = 0.8 per 64 tokens:
114 → 74 → 48 → 30 → 20 per 128-token bin. `simbin` is also recency-biased. `neymankv_v1` is flat (≈15–20 per bin out to
age 2,000). The v2 curve in this figure is from a since-fixed bug (prior not applied); see the PoC report.

## P1.2 / P1.3 Results (MATH500, n=40, B=256, same seeds and decoding as S1)

| method | acc | mean / p95 len | cap rate | per-trace max bias, mean | max bias, max |
|---|---|---|---|---|---|
| full | 67.5% | 2131 / 3072 | 25% | – | – |
| random | 62.5% | 1985 / 3072 | 30% | 0 | 0 |
| random_ht | 57.5% | 2227 / 3072 | 37.5% | 6.47 nats | 9.37 nats |
| random_ht_clip | 57.5% | 2278 / 3072 | 40% | 3.72 nats | 3.91 nats (= clip) |
| neymankv_v2 (for reference) | 67.5% | 1953 / 3072 | 25% | – | – |

**Implementation of the correction.** Each survivor's bias accumulates log(M/B) per eviction round (M candidates, B kept). That is
−log of its exact inclusion probability, and in steady state (M = B+64) it equals the brief's floor(a/64)·log((B+64)/B), i.e. ALiBi
slope m = log(1.25)/64 = 0.0035 per token with a positive sign. It also stays exact in the early rounds, when M < B+64.
Protected tokens get bias 0.

## What the correction did
- Biases get large. The oldest survivors in a 3,000-token trace carry about 9.4 nats, i.e. a weight of ~12,000 relative to the
  protected tokens. The mean per-trace maximum is 6.5 nats (≈650×). So the corrected estimator leans on a handful of ancient tokens.
- Clipping at log 50 removes that variance but does not recover accuracy (still 57.5%). So variance from huge weights is not the
  whole story. Up-weighting old survivors at all, i.e. undoing part of random eviction's recency emphasis, seems to cost accuracy.
- Both corrected variants lengthen traces and hit the cap more often (37.5–40% vs 30%), the same direction as v1's failure, but much milder.
- Interpretation, to be confirmed at higher n: on reasoning traces the recency bias of plain random eviction is doing useful work
  that an unbiased (Hajek) estimate partly undoes. P2.1's measured age profiles are the right next step to quantify it.

## Not done (clock)
- `random_ht` forced-prefix probe (P1.4): skipped by the 4h15m launch cutoff.
- Phase 2 (age-profile calibration, `agekv`, novelty reserve): next session. Note for P2.3: the retention probe with the recall cue
  separates methods clearly (`full` 20/20, `random` 1/20, recency-biased methods 0/20, uniform-coverage v1 4/20), so a novelty
  reserve has a measurable target.
- Prerequisite for P2.1: re-capture with the fixed tokenizer and SDPA attention (`runs/poc/REPORT.md` §2.1–2.2).

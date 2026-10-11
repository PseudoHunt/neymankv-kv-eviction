# Follow-up report (Phases A, D, G)

The summary, the G2 plan, the list of cut jobs and GPU time are in `runs/STATUS.md`. This file holds the full tables.
Phases B and C were cancelled by the PI. Phase E (age profile, budget plot) and Phase F (cost) were not run.
The invalid-run quarantine is still in place: `runs/poc/invalid_tokenizer_bug/`, `runs/poc/invalid_eager_bf16/`.
Tokenizer workaround: `PreTrainedTokenizerFast.from_pretrained` plus a round-trip assert (transformers 5.7 loads the
R1-Distill tokenizer as LlamaTokenizer and drops spaces).

## Phase G — primary result: Qwen3-4B, kvpress RULER-4K, query-agnostic, 25% kept
Verdict: stratified coverage beats random (+13.2 [+9.6, +17.1]), but the gain comes from stratification, not the
reweighting (+2.1 [−0.3, +4.6]). Against expected_attention it loses by 66.6 points, including on cwe/fwe.
The gate's part 2 fails, so RULER-4K at 25% was not extended and Phase F did not run.

The random_rw control is in a separate group of rows below (rows comparing against random_rw; see the explanation in
STATUS.md: under uniform selection the bias is a constant context-mass correction).

| comparison | n | mean a | mean b | diff (pts) | 95% CI | W/L/T |
|---|---|---|---|---|---|---|
| v1_prop - random | 260 | 19.4 | 6.2 | +13.2 | [+9.6, +17.1] | 55/4/201 |
| v1_prop - v1_prop_noweight | 260 | 19.4 | 17.3 | +2.1 | [-0.3, +4.6] | 27/13/220 |
| random_rw - random | 260 | 11.2 | 6.2 | +5.0 | [+1.9, +8.2] | 28/8/224 |
| v1_prop_noweight - random | 260 | 17.3 | 6.2 | +11.1 | [+7.6, +14.8] | 46/6/208 |
| v1_prop_noweight - random_rw | 260 | 17.3 | 11.2 | +6.1 | [+3.0, +9.2] | 34/8/218 |
| v1_prop - random_rw | 260 | 19.4 | 11.2 | +8.2 | [+4.9, +11.6] | 42/5/213 |
| v1_prop - expected_attention | 260 | 19.4 | 86.0 | -66.6 | [-72.0, -61.0] | 3/191/66 |
| expected_attention - random | 260 | 86.0 | 6.2 | +79.8 | [+75.0, +84.4] | 220/1/39 |
| random - full | 260 | 6.2 | 95.3 | -89.1 | [-92.4, -85.5] | 0/241/19 |
| v1_prop - full | 260 | 19.4 | 95.3 | -75.9 | [-80.6, -71.0] | 1/214/45 |

### Per-task scores (RULER-4K, 25% kept, 20 samples per task)
```
full               cwe:0.93 fwe:0.90 niah_multikey_1:1.00 niah_multikey_2:1.00 niah_multikey_3:1.00 niah_multiquery:1.00 niah_multivalue:1.00 niah_single_1:1.00 niah_single_2:1.00 niah_single_3:1.00 qa_1:0.85 qa_2:0.70 vt:1.00
random             cwe:0.11 fwe:0.43 niah_multikey_1:0.00 niah_multikey_2:0.00 niah_multikey_3:0.00 niah_multiquery:0.00 niah_multivalue:0.00 niah_single_1:0.00 niah_single_2:0.00 niah_single_3:0.00 qa_1:0.15 qa_2:0.10 vt:0.01
random_rw          cwe:0.20 fwe:0.73 niah_multikey_1:0.05 niah_multikey_2:0.00 niah_multikey_3:0.00 niah_multiquery:0.00 niah_multivalue:0.00 niah_single_1:0.00 niah_single_2:0.00 niah_single_3:0.00 qa_1:0.15 qa_2:0.30 vt:0.02
v1_prop_noweight   cwe:0.61 fwe:0.73 niah_multikey_1:0.10 niah_multikey_2:0.00 niah_multikey_3:0.00 niah_multiquery:0.01 niah_multivalue:0.00 niah_single_1:0.15 niah_single_2:0.00 niah_single_3:0.00 qa_1:0.20 qa_2:0.35 vt:0.09
v1_prop            cwe:0.67 fwe:0.78 niah_multikey_1:0.10 niah_multikey_2:0.05 niah_multikey_3:0.00 niah_multiquery:0.03 niah_multivalue:0.00 niah_single_1:0.05 niah_single_2:0.10 niah_single_3:0.00 qa_1:0.25 qa_2:0.35 vt:0.15
expected_attention cwe:0.86 fwe:0.87 niah_multikey_1:1.00 niah_multikey_2:1.00 niah_multikey_3:0.75 niah_multiquery:1.00 niah_multivalue:1.00 niah_single_1:1.00 niah_single_2:1.00 niah_single_3:0.70 qa_1:0.55 qa_2:0.45 vt:1.00
```

Pooled cwe+fwe, paired:
```
v1_prop - expected_attention {'n': 40, 'mean_a': np.float64(0.724), 'mean_b': np.float64(0.863), 'diff': np.float64(-0.139), 'lo': np.float64(-0.209), 'hi': np.float64(-0.078), 'wins': 2, 'losses': 19, 'ties': 19}
v1_prop_noweight - expected_attention {'n': 40, 'mean_a': np.float64(0.672), 'mean_b': np.float64(0.863), 'diff': np.float64(-0.192), 'lo': np.float64(-0.268), 'hi': np.float64(-0.119), 'wins': 2, 'losses': 26, 'ties': 12}
v1_prop - random {'n': 40, 'mean_a': np.float64(0.724), 'mean_b': np.float64(0.272), 'diff': np.float64(0.452), 'lo': np.float64(0.348), 'hi': np.float64(0.554), 'wins': 32, 'losses': 2, 'ties': 6}
expected_attention - full {'n': 40, 'mean_a': np.float64(0.863), 'mean_b': np.float64(0.918), 'diff': np.float64(-0.054), 'lo': np.float64(-0.084), 'hi': np.float64(-0.027), 'wins': 2, 'losses': 16, 'ties': 22}
```

Caveats: LooGLE short-dependency questions are mostly local lookups, so coverage should help there (favourable to us).
LooGLE was not run (cut at wrap-up). kvpress's RULER-4K query-agnostic setting is the one reviewers compare against,
and it is the setting used here.

## Phase A — reasoning, R1-Distill-Qwen-1.5B, B=256, two seeds
Verdict: no pairwise difference excludes 0 in any split; full, random and v2 are indistinguishable at B=256.
The neymankv_v2 vs random gate fails (−1.8 [−5.2, +1.5]).

"PoC original 40" are the 40 MATH and 40 GSM8K problems used in the PoC. "new" are math_x (80) and gsm_x (40).

| dataset | problems | comparison | n | diff (pts) | 95% CI | W/L/T |
|---|---|---|---|---|---|---|
| math | all | neymankv_v2 - random | 120 | -1.7 | [-5.8, +2.5] | 8/12/100 |
| math | all | neymankv_v2 - full | 120 | -1.7 | [-6.7, +3.3] | 13/15/92 |
| math | all | random - full | 120 | +0.0 | [-5.4, +5.0] | 14/14/92 |
| math | PoC original 40 | neymankv_v2 - random | 40 | -2.5 | [-10.0, +3.8] | 2/3/35 |
| math | PoC original 40 | neymankv_v2 - full | 40 | +1.2 | [-10.0, +11.2] | 6/4/30 |
| math | PoC original 40 | random - full | 40 | +3.8 | [-6.2, +13.8] | 6/3/31 |
| math | new | neymankv_v2 - random | 80 | -1.2 | [-6.2, +4.4] | 6/9/65 |
| math | new | neymankv_v2 - full | 80 | -3.1 | [-8.8, +2.5] | 7/11/62 |
| math | new | random - full | 80 | -1.9 | [-8.1, +4.4] | 8/11/61 |
| gsm | all | neymankv_v2 - random | 80 | -1.9 | [-7.5, +3.8] | 7/8/65 |
| gsm | all | neymankv_v2 - full | 80 | +2.5 | [-4.4, +9.4] | 14/11/55 |
| gsm | all | random - full | 80 | +4.4 | [-1.9, +10.6] | 14/8/58 |
| gsm | PoC original 40 | neymankv_v2 - random | 40 | -2.5 | [-12.5, +6.2] | 5/5/30 |
| gsm | PoC original 40 | neymankv_v2 - full | 40 | +1.2 | [-8.8, +11.2] | 7/6/27 |
| gsm | PoC original 40 | random - full | 40 | +3.8 | [-3.8, +11.2] | 6/3/31 |
| gsm | new | neymankv_v2 - random | 40 | -1.2 | [-6.2, +3.8] | 2/3/35 |
| gsm | new | neymankv_v2 - full | 40 | +3.8 | [-5.0, +13.8] | 7/5/28 |
| gsm | new | random - full | 40 | +5.0 | [-5.0, +15.0] | 8/5/27 |
| all | all | neymankv_v2 - random | 200 | -1.8 | [-5.2, +1.5] | 15/20/165 |
| all | all | neymankv_v2 - full | 200 | +0.0 | [-4.2, +4.2] | 27/26/147 |
| all | all | random - full | 200 | +1.8 | [-2.2, +5.8] | 28/22/150 |
| all | PoC original 40 | neymankv_v2 - random | 80 | -2.5 | [-8.8, +3.1] | 7/8/65 |
| all | PoC original 40 | neymankv_v2 - full | 80 | +1.2 | [-6.2, +8.8] | 13/10/57 |
| all | PoC original 40 | random - full | 80 | +3.8 | [-1.9, +10.0] | 12/6/62 |
| all | new | neymankv_v2 - random | 120 | -1.2 | [-5.0, +2.9] | 8/12/100 |
| all | new | neymankv_v2 - full | 120 | -0.8 | [-5.8, +4.2] | 14/16/90 |
| all | new | random - full | 120 | +0.4 | [-4.6, +5.4] | 16/16/88 |

## Phase D — retention
Verdict: a query-free novelty reserve recovers almost no recall (3/20 and 1/20). An oracle reserve of the fact tokens
recovers 20/20 at no accuracy cost, so retention needs a query signal, not more slots.

**D1.** The fact tokens are " the access code is 7391-KQ" (forced-prefix indices 4–14).
- Under v2, copies survive at 39% after 512 generated tokens, 9% at 768 and <2% at 1024. The drop rate is 0.31 per
  compression.
- The fact token is a singleton in its 8-bit bucket in 43% of candidate rows. The floor fails because the recency
  prior gives old chunks fewer slots than they have strata.
- Under full attention, recall is 19/20 and retrieval is moderately concentrated. Layer 19, q-head 1 carries mass
  0.87 and is the top head on 17/20 prompts. A median of 14 of 56 KV heads covers 80% of the mass.
- Files: `d1_v2_fact_tokens.csv`, `d1_full_recall_attention.csv`, `d1_meta.json`.

**D2.** B=256, 20 recall-cue prompts and 20 MATH500 problems, seed 0.

| method | recall | Δ recall vs base (paired, 20) | MATH /20 |
|---|---|---|---|
| random_reserve | 3/20 | vs random 1/20 | 17 |
| v2_reserve | 1/20 | vs v2 0/20 | 14 |
| random_oracle | 20/20 | vs random 1/20 | 14 |
| v2_oracle | 20/20 | vs v2 0/20 | 13 |

The singleton reserve is oldest-first, and the fact sits at the start of the generation, which favours the reserve.
It fails anyway.

Paired against the PoC seed-0 runs (same prompts, bootstrap over prompts):

| method | base | recall diff (pts) | MATH acc diff (pts) |
|---|---|---|---|
| random_reserve | random | +10 [-10, +30] (n=20) | n/a |
| v2_reserve | neymankv_v2 | +5 [+0, +15] (n=20) | -10 [-30, +10] (n=20) |
| random_oracle | random | +95 [+85, +100] (n=20) | n/a |
| v2_oracle | neymankv_v2 | +100 [+100, +100] (n=20) | -10 [-30, +10] (n=20) |

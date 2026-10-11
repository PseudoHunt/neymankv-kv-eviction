# NeymanKV — status at close (2026-10-11, tag `poc-close-2026-10-11`)

The reasoning (decode-time) track is closed with a negative result. On long documents, query-agnostic stratified
coverage beats random by a wide margin, but it loses badly to expected_attention, and the HT reweighting is not
shown to matter. Details: `runs/followup/REPORT.md` (follow-up phases), `runs/poc/REPORT.md` (PoC),
`runs/agekv/REPORT.md` (AgeKV), `runs/REPORT.md` (offline study; its reasoning rows are invalid, see the erratum there).

## Phase A — reasoning, B=256 (verdict: no method differs)
R1-Distill-Qwen-1.5B, decode-time eviction every 64 steps. Two seeds (1, 2), 200 problems (120 MATH500 + 80 GSM8K).
Per-problem accuracy is averaged over the two seeds; paired bootstrap 95% CI over problems (10k resamples).

| comparison | diff (pts) | 95% CI | W/L/T |
|---|---|---|---|
| neymankv_v2 − random | −1.8 | [−5.2, +1.5] | 15/20/165 |
| neymankv_v2 − full   |  0.0 | [−4.2, +4.2] | 27/26/147 |
| random − full        | +1.8 | [−2.2, +5.8] | 28/22/150 |

Accuracy: full 64.2, random 66.0, v2 64.2. No interval excludes 0 in any split: PoC-original-40 vs new problems,
or MATH vs GSM8K (`runs/followup/phaseA_paired.csv`). At B=256 the three methods cannot be told apart, so this budget
does not discriminate between eviction policies on these tasks.

## Phase D — retention of a planted fact (verdict: retention needs a query signal)
D1 (offline diagnosis): under v2, copies of the fact tokens survive at 39% after 512 generated tokens, 9% at 768,
and <2% at 1024. The 1-per-stratum floor fails because the recency prior gives old chunks fewer slots than they have
strata.

D2 (reserves), B=256, recall-cue probe (20) and 20 MATH500:

| method | recall cue | MATH (/20) |
|---|---|---|
| random (PoC seed 0, reference) | 1/20 | — |
| neymankv_v2 (PoC seed 0, reference) | 0/20 | — |
| random_reserve (singleton reserve) | 3/20 | 17 |
| v2_reserve (singleton reserve) | 1/20 | 14 |
| random_oracle (fact tokens reserved) | **20/20** | 14 |
| v2_oracle (fact tokens reserved) | **20/20** | 13 |

A query-free novelty heuristic (keep tokens that are singletons in their hash bucket, oldest first) recovers almost
nothing. This holds even though the probe's fact sits at the very start of the generation, which favours an
oldest-first reserve. Reserving the right ~11 tokens recovers recall completely at no accuracy cost, so the capacity
is there; what is missing is a signal for which tokens matter. The novelty reserve was the cheapest variant and it
failed, so it was not continued.

## Phase G — long-document, query-agnostic compression (primary result)
Qwen3-4B via kvpress 0.5.5. Each context is compressed once without its question, keeping 25%
(compression_ratio 0.75). RULER-4K, 20 samples per task × 13 tasks = 260, paired per sample. Score = kvpress string
match (string_match_part for qa_*, string_match_all otherwise).

Scores: full 95.3 · expected_attention 86.0 · v1_prop 19.4 · v1_prop_noweight 17.3 · random_rw 11.2 · random 6.2

| # | comparison | diff (pts) | 95% CI | W/L/T |
|---|---|---|---|---|
| 1 | v1_prop − random | **+13.2** | [+9.6, +17.1] | 55/4/201 |
| 2 | v1_prop − v1_prop_noweight (the reweighting) | +2.1 | [−0.3, +4.6] | 27/13/220 |
| 3 | random_rw − random (the bias alone, no strata) | +5.0 | [+2.0, +8.2] | 28/8/224 |
| 4 | v1_prop − random_rw | +8.2 | [+4.9, +11.6] | 42/5/213 |
| 5 | v1_prop − expected_attention | **−66.6** | [−72.0, −61.0] | 3/191/66 |

Also: v1_prop_noweight − random +11.1 [+7.6, +14.8]; expected_attention − random +79.8 [+75.0, +84.4].

**Gate:** part 1 (v1_prop > random, CI excludes 0) passes. Part 2 (the no-weight ablation shows the reweighting is
doing it) fails: stratification gives +11.1 of the +13.2 points, and the reweighting's interval includes 0.
Compared with the published baseline, row 5, stratified-random selection is not competitive.

**cwe / fwe** (the coverage tasks; this table decides G2). Per task, 20 samples each:

| task | full | expected_attention | v1_prop | v1_prop_noweight | random_rw | random |
|---|---|---|---|---|---|---|
| cwe | 0.93 | 0.86 | 0.67 | 0.61 | 0.20 | 0.11 |
| fwe | 0.90 | 0.87 | 0.78 | 0.73 | 0.73 | 0.43 |

Pooled cwe+fwe (n=40, paired): v1_prop − expected_attention −13.9 [−20.9, −7.8] (2 wins / 19 losses);
expected_attention − full −5.4 [−8.4, −2.7]; v1_prop − random +45.2 [+34.8, +55.4].
v1_prop loses to expected_attention even on the tasks where coverage should matter most, and expected_attention is
only 5.4 points below full there.

On the needle and variable-tracking tasks (niah_*, vt), every query-agnostic random-based method is near 0
(v1_prop ≤ 0.15). expected_attention scores 0.70–1.00 on the same tasks.

### The `random_rw` artifact
random_rw keeps the same uniform-random set as `random` but adds the HT bias log(N/n) = log 4 to every kept context key.
Because selection is uniform, that bias is the **same constant for every context key**, so it carries no
importance-weighting information. Its only effect is to raise the total softmax mass on the context relative to the
question and answer tokens. That compensates for the 75% of context mass that was removed (the Hajek-style
denominator correction), and it alone is worth +5.0 points (row 3).
So any "reweighting helps" claim has to beat random_rw, not random. Within the stratified method the per-stratum
weights add only +2.1 [−0.3, +4.6] on top of what this global mass correction already provides.

### Phase G caveats (as requested by the PI)
- LooGLE (the SCBench substitute; SCBench is not in kvpress 0.5.5 and its contexts exceed the 32K window) was **not
  run**. Its short-dependency questions are mostly local lookups, where coverage should help, so it would have
  favoured v1_prop. Its absence removes a setting favourable to us, not an unfavourable one.
- kvpress's RULER-4K query-agnostic setting is the one reviewers compare against, and it is the setting reported here.

## Jobs cut at wrap-up (PI instruction, 15:20 UTC)
- Phases B and C were cancelled earlier by the PI.
- Phase E (age profile) and the budget plot were not produced.
- Phase G, not run: kvzip; RULER-8K (`full` finished at 25%, kept unscored; `random` and `v1_prop` killed at about
  12 min and 10 min); LooGLE; every 10% budget run (ratio 0.9); and the larger RULER-4K-at-25% run (the gate did not
  pass).
- Phase F (cost) was not run, because it was conditional on G passing.
- The knee test (B=64/128) was not run.
- Code fix after the runs: `longdoc/run_g.py` now writes a stable sha1 `context_hash`. The committed G jsonl files
  hold Python's salted `hash()`, so `longdoc/score_g.py` matches samples on (task, question, answer, occurrence)
  instead.

## G2 plan (not run)
**Question:** does SimHash stratification help as a coverage term on top of a real query-agnostic scorer, rather
than as a replacement for one?

- **Methods.** Both new presses go in `longdoc/presses.py`.
  - `expected_attention`, the baseline: AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)), unchanged.
  - `ea_strata`: the same expected-attention scores, but the per-head budget is first split across SimHash strata by
    proportional allocation (floor 1, Webster rounding, the same code as `v1_prop`). Within each stratum it keeps
    the top expected-attention scores. No logit bias.
  - `ea_floor`: guarantee one slot per non-empty stratum (that stratum's top-EA token), then fill the rest of the
    budget by global expected-attention top-k. No logit bias.
- **Data.** RULER-4K, kvpress query-agnostic protocol, the same 260 samples (20/task, random_state 0), at 25% kept
  (ratio 0.75) and 10% kept (ratio 0.9).
- **Pass condition**, judged at each budget separately. Both must hold for `ea_strata` or `ea_floor` against
  `expected_attention`, using a paired per-sample bootstrap (10k resamples):
  1. The overall RULER-4K difference is > 0 with a 95% CI lower bound > 0.
  2. No regression on the needle tasks: pooled niah_* + vt difference with a 95% CI lower bound > −2 points.

  G2 passes if either variant passes at either budget. Otherwise the long-document track closes too, and the
  write-up becomes an analysis paper.
- **Cost.** Four new runs (two methods × two budgets) plus expected_attention at 10%. About 25–30 process-minutes
  each, so about 2.5 process-hours, or about 1 GPU-hour on one A30 running 3 jobs at a time.
- **Prior from the cwe/fwe table.** At 25%, expected_attention is already within 5.4 points of full on cwe/fwe, and
  v1_prop is 13.9 points below it, so there is little headroom for a coverage term. **Recommendation:** run G2 only
  at 10%, where expected_attention's coverage loss should be larger. If time is tight, skip G2.

## Captures and artifacts
- `data/captures/` (gitignored) is 13 GB in total. The Qwen3-4B long-document and needle captures (`doc_*`,
  `needle_*`) are about 11.3 GB and are kept. The R1 reasoning captures (`reason_*`) are only about 1.7 GB, not 13 GB;
  they are reproducible in about 45 GPU-minutes and the reasoning track is closed. Disk was not tight (953 GB free),
  so nothing was deleted.
- kvpress job queue config: `runs/followup/orchestrate_G.sh` and `longdoc/run_g.py` (dataset sampling and sizes).
  The kvpress venv is `/home/venv_kvpress`; `runs/pip_freeze_kvpress.txt` holds its frozen packages.
- Main env: `runs/pip_freeze_main.txt`.
- Quarantined invalid runs, kept: `runs/poc/invalid_tokenizer_bug/`, `runs/poc/invalid_eager_bf16/`.

## GPU time, follow-up brief (one NVIDIA A30; up to 3 jobs ran concurrently)
| phase | process-hours (sum of job wall time) | share of GPU wall time |
|---|---|---|
| A (6 runs × 200 problems) | 7.92 | 2.90 h |
| D1 (retention diagnosis) | 0.48 | 0.18 h |
| D2 (8 reserve runs) | 1.32 | 0.48 h |
| G (6 RULER-4K runs + RULER-8K full + 2 killed) | 2.53 | 0.93 h |
| **total** | **12.25** | **4.49 h** (10:50 → 15:20 UTC) |

The share column splits the elapsed GPU wall time across phases in proportion to process-hours.
GPU time for earlier briefs is in the PoC and AgeKV reports.

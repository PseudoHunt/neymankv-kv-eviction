# T1 results

<!-- provenance -->
- git commit: `4033bd94e6c18770596c2474032f13aaea34e63c` (dirty=True)
- GPU: NVIDIA A30; python 3.10.20; torch 2.11.0+cu130
- pip freeze: 193 packages, full list in `runs/pip_freeze.txt`

Statistic: per head, average mean_err over seeds; then median over (prompt, layer, KV head).
seed_std = std over seeds of the per-seed median (random methods only).

## Gate E

Criterion: `neyman_proxy_*` with reweighting on `pre`/`pre_pos_norm` cuts the median relative output error by
>= 15% vs `random_pnorm` at both budgets on long_doc, and is not worse than `random_pnorm` on reasoning.

Cells: median error (reduction vs random_pnorm; positive = better).

random_pnorm baseline: long_doc@10% = 0.7464, long_doc@25% = 0.5145, reasoning@10% = 1.0670, reasoning@25% = 0.6399

| method            | strata       | long@10%      | long@25%      | reas@10%      | reas@25%      | passes   |
|:------------------|:-------------|:--------------|:--------------|:--------------|:--------------|:---------|
| neyman_proxy_key  | pre          | 0.3108 (+58%) | 0.1580 (+69%) | 0.4170 (+61%) | 0.1472 (+77%) | True     |
| neyman_proxy_key  | pre_pos_norm | 0.4527 (+39%) | 0.2049 (+60%) | 0.4763 (+55%) | 0.1310 (+80%) | True     |
| neyman_proxy_val  | pre          | 0.3013 (+60%) | 0.1518 (+70%) | 0.4508 (+58%) | 0.1541 (+76%) | True     |
| neyman_proxy_val  | pre_pos_norm | 0.4445 (+40%) | 0.1978 (+62%) | 0.5094 (+52%) | 0.1333 (+79%) | True     |
| neyman_oracle_key | pre          | 0.1675 (+78%) | 0.0556 (+89%) | 0.3667 (+66%) | 0.1023 (+84%) | True     |
| neyman_oracle_key | pre_pos_norm | 0.1450 (+81%) | 0.0357 (+93%) | 0.3249 (+70%) | 0.0561 (+91%) | True     |
| neyman_oracle_val | pre          | 0.1521 (+80%) | 0.0517 (+90%) | 0.3394 (+68%) | 0.0982 (+85%) | True     |
| neyman_oracle_val | pre_pos_norm | 0.1237 (+83%) | 0.0327 (+94%) | 0.2975 (+72%) | 0.0569 (+91%) | True     |

**Gate E: PASS** (proxy passes: True; oracle passes: True).

## Ablations (median error; long_doc / reasoning; 10% / 25%)

### Reweighting on vs off (fixed allocation)

| config                                 |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:---------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_oracle_val [pre] no-rw          |              0.3596 |               0.1697 |               0.3673 |                0.177  |
| neyman_oracle_val [pre] rw             |              0.1521 |               0.0517 |               0.3394 |                0.0982 |
| neyman_oracle_val [pre_pos_norm] no-rw |              0.318  |               0.1194 |               0.3128 |                0.1016 |
| neyman_oracle_val [pre_pos_norm] rw    |              0.1237 |               0.0327 |               0.2975 |                0.0569 |
| neyman_proxy_val [pre] no-rw           |              0.3998 |               0.227  |               0.3883 |                0.1993 |
| neyman_proxy_val [pre] rw              |              0.3013 |               0.1518 |               0.4508 |                0.1541 |
| neyman_proxy_val [pre_pos_norm] no-rw  |              0.3792 |               0.1942 |               0.3389 |                0.1311 |
| neyman_proxy_val [pre_pos_norm] rw     |              0.4445 |               0.1978 |               0.5094 |                0.1333 |
| strat_prop [post] no-rw                |              0.3916 |               0.2013 |               0.4223 |                0.2217 |
| strat_prop [post] rw                   |              0.2158 |               0.0845 |               0.5108 |                0.1788 |
| strat_prop [pre] no-rw                 |              0.3878 |               0.2011 |               0.4227 |                0.228  |
| strat_prop [pre] rw                    |              0.2014 |               0.0799 |               0.5135 |                0.1768 |
| strat_prop [pre_pos_norm] no-rw        |              0.3926 |               0.2055 |               0.4032 |                0.2201 |
| strat_prop [pre_pos_norm] rw           |              0.2188 |               0.0834 |               0.5321 |                0.1847 |

### Proportional vs Neyman (fixed strata, reweighting on)

| config                                |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:--------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_observed_key [pre] rw          |              0.2545 |               0.1236 |               0.388  |                0.1349 |
| neyman_observed_key [pre_pos_norm] rw |              0.3734 |               0.1532 |               0.3644 |                0.0973 |
| neyman_observed_val [pre] rw          |              0.2437 |               0.112  |               0.4165 |                0.1367 |
| neyman_observed_val [pre_pos_norm] rw |              0.3287 |               0.1334 |               0.3786 |                0.092  |
| neyman_oracle_key [pre] rw            |              0.1675 |               0.0556 |               0.3667 |                0.1023 |
| neyman_oracle_key [pre_pos_norm] rw   |              0.145  |               0.0357 |               0.3249 |                0.0561 |
| neyman_oracle_val [pre] rw            |              0.1521 |               0.0517 |               0.3394 |                0.0982 |
| neyman_oracle_val [pre_pos_norm] rw   |              0.1237 |               0.0327 |               0.2975 |                0.0569 |
| neyman_proxy_key [pre] rw             |              0.3108 |               0.158  |               0.417  |                0.1472 |
| neyman_proxy_key [pre_pos_norm] rw    |              0.4527 |               0.2049 |               0.4763 |                0.131  |
| neyman_proxy_val [pre] rw             |              0.3013 |               0.1518 |               0.4508 |                0.1541 |
| neyman_proxy_val [pre_pos_norm] rw    |              0.4445 |               0.1978 |               0.5094 |                0.1333 |
| strat_prop [pre] rw                   |              0.2014 |               0.0799 |               0.5135 |                0.1768 |
| strat_prop [pre_pos_norm] rw          |              0.2188 |               0.0834 |               0.5321 |                0.1847 |

### post vs pre strata (proportional, reweighting on)

| config                       |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:-----------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| strat_prop [post] rw         |              0.2158 |               0.0845 |               0.5108 |                0.1788 |
| strat_prop [pre] rw          |              0.2014 |               0.0799 |               0.5135 |                0.1768 |
| strat_prop [pre_pos_norm] rw |              0.2188 |               0.0834 |               0.5321 |                0.1847 |

### key vs val S_h (reweighting on)

| config                                |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:--------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_observed_key [pre] rw          |              0.2545 |               0.1236 |               0.388  |                0.1349 |
| neyman_observed_key [pre_pos_norm] rw |              0.3734 |               0.1532 |               0.3644 |                0.0973 |
| neyman_observed_val [pre] rw          |              0.2437 |               0.112  |               0.4165 |                0.1367 |
| neyman_observed_val [pre_pos_norm] rw |              0.3287 |               0.1334 |               0.3786 |                0.092  |
| neyman_oracle_key [pre] rw            |              0.1675 |               0.0556 |               0.3667 |                0.1023 |
| neyman_oracle_key [pre_pos_norm] rw   |              0.145  |               0.0357 |               0.3249 |                0.0561 |
| neyman_oracle_val [pre] rw            |              0.1521 |               0.0517 |               0.3394 |                0.0982 |
| neyman_oracle_val [pre_pos_norm] rw   |              0.1237 |               0.0327 |               0.2975 |                0.0569 |
| neyman_proxy_key [pre] rw             |              0.3108 |               0.158  |               0.417  |                0.1472 |
| neyman_proxy_key [pre_pos_norm] rw    |              0.4527 |               0.2049 |               0.4763 |                0.131  |
| neyman_proxy_val [pre] rw             |              0.3013 |               0.1518 |               0.4508 |                0.1541 |
| neyman_proxy_val [pre_pos_norm] rw    |              0.4445 |               0.1978 |               0.5094 |                0.1333 |

### Within-stratum selection: random vs systematic (neyman_proxy_val, pre_pos_norm, rw)

| config                                          |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:------------------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_proxy_val [pre_pos_norm] (systematic) rw |              0.4455 |               0.1919 |               0.4923 |                0.1292 |
| neyman_proxy_val [pre_pos_norm] rw              |              0.4445 |               0.1978 |               0.5094 |                0.1333 |

## Full table: median mean_err

| config                                          |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:------------------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_observed_key [pre] rw                    |              0.2545 |               0.1236 |               0.388  |                0.1349 |
| neyman_observed_key [pre_pos_norm] rw           |              0.3734 |               0.1532 |               0.3644 |                0.0973 |
| neyman_observed_val [pre] rw                    |              0.2437 |               0.112  |               0.4165 |                0.1367 |
| neyman_observed_val [pre_pos_norm] rw           |              0.3287 |               0.1334 |               0.3786 |                0.092  |
| neyman_oracle_key [pre] no-rw                   |              0.3679 |               0.1721 |               0.3708 |                0.187  |
| neyman_oracle_key [pre] rw                      |              0.1675 |               0.0556 |               0.3667 |                0.1023 |
| neyman_oracle_key [pre_pos_norm] no-rw          |              0.3208 |               0.1226 |               0.3216 |                0.1055 |
| neyman_oracle_key [pre_pos_norm] rw             |              0.145  |               0.0357 |               0.3249 |                0.0561 |
| neyman_oracle_val [pre] no-rw                   |              0.3596 |               0.1697 |               0.3673 |                0.177  |
| neyman_oracle_val [pre] rw                      |              0.1521 |               0.0517 |               0.3394 |                0.0982 |
| neyman_oracle_val [pre_pos_norm] no-rw          |              0.318  |               0.1194 |               0.3128 |                0.1016 |
| neyman_oracle_val [pre_pos_norm] rw             |              0.1237 |               0.0327 |               0.2975 |                0.0569 |
| neyman_proxy_key [pre] no-rw                    |              0.4043 |               0.2293 |               0.3907 |                0.2009 |
| neyman_proxy_key [pre] rw                       |              0.3108 |               0.158  |               0.417  |                0.1472 |
| neyman_proxy_key [pre_pos_norm] no-rw           |              0.3834 |               0.1942 |               0.3393 |                0.1205 |
| neyman_proxy_key [pre_pos_norm] rw              |              0.4527 |               0.2049 |               0.4763 |                0.131  |
| neyman_proxy_val [pre] no-rw                    |              0.3998 |               0.227  |               0.3883 |                0.1993 |
| neyman_proxy_val [pre] rw                       |              0.3013 |               0.1518 |               0.4508 |                0.1541 |
| neyman_proxy_val [pre_pos_norm] (systematic) rw |              0.4455 |               0.1919 |               0.4923 |                0.1292 |
| neyman_proxy_val [pre_pos_norm] no-rw           |              0.3792 |               0.1942 |               0.3389 |                0.1311 |
| neyman_proxy_val [pre_pos_norm] rw              |              0.4445 |               0.1978 |               0.5094 |                0.1333 |
| random_pnorm                                    |              0.7464 |               0.5145 |               1.067  |                0.6399 |
| random_pnorm_naiveHT                            |              4.2008 |               4.1895 |               1.998  |                1.6988 |
| random_uniform                                  |              0.2257 |               0.0913 |               0.4756 |                0.1896 |
| rkvhash [lsh6]                                  |              0.4459 |               0.272  |               0.4547 |                0.2645 |
| rkvhash [lsh8]                                  |              0.4401 |               0.2533 |               0.4421 |                0.2744 |
| rkvhash_lam0.1 [lsh8]                           |              0.3155 |               0.1392 |               0.211  |                0.089  |
| simbin [post] (oldest_first_evict)              |              0.3396 |               0.1722 |               0.2955 |                0.141  |
| strat_prop [post] no-rw                         |              0.3916 |               0.2013 |               0.4223 |                0.2217 |
| strat_prop [post] rw                            |              0.2158 |               0.0845 |               0.5108 |                0.1788 |
| strat_prop [pre] no-rw                          |              0.3878 |               0.2011 |               0.4227 |                0.228  |
| strat_prop [pre] rw                             |              0.2014 |               0.0799 |               0.5135 |                0.1768 |
| strat_prop [pre_pos_norm] no-rw                 |              0.3926 |               0.2055 |               0.4032 |                0.2201 |
| strat_prop [pre_pos_norm] rw                    |              0.2188 |               0.0834 |               0.5321 |                0.1847 |
| topk_oracle                                     |              0.1001 |               0.0245 |               0.1173 |                0.0316 |

## Median p95 error over queries

| config                                          |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:------------------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_observed_key [pre] rw                    |              0.8727 |               0.4449 |               1.1922 |                0.4469 |
| neyman_observed_key [pre_pos_norm] rw           |              1.1967 |               0.5495 |               1.1445 |                0.3417 |
| neyman_observed_val [pre] rw                    |              0.8291 |               0.4144 |               1.2283 |                0.4329 |
| neyman_observed_val [pre_pos_norm] rw           |              1.0884 |               0.5065 |               1.1165 |                0.3051 |
| neyman_oracle_key [pre] no-rw                   |              1.0089 |               0.5544 |               1.0293 |                0.5574 |
| neyman_oracle_key [pre] rw                      |              0.5739 |               0.2097 |               1.103  |                0.3424 |
| neyman_oracle_key [pre_pos_norm] no-rw          |              0.9605 |               0.4251 |               0.8869 |                0.3399 |
| neyman_oracle_key [pre_pos_norm] rw             |              0.4752 |               0.1368 |               0.9871 |                0.1874 |
| neyman_oracle_val [pre] no-rw                   |              0.991  |               0.5539 |               0.9722 |                0.529  |
| neyman_oracle_val [pre] rw                      |              0.5376 |               0.1959 |               1.0329 |                0.3338 |
| neyman_oracle_val [pre_pos_norm] no-rw          |              0.949  |               0.425  |               0.8554 |                0.3252 |
| neyman_oracle_val [pre_pos_norm] rw             |              0.4209 |               0.1179 |               0.9281 |                0.1795 |
| neyman_proxy_key [pre] no-rw                    |              1.0815 |               0.7338 |               1.0677 |                0.6073 |
| neyman_proxy_key [pre] rw                       |              1.0332 |               0.5662 |               1.3123 |                0.5186 |
| neyman_proxy_key [pre_pos_norm] no-rw           |              1.0796 |               0.6659 |               0.9461 |                0.4138 |
| neyman_proxy_key [pre_pos_norm] rw              |              1.4608 |               0.7222 |               1.4472 |                0.476  |
| neyman_proxy_val [pre] no-rw                    |              1.07   |               0.7203 |               1.0776 |                0.5939 |
| neyman_proxy_val [pre] rw                       |              0.9915 |               0.5409 |               1.3693 |                0.5063 |
| neyman_proxy_val [pre_pos_norm] (systematic) rw |              1.4092 |               0.6926 |               1.4947 |                0.4576 |
| neyman_proxy_val [pre_pos_norm] no-rw           |              1.0765 |               0.6738 |               0.931  |                0.4125 |
| neyman_proxy_val [pre_pos_norm] rw              |              1.399  |               0.6904 |               1.5141 |                0.4653 |
| random_pnorm                                    |              2.4297 |               1.6674 |               3.6519 |                2.2586 |
| random_pnorm_naiveHT                            |             11.1198 |              10.969  |               6.1447 |                4.4827 |
| random_uniform                                  |              0.7133 |               0.2975 |               1.4067 |                0.5636 |
| rkvhash [lsh6]                                  |              1.1315 |               0.8325 |               1.2591 |                0.8094 |
| rkvhash [lsh8]                                  |              1.1066 |               0.7793 |               1.25   |                0.82   |
| rkvhash_lam0.1 [lsh8]                           |              1.044  |               0.5491 |               0.781  |                0.3512 |
| simbin [post] (oldest_first_evict)              |              1.0636 |               0.6059 |               0.8774 |                0.4668 |
| strat_prop [post] no-rw                         |              1.039  |               0.6042 |               1.1677 |                0.6243 |
| strat_prop [post] rw                            |              0.7026 |               0.2897 |               1.4766 |                0.5413 |
| strat_prop [pre] no-rw                          |              1.0355 |               0.6032 |               1.1677 |                0.6495 |
| strat_prop [pre] rw                             |              0.6758 |               0.2799 |               1.5029 |                0.5442 |
| strat_prop [pre_pos_norm] no-rw                 |              1.0402 |               0.614  |               1.1139 |                0.615  |
| strat_prop [pre_pos_norm] rw                    |              0.7066 |               0.2866 |               1.5631 |                0.5667 |
| topk_oracle                                     |              0.3805 |               0.0973 |               0.4209 |                0.1221 |

## Median attention-mass recall

| config                                          |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:------------------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_observed_key [pre] rw                    |               0.693 |                0.771 |                0.581 |                 0.704 |
| neyman_observed_key [pre_pos_norm] rw           |               0.719 |                0.81  |                0.606 |                 0.807 |
| neyman_observed_val [pre] rw                    |               0.627 |                0.718 |                0.54  |                 0.691 |
| neyman_observed_val [pre_pos_norm] rw           |               0.646 |                0.781 |                0.568 |                 0.785 |
| neyman_oracle_key [pre] no-rw                   |               0.744 |                0.814 |                0.632 |                 0.719 |
| neyman_oracle_key [pre] rw                      |               0.744 |                0.814 |                0.632 |                 0.719 |
| neyman_oracle_key [pre_pos_norm] no-rw          |               0.759 |                0.847 |                0.634 |                 0.798 |
| neyman_oracle_key [pre_pos_norm] rw             |               0.759 |                0.847 |                0.634 |                 0.798 |
| neyman_oracle_val [pre] no-rw                   |               0.658 |                0.793 |                0.584 |                 0.714 |
| neyman_oracle_val [pre] rw                      |               0.658 |                0.793 |                0.584 |                 0.714 |
| neyman_oracle_val [pre_pos_norm] no-rw          |               0.694 |                0.835 |                0.605 |                 0.796 |
| neyman_oracle_val [pre_pos_norm] rw             |               0.694 |                0.835 |                0.605 |                 0.796 |
| neyman_proxy_key [pre] no-rw                    |               0.687 |                0.77  |                0.573 |                 0.694 |
| neyman_proxy_key [pre] rw                       |               0.687 |                0.77  |                0.573 |                 0.694 |
| neyman_proxy_key [pre_pos_norm] no-rw           |               0.707 |                0.794 |                0.6   |                 0.78  |
| neyman_proxy_key [pre_pos_norm] rw              |               0.707 |                0.794 |                0.6   |                 0.78  |
| neyman_proxy_val [pre] no-rw                    |               0.63  |                0.735 |                0.524 |                 0.683 |
| neyman_proxy_val [pre] rw                       |               0.63  |                0.735 |                0.524 |                 0.683 |
| neyman_proxy_val [pre_pos_norm] (systematic) rw |               0.662 |                0.786 |                0.564 |                 0.758 |
| neyman_proxy_val [pre_pos_norm] no-rw           |               0.665 |                0.786 |                0.568 |                 0.761 |
| neyman_proxy_val [pre_pos_norm] rw              |               0.665 |                0.786 |                0.568 |                 0.761 |
| random_pnorm                                    |               0.575 |                0.632 |                0.507 |                 0.588 |
| random_pnorm_naiveHT                            |               0.576 |                0.632 |                0.507 |                 0.588 |
| random_uniform                                  |               0.589 |                0.656 |                0.489 |                 0.578 |
| rkvhash [lsh6]                                  |               0.616 |                0.669 |                0.492 |                 0.59  |
| rkvhash [lsh8]                                  |               0.604 |                0.663 |                0.495 |                 0.564 |
| rkvhash_lam0.1 [lsh8]                           |               0.773 |                0.828 |                0.764 |                 0.845 |
| simbin [post] (oldest_first_evict)              |               0.588 |                0.661 |                0.604 |                 0.702 |
| strat_prop [post] no-rw                         |               0.587 |                0.651 |                0.494 |                 0.59  |
| strat_prop [post] rw                            |               0.587 |                0.651 |                0.494 |                 0.59  |
| strat_prop [pre] no-rw                          |               0.59  |                0.659 |                0.5   |                 0.585 |
| strat_prop [pre] rw                             |               0.59  |                0.659 |                0.5   |                 0.585 |
| strat_prop [pre_pos_norm] no-rw                 |               0.592 |                0.662 |                0.502 |                 0.583 |
| strat_prop [pre_pos_norm] rw                    |               0.592 |                0.662 |                0.502 |                 0.583 |
| topk_oracle                                     |               0.88  |                0.939 |                0.818 |                 0.904 |

## Seed std of the median (mean_err)

| config                                          |   ('long_doc', 0.1) |   ('long_doc', 0.25) |   ('reasoning', 0.1) |   ('reasoning', 0.25) |
|:------------------------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| neyman_observed_key [pre] rw                    |              0.0047 |               0.0014 |               0.0146 |                0.0019 |
| neyman_observed_key [pre_pos_norm] rw           |              0.0085 |               0.0028 |               0.0126 |                0.0032 |
| neyman_observed_val [pre] rw                    |              0.0052 |               0.0013 |               0.0135 |                0.0024 |
| neyman_observed_val [pre_pos_norm] rw           |              0.0046 |               0.003  |               0.013  |                0.0027 |
| neyman_oracle_key [pre] no-rw                   |              0.0018 |               0.0008 |               0.0012 |                0.0027 |
| neyman_oracle_key [pre] rw                      |              0.0027 |               0.0007 |               0.0051 |                0.0024 |
| neyman_oracle_key [pre_pos_norm] no-rw          |              0.0007 |               0.001  |               0.0032 |                0.0008 |
| neyman_oracle_key [pre_pos_norm] rw             |              0.0019 |               0.0003 |               0.0043 |                0.0011 |
| neyman_oracle_val [pre] no-rw                   |              0.0015 |               0.0005 |               0.0047 |                0.0024 |
| neyman_oracle_val [pre] rw                      |              0.0006 |               0.0003 |               0.0066 |                0.0024 |
| neyman_oracle_val [pre_pos_norm] no-rw          |              0.0026 |               0.0004 |               0.0035 |                0.0009 |
| neyman_oracle_val [pre_pos_norm] rw             |              0.0009 |               0.0002 |               0.0064 |                0.0015 |
| neyman_proxy_key [pre] no-rw                    |              0.0014 |               0.001  |               0.0043 |                0.0016 |
| neyman_proxy_key [pre] rw                       |              0.0066 |               0.0047 |               0.0034 |                0.0022 |
| neyman_proxy_key [pre_pos_norm] no-rw           |              0.0021 |               0.0012 |               0.0043 |                0.0026 |
| neyman_proxy_key [pre_pos_norm] rw              |              0.0072 |               0.003  |               0.0262 |                0.0067 |
| neyman_proxy_val [pre] no-rw                    |              0.0028 |               0.002  |               0.0073 |                0.0029 |
| neyman_proxy_val [pre] rw                       |              0.0065 |               0.003  |               0.0044 |                0.0093 |
| neyman_proxy_val [pre_pos_norm] (systematic) rw |              0.0048 |               0.0039 |               0.013  |                0.0035 |
| neyman_proxy_val [pre_pos_norm] no-rw           |              0.0019 |               0.0012 |               0.0042 |                0.0035 |
| neyman_proxy_val [pre_pos_norm] rw              |              0.0136 |               0.0018 |               0.0316 |                0.0078 |
| random_pnorm                                    |              0.0076 |               0.0034 |               0.0251 |                0.0183 |
| random_pnorm_naiveHT                            |              0.1889 |               0.0481 |               0.1486 |                0.0818 |
| random_uniform                                  |              0.0044 |               0.0012 |               0.0051 |                0.0045 |
| rkvhash [lsh6]                                  |              0      |               0      |               0      |                0      |
| rkvhash [lsh8]                                  |              0      |               0      |               0      |                0      |
| rkvhash_lam0.1 [lsh8]                           |              0      |               0      |               0      |                0      |
| simbin [post] (oldest_first_evict)              |              0      |               0      |               0      |                0      |
| strat_prop [post] no-rw                         |              0.0023 |               0.0005 |               0.0057 |                0.0033 |
| strat_prop [post] rw                            |              0.004  |               0.0008 |               0.009  |                0.0039 |
| strat_prop [pre] no-rw                          |              0.0027 |               0.0015 |               0.0057 |                0.002  |
| strat_prop [pre] rw                             |              0.0012 |               0.0007 |               0.0116 |                0.0033 |
| strat_prop [pre_pos_norm] no-rw                 |              0.0012 |               0.0015 |               0.0036 |                0.0057 |
| strat_prop [pre_pos_norm] rw                    |              0.0026 |               0.001  |               0.0111 |                0.003  |
| topk_oracle                                     |              0      |               0      |               0      |                0      |

![error vs budget](t1_error_vs_budget.png)


# T0: R-KVHash scorer on real keys

<!-- provenance -->
- git commit: `4033bd94e6c18770596c2474032f13aaea34e63c` (dirty=True)
- GPU: NVIDIA A30; python 3.10.20; torch 2.11.0+cu130
- pip freeze: 193 packages, full list in `runs/pip_freeze.txt`

Scores replicate kvpress@dixi/rkv_lsh (bf16 normalize/projection/counts/softmax, lam=0 keeps top scores).
`max_bucket_bf16` is the bucket count after bf16 scatter_add (the code's own counts).
`tie_slots_x` = kept slots decided by torch.topk tie-breaking among equal bf16 scores.

## Medians per prompt type and bits

|                           |   ('long_doc', 6) |   ('long_doc', 8) |   ('reasoning', 6) |   ('reasoning', 8) |
|:--------------------------|------------------:|------------------:|-------------------:|-------------------:|
| n_cand                    |         5922      |         5922      |          2457      |          2457      |
| n_buckets                 |           61      |          159      |            63      |           195      |
| max_bucket_true           |          974      |          516.5    |           203      |           108      |
| max_bucket_bf16           |          256      |          256      |           203      |           108      |
| n_distinct_scores         |           14      |           17      |            14      |            16      |
| spearman_R_occupancy      |            0.1347 |            0.1512 |            -0.1234 |             0.2872 |
| tied_at_cutoff_10         |          857      |          632.5    |           224      |           280      |
| tie_slots_10              |          243      |          225      |            91      |           111      |
| evicted_singleton_frac_10 |            0.0002 |            0.0037 |             0.0004 |             0.0135 |
| singleton_frac_all        |            0.0003 |            0.0034 |             0.0005 |             0.0123 |
| mean_err_10               |            0.4459 |            0.4401 |             0.4547 |             0.4421 |
| p95_err_10                |            1.1315 |            1.1066 |             1.2591 |             1.25   |
| mass_recall_10            |            0.6155 |            0.6038 |             0.4921 |             0.4949 |
| mean_err_lam0.1_10        |            0.3172 |            0.3155 |             0.2109 |             0.211  |
| mass_recall_lam0.1_10     |            0.7739 |            0.7731 |             0.7642 |             0.7643 |
| tied_at_cutoff_25         |         1700      |         1501      |           376      |           421.5    |
| tie_slots_25              |          697      |          526.5    |           166      |           196.5    |
| evicted_singleton_frac_25 |            0.0002 |            0.0042 |             0      |             0.0161 |
| mean_err_25               |            0.272  |            0.2533 |             0.2645 |             0.2744 |
| p95_err_25                |            0.8325 |            0.7793 |             0.8094 |             0.82   |
| mass_recall_25            |            0.6687 |            0.6633 |             0.5898 |             0.564  |
| mean_err_lam0.1_25        |            0.1388 |            0.1392 |             0.0886 |             0.089  |
| mass_recall_lam0.1_25     |            0.8282 |            0.8276 |             0.8449 |             0.8452 |

# T-1 diagnostic: predicted Neyman gain (oracle S_h, held-out queries)

<!-- provenance -->
- git commit: `0f9c03b2dcdc28825dcdee5f62e310c869032cbc` (dirty=True)
- GPU: NVIDIA A30; python 3.10.20; torch 2.11.0+cu130
- pip freeze: 193 packages, full list in `runs/pip_freeze.txt`

gain = CV^2/(1+CV^2) of the size-weighted CV of S_h across strata; the predicted fractional variance
reduction of Neyman over proportional allocation (no finite-population correction).
`valvec` (extra, not in the brief) uses sqrt(trace Cov) of the vector p_i(v_i - o) instead of std of its norm.

## Medians by prompt type, strata variant, bits

|                                  |   gain_key |   gain_val |   gain_valvec |   cv_key |   cv_val |   n_strata |
|:---------------------------------|-----------:|-----------:|--------------:|---------:|---------:|-----------:|
| ('long_doc', 'post', 4)          |      0.49  |      0.184 |         0.186 |    0.98  |    0.475 |       16   |
| ('long_doc', 'post', 6)          |      0.598 |      0.275 |         0.284 |    1.22  |    0.617 |       57   |
| ('long_doc', 'post', 8)          |      0.676 |      0.361 |         0.378 |    1.446 |    0.751 |      171   |
| ('long_doc', 'pre', 4)           |      0.468 |      0.177 |         0.187 |    0.938 |    0.464 |       15   |
| ('long_doc', 'pre', 6)           |      0.586 |      0.274 |         0.283 |    1.191 |    0.615 |       43   |
| ('long_doc', 'pre', 8)           |      0.554 |      0.327 |         0.334 |    1.115 |    0.698 |      106   |
| ('long_doc', 'pre_pos_norm', 4)  |      0.722 |      0.519 |         0.516 |    1.611 |    1.039 |      108   |
| ('long_doc', 'pre_pos_norm', 6)  |      0.688 |      0.579 |         0.577 |    1.486 |    1.172 |      290.5 |
| ('long_doc', 'pre_pos_norm', 8)  |      0.714 |      0.624 |         0.628 |    1.579 |    1.289 |      575.5 |
| ('reasoning', 'post', 4)         |      0.413 |      0.142 |         0.136 |    0.839 |    0.407 |       16   |
| ('reasoning', 'post', 6)         |      0.567 |      0.269 |         0.265 |    1.145 |    0.606 |       61   |
| ('reasoning', 'post', 8)         |      0.635 |      0.39  |         0.379 |    1.32  |    0.799 |      179.5 |
| ('reasoning', 'pre', 4)          |      0.416 |      0.122 |         0.118 |    0.844 |    0.372 |       16   |
| ('reasoning', 'pre', 6)          |      0.418 |      0.21  |         0.204 |    0.848 |    0.515 |       45   |
| ('reasoning', 'pre', 8)          |      0.393 |      0.264 |         0.259 |    0.804 |    0.599 |      115   |
| ('reasoning', 'pre_pos_norm', 4) |      0.772 |      0.563 |         0.547 |    1.842 |    1.134 |      137   |
| ('reasoning', 'pre_pos_norm', 6) |      0.778 |      0.61  |         0.595 |    1.87  |    1.25  |      276.5 |
| ('reasoning', 'pre_pos_norm', 8) |      0.762 |      0.654 |         0.645 |    1.789 |    1.374 |      526   |

## Medians by prompt type and layer (all strata variants and bits pooled)

|                   |   gain_key |   gain_val |   gain_valvec |   cv_key |   cv_val |
|:------------------|-----------:|-----------:|--------------:|---------:|---------:|
| ('long_doc', 0)   |      0.055 |      0.049 |         0.055 |    0.242 |    0.227 |
| ('long_doc', 4)   |      0.935 |      0.312 |         0.371 |    3.804 |    0.673 |
| ('long_doc', 8)   |      0.145 |      0.148 |         0.147 |    0.411 |    0.416 |
| ('long_doc', 12)  |      0.249 |      0.243 |         0.243 |    0.575 |    0.567 |
| ('long_doc', 16)  |      0.355 |      0.559 |         0.554 |    0.742 |    1.125 |
| ('long_doc', 20)  |      0.833 |      0.486 |         0.48  |    2.23  |    0.972 |
| ('long_doc', 24)  |      0.861 |      0.444 |         0.434 |    2.487 |    0.893 |
| ('long_doc', 28)  |      0.926 |      0.566 |         0.563 |    3.529 |    1.142 |
| ('long_doc', 32)  |      0.962 |      0.679 |         0.684 |    5.018 |    1.454 |
| ('reasoning', 0)  |      0.251 |      0.235 |         0.244 |    0.579 |    0.554 |
| ('reasoning', 4)  |      0.205 |      0.215 |         0.202 |    0.507 |    0.524 |
| ('reasoning', 8)  |      0.52  |      0.279 |         0.268 |    1.042 |    0.622 |
| ('reasoning', 12) |      0.783 |      0.454 |         0.436 |    1.9   |    0.911 |
| ('reasoning', 16) |      0.782 |      0.409 |         0.397 |    1.895 |    0.832 |
| ('reasoning', 20) |      0.941 |      0.629 |         0.618 |    4.011 |    1.303 |
| ('reasoning', 24) |      0.446 |      0.288 |         0.284 |    0.898 |    0.636 |

## Medians by prompt type (pooled)

| prompt_type   |   gain_key |   gain_val |   gain_valvec |   cv_key |   cv_val |
|:--------------|-----------:|-----------:|--------------:|---------:|---------:|
| long_doc      |      0.613 |      0.401 |         0.404 |    1.258 |    0.818 |
| reasoning     |      0.632 |      0.366 |         0.357 |    1.309 |    0.76  |

## Gate C

Medians over pre / pre_pos_norm strata, all bits (the strata the Neyman methods use):

| prompt_type   |   gain_key |   gain_val |
|:--------------|-----------:|-----------:|
| long_doc      |      0.627 |      0.456 |
| reasoning     |      0.678 |      0.438 |

**Verdict: CONTINUE. Both rules fire: gain_val >= 0.15, and gain_val does not exceed gain_key by >= 0.03 (values add nothing to the predicted gain). The two gains measure different variances (attention weights vs output), so T1 runs both key and val S_h and lets measured error decide.**

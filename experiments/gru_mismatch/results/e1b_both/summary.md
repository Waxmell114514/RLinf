# E1 — GRU mismatch: aggregate

## Reproduction gate (trailing recall, %)

| interface | seed 42 | all seeds | reference | gate |
|---|---|---|---|---|
| both | 62.3 | 59.5 ± 3.9 | - | - |

Gate (trace_abs only): seed 42 within ±5.0 points of the source's number. Quick-mode runs are smoke tests and cannot pass it.

## Attribution ladder per interface × transform

Balanced accuracy, mean ± sd over seeds. `margin` = best hidden (max of P1, P1g) − selection floor (max of P0, P0g).

| interface | transform | n_samples | floor | perm_p95 | best_hidden (P1/P1g) | margin | P3 | P4 | P2 | P2r | C_cmd | C_out | C_eff | C_phase | n_seeds | seeds_readable | control_margin | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both | freeze | 6500 | 0.502 ± 0.003 | 0.511 ± 0.002 | 0.863 ± 0.010 | 0.362 ± 0.013 | 0.823 ± 0.011 | 0.828 ± 0.010 | 0.906 ± 0.020 | 0.925 ± 0.015 | 0.506 ± 0.006 | 0.886 ± 0.018 | 0.496 ± 0.005 | 0.501 ± 0.001 | 5 | 5 | 0.42351342380892065 | readable, but not beyond outcome-only C_out (novelty) |
| both | mirror | 6500 | 0.505 ± 0.004 | 0.512 ± 0.002 | 0.898 ± 0.007 | 0.393 ± 0.008 | 0.871 ± 0.009 | 0.889 ± 0.006 | 0.504 ± 0.008 | 0.955 ± 0.007 | 0.506 ± 0.006 | 0.507 ± 0.005 | 0.496 ± 0.005 | 0.501 ± 0.001 | 5 | 5 | 0.4497548946318264 | readable |
| both | swap | 6500 | 0.506 ± 0.004 | 0.510 ± 0.001 | 0.684 ± 0.030 | 0.177 ± 0.032 | 0.645 ± 0.013 | 0.648 ± 0.025 | 0.501 ± 0.007 | 0.887 ± 0.013 | 0.506 ± 0.006 | 0.502 ± 0.006 | 0.496 ± 0.005 | 0.501 ± 0.001 | 5 | 5 | 0.38023293939970326 | readable |

## Pre-registered readout

Rule: an arm is *readable* when, in ≥ 4 seeds, the best hidden probe beats the selection floor by ≥ 0.04 and its permutation p95; it is *uninformative* when P2r clears the floor by < 0.1.

Primary cell ('trace_abs', 'swap') not run.

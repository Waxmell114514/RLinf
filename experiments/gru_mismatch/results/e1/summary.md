# E1 — GRU mismatch: aggregate

## Reproduction gate (trailing recall, %)

| interface | seed 42 | all seeds | reference | gate |
|---|---|---|---|---|
| command | 5.8 | 15.1 ± 13.7 | - | - |
| none | 6.5 | 11.6 ± 7.4 | 12.3 | info only (outside ±5.0) |
| trace_abs | 56.5 | 53.6 ± 3.9 | 56.5 | PASS |
| trace_signed | 9.9 | 10.0 ± 4.6 | - | - |

Gate (trace_abs only): seed 42 within ±5.0 points of the source's number. Quick-mode runs are smoke tests and cannot pass it.

## Attribution ladder per interface × transform

Balanced accuracy, mean ± sd over seeds. `margin` = best hidden (max of P1, P1g) − selection floor (max of P0, P0g).

| interface | transform | n_samples | floor | perm_p95 | best_hidden (P1/P1g) | margin | P3 | P4 | P2 | P2r | C_cmd | C_out | C_eff | C_phase | n_seeds | seeds_readable | control_margin | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| command | freeze | 6500 | 0.504 ± 0.004 | 0.512 ± 0.002 | 0.832 ± 0.017 | 0.328 ± 0.018 | 0.785 ± 0.022 | 0.818 ± 0.020 | 0.917 ± 0.013 | 0.933 ± 0.010 | 0.506 ± 0.006 | 0.894 ± 0.011 | 0.496 ± 0.005 | 0.501 ± 0.001 | 5 | 5 | 0.42875839431557483 | readable, but not beyond outcome-only C_out (novelty) |
| command | mirror | 6500 | 0.506 ± 0.004 | 0.512 ± 0.002 | 0.874 ± 0.007 | 0.368 ± 0.006 | 0.839 ± 0.011 | 0.869 ± 0.012 | 0.504 ± 0.012 | 0.962 ± 0.003 | 0.506 ± 0.006 | 0.510 ± 0.007 | 0.496 ± 0.005 | 0.501 ± 0.001 | 5 | 5 | 0.4551904770544196 | readable |
| command | swap | 6500 | 0.505 ± 0.003 | 0.511 ± 0.002 | 0.640 ± 0.016 | 0.135 ± 0.018 | 0.629 ± 0.013 | 0.621 ± 0.012 | 0.502 ± 0.009 | 0.894 ± 0.009 | 0.506 ± 0.006 | 0.505 ± 0.003 | 0.496 ± 0.005 | 0.501 ± 0.001 | 5 | 5 | 0.38919094695216394 | readable |
| none | freeze | 6500 | 0.506 ± 0.007 | 0.512 ± 0.002 | 0.804 ± 0.008 | 0.297 ± 0.007 | 0.784 ± 0.008 | 0.800 ± 0.010 | 0.911 ± 0.006 | 0.931 ± 0.003 | 0.506 ± 0.006 | 0.889 ± 0.010 | 0.499 ± 0.003 | 0.501 ± 0.001 | 5 | 5 | 0.4246092154065464 | readable, but not beyond outcome-only C_out (novelty) |
| none | mirror | 6500 | 0.504 ± 0.005 | 0.512 ± 0.002 | 0.501 ± 0.004 | -0.003 ± 0.006 | 0.500 ± 0.003 | 0.499 ± 0.004 | 0.503 ± 0.008 | 0.961 ± 0.003 | 0.506 ± 0.006 | 0.507 ± 0.005 | 0.499 ± 0.003 | 0.501 ± 0.001 | 5 | 0 | 0.45671622620648444 | not readable |
| none | swap | 6500 | 0.501 ± 0.003 | 0.512 ± 0.002 | 0.501 ± 0.006 | 0.001 ± 0.005 | 0.500 ± 0.004 | 0.502 ± 0.007 | 0.502 ± 0.013 | 0.893 ± 0.004 | 0.506 ± 0.006 | 0.504 ± 0.007 | 0.499 ± 0.003 | 0.501 ± 0.001 | 5 | 0 | 0.3922296349754619 | not readable |
| trace_abs | freeze | 6500 | 0.506 ± 0.005 | 0.511 ± 0.002 | 0.834 ± 0.012 | 0.328 ± 0.015 | 0.821 ± 0.008 | 0.835 ± 0.005 | 0.910 ± 0.005 | 0.928 ± 0.006 | 0.506 ± 0.006 | 0.886 ± 0.006 | 0.499 ± 0.003 | 0.501 ± 0.001 | 5 | 5 | 0.4221708159703099 | readable, but not beyond outcome-only C_out (novelty) |
| trace_abs | mirror | 6500 | 0.506 ± 0.007 | 0.512 ± 0.001 | 0.504 ± 0.005 | -0.002 ± 0.011 | 0.502 ± 0.004 | 0.503 ± 0.007 | 0.501 ± 0.012 | 0.960 ± 0.002 | 0.506 ± 0.006 | 0.506 ± 0.006 | 0.499 ± 0.003 | 0.501 ± 0.001 | 5 | 0 | 0.454158353805605 | not readable |
| trace_abs | swap | 6500 | 0.508 ± 0.005 | 0.510 ± 0.002 | 0.507 ± 0.011 | -0.001 ± 0.012 | 0.499 ± 0.007 | 0.503 ± 0.004 | 0.502 ± 0.008 | 0.887 ± 0.007 | 0.506 ± 0.006 | 0.501 ± 0.003 | 0.499 ± 0.003 | 0.501 ± 0.001 | 5 | 0 | 0.37876963920015916 | not readable |
| trace_signed | freeze | 6500 | 0.506 ± 0.011 | 0.511 ± 0.001 | 0.807 ± 0.013 | 0.300 ± 0.014 | 0.788 ± 0.014 | 0.803 ± 0.014 | 0.921 ± 0.018 | 0.935 ± 0.013 | 0.506 ± 0.006 | 0.901 ± 0.019 | 0.502 ± 0.006 | 0.501 ± 0.001 | 5 | 5 | 0.429052759894491 | readable, but not beyond outcome-only C_out (novelty) |
| trace_signed | mirror | 6500 | 0.503 ± 0.007 | 0.512 ± 0.003 | 0.616 ± 0.018 | 0.113 ± 0.022 | 0.529 ± 0.011 | 0.570 ± 0.018 | 0.504 ± 0.016 | 0.962 ± 0.005 | 0.506 ± 0.006 | 0.506 ± 0.005 | 0.502 ± 0.006 | 0.501 ± 0.001 | 5 | 5 | 0.4588915340186217 | readable |
| trace_signed | swap | 6500 | 0.506 ± 0.004 | 0.511 ± 0.001 | 0.541 ± 0.009 | 0.036 ± 0.009 | 0.513 ± 0.009 | 0.521 ± 0.011 | 0.501 ± 0.009 | 0.899 ± 0.010 | 0.506 ± 0.006 | 0.501 ± 0.006 | 0.502 ± 0.006 | 0.501 ± 0.001 | 5 | 2 | 0.39381514777920834 | mixed |

## Pre-registered readout

Rule: an arm is *readable* when, in ≥ 4 seeds, the best hidden probe beats the selection floor by ≥ 0.04 and its permutation p95; it is *uninformative* when P2r clears the floor by < 0.1.

Primary cell `trace_abs` × `swap`: **not readable**.
Plan consequence: keeping the command is not enough -> a comparator is needed; E4 (world model as comparator) becomes the main experiment.

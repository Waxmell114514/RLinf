**INTERIM — Modal workspace spend limit hit 09:33-09:37 UTC; missing: e4_sft_long_swap_s0 and e4_grpo_long_swap_s0 collections, e4_grpo_object_swap_s0 linear, 10 of 12 MLP rounds (only sft goal_swap s1/s2 done). Every missing value is [TO VERIFY].**

# E4 report — SFT vs GRPO on the unchanged hijack→probe battery

Generated 2026-10-26T11:06:42Z by `e4_report.py` from `experiments/efference_probe/data`. Every number is read from a results file; † = transcribed from `reports/p0_wave2_report.md` because that run's analysis directory was not available here. `[TO VERIFY]` = not found, not estimated.

Rows: the existing wave-2 checkpoint for that suite (goal-traj1 on libero_goal; that suite's own SFT checkpoint elsewhere), Base-Lora (`RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`) and GRPO (`RLinf/RLinf-OpenVLAOFT-LIBERO-130`). Probe rungs are the best of 27 layer×pool cells ± fold sd, read against the same-family P0 floor.

## Pre-registered readout, applied

**R4/R5, mlp family (primary).** Base-Lora: at floor; GRPO: inconclusive (numbers missing). Contrast: (e) inconclusive: a checkpoint has no replicated verdict.

**R4/R5, linear family (secondary).** Base-Lora: at floor; GRPO: inconclusive (above floor but explained by C_dstates in a seed). Contrast: (e) inconclusive: a checkpoint has no replicated verdict.

**Per-run ladders (R1-R3, R11).** δ = 0.04 over the same-family floor; 'beyond observables' also needs ≥ C_dstates + δ and C_cmd at chance (R2).

- `main02`† linear: **at** (margins P1 +0.002, P3 -0.009, P4 -0.005) (wave-2 row; rule applied for reference)
- `main02`† mlp: **at** (margins P1 +0.006, P3 +0.008, P4 +0.011) (wave-2 row; rule applied for reference)
- `main02_mirror`† linear: **above** (margins P1 +0.046, P3 +0.047, P4 +0.076); beyond observables: none (C_dstates 0.623) (wave-2 row; rule applied for reference)
- `main02_mirror`† mlp: **above** (margins P1 +0.049, P3 +0.045, P4 +0.080); beyond observables: none (C_dstates 0.738) (wave-2 row; rule applied for reference)
- `main02_freeze`† linear: **at** (margins P1 +0.005, P3 +0.027, P4 +0.028) (wave-2 row; rule applied for reference)
- `main02_freeze`† mlp: **above** (margins P1 +0.022, P3 +0.053, P4 +0.101); beyond observables: none (C_dstates 0.825) (wave-2 row; rule applied for reference)
- `suite_spatial02`† linear: **at** (margins P1 +0.030, P3 +0.034, P4 +0.025) (wave-2 row; rule applied for reference)
- `suite_spatial02`† mlp: **at** (margins P1 +0.024, P3 +0.026, P4 +0.030) (wave-2 row; rule applied for reference)
- `suite_object01`† linear: **at** (margins P1 +0.024, P3 +0.023, P4 +0.034) (wave-2 row; rule applied for reference)
- `suite_long01`† linear: **at** (margins P1 +0.023, P3 +0.022, P4 +0.026) (wave-2 row; rule applied for reference)
- `e4_sft_goal_swap_s1` linear: **at** (margins P1 +0.018, P3 +0.016, P4 +0.031)
- `e4_sft_goal_swap_s1` mlp: **at** (margins P1 +0.011, P3 +0.020, P4 +0.026)
- `e4_grpo_goal_swap_s1` linear: **at** (margins P1 +0.035, P3 +0.029, P4 +0.036)
- `e4_sft_goal_swap_s2` linear: **at** (margins P1 +0.029, P3 +0.032, P4 +0.011)
- `e4_sft_goal_swap_s2` mlp: **at** (margins P1 +0.018, P3 +0.011, P4 -0.013)
- `e4_grpo_goal_swap_s2` linear: **above** (margins P1 +0.046, P3 +0.051, P4 +0.049); beyond observables: none (C_dstates 0.532)
- `e4_sft_goal_mirror_s1` linear: **above** (margins P1 +0.100, P3 +0.103, P4 +0.119); beyond observables: none (C_dstates 0.640)
- `e4_grpo_goal_mirror_s1` linear: **above** (margins P1 +0.149, P3 +0.156, P4 +0.170); beyond observables: ['P1', 'P3', 'P4']
- `e4_sft_goal_freeze_s1` linear: **above** (margins P1 +0.024, P3 +0.029, P4 +0.055); beyond observables: none (C_dstates 0.636)
- `e4_grpo_goal_freeze_s1` linear: **above** (margins P1 +0.119, P3 +0.112, P4 +0.209); beyond observables: ['P4']
- `e4_sft_goal_mirror_s2` linear: **above** (margins P1 +0.099, P3 +0.090, P4 +0.126); beyond observables: none (C_dstates 0.664)
- `e4_grpo_goal_mirror_s2` linear: **above** (margins P1 +0.138, P3 +0.146, P4 +0.179); beyond observables: ['P4']
- `e4_sft_goal_freeze_s2` linear: **above** (margins P1 +0.048, P3 +0.073, P4 +0.056); beyond observables: none (C_dstates 0.667)
- `e4_grpo_goal_freeze_s2` linear: **above** (margins P1 +0.074, P3 +0.074, P4 +0.158); beyond observables: none (C_dstates 0.672)
- `e4_sft_spatial_swap_s0` linear: **above** (margins P1 +0.056, P3 +0.056, P4 +0.056); beyond observables: ['P1', 'P3', 'P4'] (R10: single-seed suite)
- `e4_grpo_spatial_swap_s0` linear: **above** (margins P1 +0.050, P3 +0.057, P4 +0.050); beyond observables: ['P1', 'P3', 'P4'] (R10: single-seed suite)
- `e4_sft_object_swap_s0` linear: **above** (margins P1 +0.037, P3 +0.036, P4 +0.042); beyond observables: none (C_dstates 0.577) (R10: single-seed suite)

**R6 (mirror/freeze, MLP).** [TO VERIFY]: a Base-Lora mirror MLP round is missing.

**R7 (E3 migration).** not migrated (GRPO freeze reaction on 0/2 seeds, Base-Lora on 0/2).

**R8 (compensation).** does not strengthen on every arm (clean-probe gap Base-Lora->GRPO: +0.05->+0.06, +0.10->+0.07, +0.17->+0.17, +0.16->+0.27).

**R9 (E1 positive control).** Base-Lora s1: 0.471 (L28/act_mean); Base-Lora s2: 0.484 (L28/act_mean); GRPO s1: 0.670 (L28/act_mean); GRPO s2: 0.700 (L28/act_mean).

## libero_goal / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02` | 1 | 1266 | 0.541 | 0.543±0.035 | 0.532±0.023 | 0.536±0.011 | 0.711±0.038 | 0.488±0.035 | 0.475±0.019 | [TO VERIFY]: not in the wave-2 report | +0.0013 (0.527) | +0.145 (0.443) | 0.41 / 0.39 / 0.51 | 149 |
| Base-Lora | `e4_sft_goal_swap_s1` | 1 | 968 | 0.525 | 0.543±0.023 | 0.541±0.022 | 0.556±0.018 | 0.798±0.032 | 0.518±0.030 | 0.494±0.014 | 0.471 (L28/act_mean) | -0.0049 (0.157) | +0.756 (0.063) | 0.61 / 0.60 / 0.65 | 138 |
| GRPO | `e4_grpo_goal_swap_s1` | 1 | 952 | 0.528 | 0.563±0.033 | 0.557±0.035 | 0.564±0.040 | 0.672±0.035 | 0.460±0.041 | 0.506±0.029 | 0.670 (L28/act_mean) | -0.0234 (0.948) | -0.037 (0.383) | 0.93 / 0.92 / 0.98 | 247 |
| Base-Lora | `e4_sft_goal_swap_s2` | 2 | 948 | 0.526 | 0.555±0.028 | 0.558±0.025 | 0.537±0.017 | 0.827±0.016 | 0.537±0.049 | 0.527±0.037 | 0.484 (L28/act_mean) | +0.0167 (0.310) | -0.864 (0.166) | 0.61 / 0.60 / 0.69 | 128 |
| GRPO | `e4_grpo_goal_swap_s2` | 2 | 858 | 0.541 | 0.586±0.005 | 0.591±0.024 | 0.590±0.026 | 0.681±0.045 | 0.532±0.025 | 0.552±0.039 | 0.700 (L28/act_mean) | -0.0203 (0.422) | +0.023 (0.209) | 0.95 / 0.93 / 1.00 | 245 |

## libero_goal / swap / mlp

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02` | 1 | 1266 | 0.532 | 0.538±0.026 | 0.540±0.017 | 0.543±0.013 | 0.776±0.018 | 0.522±0.015 | 0.499±0.030 | n/a (linear-only) | +0.0013 (0.527) | +0.145 (0.443) | 0.41 / 0.39 / 0.51 | 149 |
| Base-Lora | `e4_sft_goal_swap_s1` | 1 | 968 | 0.525 | 0.536±0.047 | 0.545±0.037 | 0.552±0.018 | 0.846±0.023 | 0.541±0.042 | 0.508±0.043 | n/a (linear-only) | -0.0049 (0.157) | +0.756 (0.063) | 0.61 / 0.60 / 0.65 | 138 |
| GRPO | `e4_grpo_goal_swap_s1` | 1 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_swap_s2` | 2 | 948 | 0.538 | 0.556±0.018 | 0.549±0.023 | 0.525±0.023 | 0.856±0.026 | 0.529±0.050 | 0.521±0.040 | n/a (linear-only) | +0.0167 (0.310) | -0.864 (0.166) | 0.61 / 0.60 / 0.69 | 128 |
| GRPO | `e4_grpo_goal_swap_s2` | 2 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_goal / mirror / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_mirror` | 1 | 1226 | 0.521 | 0.567±0.028 | 0.568±0.029 | 0.597±0.011 | 0.857±0.036 | 0.623±0.040 | 0.515±0.021 | [TO VERIFY]: not in the wave-2 report | +0.0140 (0.016) | -1.939 (0.012) | 0.25 / 0.19 / 0.45 | 119 |
| Base-Lora | `e4_sft_goal_mirror_s1` | 1 | 968 | 0.534 | 0.634±0.030 | 0.637±0.032 | 0.652±0.031 | 0.906±0.011 | 0.640±0.025 | 0.446±0.016 | 0.465 (L32/act_mean) | +0.0387 (0.073) | +0.052 (0.765) | 0.51 / 0.47 / 0.65 | 112 |
| GRPO | `e4_grpo_goal_mirror_s1` | 1 | 1090 | 0.512 | 0.661±0.013 | 0.668±0.020 | 0.682±0.015 | 0.914±0.025 | 0.586±0.018 | 0.516±0.013 | 0.528 (L28/act_mean) | -0.0028 (0.948) | -0.834 (0.288) | 0.81 / 0.77 / 0.94 | 169 |
| Base-Lora | `e4_sft_goal_mirror_s2` | 2 | 902 | 0.528 | 0.627±0.031 | 0.618±0.036 | 0.654±0.056 | 0.912±0.031 | 0.664±0.035 | 0.495±0.014 | 0.469 (L32/act_mean) | -0.0237 (0.287) | +0.403 (0.967) | 0.51 / 0.48 / 0.64 | 114 |
| GRPO | `e4_grpo_goal_mirror_s2` | 2 | 1160 | 0.516 | 0.654±0.028 | 0.661±0.028 | 0.695±0.020 | 0.918±0.021 | 0.633±0.034 | 0.496±0.034 | 0.485 (L28/act_mean) | -0.0004 (0.357) | -1.968 (0.154) | 0.74 / 0.69 / 0.96 | 176 |

## libero_goal / mirror / mlp

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_mirror` | 1 | 1226 | 0.532 | 0.581±0.038 | 0.577±0.030 | 0.612±0.022 | 0.871±0.025 | 0.738±0.019 | 0.520±0.042 | n/a (linear-only) | +0.0140 (0.016) | -1.939 (0.012) | 0.25 / 0.19 / 0.45 | 119 |
| Base-Lora | `e4_sft_goal_mirror_s1` | 1 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_mirror_s1` | 1 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_mirror_s2` | 2 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_mirror_s2` | 2 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_goal / freeze / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_freeze` | 1 | 1056 | 0.538 | 0.543±0.030 | 0.565±0.017 | 0.566±0.045 | 0.886±0.035 | 0.594±0.052 | 0.503±0.018 | [TO VERIFY]: not in the wave-2 report | -0.0036 (0.956) | +0.713 (0.819) | 0.38 / 0.40 / 0.32 | 115 |
| Base-Lora | `e4_sft_goal_freeze_s1` | 1 | 760 | 0.530 | 0.554±0.049 | 0.560±0.053 | 0.585±0.043 | 0.885±0.024 | 0.636±0.024 | 0.494±0.043 | 0.493 (L32/act_mean) | -0.0165 (0.233) | -0.005 (0.633) | 0.59 / 0.61 / 0.55 | 108 |
| GRPO | `e4_grpo_goal_freeze_s1` | 1 | 806 | 0.510 | 0.629±0.089 | 0.622±0.057 | 0.719±0.071 | 0.942±0.018 | 0.591±0.042 | 0.478±0.031 | 0.703 (L28/act_mean) | -0.0547 (0.953) | +3.658 (0.795) | 0.93 / 0.92 / 0.94 | 168 |
| Base-Lora | `e4_sft_goal_freeze_s2` | 2 | 754 | 0.514 | 0.562±0.011 | 0.587±0.041 | 0.570±0.025 | 0.907±0.034 | 0.667±0.019 | 0.562±0.041 | 0.523 (L32/act_mean) | +0.0026 (0.887) | +0.942 (0.965) | 0.71 / 0.69 / 0.79 | 113 |
| GRPO | `e4_grpo_goal_freeze_s2` | 2 | 784 | 0.547 | 0.621±0.053 | 0.620±0.023 | 0.704±0.037 | 0.944±0.011 | 0.672±0.025 | 0.524±0.041 | 0.692 (L28/act_mean) | -0.0708 (0.836) | +3.331 (0.453) | 0.94 / 0.94 / 0.98 | 174 |

## libero_goal / freeze / mlp

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_freeze` | 1 | 1056 | 0.535 | 0.557±0.006 | 0.588±0.031 | 0.636±0.029 | 0.911±0.036 | 0.825±0.028 | 0.485±0.011 | n/a (linear-only) | -0.0036 (0.956) | +0.713 (0.819) | 0.38 / 0.40 / 0.32 | 115 |
| Base-Lora | `e4_sft_goal_freeze_s1` | 1 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_freeze_s1` | 1 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_freeze_s2` | 2 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_freeze_s2` | 2 | [TO VERIFY: missing: no mlp round] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_spatial / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spatial-traj1† | `suite_spatial02` | 0 | 1202 | 0.533 | 0.563±0.038 | 0.567±0.036 | 0.558±0.042 | 0.757±0.021 | 0.499±0.028 | 0.483±0.022 | [TO VERIFY]: not in the wave-2 report | +0.0044 (0.264) | -0.189 (0.295) | 0.47 / 0.44 / 0.58 | 138 |
| Base-Lora | `e4_sft_spatial_swap_s0` | 0 | 934 | 0.517 | 0.573±0.029 | 0.573±0.034 | 0.573±0.020 | 0.820±0.014 | 0.504±0.010 | 0.498±0.047 | 0.448 (L32/act_mean) | +0.0057 (0.477) | -0.251 (0.498) | 0.66 / 0.61 / 0.82 | 139 |
| GRPO | `e4_grpo_spatial_swap_s0` | 0 | 730 | 0.515 | 0.564±0.044 | 0.571±0.026 | 0.565±0.038 | 0.647±0.026 | 0.507±0.037 | 0.549±0.034 | 0.622 (L28/act_mean) | +0.0493 (0.029) | -5.202 (0.015) | 0.98 / 0.98 / 0.99 | 255 |

## libero_object / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object-traj1† | `suite_object01` | 0 | 1478 | 0.542 | 0.566±0.026 | 0.565±0.028 | 0.576±0.022 | 0.800±0.023 | 0.513±0.016 | 0.512±0.019 | [TO VERIFY]: not in the wave-2 report | +0.0128 (0.011) | -1.619 (0.0036) | 0.21 / 0.18 / 0.31 | 153 |
| Base-Lora | `e4_sft_object_swap_s0` | 0 | 1178 | 0.509 | 0.546±0.026 | 0.545±0.023 | 0.551±0.048 | 0.880±0.020 | 0.577±0.036 | 0.527±0.031 | 0.443 (L32/act_mean) | +0.0342 (0.144) | -1.157 (0.606) | 0.56 / 0.53 / 0.67 | 147 |
| GRPO | `e4_grpo_object_swap_s0` | 0 | [TO VERIFY: missing: no completed analysis directory] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_long / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| libero10-traj1† | `suite_long01` | 0 | 2354 | 0.523 | 0.546±0.013 | 0.545±0.014 | 0.549±0.026 | 0.774±0.018 | 0.512±0.016 | 0.488±0.017 | [TO VERIFY]: not in the wave-2 report | +0.0020 (0.428) | -0.473 (0.260) | 0.07 / 0.04 / 0.22 | 123 |
| Base-Lora | `e4_sft_long_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_long_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## Wall-clock ledger

| start (UTC) | kind | run | rc | wall (min) | outcome |
|---|---|---|---|---|---|
| 2026-10-26T04:12:35Z | collect | `e4_sft_smoke` | 1 | 1.7 | failed (None) |
| 2026-10-26T04:14:16Z | collect | `e4_grpo_smoke` | 1 | 1.5 | failed (None) |
| 2026-10-26T04:28:42Z | collect | `e4_sft_smoke_r2` | 0 | 1.0 | ok |
| 2026-10-26T04:29:43Z | collect | `e4_grpo_smoke_r2` | 0 | 0.8 | ok |
| 2026-10-26T04:36:05Z | collect | `e4_sft_pilot` | 0 | 2.3 | ok |
| 2026-10-26T04:38:21Z | probe-pilot | `e4_sft_pilot` | 0 | 0.2 | ok |
| 2026-10-26T04:38:37Z | collect | `e4_grpo_pilot` | 0 | 1.7 | ok |
| 2026-10-26T04:40:17Z | probe-pilot | `e4_grpo_pilot` | 0 | 0.3 | ok |
| 2026-10-26T06:29:56Z | collect | `e4_sft_pilot2` | 0 | 6.5 | ok |
| 2026-10-26T06:36:27Z | probe-pilot | `e4_sft_pilot2` | 0 | 2.1 | ok |
| 2026-10-26T06:38:35Z | collect | `e4_grpo_pilot2` | 0 | 4.7 | ok |
| 2026-10-26T06:43:15Z | probe-pilot | `e4_grpo_pilot2` | 0 | 1.3 | ok |
| 2026-10-26T06:52:55Z | collect | `e4_sft_goal_swap_s1` | 0 | 9.4 | ok |
| 2026-10-26T07:02:36Z | collect | `e4_grpo_goal_swap_s1` | 0 | 12.3 | ok |
| 2026-10-26T07:02:41Z | probe-linear | `e4_sft_goal_swap_s1` | 0 | 6.3 | ok |
| 2026-10-26T07:10:41Z | probe-mlp | `e4_sft_goal_swap_s1` | 0 | 112.0 | ok |
| 2026-10-26T07:15:08Z | collect | `e4_sft_goal_swap_s2` | 0 | 9.5 | ok |
| 2026-10-26T07:24:51Z | collect | `e4_grpo_goal_swap_s2` | 0 | 12.9 | ok |
| 2026-10-26T07:37:58Z | collect | `e4_sft_goal_mirror_s1` | 0 | 7.6 | ok |
| 2026-10-26T07:41:21Z | probe-linear | `e4_grpo_goal_swap_s1` | 0 | 18.7 | ok |
| 2026-10-26T07:41:23Z | probe-linear | `e4_grpo_goal_swap_s2` | 0 | 16.3 | ok |
| 2026-10-26T07:41:24Z | probe-linear | `e4_sft_goal_swap_s2` | 0 | 5.8 | ok |
| 2026-10-26T07:45:45Z | collect | `e4_grpo_goal_mirror_s1` | 0 | 11.1 | ok |
| 2026-10-26T07:47:20Z | probe-linear | `e4_sft_goal_mirror_s1` | 0 | 6.8 | ok |
| 2026-10-26T07:47:25Z | probe-mlp | `e4_sft_goal_swap_s2` | 0 | 102.1 | ok |
| 2026-10-26T07:57:01Z | collect | `e4_sft_goal_freeze_s1` | 0 | 6.7 | ok |
| 2026-10-26T07:57:30Z | probe-linear | `e4_grpo_goal_mirror_s1` | 0 | 11.8 | ok |
| 2026-10-26T08:03:52Z | collect | `e4_grpo_goal_freeze_s1` | 0 | 9.3 | ok |
| 2026-10-26T08:05:37Z | probe-linear | `e4_sft_goal_freeze_s1` | 0 | 14.3 | ok |
| 2026-10-26T08:13:18Z | collect | `e4_sft_goal_mirror_s2` | 0 | 7.5 | ok |
| 2026-10-26T08:13:42Z | probe-linear | `e4_grpo_goal_freeze_s1` | 0 | 7.3 | ok |
| 2026-10-26T08:21:01Z | collect | `e4_grpo_goal_mirror_s2` | 0 | 11.4 | ok |
| 2026-10-26T08:21:42Z | probe-linear | `e4_sft_goal_mirror_s2` | 0 | 6.2 | ok |
| 2026-10-26T08:32:36Z | collect | `e4_sft_goal_freeze_s2` | 0 | 6.7 | ok |
| 2026-10-26T08:33:54Z | probe-linear | `e4_grpo_goal_mirror_s2` | 0 | 10.3 | ok |
| 2026-10-26T08:39:27Z | collect | `e4_grpo_goal_freeze_s2` | 0 | 9.1 | ok |
| 2026-10-26T08:39:58Z | probe-linear | `e4_sft_goal_freeze_s2` | 0 | 6.0 | ok |
| 2026-10-26T08:48:44Z | collect | `e4_sft_spatial_swap_s0` | 0 | 10.0 | ok |
| 2026-10-26T08:50:07Z | probe-linear | `e4_grpo_goal_freeze_s2` | 0 | 13.6 | ok |
| 2026-10-26T08:58:57Z | collect | `e4_grpo_spatial_swap_s0` | 0 | 12.0 | ok |
| 2026-10-26T09:00:25Z | probe-linear | `e4_sft_spatial_swap_s0` | 0 | 5.8 | ok |
| 2026-10-26T09:11:10Z | collect | `e4_sft_object_swap_s0` | 0 | 10.3 | ok |
| 2026-10-26T09:12:21Z | probe-linear | `e4_grpo_spatial_swap_s0` | 0 | 8.5 | ok |
| 2026-10-26T09:21:44Z | collect | `e4_grpo_object_swap_s0` | 0 | 11.7 | ok |
| 2026-10-26T09:22:27Z | probe-linear | `e4_sft_object_swap_s0` | 0 | 9.8 | ok |

Collection (GPU) wall: **2.96 h**; probe (CPU) wall: **6.09 h**.

## Low-competence task flags (R12; kept in every analysis)

- `e4_sft_goal_swap_s1`: none
- `e4_grpo_goal_swap_s1`: [5]
- `e4_sft_goal_swap_s2`: none
- `e4_grpo_goal_swap_s2`: [5]
- `e4_sft_goal_mirror_s1`: none
- `e4_grpo_goal_mirror_s1`: none
- `e4_sft_goal_freeze_s1`: none
- `e4_grpo_goal_freeze_s1`: none
- `e4_sft_goal_mirror_s2`: none
- `e4_grpo_goal_mirror_s2`: none
- `e4_sft_goal_freeze_s2`: none
- `e4_grpo_goal_freeze_s2`: none
- `e4_sft_spatial_swap_s0`: none
- `e4_grpo_spatial_swap_s0`: none
- `e4_sft_object_swap_s0`: none

## Readout in plain words

This is an interim readout. The lines below restate what `e4_report.py` printed above, without new numbers.

- **R4/R5, MLP family (primary).** Base-Lora: at floor. GRPO: inconclusive (numbers missing). Contrast: (e) inconclusive: a checkpoint has no replicated verdict. **Blocked by missing MLP rounds:** GRPO's verdict needs the MLP rounds on `e4_grpo_goal_swap_s1` and `e4_grpo_goal_swap_s2`, and neither finished. So the primary pre-registered contrast cannot be decided from this data.
- **R4/R5, linear family (secondary).** Base-Lora: at floor. GRPO: inconclusive (above floor but explained by C_dstates in a seed). Contrast: (e) inconclusive: a checkpoint has no replicated verdict. In the per-run ladders above, `e4_grpo_goal_swap_s1` linear is at floor and `e4_grpo_goal_swap_s2` linear is above floor with "beyond observables: none (C_dstates 0.532)"; under R4 that is not a replicated positive.
- **R6 (mirror/freeze, MLP).** [TO VERIFY]: a Base-Lora mirror MLP round is missing. **Blocked:** all eight mirror/freeze MLP rounds are missing. The linear ladders for these arms are reported above, but R6 counts only through the MLP family's R2 test.
- **R7 (E3 migration).** not migrated (GRPO freeze reaction on 0/2 seeds, Base-Lora on 0/2).
- **R8 (compensation).** does not strengthen on every arm (clean-probe gap Base-Lora->GRPO: +0.05->+0.06, +0.10->+0.07, +0.17->+0.17, +0.16->+0.27).
- **R9 (E1 positive control).** Base-Lora s1: 0.471 (L28/act_mean); Base-Lora s2: 0.484 (L28/act_mean); GRPO s1: 0.670 (L28/act_mean); GRPO s2: 0.700 (L28/act_mean). All four are above R9's 0.30 withholding threshold.
- **R10 (suites, single seed, linear).** Spatial and object are reported for Base-Lora, and spatial for GRPO. GRPO object and both long runs are missing, so the suites cannot corroborate anything for GRPO beyond spatial.

## Surprises / could not verify

- **Spend limit.** Every collection and probe stopped when the Modal *workspace* spend limit was exceeded, sometime between 09:33 (the last ledger write, `e4_grpo_object_swap_s0` collect) and 09:37 UTC. This is the account's own cap, separate from the operator's $80 SPEND_CAP. The workspace's billed total is [TO VERIFY] on the Modal dashboard. By the operator formula, GPU is 2.96 h x $2.81 = ~$8.3. CPU is the 6.09 h of completed probe wall in the ledger, plus ten MLP containers killed in flight, started between 07:55:25 and 09:04:11 UTC by the driver lines, so ~0.5-1.7 h each before the stop; they are not in the ledger [TO VERIFY]. Running the 12 goal-suite MLP rounds in parallel is the most likely reason the limit was reached so early [interpretation].
- **GRPO P2r is low at pilot2 and in every GRPO swap run.** 0.613 at pilot2 (sft 0.772), and in the swap tables above GRPO P2r is 0.672 / 0.681 (goal), 0.647 (spatial) against Base-Lora 0.798 / 0.827 / 0.820. R3 still calls those runs informative, but the margin is smaller. GRPO's median intervention strength at pilot2 was also lower on most tasks (0.23-1.12 vs sft 0.58-1.37; per-task values in `e4_gates/s1b_*.json`). A plausible reading is that GRPO's actions are already close to the donor's, so a swap changes less [interpretation, not tested].
- **Wall clocks ran at 0.2-0.4x the expected values.** Smokes took 50-60 s against 683 s expected. S2 collections took 6.7-12.9 min against 23-66 min after k/k_base scaling (ledger table above). The wave-2 `expected_s` was measured on a slower box with a slower startup. Because the speed factor was locked at its floor of 1.0, every watchdog sat at 2x a generous expectation and never came close to firing.
- **Memory.** Peak tree RSS (summed over the collector and its env workers, which can double-count shared pages) was 60.1-62.5 GB on the goal runs and 65.05-65.77 GB on spatial and object, against the old 64 GiB (68.7 GB) container limit. The limit was raised to 96 GiB before S2, with the billed request unchanged. The cgroup's own peak usage was not recorded [TO VERIFY]. The ledger's `peak_vram_mib` read 9 MiB on the sub-minute smokes, so it looks undersampled on very short runs; it read ~25.5 GB on every longer run.
- **GRPO pilot2 P0's best cell is degenerate.** 0.500 with sd 0.000 at L0/ctx_last, which is a constant prediction. Its mean over cells, 0.451, only just clears the band's lower edge.
- **`e4_grpo_goal_swap_s1` MLP round started twice.** Driver lines at 08:01:29 and 08:16:57, the second after renaming the first's half-written `analysis_mlp` to `analysis_mlp.partial_20261026T081657Z` (kept). Only one orchestrator was running, and it never respawns a call, so this was most likely Modal re-running the input after the first container died (preemption or memory). Which one is [TO VERIFY]. Neither attempt finished before the spend limit.
- **Logbook part lost to a path.** The `init` part was written under a nested path (`.../e4_logbook/<stamp>_cpu_init_sft_/vol/ckpts/sft.md`), which `merge-logbook`'s `*.md` glob does not see. It was merged by hand from a flat local copy, so it appears at the end of the logbook rather than in time order. The label is now sanitized.
- **Driver orchestration on Modal.** The orchestrator failed twice before it ran unattended, both times in `modal_e4.py`, and both are fixed and logged: modal 1.6's builtin `TimeoutError`, then a detached `.remote()` cancelled when the local client was killed. Collections were never affected; linear rounds started up to ~26 min late (`e4_grpo_goal_swap_s1` collected 07:14:56, linear started 07:41:21).
- **R12 flags.** GRPO task 5 on both goal swap seeds. Base-Lora none on S2 (task 7 at pilot2). Flagged tasks are kept in every analysis.
- **Not collected or not probed.** See the banner. `e4_sft_long_swap_s0` has a run directory on the Volume with no completion marker; a resume collects it under a new id and leaves the old one untouched.

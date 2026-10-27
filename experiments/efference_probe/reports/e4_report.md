**INTERIM, closed with the data in hand (operator decision, 2026-10-27).** Missing, so every value for these is [TO VERIFY]:
- `e4_grpo_goal_swap_s1` MLP: three Modal launches, all preempted or on hosts past the pace rule; no result.
- All eight goal mirror/freeze MLP rounds (R6).
- `e4_grpo_long_swap_s0` linear: stopped at its 1600 s watchdog with 199/200 permutations done; the Amendment 2 re-run was not run.

# E4 report — SFT vs GRPO on the unchanged hijack→probe battery

Generated 2026-10-27T13:29:05Z by `e4_report.py` from a client-side copy of the Modal Volume's results (everything but `hidden/` and `frames/`). Every number is read from a results file; † = transcribed from `reports/p0_wave2_report.md` because that run's analysis directory was not available here. `[TO VERIFY]` = not found, not estimated.

Rows: the existing wave-2 checkpoint for that suite (goal-traj1 on libero_goal; that suite's own SFT checkpoint elsewhere), Base-Lora (`RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`) and GRPO (`RLinf/RLinf-OpenVLAOFT-LIBERO-130`). Probe rungs are the best of 27 layer×pool cells ± fold sd, read against the same-family P0 floor.

## Pre-registered readout, applied

**R4/R5, mlp family (primary).** Base-Lora: at floor; GRPO: inconclusive (numbers missing). Contrast: (e) inconclusive: a checkpoint has no replicated verdict.

**R4/R5, linear family (secondary).** Base-Lora: at floor; GRPO: inconclusive (above floor but explained by outcome perception in a seed: C_cmd off chance). Contrast: (e) inconclusive: a checkpoint has no replicated verdict.

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
- `e4_grpo_goal_swap_s2` linear: **above** (margins P1 +0.046, P3 +0.051, P4 +0.049); beyond observables: none (C_cmd 0.552 outside [0.45, 0.55]; rungs clear C_dstates 0.532 + δ)
- `e4_grpo_goal_swap_s2` mlp: **above** (margins P1 +0.064, P3 +0.060, P4 +0.051); beyond observables: none (C_dstates 0.568)
- `e4_sft_goal_mirror_s1` linear: **above** (margins P1 +0.100, P3 +0.103, P4 +0.119); beyond observables: none (C_dstates 0.640; C_cmd 0.446 outside [0.45, 0.55])
- `e4_grpo_goal_mirror_s1` linear: **above** (margins P1 +0.149, P3 +0.156, P4 +0.170); beyond observables: ['P1', 'P3', 'P4']
- `e4_sft_goal_freeze_s1` linear: **above** (margins P1 +0.024, P3 +0.029, P4 +0.055); beyond observables: none (C_dstates 0.636)
- `e4_grpo_goal_freeze_s1` linear: **above** (margins P1 +0.119, P3 +0.112, P4 +0.209); beyond observables: ['P4']
- `e4_sft_goal_mirror_s2` linear: **above** (margins P1 +0.099, P3 +0.090, P4 +0.126); beyond observables: none (C_dstates 0.664)
- `e4_grpo_goal_mirror_s2` linear: **above** (margins P1 +0.138, P3 +0.146, P4 +0.179); beyond observables: ['P4']
- `e4_sft_goal_freeze_s2` linear: **above** (margins P1 +0.048, P3 +0.073, P4 +0.056); beyond observables: none (C_dstates 0.667; C_cmd 0.562 outside [0.45, 0.55])
- `e4_grpo_goal_freeze_s2` linear: **above** (margins P1 +0.074, P3 +0.074, P4 +0.158); beyond observables: none (C_dstates 0.672)
- `e4_sft_spatial_swap_s0` linear: **above** (margins P1 +0.056, P3 +0.056, P4 +0.056); beyond observables: ['P1', 'P3', 'P4'] (R10: single-seed suite)
- `e4_grpo_spatial_swap_s0` linear: **above** (margins P1 +0.050, P3 +0.057, P4 +0.050); beyond observables: ['P1', 'P3', 'P4'] (R10: single-seed suite)
- `e4_sft_object_swap_s0` linear: **above** (margins P1 +0.037, P3 +0.036, P4 +0.042); beyond observables: none (C_dstates 0.577) (R10: single-seed suite)
- `e4_grpo_object_swap_s0` linear: **above** (margins P1 +0.062, P3 +0.059, P4 +0.053); beyond observables: none (C_dstates 0.531) (R10: single-seed suite)
- `e4_sft_long_swap_s0_r2` linear: **at** (margins P1 +0.021, P3 +0.021, P4 +0.033) (R10: single-seed suite)

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
| GRPO | `e4_grpo_goal_swap_s2` | 2 | 858 | 0.527 | 0.591±0.027 | 0.586±0.023 | 0.578±0.012 | 0.705±0.025 | 0.568±0.058 | 0.513±0.049 | n/a (linear-only) | -0.0203 (0.422) | +0.023 (0.209) | 0.95 / 0.93 / 1.00 | 245 |

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
| GRPO | `e4_grpo_object_swap_s0` | 0 | 1082 | 0.505 | 0.567±0.036 | 0.564±0.027 | 0.558±0.033 | 0.625±0.038 | 0.531±0.042 | 0.510±0.030 | 0.596 (L28/act_mean) | +0.0247 (0.591) | -1.835 (0.457) | 0.97 / 0.96 / 1.00 | 284 |

## libero_long / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| libero10-traj1† | `suite_long01` | 0 | 2354 | 0.523 | 0.546±0.013 | 0.545±0.014 | 0.549±0.026 | 0.774±0.018 | 0.512±0.016 | 0.488±0.017 | [TO VERIFY]: not in the wave-2 report | +0.0020 (0.428) | -0.473 (0.260) | 0.07 / 0.04 / 0.22 | 123 |
| Base-Lora | `e4_sft_long_swap_s0_r2` | 0 | 1974 | 0.515 | 0.536±0.029 | 0.536±0.016 | 0.548±0.020 | 0.844±0.015 | 0.505±0.023 | 0.481±0.028 | 0.390 (L32/act_mean) | +0.0394 (0.015) | -1.990 (0.0075) | 0.33 / 0.30 / 0.44 | 123 |
| GRPO | `e4_grpo_long_swap_s0` | 0 | [TO VERIFY: missing: no completed analysis directory] |  |  |  |  |  |  |  |  |  |  |  |  |

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
| 2026-10-26T14:42:13Z | probe-linear | `e4_grpo_object_swap_s0` | 0 | 14.6 | ok |
| 2026-10-26T14:42:25Z | collect | `e4_sft_long_swap_s0_r2` | 0 | 16.6 | ok |
| 2026-10-26T15:01:00Z | collect | `e4_grpo_long_swap_s0` | 0 | 22.7 | ok |
| 2026-10-26T15:01:41Z | probe-linear | `e4_sft_long_swap_s0_r2` | 0 | 25.1 | ok |
| 2026-10-26T15:25:33Z | probe-linear | `e4_grpo_long_swap_s0` | -2 | 26.8 | failed |
| 2026-10-27T08:51:50Z | probe-mlp | `e4_grpo_goal_swap_s2` | 0 | 209.8 | ok |

Collection (GPU) wall: **3.62 h**; probe (CPU) wall: **10.69 h**.

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
- `e4_grpo_object_swap_s0`: none
- `e4_sft_long_swap_s0_r2`: [4, 8]

## Readout in plain words

The lines below restate what `e4_report.py` printed above; they add no numbers.

- **R4/R5, MLP family (primary).**
  - Base-Lora is at floor on both goal-swap seeds.
  - GRPO is inconclusive. The generator says "numbers missing" because the s1 MLP round is absent, but s2 alone already fixes the verdict:
    - s2 is above floor (margins P1 +0.064, P3 +0.060, P4 +0.051);
    - s2 is informative (P2r 0.705);
    - no rung reaches C_dstates (0.568) + δ, so R2 calls it outcome perception;
    - under R4 a seed explained by C_dstates makes the checkpoint inconclusive, whatever s1 would have shown.
  - Contrast: (e) inconclusive.
- **R4/R5, linear family (secondary).**
  - Base-Lora is at floor on both seeds.
  - GRPO s1 is at floor (margins +0.035/+0.029/+0.036, all below δ).
  - GRPO s2 is above floor, and its rungs clear C_dstates 0.532 + δ. But its C_cmd is 0.552, just outside [0.45, 0.55], so R2 calls it explained by outcome perception.
  - Contrast: (e) inconclusive.
- **R6 (mirror/freeze, MLP).** Blocked: all eight MLP rounds are missing.
  - In the linear family, which R6 does not count, GRPO is beyond observables on mirror s1 (P1/P3/P4), mirror s2 (P4) and freeze s1 (P4); Base-Lora is beyond observables on none.
  - That is the only place the GRPO-vs-SFT pattern points toward (a), and it would count only through the MLP R2 test on both seeds.
- **R7 (E3 migration).** Not migrated: GRPO freeze reacts on 0/2 seeds, Base-Lora on 0/2.
- **R8 (compensation).** Does not strengthen on every arm.
- **R9 (E1 positive control).** All four goal-swap runs are above 0.30, so no verdict is withheld.
- **R10 (suites, single seed, linear; corroborate only).**
  - Spatial: both checkpoints are above floor and beyond observables.
  - Object: both are above floor and explained by C_dstates (Base-Lora 0.577, GRPO 0.531).
  - Long: Base-Lora (`_r2`) is at floor; GRPO is missing.
  - With R5 at (e) there is nothing for the suites to corroborate.

## What this means for the paper's hypotheses [interpretation, not a pre-registered verdict]

- **The RL-attribution hypothesis is not supported on the primary arm.**
  - The paper asks whether anything inside the policy registers that an executed action was not its own.
  - Its pre-registered RL comparison expects an RL-post-trained sibling to show this beyond what observables explain, if RL builds attribution-relevant representation.
  - On the stealthy swap arm, the one seed with a complete GRPO MLP ladder rises above floor, while Base-Lora stays at floor on both seeds. But an observables-only probe on Δs reaches 0.568 in the same family, within δ of every rung (P1 − C_dstates = +0.023).
  - What GRPO adds over SFT is, at this detection limit, readable from the outcome alone: the paper's T5 pattern, now on a policy trained on the consequences of its own actions.
- **It is not the clean (b) either.**
  - GRPO is not at floor: on s2 its ladder separates from floor in both families, which Base-Lora's never does on goal swap (on spatial, single seed and linear only, both checkpoints are above floor and beyond observables).
  - The pre-registration keeps this as "inconclusive / not replicated" and never as a positive. Its R2 call rests on C_dstates in the MLP family, whose fold sd is 0.058, and on C_cmd only 0.002 past the band edge in the linear family. Both are close calls.
- **The open question is the mirror/freeze arms (R6).**
  - There, GRPO's linear rungs exceed C_dstates + δ on three of four arm×seed cells and Base-Lora's on none.
  - If that held under the MLP twin on both seeds, it would be the evidence for (a) that swap could not give.
  - Until those rounds run, it is a lead, not a result.

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
- **R2 reason labels corrected after generation (`e4_report.py` fix, same commit).** The generator printed every above-floor run that failed R2 as "none (C_dstates …)" and the R4 verdict as "explained by C_dstates", even when the failing R2 condition was C_cmd off its [0.45, 0.55] chance band. Verdicts are unchanged; only the reasons are relabelled, using the C_cmd values already in the tables: `e4_grpo_goal_swap_s2` linear fails on C_cmd alone (0.552, its rungs clear C_dstates + δ), and `e4_sft_goal_mirror_s1` (C_cmd 0.446) and `e4_sft_goal_freeze_s2` (C_cmd 0.562) fail on both conditions. The lines were edited by hand because the Volume data is not readable from this session; a regeneration on resume prints the same labels.
- **Second Modal workspace.** After the first workspace's spend limit, the Volume (9151 files, 47.50 GB) was copied client-side to a new workspace and verified file by file by size (`e4_copy_volume.py`). `init` re-registered the checkpoints there, re-checking their sha256, because `e4_state.json` recorded the old Volume's mount path.
- **Preemption made long CPU rounds unreliable on Modal.** On the new workspace, CPU probe containers were preempted six times: four times with Modal's "Container terminated due to preemption" message, and two simultaneous deaths at ~15:00 UTC on 2026-10-26 [cause TO VERIFY]. An MLP round cannot checkpoint, so each preemption restarted it from zero. `e4_grpo_goal_swap_s1` lost an attempt at ~70% of its round (12:43 UTC, 2026-10-27).
- **Host speed varied ~2x.** Preemptible 24-CPU hosts on the new workspace ran 1.9-2.8x the old workspace's SFT reference pace, by stage. A non-preemptible host ran 2.3x slower per core while using 23.1 of its 24 cores, at 3x the price, and was stopped after 30 min. Amendment 3 (logbook) relaxed the pace line to 3x and the MLP watchdog to 7 h for the last s1 launch only.
- **A launcher change of mine halved the probe CPUs.** The 12-core limit in b4963f0 gave probe containers 12 CPUs instead of the 24 a bare `cpu=8.0` had given, because one Modal core is one container CPU. It is now 24.
- **Orchestrator gap.** After a preemption, Modal can mark a call FAILURE in its call graph while `get(timeout=0)` still raises `TimeoutError`, so the orchestrator kept that call's slot. This is documented in `modal_e4.py`.
- **libero_long linear rounds need more time.** The SFT long round took 1506.6 s against the 1600 s watchdog. GRPO's was stopped at 1600 s with 199/200 permutations done, an estimated ~1800-2000 s round. Amendment 2 sets long-suite linear rounds to 1600 s expected; the re-run was not done.
- **Spend.** The new workspace's dashboard read $7.11 when the first resume was stopped (operator). List-price estimates since then:
  - non-preemptible try ~$2.0;
  - two s1 pace-rule stops ~$0.6 each;
  - s2 ~$3.9;
  - s1 launch 3 ~$3.6;
  - total ~$17.8 [TO VERIFY on the dashboard].
- **R12.** `e4_sft_long_swap_s0_r2` flags tasks 4 and 8 (kept in every analysis). The run was collected as `_r2` because the interrupted first attempt's directory exists and is kept.

# E4 report — SFT vs GRPO on the unchanged hijack→probe battery

Generated 2026-10-26T03:32:41Z by `e4_report.py` from `experiments/efference_probe/data`. Every number is read from a results file; † = transcribed from `reports/p0_wave2_report.md` because that run's analysis directory was not available here. `[TO VERIFY]` = not found, not estimated.

Rows: the existing wave-2 checkpoint for that suite (goal-traj1 on libero_goal; that suite's own SFT checkpoint elsewhere), Base-Lora (`RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`) and GRPO (`RLinf/RLinf-OpenVLAOFT-LIBERO-130`). Probe rungs are the best of 27 layer×pool cells ± fold sd, read against the same-family P0 floor.

## Pre-registered readout, applied

**R4/R5, mlp family (primary).** Base-Lora: inconclusive (a seed is missing); GRPO: inconclusive (a seed is missing). Contrast: (e) inconclusive: a checkpoint has no replicated verdict.

**R4/R5, linear family (secondary).** Base-Lora: inconclusive (a seed is missing); GRPO: inconclusive (a seed is missing). Contrast: (e) inconclusive: a checkpoint has no replicated verdict.

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

**R6 (mirror/freeze, MLP).** [TO VERIFY]: a Base-Lora mirror seed is missing.

**R7 (E3 migration).** [TO VERIFY]: a freeze arm is missing.

**R8 (compensation).** [TO VERIFY]: a swap/mirror arm or its probe/clean success split is missing.

**R9 (E1 positive control).** Base-Lora s1: [TO VERIFY]; Base-Lora s2: [TO VERIFY]; GRPO s1: [TO VERIFY]; GRPO s2: [TO VERIFY].

## libero_goal / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02` | 1 | 1266 | 0.541 | 0.543±0.035 | 0.532±0.023 | 0.536±0.011 | 0.711±0.038 | 0.488±0.035 | 0.475±0.019 | [TO VERIFY]: not in the wave-2 report | +0.0013 (0.527) | +0.145 (0.443) | 0.41 / 0.39 / 0.51 | 149 |
| Base-Lora | `e4_sft_goal_swap_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_swap_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_swap_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_swap_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_goal / swap / mlp

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02` | 1 | 1266 | 0.532 | 0.538±0.026 | 0.540±0.017 | 0.543±0.013 | 0.776±0.018 | 0.522±0.015 | 0.499±0.030 | n/a (linear-only) | +0.0013 (0.527) | +0.145 (0.443) | 0.41 / 0.39 / 0.51 | 149 |
| Base-Lora | `e4_sft_goal_swap_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_swap_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_swap_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_swap_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_goal / mirror / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_mirror` | 1 | 1226 | 0.521 | 0.567±0.028 | 0.568±0.029 | 0.597±0.011 | 0.857±0.036 | 0.623±0.040 | 0.515±0.021 | [TO VERIFY]: not in the wave-2 report | +0.0140 (0.016) | -1.939 (0.012) | 0.25 / 0.19 / 0.45 | 119 |
| Base-Lora | `e4_sft_goal_mirror_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_mirror_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_mirror_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_mirror_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_goal / mirror / mlp

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_mirror` | 1 | 1226 | 0.532 | 0.581±0.038 | 0.577±0.030 | 0.612±0.022 | 0.871±0.025 | 0.738±0.019 | 0.520±0.042 | n/a (linear-only) | +0.0140 (0.016) | -1.939 (0.012) | 0.25 / 0.19 / 0.45 | 119 |
| Base-Lora | `e4_sft_goal_mirror_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_mirror_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_mirror_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_mirror_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_goal / freeze / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_freeze` | 1 | 1056 | 0.538 | 0.543±0.030 | 0.565±0.017 | 0.566±0.045 | 0.886±0.035 | 0.594±0.052 | 0.503±0.018 | [TO VERIFY]: not in the wave-2 report | -0.0036 (0.956) | +0.713 (0.819) | 0.38 / 0.40 / 0.32 | 115 |
| Base-Lora | `e4_sft_goal_freeze_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_freeze_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_freeze_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_freeze_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_goal / freeze / mlp

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goal-traj1† | `main02_freeze` | 1 | 1056 | 0.535 | 0.557±0.006 | 0.588±0.031 | 0.636±0.029 | 0.911±0.036 | 0.825±0.028 | 0.485±0.011 | n/a (linear-only) | -0.0036 (0.956) | +0.713 (0.819) | 0.38 / 0.40 / 0.32 | 115 |
| Base-Lora | `e4_sft_goal_freeze_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_freeze_s1` | 1 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| Base-Lora | `e4_sft_goal_freeze_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_goal_freeze_s2` | 2 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_spatial / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spatial-traj1† | `suite_spatial02` | 0 | 1202 | 0.533 | 0.563±0.038 | 0.567±0.036 | 0.558±0.042 | 0.757±0.021 | 0.499±0.028 | 0.483±0.022 | [TO VERIFY]: not in the wave-2 report | +0.0044 (0.264) | -0.189 (0.295) | 0.47 / 0.44 / 0.58 | 138 |
| Base-Lora | `e4_sft_spatial_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_spatial_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_object / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object-traj1† | `suite_object01` | 0 | 1478 | 0.542 | 0.566±0.026 | 0.565±0.028 | 0.576±0.022 | 0.800±0.023 | 0.513±0.016 | 0.512±0.019 | [TO VERIFY]: not in the wave-2 report | +0.0128 (0.011) | -1.619 (0.0036) | 0.21 / 0.18 / 0.31 | 153 |
| Base-Lora | `e4_sft_object_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_object_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## libero_long / swap / linear

| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| libero10-traj1† | `suite_long01` | 0 | 2354 | 0.523 | 0.546±0.013 | 0.545±0.014 | 0.549±0.026 | 0.774±0.018 | 0.512±0.016 | 0.488±0.017 | [TO VERIFY]: not in the wave-2 report | +0.0020 (0.428) | -0.473 (0.260) | 0.07 / 0.04 / 0.22 | 123 |
| Base-Lora | `e4_sft_long_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |
| GRPO | `e4_grpo_long_swap_s0` | 0 | [TO VERIFY: missing: not collected] |  |  |  |  |  |  |  |  |  |  |  |  |

## Wall-clock ledger

[TO VERIFY]: no ledger records under `experiments/efference_probe/data/e4_ledger`.

## Low-competence task flags (R12; kept in every analysis)



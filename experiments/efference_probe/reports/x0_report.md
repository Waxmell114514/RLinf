**Exploratory.** X0 re-analyses the hidden states E4 already collected. Its analyses were pre-registered in `logbook.md` before any of their numbers existed:
- "X0 PRE-REGISTRATION" (2026-10-28 03:24) covers A1–A7, plus the implementation notes that follow it;
- A2b and A8 were added mid-run, each pre-registered before its own numbers;
- "X0 FOLLOW-UP PRE-REGISTRATION F1-F7" (2026-10-28 15:04) covers F1–F7.

No R1–R12 verdict changes. Every number is read from `reports/x0_results/` (F1–F7 in `followup/`); the logbook entries carry the same numbers with their file names. Cost: $0 Modal, this container's CPU only.

**Revised** after Fable's review (`next_steps_fable_x0.md`) and F1–F7. The first version overreached on three points; the corrections are folded in below:
- **"comparator missing":** a single-image policy never has the command and the outcome in the same forward pass;
- **"the policy is a state estimator":** the state is readable from the input embedding on, and RL's advantage is mostly a data effect;
- **"GRPO s1's P1 excursion":** it is at floor by R1.

# X0 — what the E4 hidden states say about the mechanism

## Bottom line

1. **There is no efference increment anywhere.**
   - |P3 − P1| < 0.04 in all 30 run × family rows, wave-2 rows included (A1).
   - The paired bootstrap CI covers 0 in 16 of 17 E4 runs (A7). The exception, Base-Lora freeze s2 at +0.025, sits within the ~0.01 BLAS resolution (F1).
2. **The end-effector state is in the image embedding, and the last motion is there only through it.**
   - h(c+1) reads out s(c+1) at R² 0.81–0.97 (A2). At layer 0, the projected image embedding plus fixed prompt, it is already 0.86–0.97; later layers add at most 0.015.
   - A per-task k-NN on the true 8-dim end state predicts the last chunk's Δs as well as h(c+1) (A2b). Adding h(c+1) on top gains at most 0.009 R² overall and 0.005 on hijacked calls (F2: "no motion trace beyond the end-effector state", 4 of 4).
3. **Command and outcome never meet in one forward pass, and a two-call external reader can partly recombine them.**
   - The command is linearly decodable from h(c) (E1) and the outcome from h(c+1) (A2).
   - P2r's comparison features built from both decodes detect the swap on Base-Lora 5 of 5 runs (0.602–0.680). On GRPO they do so on 3 of 4, with s2 passing by 0.007 and s1 missing (A3).
   - The decodes are lossy on both sides. Replacing only the outcome, or only the command, with its decode costs 0.10–0.18 against true inputs on Base-Lora (F3: "both decodes lossy").
   - So the activations hold a degraded, linearly recoverable version of each ingredient. They do not hold the comparison, and the architecture never puts both ingredients in one forward pass.
4. **GRPO's P1 margins are small and end-state-borne.**
   - GRPO's linear P1 margin over floor is +0.035 to +0.062, larger than Base-Lora's on 3 of 4 matched cells (sign test p = 0.31; F6).
   - The one goal-swap margin beyond δ, GRPO s2, drops from 0.586 to 0.549 (floor 0.541) once a per-task linear end-state fit is removed (F5: state-explained). GRPO s1 (+0.035) is at floor by R1, and F5 is indeterminate there.
   - The margins sit at L4–L20 (A5). They do not track outcome visibility (A4), novelty (A6), phase (F6, C_phase ≤ floor + 0.034) or low-competence tasks (F7).
5. **RL narrows the state distribution; it does not sharpen the state code.**
   - GRPO's state readout error is about half Base-Lora's on all calls.
   - On matched support the gap shrinks to 21–38% of that (F4: "data effect", both cells, both seed pairs).

## Results by analysis

| Analysis | Test | Pre-registered prediction / rule | Result | Reading |
|---|---|---|---|---|
| A1 | efference increment P3 − P1 | \|·\| < 0.04 everywhere | max +0.031 (wave-2), +0.025 (E4) | holds |
| A2 | s(c+1), Δs from h(c+1) | premise: (i) ≥ 0.5, (ii) < 0.3; M2 if (i) < 0.3 | (i) 0.81–0.97, (ii) 0.40–0.85 | M2 rejected; (ii) explained by A2b/F2 |
| A2b, F2 | h(c+1) vs k-NN on true s(c+1), then residual | no motion trace iff gain < 0.05 (4 of 4) | gain ≤ 0.009 (all), ≤ 0.005 (HIJACK) | **no motion trace beyond the end state** |
| A3 | decoded-ingredients oracle | ≥ floor + 0.04 on ≥ 3 of 4 goal swap | 3 of 4; Base-Lora 5/5, GRPO 3/4 (one marginal, one miss) | a two-call reader recovers it; not an internal comparator |
| F3 | one side decoded, one true | outcome or command side ≥ P2r − 0.10 on ≥ 4 of 5 Base-Lora | outcome 1/5, command 0/5 | **both decodes lossy** |
| A4 | Mahalanobis(Δs) AUROC | GRPO > Base-Lora on both seeds | s1 0.520 < 0.528; s2 0.563 > 0.539 | fails |
| A5 | top-3 P1 cells | GRPO at layer ≥ 24 or `ctx_last` | 0 of 9 (L4–L20) | falsified |
| A6 | ρ(Δnovelty, Δentropy) | pooled ρ > 0, CI excludes 0 | +0.026 [−0.003, +0.055] | fails |
| A7 | paired bootstrap | CI covering 0 = not separable | P3 − P1 covers 0 in 16/17 | see A1 |
| A8, F5 | end-state controls; residualised P1 | F5: state-explained iff P1_resid ≤ floor + 0.02 | GRPO s2, Base-Lora s2 state-explained; s1 runs indeterminate (at floor) | margins beyond δ are end-state-borne |
| F4 | state RMSE, matched support | data effect iff matched gap ≤ 0.5 × all-calls gap | ratios 0.21–0.38 | **data effect** |
| F1 | BLAS resolution | descriptive | ≤ 0.011 pooled, ≤ 0.054 single fold | Methods sentence |
| F6 | C_phase; margin sign test | descriptive | C_phase 0.474–0.561; GRPO > Base-Lora 3/4, p = 0.31 | phase does not carry P1 |
| F7 | per-task margin, hit vs strength | flag if flagged tasks carry > 0.5 | flag only on Base-Lora long (0.58); strength effect only on object | not competence-driven |

## Details

**What h(c+1) knows** (A2, A2b, F2; best-cell uniform R² over the 8 state dims, 4000 calls per run):

| run | s(c+1), best | s(c+1), L0 | Δs ← h(c+1) | Δs ← k-NN(true s(c+1)) | F2 gain, all / HIJACK |
|---|---|---|---|---|---|
| GRPO goal s1 | 0.972 | 0.965 | 0.834 | 0.826 | +0.008 / +0.003 |
| GRPO goal s2 | 0.965 | 0.961 | 0.850 | 0.840 | +0.007 / +0.005 |
| Base-Lora goal s1 | 0.881 | 0.873 | 0.584 | 0.577 | +0.009 / −0.007 |
| Base-Lora goal s2 | 0.875 | 0.860 | 0.591 | 0.580 | +0.005 / +0.002 |
| GRPO spatial | 0.946 | — | 0.780 | — | — |
| Base-Lora spatial | 0.921 | — | 0.557 | — | — |
| GRPO object | 0.960 | — | 0.761 | — | — |
| Base-Lora object | 0.916 | — | 0.557 | — | — |
| Base-Lora long | 0.811 | — | 0.395 | — | — |

**F4 — state readout error at L0** (physical units):
- Translation RMSE is 1.5–2.1 cm for Base-Lora and 0.9–1.3 cm for GRPO.
- The nRMSE gap (pooled-SD scale) shrinks from 0.151–0.165 on all calls to 0.044–0.049 on matched support.
- Caveat: only 321–327 of Base-Lora's 4000 calls lie inside GRPO's state support, against 1706–1863 of GRPO's inside Base-Lora's.

**Decoded comparison** (A3, F3; linear, P2r's features; floor = best-of-27 P0):

| run | A3: â + Δŝ | F3: true a + Δŝ | F3: â + true Δs | P2r (true) | floor + 0.04 | best of P1/P3/P4 |
|---|---|---|---|---|---|---|
| Base-Lora goal s1 | 0.602 | 0.688 | 0.645 | 0.798 | 0.565 | 0.556 |
| Base-Lora goal s2 | 0.606 | 0.678 | 0.666 | 0.827 | 0.566 | 0.558 |
| Base-Lora spatial | 0.627 | 0.711 | 0.656 | 0.820 | 0.557 | 0.573 |
| Base-Lora object | 0.680 | 0.784 | 0.734 | 0.880 | 0.549 | 0.551 |
| Base-Lora long | 0.608 | 0.666 | 0.690 | 0.844 | 0.555 | 0.548 |
| GRPO goal s1 | 0.514 | 0.589 | 0.581 | 0.672 | 0.568 | 0.564 |
| GRPO goal s2 | 0.588 | 0.607 | 0.611 | 0.681 | 0.581 | 0.591 |
| GRPO spatial | 0.583 | 0.594 | 0.586 | 0.647 | 0.555 | 0.571 |
| GRPO object | 0.577 | 0.578 | 0.578 | 0.625 | 0.545 | 0.567 |

- **Base-Lora.** A two-stage reader (supervised decode, then the fixed comparison) beats every one-stage probe on the same samples by +0.046 to +0.129. Each decoded side alone loses 25–52% of the true-input comparison's margin over chance.
- **What P3 − P1 = 0 means.** Part of it is a one-stage probe's sample complexity at n ≈ 1k, rather than the absence of any outcome code. The rest is decode loss on both sides.
- **GRPO.** Its comparison has less to work with (P2r 0.625–0.681 against 0.798–0.880), and its one-stage probes already match the two-stage reader.

**A7 — paired episode-cluster bootstrap** (B = 2000; every refit reproduces its CSV's fold scores and selected C exactly, in one of three BLAS settings):
- **P3 − P1:** the CI covers 0 in 16 of 17 linear runs and in the GRPO s2 MLP run.
- **P1 − C_dstates, goal swap linear:** GRPO s1 +0.103 [+0.056, +0.149], GRPO s2 +0.055 [+0.008, +0.102]. Base-Lora covers 0; GRPO s2 MLP +0.023 [−0.019, +0.065].
- **Caveat:** GRPO s1's C_dstates is 0.460, below chance, so that difference is not evidence that P1 is beyond observables.
- **Fixed reading:** a CI conditional on the selected cell is optimistic, and an excluding CI is not by itself a positive.

## What the paper can now say

> A single-image VLA cannot tell that its last action was overridden, and the reason is structural: the command exists only in the forward pass that produced it and the outcome only in the next. The robot's end-effector state is linearly present from the projected image embedding on (R² 0.86–0.97), and the last motion only through it (adding the hidden state to a k-NN on the true end state gains ≤ 0.009 R²). Adding the command to the post-action state changes nothing a probe can read (P3 − P1 within ±0.04 in 30/30 rows). A two-stage reader with access to both calls (a supervised decode of each, then a fixed comparison) detects the override where one-stage probes on the same samples do not, but each decode loses a quarter to a half of the margin a comparison of true quantities carries. RL post-training narrows the state distribution, which makes the state more readable; it adds no efference increment within a 0.04 detection limit on one SFT→GRPO pair, and its small anomaly margins are carried by the end state. Override detection on these policies therefore has to be external, or come from an input channel that carries the command.

## What remains open

- **One SFT→GRPO pair.** The RL statements need a second pair (X1 in `next_steps_fable_x0.md`).
- **8-dim end state only.** The "observables" controls see eef pose and gripper; the camera also sees objects and arm posture. Spatial is "beyond observables" for both checkpoints (linear, one seed) with no explanation. Object poses and joint positions are not logged (`capture_vision: false` in every config).
- **GRPO goal-swap s1 has no MLP ladder.**
- **Suites have one seed.**
- **Best-cell statistics favour P1** (max over 27 cells).

## Process notes

- **BLAS.** Probe rounds ran on Modal hosts with different BLAS kernels and thread counts. Refits here reproduce each CSV exactly in one of three settings, and the reproducing one is used per run (`a7_select`, dce6ba8; F3/F5/F7 run in children under that setting). The resolution is ≤ 0.011 on pooled differences and ≤ 0.054 on a single fold (F1).
- **F5's two conditions overlap when P1 is within 0.04 of floor.** The first code version labelled such runs by check order; they are now reported as indeterminate (4bbeeaf, before the F5 log entry).
- **Lost compute time.** Background jobs hit the session's time limits four times, and one waiting loop matched its own command line. Partial runs that restarted cost roughly 3–4 h of compute [estimate]; no results were lost.
- **Not in the original X0 plan:** A2b, A8 and F1–F7, each pre-registered before computing. A4's |AUROC − 0.5| correlation is post hoc.

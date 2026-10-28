**Exploratory.** X0 re-analyses the hidden states E4 already collected. Its analyses were pre-registered in `logbook.md` before any of their numbers existed:
- "X0 PRE-REGISTRATION" (2026-10-28 03:24) covers A1–A7, plus the implementation notes that follow it;
- A2b and A8 were added mid-run, each pre-registered before its own numbers.

No R1–R12 verdict changes. Every number below is read from the result files in `reports/x0_results/` (a copy of the data root's `x0/`); the logbook entries carry the same numbers with their file names. Cost: $0 Modal, this container's CPU only.

# X0 — what the E4 hidden states say about the mechanism

## Bottom line

1. **There is no efference increment anywhere.**
   - |P3 − P1| < 0.04 in all 30 run × family rows, wave-2 rows included (A1).
   - The paired bootstrap CI of P3 − P1 covers 0 in 16 of 17 E4 runs (A7).
2. **The policy is a high-fidelity state estimator and is memoryless in substance.**
   - h(c+1) reads out the current robot state at R² 0.81–0.97 (A2).
   - It also reads out the last chunk's motion Δs at R² 0.40–0.85, which looked like memory.
   - A per-task k-NN on the true current state predicts that motion just as well, on hijacked calls too (A2b). h(c+1) knows how the arm moved only through where it now is.
3. **The ingredients of a comparator are present; the comparator is not.**
   - Decoded command and decoded outcome, fed to P2r's comparison features, detect the hijack at floor + 0.04 or better on 3 of 4 goal-swap runs (A3; pre-registered reading "M1"). All 5 suite runs clear it too, so 8 of 9 swap runs do.
   - On Base-Lora, the decoded comparison beats the model's own best linear rung (P1, P3 or P4) by +0.046 to +0.129 in every run. On GRPO it adds −0.05 to +0.01.
4. **Three candidate explanations of GRPO's above-floor P1 fail their pre-registered tests:**
   - late-layer re-planning (A5: GRPO's excursion sits at L4–L20, not late);
   - higher outcome visibility (A4: GRPO > Base-Lora on one seed of two);
   - novelty-driven reaction (A6: pooled ρ = +0.026, 95% CI [−0.003, +0.055]).
5. **What GRPO s1's P1 excursion is made of is still open.** It is not the robot's own end state (A8: per-task state probes at 0.49 against P1 0.563). The scene, such as object positions, is the remaining observable candidate and was not tested.

## Results by mechanism

M1–M4 are the candidate mechanisms of `next_steps_fable.md` §B.

| Mechanism | Test | Pre-registered prediction | Result | Verdict |
|---|---|---|---|---|
| M1, no efference channel | A1, efference increment | \|P3 − P1\| < 0.04 everywhere | max +0.031 (wave-2), +0.025 (E4) | holds |
| M1 | A3, decoded-ingredients oracle | ≥ floor + 0.04 on ≥ 3 of 4 goal swap | 3 of 4 (0.602, 0.606, 0.588; GRPO s1 0.514); suites 5 of 5 | **M1: ingredients present, comparator missing** |
| M1 | A2 (iii), Δs from h(c) ⊕ h(c+1) | ≥ 0.5 | 0.58–0.90 | compatible |
| M2, outcome not metric | A2 (i), s(c+1) from h(c+1) | M2 if < 0.3 | 0.81–0.97 | **rejected** |
| Memoryless premise | A2 (ii), Δs from h(c+1) | < 0.3 | 0.40–0.85 | fails as stated; A2b explains it |
| (A2b) motion beyond state | H vs per-task k-NN on true s(c+1) | — | H − kNN on all calls +0.008 to +0.014; on HIJACK −0.040 to +0.001 | regularity explains A2 (ii) in 4 of 4 |
| M3, anomaly visibility | A4, Mahalanobis(Δs) AUROC | GRPO > Base-Lora on both seeds | s1 0.520 < 0.528; s2 0.563 > 0.539 | fails |
| M3, late locus | A5, top-3 P1 cells | GRPO at layer ≥ 24 or `ctx_last` | 0 of 9 (all L4–L20) | **falsified, reversed** |
| M4, novelty | A6, ρ(Δnovelty, Δentropy) | pooled ρ > 0, CI excludes 0 | +0.026 [−0.003, +0.055] | fails |
| (A8) end-state control | C_state vs P1 | — | GRPO s1 P1 beyond state; others mixed | open |

## Details

**A2/A2b — what h(c+1) knows.** Best-cell uniform R² over the 8 state dims (4000 calls per run):

| run | s(c+1) | Δs ← h(c+1) | Δs ← h(c) ⊕ h(c+1) | Δs ← k-NN(true s(c+1)), A2b out-of-fold | HIJACK calls only: h vs k-NN (A2b) |
|---|---|---|---|---|---|
| GRPO goal s1 | 0.972 | 0.834 | 0.890 | 0.826 | 0.666 vs 0.673 |
| GRPO goal s2 | 0.965 | 0.850 | 0.899 | 0.840 | 0.718 vs 0.717 |
| Base-Lora goal s1 | 0.881 | 0.584 | 0.706 | 0.577 | 0.299 vs 0.339 |
| Base-Lora goal s2 | 0.875 | 0.591 | 0.692 | 0.580 | 0.336 vs 0.351 |
| GRPO spatial | 0.946 | 0.780 | 0.878 | — | — |
| Base-Lora spatial | 0.921 | 0.557 | 0.755 | — | — |
| GRPO object | 0.960 | 0.761 | 0.864 | — | — |
| Base-Lora object | 0.916 | 0.557 | 0.765 | — | — |
| Base-Lora long | 0.811 | 0.395 | 0.579 | — | — |

GRPO's states are more readable than Base-Lora's in every suite. A stronger state readout is something RL training buys; an efference read is not.

**A3 — decoded-ingredients oracle** (linear, P2r's features on decodes; floor = best-of-27 P0):

| run | variant A | variant B | floor + 0.04 | P1 | P3 | P2r (true) |
|---|---|---|---|---|---|---|
| Base-Lora goal s1 | **0.602** | 0.567 | 0.565 | 0.543 | 0.541 | 0.798 |
| GRPO goal s1 | 0.514 | 0.505 | 0.568 | 0.563 | 0.557 | 0.672 |
| Base-Lora goal s2 | **0.606** | 0.515 | 0.566 | 0.555 | 0.558 | 0.827 |
| GRPO goal s2 | **0.588** | 0.574 | 0.581 | 0.586 | 0.591 | 0.681 |
| Base-Lora spatial | **0.627** | 0.551 | 0.557 | 0.573 | 0.573 | 0.820 |
| GRPO spatial | **0.583** | 0.548 | 0.555 | 0.564 | 0.571 | 0.647 |
| Base-Lora object | **0.680** | 0.593 | 0.549 | 0.546 | 0.545 | 0.880 |
| GRPO object | **0.577** | 0.559 | 0.545 | 0.567 | 0.564 | 0.625 |
| Base-Lora long | **0.608** | 0.563 | 0.555 | 0.536 | 0.536 | 0.844 |

**Base-Lora.** A comparison computed from the model's own decodable signals beats every linear read of the model by +0.046 to +0.129 in all five runs. The information to tell "I did this" from "this was done to me" is in the activations; nothing in the network combines it.

**GRPO.** Its linear read already matches the decoded comparison, within −0.05 to +0.01. Its comparison also has less to work with (P2r 0.625–0.681 against 0.798–0.880), consistent with a tighter on-policy distribution.

**A7 — paired episode-cluster bootstrap** (B = 2000; every refit reproduces its CSV's fold scores and selected C exactly):
- **P3 − P1:** the CI covers 0 in 16 of 17 linear runs and in the GRPO s2 MLP run. The exception is Base-Lora freeze s2, +0.025 [+0.006, +0.046].
- **P1 − C_dstates on goal swap:**
  - GRPO s1 +0.103 [+0.056, +0.149] and s2 +0.055 [+0.008, +0.102] (linear);
  - GRPO s2 MLP +0.023 [−0.019, +0.065];
  - Base-Lora +0.026 and +0.018, both covering 0.
- **Fixed reading:** a CI conditional on the selected cell is optimistic, and an excluding CI is not by itself a positive.

## What the paper can now say

> The policies are memoryless state estimators: a single observation's embedding reads out the robot state almost perfectly (R² up to 0.97), and how the arm last moved only through where it now is. The command that produced the motion is also linearly present (E1). A comparison computed from these two decodable signals detects externally imposed motion on 8 of 9 runs, but the network never computes it. Its own representation, read with or without the command, sits at the chance floor, and the efference increment P3 − P1 is zero in every run, checkpoint and suite. RL fine-tuning sharpens the state estimate (GRPO's motion readout 0.76–0.85 against 0.40–0.59) but adds no comparator. Its above-floor anomaly signal sits in early layers, does not track outcome visibility, and is not novelty-driven.

That turns the null into a located deficit: the policy has the ingredients but lacks the comparison. It points straight at the design experiments:
- **X3 / X6:** feed the command back in.
- **X4:** an external ten-line comparator closing the loop.

## What X0 does not settle

- **GRPO s1's P1 excursion** is beyond its robot-state controls (A8). The scene, such as object displacement by the donor chunk, is the candidate. Testing it needs object poses, which are not logged; a scene-only control would need new collection.
- **GRPO goal-swap s1 has no MLP ladder** (three preempted Modal attempts), so A5's and A7's primary-family checks rest on s2 alone.
- **Suites have one seed.**
- **Best-cell statistics favour P1** (max over 27 cells) in A7 and A8.

## Process notes

- **Reproduction depended on BLAS.** The Modal probe rounds ran on hosts with different BLAS kernels and thread counts, and a refit here missed its CSV by one or two boundary samples per fold. A7 was therefore run under three BLAS settings, and the reproducing one is used per run (`a7_select`, commit dce6ba8). The first pass, at this container's 4-thread AVX-512 default, is kept in `x0/a7_passes/avx512_4t/`.
- **Lost compute time:**
  - Background jobs hit the session's 30-minute or 2-hour limits four times, and one waiting loop matched its own command line, which delayed a download by an hour. The partial runs that restarted from scratch cost roughly 3–4 h of compute [estimate]. No results were lost.
  - A CLI choice was missing for A2b (`5bc4eb1`).
- **Not pre-registered** in the original X0 plan:
  - A2b and A8 were added after A2 and pre-registered before computing;
  - A4's |AUROC − 0.5| correlation is post hoc and labelled so in the logbook.

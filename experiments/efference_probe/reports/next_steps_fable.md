# Memo: from "a null result" to an RSS paper

*Advisory memo written by Fable on 2026-10-28, from a read-only pass over the repo. It is not a pre-registered verdict. Claims were spot-checked against the repo:*
- *E1 R² 0.639/0.638 (`paper_icml/main.tex`);*
- *the P3 − P1 values (`reports/e4_report.md` tables);*
- *`_with_previous_chunk` (`origin/main:vram_probe.py`);*
- *`use_proprio` (`rlinf/models/embodiment/openvla_oft/official`);*
- *GRPO's `value_head` at the `ctx_last` position (logbook line ~1018, `capture.py`).*

All numbers below are quoted from `experiments/efference_probe/paper_icml/main.tex`, `reports/e4_report.md`, the logbook rules (`E4 PRE-REGISTERED READOUT`, R1–R12) and, on `origin/main`, `PREREG_wave3.md` and the `E1 DONE` / `E1b DONE` logbook entries. Anything marked **[inference]** is mine.

## A. The "so what"

### Thesis an RSS paper can stand on
A deployed VLA cannot tell that its last action was overridden. This is a *design* fact, not a training accident:
- **What the representation holds:** the motor plan (E1 R² = 0.639/0.638, same cell on two seeds) and the outcome's appearance (mirror/freeze probes clear floor exactly as far as the observables twin C_Δs allows).
- **What it never holds:** the outcome *in a form comparable to the command* (P3 − P1 ≈ 0 in every run, both families).
- **The comparison itself is cheap:** a 256-unit MLP computes it from the raw ingredients at 0.687, against 0.463 linear.
- **RL post-training on the policy's own consequences (GRPO from Base-Lora) does not buy it either:**
  - Base-Lora is at floor on both goal-swap seeds in both families.
  - GRPO's one completed MLP seed rises to P1 0.591 against floor 0.527, but its observables twin reaches 0.568, inside δ = 0.04, and P3/P4 (0.586/0.578) add nothing over P1.
- **The toy agent says what *would* buy it:** a signed, instantaneous copy of the command on the input (GRU swap +0.135, mirror +0.368, 5/5 seeds), not a magnitude trace (−0.001). One state can carry self and mismatch together (E1b: swap +0.177, trailing recall 59.5%).

Roboticists should care for two reasons:
1. Failure or override detection on these policies must be external, or must come from an added input channel, and the paper says which channel and in what format.
2. The audit costs under 5 GPU-h per checkpoint and is pre-registered end to end.

### Framings
**Defensible:**
- "VLAs do not represent their own agency; here is the audit and here is the input-format requirement for fixing it."
- "RL post-training does not create self-attribution within a stated detection limit."

**Overclaims to drop:**
- "Novelty-gated compensation" as a mechanism (see D: the mirror reaction did not replicate on either E4 checkpoint).
- "RL builds attribution", in any form.
- Anything that reads the GRU as evidence about 7B VLAs rather than as a format hypothesis.

## B. Mechanistic explanation

The key logical point is already implicit in `results.md` ("Q1's null is the expected outcome") and should become the paper's spine:
- Because the checkpoints are memoryless and vision-only per call, h(t+1) is a deterministic function of (o(t+1), instruction). *Anything* P1 decodes is outcome perception by construction.
- P4 = h(t) ⊕ h(t+1) is bounded by what two observations contain.
- The only rung that can carry efference is P3 (a ⊕ h(t+1)), so the informative statistic is the **efference increment P3 − P1**.

From `e4_report.md`, the efference increment is −0.005 for GRPO goal-swap s2 MLP; the four mirror/freeze cells below are linear-family:

| run | P3 − P1 |
|---|---|
| GRPO goal-swap s2, MLP | −0.005 |
| GRPO mirror s1 | +0.007 |
| GRPO freeze s1 | −0.007 |
| GRPO mirror s2 | +0.007 |
| GRPO freeze s2 | −0.001 |
| wave-2 replication, MLP | +0.002 |

It is the cleanest zero in the dataset, and it is never reported as such.

### M1 — no efference channel (architectural)
The command never re-enters the network, so no comparator can form.
- **Predicts:** supplying the command as *input* (not at the probe) makes swap readable beyond C_Δs; supplying |a| or proprioception alone does not.
- **Discriminators:** E5-frozen and E5 (section C, X3 and X6). The GRU already supports the format half (`command` readable, `trace_abs` not).

### M2 — outcome not represented metrically, so the comparison has nothing to grip
P3 MLP is at floor while P2 MLP on raw (a, Δs) reaches 0.687, so h(t+1) does not expose Δs in a command-comparable form *at this n*.

The cheapest discriminator uses existing hidden states and takes CPU-minutes:
1. A ridge readout of s(t+1) and Δs from h(t+1) and from h(t) ⊕ h(t+1) (the same `ridge_readout` as E1).
2. A **decoded-ingredients oracle**: within each outer fold, fit â = ridge(h(t)) and Δŝ = ridge(h(t+1)) on the training split, build P2r's comparison features from (â, Δŝ), and score the hijack label.

How to read the outcome:

| result | meaning |
|---|---|
| The oracle clears floor | The ingredients are linearly present and the missing piece is the comparator: M1 wins. |
| The oracle sits at floor while Δs R² is high | The outcome is present but too coarse for a same-task swap. |
| Δs R² is low | M2 wins outright. |

This also settles the dimensionality objection in Limitations.

### M3 — GRPO's increment is anomaly or progress, not authorship
GRPO's episodes are short and near-deterministic:
- success 0.93–0.98 against 0.61;
- E1 R² 0.67–0.70 against 0.47–0.48;
- P2r lower, 0.672/0.681 against 0.798/0.827;
- intervention strength lower at pilot2.

**[inference]** A tighter on-policy state distribution makes any deviation, even a same-task donor chunk, more visible from the outcome alone. The GRPO checkpoint also carried a value head at the `ctx_last` position, which shapes h toward "on track vs off track".

Tests on existing data:
1. Per-task dispersion of SELF Δs, and the Mahalanobis distance of hijacked Δs from it, GRPO against Base-Lora. M3 predicts the P1 margin tracks it.
2. The layer×pool locus of GRPO's excursion in `probe_results.csv`. M3 predicts late layers and `ctx_last`.
3. A paired per-episode bootstrap of P1 − C_Δs on the same samples, which the README admits is missing.

Caution: C_Δs is not uniformly higher for GRPO (MLP s2 0.568 against 0.529/0.541, but object 0.531 against 0.577), so this is a hypothesis to test, not a result.

### M4 — visual servoing explains the behaviour
The next-call reaction is a function of o(t+1) alone.
- **Test:** operationalise "novelty" as the distance of layer-0/4 `ctx_mean` of h(t+1) (essentially the projected vision embedding) from the per-task SELF distribution, and regress Δentropy on it within and across arms.
- Frames exist for only 5 episodes per run, so hidden states are the only free route.
- This is what would rescue or bury the "novelty" claim.

## C. Next experiments, ranked cheapest-informative first

**Costs:**
- L40S ≈ $2.81/h, and collections took 6.7–22.7 min (ledger).
- A linear round takes 6–27 min at 24 CPUs.
- An MLP round takes 102–210 min on Modal, ~$1.8–2.5 per attempt, and cannot checkpoint.
- The Windows box is ~7× slower (logbook 08:50): **[inference]** 12–25 h per MLP round, not the 2 h in `LOCAL_RUN.md`.

### X0. Mechanistic re-analysis of existing data — $0 Modal, local CPU hours
Volume reads are free (`LOCAL_RUN.md`); the hidden states are 17.2 GB.
- **Deliverables:**
  - the efference-increment table for all 20+ runs;
  - outcome readout R²;
  - the decoded-ingredients oracle;
  - GRPO dispersion and locus;
  - the novelty operationalisation;
  - a paired bootstrap of P1 − C_Δs.
- **Predictions and decision rule:**
  - M1 predicts the oracle clears floor + 0.04 on ≥ 3 of 5 swap runs, with Δs R² ≥ 0.5.
  - M2 predicts Δs R² < 0.3.
- Either outcome gives the paper the mechanism paragraph it currently lacks. Do this first: it changes the writing more than any new collection under $10.

### X1. Wave-3 E2: second SFT→GRPO pair — ~$4–6
The SFT side (`main02`, `main02_mirror`, `main02_freeze`) exists. Collect only `e2_rl`, `e2_rl_mirror` and `e2_rl_freeze` (~15–25 min each, **[inference]** from the E4 ledger) and run the linear probes locally.
- **Hypothesis:** GRPO's pattern is a property of the objective, not of the 130-Base-Lora pair. The pattern: P1 up at most by outcome perception, P3 − P1 ≈ 0, and no E3 migration.
- **Rule:** R1–R3 per run, plus "efference increment ≥ 0.04 on both seeds" as the only positive.
- Replicating E4 on an independent pair is what lets you write "RL does not buy it" without hedging on one seed.

### X2. E5 step 1 (VRAM gate) — ~$0.5–1
Already coded (`vram_probe.py`). It is the pre-registered go/no-go, and it costs minutes.

### X3. E5-frozen: previous chunk spliced into the prompt at inference, no training — ~$1–2
`vram_probe._with_previous_chunk` already builds the input. Run one goal-swap collection with the previous commanded chunk as 56 prompt tokens, plus a linear round and (if affordable) one MLP round.
- **Hypothesis:** under M1, a frozen transformer that already reads action tokens may mix command and image enough for a linear mismatch readout, as the frozen GRU does in phase 3.
- **Prediction (M1-favouring):** P1 ≥ floor + 0.04 and ≥ C_Δs + 0.04, with C_cmd in band.
- **Null:** at floor, consistent with the PREREG warning that the backbone must be trained.
- Report the success rate: the tokens are off-distribution and may degrade the policy, which is itself a design datum.
- This is exploratory and outside the PREREG; log it as an amendment before collecting.

### X4. External efference monitor, closed loop — ~$3–6
The P2r comparator on (a_cmd, Δs) detects hijacks at 0.69–0.91, with no learning in the policy.
- **Design:** run freeze and swap arms where, when the monitor fires, the executed chunk is re-issued. Measure the clean-minus-probe success gap against the unmonitored arm.
- **Rule:** the gap shrinks by at least half, with a bootstrap CI excluding 0.
- This is a systems result, not a representational one. But it is the practical sentence an RSS reviewer wants: a ten-line proprioceptive comparator does what the 7B policy cannot.

### X5. Finish E4's missing rounds
To be candid, there are two parts:
- **The GRPO s1 goal-swap MLP round** (~$2–2.5; three preempted attempts) **cannot move R4**: s2 is already "explained by C_dstates", and R4 requires both seeds beyond observables.
- **The eight R6 mirror/freeze MLP rounds** (~$16–20 on Modal, or 4–8 days locally) are the only pre-registered path to (a), and the linear hint is real: GRPO beyond C_Δs on 3/4 cells, Base-Lora on 0/4. But under the memoryless argument, the P1/P4 excursions there are bounded by observables, and GRPO's P3 − P1 is ≈ 0 on all four cells.

So run them locally in the background, do not block the paper on them, and state the gap as "not run".

### X6. E5 proper: LoRA fine-tune with the previous command as input, then the full battery — ~$25–60 plus a demo pipeline; needs new compute
- **Hypothesis:** with a signed instantaneous command on the input, swap becomes readable beyond observables and compensation becomes mismatch-gated.
- **Rule:** on two seeds, P1 or P3 ≥ floor + 0.04 and ≥ C_Δs + 0.04, with C_cmd in band, and freeze E3 reacts (R7's test).
- **This is the one experiment that most changes the paper:** it turns a null into a positive design result.
- **A cheaper sibling worth checking:** an off-the-shelf OFT checkpoint that already takes proprioception. The official LIBERO releases use proprio + wrist, and RLinf's `official` path has `use_proprio`. It tests M2 without training, but the wrist-camera confound and a different action head have to be handled.

### X7. Wave-3 E3/E4-WM — ~$25–45; park
Substep-frame collections plus Wan-5B inference exceed the budget, and they belong to line B's own paper.

## D. What to cut or reframe

1. **Demote "novelty-gated compensation" from the title and from Contribution 4.**
   - In E4 the mirror reaction is null on all four checkpoint × seed cells: Base-Lora s1 +0.0387 (p = 0.073), s2 −0.0237 (p = 0.287); GRPO −0.0028 and −0.0004.
   - Meanwhile swap reactions appear in two more runs: Base-Lora long (p = 0.015/0.0075) and GRPO spatial (p = 0.029/0.015).
   - Across checkpoints the reaction is not ordered by visual novelty.
   - Keep the behavioural battery, report all cells, and claim only "small, checkpoint-dependent reactions; freeze never reacts". If X0's novelty score shows a dose-response, the mechanism can return.
2. **Lead with the architectural premise, not the null.** Q1 becomes the premise. The results are the format finding (P3 − P1 ≈ 0; outcome present but incomparable), the RL finding, and the GRU format requirement. Add the efference-increment table.
3. **Add E4-RL as a main section**, with the full "not run" ledger and three robust statements:
   - Base-Lora is at floor on both seeds and both families.
   - GRPO's increment is within δ of its observables twin.
   - There is no E3 migration, and R8 does not strengthen uniformly.

   Rename wave 3's E4 to "W3-WM" to remove the naming clash.
4. **Compress T1–T5 to a methods box plus an appendix.** RSS readers want the mechanism and the design consequence, not five probing traps. Move the ICML-flavoured world-model-vs-action-blind positioning to a paragraph.
5. **Fix the stated scope.** The abstract's "seven runs" is now ~27 runs across three checkpoints, and the Object/Long MLP passes and the GRPO long linear round are open.

## E. The five hardest objections

1. ***"A memoryless vision-only policy cannot attribute; the null is a tautology."***
   - Concede it for P1, and make it the premise.
   - The non-trivial content is P3 at floor while P2-MLP is 0.687 on identical samples, plus three further results:
     - the GRU shows which input format suffices;
     - E4 shows RL does not add it;
     - X3/X6 show what does.
2. ***"Probes are too weak: 4096-dim states against 8–56-dim mechanics."***
   - The same MLP family clears floors on 4096/8192-dim inputs when outcome information exists (mirror P4 0.612, freeze 0.636).
   - The decoded-ingredients oracle (X0) tests directly whether linearly compressing the states to 56 + 8 dims recovers the label.
   - Token-resolved capture remains a stated falsifier.
3. ***"Your novelty variable was never measured; the mirror/freeze contrast is two points, and it did not replicate on the E4 checkpoints."***
   - Correct.
   - Answer with X0's operationalised novelty and the pooled nine mirror/freeze runs, or withdraw the mechanism (D1).
4. ***"GRPO's increment is within δ of C_Δs by 0.023 in one family, and C_cmd is 0.002 past the band in the other; δ = 0.04 is doing all the work."***
   - δ was calibrated before E4 to reproduce every wave-2 verdict (logbook, pinned by a test).
   - Add the paired per-episode bootstrap of P1 − C_Δs and P3 − P1 (X0), and replicate on an independent pair (X1).
   - The claim is "not readable within the stated limit", never "absent".
5. ***"Sim only, one model family, sampling at T = 1.6; what should a practitioner do?"***
   - State the scope plainly.
   - Show the external monitor closing the loop (X4) and the proprio/command-input condition (X3/X6) as the actionable design guidance.
   - Note that the audit's cost (<5 GPU-h per checkpoint) makes it a per-checkpoint check rather than a one-off study.

## Bottom line

- Spend the remaining ~$12 on X1, X2 and X3 (about $6–9 in total).
- Run X0 and the background R6 rounds locally.
- Treat X6 as the compute ask that turns this into an RSS paper with a positive result.

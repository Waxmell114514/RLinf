# Memo: after X1 and X4 — what closes "so what?", what still overreaches, and where to stop

*Advisory memo by Fable, 2026-10-30, from a read-only pass over the repo; the follow-up to `next_steps_fable_x0.md`. Not a pre-registered verdict; nothing here changes an R1–R12, X1-R1–R7 or X4 call. Every number is quoted from a named file. `[estimate]`, `[TO VERIFY]` and `[inference]` are marked as before. New in this memo: `[quick check]` marks a number I computed in this pass with pandas from the named `calls.parquet` / `manifest.json` / `hidden/*.npz`; it is a diagnostic, not a result, and must be re-run under a logged rule before it enters the paper. Prices use the billed L40S rate from the logbook's "Pricing correction" ($2.8/h): ≈ $0.6 per 250-episode goal collection plus ≈ $0.1 of start-up per call. Credit left ≈ $3.1 (logbook, "X4 READOUT", 05:14) `[TO VERIFY on the dashboard]`.*

Files read: `reports/next_steps_fable_x0.md`, `reports/x0_report.md`, `reports/e4_report.md`, `reports/x1_report.md`, `reports/x4_report.md`, `PREREG_X1_X4.md` (Amendments 1–2 included), `logbook.md` from the F1–F7 results (15:08–16:01, 2026-10-28) through the pricing correction, the X1/X4 pre-registration, "X4 monitors frozen", the X1 S2 operator summary, the local probe lines, "X1 READOUT" and its correction, the X4 approval, "X4 S0 FAILED", "X4 AMENDMENT 2", the GRPO S2 and budget-stop entries, the s5 decision and "X4 READOUT"; every JSON in `data/x1/x1_readout/`; `data/x1/x0/a2_x1_*.csv`; `data/x1/x1_*/analysis/probe_results.csv`, `manifest.json`, `calls.parquet` and one `hidden/ep*.npz` per run; `data/x4/x4_*_mirror_s*/manifest.json` and `calls.parquet`; `x4/monitor_{grpo,sft}.json`; the `reports/x0_results/` and `followup/` listings; `x1_report.py`, `x4_report.py`, `monitor.py`, `observables.py`, `hijack.py`, `x1_plan.yaml`, and by grep `harness.py`, `capture.py`, `config.py`, `configs/main02_mirror.yaml`, `x0_followup.py`, `x0_mechanism.py`; `paper_icml/main.tex` (abstract, introduction, positioning, protocol, Results I, receipts, traps, limitations, conclusion); the `local_check/` listing and `e4_derived/*/competence.csv`.

---

## 0. Bottom line in eight sentences

1. **The spine is now as strong as it will get.** P3 − P1 is inside ±0.04 in 36 of 36 run×family rows (30 in `a1_efference_increment.csv`; six in `x1_report.md`, where |P3 − P1| ≤ 0.006), the paired CI covers 0 in 17 of 18 linear bootstraps (16 of 17 in `a7_linear_selected.csv`, the exception inside F1's 0.011 BLAS resolution; plus `x1_grpo_swap` +0.003 [−0.014, +0.020]), and the registered X1-R3 reading is "no efference increment on two pairs". Nothing more needs collecting for this claim.
2. **The RL question is answered as far as this budget can answer it: a null with a sliver.** At the pre-named cell L12/`ctx_mean`, GRPO − SFT is +0.035 [−0.004, +0.074] on X1 (registered reading "no RL effect at the fixed cell"), +0.050 [+0.002, +0.097] and +0.064 [+0.021, +0.110] on the two post-hoc E4 seeds (`fixed_cell.json`); the one X1 swap margin beyond δ is state-explained (R5: 0.588 → 0.559, floor 0.549), and GRPO's mirror/freeze margins sit under C_obs's bar (0.715 / 0.738; `x1_report.md` R2). Write "RL raises outcome perception by at most ≈ 0.06 and attribution by nothing we can detect".
3. **Three lines in your summary overreach, and the files do not support them.** "X1 6/6 with bootstrap CIs covering 0" (one X1 bootstrap exists); the two-call reader "detects hijacks at 0.68–0.93" (that is P2r on *true* inputs; the activation reader A3 is 0.54–0.82 on X1 and fails on `x1_sft_swap` and E4 GRPO goal s1); and "RL narrows the state distribution" as an RL fact (X1's all-call readout gap is 0.006 nRMSE, `f4_x1_swap.json`; E4's gap is Base-Lora vs GRPO-130, not SFT vs RL). Also, the F2 "exception" on GRPO freeze is an R² on a target whose variance is 6 % of SELF's `[quick check]`; only mirror's +0.057 is real, and small. Section 2.
4. **X4 says more than its registered reading, provided the reading comes first.** In all 1,750 X4 episodes, every failed episode ended at the 40-call cap `[quick check]`: "hijack cost" in this protocol is time. The oracle undo saves 1.9 calls per GRPO probe episode (−1.90 [−3.74, −0.04] `[quick check]`) and is taxed by the cap it is trying to beat. Detection is not the bottleneck: B_mon / B_orc = 0.059 / 0.083 = 0.71 ≈ the online TPR 0.754 (`x4_report.md`). Section 4.
5. **The external monitor is the actionable deliverable, and it has one measured hole.** The frozen mirror-trained monitors transfer across checkpoints (AUROC 0.961–0.977 on the other checkpoint's X4 off arm; 0.931–0.974 on X1's mirror runs) but not across transforms (swap 0.650–0.692, freeze 0.703–0.753) `[quick check]`. A pooled *linear* monitor does not close it (held-out TPR 0.31–0.44 at 2 % FPR); the free thing to test is an MLP on the same ≈ 35 features (F9). Section 5.
6. **Strongest honest thesis:** a single-image VLA cannot know that its last action was overridden, before or after RL on its own consequences; its states hold the current state, from the projected image embedding on, and nothing about authorship; an efference copy handed to a ten-line external reader with proprioception detects the override online at TPR ≈ 0.75 for 2 % false alarms; and once detection is solved, the cost of an override is the time it takes to recover, which no one-call recovery repays under a call budget. Section 6.
7. **Spend:** free CPU on F8–F12 (survival-corrected costs, monitor generality, the F2 relabel, a vision-only twin, the residualised fixed cell), and ≈ $0.7 of the $3.1 on one collection, **X5**: GRPO-130 goal mirror, a new seed, with objects, joints and vision logged. It closes the only E4 cell still "beyond observables" on all three rungs (GRPO mirror s1, `e4_report.md`). Keep the rest as reserve. Section 5.
8. **Then stop.** More X4 seeds, other recoveries, prompt-spliced "efference copies", image ablations and activation patching on a frozen model would not change a sentence; the channel experiment (X6) needs training and belongs in the compute ask. Section 7.

---

## 1. What the two waves settled, checked against the files

| # | Claim | E4 / X0 | X1 | X4 | Status |
|---|---|---|---|---|---|
| 1 | No efference increment | 30/30 rows within δ (`a1`); CI covers 0 in 16/17 (`a7_linear_selected.csv`) | 6/6, max |P3 − P1| 0.006; `x1_grpo_swap` +0.003 [−0.014, +0.020]; E4 s1 −0.006 [−0.038, +0.023], s2 +0.005 [−0.014, +0.022] (`x1_report.md` R3) | — | settled on two pairs |
| 2 | State readable from the input embedding | L0/`ctx_mean` R² 0.860–0.965 (`a2_e4_*goal*.csv`) | L0 0.841–0.908, best cell 0.880–0.929 (`data/x1/x0/a2_x1_*.csv`) | — | settled, both checkpoints of both pairs |
| 3 | Last motion only through the end state | F2 gain ≤ 0.009 all calls, ≤ 0.005 HIJACK, 4/4 | all-call gain 0.010–0.016 in 6/6; HIJACK gain ≤ 0.021 on swap and on SFT runs; GRPO mirror +0.057, GRPO freeze +0.104 (`f2_x1_*.json`) | — | holds on all calls; freeze exception undefined, mirror exception small (§2, §4a) |
| 4 | Two-call reader on activation decodes | A3 ≥ floor + δ on Base-Lora 5/5, GRPO 3/4 (one by +0.007) | 0.539 (`x1_sft_swap`, under floor + δ = 0.567), 0.603, 0.675, 0.712, 0.745, 0.823 (`a3_x1_*.json`) | — | works on gross violations and on most swaps; not universal |
| 5 | Decodes lossy (F3) | outcome side OK 1/5, command 0/5 (Base-Lora) | command side within 0.10 of P2r on 6/6 (0.602–0.925 vs 0.683–0.931); outcome side only on `x1_grpo_swap` (`f3_x1_*.json`) | — | "both lossy" holds on E4; on X1 the command decode is the better side |
| 6 | GRPO margins are state-borne | F5 state-explained on both s2 runs; s1 runs indeterminate | R5 state-explained on `x1_grpo_swap` (C_obs residualiser); F5 "partial" on GRPO mirror 0.623 → 0.586 and freeze 0.596 → 0.560, yet both rungs under C_obs + δ (R2) | — | swap: yes; mirror/freeze: explained by full observables, not by the 8-dim state alone |
| 7 | RL readout advantage | F4 "data effect", matched/all ratios 0.21–0.38 | no gap to explain: nRMSE 0.337 (SFT) vs 0.332 (GRPO), `f4_x1_swap.json` | — | an E4 fact about Base-Lora vs GRPO-130; not an RL claim |
| 8 | Fixed cell L12/`ctx_mean` | post hoc +0.050 [+0.002, +0.097], +0.064 [+0.021, +0.110] | +0.035 [−0.004, +0.074], "no RL effect" (`fixed_cell.json`) | — | registered null; three positive points, all < 0.07 |
| 9 | Online detection | — | — | TPR 0.754 / 0.732 at FPR 0.019 / 0.024 on 917 / 917 HIJACK calls (`x4_report.md`) | settled on mirror, both E4 checkpoints |
| 10 | Recovery | — | — | Rule O false on both: B_orc +0.083 [+0.017, +0.150] vs 0.5·G = 0.110 (GRPO); +0.002 [−0.093, +0.098] vs 0.084 (Base-Lora) | "undo does not recover the cost; X4 is uninformative about the comparator" |
| 11 | Hijack cost, RL vs SFT | R8 mirror 0.175/0.159 → 0.170/0.268 | R6 mirror +0.275 [+0.050, +0.488]; swap −0.004 [−0.200, +0.194]; freeze +0.091 [−0.129, +0.302] | off-arm gaps 0.150 / 0.119 (clean n = 64 / 49) `[quick check]` | GRPO's *absolute* mirror cost larger in 2 of 3 pair-seeds; *relative* cost equal (§2) |

Two reproduction facts worth a Methods sentence: `x1_sft_swap` reproduces wave-2 `main02` within 0.04 on P1/P3/P4/P2r (−0.005, −0.000, +0.009, −0.028; `x1_report.md`), on a different machine and stack a year of code later; and X4's off arm reproduces E4's pooled probe success (0.756 vs 0.730; 0.493 vs 0.477; no sanity flag).

---

## 2. Overreach: your summary, then the paper

### In the summary you sent me

1. **"X1 6/6 runs, with bootstrap CIs covering 0."** X1 has one paired bootstrap, on `x1_grpo_swap` (PREREG X1-R3 asks for exactly that one). The honest line: "|P3 − P1| ≤ 0.006 in all six X1 runs; the registered bootstrap on the RL swap arm covers 0; in E4, 16 of 17 cover 0 and the exception (+0.025, Base-Lora freeze s2) sits inside the 0.011 BLAS resolution (F1)."
2. **"The last motion is visible only through the state (F2). Exception in X1: GRPO mirror/freeze F2 gain 0.057 / 0.104."** The freeze exception is not a motion trace. On freeze HIJACK calls the arm barely moves — median |Δpos| 1.01 cm (GRPO) and 0.73 cm (SFT) against 4.90 / 2.96 cm on SELF calls, and the summed position variance is 2.0 / 1.4 cm² against 31.6 / 19.4 cm² `[quick check, data/x1/x1_*_freeze/calls.parquet]`. On that near-constant target the k-NN's R² is −2.60 (GRPO) and −2.04 (SFT) (`f2_x1_*_freeze.json`, `hijack.S2`); a "gain" of +0.104 is a predictor going from very wrong to slightly less wrong. The X1-R7 mark ("does not replicate") stands as computed; the *text* should say the freeze cell is undefined (F10 below gives the rule) and keep only mirror's +0.057 (0.157 → 0.213), where the HIJACK target variance is 84 % of SELF's. Mirror's +0.057 is compatible with joint posture or object pose rather than efference: C_obs is the top twin on that run (0.675) and C_joint 0.610, C_scene 0.606 (`controls_x1_grpo_mirror.json`).
3. **"An external two-call reader detects hijacks (P2r 0.68–0.93), but decoded inputs are lossy (F3)."** This conflates two readers. P2r reads the *true* command and the *true* state change — it is the simulator's monitor, and 0.683–0.931 is its X1 range. The two-call reader on *activation decodes* is A3, which on X1 scores 0.539–0.823 and fails on `x1_sft_swap` (0.539 against floor + δ = 0.567); on E4 it failed on GRPO goal s1 (0.514 vs 0.568, `x0_report.md`). Write: "on true inputs the comparison is easy (P2r 0.68–0.93); on linear decodes from the two calls' activations it is recoverable on gross violations (0.68–0.82) and on most but not all swaps (0.54–0.68)".
4. **"RL: E4 small, end-state-borne (F5), and a data effect (F4)."** F4 does not replicate on X1 (`x1_report.md` R7: "representation effect" under the unchanged rule, because the all-call gap is 0.006 — there is no RL readout advantage to attribute; `f4_x1_swap.json` nRMSE 0.337 vs 0.332, L0). E4's sharper GRPO readout (translation RMSE 0.9–1.3 cm vs 1.5–2.1 cm, F4 logbook entry) separates a LoRA SFT on 130 tasks from a full-parameter RL checkpoint; X1's full-parameter SFT and its RL sibling read out alike. Drop "RL narrows the state distribution / makes the state more readable" as an RL statement; keep it as an E4 observation.
5. **"X1: GRPO above floor but explained by full observables, except a marginal swap case."** Right, with one tightening: the "marginal case" is `x1_grpo_swap` P1 0.588 and P3 0.590 against a bar of 0.587 (by 0.001 and 0.003; `x1_report.md` R2), and R5 on the same run is "state-explained" (0.588 → 0.559). Call it "not beyond observables"; do not keep a 0.001 margin alive in the text.
6. **"X4: undo recovers only ~38 % of GRPO's hijack cost under oracle detection."** True against the registered G (0.083 / 0.219 = 0.38), and the registered verdict stands. Two things to add: Base-Lora's undo recovers nothing (+0.002 [−0.093, +0.098]); and against X4's own off-arm gap — clean-off 0.906 (n = 64) minus probe-off 0.756 = 0.150 `[quick check]` — the same +0.083 is 55 %. Report both, interpret neither as a pass; §4b explains why the gap is a time budget.
7. **"Behaviour: on mirror, GRPO pays a larger hijack cost (X1 R6 +0.27)."** Absolute, yes. Relative to its own clean success, no: `x1_grpo_mirror` loses 0.565 of 0.938 (60.3 %), `x1_sft_mirror` 0.291 of 0.484 (60.1 %) `[quick check, manifests]`. GRPO starts higher and falls to the same fraction. E4 s1 shows no GRPO excess at all (0.170 vs 0.175, `e4_report.md` R8). The time view is the one that separates them (§4b): hijacks add +13.8 calls [+11.1, +16.4] to a GRPO mirror episode and +5.4 [+1.4, +9.5] to an SFT one, the latter censored by the cap `[quick check]`.

### In `paper_icml/main.tex` (still as of this pass)

- Abstract and title still lead with **novelty-gated compensation**; A6 pooled ρ = +0.026 [−0.003, +0.055] and E4's mirror reaction nulls on four cells already retired it (`next_steps_fable_x0.md` §6). Unchanged since.
- **Contribution 3** ("the comparison is cheap; the policy never buys it") is still the capacity argument that A3/F3 undercut; X1 adds a case where even the two-stage reader fails (`x1_sft_swap`). Rewrite as the previous memo said, with F3's numbers.
- **"The RL comparison (pre-registered, not yet run)"** is now two pairs, 11 collections and a fixed-cell test; "seven runs" in the abstract is ≈ 30 probed runs plus seven X4 collections, across seven checkpoints (the six in `a1_efference_increment.csv` plus GRPO-goal).
- **Intro:** "no proprioception, no action history" is right and now carries more weight: it is why the monitor, not the policy, gets the efference copy (§6). Say it as the design principle, not a footnote.
- **"The policy is a high-fidelity state estimator"** → "the projected image embedding is" (L0 R² 0.84–0.97 across both waves; the LM adds ≤ 0.04: X1 best-cell minus L0 is 0.012–0.039 in `a2_x1_*.csv`, E4 ≤ 0.015).

---

## 3. What a strong RSS reviewer still finds missing

Each item: the claim the paper makes or implies → the gap → why it matters → what closes it (§5 numbers).

1. **"Override detection must be external, or come from an added input channel."** The external half is now measured (X4 detection); the channel half is untested and cannot be tested without training (X6). *Why it matters:* it is the paper's one design recommendation. *Close:* say it is untested; write X6 as the compute ask with the spec the data imply — the channel must carry the **signed command**, because F3 shows the command decode loses 0.15–0.16 against the true command on the five E4 Base-Lora runs (decoded â + true Δs vs P2r, `f3_e4_sft_*.json`) and the one-stage probes cannot build it; and include the X2 VRAM gate number if you run it (unchanged advice).
2. **"The monitor fires — then what?"** X4's registered answer is "uninformative about the comparator", and reviewers at a robotics venue will read "the recovery did not work" as "the detector is useless". *Gap:* the paper has no account of *why* undo fails. *Close (free, §4b, F8):* every X4 failure is a timeout at 40 calls; the undo repays ≈ 1.9 calls on GRPO; relative to the cap, a hijack plus its undo costs two calls out of a budget GRPO's clean episodes use 15.6 of and SFT's 24.6 of `[quick check]`. Report detection as the deliverable and recovery as a time-budget problem the protocol can measure but was not designed to solve.
3. **"A ten-line monitor."** It was trained on mirror and tested on mirror. *Gap:* on X1 it transfers across checkpoints but detects swap at AUROC 0.65–0.69 and freeze at 0.70–0.75 `[quick check]`, while per-transform P2r reaches 0.68–0.69 and 0.86–0.93 on the same runs (`x1_report.md`). *Why it matters:* a deployable monitor does not know the perturbation type. *Close (free, F9):* pooled training, and the MLP family on the same features; report per-transform held-out TPR at 2 % FPR.
4. **E4's three "beyond observables" GRPO-130 cells** (mirror s1 P1/P3/P4 at 0.661/0.668/0.682 vs C_dstates 0.586; mirror s2 P4 0.695 vs 0.633; freeze s1 P4 0.719 vs 0.591; `e4_report.md`) have only an 8-dim twin. X1 shows what the full twin does on a *different* GRPO checkpoint (C_obs 0.675 / 0.698 absorbs everything), but the reviewer is entitled to say that does not transfer. *Close (Modal, X5):* one GRPO-130 mirror collection with objects, joints and vision; ≈ $0.7.
5. **"Hijack cost" and "compensation" are success rates under a 40-call cap.** Clean SFT episodes already run 27–32 calls on average and 46–61 % of them hit the cap (`[quick check]`, X1 manifests); probe episodes hit it in 62–81 %. *Why it matters:* every behavioural statement — R6, R8, X4's B's — conflates failing with being slow, and the cap censors SFT more than GRPO, which biases "GRPO pays more". *Close (free, F8):* restricted-mean time-to-success at 40 calls with episode bootstrap; it is the metric the paper should have used from wave 1, and it makes X4's oracle effect visible where the success rate hides it.
6. **The RL null rests on two GRPO pairs from one codebase, both on LIBERO-Goal, with linear probes as the primary family in X1.** *Close:* scope statement; the MLP family added nothing anywhere it ran (E4), and X1's linear twins are complete. No experiment.
7. **"Your null is sample complexity."** A3/F3 answered half of it: a supervised two-stage reader finds 0.60–0.68 on E4 Base-Lora where one-stage probes find floor, and on X1 that reader fails on the SFT swap arm. *Gap:* no learning curve. *Close (free, optional, F13):* Base-Lora s1 ∪ s2 pooled (n = 968 + 948) at n/2, n, 2n; one Limitations sentence.
8. **Real robots and real proprioception.** The monitor read the simulator's eef pose. *Close:* none in budget; say the monitor needs the commanded chunk and the end-effector pose before and after it — both available on a real arm without a camera — and that nothing in it uses the image.

---

## 4. Three things the existing data say that nobody has written down yet `[quick check]`

### 4a. The F2 freeze exception is undefined; the mirror exception is small and twin-shaped

Numbers in §2 item 2. Rule to log before re-running (F10): at the F2 cell, mark the HIJACK gain "undefined" iff var(Δpos | HIJACK) / var(Δpos | SELF) < 0.2. By the quick check that fires on both freeze runs (0.06 and 0.07) and on no other X1 run (mirror 0.84 / 0.82, swap 0.67 / 0.77).

### 4b. In X4, failing *is* timing out, so the undo is taxed by the outcome measure

From the seven X4 manifests (`data/x4/x4_*_mirror_s*/manifest.json`), probe episodes, by condition:

| ckpt | condition | n | success | mean calls | at cap | hijacks / ep | recoveries / ep |
|---|---|---|---|---|---|---|---|
| GRPO | off | 291 | 0.756 | 22.8 | 0.247 | 3.15 | 0 |
| GRPO | monitor | 238 | 0.815 | 21.6 | 0.189 | 3.02 | 2.33 |
| GRPO | oracle | 249 | 0.839 | 20.9 | 0.161 | 2.82 | 2.66 |
| Base-Lora | off | 213 | 0.493 | 30.1 | 0.516 | 4.31 | 0 |
| Base-Lora | monitor | 183 | 0.508 | 29.0 | 0.497 | 4.29 | 3.52 |
| Base-Lora | oracle | 190 | 0.495 | 28.8 | 0.511 | 4.13 | 3.94 |

- **Every failure is a truncation.** The fraction of probe episodes that failed without reaching 40 calls is 0.000 in all six cells (LIBERO ends an episode only on success or the cap). "1 − success" and "at cap" are the same column.
- **Success is a steep function of how many hijacks an episode absorbed**, in every condition: GRPO off 1.000 at ≤ 2 hijacks, 0.935 at 3, 0.885 at 4, 0.500 at 5, 0.167 at 6; Base-Lora off 1.000, 0.842, 0.647, 0.273, 0.103. Oracle-arm successes had 1.94 recoveries, failures 6.45 (GRPO); 1.88 vs 5.95 (Base-Lora). Hijack count and episode length are confounded (longer episodes offer more p = 0.25 draws), so this is descriptive — but it says the recovery's own call cost lands on exactly the episodes that are already near the cap.
- **In time, the oracle helps:** restricted mean calls-to-success at 40, oracle − off, GRPO −1.90 [−3.74, −0.04], Base-Lora −1.28 [−3.74, +1.06]; monitor − off −1.18 [−3.09, +0.72] and −1.11 [−3.30, +1.23] (episode bootstrap, B = 2000, seed 0). Hijacks themselves add +5.63 [+3.11, +7.99] calls (GRPO) and +4.68 [+0.88, +8.63] (Base-Lora) to a probe episode against clean-off.
- **The undo is not broken, and it is not the scene.** Median undo error is 1.86 / 1.39 cm over 1178 / 1330 undos (`x4_report.md`), i.e. the undo returns most of the displacement (Amendment 2: 71–84 %). And hijacks do not disturb objects more than the policy's own chunks: P(max object displacement > 1 cm during the chunk) on X1 mirror is 0.257 (HIJACK) vs 0.220 (SELF) for SFT and 0.381 vs 0.311 for GRPO; on freeze it is *lower* under hijack (0.162 vs 0.220; 0.307 vs 0.354) `[quick check, data/x1/x1_*/calls.parquet, objects_before/after]`. Probe episodes in which some hijack moved an object succeed *more* often (GRPO mirror 0.424 vs 0.259), because object contact happens late in a task. The cost is lost progress and lost calls, not a knocked-over scene.
- **Design note for the paper:** PREREG X4 made recovery calls count toward `max_calls_per_episode = 40`. With the outcome being success-by-40, any recovery that costs a call is charged against the measure it is meant to improve; a hijack plus its undo is two lost calls. Report that as a limitation of the X4 design, keep Rule O's verdict, and give the RMST numbers next to it (F8 makes them pre-registered).

### 4c. The frozen monitors transfer across checkpoints, not across transforms

Replay of `x4/monitor_grpo.json` and `x4/monitor_sft.json` (frozen; `monitor.mismatch_features` on `a_cmd_env`, `states_before`, `states_after`; calls with `call_idx ≥ 4`), own thresholds τ:

| test set | n (HIJACK) | GRPO-monitor AUROC / TPR / FPR | SFT-monitor AUROC / TPR / FPR |
|---|---|---|---|
| X4 GRPO off arm (s3–s4) | 3358 (481) | 0.980 / 0.761 / 0.019 | 0.977 / 0.807 / 0.026 |
| X4 Base-Lora off arm (s3–s4) | 4389 (599) | 0.961 / 0.631 / 0.015 | 0.968 / 0.711 / 0.023 |
| `x1_grpo_mirror` | 5623 (842) | 0.971 / 0.728 / 0.012 | 0.974 / 0.831 / 0.040 |
| `x1_sft_mirror` | 4639 (651) | 0.931 / 0.647 / 0.047 | 0.938 / 0.710 / 0.055 |
| `x1_grpo_freeze` | 4011 (551) | 0.753 / 0.209 / 0.012 | 0.748 / 0.318 / 0.043 |
| `x1_sft_freeze` | 4299 (586) | 0.704 / 0.312 / 0.057 | 0.703 / 0.334 / 0.066 |
| `x1_grpo_swap` | 4890 (660) | 0.653 / 0.223 / 0.023 | 0.650 / 0.292 / 0.048 |
| `x1_sft_swap` | 5153 (711) | 0.692 / 0.293 / 0.048 | 0.689 / 0.307 / 0.060 |

A linear logistic on the same features trained on all three X1 SFT transforms (episode-grouped 5-fold, τ at the 98th SELF percentile of each training fold) scores held-out AUROC / TPR / FPR of 0.762 / 0.308 / 0.026 (swap), 0.875 / 0.441 / 0.019 (mirror), 0.875 / 0.352 / 0.022 (freeze), and transfers to the GRPO-goal runs at 0.712 / 0.277, 0.933 / 0.601, 0.912 / 0.403. The linear family cannot represent "mirror or freeze or swap" well in one decision function (the transforms live in different parts of P2r's feature block); this is what F9's MLP test is for.

---

## 5. Next steps, ranked by information per dollar and per hour

Costs: billed L40S ≈ $2.8/h (logbook "Pricing correction"); E4 GRPO goal mirror collections took 11.1 / 11.4 min, X4 GRPO seeds 10.1–11.8 min, X1 GRPO mirror 13.3 min for 220 episodes (`e4_report.md` ledger; logbook X1 S2 and X4 entries) → **≈ $0.5–0.6 per GRPO mirror collection plus ≈ $0.1 start-up** `[estimate]`. Local linear rounds: X1 took 25.8–86.5 min per run at `--jobs 4` (logbook), controls and residuals minutes. Every item needs a logbook pre-registration before any number is computed; rules are written to be pasted.

### Tier 0 — free, this container, in this order

| # | Analysis | Files / code | Rule (to pre-register) | Cost | What each outcome lets the paper say |
|---|---|---|---|---|---|
| **F8** | **Survival-corrected hijack cost.** Restricted mean calls-to-success at 40 (RMST; `n_calls` already equals it), episode bootstrap B = 2000, for: probe − clean per run; RL − SFT difference per arm; X4 oracle − off and monitor − off per checkpoint. Optionally Cox with censoring at 40. | X1 manifests (6), X4 manifests (7), E4 goal-swap manifests (4, in `local_check/`); E4 mirror/freeze manifests must be fetched from the Volume (8 small files; CPU call, ≈ $0 `[TO VERIFY]`). New `x_survival.py`; nothing existing changes. | (i) "hijacks cost time" iff RMST(probe) − RMST(clean) > 0 with CI excluding 0; (ii) RL − SFT added calls reported with CI, sign test over arms; (iii) X4: RMST(oracle) − RMST(off) reported *next to* Rule O, never in place of it. | < 1 CPU-h | Replaces every success-rate "cost" sentence with a censoring-correct one; by the quick check: GRPO mirror +13.8 [+11.1, +16.4] calls vs SFT +5.4 [+1.4, +9.5]; X4 oracle saves 1.9 [0.04, 3.7] calls on GRPO. If (iii)'s CI excludes 0 on GRPO, the paper may say "the undo shortens recovery; it does not repay the hijack's call cost under the cap". |
| **F9** | **Monitor generality.** (a) Re-run §4c's replay under a logged rule. (b) Pooled linear P2r monitor across swap/mirror/freeze, grouped CV. (c) The ladder's MLP family (256 hidden) on the same ≈ 35 features, pooled, held out per transform and per checkpoint (X1 SFT ↔ X1 GRPO; X4 off arms as a second test set). | `monitor.mismatch_features`, `probes.run_probe` (MLP family exists); new `x_monitor_generality.py`. Data: X1 `calls.parquet` ×6, X4 off arms. | "One monitor covers gross overrides" iff pooled MLP held-out TPR ≥ 0.5 at FPR 0.02 on mirror *and* freeze, on both checkpoints; swap reported, no bar. Cross-checkpoint transfer "holds" iff AUROC drops ≤ 0.03 against own-checkpoint. | < 1 CPU-h (features are 35-d) | Either "a single command-conditioned monitor, trained once, detects reversed and frozen overrides on a checkpoint it never saw" (the deliverable, generalised), or "the monitor must be per-perturbation; swap stays hard by design" (P2r 0.68–0.69). |
| **F10** | **F2 relabel.** Target-variance ratio at each F2 cell. | 20 lines on `f2_residual`'s inputs; `f2_x1_*.json` unchanged. | HIJACK gain "undefined" iff var(Δpos ∣ HIJACK)/var(Δpos ∣ SELF) < 0.2. | minutes | Removes the freeze "motion trace" from the text; keeps mirror +0.057 as the one small exception, attributed to the C_obs twin. |
| **F11** | **Vision-only twin.** C_vis: linear probe on `vis(c+1)` (the 4096-d projected patch mean captured in X1), and on `vis(c) ⊕ vis(c+1)`; P1 at its best cell residualised on `vis(c+1)` (as F5 does on the state). | `hidden/ep*.npz` field `vis` (present in all six X1 runs); `x1_report.residualised_p1` pattern; new flags in a new module. | "Vision-borne" iff P1_best − C_vis ≤ δ *and* P1_resid(vis) ≤ floor + 0.02, on ≥ 5 of 6 X1 runs. | ≈ 10 min per run | "Everything the LM's states carry about the override is already in the projected image; the LM adds nothing" — the architectural sentence with a number, and a direct answer to "maybe the LM computes something from the image". A failure (P1 ≫ C_vis) would be the first sign of an LM-internal signal and would deserve the MLP twin. |
| **F12** | **Residualised fixed cell.** GRPO − SFT at L12/`ctx_mean` after residualising h on C_obs's features (X1) or the per-task 8-dim state (E4), paired OOF, unpaired episode bootstrap as in R4. | `x1_report.fixed_cell`, `residualised_p1`; `fixed_cell.json` for the unresidualised reference. | "State-borne" iff the residualised difference ≤ 0.02 with CI covering 0 on X1 and on both E4 seeds. | minutes | Turns "three positive points at the fixed cell" into "outcome perception, +0.03 to +0.06, removed by the observables" — or flags the one thing that would reopen the RL question. |
| F13 (optional) | **Learning curve** on Base-Lora goal swap s1 ∪ s2 (n = 968 + 948): P1/P3/P4 and the P0 floor at n/2, n, 2n, same 27 cells. | `run_probes.py` on a pooled sample table (new loader, no probe-code change); hidden states in `local_check/`. | "Not sample-limited at this scale" iff (P1 − floor) rises ≤ 0.01 per doubling and P3 − P1 stays within δ at 2n. | 2–4 CPU-h `[estimate]` | One Limitations sentence answering "your null is n". |
| F14 | **Object displacement under hijack** (§4b, last bullet) written up as a check. | X1 `calls.parquet`, `objects_before/after`. | descriptive | done; re-run logged | "Hijacks do not disturb the scene more than self chunks; the cost is lost progress." |

Order: F8 and F10 first (they fix sentences already in the draft), then F9 and F11 (they add the two new sentences the RSS version needs), F12, then F13 if there is idle CPU.

### Tier 1 — Modal, one item, ≈ $0.7

**X5. GRPO-130 goal mirror, seed 3, full observables.** `main02_mirror.yaml` unchanged; overrides `capture.capture_vision=true`, `capture.extra_observables=true`, `model.do_sample=true`, `model.temperature=1.6`; GRPO's k set to reproduce E4 mirror s1's size (169 episodes ⇒ k ≈ 34 states on tasks {2,3,6,8,9}) rather than re-sized; both E4 checkpoints are already registered on the X-waves Volume (logbook 02:48:06, sha256 verified), so no download. Driver: a two-line `x5_plan.yaml` in the `x1_plan.yaml` pattern. Then locally: the linear round (≈ 1 h; X1 GRPO mirror took 55.9 min), `x1_report.py controls` and `resid` (minutes), F11 on the new `vis`.
- *Rules:* X1-R2 verbatim (beyond iff ≥ max(C_dstates, C_state+d, C_scene, C_joint, C_obs) + δ with C_cmd in [0.45, 0.55]); X1-R5 (residualised on C_obs); A1 on the new run.
- *Cost:* ≈ 11 min GPU ≈ $0.5 + $0.1 start-up → **$0.6–0.7** `[estimate]`; ≈ 1.5 h local wall.
- *Why this and not something else:* it is the only remaining cell in any wave where a GRPO checkpoint was "beyond observables" on all three rungs and no full-observables twin exists (`e4_report.md`, GRPO mirror s1), and the E4 report itself calls those cells "the only place the GRPO-vs-SFT pattern points toward (a)". X1's twins (C_obs 0.675 absorbing P1 0.623 on GRPO-goal mirror) predict it closes; a prediction the paper can state before the run.
- *Outcomes:* ≤ bar → "once objects and joints are logged, no rung on any checkpoint, SFT or RL, in any wave, is beyond observables; the three E4 cells that were beyond the 8-dim twin are explained on a fresh GRPO-130 seed" — and E4's R6 lead closes without the eight MLP rounds (R6 itself stays "blocked", as a ledger line). Beyond → a GRPO-130-specific mirror signal survives the full twin on one seed; then and only then is the MLP twin worth the Windows box.
- **X5b (optional, ≈ $0.55):** the same for freeze (E4 GRPO freeze s1 P4 0.719 vs C_state+d 0.606), only if X5 runs clean and ≥ $2 remains.

### Tier 2 — do not run (with the reason, since each was on the table)

- **Previous chunk spliced into the prompt as an "efference copy".** The frozen model was never trained to attend to action tokens in the prompt (previous memo §5, unchanged). A null says nothing about the channel; a positive would need a training story anyway. ≈ $0.7 + a probe round; its only value is X6 feasibility (does success survive the extra tokens). Run it only when the X6 ask is being written, after X5.
- **Ablate the image so state comes only from proprio.** This checkpoint has no proprioceptive input (`main.tex`: "no proprioception, no wrist camera"; `config.py` `num_images_in_input: 1`). Blanking the image removes the state channel entirely; what remains is a sanity check that the policy fails, not the experiment the name suggests. Cut.
- **Activation patching / steering between calls** (patch E1's command directions from h(c) into call c + 1). Without a circuit trained to consume the command there is no reason to expect a behavioural or probe effect, so a null is uninformative and a positive implausible; it also needs a new forward hook in the collection path and a GPU run (≥ $1, a day of code). Park; it is X6's job.
- **More X4 seeds (Base-Lora s6).** Rule O's point estimate is +0.002 on 190 vs 213 episodes; another 250 episodes cannot move it to 0.084. GRPO's +0.083 would need to reach 0.110 with CI_low > 0; one more seed will not do that either. The money is better in X5.
- **X4 with a higher call cap, or recovery calls exempt from the cap.** The censoring question is answered for free by F8; a new design would need a new pre-registration and $0.7 per checkpoint. Only if a reviewer demands it.
- **The eight E4 mirror/freeze MLP rounds.** X1's linear twins already explain the mirror/freeze margins with full observables, and the MLP family added nothing to any hidden rung where it ran (`e4_report.md`, wave 2). Report R6 as blocked, with X1-R2 as the reason it no longer matters.
- **GRPO goal-swap s1 MLP.** Windows box, background, if it is idle; otherwise a ledger line. Never Modal.
- **X3, X6-sibling, X7:** as in the previous memo.

### Budget plan `[estimate]`

F8–F14: $0 → X5: $0.6–0.7 → (X5b: $0.55, optional) → reserve $1.8–2.4 for one failed X5 attempt. Nothing on Modal CPU; no MLP round anywhere but Windows.

---

## 6. The thesis, and the headline claims with their files

### Thesis (RSS framing)

> **A deployed single-image VLA cannot tell that its last action was overridden, and post-training it on the consequences of its own actions does not teach it to.** The command exists only in the forward pass that produced it and the outcome only in the next, so no internal comparator is possible; the states confirm it, exposing the current end-effector state from the projected image embedding on (R² 0.84–0.97) and the last motion only through that state, with every above-floor probe matched by an observables-only twin once objects and joints are logged. Across 36 run×family rows on seven checkpoints, adding the command to the post-action state changes nothing a probe can read (P3 − P1 within ±0.04; 17 of 18 linear bootstrap CIs cover 0), and two independent SFT→GRPO pairs move the pre-named cell by +0.035 to +0.064, all of it outcome perception. The efference copy the policy lacks is cheap to use outside it: a linear monitor on the commanded chunk and the proprioceptive state change detects overrides online at TPR 0.73–0.75 for 2 % false alarms and transfers to a checkpoint it never saw. With detection solved, the cost of an override is the time it takes to recover — the hijacks in a swap or mirror episode add 3–14 calls to it — and a one-call undo repays one or two of them; under a call budget that is not enough, which is why the override problem is a recovery problem, not a detection problem, and why the policy itself should carry the command.

### Headline claims (five), each with its evidence

| # | Claim | Evidence files |
|---|---|---|
| 1 | **No readable self-attribution, before or after RL.** 36/36 rows within δ; 17/18 linear bootstrap CIs cover 0; registered "no efference increment on two pairs". | `reports/x0_results/a1_efference_increment.csv`, `a7_linear_selected.csv`, `reports/x1_report.md` (R3), `data/x1/x1_readout/increment_x1_grpo_swap.json` |
| 2 | **The states are a state estimate, not an authorship record.** L0 R² 0.84–0.97; F2 all-call gain ≤ 0.016; every rung ≤ C_obs + δ in X1; F5/R5 state-explained; (pending F11) ≤ C_vis + δ. | `a2_*.csv` (E4), `data/x1/x0/a2_x1_*.csv`, `followup/f2_*.json`, `x1_readout/f2_*.json`, `x1_report.md` (R2, R5), `x1_readout/controls_*.json` |
| 3 | **RL on its own consequences raises outcome perception slightly and attribution not at all.** Fixed cell +0.035 [−0.004, +0.074] (X1), +0.050 / +0.064 (E4, post hoc); state-explained; no readout gap in X1. | `x1_readout/fixed_cell.json`, `resid_x1_grpo_swap.json`, `f4_x1_swap.json`, `reports/e4_report.md` (R4/R5) |
| 4 | **An efference copy makes an external monitor cheap, and the policy's own state does not.** P2r 0.68–0.93 vs C_dstates 0.48–0.64 (X1); online TPR 0.754 / 0.732 at FPR 0.019 / 0.024; cross-checkpoint AUROC 0.96–0.98 (pending F9 under a rule). | `x1_report.md` (R1 `p2r`, `c_dstates`), `reports/x4_report.md`, `x4/monitor_{grpo,sft}.json` (`meta`), logbook "X4 monitors frozen" |
| 5 | **Detection is not recovery.** Rule O false on both checkpoints; B_mon/B_orc = 0.71 ≈ TPR; every failure a 40-call truncation; undo returns 71–84 % of the displacement and saves ≈ 2 calls (pending F8). | `reports/x4_report.md`, `PREREG_X1_X4.md` Amendment 2, logbook "X4 READOUT", `data/x4/*/manifest.json` |

A sixth, descriptive, if F8 confirms it: **hijacks cost the RL policy more time** (+13.8 vs +5.4 calls on mirror, X1) **while both policies lose the same fraction of their clean success** (0.60); `data/x1/x1_*_mirror/manifest.json`, `x1_report.md` (R6).

What this thesis does *not* need, and the draft should stop defending: novelty-gated compensation; "the comparator the network fails to compute"; "RL sharpens / narrows the state estimate"; GRPO "excursions"; a 0.001 margin on `x1_grpo_swap`.

---

## 7. Stopping point

Stop collecting after X5 (and X5b only if it is already paid for by a clean X5). Stop analysing after F8–F12 and, if idle, F13. The test for anything else is: *which of the five claims in §6 would change?* None of the items in Tier 2 can change one; a positive X6 would add a sixth, and that is a different paper's budget.

Concretely, do not:
- run another X4 seed, another recovery policy, or a higher-cap X4;
- run mirror/freeze MLP rounds anywhere, or the GRPO s1 MLP on Modal;
- splice actions into prompts, ablate images, or patch activations on a frozen model;
- re-collect wave-2 suites (spatial "beyond observables" for both checkpoints stays a single-seed, 8-dim-twin observation in the appendix — X1's full twins on goal are the paper's answer to it, stated as such);
- chase the fixed-cell sliver beyond F12.

And do write: the §2 corrections; the X4 time-budget limitation; the monitor as a deliverable with its transfer table; X6 as the compute ask with the signed-command specification; the scope sentence (one model family, one simulator, two GRPO pairs from one codebase, linear primary in X1, sim proprio for the monitor).

---

## 8. Checklist before any launch or any new number

- Logbook pre-registration for F8–F12 (rules in §5) and for X5 (rules X1-R2, X1-R5, A1; k; seed 3; overrides), before the first number and before the container starts.
- `x5_plan.yaml` tested offline through the existing plan tests; `gate_s0` smoke for extra observables on the GRPO-130 checkpoint (the X1 S0 checks, unchanged).
- `[TO VERIFY]`: the Modal balance; that the E4 checkpoints are still registered on the X-waves Volume; the cost of fetching eight E4 manifests for F8; whether the Windows box is idle for the GRPO s1 MLP.
- Nothing in this memo edits probe code, transforms or gates, trains anything, or asks for an MLP round on Modal.

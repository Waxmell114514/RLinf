# Memo: after X0 — what the data support, where the reading overreaches, what to run next

*Advisory memo by Fable, 2026-10-28, from a read-only pass over the repo. Not a pre-registered verdict; nothing here changes an R1–R12 call. Every number is quoted from a named file. `[estimate]`, `[TO VERIFY]` and `[inference]` are marked. Prices use the L40S rate you gave ($1.95/h); the E4 ledger was costed at $2.81/h, so earlier dollar figures in `next_steps_fable.md` are ~30 % high at today's rate.*

Files read: `reports/x0_report.md`, every file in `reports/x0_results/` (A1–A8, the three `a7_passes/`), `reports/next_steps_fable.md`, `reports/e4_report.md`, the X0 block of `logbook.md` (lines ~1566–1725) and the E4 spend/checkpoint entries, `x0_mechanism.py`, `datasets.py`, `hijack.py`, `harness.py` (grep), `capture.py` (grep), `configs/*.yaml`, `e4_plan.yaml`, `paper_icml/main.tex`, and on `origin/main` `PREREG_wave3.md`, `WAVE3_RUNBOOK.md`, `vram_probe.py`.

---

## 0. Bottom line in six sentences

1. **The zero is real and is the paper's cleanest number.** P3 − P1 is inside ±0.04 in all 30 run×family rows (`a1_efference_increment.csv`; largest +0.031 wave-2, +0.025 E4) and its paired CI covers 0 in 16 of 17 linear runs and the one MLP run (`a7_linear_selected.csv`). Keep it as the spine.
2. **"Ingredients present, comparator missing" overreaches in two ways.** The oracle's command comes from h(c), the *previous* call; a memoryless network never has h(c) and h(c+1) in the same forward pass, so A3 is a statement about an **external two-call reader**, not about a comparator the network "fails to compute". And A3 shows that P3 − P1 = 0 is at least partly a **probe sample-complexity** result, which cuts against the paper's current "capacity argument" (Contribution 3). Section 1.
3. **"Memoryless state estimator" is right as a premise and mostly right as a finding, but two corrections are needed:** the state readout is already at layer 0 (R² 0.965 GRPO s1, 0.873 Base-Lora s1, `a2_*.csv`), so it belongs to the vision encoder and the on-policy image distribution, not to "the policy"; and A2b lacks the residual test that would actually license "only through the state". Section 2.
4. **GRPO s1's "P1 excursion" is not an excursion by your own rules.** P1 − floor = +0.035 < δ (`e4_report.md`: "at"). It looks like one only against controls that sit *below chance* (C_dstates 0.460, C_state 0.484, C_state+d 0.488; `a8_e4_grpo_goal_swap_s1_linear.json`). The scene is a legitimate candidate for GRPO's small, consistent P1 margins, but it cannot be tested on existing data (no object poses, no joint posture, no vision features captured: `capture_vision: false` in every config). Section 3.
5. **Strongest honest thesis:** a single-image VLA cannot attribute its last transition because the command and the outcome never share a forward pass; its activations expose the *current* state (nearly perfectly, from the input embedding on) and nothing about authorship; a two-call external reader recovers the comparison from linear decodes; RL post-training narrows and sharpens the state readout but adds no efference increment, within a 0.04 detection limit on one SFT→GRPO pair. The positive design result the paper still lacks is an in-the-loop comparator (X4) or a command-input fine-tune (X6). Section 4.
6. **Spend the ~$12 on X1 (second SFT→GRPO pair, ~$2–3.5) and X4 (closed-loop monitor, ~$1.5–2.5), both with the collection code extended to log object poses, joint positions and vision features first.** Run the free analyses F1–F7 before either. Finish GRPO s1's MLP and the R6 rounds locally; never on Modal. Section 5.

---

## 1. Is "ingredients present, comparator missing" (A3) a fair reading?

### What A3 actually shows

`x0_mechanism.py::a3_oracle`: per outer fold, three ridge decoders are fit on training-episode calls with *true* targets — â(c) from h(c) at L28/`act_mean`, ŝ from h(k) at the A2 state cell, Δŝ from h(c+1) — then P2r's hand-built comparison features (sums, products, norms, cosines; `datasets._mismatch_features`) are computed from the decodes and scored by the ladder's linear logistic. Variant A clears floor + 0.04 on 8 of 9 swap runs (`a3_*.json`).

What this establishes: **the pair (h(c), h(c+1)) contains, in linearly decodable form, enough to detect a same-task swap, provided a supervised reader first compresses each state to its 8 relevant dimensions and then applies a fixed nonlinear comparison.** That is a true and useful fact. It is not the fact the report states.

### Three ways the reading overreaches

**(i) The command is read from the previous call.** Every oracle variant uses h(c) (`d_hcmd = h(cur, cmd_cell)`; variant A also uses h(c) for ŝ_before). At call c+1 the network's forward pass sees o(c+1) and the instruction only. So "nothing in the network combines it" is not a deficit the network could have avoided: the two ingredients are never co-present. The honest sentence is: *"the command at c and the outcome at c+1 are each linearly present in their own call's activations; no single forward pass holds both, so no internal comparator is possible, and an external reader with access to both calls can build one."* That is M1 stated as architecture, and it points at X4 (an activation-level monitor, if you want one) and X6, not at a "located deficit" inside the policy.

**(ii) A3 undercuts the paper's capacity argument more than it supports M1.** `main.tex` Contribution 3 and §Results I say: the same MLP computes the comparison from raw a ⊕ Δs (0.687 vs 0.463 linear) but not from a ⊕ h(t+1) (P3 MLP at floor), "absence in the states is therefore not explained by the comparison being hard." A3 shows the opposite is partly true: give a *linear* classifier the state decoded by a supervised 4096→8 ridge (fit on ~4000 calls with 8-dim regression targets) and it finds the label at 0.60–0.68 on Base-Lora; the MLP on a ⊕ h(t+1) has to discover that 4096→8 map from ~950 binary labels and does not. P3 − P1 = 0 is therefore consistent with two readings — (a) h(t+1) holds no command-comparable outcome code, (b) the probe cannot learn state extraction and comparison jointly at this n — and A3 says (b) is live. The paper currently asserts (a). The decomposition F3 below settles it for free.

**(iii) The pre-registered rule passes on thin margins for GRPO.** Fold lists in `a3_*.json` (sd computed from them): Base-Lora goal s1/s2 clear floor + 0.04 by +0.037/+0.040 with every fold ≥ 0.531/0.565; the suites by +0.053 to +0.131. GRPO goal s2 clears by **+0.007** with fold sd 0.060 (folds 0.599, 0.552, 0.688, 0.547, 0.553); GRPO spatial/object by +0.029/+0.031; GRPO goal s1 misses by −0.054. So "3 of 4 goal-swap runs" rests on s2's 0.007, and the "8 of 9" headline should be "Base-Lora 5 of 5, GRPO 3 of 4 with two marginal and one miss."

### What is solid in A3 and should stay

- For Base-Lora the decoded comparison beats P4 — the rung with the *same* two-call information in raw 8192-dim form — by +0.046 (goal s1), +0.069 (s2), +0.054 (spatial), +0.129 (object), +0.060 (long) (`a3_*.json` vs the P4 column of `e4_report.md`). That is the clean statement of the dimensionality problem the Limitations section already worries about: the signal is there, a 950-label probe on 8192 dims cannot find it, a supervised compression can.
- For GRPO the decoded comparison adds nothing over P4 (−0.050, −0.002, +0.018, +0.019) and P2r itself is lower (0.625–0.681 vs 0.798–0.880). Reading: GRPO's donor chunks are closer to its own (stereotyped on-policy behaviour), so there is less to compare; the report already says this and it is right.

### A cleaner test of "ingredients present"

A3 cannot be made cleaner about the *network*, because the network has no access to h(c). It can be made cleaner about the *format*, which is the claim the paper needs:

- **F3 (free, CPU-minutes once hidden states are re-fetched): decompose the oracle.** Score P2r features on (true a, decoded Δŝ) and on (decoded â, true Δs), same folds and cross-fitting as A3. If (true a, Δŝ) is close to P2r(true) (0.80–0.88 on Base-Lora), the outcome side of h(c+1) *is* in command-comparable form and reading (b) above wins: the paper rewrites Contribution 3 as "the representation format supports the comparison given a supervised state readout; what is missing is the channel." If it stays near the oracle (0.60), the outcome decode is the bottleneck and reading (a) partly survives. Pre-registrable rule: "format supports the comparison" iff (true a, Δŝ_A) ≥ P2r(true) − 0.10 on ≥ 4 of 5 Base-Lora swap runs.
- The genuinely causal test of M1 remains supplying the command as *input* and training (X6). X3 (frozen model, tokens spliced) is not that test; see §5.

---

## 2. Is "memoryless state estimator; knows past motion only through current state" justified?

### What holds

- A2 (`a2_*.csv`): s(c+1) reads out of h(c+1) at R² 0.81–0.97 (best cell); Δs(c) at 0.40–0.85. M2 ("outcome not metric") is rejected cleanly.
- A2b (`a2b_*.json`): on all calls, H (ridge on h(c+1)) ≈ S2 (per-task k-NN on the true s(c+1)): 0.834 vs 0.826, 0.849 vs 0.840, 0.585 vs 0.577, 0.594 vs 0.580. On HIJACK calls H is at or below S2 in all four (0.666 vs 0.673, 0.718 vs 0.717, 0.299 vs 0.339, 0.336 vs 0.351). H+S1 is within 0.005 of H. The pre-set reading "regularity explains A2(ii)" fires in 4 of 4.

### Four corrections

**(a) The estimator is the vision encoder, and the readout is already at the input.** From `a2_*.csv`, `ctx_mean` at **layer 0** — the input embedding, i.e. mean of the 256 projected vision patches plus fixed prompt tokens (README pool table; `capture.py` averages `vis` over patches the same way) — reads out s(c+1) at R² 0.965 / 0.961 (GRPO s1/s2) and 0.873 / 0.860 (Base-Lora s1/s2); every later `ctx_mean` cell adds ≤ 0.015. Δs at L0 is 0.806/0.820 (GRPO) and 0.552/0.551 (Base-Lora), rising to 0.834/0.850 and 0.584/0.591 by L20–L24 `act_mean`. So "the policy is a high-fidelity state estimator" should read "the robot state is linearly present in the projected image embedding and stays present through the LM; the LM adds little." That matters for the paper's framing: nothing about the *policy's* computation is being credited.

**(b) "RL sharpens the state estimate" is, at layer 0, a data-distribution statement until shown otherwise.** The GRPO–Base-Lora gap exists at L0 (0.965 vs 0.873). The GRPO root differs from the Base-Lora root by relative 0.00045 (an LLM q_proj) and 0.0033 (a vision qkv), with layernorms moved (full-parameter RL; logbook entry "How A loads", ~line 1001). A 0.3 % weight change can change a linear readout, so this is not settled, but the simpler account is that GRPO's near-deterministic trajectories (success 0.93–0.98 vs 0.61, `e4_report.md`) put the arm on a tighter image→state manifold. R² also depends on the target variance. **F4 (free):** report readout RMSE in physical units per state dimension alongside R², and the SELF state variance per checkpoint; and evaluate each checkpoint's ridge only on calls inside the overlap of the two state distributions (per-task matched support). If the RMSE gap closes on matched support, write "RL narrows the state distribution and the state becomes more readable", not "RL buys a sharper state estimate".

**(c) A2b did not run the residual test.** H ≈ S2 says h is no better than the best state-only predictor; it does not say h carries nothing *beyond* the state, because two predictors of equal R² can be complementary (both are at 0.83, far from 1). The pre-registered flag "h carries motion beyond the state" was tested only on HIJACK calls and only as H vs S2, never as H+S2 vs S2. **F2 (free):** regress the k-NN residual (Δs − S2 prediction, out-of-fold) on h(c+1); the gain in R² over S2 is the motion information beyond the 8-dim state. Rule (to pre-register): "no motion trace beyond state" iff the gain < 0.05 on all calls and on HIJACK calls, all four goal-swap runs. I expect it to pass, but the current wording claims it without the test.

**(d) "Memoryless" is architecture, not a finding.** The policy sees one image and the instruction (`main.tex` §Intro already says so). A2/A2b test something narrower and worth saying precisely: *the observation at c+1 carries no visible trace of the last motion beyond where the arm now is* (no motion blur, no settling, no displaced objects that the 8-dim state misses — to the extent the test can see them). Note the 8-dim state is an incomplete observable (§3); the honest version is "beyond the end-effector state."

Net: keep the claim, demote it from "the policy is a memoryless state estimator" to "the premise is architectural; empirically the input embedding exposes the end-effector state almost perfectly and the last motion only through it (A2b, pending the residual check)."

---

## 3. What explains GRPO goal-swap s1's P1?

### First, the size of the thing

From `e4_report.md` and `a8_e4_grpo_goal_swap_s1_linear.json`: P1 0.563 ± 0.033 at L12/`ctx_mean`, floor 0.528, margin **+0.035**, which R1 calls "at floor" (δ = 0.04). The three comparisons that make it look like an excursion are all against controls below 0.5: C_dstates 0.460 ± 0.041, C_state 0.484, C_state+d 0.488. Half of A7's +0.103 [+0.056, +0.149] is C_dstates being 0.04 under chance; against 0.5 the gap is 0.063, against the floor 0.035. A8's "P1 beyond state" rule (P1 ≥ max control + 0.04) is met only because the controls are at 0.488. Sub-chance balanced accuracy on grouped CV is ordinary (a per-task feature set that fits within-episode structure and reverses on held-out episodes; 0.46 is ~2.5 se below 0.5 at n = 952, so partly noise too), but it means these comparisons carry no evidence that P1 is "beyond" anything. The s2 excursion is the real one (+0.046 linear, +0.064 MLP; C_cmd 0.552 off-band in linear; MLP "state accounts for P1", `a8_e4_grpo_goal_swap_s2_mlp.json`).

What survives is a weaker, more interesting pattern: GRPO's linear P1 margin exceeds Base-Lora's on 3 of 4 matched suite×seed cells (goal s1 +0.035 vs +0.018, goal s2 +0.046 vs +0.029, object +0.062 vs +0.037; spatial +0.050 vs +0.056 goes the other way; `a4_visibility.csv` column `p1_margin_linear`), every GRPO excursion sits at L4–L20 `ctx_mean`/`act_mean` (`a5_locus.csv`), and on both GRPO goal seeds P1's best cell is exactly A2's best state-readout cell (s1 L12/`ctx_mean`, s2 L8/`ctx_mean`; `a3_*.json` `state_cell`) — weak evidence, because the A2 profile is flat to 0.01 across `ctx_mean` cells, but consistent with P1 reading a state-like quantity.

### Candidate explanations, ranked, with what existing data can and cannot say

1. **A sharper (or narrower) state code makes the same small Δs anomaly more linearly separable.** Consistent with A2 (GRPO Δs readout 0.83–0.85 vs 0.58–0.59), with the locus, and with the cell coincidence. Against it: A4 says the anomaly is *not* more visible in true-Δs space on s1 (AUROC 0.520 vs 0.528; `a4_visibility.csv`), and the linear C_state/C_dstates controls, which have the true 8-dim state, find nothing. That leaves the model reading something the 8-dim state does not contain.
2. **Observables the 8-dim end-effector state misses.** Two concrete ones: (i) **object poses** — a donor chunk from a different initial state can grasp air, nudge an object, or toggle the gripper at the wrong moment; (ii) **joint posture** — the 7-DoF arm's null-space configuration under OSC is history-dependent and visible in the image but not in eef pose + gripper; on GRPO's stereotyped trajectories posture is a regular function of phase, so a donor chunk pushes it off the manifold. Neither is logged (`harness.py` writes `states_before/after` from `obs["states"]`, 8 dims; `frames` for 5 episodes only). The P4 − P1 = +0.090/+0.084 on GRPO freeze (`a1`, "descriptive") against +0.031/+0.008 on Base-Lora, and P4 0.719 beating C_state+d 0.606 on GRPO freeze s1 (`a8_e4_grpo_goal_freeze_s1_linear.json`), point the same way: a frozen command leaves the eef state fixed but lets posture/scene settle.
3. **Phase or competence structure.** GRPO task 5 is flagged low-competence on both seeds (`e4_report.md` R12). C_phase for the GRPO runs is in `probe_results.csv` (read by `e4_report._family_numbers`) but not in any X0 table.
4. **Value-head shaping at `ctx_last`.** Not supported: the excursion is at `ctx_mean`/`act_mean`, never `ctx_last` (`a5_locus.csv`).

### Can the scene explanation be tested for free?

- **`HiddenStore.vision` — no.** `capture_vision: false` in every E4 config (`configs/main02*.yaml`, `suite_*.yaml`; `e4_plan.yaml` overrides only decoding). The closest free proxy is L0/`ctx_mean` (projected patches + fixed prompt), which A6 already used; for GRPO s1 its P1 margin is < +0.021 (not in the top 3 of `a5_locus.csv`), so the signal is not linearly present at the input level and grows by L8–L12. That says "computed from the image", not "scene vs robot".
- **Donor chunk — partly.** `donor_episode_id` and `a_exec` are logged (`harness.py` ~l. 365–378). **F7 (free):** correlate per-sample P1 out-of-fold correctness (A7 keeps OOF predictions) with intervention strength ‖a_cmd − a_exec‖ and with the gripper-toggle indicator; and tabulate OOF P1 balanced accuracy per task. If hits concentrate on large-deviation or gripper-toggle samples, P1 is reading outcome magnitude through a sharper code (candidate 1 + a scene-ish component); if they concentrate on task 5, it is competence (candidate 3).
- **F5 (free): residualised P1.** Regress h(c+1) at the P1 cell on the per-task linear state features (A8's `_per_task_state`) and probe the residual. This replaces the sub-chance-control comparison with a direct one: if P1_resid ≈ P1, the signal is orthogonal to a linear state read; if it drops to floor, candidate 1 wins and the "beyond state" line should be deleted.
- **A decisive test needs one new collection** that logs object poses and joint positions per call (a few floats; robosuite exposes both in `obs`/`sim`). Then C_scene and C_joint are observables twins at zero probe cost. Fold this into X1 and X4 (§5) rather than running a separate collection for it.

Recommendation for the paper: state GRPO's P1 margins as "small (+0.035 to +0.064), early-to-mid-layer, not explained by the 8-dim end-effector state, not larger than the outcome anomaly in true-Δs space; candidates are a sharper state code and observables the state vector omits; tested in X1 with scene logging." Do not promote it beyond that.

---

## 4. The strongest honest thesis, and what is fragile

### Thesis (RSS framing)

> **A deployed single-image VLA cannot tell that its last action was overridden, and the reason is structural, not a training accident.** The command exists only in the forward pass that produced it and the outcome only in the next; the policy's activations expose the current end-effector state almost perfectly — already in the projected image embedding (R² 0.86–0.97) — and the last motion only through that state. Across 30 run×family rows on three checkpoints and four suites, adding the command to the post-action state changes nothing a probe can read (P3 − P1 within ±0.04; CIs cover 0), and on every kinematically plausible swap the hidden rungs sit at a selection-matched floor. The ingredients are recoverable by a reader with access to both calls, which is exactly what a ten-line external comparator can exploit and what the policy cannot. RL post-training on the policy's own consequences narrows and sharpens the state readout and adds no efference increment, within a 0.04 detection limit on one SFT→GRPO pair. **Consequence for practice:** override and failure detection on these policies must be external, or must come from an added input channel carrying the signed, instantaneous command (the GRU format result).

What this thesis does *not* need: novelty-gated compensation (A6 pooled ρ = +0.026 [−0.003, +0.055]; E4 mirror reaction null on all four cells per `next_steps_fable.md` §D), a "comparator the network fails to compute", or "RL buys a sharper state estimate" as a representational claim.

### Fragile claims, honestly labelled

| Claim | Why fragile | Fix |
|---|---|---|
| "RL adds no efference increment" | One pair (Base-Lora → GRPO-130); GRPO s1 has no MLP ladder; R4 verdict is "inconclusive" by the pre-registered rule, not "no" | X1 (second pair); finish s1 MLP locally; write "within δ, on one pair, replicated on a second" |
| GRPO P1 margins > Base-Lora's | 3 of 4 cells, each inside or just past δ; best-of-27 selection on both sides; controls sub-chance on s1 | Pre-register a **fixed cell** for X1 (L12/`ctx_mean`, from E4), compare OOF accuracies paired by episode; F5/F7 |
| A3 "3 of 4 / 8 of 9" | GRPO s2 passes by 0.007 with fold sd 0.06 | Report per-checkpoint; Base-Lora 5/5 is the solid part |
| Best-cell numbers | A7/A8 CIs are conditional on the selected cell (report says so); BLAS changes fold scores | Quantify: across the three `a7_passes/` settings, pooled P1 − C_dstates moves ≤ 0.010 (SFT freeze s2: −0.105 vs −0.115; its P3 − P1 +0.025 vs +0.036), and a single fold's balanced accuracy moved up to 0.054 (SFT freeze s2 P1, avx2 vs avx512). State this as the reproducibility resolution (~0.01 on pooled differences) in Methods; it is a quarter of δ, which is reassuring, and the one P3 − P1 CI that excludes 0 (SFT freeze s2) sits inside that resolution |
| Suites single-seed | Spatial is "beyond observables" for *both* checkpoints (linear only, one seed) and has no explanation | Say so; the scene/joint logging in X1 is the only cheap way to get an observables twin for it |
| "Observables-only twin" | C_dstates/C_state see 8 dims; the camera sees objects and posture | Log them; until then write "end-effector observables" |
| A6 novelty operationalisation | L0/`ctx_mean` Mahalanobis in PCA-50 is one choice; a null here is weak evidence against the mechanism | Fine to drop the mechanism anyway (E4 behavioural nulls carry that); do not claim A6 refutes it |

---

## 5. Next steps, ranked

Costs: L40S $1.95/h (yours); E4 collections took 6.7–12.9 min for goal arms, 10–12 min spatial/object, 16.6–22.7 min long (`e4_report.md` ledger) → **$0.25–0.45 per goal-suite collection, $0.55–0.75 long** `[estimate]`. Linear probe rounds: 5.8–25 min on 24 Modal CPUs; the orchestrator reserved $0.50 worst case per round (logbook) → **$0.1–0.5** `[estimate]`, or free on this container at 4 CPUs (~30–120 min per round `[estimate]`, hidden states ~1 GB per run fetched free from the Volume). MLP rounds: $1.8–3.9 per attempt on Modal with six preemptions seen (logbook 13:22 entry); 12–25 h on the Windows box `[inference from "~7× slower"]`. Credit: ~$12 `[TO VERIFY on the dashboard]`.

Everything below needs a logbook pre-registration before any number is computed or any container starts; the rules are written so they can be pasted.

### Tier 0 — free analyses on existing data (this container; re-fetch the four goal-swap runs' hidden states)

| # | Analysis | Hypothesis / rule | Changes in the paper |
|---|---|---|---|
| **F3** | Oracle decomposition: (true a, decoded Δŝ) and (decoded â, true Δs) → P2r features, A3's folds | "Format supports the comparison" iff (true a, Δŝ_A) ≥ P2r(true) − 0.10 on ≥ 4/5 Base-Lora swap runs | Rewrites Contribution 3 correctly; decides which reading of P3 − P1 = 0 the paper defends. **Do first.** |
| **F2** | A2b residual: ridge of (Δs − S2 OOF prediction) on h(c+1) | "No motion beyond end-effector state" iff ΔR² < 0.05 on all and on HIJACK calls, 4/4 runs | Licenses the "only through the state" sentence |
| **F4** | State readout in physical RMSE + matched-support comparison GRPO vs Base-Lora, L0 and best cell | "Data effect" iff the RMSE gap on matched support shrinks by ≥ half | Decides "RL sharpens the estimate" vs "RL narrows the distribution" |
| **F5** | Residualised P1 (h at the P1 cell minus per-task linear state fit), GRPO s1/s2 and Base-Lora s1/s2 | "State-orthogonal" iff P1_resid ≥ P1 − 0.02; "state-explained" iff ≤ floor + 0.02 | Replaces the sub-chance-control comparison in §3 |
| **F7** | Per-task OOF P1 accuracy; P1 hit rate vs ‖a_cmd − a_exec‖ and gripper toggles (donor fields are logged) | Descriptive; flag if task 5 carries > half the margin | Competence vs scene reading for GRPO's margin |
| F6 | Report C_phase for the GRPO runs; pooled sign test of GRPO − Base-Lora P1 margins (4 cells) | Descriptive | Closes a hole a reviewer will find |
| F1 | Quantify BLAS resolution from `a7_passes/` (done above: ≤ 0.010 pooled, ≤ 0.054 single fold) | — | One Methods sentence |

CPU: hours, not days `[estimate]`; nothing on Modal. Expected effect on the writing: larger than any $5 collection.

### Tier 1 — Modal, in this order

**X1. Second SFT→GRPO pair on libero_goal (goal-traj1 SFT → `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal`).** `PREREG_wave3.md` E2 already specifies the configs (`e2_rl` swap 10 tasks × 50 states; `e2_rl_mirror/freeze` tasks {2,3,6,8,9} × 50) and the readout.
- *Hypothesis:* GRPO's pattern is a property of the objective: P3 − P1 ≈ 0, P1 up at most by outcome perception, no E3 migration, state readout narrower/sharper.
- *Rules:* R1–R3 per run with the same δ; the only positive is "P3 − P1 ≥ 0.04 on both pairs" or "P1 ≥ C_dstates + δ and ≥ C_state+d + δ and ≥ C_scene + δ in the MLP family on both seeds". Add one **fixed-cell** test to escape selection: L12/`ctx_mean`, pre-named from E4, paired OOF comparison RL vs SFT, bootstrap CI.
- *Design changes before collecting (code, not analysis):* log object poses and the 7 joint positions per call; set `capture_vision: true`; size episodes by the Amendment-1 rule to match `main02`'s 1266 samples rather than 500 episodes; **re-collect the SFT side on the same stack** (one `main02`-equivalent, ~9 min) so the pair is like-for-like; wave-2 `main02` was collected on a different machine and its hidden states are not on the Volume `[TO VERIFY]`.
- *Cost:* RL swap ~13–25 min, mirror + freeze ~12 min each, SFT swap ~9 min → ~0.8–1.0 GPU-h → **$1.6–2.0**; linear rounds free locally or ~$1 on Modal → **$2–3.5 total** `[estimate]`. (Earlier memo: $4–6 at $2.81/h and 500 episodes.)
- *Changes in the paper:* turns "one pair, inconclusive" into a replicated statement either way; supplies the first observables twins for scene and posture, which also retire the spatial-suite anomaly or make it a finding.

**X4. External comparator closing the loop.**
- *Design:* the P2r logistic on true (a_cmd, Δs), fit on an existing run's samples (e.g. `e4_sft_goal_mirror_s1`), applied online in a new seed; when it fires, re-issue the commanded chunk once. Arms: mirror on both E4 checkpoints (largest clean − probe success gaps: Base-Lora 0.65 vs 0.47, GRPO 0.94 vs 0.77, `e4_report.md`) and swap on Base-Lora (0.65 vs 0.60). The existing E4 runs are the unmonitored arms (same checkpoint, same stack).
- *Rule:* clean − probe success gap shrinks by ≥ half, episode-bootstrap CI excluding 0; report the monitor's false-positive rate on SELF calls and the success cost of false recoveries.
- *Cost:* 3 collections ~10 min each → **$1–1.5**; no probing → **$1.5–2.5 with margin** `[estimate]`; code: ~a day (monitor in the harness loop, recovery action, logging).
- *Caveats to write in:* the monitor sees simulator state, so this is the "proprioceptive comparator" sentence, not a learned-representation result; at GRPO's 0.92–0.98 success the swap gap is compressed; the result is a systems datum that gives an RSS reviewer the actionable line.

**X2. E5 step 1 (VRAM gate), $0.5–1 `[estimate]`.** Already coded (`vram_probe.py`) and pre-registered (`PREREG_wave3.md` E5: batch ≥ 4 under 22 GB on a 24 GB card). Only worth running if the X6 compute ask will be written; run it on a 24 GB-class GPU (A10G) rather than the 48 GB L40S so the gate means what it says `[TO VERIFY A10G availability/price]`. It changes nothing in the paper by itself.

### Tier 2 — local, background, no Modal

- **GRPO goal-swap s1 MLP.** Cannot move R4 (s2 is already "explained by C_dstates", `e4_report.md`), but closes the most visible gap in the tables and gives A5/A7 a second primary-family seed. 12–25 h on the Windows box `[inference]`; three Modal attempts already lost. Run it there; do not buy it.
- **The eight R6 mirror/freeze MLP rounds.** After A8 the linear lead is narrower than the report says (GRPO mirror s2 "P1 beyond state", mirror s1 and freeze s1 "unresolved", freeze s2 "state accounts", `a8_*`), and the 8-dim twin is incomplete anyway. Run in the background; report "not run" if they are not done when the paper is due.

### Tier 3 — park or cut

- **X3 (frozen model, previous chunk spliced into the prompt).** Cheap (~$0.5 + a probe round) but nearly uninformative: the LM was never trained to attend to action tokens in the prompt, and `PREREG_wave3.md` already says a frozen model is not E5. A null says nothing about M1; a positive would be surprising. Its only value is as a feasibility pilot for X6 (does the policy survive the tokens; success rate). Run it only if X6 is going ahead, and pre-register it as a feasibility check, not a test.
- **X6-sibling (off-the-shelf OFT with proprio input).** X0 removed its rationale: M2 is rejected (the state is already readable from vision), so giving the policy the state explicitly tests nothing the paper needs, and the wrist camera changes outcome visibility. Cut.
- **X6 proper** (LoRA fine-tune with the previous command as input): still the experiment that turns the null into a positive design result, still $25–60 + a demo pipeline `[earlier estimate, unchanged]`. Out of this budget; write it as the compute ask, with X2's number.
- **X7 (world-model comparator):** park, as before.

### Budget plan `[estimate]`

F-analyses $0 → X1 $2–3.5 → X4 $1.5–2.5 → X2 $0.5–1 only if the X6 ask is being written → reserve ~$4–5 for one re-run. Never an MLP round on Modal.

---

## 6. What to cut or reframe (after X0)

1. **Title and Contribution 4.** "Novelty-gated compensation" does not survive: A6 pooled ρ = +0.026 [−0.003, +0.055] (`a6_pooled.json`), and the mirror reaction is null on all four E4 cells. Keep the behavioural battery as a descriptive section ("small, checkpoint-dependent reactions; freeze never reacts"). Alternative titles already in `main.tex`'s comments ("The policy reacts to what it sees, not what it did") are closer to the data than the current one.
2. **Contribution 3 (capacity argument).** Rewrite per §1(ii): "a two-stage reader — supervised state decode, then a fixed comparison — recovers the label from linear decodes of the two calls' activations (0.60–0.68); a one-stage probe at n ≈ 1k does not. The information is present and compactly recoverable; it is never co-present in one forward pass." Pending F3.
3. **"The policy is a state estimator."** Attribute to the input embedding (L0 R² 0.86–0.97) and say the LM adds ≤ 0.015. "RL sharpens the state estimate" → pending F4, default to "narrows the state distribution and makes it more readable".
4. **"Located deficit."** Replace with the architectural sentence (§1(i)). The deficit is not located inside the network; it is the absence of a channel.
5. **GRPO s1.** Delete "excursion"; it is at floor by R1. Keep the pooled, cautious pattern of §3.
6. **Observables twin.** Rename C_dstates/C_state as *end-effector* observables until scene/joint logging exists; say explicitly that the camera sees more than the 8-dim state.
7. **Scope.** Abstract's "seven runs" → ~27 runs across three checkpoints; E4-RL as a main section with the not-run ledger (GRPO s1 MLP, eight R6 rounds, GRPO long linear); rename wave 3's "E4" to avoid the clash (already noted in the logbook).
8. **T1–T5** to a methods box plus appendix, as the previous memo said; add T6: "a supervised two-stage reader can find what a one-stage probe cannot — report both or the null overclaims."

---

## 7. The hardest reviewer objections after X0

1. **"A memoryless vision-only policy cannot attribute by construction; your null is a tautology, and now A2b says everything is state regularity."** Concede it and make it the premise (the paper half does). The content is then: (a) the measured bound on outcome perception — floor on same-task swaps in two families and three checkpoints, bounded by end-effector observables on gross violations; (b) the two-call reader result (A3/F3), which is the design datum; (c) the RL comparison; (d) the GRU format requirement. Do not argue the network "fails" to compare.
2. **"Your oracle is supervised with ground truth and uses the previous call's activations; it says nothing about the policy."** Correct, and say so first. It is a statement about the activations' format and about what an external reader can do, offered as the dimensionality check Limitations asks for. F3 makes it sharper; X4 makes it actionable.
3. **"Then Contribution 3 is wrong: the MLP's failure on a ⊕ h(t+1) is sample complexity, not absence."** Partly right; §1(ii). Rewrite the contribution; keep the empirical fact (same family, same samples, raw ingredients 0.687 vs hidden 0.540) and add the supervised-reader number. This is the objection most likely to sink the current draft, and X0 is what exposed it.
4. **"RL: one pair, s1 MLP missing, margins inside δ, controls below chance, C_cmd off band on s2 — 'RL does not buy it' is underpowered, and your own 3-of-4 pattern hints it buys something."** Answer with X1 (fixed-cell pre-registered test, scene twins), the bootstrap CIs (A7: P3 − P1 covers 0 on every GRPO run), and the wording "within δ; no efference increment; outcome perception possibly stronger, candidates named and tested."
5. **"Your observables-only controls see 8 numbers; the camera sees objects and the arm's posture. Spatial is 'beyond observables' for both checkpoints and you have no explanation."** Concede; it is the one hole X0 could not close. X1's logging closes it at zero marginal cost; until then write "end-effector observables" and list the candidates.
6. **"Sim only, one model family, T = 1.6, and results depend on the BLAS kernel."** Scope statement; the BLAS resolution is ~0.01 on pooled differences and ≤ 0.054 on a single fold (`a7_passes/`), a quarter of δ — say it in Methods rather than let a reviewer find it in the logbook.

---

## 8. Checklist before any launch

- Logbook pre-registration for each of F2–F7 (rules above), X1 and X4 (rules above), including the fixed cell for X1 and the monitor's training run for X4.
- Collection-code change, tested offline: per-call object poses, 7 joint positions, `capture_vision: true`; re-run `verify_indexing`.
- `[TO VERIFY]`: Modal balance; whether wave-2 `main02` hidden states exist anywhere (decides whether the SFT side of X1 must be re-collected — recommended regardless); the exact HF id and sampling settings of the goal-traj1 GRPO sibling; A10G price for X2.
- Nothing in this memo asks for an MLP round on Modal.

# Wave 3 pre-registration — what makes "that wasn't me" readable?

Append-only. Written 2026-10-10, **before any wave-3 data exists**; the git
commit that adds this file is the timestamp. Changes after that are dated
amendments at the bottom, never edits above them.

Context: the RobotPAD workshop paper (wave 1) and its wave-2 replication found
no linearly readable self-vs-hijacked signal in OpenVLA-OFT (SFT, LIBERO-Goal),
a working comparison-feature positive control, and a readable motor plan
(E1 R² = 0.64). Wave 2 did **not** replicate the SWAP behavioural reaction
(main02: Wilcoxon p = 0.53 / 0.44); no claim below depends on it.

Working hypothesis (from the merged plan): readable attribution needs three
parts — a retained command, the outcome, and a comparator. The SFT VLA has
only the outcome. Line A (Evan Ye) asks where the command must live; line B
(Jiazhuo Li) asks whether a frozen world model can be the comparator.

Shared yardstick. **MIN_EFFECT = 0.04** balanced accuracy above the
selection-aware floor: the paper's detectability limit at n ≈ 1.7k, used for
the GRU and the VLA alike so the two are judged on one scale.

---

## E1 — GRU mismatch (line A, cheapest)

Code: `experiments/gru_mismatch/` (a port of Ye 2026's exp3/exp4, MIT).

**Gate (done before this file was written, not the hypothesis).** The port
reproduces exp4's trailing recall at the source seed exactly: **56.5%**
(seed 42, full scale). exp3 (no trace) gives **6.5%** against the source's
"~12.3%" (its exp4 docstring says 9–15%); the exp3 code path was checked line
by line against the source and is identical, so this is recorded as an open
question for the author, not a port bug. The plan gates on exp4 only.

**Disclosure.** While debugging, one quick-mode smoke run (step counts / 5,
seed 42, `trace_abs`) was looked at: swap margin −0.05, mirror −0.03, freeze
+0.25 with outcome-only C_out 0.88. No full-scale phase-3 number exists.

**Design.** Interfaces `none` (exp3), `trace_abs` (exp4), `trace_signed`,
`command` (instantaneous signed `a_t` as input) × seeds 42–46 × arms swap /
mirror / freeze. Training exactly as the source; then 40k frozen steps with
hijacks (p = 0.25 on active steps, warm-up 4 steps after burst onset, clean
gap 2). On a hijacked step the efference input gets the command `a`, the world
moves by `γ·a'`: swap `a'` ~ U(−2, 2) independent, mirror `a' = −a`, freeze
`a' = 0`. The three arms replay the identical world (paired).

**Probes.** The VLA pipeline's own sample builder and `run_probe` (5-fold CV
grouped by burst, phase-matched negatives, nested C, linear family).
Selection floor = max(P0 on `h_multi`, P0 on `h_gru`); best hidden =
max(P1, P1g); permutation null (50) on the `h_multi` cell.

**Readout.** An arm is *readable* when in ≥ 4/5 seeds best hidden ≥ floor +
0.04 **and** > permutation p95; *uninformative* when P2r − floor < 0.10.
**Primary cell: `trace_abs` × swap** (Evan's model, the paper's primary
transform). Swap and mirror are the clean comparison tests (outcome-only
C_out cannot see them); freeze is also an outcome anomaly in this world, so a
readable freeze arm only counts as attribution if it exceeds C_out.

**Plan consequence (from the merged plan).** Primary readable → keeping the
command suffices → E5 is the main line-A experiment. Not readable → a
comparator is needed → E4 is the main experiment. If `command` is readable
while `trace_abs` is not, the reading is "the command must be on the input in
usable (signed, instantaneous) form", which is what E5 gives the VLA.

Predictions written now: `trace_abs` swap and mirror not readable (the |a|
EMA carries ≤ 0.05·|a_t| of the instantaneous command and no sign); `command`
swap readable.

## E2 — RL control (line A)

Configs: `e2_sft.yaml`, `e2_rl.yaml` (swap, 10 tasks × 50 states, seed 1,
identical episode set), `e2_rl_mirror.yaml`, `e2_rl_freeze.yaml` (tasks
{2,3,6,8,9} × 50; SFT counterparts main02_mirror / main02_freeze). Checkpoint:
`RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal` (base: the probed traj1 SFT; card
sampling temperature 1.6). Hongzhi Zang confirmed only the final checkpoint
exists, so there is no training-time sweep.

**Analysis.** `run_probes.py --stage main --permutations 200` per run
(linear; MLP family on e2_sft and e2_rl if budget allows), then
`compare_runs.py --run sft:none=… --run rl:none=… --baseline sft:none`.

**Readout, reported in two separate tables.**
- *Attribution*: per run, readable iff any of P1/P3/P4 ≥ the run's own floor
  + 0.04 and > its permutation p95. Raw accuracies are never compared across
  runs (floors depend on n).
- *Compensation*: hijack cost = success(clean) − success(probe episodes)
  (randomised per episode), bootstrap 95% CI; RL − SFT difference with CI.
  E3 next-call reaction and innovation-corrected undo alignment alongside.
- RL's swap arm is uninformative if its P2r − floor < 0.10 (competence can
  change intervention strength).

**Mapping.** RL not readable and RL cost < SFT cost (CI excludes 0) →
supports Ye's prediction (compensation without encoding survives RL). RL
readable → the training objective can produce attribution; revise the
three-part hypothesis. Caveat recorded now: at ~95% success the cost is
compressed; read it with `success_clean`.

## E3 — world-model action sensitivity (line B, the gate)

Model: `RLinf/RLinf-Wan-LIBERO-Goal` (Wan2.2-TI2V-5B-shaped DiT + Wan2.2 VAE),
called as `WanEnv` does: reference = episode's first frame, 4 context frames,
5 condition actions + the 8 × 7 chunk, 5 denoising steps, cfg 1.0.

**Data.** The gate runs on `e4_wm_swap` (substep frames; faithful context).
An exploratory pass on wave-1/2 frames (`static` context: the call frame × 4,
"held still" condition actions) may be run first; it is reported but never
used for the gate, because a static mid-episode context is off-distribution.

**Queries.** 300 SELF calls with call index ≥ 1, sampled uniformly (seed 0).
Candidates: `cmd`, `cmd` at a second noise seed, `mirror(cmd)`,
`freeze(cmd)`, `swap` (a same-task episode's commanded chunk at matched
phase), and 6 more such alternatives.

**Metric.** Frames → [0, 1], 4 × 4 block-mean (256 → 64 px); d = MSE over
the 8 predicted frames. Noise = d(P_cmd^seed0, P_cmd^seed1);
r_T = d(P_cmd, P_T) / noise (same noise seed for both).

**Gate.** Proceed to E4 iff **median r_freeze ≥ 1 and median r_swap ≥ 1**
(changing the action moves the prediction more than resampling the noise).
Reported without threshold: Jiazhuo's ratio (spread across the alternatives'
predictions) / (mean prediction's motion from the last context frame), and
SELF fidelity: the fraction of queries where d(A, P_cmd) is below the median
of d(A, P_alt). Below the gate: stop line B and report to Jiazhuo.

## E4 — world model as comparator (line B)

Data: `e4_wm_{swap,mirror,freeze}` (SFT traj1, tasks {2,3,6,8,9} × 20 states,
seed 2, every substep frame). Samples: the probe pipeline's phase-matched
pairs, so the WM and the policy's own probes see the same transitions.

**Scores** (A = the 8 real substep frames, P_k = WM prediction for candidate
k, P̄ = mean over the 6 alternatives; all at the same noise seed):
- *Primary comparator — controllable-subspace distance*:
  U = orthonormal basis of {P_cmd − P̄, P_alt_k − P̄}; score =
  ‖Uᵀ(A − P̄) − Uᵀ(P_cmd − P̄)‖² / dim. (The naive d(A − P̄, P_cmd − P̄) is
  identically d(A, P_cmd); projecting onto what actions can change is what
  removes the WM's action-independent error.)
- Secondary comparators: raw error d(A, P_cmd); rank score
  d(A, P_cmd) − mean_k d(A, P_alt_k); controllable-subspace cosine.
- *Primary novelty baseline* — the same machinery without the command:
  ‖U_altᵀ(A − P̄)‖² / dim, U_alt spanned by the alternatives only.
- Other baselines: d(A, P̄); image motion d(A_last, context_last); the
  policy's next-call action entropy.

AUROC per score per transform, 95% CI from 2000 episode-level bootstraps;
differences by paired bootstrap.

**Readout (FREEZE is the key condition).**
- Comparator supplies what the policy lacks: on FREEZE, AUROC(primary) ≥ 0.65
  with CI lower bound > 0.5, **and** AUROC(primary) − AUROC(primary novelty)
  ≥ 0.10 with CI excluding 0.
- Comparator only reads novelty: neither clears 0.6 on FREEZE, or their
  difference's CI includes 0.
- Swap and mirror reported with the same statistics; the policy-entropy
  baseline on FREEZE is expected near 0.5 (the paper's null reaction).

## E5 — command input on the VLA (line A, most expensive)

Step 1 only is pre-registered: `vram_probe.py` measures peak memory of a
LoRA (r = 32) forward + backward on OpenVLA-OFT with the previous command
chunk in the prompt, gradient checkpointing on. **Proceed iff batch ≥ 4 peaks
below 22 GB** on the 24 GB card; otherwise defer, as the plan says. The
training and probing design will be appended here as an amendment before any
training run. Warning carried from the plan: a read-out head on the frozen
model is not E5 (command + frozen hidden state already sits at the floor in
the wave-2 MLP probes); the backbone must be trained.

E2 and E5 together fill the interface (command input or not) × objective
(SFT or RL) grid that `compare_runs.py` lays out.

---

## Amendments

### Amendment 1 — 2026-10-11 — E1b: both channels at once

Written after the E1 results (logbook, 2026-10-10) and **before any `both`
run**. Exploratory extension of E1; the E1 readout above is unchanged.

*Why.* E1 found a format dissociation. The exp4 |a| trace carries the self
(trailing recall 53.6 ± 3.9%) but no mismatch (swap −0.001, mirror −0.002);
the signed instantaneous command carries the mismatch (swap +0.135, mirror
+0.368) but trailing recall drops to 15.1 ± 13.7% (6–14% in four seeds).
Question: given both channels, does one state carry both readouts?

*Design.* Interface `both` = observation + the exp4 |a| trace + the signed
command a_t (GRU input 6-d). Everything else as E1: seeds 42–46, source
training protocol, 40k frozen phase-3 steps, swap / mirror / freeze, linear
probes, 50 permutations. Output: `experiments/gru_mismatch/results/e1b_both/`
(E1's directory is not touched).

*Readout.*
- Self intact: 5-seed mean trailing recall ≥ 40% (exp4's lowest seed is
  49.1%; every interface without the |a| trace averaged ≤ 15.1%).
- Mismatch readable: the E1 rule on swap and on mirror (≥ 4/5 seeds with
  margin ≥ 0.04 and above the permutation p95); freeze counts only beyond
  outcome-only C_out.
- **One state carries both** iff self intact and swap readable.
  **Interference** iff either fails although it held with that channel alone
  (trailing under `trace_abs`, swap under `command`).

*Prediction.* Both carried: the two channels enter as separate inputs, and
the trailing supervision can use the trace as in exp4. The swap margin may be
smaller than `command`'s +0.135.

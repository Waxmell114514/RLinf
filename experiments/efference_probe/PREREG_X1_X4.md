# X1 and X4 pre-registration — a second SFT→GRPO pair, and an external comparator in the loop

**Append-only.** Written 2026-10-29, before any X1 or X4 code change, collection or number exists; the git commit that adds this file is the timestamp. Later changes are dated amendments at the bottom, never edits above them.

**Approval.** Writing this file does not approve any Modal spend. Nothing here launches until the user approves it explicitly, after the code changes in §4 pass their offline tests.

## Where this comes from

- **The plan:** `reports/next_steps_fable_x0.md` §5, Tier 1.
- **What each experiment answers:**
  - X1 replicates the E4 SFT-vs-GRPO contrast on an independent pair, and adds observables the 8-dim end-effector state omits (object poses, joint positions).
  - X4 asks whether a comparator the policy lacks, run outside it, recovers the cost of a hijack.
- **What came before:**
  - **E4 and X0/F1–F7** (`reports/e4_report.md`, `reports/x0_report.md`) found:
    - no efference increment: |P3 − P1| < 0.04 in 30 of 30 rows;
    - the last motion is visible only through the end state (F2);
    - GRPO's small P1 margins are carried by the end state (F5);
    - RL's sharper state readout is a data effect (F4).
  - **Wave-3 E2 (`PREREG_wave3.md` on `main`, commit 9b83364)** pre-registered the same pair but was never run. X1 supersedes it. An amendment saying so goes into `PREREG_wave3.md` when this branch and `main` are merged.

## 1. Shared conventions

- **δ:** 0.04, the yardstick of every earlier wave.
- **Probe ladder and gates:** unchanged code (`run_probes.py --stage main --permutations 200`, linear). The readout functions are E4's (`e4_report.ladder_verdict`: floor = best-of-27 P0; beyond observables needs ≥ C_dstates + δ and C_cmd in [0.45, 0.55]).
- **Primary family: linear.** This deviates from E4, whose primary was MLP. MLP rounds cost $1.8–3.9 per attempt on Modal and were preempted six times in E4 (logbook), and this budget cannot carry them. MLP rounds may run locally afterwards; they are reported, never gating.
- **Bootstrap:** episode-cluster, B = 2000, seed 0, 95% percentile CI.
- **BLAS:** every refit compared with a CSV number runs under the setting that reproduces that CSV (`x0_mechanism.a7_select`).
- **Decoding:** `do_sample: true`, `temperature: 1.6` for every checkpoint and every collection, as in wave 2 and E4.
- **Budget rule:** a pre-registered GPU-hour cap per experiment (§5). The E4 orchestrator's budget gate and spend journal stay on, and a run at 2× its expected wall clock is stopped and reported.

## 2. X1 — second SFT→GRPO pair on libero_goal

### Checkpoints

| role | HF repo | note |
|---|---|---|
| SFT | `Haozhan72/Openvla-oft-SFT-libero-goal-traj1` | the waves 1–2 checkpoint (`main02`) |
| GRPO | `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal` | its GRPO sibling; the model card names traj1 as base [TO VERIFY at pinning] |

- **Pinning:** revisions and shard sha256s are pinned in `x1_plan.yaml` before S0, as `e4_plan.yaml` does.
- **Normalisation:** `unnorm_key: libero_goal_no_noops` for both.

### Collections

Configs are the wave-2 ones, unchanged except for the overrides listed here.

| key | config | tasks × states | seed | checkpoint |
|---|---|---|---|---|
| x1_sft_swap | main02.yaml | 10 × 20 | 1 | SFT |
| x1_grpo_swap | main02.yaml | 10 × k | 1 | GRPO |
| x1_sft_mirror | main02_mirror.yaml | {2,3,6,8,9} × 30 | 1 | SFT |
| x1_grpo_mirror | main02_mirror.yaml | {2,3,6,8,9} × k | 1 | GRPO |
| x1_sft_freeze | main02_freeze.yaml | {2,3,6,8,9} × 30 | 1 | SFT |
| x1_grpo_freeze | main02_freeze.yaml | {2,3,6,8,9} × k | 1 | GRPO |

- **Sizing (GRPO's k):** E4 Amendment 1's rule, k = min(50, max(k_base, ⌈k_base·r⌉)). r is SFT/GRPO positives per episode, measured on a pilot2 of 10 tasks × 8 states per checkpoint. SFT arms are never resized.
- **Overrides on every X1 collection:**
  - `capture.capture_vision: true`;
  - `capture.extra_observables: true` (new, §4);
  - decoding as in §1.
- **The SFT swap arm re-collects `main02`** (same checkpoint, config and seed) on the current stack. Its ladder is compared with `main02`'s wave-2 numbers as a reproduction check: P1, P3, P4 and P2r each within ±0.04. A miss is flagged and reported, not corrected.

### New observables (logged per call, before and after the chunk)

- **Joints:** the 7 arm joint positions (`robot0_joint_pos`).
- **Objects:** every object's position and quaternion from LIBERO's object observables in the raw robosuite observation. The keys are those ending in `_pos` / `_quat` that are not `robot0_*`; their order is fixed per run and written to the manifest. [TO VERIFY at S0: the exact key list.]
- **Vision:** the mean projected vision-patch embedding (`capture_vision`), unchanged code.

### New controls (same ladder machinery, linear)

- **C_joint:** joint positions after the chunk, plus their change over the chunk, plus task one-hot.
- **C_scene:** object poses after the chunk, plus their change over the chunk, plus task one-hot. LIBERO-Goal tasks share one scene, so no per-task interaction is used.
- **C_obs:** A8's C_state+d (8-dim per-task linear end state plus Δs), plus C_scene, plus C_joint.

### Pre-registered readouts

Each is applied per run unless stated.

1. **X1-R1, ladder:** R1–R3 per run, as E4 applies them.
2. **X1-R2, beyond full observables:** P1 (and P3, P4) counts as beyond observables iff it is ≥ max(C_dstates, C_state+d, C_scene, C_joint, C_obs) + δ, with C_cmd in band. This replaces the 8-dim-only "observables twin" for X1 runs.
3. **X1-R3, efference increment** (the replication):
   - **"No efference increment on two pairs":** |P3 − P1| < δ in every X1 run, and the A7-style paired bootstrap CI of P3 − P1 covers 0 on x1_grpo_swap.
   - **"RL produces an efference increment":** P3 − P1 ≥ δ with CI excluding 0 on x1_grpo_swap *and* on an E4 GRPO goal-swap seed.
   - Otherwise "unresolved".
4. **X1-R4, fixed cell (escapes best-of-27 selection).** Linear P1 at **L12/`ctx_mean`**, named now, refit out-of-fold on the ladder's folds. The statistic is GRPO − SFT balanced accuracy, with an unpaired episode bootstrap CI (episodes resampled within each run).
   - It is computed for the X1 pair and, as secondary, for the E4 pair (s1 and s2). L12/`ctx_mean` was GRPO s1's best E4 cell, so the E4 numbers at this cell are post hoc.
   - **"RL raises P1 at the fixed cell":** difference ≥ δ with CI excluding 0 on X1 and on both E4 seeds.
   - **"No RL effect at the fixed cell":** the X1 CI covers 0.
   - Otherwise "mixed".
5. **X1-R5, GRPO margins beyond the end state.** F5 (residualised P1) rerun on x1_grpo_swap with the residualiser extended to C_obs's features, using F5's conditions and the indeterminate case.
6. **X1-R6, hijack cost.** clean − probe success per run, RL − SFT difference with CI (E2's compensation table). Descriptive.
7. **X1-R7, X0 battery.** A1, A2, A2b/F2, A3, F3, F4 and F5 rerun on the X1 runs with unchanged code. They are reported next to E4's, each marked "replicates" or "does not" against its own pre-registered rule.

**Uninformative-arm rule (E2's):** an RL arm whose P2r − floor < 0.10 is reported as uninformative for attribution, because competence can change intervention strength.

### X1 stages and gates

- **S0, smoke:** both checkpoints, 1 task × 8 states, new logging on. **Gate:**
  - every logged call has 7 joint positions and ≥ 1 object pose;
  - the raw-observation eef position equals `states_after[:3]` within 1e-6 on every call;
  - the vision vector is present.

  Fail: stop, fix and re-smoke. X1 does not run without the new observables.
- **S1, pilot2:** both checkpoints, 10 tasks × 8 states. **Gate:** E4's amended pilot gate. Its yields give k.
- **S2:** the six collections above, in the order swap (SFT, GRPO), mirror, freeze, so that a budget stop cuts from the tail.
- **Probes:** linear rounds, on this container (free) or on Modal CPU if the user prefers speed (~$0.1–0.5 per round [estimate]).

## 3. X4 — external comparator closing the loop

### Design

- **Arm:** mirror on goal tasks {2, 3, 6, 8, 9}, for both E4 checkpoints (Base-Lora and GRPO-130), with `main02_mirror.yaml`'s hijack settings unchanged (p 0.25, warm-up 4, gap 2, probe fraction 0.8).
- **Why mirror only:** in E4 it is the only arm with a large hijack cost on both checkpoints. Clean − probe success, from the run manifests:

  | arm | Base-Lora s1 / s2 | GRPO s1 / s2 |
  |---|---|---|
  | mirror | 0.175 / 0.159 | 0.170 / 0.268 |
  | swap | 0.047 / 0.096 | 0.059 / 0.066 |
  | freeze | −0.057 / 0.097 | 0.018 / 0.041 |

  A monitor cannot show recovery where there is little to recover.
- **Episodes:** 50 initial states per task, seeds 3, 4, 5 and 6 (one collection each), so 4 × 250 = 1000 episodes per checkpoint. Seeds 1–2 are E4's.
- **Conditions:** randomised per episode, independently of the probe/clean flag, from a dedicated RNG stream (seed, task, batch, 0x4D4F4E), so the hijack schedule's stream is untouched:
  - **off:** monitor scores are logged, nothing is acted on;
  - **monitor:** a recovery follows each call the monitor fires on;
  - **oracle:** a recovery follows each truly hijacked call.

  Each condition has probability 1/3. That gives ~267 probe and ~67 clean episodes per condition per checkpoint.
- **Hidden capture:** reduced to L32/`ctx_mean` (not analysed), to save disk.

### The monitor (fixed before collection)

- **Features.** P2r's comparison block on the true command and the true state change of the chunk just executed: `mismatch_from_arrays(a_cmd_env(c), Δs(c))`, which equals `datasets._mismatch_features` exactly (tested in X0). They are computed online from `states_before` / `states_after`.
- **Model.** The ladder's linear pipeline (StandardScaler, class-balanced LogisticRegression, C by `_select_c`), fit on the probe samples of `e4_<ckpt>_goal_mirror_s1`.
- **Threshold τ.** The 98th percentile of the model's scores on all SELF calls with call_idx ≥ 4 in the *held-out* run `e4_<ckpt>_goal_mirror_s2`, so the nominal online false-positive rate is 2%.
- **Freezing.** The fitted model, τ and their sha256 go into the logbook before S0. On true inputs P2r scores 0.906–0.918 balanced accuracy and AUROC 0.971–0.978 on these runs (`probe_results.csv`).
- **Replay gate (free).** On `e4_<ckpt>_goal_mirror_s2`, the replay TPR on HIJACK calls at τ must be ≥ 0.5. If it fails, the monitor condition is dropped for that checkpoint and the run proceeds with off/oracle at 1/2 each.
- **What is scored.** Every policy call with call_idx ≥ 4, in all conditions. Recovery calls are never scored.

### The recovery (fixed before collection)

**Undo** the chunk just executed: the next call executes the executed chunk reversed and negated on the six delta dimensions, with the gripper commands reversed: undo[k] = (−a_exec[7−k][0:6], a_exec[7−k][6]).

- **Why undo rather than completing the intended motion.** It is transform-agnostic and stays inside the action bounds by construction. Completing a mirror's intended motion needs 2·a_cmd on x/y, which exceeds the [−1, 1] bound on 67% (Base-Lora) and 76% (GRPO) of E4 mirror hijacks (`calls.parquet`, mirror s1).
- **How a recovery call works:**
  - it counts toward `max_calls_per_episode` (40);
  - it is never hijacked (excluded from the scheduler's `active` mask) and never scored;
  - the policy is still queried in the batched step, and its output for that slot is discarded and logged;
  - the row is labelled `RECOVERY`;
  - normal operation resumes at the next call.
- **Cost of a false positive:** undoing a self-commanded chunk costs about two calls of progress. It is measured on clean episodes (below).

### Pre-registered readouts (per checkpoint)

- **Yardstick G,** fixed now from E4's pooled mirror s1 + s2 manifests:
  - Base-Lora: probe 0.477 (n 241), clean 0.644 (n 59), so **G = 0.167**;
  - GRPO: probe 0.730 (n 403), clean 0.948 (n 97), so **G = 0.219**.
- **B_mon** = success(probe, monitor) − success(probe, off), and **B_orc** = success(probe, oracle) − success(probe, off), with episode-bootstrap CIs resampled within condition.
- **Rule M, "the external comparator closes at least half the hijack cost":** B_mon ≥ 0.5·G and CI_low(B_mon) > 0.
- **Rule O, "undo recovers the cost when detection is perfect":** B_orc ≥ 0.5·G and CI_low(B_orc) > 0.
- **Readings:**
  - M → the paper's actionable sentence;
  - O but not M → "recovery works; detection limits";
  - not O → "undo does not recover the cost". X4 is then uninformative about the comparator, and M is not interpreted.
- **Reported alongside:**
  - false-positive cost = success(clean, monitor) − success(clean, off ∪ oracle), with CI;
  - online FPR and TPR at τ, in the off condition;
  - undo accuracy: ‖eef pos after the undo − eef pos before the hijacked call‖, as a median in cm.
- **Sanity flag:** success(probe, off) must lie within ±0.10 of E4's pooled probe success (0.477 / 0.730). Otherwise "off arm does not reproduce E4", reported with the result.

### X4 stages and gates

- **S0, smoke:** 1 task × 8 states per checkpoint, all three conditions forced to occur. **Gate:**
  - online features equal an offline recomputation from the logged rows within 1e-6;
  - recovery rows are present and never hijacked;
  - median undo error ≤ 2 cm.

  Fail: stop and report.
- **S2:** GRPO first (larger G, so better power), then Base-Lora. One collection per seed.

## 4. Code changes required (in `experiments/efference_probe/`; nothing under `rlinf/`)

All of them are behind config flags that default off, so every earlier config collects exactly as before. Offline tests come first; they cover the flags-off path, so a regression shows up as a failing earlier test.

1. **Harness, extra observables:** read `env.current_raw_obs` after `chunk_step` and log joint and object fields plus the eef consistency check (`capture.extra_observables`).
2. **Harness, monitor and recovery:** the per-episode condition RNG, online scoring, the undo chunk, `RECOVERY` rows and the scheduler mask (`monitor:` config section).
3. **`x4_monitor.py`:** fit the monitor, set τ, run the replay gate, and write the frozen artifact with its sha256.
4. **Plan and driver:** `x1_plan.yaml` and `x4_plan.yaml`, reusing `e4_wave.py` / `modal_e4.py` (generalised only as far as a plan path and checkpoint keys).
5. **Analysis and reporting:** `x1_report.py` (X1-R1 to R7, the new controls) and `x4_report.py` (B_mon, B_orc, the readings).

## 5. Budget and order

**Prices:** L40S at $1.95/h; E4 collection times from the ledger (1.9–3.0 s per episode on goal mirror/swap, plus ~1 min startup); all figures [estimate].

| item | episodes | GPU time | cost |
|---|---|---|---|
| X1 S0 + S1 | 2 smokes + 2 × 80 | ~15 min | ~$0.5 |
| X1 S2 | 200 + ≤500 + 150 + ≤250 + 150 + ≤250 | ~60 min | ~$2.0 |
| X4 S0 | 2 × 8 | ~5 min | ~$0.2 |
| X4 S2 | 8 × 250 | ~100 min | ~$3.3 |
| **total** | | **~3 h** | **~$6** |

- **Hard caps:** X1 1.6 GPU-h, X4 2.0 GPU-h, and $8 combined on the spend journal. At a cap the run stops and is reported.
- **Order:** the X1 code and S0, then X1 S1–S2, then the X4 code and S0, then X4 S2.
- **Separate approval:** the user approves X1 and X4 separately. Either can run without the other.
- **Probes:** X1's linear probe rounds run on this container by default ($0).
- **Credit:** ~$12 of the new workspace's $30 [TO VERIFY on the Modal dashboard before any launch].

## 6. [TO VERIFY] before any launch

- The GRPO-goal checkpoint's base, revision and shard hashes.
- LIBERO's object-observable key list, at S0.
- The Modal balance.
- `e4_wave.py` / `modal_e4.py` are reused with only a plan-path and checkpoint-key generalisation. If a larger change turns out to be needed, it is written up as an amendment here before collection.

## Amendments

### Amendment 1 (2026-10-29, before any X1 or X4 data)

Written after an independent review of the §4 code and before any X1 or X4 collection has started. No result informs it. Items 1–4 change or extend what is written above; items 5–6 bring the code in line with the existing wording and record how it is applied.

1. **Object keys (§2 "New observables", §6).** The object list also leaves out keys containing `_to_robot0` (LIBERO's `<obj>_to_robot0_eef_pos` / `_quat`, the object relative to the gripper). They are functions of the end-effector and object poses, so keeping them would put robot state into C_scene and blur its contrast with C_state. The S0 manifest's `object_keys` is logged as the §6 record.
2. **Driver reuse (§4 item 4, §6).** The reuse went beyond a plan path and checkpoint keys:
   - plan fields `wave`, `ckpt_order`, `prereg_markers`, `s2_requires` and `collect_overrides` (plan-wide and per checkpoint), and stages a plan lacks expand to no runs;
   - the S0 gate adds the §2 and §3 smoke checks (X1: 7 joints, ≥ 1 object pose, eef within 1e-6, vision in every archive; X4: online features within 1e-6 of a recomputation, recovery rows present and never hijacked, all three conditions present, median undo error ≤ 2 cm);
   - `modal_e4.py stage --wave` routes a call to the wave's plan and data root, and keeps the spend journal of item 4.

   No E4 gate rule changes; E4's plan expands and resolves as before (its tests pass unchanged).
3. **X4 smoke at 10 envs (§3 "X4 stages and gates").** S0 runs 10 envs, the count S2 locks, instead of `main02_mirror.yaml`'s 20. At 20 the smoke would pad 12 envs, peak near E4's ~79 GB RSS against the 96 GiB container, and not exercise S2's batch shape.
4. **Spend journal (§1, §5).** The E4 orchestrator's budget gate covers only E4's plan, so X1 and X4 get their own gate in `stage`:
   - one journal on the Volume, shared by X1 and X4 (session `x_waves`), recording each call's start and end;
   - an X call is refused without `--budget-usd`, the combined cap, and is started only if committed spend plus the call's worst case fits under it;
   - a collection's worst case is the watchdog × its expected wall clock (scaled to 50 states for a sized arm) plus the hang grace and a 300 s start-up. A stage command reserves that for each of its runs, plus a pilot probe round per S1b run, and a CPU command reserves 1800 s;
   - spend is priced at the container's request limits: $3.28/h for a GPU container and $1.52/h for a CPU one. That is above the ~$2.8/h actually billed (logbook, "Pricing correction"), so the gate is conservative.

   The cap value is the operator's to set at launch (§5 registers $8).
5. **Code brought in line with the wording above.**
   - **X1 smoke sampled:** X1 S0 samples at T = 1.6, as §1 says of every collection. The first plan had it greedy, copying E4's smoke.
   - **X4 order:** "GRPO first, then Base-Lora" means all four GRPO seeds before any Base-Lora seed, so a budget stop leaves GRPO's 1000 episodes whole. The runbook issues them in that order.
   - **Undo accuracy and the S0 undo gate:** computed only on undos whose preceding call was HIJACK, per "before the hijacked call". A false-positive undo has no hijacked call to return to; its cost is the false-positive cost.
   - **X1-R3 completeness:** "in every X1 run" needs all six planned runs, otherwise "unresolved (k of 6 X1 runs)".
   - **main02 reproduction check:** implemented as §2 states (P1, P3, P4 and P2r each within 0.04).
6. **How X1-R7 is marked, and what is added to the reports (descriptive; no rule changes).**
   - **"Replicates":** each X0 rule is applied to the X1 run with unchanged code. The result is marked "replicates" iff it equals E4's outcome for the same role (SFT or GRPO), as recorded in `x1_report.E4_REFERENCE`.
   - **F4:** run on the X1 swap pair, through a `pairs` argument to the unchanged F4 code.
   - **Hijack counts:** X4 reports hijacks per probe episode by condition.
   - **Labels:** X1 runs are labelled goal-traj1 and GRPO-goal, not with E4's names.

### Amendment 2 (2026-10-30, after X4 S0, before any X4 S2 data)

**What happened.** X4 S0 passed for Base-Lora: median undo error 1.60 cm over 6 hijacked undos, and every check true. GRPO failed only `undo_error_median_at_most_2cm`:
- 4 hijacked undos, with errors of 1.47, 1.71, 2.75 and 2.76 cm (median 2.23 cm);
- every other check (recovery rows, never hijacked, all conditions, online scores within 1e-6) passed.

As registered, X4 stopped and the failure was reported (logbook).

**Diagnosis.** The open-loop undo returns 71–84% of the hijack's displacement (GRPO: 5.8–10.9 cm displaced, 1.5–2.8 cm left). The residual grows with the displacement, so the absolute 2 cm bound binds on GRPO's larger motions. A sign or indexing error would leave about twice the displacement, so this is not an implementation error.

**Decision (operator).** Continue with the registered design and record the GRPO S0 undo failure as a deviation.
- **Waiver:** `x4_plan.yaml` waives exactly that check (`gate_waivers`: S0, GRPO, `undo_error_median_at_most_2cm`). The waiver applies only once this amendment's logbook entry ("X4 AMENDMENT 2") exists. The gate file keeps its FAILED verdict.
- **Unchanged:** the recovery (undo), the conditions and their RNG, the frozen monitors, G, Rules M and O, the readings, the sanity flag and the S2 sizes.
- **Reporting:** undo accuracy is reported per checkpoint from the S2 data (median, as registered). Every GRPO X4 result is reported with this deviation stated: the GRPO S0 undo accuracy missed the registered 2 cm bound. Rule O remains the outcome test of whether the undo recovers the hijack cost.

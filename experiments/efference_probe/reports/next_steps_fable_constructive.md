# Memo: the constructive result — an external efference-copy loop that makes a frozen VLA fault-tolerant

*Advisory memo by Fable, 2026-10-30, from a read-only pass over the repo; follows `next_steps_fable_x1x4.md`. Not a pre-registered verdict; nothing here changes an E4, X0, X1, X4 or EC0 call. Every number is quoted from a named file. `[estimate]`, `[inference]`, `[TO VERIFY]` as before; `[quick check]` marks a number I computed in this pass from the named `calls.parquet` / `manifest.json` with pandas and scikit-learn — a diagnostic, to be re-run under a logged rule before it enters the paper. Prices: L40S ≈ $2.8/h billed (logbook "Pricing correction"), ≈ $0.6 per 250-episode goal collection plus ≈ $0.1 start-up per call; H100 ≈ $4.8/h (your figure; collections are simulator-bound at 10–13 min on the L40S, so the H100 buys nothing here). Credit ≈ $3.1 `[TO VERIFY on the dashboard]`. Training and fine-tuning are excluded unless the operator lifts that; nothing recommended below needs them.*

Files read: the logbook entries "EC0 PRE-REGISTRATION" (06:50), "EC0 RESULT" (07:34) and "EC0b RESULT" (07:37); `ec0_precheck.py`; `data/x1/ec0/ec0_summary_all.json`, every `ec0_x1_*.json` and `ec0b_x1_*.json`, and `local_check/ec0/ec0_e4_*.json`, `ec0b_e4_*.json`; `harness.py` (`_run_batch`, `_policy_step`, `_to_env_action_space`); `hijack.py`; `monitor.py`; `rlinf/envs/libero/libero_env.py` by grep (read-only: `step`, `chunk_step`, `current_raw_obs`); `reports/x4_report.md`, `reports/x1_report.md`, `PREREG_X1_X4.md`; my own `next_steps_fable_x1x4.md` and `next_steps_fable_x0.md`; and for the quick checks `data/x1/x1_{sft,grpo}_swap/calls.parquet`, `data/x1/x1_grpo_mirror/calls.parquet`, `data/x4/x4_*_mirror_s*/manifest.json`.

---

## 0. Bottom line in seven sentences

1. **Recommended contribution:** a training-free *external efference-copy loop* for persistent actuation faults — the comparator compares the commanded chunk with the proprioceptive state change, identifies a constant actuator bias or gain loss (or an axis inversion) within a few chunks, and pre-compensates the next commands; it adds **zero policy calls**, which is exactly what X4's undo could not do under the 40-call cap. Measured as success and time-to-success against a no-loop arm and an oracle-inverse ceiling, on the X4 checkpoints. Call it **EC1**.
2. **Why it beats the alternatives:** it turns the paper's thesis into a system ("the policy lacks the copy; the copy, used outside it, fixes the robot"), it is the only candidate whose recovery does not cost calls, and it has a measurable dose–response. The safety-layer framing (b) adds no new truth; "replan instead of undo" (c) is what the memoryless policy already does every call; monitor generality (d) is a free supporting table, not a headline.
3. **The de-risking is already half done and favourable `[quick check]`:** a forward model fit only on the policy's own transitions predicts the chunk's translation with OOF R² 0.81–0.87 and residual sd 0.8–1.5 cm; its gain is ≈ 1 cm per command unit; a bias of 0.2 per step shows up as a 1.6 cm per-chunk shift and is identified in a median of 2 chunks (0.3: 1–2 chunks; 0.1: 13–17 chunks); and commands run well inside the action bounds (|a_x| > 0.7 on 8–11 % of steps), so bias and moderate gain faults are compensable by inversion.
4. **EC0 does not argue against this.** EC0 bounds a *one-shot, label-free* override detector (V_true TPR 0.20–0.51 at FPR 0.02; `ec0_summary_all.json`), and the bound comes from the heavy tail of the policy's own transitions (|z| > 3 on 0.6–1.2 % of SELF chunks `[quick check]`), not from missing information — the supervised monitor reaches 0.75 on the same inputs. A persistent fault is a *sequential* problem: the signal-to-noise grows with √k and the tail stops mattering. Your advice not to fund the fine-tune holds for a detector; the memo says where it does not hold (§5).
5. **Cost:** code 2–3 days `[estimate]`; first paid step a $0.5–0.9 magnitude pilot; then ≈ $0.75 per fault × checkpoint collection (300 episodes, three arms). **≤ $3 buys GRPO-130 × {bias, gain} with the ceiling arm (≈ $2.0–2.3); $10–20 buys both checkpoints, three faults, two magnitudes, a second seed and a fault-onset variant.** Section 3.
6. **Framing:** *Give the VLA an efference copy — from outside.* Negative result (no self-attribution, SFT or RL) → mechanism (command and outcome never share a forward pass; states hold the current state) → consequence (the copy used externally detects overrides, X4) → construction (the copy used externally identifies and cancels actuation faults, EC1). Section 4.
7. **First step, $0, this week:** freeze the per-checkpoint forward models from X4's clean and off-arm SELF transitions; calibrate the loop's trigger by replay on the logged clean episodes to a registered false-alarm rate; verify identification on logged command series with injected faults; write the fault module with fake-env tests. Section 6.

---

## 1. The candidates, weighed

| | (a) actuation faults + external loop | (b) monitor as safety layer | (c) a recovery that pays off under the cap | (d) monitor generality | (e2) own-failure prediction from the residual |
|---|---|---|---|---|---|
| New truth it produces | fault tolerance of a frozen VLA, with and without the copy; dose–response; identification latency | none beyond X4's TPR/FPR; "prevented failures" is undefined when every failure is a timeout (`next_steps_fable_x1x4.md` §4b) | for one-off hijacks: the policy re-plans from the next image anyway (off arm); undo already saves 1.9 calls `[quick check]`; shorter chunks cost calls | per-transform / per-checkpoint TPR table (F9) | whether high command–outcome residual on the policy's *own* chunks predicts episode failure |
| Costs calls? | **no** (changes commands, adds none) | pause = episode ends | yes, under the cap | — | — |
| Uses the thesis | directly: the copy the audit says is missing | partly | weakly | partly | directly, label-free |
| Risk | the VLA's visual loop may already absorb mild faults; clipping limits gain compensation | reads as repackaging | X4 already answered it | not a headline | new topic, own literature, confounded by episode length |
| Cost | $0.5 pilot + $0.75 per cell | $0 | ≥ $0.7 per variant | $0 | $0 |
| Verdict | **headline** | one paragraph of the X4 section | drop | supporting table | optional free afternoon; §6 |

(a) wins because it is the only option where the efference copy *does* something a reviewer can see on a success curve, and because its recovery is free in the currency X4 showed matters (calls). (e2) is the only other genuinely new result on the table and is free; it is a different paper's headline.

---

## 2. EC1 — exact design

### 2.1 Faults (applied to the executed chunk, in env action space, on every call of a faulted episode from call 0)

All three are standard actuator-fault models; each is a pure function of the command, implemented in a new `faults.py`, dispatched by a new `fault:` config section in the harness (next to `monitor:`), with `hijack.enabled: false`. No existing transform or gate changes.

| fault | executed translation dims | parameters (registered ladder) | compensable by inversion? |
|---|---|---|---|
| **bias** | a_exec = clip(a_cmd + b, −1, 1) | ‖b‖ ∈ {0.15, 0.30} per step along a per-episode horizontal direction drawn from the fault RNG stream | yes, within bounds for ≥ 90 % of steps at 0.30 (|a_x| > 0.7 on 7.7 % / 11.4 % of SFT / GRPO steps `[quick check]`) |
| **gain loss** | a_exec = g · a_cmd | g ∈ {0.6, 0.4} | partly: inversion needs a_cmd / g; clipped on steps with |a| > g (90th percentile of |a_x| is 0.64 / 0.73 `[quick check]`); the oracle arm measures the clipping loss |
| **axis inversion** | a_exec[x] = −a_cmd[x] | none (the persistent one-axis mirror) | yes, exactly |

Gripper, rotation and z are untouched. Gripper slip and latency are *not* in the ladder: neither is compensable by command scaling, so they would be negative controls for the loop, worth a sentence and not $0.75 each in this budget.

Scale, so the magnitudes mean something: in the logged data one command unit of chunk-summed x moves the end-effector ≈ 1.0 cm (ridge slope 0.98–1.03 cm per unit `[quick check]`), per-step |a| has median 0.17–0.42 and a chunk displaces 2.7–5.5 cm (medians, `next_steps_fable_x1x4.md`). A bias of 0.30 per step is therefore a 2.4 cm per-chunk drift, comparable to the chunk's own motion; gain 0.4 makes the policy's chunks 60 % too short.

### 2.2 The loop (training-free, outside the VLA, uses the command and the proprioceptive state)

- **Forward model F** (per checkpoint, frozen before collection with a sha256 like the X4 monitor): ridge [s(c), a_sent(c) flattened, task one-hot] → Δs(c), EC0's design, fit on the policy's **own clean transitions already on disk** — for GRPO-130 and Base-Lora the X4 off-arm and clean-episode SELF calls (5,387 / 5,684 SELF calls in the off arms alone, `x4_report.md`). No label, no fault data in the fit.
- **Identifier** (translation dims only; the rotation and gripper dims add noise, EC0b): per episode, per dim d, recursive least squares on r_d(c) = Δs_d(c) − F_d(·) = k_d·[(g_d − 1)·Σ_t a_sent,d(c,t) + 8·b_d] with k_d the model's slope; regressors are known, two parameters per dim.
- **Trigger** (so the loop does not act on noise): a CUSUM on the standardised translation residual, with its threshold **calibrated by replay on logged clean episodes** (X4 clean + off-arm episodes, X1 clean episodes) to a registered per-episode false-alarm rate of 5 %. The naive independence threshold gives 6–19 % per clean episode `[quick check]` because residuals are autocorrelated within an episode, so the threshold must be empirical, not analytic.
- **Compensation**, from the call after the trigger: a_sent = clip((a_cmd − b̂)/ĝ, −1, 1) on translation; identification continues on a_sent. Everything is logged: `a_cmd_env` (policy), `a_sent` (after compensation), `a_exec` (after fault), the estimates, the trigger call.
- **Oracle arm:** a_sent = clip((a_cmd − b)/g, −1, 1) with the true fault — the ceiling, and the measure of the clipping loss.

### 2.3 Conditions and sizes (one collection per fault × magnitude × checkpoint)

`main02.yaml` (all ten goal tasks) × 30 initial states = 300 episodes, ≈ 13–15 min `[estimate from 12.1–13.6 min for 200–300 episodes in X1]` → **≈ $0.65 + $0.1 start-up ≈ $0.75**. Faulted episodes 80 % (as `probe_episode_fraction`), conditions per episode from a dedicated RNG stream as X4 did: **off 0.4 / loop 0.4 / oracle 0.2** → n ≈ 96 / 96 / 48. Clean episodes 20 % with the loop **on** for half → n ≈ 30 / 30 for the false-compensation cost.

### 2.4 Pre-registered readouts and rules (per fault × magnitude × checkpoint)

- **G_f** = success(clean) − success(fault, off), clean = the run's own clean episodes (both loop states pooled unless Rule FP below fails, then clean-off only); episode bootstrap B = 2000, seed 0.
- **B_loop** = success(fault, loop) − success(fault, off); **B_orc** likewise for the oracle.
- **Rule L ("the loop restores at least half the fault's cost"):** B_loop ≥ 0.5·G_f and CI_low(B_loop) > 0.
- **Rule C ("inversion can restore it"):** B_orc ≥ 0.5·G_f and CI_low > 0. Not C → "not compensable by command inversion within bounds; uninformative about identification" (X4's logic, kept).
- **Rule FP:** success(clean, loop) − success(clean, off) with CI; the loop is "safe on a healthy robot" iff the point is ≥ −0.05.
- **Uninformative-fault rule:** G_f < 0.15 → "the policy's visual loop absorbs this fault"; reported as a finding about the VLA, the loop rules are not interpreted.
- **Alongside, descriptive:** RMST(40) for every contrast (the censoring-correct metric from F8); identification latency (trigger call, median and 90th percentile); terminal |ĝ − g|, ‖b̂ − b‖; the clipped fraction of compensated steps.
- **Readings:** L and C → the paper's constructive sentence; C but not L → "compensable; identification limits" (report latency and estimate error); not C → "beyond command inversion".

### 2.5 Power at the planned n, from measured quantities `[estimate]`

Clean success: GRPO-130 0.968 over all X4 clean episodes, Base-Lora 0.634 `[quick check]`; the fault magnitudes are chosen in the pilot so that success(fault, off) lands in 0.3–0.6 (§3). Two-sided α = 0.05; power is for CI_low(B) > 0, with Bernoulli variance at the stated rates.

| checkpoint | success off → loop | B | n per arm | SE | power |
|---|---|---|---|---|---|
| GRPO-130 | 0.55 → 0.85 | 0.30 | 96 / 96 | 0.062 | > 0.99 |
| GRPO-130 | 0.55 → 0.75 | 0.20 | 96 / 96 | 0.067 | 0.85 |
| GRPO-130, oracle | 0.55 → 0.85 | 0.30 | 96 / 48 | 0.072 | 0.99 |
| Base-Lora | 0.35 → 0.55 | 0.20 | 96 / 96 | 0.070 | 0.81 |
| Base-Lora | 0.35 → 0.50 | 0.15 | 96 / 96 | 0.071 | 0.56 |

The second half of Rule L (point ≥ 0.5·G_f) costs little power when the true restoration is ≥ 0.75·G_f: with G_f = 0.4 and SE 0.065, P(B̂ ≥ 0.20 ∣ B = 0.30) = 0.94. For reference, X4's B_orc CI half-width was 0.067 at n = 249 / 291 (`x4_report.md`); the design above has half-widths of 0.12–0.14, which is why the fault must be sized to cost ≥ 0.3 and why Base-Lora, if run, should get two seeds or a two-arm design (n = 120 / 120 → power 0.90 at B = 0.20).

### 2.6 What each outcome lets the paper claim

- **L and C on both faults, one checkpoint ($3 tier):** "A frozen VLA loses G_f of its success under a constant actuator bias or gain loss and does not recover it through vision; an external efference-copy loop that needs only the commanded chunk and the end-effector pose identifies the fault within k chunks and restores ≥ half of the loss with no additional policy calls." Plus the latency and estimate-error numbers.
- **L and C on both checkpoints, three faults, two magnitudes ($10–20 tier):** the same, as a dose–response figure with an SFT and an RL checkpoint, and the inversion case where the VLA alone collapses.
- **C but not L:** "compensable; our identifier is too slow / too noisy at this magnitude" — still a positive ceiling result and an honest systems datum.
- **Not C (expected only for gain 0.4):** "beyond command inversion; the compensable range is bounded by the action limits" — a quantified boundary, stated next to the headroom numbers.
- **G_f < 0.15 on bias:** "the VLA's visual servoing absorbs a constant bias of this size" — a finding about the policy's closed loop that replaces the retired "novelty-gated compensation" with a measured one; the inversion arm then carries the constructive result.

---

## 3. Options by budget

### $0 — the de-risking (this week; §6 has the order)

1. Fit and freeze F for GRPO-130 and Base-Lora from X4 SELF transitions; report OOF R² per translation dim and residual sd (expected 0.8–1.5 cm from the X1 checkpoints `[quick check]`).
2. Replay-calibrate the CUSUM threshold on all logged clean episodes of both checkpoints to 5 % per-episode false alarm; report the achieved rate on held-out clean episodes.
3. Identification on logged command series with injected faults (a' = g·a + b applied to the logged `a_cmd_env`, real residual noise added): chunks-to-trigger and estimate error vs magnitude — the table that fixes the pilot ladder.
4. `faults.py` + harness `fault:` section + `ec1_plan.yaml` + `ec1_report.py`, fake-env tests (flags-off rows identical, as the X1/X4 change was tested: logbook "X1 / X4 code ready").
5. Optional, free, one afternoon: (e2) — do EC0's OOF SELF residual scores in the first k calls predict episode failure (episode-grouped AUROC, k fixed in advance)? If AUROC ≥ 0.70 on both X1 checkpoints it is a paragraph; otherwise nothing.

### ≤ $3 — with the credit in hand

| step | what | cost `[estimate]` |
|---|---|---|
| pilot | GRPO-130, 2 tasks × 8 states, off arm only, settings {bias 0.15, 0.30; gain 0.6, 0.4; inversion}; rule: a magnitude enters the main run iff its pilot success ≤ 0.6 | $0.5 if the driver chains the five settings in one container call `[TO VERIFY]`, else ≈ $0.9 |
| main 1 | GRPO-130, bias at the chosen magnitude, 300 episodes, three arms | $0.75 |
| main 2 | GRPO-130, gain at the chosen magnitude | $0.75 |
| total | | **$2.0–2.4**, reserve $0.7–1.1 |

Inversion on GRPO-130 ($0.75) fits only if the pilot is chained and nothing fails; otherwise it is the first item of the next tier.

### ≈ $10–20 — if the operator tops up

Pilot ($0.9) + GRPO-130 × {bias, gain, inversion} × 2 magnitudes (inversion has one) = 5 runs ($3.75) + Base-Lora × 3 faults at one magnitude each (3 runs, $2.25; two-arm design for power) + a second seed of the GRPO-130 primary cells (3 runs, $2.25) + one fault-onset variant (fault starts at call 6, GRPO-130 bias; measures detection latency with a clean reference inside the episode; $0.75) = **≈ $10 of collections; $12–14 with reserve.** Buys: dose–response curves on an SFT and an RL checkpoint, a replication, the inversion extreme, and the onset latency.

Wall-clock: code 2–3 days; each collection 13–15 min; readouts minutes; the whole $10 tier is an afternoon of GPU once the code is in.

---

## 4. The paper framing

**Title-level thesis.** *Give the VLA an efference copy — from outside.* A single-image VLA cannot tell what it just did, because the command and its outcome never share a forward pass; RL on its own consequences does not change that; and the copy it lacks is cheap to use outside it — to detect overrides, and to identify and cancel actuation faults the policy's own visual loop cannot.

**Claims (five), each with its evidence.**

1. **No readable self-attribution, before or after RL.** P3 − P1 within ±0.04 in 36/36 run×family rows; 17/18 linear bootstrap CIs cover 0; fixed cell +0.035 [−0.004, +0.074] (X1), +0.050 / +0.064 (E4, post hoc), state-explained. — `a1_efference_increment.csv`, `a7_linear_selected.csv`, `x1_report.md`, `fixed_cell.json`.
2. **Why: the states hold the current state and nothing about authorship.** L0 R² 0.84–0.97; F2 all-call gain ≤ 0.016; every above-floor rung ≤ C_obs + δ once objects and joints are logged. — `a2_*.csv`, `f2_*.json`, `x1_report.md` R2/R5.
3. **The copy, used externally, detects overrides online** at TPR 0.75 / 0.73 for FPR 0.02 and transfers across checkpoints (AUROC 0.96–0.98); one-off perturbations are then a recovery-time problem, not a detection problem (every X4 failure is a timeout; the undo saves ≈ 2 calls). — `x4_report.md`, `x4/monitor_*.json`, F8.
4. **[EC1] The copy, used externally, makes the frozen policy fault-tolerant:** under persistent actuator bias / gain loss / inversion the policy loses G_f; the loop identifies the fault in k chunks and restores ≥ half of G_f with no extra calls, on an SFT and an RL checkpoint; the compensable range is bounded by the action limits. — `ec1_report.md` (to be).
5. **A one-shot label-free detector is bounded** (EC0: TPR ≤ 0.51 at FPR 0.02 even with true states; ≤ 0.16 from the VLA's features), **which is why the right use of a self-supervised forward model is sequential identification, not anomaly detection.** — `ec0_summary_all.json`, `ec0b_*.json`.

Claim 5 is what turns EC0 from a dead end into the reason for EC1's design, and it answers in advance the reviewer who asks why the loop is not "just an anomaly detector on the forward model".

What the draft stops defending: novelty-gated compensation as a mechanism (its measured successor is the off arm's dose–response: how much fault the visual loop absorbs); "the comparison is cheap, the policy never buys it" (Contribution 3); any claim that RL buys monitoring.

---

## 5. Risks, and the cheapest de-risking step for each

| risk | evidence now | de-risk | cost |
|---|---|---|---|
| The VLA's visual loop absorbs mild bias, G_f small | unknown; the policy re-plans every 8 steps from the image | pilot ladder with the ≤ 0.6 gate; inversion as the guaranteed-large case | $0.5–0.9 |
| Gain compensation clipped by the bounds | |a_x| > 0.7 on 8–11 % of steps; 90th pct 0.64–0.73 `[quick check]` | g ≥ 0.6 in the ladder; the oracle arm measures the loss; report the clipped fraction | $0 |
| Identification too slow / too noisy | bias 0.2: median 2 chunks; 0.1: 13–17 `[quick check]` | magnitudes ≥ 0.15; latency is a reported quantity, not a hidden assumption | $0 |
| False compensation on a healthy robot | naive threshold → 6–19 % per clean episode `[quick check]` | empirical CUSUM calibration on logged clean episodes; Rule FP in-run | $0 + 30 clean-loop episodes per run |
| Forward model mis-specified under the fault (the fault changes the state distribution) | X1 F4: matched-support readout gaps are small | identifier uses the model's slope only, and a_sent is known; oracle arm bounds it | $0 |
| "Adaptive control 101 around a VLA" | true | the contribution is the measurement and the chain: audit → missing copy → external loop → restored success; cite MRAC / fault-tolerant control; keep the off arm's dose–response as a result about the VLA | writing |
| Reviewer: faults are synthetic, sim-only, perfect proprio | yes | scope sentence: bias / gain / inversion are the textbook actuator-fault models; the loop uses the command and the end-effector pose, both available on a real arm without a camera | writing |
| Budget: $3.1 covers one checkpoint × two faults | — | Base-Lora, replication and the ladder need the top-up; say so before the pilot | — |

**The single cheapest first step** is item 2 of §3's $0 list: calibrate the trigger by replay on the ≈ 510 logged clean episodes that never saw an undo (X4 off ∪ oracle clean episodes: 150 GRPO + 108 Base-Lora, the `fp_cost` controls in `x4_report.md`; X1: 253 clean across the six runs `[quick check]`; X4's 72 + 56 monitor-condition clean episodes are excluded because a false positive could have fired an undo in them) and report the per-episode false-alarm rate and the chunks-to-trigger on injected faults. If a 5 % false-alarm threshold needs more than 4 chunks to trigger on a 0.2 bias, the ladder moves to 0.3 before any GPU is spent.

---

## 6. Order of work

1. `[$0, day 1]` Freeze F per checkpoint from X4 SELF transitions; CUSUM calibration by replay; injected-fault identification table. Logbook entry with the thresholds and their sha256.
2. `[$0, days 1–3]` `faults.py`, harness `fault:` section, plan, report, fake-env tests; EC1 pre-registration appended to a new `PREREG_EC1.md` (rules of §2.4, the ladder, the pilot gate, the sizes, the budget cap), committed before any collection.
3. `[$0.5–0.9]` Pilot; log the chosen magnitudes.
4. `[$1.5]` GRPO-130 bias and gain mains; readout; logbook.
5. `[top-up]` Inversion, Base-Lora, second seed, ladder, onset variant, in that order, each cut-able from the tail.
6. `[$0, in parallel]` F8 (survival metric) and F9 (monitor generality) from the previous memo, because EC1's tables use RMST and the paper's claim 3 uses F9.

---

## 7. Statements in the brief that the files do not support, or support only in part

1. **"V_feat TPR 0.03–0.13, AUROC 0.56–0.73."** X1: TPR 0.026–0.128, AUROC 0.555–0.728; the E4 swap runs go to TPR 0.156 and AUROC 0.711 (`ec0_summary_all.json`). Write 0.03–0.16.
2. **"A translation-only score gets 0.50 / 0.62 on mirror only."** It also lifts GRPO freeze (0.462 vs V_true 0.419, `ec0b_x1_grpo_freeze.json`); it drops on swap and on SFT freeze. The point stands: the omnibus 8-dim residual is the wrong statistic for direction-flipping faults, which is itself evidence that the gap is partly the statistic, not only noise.
3. **"Weak at a strict FPR because the policy's own transitions are noisy."** Half right. The SELF residual on translation has sd 0.8–1.5 cm against hijack displacements of several cm, but its tail is heavy (98th percentile 2.5–2.7 sd; |z| > 3 on 0.6–1.2 % of chunks `[quick check]`), and a 2 % FPR threshold sits in that tail. The supervised monitor reaches 0.75 on the *same* two inputs, so the information is there; what the label-free one-shot score lacks is the shape of the override. For a persistent fault the sequential statistic averages the tail away (median 1–2 chunks to detect a 0.2–0.3 bias). So: EC0 bounds one-shot label-free detection; it does not bound label-free *identification*.
4. **"So I advised the operator NOT to fund the efference-token + self-supervised-forward-loss fine-tune."** Agreed for its detection purpose, and it is excluded anyway. Two caveats for the record: (i) a fine-tune with the command as *input* (X6) was never a detector proposal — its purpose was to test whether the channel changes the policy's behaviour, which EC0 says nothing about; (ii) the "$5–7" estimate `[TO VERIFY]` is far below the $25–60 the X0 memo carried for X6; if the cheaper figure is for a small LoRA on a few thousand transitions, say so when the ask is written, because reviewers will ask why it was not run.
5. **"X4: detection works, open-loop undo fails Rule O."** Correct, and incomplete in the same way as before: every X4 failure is a 40-call truncation and the undo saves 1.9 [0.04, 3.7] calls on GRPO `[quick check]`. The constructive result in §2 is designed around that fact; the paper should state it next to Rule O.
6. **"Operator's verdict: not enough for RSS."** Not checkable from files; the memo takes it as given.

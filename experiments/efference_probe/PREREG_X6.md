# X6 pre-registration — a cross-architecture fault benchmark, and an efference-copy input that should close the gap it shows

**Append-only.** Written 2026-10-30, after the X6 stage-0 pilot (n = 16 per setting) and before any benchmark run at n = 300, any training collection, or any fine-tune. The git commit that adds this file is the timestamp. Later changes are dated amendments at the bottom, never edits above them.

**Approval.** Writing this file approves no spend and lifts no rule. Part B trains models, which the E4 rules forbid until the user lifts "no training" for X6-EC explicitly. Every stage in §5 runs only after the user approves its tier, its platform and its cap.

## Where this comes from

- **The plan:** `reports/next_steps_fable_reframe.md` (thesis, §2.3, §3), with the data and training design of `reports/next_steps_fable_positive.md` §3.2–3.5.
- **The thesis under test:** *policies that cannot know what they did.* A VLA that sees only the current image (and at most the current state) carries no record of the command it sent, so it cannot tell when its actuation changes. The lens makes two predictions:
  - **Part A (the control):** every such policy fails persistent actuation faults it cannot see. This holds whatever its training: SFT, RL, a proprio input, or world-model pretraining.
  - **Part B (the design guidance):** a short command–outcome history, given through the model's proprio path and trained under randomised faults, closes the gap. A forward-model loss without that input does not.
- **What stage 0 measured** (logbook, "X6 stage 0" entries):
  - **Identifiability** (`data/x6/identify.json`): K = 2 chunks of the frozen policy's own history identify a 0.30 bias (median z 4.28) and an axis inversion (z 17.5).
  - **Rule P pilot** (frozen GRPO-130, tasks {2, 8} × 8 states, n = 16 per setting): success under each setting.

    | setting | success |
    |---|---|
    | clean | 0.938 |
    | bias_x 0.15 | 0.562 |
    | bias_x 0.30 | 0.000 |
    | invert_x | 0.000 |
    | gain 0.6 | 0.875 |
    | gain 0.4 | 0.562 |

    Every failure was a timeout. Under bias the policy reacts (mean commanded x −0.232) but never cancels the fault.
  - **Loading smoke:** the `official` and `rlinf` OFT classes agree on 95.7% of greedy tokens, so every Part B arm runs through the `rlinf` class (`ec_policy.py`).
- **Model checks** (`reports/x6_model_checks.md`): every LIBERO policy within reach is memoryless across calls. DreamZero is marked `[inference]` until its smoke logs `current_start_frame`. No WoVR-trained OFT checkpoint is released.

## 1. Shared conventions

- **Task suite:** `libero_goal`, all 10 tasks, 320 simulator steps per episode for every architecture (OFT 40 calls × 8; π0 64 calls × 5).
- **Faults** (`faults.py`):
  - **Where:** applied to the executed env action after the policy's command, so they are independent of the action head.
  - **Clipping:** dims 0–5 are clipped to ±1.
  - **Gripper:** never touched.
  - **One fault per episode:** each fault lasts the whole episode.
- **Reset-state split:**
  - **Evaluation:** states 0–29 of each task. This covers every Part A run and every Part B evaluation.
  - **Training only:** states 30–49 of each task are the Part B training collection.
  - **No overlap:** no evaluation episode starts from a training state.
- **Pairing:** every setting of a run, and every arm of Part B, runs the same reset states with the same env seed (3). Differences are bootstrapped over (task, reset state) pairs.
- **Bootstrap:** B = 2000, seed 0, 95% percentile CI.
- **Decoding:** as in each architecture's repo evaluation. OFT is sampled at T = 1.6, as in every earlier wave; π0 uses `mode: eval`.
- **Metrics** (`x6_fault_report.py`):
  - success;
  - RMST in simulator steps at the 320-step budget (a failure counts 320);
  - paired drop against clean;
  - reaction trace: the mean commanded x per call.
- **Budget rule:** each stage has a cap (§5). A run at 2× its expected wall-clock is stopped and reported.

## 2. Part A — cross-architecture fault benchmark (the control)

### Models

| tier | model | checkpoint | clean-success caveat |
|---|---|---|---|
| minimal | OpenVLA-OFT GRPO-130 (= Part B arm F) | `RLinf/RLinf-OpenVLAOFT-LIBERO-130` | pilot 0.938 |
| minimal | π0 LIBERO SFT (current-state input) | `RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT` | `[TO VERIFY]` on goal |
| solid | GR00T N1.5 Goal | `RLinf/RLinf-Gr00t-SFT-Goal` | documented 48.2% |
| solid | DreamZero 5B (world-model policy) | `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers` | its smoke must confirm the cache resets every call |
| solid | OpenVLA-OFT Base-Lora | `RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora` | — |

### Protocol

- **Configs:** `configs/x6_fault_oft.yaml` and `configs/x6_fault_pi0.yaml`; later models get configs of the same shape.
- **Sweep:** `[clean, bias_x:0.15, bias_x:0.3, invert_x, gain:0.6, gain:0.4]`.
- **Size:** 10 tasks × 30 states = 300 episodes per setting, 1,800 per model.
- **Capture:** hidden-state capture, hijacks and frames are off.

### Rule M (the categorical prediction, per model)

- **Scope:** applies to every model whose clean success is ≥ 0.40.
- **Test:** for both rule settings, bias_x 0.30 and invert_x:
  - the point estimate satisfies success(setting) ≤ 0.5 × success(clean), **and**
  - the paired drop has CI_low > 0.
- **Verdict:** the model "fails as predicted" iff both settings pass.
- **Cross-architecture claim:** "memoryless VLAs fail persistent bias and inversion" holds iff Rule M passes for every model that was run.
- **A model that passes a rule setting is a counterexample.** It is reported as such.
  - If the setting is inversion, the lens is wrong about what the image carries, and the paper says so.
- **Below 0.40 clean:** a model is reported descriptively, without the rule.

### Descriptive, no rule

- **Other settings:** bias 0.15 and gain 0.6 / 0.4, where the lens expects partial absorption. The pilot absorbed gain 0.6.
- **Every setting:** the reaction trace and RMST.

## 3. Part B — X6-EC: the efference-copy input, and the 2 × 2 that isolates it

### 3.1 Class and arms

- **Class:** every arm runs through the `rlinf` OFT class with checkpoint GRPO-130.
  - **F** is the plain class, frozen.
  - **A–E** are made by `ec_policy.adopt`. It adds one projected token after the image patches, as in the official proprio path. The projector is Linear–GELU–Linear, its output layer zero-initialised.
- **The arms:**

  | arm | token input | auxiliary loss | lens prediction (rule cells, §3.4) |
  |---|---|---|---|
  | F | none (frozen) | — | fails |
  | A | copy, 136-d | — | passes Rules E, N |
  | B | state only, 8-d | — | fails on inversion; at most hedges on bias |
  | C | copy, 136-d, history shuffled at training time | — | fails |
  | D | state only, 8-d | forward-model head | as B (Rule L) |
  | E | copy, 136-d | forward-model head | ≥ A; reported, no rule |

- **Copy input** (`ec_policy.EfferenceHistory`, K = 2), 136 values in this order:
  - **s(c):** the current state, normalised by the checkpoint's proprio q01/q99, or by the collection's own q01/q99 if the checkpoint has none (smoke (d));
  - **a_sent(c−1):** 56 values;
  - **Δs(c−1):** 8 values;
  - **a_sent(c−2):** 56 values;
  - **Δs(c−2):** 8 values.

  More precisely:
  - a_sent is what the controller sent, before any fault.
  - Δs is the measured state change over that chunk, divided by the per-dimension sd of Δs over the collection's clean episodes and clipped to ±10.
  - Both are zero-padded at episode start.
- **Shuffled copy (C):** each training sample's four history blocks are replaced by those of a random sample from a different episode. The draw is redone every epoch. The true history is used at test.
- **Forward-model head (D, E):** a 2-layer MLP (width 512, GELU).
  - **Input:** the mean of the last-layer hidden states over the 56 action positions of `default_forward`, plus a_sent(c).
  - **Target:** the scaled Δs(c), trained by MSE with weight λ = 1.0 against the mean token NLL.
  - **At evaluation:** the head is discarded.

### 3.2 Training data

- **Collector:** frozen F (`rlinf` class, sampled at T = 1.6) on states 30–49 of all 10 tasks. Six passes with env seeds 101–106 give 1,200 episodes.
- **Fault drawn per episode:**
  - **clean, 0.30.**
  - **bias, 0.45:**
    - magnitude ‖b‖ ~ U[0.10, 0.35];
    - direction θ uniform on [0°, 360°), excluding four 20°-wide sectors centred on 0°, 90°, 180° and 270°;
    - the bias is (‖b‖ cos θ, ‖b‖ sin θ) added to (x, y).

    The axis directions are therefore never trained, and are the held-out directions at test.
  - **inversion, 0.25:** of x or of y, 50/50.
- **Oracle compensation:** on in 70% of faulted episodes, off in 30%.
  - **What the oracle sends:** the inverse fault applied to the policy's command. For bias it is a_cmd − b on (x, y); for inversion it negates the inverted axis. Dims 0–5 are clipped to ±1.
  - **Episodes with compensation off:** these still log the compensated command as their target, so drifted states are seen.
- **Target per call:** the compensated command (in clean episodes, a_cmd). It is tokenised through the model's own bin mapping.
- **Logged per call:**
  - main image (JPEG q95) and task;
  - s(c), a_cmd, a_sent, a_exec, Δs;
  - the fault parameters and the compensation flag.

### 3.3 Training

- **Trained parameters:**
  - LoRA r = 32 on the LLM blocks (the repo's target list, vision targets dropped);
  - the projector, fully;
  - the forward-model head (D, E).

  Everything else is frozen.
- **Loss:** token NLL of the 56 target tokens through `default_forward` (`compute_logprobs`, T = 1.0, top_k = −1), plus λ·MSE for D and E.
- **Schedule:**
  - batch 16, 2 epochs, AdamW;
  - learning rate 1e-4 for the projector and head, 5e-5 for LoRA;
  - cosine schedule with 100 warm-up steps;
  - bf16, gradient clip 1.0.
- **Seeds:** training seed 0 for every arm. A second seed (1) for A and B if the tier's cap allows; it is reported per seed and averaged.
- **No checkpoint selection on evaluation states:** the last step is evaluated.

### 3.4 Evaluation

- **Episodes:** states 0–29, 10 tasks, 300 episodes per cell, env seed 3, T = 1.6. All arms are paired by (task, state).
- **Rule cells:**
  - clean;
  - bias_x 0.30: direction 0°, held out of training; this is Part A's `bias_x:0.3`;
  - invert_x: the fault family was trained, but not on these states.
- **Descriptive cells** (if the cap allows): −y bias at 0.20 (held-out direction); +x bias at 0.40 (beyond the trained magnitude range); invert_y; gain 0.6 (an untrained family).
- **F's numbers:** F's rule-cell numbers are Part A's OFT run. Same config, states and seed, so they are not re-run.
- **Adopted arms at test:**
  - the history is built online from what the policy sent (no compensation at test) and the measured Δs;
  - A also runs invert_x with its history blocks zeroed (descriptive input ablation).

### 3.5 Rules

Per family f ∈ {bias (cell bias_x 0.30), inversion (cell invert_x)}. G_f = success(F, clean) − success(F, f). Each difference is bootstrapped over paired (task, state).

- **Rule E (efficacy):** success(A, f) − success(F, f) ≥ 0.5·G_f, with CI_low > 0.
- **Rule N (the copy, not the fine-tune):** both of the following, each with CI_low > 0:
  - success(A, f) − success(B, f) ≥ 0.15;
  - success(A, f) − success(C, f) ≥ 0.15.
- **Rule L (the loss without the channel is inert):** |success(D, f) − success(B, f)| ≤ 0.05, with the 95% CI inside [−0.10, +0.10].
- **Rule 0 (no nominal cost):** success(A, clean) − success(F, clean) ≥ −0.05, with CI_low ≥ −0.10.
- **Readings:**

  | outcome | reading |
  |---|---|
  | E ∧ N ∧ 0 on both families | the headline claim |
  | the same on one family | the claim is scoped to that family |
  | E ∧ ¬N | fine-tuning under faults, not the copy, restores success. Reported as such; it removes the mechanism |
  | N ∧ ¬E | the copy is read, but not enough |
  | ¬0 | the fine-tune damaged the policy. Reported, and no further arms are trained |
  | L fails with D > B | the lens's warning about forward-model losses is wrong |
  | L fails with D < B | the loss hurts |

- **Power** `[estimate, next_steps_fable_reframe.md §3.1, at 240 faulted episodes; 300 paired is tighter]`:
  - Rule E > 0.99 at a 0.30 effect;
  - Rule N 0.96 at 0.15;
  - Rule L is an equivalence test with half-width ≈ 0.08.

## 4. Wrapper smoke (gate S0, before any Part B spend)

`x6_ec_smoke.py` runs on GRPO-130, one batch of real observations.

- **Pass iff all three hold:**
  - **(a) Plain class:** greedy `predict_action_batch` logprobs agree with a `default_forward` replay of the same tokens.
  - **(b) Adopted class:** the same agreement holds with a random projector output layer.
  - **Finite actions:** every action is finite in both cases.

  Agreement means a difference within 0.05 + 0.02 × max |logprob|, the capture check's tolerance.
- **Reported, no gate:**
  - (c) the zero-init token agreement with (a);
  - (d) whether the checkpoint's norm stats carry 8-d proprio bounds.
- **On failure:** stop, report, and train nothing.

## 5. Stages, gates and caps (minimal tier)

Platform: a rental (AutoDL A800 or Vast.ai A100), chosen by the user. Modal credit is ≈ $2.6, which is not enough. All costs are `[estimate]`; GPU-hours are scaled from the pilot (96 episodes in 307 s).

| stage | content | gate to pass before the next stage | GPU-h |
|---|---|---|---|
| S0 | wrapper smoke (§4) | smoke passes | ≈ 0.2 |
| S1 | Part A, OFT: 1,800 episodes (= F) | clean success ≥ 0.80. Otherwise Part B stops: F is too weak to show a gap | 1–1.5 |
| S2 | Part A, π0: 1,800 episodes | its own first-batch smoke (finite actions, clean success logged) | 1.5–3 |
| S3 | training collection: 1,200 episodes, frames on | ≥ 95% of episodes logged; compensation recovers clean-level success on compensated episodes (reported) | ≈ 1 |
| S4 | fine-tunes A–E, seed 0 | loss finite and decreasing | 4–6 |
| S5 | evaluations: 5 arms × 3 rule cells × 300 episodes | — | 2.5–3 |
| S4b/S5b | seed 1 for A and B, then descriptive cells | if the cap allows | 3–4 |

- **Minimal-tier cap:** $25 for everything above, including failures, on rental pricing (≈ $0.6–0.8 per GPU-hour).
- **Solid tier:** GR00T, DreamZero, Base-Lora and the gain family. Each is pre-registered by amendment before it runs.

## 6. Code changes required (all in `experiments/efference_probe/`; nothing under `rlinf/`)

| status | change |
|---|---|
| done | `ec_policy.py` (wrapper, projector, history), `x6_ec_smoke.py`, `faults.py`, `x6_fault_report.py`, the capture-off and multi-architecture harness path |
| to do | `faults.py`: 2-D bias (`bias:<mag>@<deg>`), `invert_y`, the oracle inverse, and a per-episode fault draw with the compensation flag |
| to do | config: a reset-state offset, so collections can use states 30–49 |
| to do | harness: `EfferenceHistory` wiring and a loader for adopted arms (projector state plus LoRA adapter); logging of a_sent, the compensation flag and the target |
| to do | new files: `ec_dataset.py` (calls plus frames → samples, shuffled-history option), `train_ec.py` (LoRA, NLL, forward-model head), `x6_ec_report.py` (Rules M, E, N, L, 0) |
| test | offline tests with the fake env and the fake OFT for each of the above |

## 7. [TO VERIFY] before any launch

- π0 runs through the harness: its first batch is part of S2.
- The checkpoint's proprio norm stats: smoke (d).
- The repo's LoRA target list for this class.
- The rental platform's image installs `requirements/install.sh embodied --model openvla-oft --env libero`, plus `openpi` for π0.

## Amendments

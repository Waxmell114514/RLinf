# Memo: the positive result under the operator's narrative — an efference-copy input for the VLA, and where the narrative holds

*Advisory memo by Fable, 2026-10-30, from a read-only pass over the repo; follows `next_steps_fable_x1x4.md` and `next_steps_fable_constructive.md`. Not a pre-registered verdict; nothing here changes an E4, X0, X1, X4 or EC0 call. Every number is quoted from a named file. `[estimate]`, `[inference]`, `[TO VERIFY]` as before; `[quick check]` marks a number I computed in this pass from the named `calls.parquet` / `manifest.json` with pandas and scikit-learn — a diagnostic, to be re-run under a logged rule before it enters the paper. Prices: L40S ≈ $2.8/h billed (logbook "Pricing correction"), ≈ $0.6 per 250-episode goal collection plus ≈ $0.1 start-up per call; H100 ≈ $4.8/h (your figure). Credit ≈ $3.1 `[TO VERIFY on the dashboard]`. **Everything in §3 needs the operator to lift the no-training rule; §1 and the $3.1 tier do not.***

Files read: `rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py` (`OpenVLAOFTRLConfig`, `prepare_inputs`, `_build_embedding`, `default_forward`, `load_proprio_projector_weights`), `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py` (`__init__`, `_build_embedding`, `predict_action_batch`), `rlinf/models/embodiment/openvla_oft/__init__.py`, `rlinf/models/embodiment/openvla_oft/openvla_utils.py` (`normalize_proprio`, the LoRA block, `norm_stats` loading), `rlinf/models/embodiment/base_policy.py` (`ForwardType`), `rlinf/models/__init__.py` (LoRA block), `examples/embodiment/config/model/openvla_oft.yaml`, `examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml`, `examples/embodiment/config/libero_spatial_opd_openvlaoft.yaml`, `examples/embodiment/config/env/libero_goal.yaml` (camera size), `examples/embodiment/train_offline_rl.py` (imports), `rlinf/algorithms/losses.py` (function list); in `experiments/efference_probe/`: `harness.py` (`build_model_config`, `_run_batch`, `_policy_step`, `_to_env_action_space`), `config.py`, `storage.py`, `hijack.py`, `monitor.py`, `ec0_precheck.py`, `data/x1/x1_grpo_swap/run_config.yaml` and `frames/ep00000/`, `data/x1/ec0/*.json`, `local_check/ec0/*.json`, the logbook entries "EC0 PRE-REGISTRATION", "EC0 RESULT", "EC0b RESULT"; `reports/x4_report.md`, `reports/x1_report.md`, `reports/e4_report.md`, `PREREG_X1_X4.md`; and for the quick checks `data/x1/x1_{sft,grpo}_swap/calls.parquet`, `data/x1/x1_grpo_mirror/calls.parquet`, `data/x4/x4_*_mirror_s*/manifest.json`. The `prismatic` package (`ProprioProjector`, `_process_proprio_features`) is not installed in this container, so its internals are `[TO VERIFY]`; the model code shows it takes `proprio_dim` as a free argument and adds exactly one token (`num_patches += 1`).

---

## 0. Bottom line in eight sentences

1. **The premise splits, and our own data decide the split.** For *transient* overrides — the hijacks we have measured — "no efference copy → fragile" is **false**: the memoryless policy re-plans from the next image, the state is fully in that image (L0 R² 0.84–0.97; F2 gain ≤ 0.016), and even perfect knowledge of the override plus an undo repays at most 38 % of the cost on GRPO and nothing on Base-Lora (X4 oracle: +0.083 [+0.017, +0.150], +0.002 [−0.093, +0.098]; every failure a 40-call timeout). For *persistent or latent* actuation faults — bias, gain loss, axis inversion — the premise is **plausible and unmeasured**: there the optimal action depends on a latent the image cannot show and only the command–outcome history can identify. The narrative must be scoped to that class or the X4 table sinks it. Section 1.
2. **The positive result that fits the operator's framing:** give the VLA its efference copy as an *input* — a short history of its own sent commands and the resulting state changes — through OFT's proprio token, fine-tune it (LoRA + fresh projector) under randomised persistent faults with oracle-compensated targets, and show that under *unseen* fault magnitudes it recovers the success the frozen policy loses, while the same fine-tune without the copy, or with a shuffled copy, does not. Call it **X6-EC**. Section 3.
3. **Training is required.** No inference-time trick makes a frozen model read an input it was never trained on (EC0's V_feat 0.03–0.16 and the previous memos' X3 verdict). The no-training rule must be lifted for §3; the $3.1 tier below needs no training and is the premise test that must come first anyway.
4. **The copy can only carry what the history identifies.** With the measured residual noise (0.98–1.15 cm per chunk) and command sizes (median chunk-sum |A_x| 1.07–1.59), a bias of 0.2 per step reaches z ≈ 2.8–3.2 after 4 chunks, an axis inversion z ≈ 8–9 after one chunk (≥ 3 in 93–94 % of episodes), a gain loss to 0.6 only z ≈ 2.7–3.0 after 4 chunks and 3.2–3.3 after 8, and a jointly unknown gain and bias is not recoverable from short histories by a learned regressor (R²(g) ≈ 0, R²(b) 0.56–0.76 `[quick check]`). So the first version trains and evaluates on **bias and inversion**, with gain loss as the stated boundary. This is the same information bound an external loop faces; the in-policy version cannot beat it, it can only match it without a box beside the robot.
5. **Costs.** $3.1: the premise pilot and one frozen-VLA fault collection (≈ $1.5–2.3), no training. $15–30: the minimal credible X6-EC on GRPO-130 — one collection of oracle-compensated rollouts (≈ $2.5), three LoRA fine-tunes (copy / no copy / shuffled copy, ≈ $2.5–3.5 each on an H100), ten evaluations (≈ $6–7) → **≈ $17 with one training seed, ≈ $28 with two seeds on the two main arms and two evaluation seeds**. More: gain loss, Base-Lora, a DAgger round, the transient-hijack evaluation (≈ $10–15 extra). Section 3.6 reconciles my $25–60 with your $5–7: $5–7 is one fine-tune plus one evaluation, which no reviewer accepts without the no-copy arm and held-out faults.
6. **First step, $0 now, $0.2 next:** the identifiability and copycat tables (done here, to be re-run under a logged rule), then a 1-task smoke that loads our `implement_version: rlinf` checkpoint through the `official` class with `use_proprio: false` and reproduces its success — because the proprio token exists only in the `official` implementation and the `rlinf` one builds embeddings from vision and text alone (`rlinf/openvla_oft_action_model.py` `_build_embedding`). If that smoke fails, an edit under `rlinf/` is unavoidable. Section 4.
7. **Framing that holds:** *VLAs are blind to their own actions: they carry no efference copy, which costs nothing while the world is fully observed and disturbances are momentary, and everything when the actuation itself changes — then the policy cannot tell that its commands are no longer doing what they did, and no amount of looking tells it. Giving the policy its own command–outcome history, and training it where that history matters, restores task success under faults it has never seen.* Section 5.
8. **Push back:** the narrative as stated ("no efference copy → poor disturbance robustness") will not survive review, because our strongest data set (X4) shows a disturbance class where the copy would not help. The scoped version can, if a $1 pilot shows that the frozen policy loses ≥ 0.3 of its success under realistic bias or inversion. And EC1 is not "conjured from nothing": it is the oracle teacher that generates X6-EC's training targets and the ceiling X6-EC is measured against. Section 6.

---

## 1. The causal premise, class by class

### 1.1 Transient overrides (our hijacks): the premise is false here, by our data

| what we measured | file | what it says about the copy |
|---|---|---|
| Hijack cost on mirror: 0.219 / 0.167 (E4 pooled), 0.565 / 0.291 (X1 GRPO-goal / traj1), both 60 % of clean success | `PREREG_X1_X4.md` (G), `x1_report.md` R6, X1 manifests `[quick check]` | VLAs are fragile to overrides — true |
| Perfect knowledge of the override + a one-call undo: B_orc +0.083 [+0.017, +0.150] (GRPO), +0.002 [−0.093, +0.098] (Base-Lora); Rule O fails on both | `x4_report.md` | knowing *does not fix it*; the oracle arm is exactly "a policy with an efference copy and the obvious reaction" |
| Every X4 failure is a 40-call timeout; the undo saves 1.9 [0.04, 3.7] calls | X4 manifests `[quick check]` | the cost is recovery time from an off-manifold state, i.e. competence of the visual loop, not attribution |
| The end-effector state is in the image embedding (L0 R² 0.84–0.97) and the last motion only through it (F2 gain ≤ 0.016 on all calls) | `a2_*.csv`, `f2_*.json` | in a fully observed Markov simulator the next optimal action is a function of the current state; the history adds no decision-relevant information for a one-off perturbation |
| GRPO's relative cost equals SFT's (0.603 vs 0.601 on mirror) | X1 manifests `[quick check]` | RL on own consequences did not change robustness to overrides either |

Honest sentence for the paper: *overrides hurt these policies (−17 to −57 points on mirror), and an efference copy would not have helped, because what is lost is time to recover from where the override left the arm, and the policy already knows where that is.* The only transient case where the copy could matter is partial observability (occlusion, no wrist camera, a push during a blind phase) — not in our data; do not claim it.

### 1.2 Persistent or latent actuation faults: the premise is plausible and unmeasured

Bias (miscalibration), gain loss (wear, a scaled drive), axis inversion (mis-wiring, a frame error), latency, payload change: the dynamics change, the image shows only the *result* of each chunk, and the optimal next command depends on the latent fault parameter. A memoryless policy can only react chunk by chunk (visual servoing); a policy with its command–outcome history can identify the latent and pre-compensate. Whether the copy is the *binding* limitation depends on whether reactive correction already suffices for task success at realistic magnitudes — nobody has measured that; it is the $1 pilot in §3.1.

What the history can identify, from the logged SELF transitions of `x1_grpo_swap` / `x1_sft_swap` (ridge forward model Δpos ~ [s, Σa, task], OOF R² 0.81–0.87, residual sd 1.15 / 0.98 cm on x, gain 1.02 / 0.98 cm per command unit) `[quick check]`:

| fault | statistic after K chunks | K = 1 | K = 2 | K = 4 | K = 8 |
|---|---|---|---|---|---|
| bias b = 0.2/step on x | z of b̂ | 1.4 / 1.6 | 2.0 / 2.3 | 2.8 / 3.2 | 4.0 / 4.5 |
| gain loss g = 0.6 (bias known) | median z of ĝ − 1 | 1.6 / 1.8 | 2.6 / 2.8 | 2.7 / 3.0 | 3.2 / 3.3 |
| gain loss g = 0.4 (bias known) | median z | 2.4 / 2.6 | 3.8 / 4.2 | 4.1 / 4.5 | 4.8 / 5.0 |
| axis inversion on x | median z; share of episodes with z ≥ 3 | 7.9 / 8.8; 0.93 / 0.94 | 12.8 / 13.9; 1.00 / 0.99 | 13.7 / 14.9; 1.00 | 16.0 / 16.7; 1.00 |

(GRPO-goal / traj1.) A learned regressor given K chunks of (a_sent, residual) with gain *and* bias unknown recovers the bias (R² 0.56–0.76 for K = 1–4) and not the gain (R² −0.06 to 0.03) `[quick check, MLP 64×64, grouped 5-fold]`: the two are confounded unless the commands vary enough within the window. Consequences for the design: bias and inversion first; gain loss as a reported boundary; a history of K ≥ 2 chunks, 4 if the token budget allows.

### 1.3 Supervisory and safety modifications: a different objective, out of scope

Speed-and-separation scaling, force limits that clip commands, a teleoperator's override: the copy would tell the policy that its commands are being modified, but the right response is to *yield* or re-plan at the imposed speed, not to compensate — compensating would defeat the safety function. That is a shared-autonomy objective with its own metrics. Say explicitly that the paper does not address it; it is the most likely place a robotics reviewer will push, and conceding it costs nothing.

### 1.4 How to scope the narrative so it survives

> VLAs carry no efference copy. That costs nothing while the environment is fully observed and disturbances are momentary — the policy re-plans from what it sees, and we show that even perfect knowledge of an override would not have helped (X4). It costs a great deal when the actuation itself changes: then the policy cannot tell that its commands have stopped doing what they did, and no image tells it. We give the policy its own command–outcome history and train it where that history matters.

---

## 2. The positive result: options, and the one to build

### 2.1 Why the copy must be an input, and why training is unavoidable

The incentive argument: under nominal dynamics the previous command carries no information about the optimal next action beyond the current state, so no training signal on clean data makes the network read it, and the strong action autocorrelation (consecutive chunk-sums correlate at 0.72 (x), 0.78 (y), 0.63–0.68 (z) `[quick check]`) makes a copy input a liability — the copycat failure. The copy is used only if the training data contain situations where the target action depends on it, i.e. randomised persistent faults with fault-dependent targets. EC0 adds the second constraint: a one-chunk residual is a weak signal at a strict threshold (V_true TPR 0.20–0.51 at FPR 0.02, `ec0_summary_all.json`), so the input must carry ≥ 2 chunks of history, and the paper should expect identification to take 2–4 chunks for bias (table above), which the evaluation must allow for.

### 2.2 Candidates

| option | training signal | why / why not |
|---|---|---|
| **(i) X6-EC: efference-copy token + LoRA, BC on oracle-compensated self-rollouts under randomised faults** | token NLL on targets a_sent = clip((a_cmd − b)/g) produced by the frozen policy + the known fault (EC1's oracle arm) | cheapest signal that makes the copy necessary; the teacher is the policy's own intent, so nominal behaviour is preserved; one collection, one short fine-tune per arm. **Chosen.** |
| (ii) + a DAgger round | relabel the student's own rollouts under faults with the same oracle | fixes the drifted-state gap of (i); adds one collection and one fine-tune per arm; second tier |
| (iii) GRPO under faults with RLinf's stack | task reward | the shipped config runs 64 envs × group 8 × 16 rollout epochs per step with `global_batch_size 16384` on one node, placement "all" GPUs (count not in the yaml; `[TO VERIFY]`); on one H100 a single step is hours `[inference]`; the reward carries no gradient toward *reading the copy* specifically. Not at these tiers. |
| (iv) history as prompt tokens, no projector | NLL | the LM must learn new token semantics from scratch; the projector path is built for exactly this; worse per dollar |
| (v) RMA: a separate adaptation module regressing (g, b) from history, fed to the projector | two-stage | more parts to ablate; the end-to-end projector on a K-chunk history is the minimal version; (v) is the fallback if (i) fails to use the copy |

### 2.3 What must be in the training data (from EC0 and the identifiability table)

1. Faults randomised **per episode** over the family and the magnitude range, including a sign-flip family (inversion) and a clean share of ≥ 30 % so nominal behaviour is anchored.
2. Targets that **depend on the fault** (the compensated command), not the policy's raw output — otherwise the copy predicts nothing.
3. **Drifted states**: an oracle-compensated collection keeps trajectories nominal, so the student never sees where an uncompensated fault leads; include ≈ 30 % of faulted episodes with compensation off (targets still the compensated commands), or run the DAgger round.
4. **History length K ≥ 2** in the input, zero-padded at episode start, with the state change Δs — not just the state — because the forward model's slope is what makes the residual readable.
5. A **shuffled-copy** arm trained on the same data, so that "the fine-tune under faults did it" and "the copy did it" are separable.

---

## 3. X6-EC — the minimal credible version

### 3.1 Stage 0 — the premise test (no training; this is also the $3.1 tier)

- **Pilot** (GRPO-130, 2 tasks × 8 states, frozen policy, fault on, no compensation): bias ‖b‖ ∈ {0.15, 0.30}, inversion on x, gain g ∈ {0.6, 0.4}; ≈ 5 × 2 min + start-up → **$0.5–0.9** `[estimate; $0.5 if the driver chains the settings in one container call, TO VERIFY]`.
- **Rule P (premise):** a fault family enters the paper iff the frozen policy's success under it is ≤ 0.6 at a magnitude inside the compensable range (bias ≤ 0.35: |a_x| > 0.7 on 7.7 % / 11.4 % of steps `[quick check]`; inversion: always). If bias fails Rule P at 0.30, the narrative holds for inversion only, and §6 applies.
- **Frozen-policy fault collection** (GRPO-130, 10 goal tasks × 30 states = 300 episodes, three arms: fault / fault + oracle inverse / clean; ≈ 13–15 min) → **$0.75**. This is EC1's run; under the operator's framing it is the baseline ("the VLA alone loses G_f"), the teacher's ceiling, and the first half of the training set.
- **$3.1 total: ≈ $1.5–2.3.** Nothing is trained; the paper gains the fragility measurement that the narrative currently only asserts.

### 3.2 Data for training (needs the lifted rule only for what follows)

- **Source:** the EC1 harness (`fault:` section, oracle compensation) with `frames.mode: all`, `jpeg_quality: 95`, hidden capture off, `extra_observables: true`. GRPO-130, all ten goal tasks. Per episode: fault family drawn {clean 0.30, bias 0.45, inversion 0.25}; bias direction uniform in the horizontal plane, ‖b‖ ~ U[0.10, 0.35]; inversion on x or y; compensation on for 70 % of faulted episodes, off for 30 %.
- **Size:** 10 tasks × 60 states = 600 episodes per seed ≈ 11k calls `[estimate: X4 GRPO ≈ 18 calls per episode]`; two seeds → **≈ 1,200 episodes, ≈ 22k samples**; frames ≈ 10 KB per call at the current quality (`frames/ep00000`: 9.9 KB main JPEG) → < 1 GB. Collection ≈ 2 × 25 min ≈ **$2.5** `[estimate]`.
- **Logged per call:** main image, instruction (recoverable from `task_id`), s(c), a_cmd(c) (policy), a_sent(c) (compensated; the target), a_exec(c), Δs(c), the fault parameters, condition.

### 3.3 Model and training

- **Load path:** `implement_version: official` (the only class with `use_proprio` / `ProprioProjector`; `model/openvla_oft.yaml` says so and the `rlinf` class's `_build_embedding` uses vision and text only), `use_proprio: true`, `proprio_dim: 136`, `is_lora: true`, `lora_rank: 32`. Weight compatibility of our `rlinf`-version checkpoints with the `official` class is **`[TO VERIFY]`** by the §4 smoke; if they do not load, an `rlinf/` change is unavoidable and the operator must decide.
- **Proprio vector (136-d, K = 2):** [s(c) (8), a_sent(c−1) (56), Δs(c−1) (8), a_sent(c−2) (56), Δs(c−2) (8)]; a_sent already in [−1, 1]; Δs scaled by per-dim SELF sd; s by the dataset's proprio stats; zero-padded at episode start. `normalize_proprio` clips to [−1, 1] after scaling by `norm_stats[unnorm_key]["proprio"]` (q01/q99 or min/max, `openvla_utils.py`), so the harness and the trainer inject a 136-d stats entry into `model.norm_stats` after loading — a dict attribute, no `rlinf/` edit `[TO VERIFY that the vendored `prismatic` projector accepts proprio_dim = 136; the constructor takes it as a free argument]`.
- **Trained parameters:** the projector (fresh, full) and LoRA r = 32 on the LLM blocks (the repo's target list, `openvla_utils.py`; drop the vision targets for a first run). Everything else frozen.
- **Loss:** token NLL of the 56 target tokens, obtained from `default_forward(forward_inputs={..., action_tokens: targets}, compute_logprobs=True, temperature=1.0, top_k=-1)` — the path the GRPO actor already uses; targets by inverting the model's own bin mapping (`bin_centers`, `vocab_size − 1 − bin`). New training loop in `experiments/efference_probe/train_ec.py`; the model's `forward` has no SFT branch (`ForwardType.SFT` raises in this class), so the loop stays outside `rlinf/`.
- **Arms:** **A** copy (136-d token); **B** no copy (8-d token: s(c) only — same projector, same LoRA, same data; isolates the history); **C** shuffled copy (history drawn from a random other sample at training time; true history at test). Plus **F**, the frozen policy.
- **Schedule:** batch 16, 2 epochs over 22k samples ≈ 2,800 steps; LR 1e-4 projector, 5e-5 LoRA, cosine, bf16, flash-attention `[inference from OFT fine-tuning defaults]`. **GPU: H100** (80 GB; batch 16 without checkpointing; ≈ 0.6 s/step `[inference]` → ≈ 30–45 min ≈ **$2.5–3.5 per fine-tune**). On an L40S (48 GB) with gradient checkpointing ≈ 2× the wall-clock at 0.58× the price — similar cost, longer `[inference]`.
- **Seeds:** one training seed per arm in the $17 version; two for A and B in the $28 version (the between-seed variance of fine-tunes is the dominant uncertainty, not the evaluation's).

### 3.4 Evaluation protocol

- Per (fault family × arm): 300 episodes (10 tasks × 30 states), 80 % faulted at **held-out** magnitudes and directions (bias ‖b‖ ∈ {0.20, 0.30} on directions not used in training; an extrapolation point 0.40; inversion on the axis not emphasised in training), 20 % clean. No hidden capture → ≈ 10 min ≈ **$0.6** each `[estimate]`.
- **Minimal set (10 evaluations ≈ $6–7):** bias × {A, B, C, F}, inversion × {A, B, C, F}, clean × {A, B}; F's clean and fault numbers come from §3.1.
- **Metrics:** success; RMST(40) (F8's censoring-correct metric); identification: slope of the student's command deviation (a_student − a_F on the same image, where a_F is the frozen policy's command logged in the same call) on the true b, ideal −1; time-to-compensation (first call at which that deviation exceeds half the ideal).

### 3.5 Pre-registered rules and power

Per fault family; G_f = success(F, clean) − success(F, fault) from §3.1; episode bootstrap B = 2000.

- **Rule E (efficacy):** success(A, fault) − success(F, fault) ≥ 0.5·G_f with CI_low > 0, at the held-out magnitudes.
- **Rule N (the copy, not the fine-tune):** success(A, fault) − success(B, fault) ≥ 0.15 with CI_low > 0, **and** success(A, fault) − success(C, fault) ≥ 0.15 with CI_low > 0.
- **Rule 0 (no nominal cost):** success(A, clean) ≥ success(F, clean) − 0.05 (CI).
- **Rule X (extrapolation, descriptive):** A's success at 0.40 reported against 0.30.
- **Readings:** E ∧ N ∧ 0 → the positive claim; E ∧ ¬N → "fine-tuning under faults, not the copy, restores success" (honest, publishable as domain randomisation, kills the narrative's mechanism); N ∧ ¬E → "the copy is read but not enough"; ¬0 → the fine-tune damaged the policy, report and stop.
- **Power `[estimate]`** at 240 faulted episodes per arm per evaluation, two-sided α = 0.05: F at 0.50 and A at 0.80 (B = 0.30, G_f ≈ 0.47) → SE 0.041, power > 0.99 for Rule E; A 0.80 vs B 0.65 (Rule N at 0.15) → SE 0.040, z 3.7, power 0.96; Rule 0 with F clean 0.968 (X4) and 60 clean episodes per arm: the SE of the difference is ≈ 0.04, so a 0.05 drop is below the detection edge — the $28 version doubles the clean episodes, and Rule 0 should be read with its CI rather than as a pass/fail. The evaluation is not the bottleneck; a second training seed is.

### 3.6 Cost, wall-clock, and the reconciliation

| tier | contents | GPU cost `[estimate]` | wall-clock |
|---|---|---|---|
| **$3.1** (no training) | pilot + frozen-policy fault collection (three arms) | $1.5–2.3 | 1 day of code (EC1's fault module), 30 min GPU |
| **≈ $17** | + training collection (2 seeds, $2.5) + 3 fine-tunes (A, B, C; $7.5–10.5 on H100) + 10 evaluations ($6–7) | **$17–22** | 1–2 weeks of code (trainer, dataset, official-class smoke, plans, report), ≈ 5 GPU-h |
| **≈ $28** | + second training seed for A and B (+$5–7), second evaluation seed on the two fault families (+$5), doubled clean evaluations | **$27–34** | + ≈ 3 GPU-h |
| more | + gain loss (collection, evaluation, +$3), + Base-Lora as a second base (+$12), + one DAgger round (+$6), + transient-hijack evaluation of A vs F on mirror (+$1.2) | +$10–25 | + 1 week |

**Reconciling $5–7 with $25–60.** $5–7 buys one fine-tune ($2.5–3.5) and one or two evaluations — enough to see whether the copy-conditioned policy succeeds under a fault, not enough to say *why*: no no-copy arm, no shuffled-copy arm, no held-out magnitudes, one seed. The X0 memo's $25–60 priced a demonstration-data pipeline built from scratch plus a broad evaluation; the oracle-compensated self-rollout teacher removes the pipeline and the shipped LoRA path removes most of the training code, which is why the credible version now lands at $17–34. Your number is the training; mine was the experiment; the honest figure for an RSS-grade result is between them.

### 3.7 New code (all under `experiments/efference_probe/`)

`faults.py` (fault transforms, oracle inverse, history buffer; EC1's module), harness `fault:` section and history logging, a flag that hands the model a separate 136-d `states` vector while the 8-d state is still logged, `ec_dataset.py` (calls + frames → samples; target tokenisation through the model's own bin mapping), `train_ec.py` (model via `rlinf.models.embodiment.openvla_oft.official.get_model`, LoRA via the shipped config keys, NLL loss, projector + LoRA checkpointing, `load_proprio_projector_weights` / `lora_path` for reload), `x6_plan.yaml`, `x6_report.py`, offline tests with the fake env. **Unavoidable `rlinf/` edit:** none if the `official` class loads our checkpoints and accepts a 136-d proprio stats entry; both are `[TO VERIFY]` by the smoke in §4, and the second is more likely to need a wrapper than a patch.

---

## 4. Failure modes and the cheapest de-risking step for each

| failure mode | evidence now | de-risk | cost |
|---|---|---|---|
| **Copycat / causal confusion** (the student copies a_sent(c−1)) | consecutive-chunk correlation 0.72 (x), 0.78 (y) `[quick check]` | faults randomised per episode so the history is informative; arm C (shuffled copy) separates "uses the copy" from "copies the copy"; Rule 0 on clean episodes; report A's clean action error against F's commands | in the design |
| **The visual loop absorbs the fault** (G_f small) | unmeasured | Rule P pilot before any training; inversion as the guaranteed-large case | $0.5–0.9 |
| **The copy is not read** (projector ignored) | risk inherent to a fresh token | arm C; an input-ablation at test (zero the history → A should fall to B); identification slope | in the design |
| **Clipping** | |a_x| > 0.7 on 8–11 % of steps; 90th pct 0.64–0.73 `[quick check]` | bias ≤ 0.35; the oracle arm's success is the ceiling; report the clipped share | $0 |
| **Gain and bias confounded; gain loss unidentifiable from short histories** | R²(g) ≈ 0 jointly; z ≈ 2.7–3.0 at K = 4 alone `[quick check]` | bias + inversion first; gain loss only in the "more" tier, with K = 4 and large-command tasks; state it as the boundary | $0 |
| **Our checkpoints do not load through the `official` class** | `implement_version: rlinf` in `run_config.yaml`; proprio path only in `official` | 1-task smoke, `use_proprio: false`, compare success with the `rlinf` path | ≈ $0.2 |
| **Nominal performance drops after the fine-tune** | — | LoRA, short schedule, 30 % clean data, Rule 0 | in the design |
| **Drifted states unseen in training** | oracle-compensated rollouts stay nominal | 30 % uncompensated faulted episodes in the collection; DAgger round in the larger tier | $0 / +$6 |
| **Training-seed variance** | fine-tunes of 7B VLAs vary by seed `[inference]` | two seeds on A and B in the $28 version; report both | +$5–7 |
| **Success under a 40-call cap is a time metric** | every X4 failure is a timeout | RMST(40) alongside; faults slow episodes, so the clean cap stays 40 | $0 |
| **"In-context adaptation under domain randomisation is known"** (RMA, sim-to-real literature) | true | the contribution is the diagnosis (no copy, replicated across SFT and RL), the minimal fix through the model's existing input path, and the ablation that isolates the copy; cite RMA and say so | writing |

**The cheapest first step** is free and already half done: re-run the identifiability and copycat tables under a logged rule (`[quick check]` → pre-registered), then the $0.2 official-class smoke. The first step that costs real money is the Rule P pilot, and it is the same pilot EC1 needed.

---

## 5. Framing under the operator's narrative

**One-sentence thesis.** *VLAs are blind to their own actions: they carry no efference copy, which costs nothing while the world is fully observed and disturbances are momentary, but leaves them unable to notice when their actuation changes; giving the policy its own command–outcome history, and training it where that history matters, restores task success under actuator faults it has never seen.*

**Claims, and how the existing results feed them.**

1. **No efference copy, before or after RL.** P3 − P1 within ±0.04 in 36/36 rows; 17/18 linear bootstrap CIs cover 0; fixed cell +0.035 [−0.004, +0.074] (X1), +0.050 / +0.064 (E4, post hoc), state-explained. — E4, X0, X1 (`a1`, `a7`, `x1_report.md`, `fixed_cell.json`). *Role:* the diagnosis.
2. **Why: the states hold the current state and nothing about authorship.** L0 R² 0.84–0.97; F2 gain ≤ 0.016; every above-floor rung ≤ C_obs + δ. — X0, X1. *Role:* the mechanism, and the reason a copy must be an *input*, not something to probe for.
3. **For momentary overrides the copy would not help — recovery, not attribution, is the limit.** X4 oracle +0.083 / +0.002; every failure a timeout; relative costs equal across SFT and RL. — X4, X1 manifests. *Role:* the honest scoping that makes claim 4 credible.
4. **For persistent actuation faults the frozen policy loses G_f and cannot recover; the frozen policy's own command–outcome history identifies the fault in 1–4 chunks** (identifiability table). — §3.1 collection, `[quick check]` tables. *Role:* the premise, measured.
5. **The copy, given as input and trained under randomised faults, restores ≥ half of G_f under unseen faults, at no nominal cost; the same fine-tune without the copy, or with it shuffled, does not.** — X6-EC. *Role:* the positive result.
6. (Supporting) **A one-shot label-free detector is bounded** (EC0 V_true ≤ 0.51 at FPR 0.02; V_feat ≤ 0.16), which is why the policy is given a history rather than a flag, and why identification takes chunks, not a call. — EC0/EC0b.

Where the earlier constructs go: X4's monitor becomes the external baseline in claim 3 ("even a perfect external detector…"); EC1's oracle compensator is the teacher of claim 5 and its ceiling; the novelty-gated-compensation mechanism is retired, its measured successor being the frozen policy's dose–response under faults (how much the visual loop absorbs).

---

## 6. Push back, plainly

1. **The narrative as stated cannot be made to hold with any experiment, because X4 already refutes it for the disturbances we have.** "No efference copy → poor disturbance robustness" is wrong for momentary overrides in a fully observed environment, and the paper's own oracle arm is the proof. A reviewer who reads X4 and then reads that sentence stops trusting the rest. The scoped narrative of §1.4 is the closest honest version, and it is a good one.
2. **The scoped narrative is itself an empirical bet with a $1 price tag.** If the frozen policy keeps ≥ 0.6 of its success under a bias of 0.30 per step, the visual loop absorbs realistic miscalibration and the premise holds only for inversion, which reviewers will call contrived. Run Rule P before anyone writes "fragile" in the abstract.
3. **"Fix it in the policy" cannot do better than the information allows.** The in-policy copy faces the same identifiability bound as an external estimator (§1.2); bias in 2–4 chunks, inversion in 1, gain loss slowly or not at all with short histories. The claim is "the policy can internalise the loop", not "the policy becomes robust to actuation faults in general". Write the boundary into the abstract.
4. **EC1 is not conjured from nothing; it is the teacher.** X6-EC's training targets *are* EC1's oracle-compensated commands, and EC1's oracle arm is the ceiling every X6-EC number is read against. Dropping EC1 does not remove the external loop from the paper; it removes the baseline that shows the fine-tune learned what the loop computes. Keep it as a baseline, not a headline, and the operator's objection is met.
5. **The operator's objection is to the framing, and the framing is the cheap part.** The constructive memo's experiment and this memo's experiment share a fault module, a pilot, a forward model and a collection; the only addition here is the fine-tune and its ablations ($15–30 and 1–2 weeks). If the no-training rule stays, the honest paper is the scoped narrative with claims 1–4 and 6, with X6-EC written as the pre-registered next experiment with the design above — and that is still a stronger RSS paper than the current draft.

---

## 7. Statements in the brief that the files support, and those they do not

- **"The memoryless visual loop already re-plans every call"** — supported (X4 off arm; the policy is queried every call; `harness._run_batch`).
- **"Fed through OFT's existing proprio-projector path (`use_proprio`, `ProprioProjector`, `proprio_dim`)"** — exists in the `official` implementation only; our checkpoints run through `implement_version: rlinf` (`data/x1/x1_grpo_swap/run_config.yaml`), whose `_build_embedding` has no proprio token. Loading them through `official` is `[TO VERIFY]`.
- **"RMA-style adaptation from state–action history"** — the right reference; note RMA's adaptation module is separate, whereas the minimal version here is end-to-end through the projector (§2.2 (v) is the RMA form, kept as the fallback).
- **"A small RL budget via RLinf's existing GRPO stack"** — the shipped LIBERO-Goal GRPO config is a multi-GPU, 16,384-sample-per-step job (`libero_goal_grpo_openvlaoft.yaml`); nothing "small" fits these tiers `[inference on footprint; GPU count TO VERIFY]`.
- **"$5–7 minimal estimate"** — correct for one fine-tune plus one evaluation (§3.6); not for a result with the ablations a reviewer will demand.
- **"The missing copy hurts robustness to disturbances"** — supported for none of the disturbances measured so far; plausible for persistent actuation faults; §1.

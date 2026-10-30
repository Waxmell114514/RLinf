# Memo: the efference copy as a lens — benchmark (A), design guidance (B), and which one the evidence can carry

*Advisory memo by Fable, 2026-10-30, from a read-only pass over the repo; follows `next_steps_fable_positive.md`. Not a pre-registered verdict; nothing here changes an E4, X0, X1, X4, EC0 or X6 stage-0 call. Every number is quoted from a named file. `[estimate]`, `[inference]`, `[TO VERIFY]` as before; `[quick check]` marks a number I computed in this pass from the named `calls.parquet` / `manifest.json` with pandas — a diagnostic, to be re-run under a logged rule before it enters the paper. Prices: rentals at ≈ $0.6–0.8/h for an 80 GB card (your Vast/AutoDL figures), Modal L40S ≈ $2.8/h; a 300-episode OFT collection takes 13–15 min (X1/X4 ledgers) → ≈ $0.2 on a rental, ≈ $0.75 on Modal. Collection needs ≥ 64 GB RAM and ≈ 25 GB VRAM at 10 envs (your measurement). Everything that fine-tunes a model needs the no-training rule lifted; the fault curves of frozen models do not.*

Files read: logbook entries "X6 stage 0, step 1/2/3 RESULT" and the steps 2–3 procedure entry; `data/x6/identify.json`; `data/x6/x6_pilot_grpo/manifest.json` and `calls.parquet`; `rlinf/models/embodiment/` listing and the module headers of `cosmos3`, `dreamzero`, `evo1`, `gr00t`, `starvla`, `molmoact2`, `openpi`, `lingbotvla`, `abot_m0`; `rlinf/models/embodiment/cosmos3/cosmos3_policy.py`, `cosmos3/sglang_adapter.py` (grep), `dreamzero/dreamzero_policy.py`, `gr00t/simulation_io.py` (grep), `openpi/openpi_action_model.py` (grep), `base_policy.py`; `examples/sft/config/` listing and `libero_sft_{cosmos3,dreamzero_5b,openpi,evo1}.yaml`; `examples/sft/config/model/cosmos3.yaml`; `examples/embodiment/config/` LIBERO listing and `model/{dreamzero_5b,evo1,gr00t,gr00t_n1d6,gr00t_n1d7,starvla,pi0,pi0_5}.yaml`; the LIBERO configs for π0, π0.5, GR00T (N1.5, N1.7), Evo-1, starVLA, Dexbotic π0, aBot-M0 (grep for checkpoints); all `huggingface.co` references in those configs; `docs/source-en/rst_source/examples/embodied/sft_cosmos3.rst`, `sft_dreamzero.rst`, `examples/world_model_index.rst`, `evaluations/guides/{cosmos3,dreamzero}_sglang.rst`; `README.md` (world-model entries); `experiments/efference_probe/rental_runbook.sh` (header). HF model cards were not fetched (no network use); every checkpoint fact below comes from the repo's configs and docs and is marked `[TO VERIFY]` where the card would decide it.

---

## 0. Bottom line in eight sentences

1. **(A) as a ranking of training methods is, for the models within reach, 天方夜谭 — and our own results say why.** Every LIBERO policy this repo can run takes the current image and at most the current state: OpenVLA-OFT (one image, no state), π0 / π0.5 / GR00T / Evo-1 / starVLA / Dexbotic (images + current proprio), DreamZero and Cosmos3 (current frame + state, then they *generate* forward). None receives its previous command. The efference increment is therefore zero by architecture for all of them, and a benchmark that scores it would return the same null for SFT, RL, proprio-input and world-model-pretrained policies alike — which is the "obvious by architecture" shrug, now across seven architectures instead of two.
2. **What survives of (A) is a behavioural benchmark with an architectural prediction attached:** persistent-fault tolerance (the Rule P curves), which the lens predicts every memoryless policy fails regardless of objective — including a 5B video world model fine-tuned into a policy. That prediction is cheap to test on three or four public LIBERO checkpoints and is more interesting as a *confirmation of the lens* than as a leaderboard.
3. **(B) is the defensible story, and X6 stage 0 made it concrete.** Frozen GRPO-130: clean 0.938 → bias 0.15 0.562, bias 0.30 0.000, inversion 0.000, gain 0.6 0.875 (absorbed), gain 0.4 0.562 (`x6_pilot_grpo/manifest.json`, n = 16 each). Under the 0.30 bias the policy *reacts* — by call 15 its mean commanded x is −0.31 against the +0.30 fault `[quick check]` — and still times out in 16 of 16 episodes; under inversion it never adapts (mean commanded x +0.35 throughout, executed −0.35). Identifiability from the policy's own command–outcome history: K = 2 chunks suffice for bias 0.30 and inversion (`identify.json`).
4. **Recommended narrative: (B) with one slice of (A) as its control.** *The efference copy is a design lens: it predicts that neither RL post-training (shown), nor a proprio input (π0, GR00T — cheap to test), nor world-model pretraining (DreamZero LIBERO checkpoint, testable at inference) gives a policy knowledge of its own actions; and it says what does — a two-chunk history of sent commands and state changes through the model's existing proprio path, trained where that history matters.* A 2 × 2 ablation (copy input × auxiliary forward-model loss) makes the guidance sharp: the lens predicts the loss without the channel does nothing for fault tolerance.
5. **The efficiency claim that comes with it:** one fault-randomised fine-tune gives in-context adaptation to faults never seen in training, replacing per-fault re-tuning at deployment; measured against a per-fault specialist fine-tune as the ceiling. (Sample efficiency of RL with the copy is a separate, expensive question; ambitious tier only.)
6. **Costs on rentals:** minimal ≈ $15–25 and 3–4 engineering weeks (X6-EC with the 2 × 2, π0's fault curve); solid ≈ $50–80 and 6–8 weeks (+ seeds, a second base, gain loss, GR00T and DreamZero fault curves, probe ladders on two more architectures); ambitious ≥ $300 and 3+ months (RL post-training with the copy, a second simulator, a real arm). Section 5.
7. **First $0 step:** a generic fault-evaluation harness over `BasePolicy.predict_action_batch` (fault injection at the env action, capture off), the pre-registration of the 2 × 2 and of the cross-architecture prediction, and the `[TO VERIFY]` checks on HF (DreamZero inference speed, released RL siblings of π0, any WoVR-trained OFT checkpoint — the cheapest "world-model-trained" comparison of all, because our whole pipeline applies to it unchanged).
8. **The loading smoke decides the class.** The `official` and `rlinf` OFT classes disagree on 4.3 % of greedy tokens on the same observations (logbook, step 2), so every X6-EC arm, frozen baseline included, must run through one class; a wrapper that adds the proprio token to the `rlinf` class keeps continuity with E4, X1, X4 and the pilot, and is the option to take.

---

## 1. Verdict on (A) and (B)

### 1.1 Why (A) cannot rank training methods with these models

The paper's own mechanism result is the obstacle: the command exists only in the forward pass that produced it (X0 §1; `x0_report.md`), so an *internal* efference comparison requires the previous command, or something that carries it, to be an input of the next forward pass. Which models have that?

| model (repo checkpoint) | inputs at call c | previous command in the forward pass? | training objective | lens prediction |
|---|---|---|---|---|
| OpenVLA-OFT (goal-traj1, GRPO-goal, Base-Lora, GRPO-130) | 1 image, instruction; no state (`model/openvla_oft.yaml` `use_proprio: False`) | no | token NLL on action bins; GRPO | increment 0 — **measured** (36/36 rows) |
| π0 / π0.5 (`RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT`, `RLinf-Pi05-LIBERO-SFT`) | 2 images + current state (`pi0.yaml` `use_proprio: True`, `num_images_in_input: 2`) | no | flow matching on action chunks | increment 0; state readability trivial (it is an input) |
| GR00T N1.5 (`RLinf-Gr00t-SFT-{Goal,Object,Spatial}`) | main + wrist image, state x/y/z/rpy/gripper (`simulation_io.py`) | no | flow-matching DiT | increment 0 |
| Evo-1 (`MINT-SJTU/Evo1_LIBERO`), starVLA (`StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1`), Dexbotic π0 / DM0 | images (+ state, `use_proprio: True` for starVLA; Evo-1 `max_state_dim` `[TO VERIFY]`) | no | flow / OFT-style heads | increment 0 |
| DreamZero 5B LIBERO (`RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000`, `-Diffusers`) | current frame(s) + state (`state_horizon: 1`), generates 33 frames and 64 actions jointly, executes 16 (`dreamzero_5b.yaml`, docs) | no `[inference from `state_horizon: 1`; whether the SGLang server keeps context across calls is TO VERIFY]` | joint video + action generation (a world model fine-tuned into a policy) | increment 0 at the *policy* level; the generated video is a forward model of its *planned* actions, not a record of the executed one |
| Cosmos3-Nano LIBERO | `num_frames: 17` video context, state; action heads on a frozen understanding tower (`model/cosmos3.yaml`) | no | video world model + action heads | increment 0; **no public LIBERO checkpoint in the docs; SFT recipe is 8 × A800, 5,000 steps** — out of reach |

The only training-method contrast the lens can rank is between *representations that could carry the relation* and those that cannot, and every candidate falls on the same side. A benchmark whose predicted outcome is "0 for everyone" is a demonstration of the lens, not a measuring stick for training methods. Two caveats to keep the verdict honest: (i) a world-model policy that *does* re-condition on its own generated context across calls (a KV cache, a rolling frame window including its own past predictions) would be the one architecture where the comparison could happen internally; the repo's DreamZero serving path makes this a `[TO VERIFY]`, and if it turns out to carry context, DreamZero becomes the single most interesting subject for the probes; (ii) a WoVR-style OFT (RL inside a learned world model, `wan_libero_goal_grpo_openvlaoft.yaml`) is still a one-image OFT at inference — the lens predicts nothing changes, and testing it costs three collections.

### 1.2 What survives of (A): the behavioural benchmark, and its validation link

The quantity that *does* vary across models is how much persistent actuation fault each policy's visual loop absorbs — the Rule P curve. The lens makes a falsifiable prediction about it: **every memoryless policy fails persistent bias and inversion at magnitudes its own history would identify in ≤ 2 chunks, whatever its objective, input set or size.** That is a benchmark result worth one figure (success vs fault magnitude, one curve per architecture, the copy-trained OFT as the only curve that stays up), and it is the control that (B) needs. Its validation link is the prediction itself, pre-registered as a binary: per memoryless model, success(bias 0.30) / success(clean) ≤ 0.5 and success(inversion) ≤ 0.2; for the copy-trained policy, ≥ 0.8 of clean at held-out magnitudes. With four or five models there is no regression to run; the claim is categorical, and a single memoryless model that passes would falsify the lens as a design rule — which is exactly what makes it worth running.

What I would not build: an "efference-copy score" from P3 − P1, E1, A2, A3 and EC0 combined. P3 − P1 is 0 everywhere by architecture; E1 and A2 are readability numbers that are trivially 1 for models with a state input; A3 and EC0 measure the *external* reader. None of them predicts fault tolerance across these models better than the one bit "does the forward pass see the previous command".

### 1.3 Why (B) is the story

The pilot gives the problem a number and a mechanism: the frozen policy under a 0.30 bias drives its commanded x from +0.41 (calls 0–1) to −0.39 (calls 2–3) and settles at −0.31 (calls 15–39), i.e. it learns nothing but reacts hard, nets +0.07 per step of unwanted drift, and reaches the 40-call cap in 16 of 16 episodes `[quick check, x6_pilot_grpo/calls.parquet]`; under inversion it commands +0.33 to +0.51 throughout, because nothing in its input tells it the sign is flipped. The lens says the missing input is a two-chunk history of sent commands and state changes (`identify.json`: bias 0.30 median z 4.28 at K = 2, inversion 17.5), the existing proprio path is where it goes, and training under randomised faults is what makes the network read it (the Markov/incentive argument of the previous memo). That is design guidance with a prediction, an ablation and a cost.

---

## 2. (A), designed as far as it should be

### 2.1 Measurements (keep two, drop the rest)

- **Fault-tolerance curve:** success and RMST(40) vs fault magnitude for bias (0.15, 0.30), inversion, gain (0.6, 0.4), 300 episodes per cell, clean reference in-run. This is the benchmark.
- **Reaction trace:** mean commanded translation per call under each fault (the "reacts but never adapts" figure), which separates visual-loop reaction from identification without any probe.
- **Optional, only for architectures that could carry the relation** (a DreamZero that keeps context, a future history-conditioned policy): the probe ladder P1/P3/P4 with the selection-matched floor, as in E4. For memoryless models it is a predicted zero and one line in the appendix.
- Dropped as benchmark quantities: E1, A2 (trivial with a state input), A3 and EC0 (they measure an external reader; keep them as the paper's measurement tools, not as model scores).

### 2.2 Model-agnostic protocol

The fault is injected at the executed env action — exactly where `build_executed_chunk` sits in `harness._run_batch` — so it is independent of the action head (tokens, flow, diffusion, video). The harness already calls `self.model.predict_action_batch(env_obs=...)` and `self.env.chunk_step(executed)` and maps model to env actions through `prepare_actions(model_type=...)`; with hidden-state capture off, a `model_type` switch plus per-model chunk length and image set (π0 and GR00T use the wrist camera, which `obs["wrist_images"]` already provides) is what is needed. **Engineering ≈ 1 week** `[estimate]` for the HuggingFace-backend models (OFT, π0, π0.5, GR00T, starVLA, Evo-1); DreamZero and Cosmos3 run through an SGLang server (`evaluations/guides/*_sglang.rst`) and need the client path instead, **≈ 1–2 weeks and 1–2 GPUs** `[estimate]`.

### 2.3 Models to compare, expected result, cost per model on rentals `[estimate]`

| model | checkpoint source | prediction under bias 0.30 / inversion | cost (3 fault settings + clean, 300 episodes each) | effort |
|---|---|---|---|---|
| OFT GRPO-130 (done at n = 16) | X4 Volume | fails / fails — measured 0.00 / 0.00 | $0.8 | 0 |
| OFT Base-Lora | X4 Volume | fails / fails | $0.8 | 0 |
| π0 LIBERO SFT | `RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT` | fails / fails (state input ≠ efference) | $0.8 | shared harness week |
| GR00T N1.5 Goal | `RLinf/RLinf-Gr00t-SFT-Goal` | fails / fails | $0.8 | shared |
| DreamZero 5B LIBERO | `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers` | fails / fails (world-model pretraining ≠ efference) | $3–6 `[inference speed TO VERIFY; 1–2 GPUs]` | 1–2 weeks |
| WoVR-trained OFT (if released) | `[TO VERIFY on HF]` | fails / fails; probes also 0 | $0.8 (+ $0.6 for the probe ladders) | 0 |
| copy-trained OFT (X6-EC arm A) | ours | ≥ 0.8 of clean / ≥ 0.8 | $0.8 | §3 |

Cosmos3 is not in the table: no LIBERO checkpoint is documented and its SFT is an 8 × A800 job.

### 2.4 Verdict on (A)

Run it as the control of (B), on three memoryless architectures beyond OFT (π0, GR00T, DreamZero) — one figure, one categorical pre-registered prediction. Do not build it as a leaderboard; do not promise it ranks SFT against RL against world-model pretraining, because the honest ranking is a tie at zero.

---

## 3. (B) design guidance: the experiment that shows the lens pays

### 3.1 The design change, and the 2 × 2 that isolates it

The lens implies one input-level change (a short command–outcome history) and warns against one tempting objective-level change (a consequence-prediction loss that gives the network a forward model it can never compare against at the next call). Test both at once:

| arm | input token | auxiliary loss | lens prediction under unseen bias / inversion |
|---|---|---|---|
| F | frozen GRPO-130 | — | fails (measured at n = 16) |
| B | state only (8-d) | — | fails on inversion; on bias at most hedges (direction randomised, so no hedge works on average) |
| D | state only | forward-model head on [h, a_cmd] → Δs (clean + fault data) | same as B — the loss cannot be used at the next call |
| A | state + 2-chunk history (136-d) | — | passes (Rule E, N) |
| E | state + history | forward-model head | ≥ A; reported, no rule |
| C | state + *shuffled* history at training time | — | fails — separates "reads the copy" from "copies the copy" |

All five fine-tunes share the data of `next_steps_fable_positive.md` §3.2 (oracle-compensated self-rollouts, faults randomised per episode, 30 % clean, 30 % uncompensated), the same LoRA rank, steps and seeds. Arm D's head is a small MLP on the pooled hidden state and the commanded chunk, trained on the logged Δs; its only role is to show that a forward model *inside* the policy, without the channel, does not change fault tolerance.

**Rules** (per fault family, held-out magnitudes, 240 faulted episodes per arm, episode bootstrap B = 2000): Rule E (A − F ≥ 0.5·G_f, CI_low > 0); Rule N (A − B ≥ 0.15 and A − C ≥ 0.15, CI_low > 0); **Rule L ("loss without channel is inert"): |D − B| ≤ 0.05 with the CI inside ±0.10**; Rule 0 (A's clean success ≥ F − 0.05, CI). Power at these n: E > 0.99 at a 0.30 effect; N 0.96 at 0.15; L is an equivalence test with half-width ≈ 0.08 at these sizes `[estimate]`.

**Why inversion is the discriminator.** A reactive policy without the copy cannot know that it moved the wrong way — the state is in the image, the direction it *intended* is not — so no hedge solves a sign flip, while bias has a weak hedge (move cautiously). If B passes inversion, the lens is wrong about what the input carries, and the paper should say so.

### 3.2 The efficiency claim

- **Deployment data efficiency:** one fault-randomised fine-tune vs per-fault specialists. Add two specialist fine-tunes (bias 0.30 only; inversion only; no copy) as the ceiling for their own fault. Rule S: A within 0.05 of each specialist on the specialist's fault, and ≥ specialist + 0.20 on the other fault. Cost: two fine-tunes and four evaluations.
- **Sample efficiency of RL post-training** with the copy: the only way to claim it is GRPO from A vs from B in a fault-randomised environment, with learning curves — RLinf's LIBERO GRPO recipe is a multi-GPU, 16,384-sample-per-step job; ≥ $100–300 per run on rentals `[inference]`. Ambitious tier; do not promise it.
- **Nominal efficiency** (fewer calls to success on clean tasks): report RMST(40) of A vs F on clean episodes; no claim expected.

### 3.3 How X6-EC changes

- The class: a wrapper adding the proprio token to the `rlinf` OFT class, in `experiments/`, so F, A–E and every earlier result share one forward path (the 95.7 % agreement rules out mixing classes).
- Arms: A, B, C stay; D and E are added (+2 fine-tunes, +4 evaluations); the two specialists are optional (+2, +4).
- The frozen-policy fault curve (300 episodes, three arms) is run once and reused as F, as the oracle ceiling, and as the OFT row of §2.3.
- Everything else (data, K = 2, 136-d token, LoRA r = 32, NLL loss, evaluation protocol, Rules E/N/0) as in the previous memo.

---

## 4. The recommended narrative

**Title-level thesis.** *Policies that cannot know what they did: the efference copy as a design lens for VLAs.* A single-image VLA carries no record of its own command when it sees the outcome; that is harmless while the world is fully observed and disturbances are momentary, and crippling when the actuation itself changes. We show the gap is architectural, not a training deficit — RL post-training, proprio inputs and world-model pretraining leave it untouched — and that a two-chunk command–outcome history through the model's existing proprio path, trained under randomised faults, closes it: the policy identifies and cancels actuator faults it has never seen, with no change to its nominal behaviour, while a forward-model loss without the channel does nothing.

**Claims.**

1. **No readable self-attribution, before or after RL** — P3 − P1 within ±0.04 in 36/36 rows, 17/18 linear bootstrap CIs covering 0, fixed cell +0.035 [−0.004, +0.074] on X1 (`a1`, `a7`, `x1_report.md`, `fixed_cell.json`). *Reframed from "a null result" to "the lens applied to two training methods".*
2. **The reason is where the command lives**: states hold the current end-effector state from the image embedding on (L0 R² 0.84–0.97) and the last motion only through it (F2 gain ≤ 0.016); every above-floor probe is explained by observables once objects and joints are logged (`a2_*.csv`, `f2_*.json`, `x1_report.md` R2). *Reframed as the architectural premise of the design rule.*
3. **Momentary overrides are a recovery problem, not an attribution problem** — the X4 oracle (+0.083 [+0.017, +0.150] GRPO; +0.002 Base-Lora; every failure a 40-call timeout) and the equal relative costs of SFT and RL. *Kept as the honest boundary: where the copy would not help.*
4. **Persistent actuation faults are where the gap bites, for every memoryless policy** — frozen GRPO-130 0.938 → 0.000 under bias 0.30 and inversion, reacting but never adapting (`x6_pilot_grpo`, to be firmed at n = 300); predicted and tested on π0, GR00T and DreamZero. *New; the slice of (A).*
5. **The lens tells you the fix and what is not a fix** — the copy-trained policy restores ≥ half the loss under unseen faults (Rules E, N), at no nominal cost (Rule 0), while the no-copy fine-tune, the shuffled copy and the forward-model loss without the channel do not (Rule L). *New; the design guidance.*
6. (Supporting) **The lens also gives the measurement tools:** an external comparator that detects overrides online at TPR 0.75 / FPR 0.02 (`x4_report.md`), and the bound on one-shot label-free detection (EC0 V_true ≤ 0.51) that explains why the policy needs a history rather than a flag.

**Dropped:** novelty-gated compensation as a mechanism; "the comparison is cheap, the policy never buys it" (Contribution 3); "RL narrows the state distribution" as an RL claim; EC1 as a headline (it stays as the teacher and the ceiling); any efference-copy "score" that ranks training methods.

---

## 5. Budget and timeline (rentals; Modal only for what is already there)

| tier | contents | GPU cost `[estimate]` | engineering | what it buys |
|---|---|---|---|---|
| **$0 first** | generic fault-evaluation harness over `BasePolicy` (capture off); the `rlinf`-class proprio wrapper; pre-registration of §2.3's prediction and §3.1's 2 × 2; HF checks: DreamZero inference speed and context handling, released RL siblings of π0/π0.5, a WoVR-trained OFT checkpoint, Evo-1's state input | $0 | 1–2 weeks | the plan, and whether DreamZero and WoVR are testable |
| **minimal, ≈ $15–25** | frozen GRPO-130 fault curve at n = 300 (3 arms, $0.3); training collection 1,200 episodes ($0.6); fine-tunes A, B, C, D, E ($0.7 each on an A100/A800, ≈ 45–60 min `[inference]`), 2 seeds on A and B (+$1.4); ≈ 18 evaluations ($0.2 each); π0 fault curve ($0.8); retries and egress | $12–18, call it **$15–25** with failures | 3–4 weeks | claims 4 and 5 on one base and one extra architecture; the 2 × 2 |
| **solid, ≈ $50–80** | + GR00T N1.5 and starVLA fault curves (+$2); + DreamZero fault curve via SGLang (+$3–6, 1–2 GPUs `[TO VERIFY]`); + Base-Lora as a second base for A/B/D (+$5); + gain-loss family (+$3); + third seed (+$3); + specialists for the efficiency claim (+$2.5); + probe ladders on π0 and GR00T with capture adapters (+$2); + transient-hijack evaluation of A vs F (+$0.5); + WoVR OFT if released (+$1.5) | **$50–80** | 6–8 weeks | claim 4 across four architectures including a world-model policy; claim 5 replicated on two bases; the efficiency claim; the appendix probes |
| **ambitious, ≥ $300** | GRPO post-training from A vs B in a fault-randomised environment (sample-efficiency curves; multi-GPU, $100–300 per run); a second simulator (ManiSkill, which the repo supports); a Franka bias/inversion demonstration | ≥ $300 | 3+ months | the RL-efficiency sentence and a real-robot figure |

Hardware notes: training on an A100/A800-80 GB (batch 16 bf16 without checkpointing); collections on the same card or a 4090 only where the box has ≥ 64 GB RAM (the 4090's 24 GB VRAM clears the 25 GB measured at 10 envs only with fewer envs — use 8); DreamZero's evaluation path runs through an SGLang server with its own dependency stack (`dreamzero_sglang.rst`; the docs note that the Cosmos3 model dependencies conflict with that stack, `cosmos3_sglang.rst`), so give it its own box rather than the training box. Modal's $2.6 covers nothing beyond one more smoke; keep it for emergencies.

---

## 6. Risks, biggest first

| risk | what it would do | mitigation | cost |
|---|---|---|---|
| **B (no copy, faults randomised) passes bias by hedging**, narrowing A − B | the design guidance becomes "train under faults", which is not new | inversion is the discriminator (no hedge solves a sign flip without the copy); Rule N on both families; report B's reaction trace | in the design |
| **A degrades nominal behaviour or copies its last command** (consecutive-chunk correlation 0.60–0.69 across x/y/z on both checkpoints, logbook step 1 / `identify.json`) | Rule 0 fails; the fix costs more than it gives | 30 % clean data; arm C; LoRA and a short schedule; Rule 0 read with its CI | in the design |
| **The `rlinf`-class wrapper cannot host the token cleanly** (the class builds embeddings from vision and text only) | fall back to the `official` class for all arms, re-running F through it | the 93.6 % zero-token agreement already shows the official path tolerates the token; the wrapper is a `_build_embedding` override | 1–2 days |
| **DreamZero evaluation is infeasible on rentals** (SGLang install, 1–2 GPUs, unknown speed) | claim 4 loses its world-model row | keep it in the solid tier; the lens's prediction stands on π0 and GR00T; say DreamZero is untested rather than guess | — |
| **The $0.30 bias is "unrealistic"** (reviewer) | the premise is contrived | report the full ladder (0.15 already costs 0.38 of success); cite miscalibration magnitudes `[TO VERIFY in the literature]`; inversion as the mis-wiring case | writing |
| **n = 16 pilot numbers move at n = 300** | magnitudes change | the 300-episode frozen curve is the first run of every tier | $0.3 |
| **Cosmos3 or a context-keeping DreamZero turns out to carry the relation internally** | the categorical prediction fails for one model | that would be the most interesting result in the paper: run the probe ladder on it; the lens is falsifiable, say so | +$2 |

The biggest risk is the first: that fault randomisation alone buys most of the tolerance on bias. The design already contains the answer (inversion, Rule N on both families), and the honest fallback — "domain randomisation restores bias tolerance; only the copy restores inversion" — is still a design-guidance result, smaller.

---

## 7. What the brief says that the files support only in part

- **"Models trained with world-model objectives should carry the consequences of their own actions."** In their *generated* rollout, yes; in the forward pass that sees the *executed* outcome, only if the architecture re-conditions on its own past prediction. For the DreamZero policy as configured (`state_horizon: 1`, joint generation from the current observation) that is `[TO VERIFY]`; Cosmos3 is untestable here (no LIBERO checkpoint; 8 × A800 SFT).
- **"Efference-copy probes become a way to compare training methods."** Only across architectures that differ in whether the previous command reaches the forward pass; among the LIBERO checkpoints in this repo none does, so the comparison is predicted to tie at zero.
- **"Collection needs ≥ 64 GB RAM and ≈ 25 GB VRAM at 10 envs."** Consistent with the X4 ledger lines (peak RSS 59–63 GB, peak VRAM 21.7–23.1 GB on the smokes at 10 envs).
- **"≈ $10–15 now buys what $27–34 bought on Modal."** Consistent with the rates (0.2–0.3 × Modal's price per GPU-hour), provided the rental box has the RAM.
- **"X6 stage 0: K = 2 suffices for bias 0.30 and inversion."** `identify.json`: bias 0.30 median z 4.28 with 100 % of episodes at z ≥ 3 at K = 2; inversion 17.5. Bias 0.15 needs K = 4 (z 3.52, 76 %); gain 0.6 is a boundary case (59–65 %).

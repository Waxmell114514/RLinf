# RMA, privileged teacher–student training, and history-based adaptation: implications for a fault-adaptive VLA

Context: OpenVLA-OFT 7B, LIBERO-goal, behavior-cloning LoRA fine-tune on its own rollouts with oracle-compensated targets on faulted dimensions. One extra "efference copy" token carries the last 2 action chunks and the measured state deltas. Results: clean success 0.963 -> 0.863, bias 0.013 -> 0.327, inversion 0.02.

Note on sourcing: items marked "(not re-fetched)" come from well-known abstracts or my prior knowledge, with the canonical URL given. They were not re-verified in this session, so the report writer should treat their numbers more cautiously.

## Q1. RMA (Kumar et al. 2021) and successors: extrinsics encoder, adaptation module, faults/payloads, nominal cost, history length, latency

### Takeaway
RMA splits training into two phases. Phase 1 trains the base policy jointly with an encoder μ, using RL and the true environment vector e_t, which gives an 8-D latent z. Phase 2 freezes the policy and regresses z from 0.5 s of state–action history by MSE, on the policy's own rollouts. In simulation, RMA with the estimated z nearly matches the oracle-z "Expert" (73.5% vs 76.2% success). A policy that conditions on z but gets no live estimate drops to 52.1%. The papers report no quantitative nominal-condition cost.

### Cited Findings
- **Citation.** Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik, "RMA: Rapid Motor Adaptation for Legged Robots," RSS 2021, arXiv:2107.04034 — [arXiv](https://arxiv.org/abs/2107.04034)
- **Privileged vector and latent.** e_t is 17-D: mass and its position (3), motor strength (12), friction (1), local terrain height (1). The encoder μ compresses it to an 8-D extrinsics vector z_t — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Randomization ranges (train / test):**
  - friction [0.05, 4.5] / [0.04, 6.0]
  - payload [0, 6] / [0, 7] kg
  - motor strength [0.90, 1.10] / [0.88, 1.22]
  - Kp [50, 60] / [45, 65]; Kd [0.4, 0.8] / [0.3, 0.9]
  - CoM ±0.15 / ±0.18
  - per-step resample probability 0.004 / 0.01

  "Motor faults" here are only ±10% strength scaling. There is no sign reversal — [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Adaptation module.**
  - History: k = 50 steps (0.5 s) of states and actions.
  - Architecture: a 2-layer MLP embeds each step to 32-D, then a 3-layer 1-D CNN (kernel/stride [8,4], [5,1], [5,1]), then a linear projection to ẑ.
  - Loss: MSE ‖ẑ_t − z_t‖², Adam lr 5e-4 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Phase 2 is on-policy.** The base policy is unrolled with ẑ from the current adaptation module, not with ground-truth z, and each visited history is labelled with the true z_t. The module therefore trains on the state distribution its own imperfect estimates induce, which makes it DAgger-like. Budget: 1000 iterations, batch 80k, about 3 h, about 80M simulation steps — [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Latency.** The base policy runs at 100 Hz and the adaptation module asynchronously at 10 Hz; the policy uses the latest ẑ. The authors report adaptation in under 1 s ("fractions of a second"), against 4–8 min of data for the Peng et al. 2020 real-world adaptation baseline — [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Simulation results (Table II; 3 policies × 1000 episodes; success % / TTF / reward):**

  | Method | Success (%) | TTF | Reward |
  |---|---|---|---|
  | Robust (domain randomization, no z) | 62.4 | 0.80 | 4.62 |
  | SysID | 56.5 | 0.74 | 4.82 |
  | AWR (40k adaptation samples) | 41.7 | 0.65 | 4.17 |
  | RMA w/o adapt | 52.1 | 0.75 | 4.72 |
  | RMA | 73.5 | 0.85 | 5.22 |
  | Expert (true z) | 76.2 | 0.86 | 5.23 |

  RMA also has lower torque (500 vs 527.6) and better smoothness (92.9 vs 122.5) than Robust — [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Real world (A1 robot).**
  - Payload: RMA carries up to 12 kg, about 100% of body weight. The stock A1 controller degrades at 8 kg. RMA without adaptation does not move above 8 kg.
  - Oily surface: 90% success.
  - 15 cm step-down: 80% success (5 trials) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Nominal cost.** No quantitative clean-condition comparison is reported. The paper only says qualitatively that the base policy transfers to wood or concrete — [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **A-RMA (Kumar, Li, Zeng, Pathak, Sreenath, Malik, arXiv:2205.15299, 2022, Cassie biped).** Standard RMA assumes a near-perfect ẑ, and the base policy "can perform poorly" when the estimator is imperfect. A-RMA adds a third phase that fine-tunes the base policy with model-free RL while it consumes the imperfect ẑ. The abstract gives no numbers — [arXiv](https://arxiv.org/abs/2205.15299)

### Inferences
- The Expert-to-RMA gap is small (about 3 points of success), so phase-2 regression recovers most of the oracle's value. RMA w/o adapt (52.1) falls *below* Robust (62.4). A z-conditioned policy fed a wrong or stale z is worse than a policy that never relied on z. For a VLA, this means an oracle-conditioned policy must never see a badly wrong ẑ at deployment unless it has been robustified to estimator error (A-RMA phase 3).
- History lengths that work in RMA-family papers run from 0.5 s to 2 s of dense, high-rate proprioception (50–100 steps). The VLA's efference token holds 2 chunks, i.e. about 16 low-rate end-effector actions plus state deltas. That is shorter in decisions but may be informative enough for a 2-D bias. Identifiability is a property of the signal, not of the step count.

### Gaps
- No RMA paper found tests actuator sign reversal. "Motor strength" stays in [0.88, 1.22], always positive.
- No published nominal-only (clean) metric for RMA vs Robust.
- A-RMA's quantitative gains were not retrieved; only the abstract was read.

## Q2. Manipulation analogues (Hora / in-hand RMA, DexterityGen, Nagabandi, PEARL) and use with VLAs or large pretrained policies

### Takeaway
In-hand rotation ("Hora," Qi et al. CoRL 2022) carries the RMA recipe to manipulation unchanged: an 8-D z from a 9-D privileged object vector, 30 steps (1.5 s) of joint positions and actions, and MSE regression with π frozen. The student reaches 95% of the oracle's rotation reward. Model-based meta-RL (Nagabandi) adapts from 16–32 recent steps. In the 2024–2026 VLA literature I found history-aware and test-time-adaptive VLAs, but none that apply an RMA-style privileged latent to actuation faults.

### Cited Findings
- **Hora citation.** Haozhi Qi, Ashish Kumar, Roberto Calandra, Yi Ma, Jitendra Malik, "In-Hand Object Rotation via Rapid Motor Adaptation," CoRL 2022, arXiv:2210.04887 — [arXiv](https://arxiv.org/abs/2210.04887)
- **Hora design.**
  - Privileged input: a 9-D e_t (object position, size, mass, friction, center of mass; scale is also randomized), encoded to an 8-D z_t.
  - Adaptation input: 30 timesteps (1.5 s) of joint positions and actions.
  - History ablation: T = 10, 20, 30, saturating at T = 30.
  - Architecture: per-step MLP [32, 32], then three 1-D temporal convolutions (kernels 9, 5, 5; strides 2, 1, 1), then a linear layer.
  - Training: MSE, Adam 3e-4; π is frozen and data comes from rolling π out with the predicted ẑ — [arXiv HTML](https://arxiv.org/html/2210.04887)
- **Hora simulation, in-distribution (rotation reward / TTF):**

  | Method | Rotation reward | TTF |
  |---|---|---|
  | Expert (oracle z) | 233.7 | 0.85 |
  | Ours (adaptation) | 222.3 | 0.82 |
  | DR (no privileged input) | 176.1 | 0.81 |
  | SysID | 174.4 | 0.81 |
  | NoAdapt (z estimated once, then frozen) | 90.9 | 0.65 |
  | Periodic | 43.6 | 0.44 |

  Out of distribution: Expert 165.1, Ours 160.6, DR 140.8 — [arXiv HTML](https://arxiv.org/html/2210.04887)
- **Hora real world, heavy objects (rotations in radians / TTF):** Ours 23.96 / 0.98, SysID 10.36 / 0.61, DR 9.67 / 0.72, NoAdapt N.A. / 0.35 — [arXiv HTML](https://arxiv.org/html/2210.04887)
- **Nagabandi et al.** Anusha Nagabandi, Ignasi Clavera, Simin Liu, Ronald Fearing, Pieter Abbeel, Sergey Levine, Chelsea Finn, "Learning to Adapt in Dynamic, Real-World Environments through Meta-Reinforcement Learning," arXiv:1803.11347, published at ICLR 2019 (venue from prior knowledge, not re-fetched).
  - Methods: model-based meta-RL in two variants. GrBAL takes a MAML gradient step on the dynamics model from recent data. ReBAL uses a recurrent learned update.
  - Adaptation window and horizon: M = 16–32 past steps and K = 10–32 (half-cheetah 32 / 32; ant 16 / 10; 7-DoF arm 16 / 32).
  - Perturbations: disabled joints, crippled leg, slopes, a floating pier; on a real millirobot, missing leg, payload, pose miscalibration.
  - On seen real terrains GrBAL ≈ MB (carpet left turn 4.07 vs 3.94), i.e. no nominal penalty. It is substantially better under unseen conditions, and uses about 1000× less data than model-free methods — [ar5iv](https://ar5iv.labs.arxiv.org/html/1803.11347)
- **PEARL.** Rakelly, Zhou, Quillen, Finn, Levine, ICML 2019, arXiv:1903.08254. A probabilistic context encoder infers a task latent z from recent transitions, and the actor and critic condition on z. Training is off-policy, with 20–100× better meta-training sample efficiency. Task uncertainty is handled by posterior sampling: z is sampled from the prior when there is no context yet — [arXiv](https://arxiv.org/abs/1903.08254) (not re-fetched)
- **DexterityGen.** Zhao-Heng Yin et al., RSS 2025, arXiv:2502.04307. A generative controller pretrained on simulated RL data and prompted by human teleoperation. It reports 10–100× better holding stability. This is a foundation low-level controller, not a dynamics- or fault-inference module — [arXiv](https://arxiv.org/pdf/2502.04307); [RSS proceedings](https://roboticsproceedings.org/rss21/p103.html)
- **Nearby VLA work (2025–2026), none of which addresses actuation faults:**
  - HAMLET (arXiv:2510.00695, ICLR 2026) makes VLAs history-aware; 76.4% on history-dependent real tasks with GR00T N1.5 — [arXiv](https://arxiv.org/abs/2510.00695)
  - SALT (arXiv:2610.07946) does test-time adaptation to visual disruptions, using the unexecuted remainder of the previous action chunk as self-supervision — [arXiv](https://arxiv.org/abs/2610.07946)
  - VLRLA (arXiv:2610.19517) does in-context RL for deployment-time adaptation with frozen parameters — [arXiv](https://arxiv.org/abs/2610.19517v1)
  - MetaVLA (arXiv:2510.05580) adds an Attentive Neural Process meta-learning module — [arXiv](https://arxiv.org/html/2510.05580v1)
  - PriorVLA (arXiv:2605.10925) freezes a prior action expert and adds a separate adaptation expert — [arXiv](https://arxiv.org/html/2605.10925v1)

  These were found by search and only skimmed at abstract level.

### Inferences
- **NoAdapt is catastrophic.** In Hora, NoAdapt (z estimated once at the start, then frozen) gets 90.9 vs 222.3. That is the manipulation analogue of the VLA's "first call with empty history" problem: committing to an early estimate is far worse than continuous re-estimation, and far worse than a robust DR policy.
- **PriorVLA's frozen-prior split.** Its frozen prior expert plus separate adaptation expert is architecturally the closest VLA analogue to "keep nominal behaviour intact and add a fault pathway."

### Gaps
- I found no paper that applies an RMA or privileged-latent estimator to a large pretrained VLA for actuation or dynamics faults. This appears to be an open niche, though the search was not exhaustive.
- Arm-manipulation RMA variants (e.g. payload-adaptive arm controllers) were not retrieved in this pass.

## Q3. Teacher–student distillation with privileged info (LbC, Lee 2020, asymmetric actor-critic), DAgger, and the imitation gap

### Takeaway
Privileged schemes first solve "act given full information" with a teacher, which makes targets unambiguous. A student then imitates on its own state distribution using DAgger. Lee et al. add a latent-regression term, which helps on the hardest terrain. When the student genuinely cannot infer what the teacher sees, pure imitation fails; this is the "imitation gap," and the fixes either mix in RL (ADVISOR) or constrain the teacher (SITT).

### Cited Findings
- **Learning by Cheating.** Dian Chen, Brady Zhou, Vladlen Koltun, Philipp Krähenbühl, CoRL 2019, arXiv:1912.12294. A privileged agent sees the ground-truth map and the positions of all traffic participants, then teaches a vision-only student. The approach reports 100% success on all tasks of the original CARLA benchmark and about 10× fewer infractions on NoCrash — [arXiv](https://arxiv.org/abs/1912.12294); [CoRL listing](https://mlanthology.org/corl/2019/chen2019corl-learning)
- **Lee et al. citation.** Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter, "Learning quadrupedal locomotion over challenging terrain," Science Robotics 2020, arXiv:2010.11251 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.11251)
- **Lee et al. method.**
  - Teacher: sees terrain heights (9 points per foot), contacts and forces, friction, and external forces. An MLP encodes these to a latent l_t. Trained with TRPO for about 12 h.
  - Student: a TCN over N proprioceptive steps at 0.02 s per step; the deployed model uses N = 100 (2 s).
  - Student training: DAgger (student rolls out, teacher labels), about 4 h.
  - Loss: action imitation² + latent regression² — [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.11251)
- **Lee et al. ablations.**
  - Action-only imitation, without the latent loss, is comparable on slopes and disturbances but has lower success on steps.
  - A TCN-20 student trained directly by RL, without privileged training, fails to learn to balance or locomote.
  - Longer memory matters on steps but not slopes; TCN-100 deviates 35.5% less than TCN-1 under a 50 N push.
  - Real-world payload test (10 kg, never simulated): the robot crosses steps up to 13.4 cm — [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.11251)
- **Asymmetric actor-critic.** Pinto, Andrychowicz, Welinder, Zaremba, Abbeel, arXiv:1710.06542 (RSS 2018). The critic sees full state and the actor sees images only. This puts privileged information into the value or advantage estimate without asking the actor to imitate unobservable behaviour — [arXiv](https://arxiv.org/abs/1710.06542) (not re-fetched)
- **Imitation gap / ADVISOR.** Weihs et al., "Bridging the Imitation Gap by Adaptive Insubordination," arXiv:2007.12173 (NeurIPS 2021). When the teacher uses information the student cannot observe, imitation yields a suboptimal or incoherent student. ADVISOR weights the imitation and RL losses per state, using an auxiliary imitation head. It was evaluated on 13 tasks designed to probe the gap — [arXiv](https://arxiv.org/pdf/2007.12173)
- **Student-Informed Teacher Training (SITT).** Messikommer et al., ICLR 2025, arXiv:2412.09149. It penalizes teacher actions the student cannot reproduce and aligns the teacher with the student. In a maze, an unaligned teacher's optimal path cannot be imitated, while the aligned teacher picks a route the student can copy — [arXiv](https://arxiv.org/html/2412.09149v2); [project page](https://rpg.ifi.uzh.ch/sitt/index.html)

### Inferences
- **The VLA setup is a student trained straight on teacher labels.** The oracle's compensating command, which knows the fault, labels each state, but there is no intermediate oracle-conditioned policy and no latent target. At an empty-history call the label depends on unobservable information, which is exactly ADVISOR's imitation gap. With MSE or cross-entropy over 256 bins, the BC target becomes a mixture over faults. For a bias it averages toward roughly zero offset; for a sign flip the two modes ±a average to about 0 or split probability mass. That plausibly explains bias partially learned and inversion never learned.
- **What the separation buys.** Lee/RMA-style separation gives the student a regression target z that is a deterministic function of the episode's fault, so it is learnable. It also hands the "how to act" part to a policy that always has z in its input, so its targets are never ambiguous. Ambiguity remains only in the estimator, where MSE toward the posterior mean is the Bayes-correct answer under uncertainty, and the conditioned policy can be trained to handle the uncertain-z regime explicitly.

### Gaps
- Quantitative teacher-vs-student numbers for Lee 2020 appear only in figures (Fig. 5) and were not extracted.
- LbC's per-benchmark tables were not extracted.

## Q4. Sign flips (inversion / mirror) vs offsets: any results on actuator reversal?

### Takeaway
Sign inversion is qualitatively harder than an offset. The correct response is the negated policy, and no interpolation between the two contexts is valid. A single network that averages over contexts produces incoherent behaviour. Recent work (DMA\*-SH, 2026) treats inversion as a discrete context and modulates the network multiplicatively through a hypernetwork. Human motor-learning studies show that error-driven (gradient-like) adaptation actively diverges under mirror reversal, and that compensation needs "de novo" learning of a separate policy.

### Cited Findings
- **DMA\*-SH citation.** Benad, Banerjee, Röder, Ay, Butz, Eppe, "Dynamics-Aligned Shared Hypernetworks for Contextual RL under Discontinuous Shifts," arXiv:2602.06550 (Feb 2026, revised May 2026, preprint) — [arXiv](https://arxiv.org/abs/2602.06550)
- **DMA\*-SH problem framing.** It identifies a failure mode in which the latent context discontinuously changes how actions affect the environment, so contexts require "incompatible control responses." It introduces an Actuator Inversion Benchmark (AIB) that also includes actuator permutations — [arXiv](https://arxiv.org/abs/2602.06550)
- **DMA\*-SH method and results.** A hypernetwork trained only through dynamics prediction generates adapter weights shared by the dynamics model, policy, and Q-function. Input normalization and random input masking stabilize context inference. On held-out AIB tasks it beats domain randomization by 58.1% and a standard context-aware baseline by 11.5% on average. The abstract does not say whether these are relative or absolute — [arXiv](https://arxiv.org/abs/2602.06550)
- **Hadjiosif, Krakauer, Haith, J. Neurosci. 2021;41(12):2747–2761.** Under mirror reversal, implicit adaptation "acts to amplify rather than to reduce errors": on-axis error rose from 2.5° to 17.8°. This is better fit by direct policy-gradient-like updating than by forward-model learning. Compensating for mirror reversal requires de novo learning of a new policy — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8018745)
- **AcroRL (arXiv:2605.24301).** Quadrotor flight with bidirectional (reversible) thrust. It trains separate PPO policies per regime and includes action history in observations so the policy can infer actuator effects — [arXiv](https://arxiv.org/pdf/2605.24301) (abstract-level only)

### Inferences
- **Additive conditioning can't flip a sign.** For a pretrained VLA, a single extra token can shift outputs additively but struggles to implement output = −f(obs) on one axis. A multiplicative or gated pathway is a better match: a per-dimension sign or gain head, or a hypernetwork / FiLM over the action head driven by z.
- **Fold the inversion out of the policy.** The simplest analytic option for inversion is to predict s ∈ {+1, −1} per axis, and a bias b, from history. The nominal VLA keeps acting, and the actuation command is corrected as a' = s ⊙ a − b, i.e. an explicit inverse fault model. This needs no policy relearning for the parametric fault families used here.
- **Inversion is detectable from the efference copy.** Its signature is a negative correlation between commanded and measured Δ on an axis. One chunk with non-trivial motion on that axis should suffice; a stationary first call cannot be disambiguated.

### Gaps
- I found no RMA-style legged or manipulation paper that tests actuator *reversal*; DMA\*-SH is the closest.
- The underlying DMA\*-SH metrics were not retrieved.

## Q5. Concrete recipe implications (oracle-latent conditioning, separate estimator, z=0 for clean, history length, loss choices)

### Takeaway
The literature consistently supports a two-phase recipe:

1. Train the policy conditioned on the true fault parameters, so the targets are unambiguous.
2. Train a separate estimator from history on the policy's own rollouts (DAgger / on-policy), with MSE to the true latent.
3. Optionally fine-tune the policy on the estimator's outputs (A-RMA) so it tolerates estimator error.

Sign faults likely need a multiplicative or discrete pathway rather than an additive latent token.

### Cited Findings
- **Oracle-conditioned policy first, then on-policy regression.** RMA and Hora use privileged z for policy training, then on-policy MSE regression of z with π frozen. The student reaches 96–97% of the oracle in RMA (73.5 vs 76.2 success) and 95% in Hora (222.3 vs 233.7 rotation reward) — [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034); [Hora](https://arxiv.org/html/2210.04887)
- **On-policy collection with the student's own estimate is part of the recipe** (RMA phase 2; Lee 2020's DAgger) — [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034); [Lee](https://ar5iv.labs.arxiv.org/html/2010.11251)
- **Fine-tune on the imperfect estimate.** An imperfect estimator hurts a policy trained only on the true z; A-RMA fine-tunes the policy with ẑ in the loop — [A-RMA](https://arxiv.org/abs/2205.15299)
- **Never freeze an early estimate.** In Hora, NoAdapt (frozen early estimate) scores 90.9 vs 222.3 for continuous estimation; in RMA, w/o-adapt scores 52.1 vs 73.5 and even falls below Robust's 62.4 — [Hora](https://arxiv.org/html/2210.04887); [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **History lengths used:**

  | Paper | History |
  |---|---|
  | RMA | 50 steps (0.5 s at 100 Hz) |
  | Hora | 30 steps (1.5 s), saturating at 30 in ablation |
  | Lee 2020 | 100 steps (2 s) |
  | Nagabandi | 16–32 steps |

  Sources: [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034); [Hora](https://arxiv.org/html/2210.04887); [Lee](https://ar5iv.labs.arxiv.org/html/2010.11251); [Nagabandi](https://ar5iv.labs.arxiv.org/html/1803.11347)
- **Loss choices.** RMA and Hora use pure latent MSE. Lee uses action imitation plus latent MSE, and the latent term helps on the hardest terrain. DMA\*-SH trains its context module only through dynamics prediction, i.e. forward-model loss rather than latent regression — [Lee](https://ar5iv.labs.arxiv.org/html/2010.11251); [DMA\*-SH](https://arxiv.org/abs/2602.06550)
- **Nominal preservation.** Nagabandi's GrBAL matches the non-adaptive model on seen terrains (4.07 vs 3.94), so adaptation did not cost nominal performance. RMA reports lower torque and smoother gait than Robust, but no clean-only success numbers — [Nagabandi](https://ar5iv.labs.arxiv.org/html/1803.11347); [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034)
- **Uncertainty at zero context.** PEARL samples z from the prior when no context exists (posterior sampling) — [PEARL](https://arxiv.org/abs/1903.08254) (not re-fetched)

### Inferences (recipe for the VLA, derived from the above, not tested)
1. **Phase 1: oracle-conditioned VLA.** Feed the true fault parameters through the extra token or a FiLM head: bias b ∈ R² (scaled), and the sign vector s ∈ {±1}^d as separate binary features. Clean episodes get the explicit null fault: b = 0, s = +1, plus a "known clean" flag. With z given, the BC target is a deterministic function of (obs, z), which removes the first-call ambiguity. Including many clean episodes with z = null should pin down nominal behaviour; the 0.963 → 0.863 clean drop is consistent with clean states receiving averaged, fault-contaminated targets in the current recipe.
2. **Phase 2: separate small estimator.** Use a TCN or MLP over the efference-copy history: commanded chunk, measured Δ state, and their per-axis correlation. Train it on rollouts of the phase-1 policy driven by ẑ (DAgger), with MSE on b and BCE on the sign bits; heteroscedastic or Gaussian NLL would also expose uncertainty. At empty history, output the prior: b = 0 and P(s = −1) = prior. The policy then behaves nominally until evidence arrives, which matches the PEARL/RMA behaviour of "act on the prior, re-estimate continuously."
3. **Phase 3, optional (A-RMA style).** Briefly fine-tune the LoRA policy with ẑ in the loop, so it degrades gracefully on wrong estimates. Alternatively, randomize z noise during phase 1.
4. **Inversion.** Prefer a multiplicative pathway: a per-axis sign gate on the decoded action, or a hypernetwork / FiLM adapter, per DMA\*-SH. An additive token is unlikely to produce a sign flip in a 256-bin tokenized head. The cheapest baseline is an explicit inverse-fault wrapper outside the VLA (a' = s ⊙ a − b) using the estimator. This upper-bounds what the estimator alone can buy, and leaves the nominal VLA untouched, so clean success stays ≈ 0.963 by construction when ẑ = null.
5. **History.** Two chunks (about 16 actions) is short compared with RMA's 50 high-rate steps, but bias and sign are low-dimensional. The main constraint is excitation: there must be non-trivial motion on the faulted axis. Consider estimating after the first executed chunk and re-querying, rather than at the first call.

### Gaps
- No source directly quantifies the "z = 0 clean flag preserves nominal" claim. It is a design inference.
- No source reports estimator losses for discrete sign variables in RMA-style work; BCE is a suggestion.
- No paper was found with numbers for 7B VLAs under this two-phase scheme.

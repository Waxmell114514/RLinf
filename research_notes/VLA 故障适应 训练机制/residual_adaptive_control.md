# Residual policies on frozen base policies, and adaptive control / disturbance observers wrapped around learned policies, for fault compensation that preserves nominal behavior by construction

Scope: (a) residual policy learning on frozen base policies, including frozen VLAs; (b) L1 adaptive control, disturbance observers (DOB), history-based latent regression (RMA) and Nussbaum / unknown-control-direction methods combined with learned controllers; (c) actuator fault-tolerant control (FTC). Context: frozen OpenVLA-OFT (LIBERO, EE delta actions, 8-step chunks), persistent 2-D additive bias and axis sign inversion. Full LoRA fine-tuning with a history token took clean success from 0.963 to 0.863 and recovered about 1/3 of the bias loss. Inversion was never learned.

Source caveat: most numbers below come from arXiv HTML full texts read through a summarizing fetch tool. Where a paper reports results only in figures, these notes say so rather than reading values off plots.

---

## Q1. Residual RL and residual policy learning on frozen base policies and VLAs: how is the residual bounded, and what nominal numbers are reported?

### Takeaway
Every residual-on-frozen-base method found keeps the base weights frozen and adds a bounded correction, a = a_base + a_res. They differ in how strongly they guarantee "zero change at the start":
- RPL zero-initializes the last layer, so the composite policy starts exactly equal to the base.
- ResiP uses a small-gain last layer and a 0.1 action scale.
- Policy Decorator and PLD bound the residual with tanh·α or [−ξ, ξ] but initialize it randomly. They rely on a probabilistic ramp-in (Policy Decorator) or a base-only warm-up (PLD).

Even so, several of these papers report an early performance dip. "Preserving nominal by construction" therefore needs zero-init plus a bound plus gating, not just a frozen base. None of the papers measures clean-condition retention as a separate metric. They report task improvement on the same distribution the base was trained on.

### Cited Findings
- **Residual Policy Learning (Silver, Allen, Tenenbaum, Kaelbling, MIT, arXiv 1812.06298, Dec 2018).**
  - The residual network's last layer is zero-initialized ("We do this by initializing the last layer of the network to be zero"), so the composite policy initially equals the base controller. — [arXiv 1812.06298](https://arxiv.org/html/1812.06298)
  - The critic is trained alone (policy fixed) until its loss falls below β = 1.0. This prevents a bad critic from degrading a good base. — [arXiv 1812.06298](https://arxiv.org/html/1812.06298)
  - Performance still dips early. Lowering β reduced the dip without significantly changing convergence time. — [arXiv 1812.06298](https://arxiv.org/html/1812.06298)
  - Tasks include SlipperyPush (friction 0.18 vs 1.0), NoisyHook (observation noise), ComplexHook (unobserved bumps), and a "miscalibrated controller" (gains artificially increased, causing oscillations). These are the closest analogues to actuation faults in this literature. — [arXiv 1812.06298](https://arxiv.org/html/1812.06298)
  - Success rates, base → RPL:

    | Task | Base | RPL | Scratch RL |
    |---|---|---|---|
    | Push | ≈0.5 | 0.9 with ~10x fewer samples than scratch | ≈1.0 after ~2M steps |
    | PickAndPlace | ≈0.5 | 1.0 after ~1M steps (~10x faster) | slower |
    | NoisyHook | ≈0.15 | ≈0.8 | never nontrivial |
    | ComplexHook | ≈0.55 | just below 0.8 | never nontrivial |

    — [arXiv 1812.06298](https://arxiv.org/html/1812.06298)
- **Residual RL for Robot Control (Johannink et al., Siemens and UC Berkeley, arXiv 1812.03201, Dec 2018; usually cited as ICRA 2019, venue not verified here).** Control is split into a conventional hand-designed feedback controller plus an RL residual. It was demonstrated on real block assembly with contacts and unstable objects, with faster learning than pure RL in sim and real. — [arXiv 1812.03201](https://arxiv.org/abs/1812.03201v2); [DeepAI summary](https://www.deepai.org/publication/residual-reinforcement-learning-for-robot-control)
- **Policy Decorator (Yuan et al., arXiv 2412.13630, Dec 2024).**
  - Base policies (Behavior Transformer, Diffusion Policy) stay frozen and only the residual is trained (SAC). — [arXiv 2412.13630](https://arxiv.org/html/2412.13630)
  - The residual is tanh-squashed and scaled by α, so a_res ∈ (−α, α). α is "typically set close to the action scale observed in the demonstration dataset." — [arXiv 2412.13630](https://arxiv.org/html/2412.13630)
  - Per-task α ranged from 0.03 (StackCube, BeT) to 0.8 (Pen, DP visual). Most values are 0.1–0.3. — [arXiv 2412.13630](https://arxiv.org/html/2412.13630)
  - "Progressive exploration": with probability ε the agent executes base + residual, otherwise base only. ε rises linearly 0→1 over H steps, with H from 30K to 8M. Too small an H "can cause complete failure from aggressive exploration." — [arXiv 2412.13630](https://arxiv.org/html/2412.13630)
  - The residual is randomly initialized; zero-init is not stated. — [arXiv 2412.13630](https://arxiv.org/html/2412.13630)
  - α ablation: too small hurts final performance, too large hurts sample efficiency. 0.1–0.5 (PushChair) and 0.03–0.1 (StackCube) converge to similar final success. — [arXiv 2412.13630](https://arxiv.org/html/2412.13630)
  - Results are reported as reaching "a near-perfect level on all tasks" with state observations. Exact numbers are only in figures. Smoother behavior than pure RL is supported only qualitatively (videos). — [arXiv 2412.13630](https://arxiv.org/html/2412.13630)
- **ResiP, "From Imitation to Refinement" (Ankile et al., arXiv 2407.16677, 2024).**
  - Base: a frozen chunked Diffusion Policy (66M U-Net) that predicts Ta = 32 and executes Texec = 8 actions at about 10 Hz. — [arXiv 2407.16677](https://arxiv.org/html/2407.16677)
  - Residual: a small Gaussian MLP trained with PPO that runs at every control step. This makes it closed-loop inside the open-loop chunk. — [arXiv 2407.16677](https://arxiv.org/html/2407.16677)
  - The residual's input is state concatenated with the base action for that step, and a = a_base + a_res with residual scale 0.1. The final layer has orthogonal init with a small gain, giving near-zero initial corrections. — [arXiv 2407.16677](https://arxiv.org/html/2407.16677)
  - The authors argue that chunking makes BC behave like an open-loop planner, which the per-step residual fixes. — [arXiv 2407.16677](https://arxiv.org/abs/2407.16677)
  - PPO settings: 1024 parallel envs, up to 500M env steps, γ = 0.999. — [arXiv 2407.16677](https://arxiv.org/html/2407.16677)
  - Simulation success (Table I, %, low/med randomness), base DP → ResiP:

    | Task | Base DP | ResiP |
    |---|---|---|
    | one_leg | 54/26 | 98/76 |
    | round_table | 12/4 | 94/77 |
    | lamp | 7/2 | 97/70 |
    | mug-rack | 26 | 88 |
    | peg-in-hole | 5 | 99 |
    | biman-insert | 33 | 93 |

    — [arXiv 2407.16677](https://arxiv.org/html/2407.16677)
  - Real world (one_leg, distilled sim+real): about 50–60% vs 20–30% real-only. — [arXiv 2407.16677](https://arxiv.org/html/2407.16677)
- **ResFiT, "Residual Off-Policy RL for Finetuning BC Policies" (arXiv 2509.19301, 2025).**
  - Treats a BC action-chunking base as a black box and learns per-step residuals with off-policy RL from sparse binary rewards. — [arXiv 2509.19301](https://arxiv.org/abs/2509.19301)
  - Details: n-step returns with n = 3, LayerNorm in the critic, and UTD > 1 (4 captures most of the benefit). It uses randomized ensembled double-Q and 50/50 sampling between demos and online data. The text gives no residual bound or init details. — [arXiv 2509.19301](https://arxiv.org/html/2509.19301v2)
  - BoxCleanup converges in about 200k steps vs 40M for PPO-based residual RL (≈200x more sample-efficient). — [arXiv 2509.19301](https://arxiv.org/html/2509.19301v2)
  - Real humanoid (29-DoF Vega, absolute joint position control): WoollyBallPnP 14% → 64% after 134 rollouts (~15 min); PackageHandover 23% → 64% after 343 episodes (~76 min). — [arXiv 2509.19301](https://arxiv.org/html/2509.19301v2)
- **PLD, "Self-Improving VLA Models with Data Generation via Residual RL" (Xiao et al., arXiv 2511.00091, Oct 2025; listed under ICLR 2026 by UT Austin RPL).**
  - Bases: π0 and OpenVLA. The VLA backbone is frozen and a lightweight Gaussian residual actor πδ(·|s, a_b) is trained, giving ā = a_b + a_δ. — [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1); [UT RPL](https://rpl.cs.utexas.edu/publications/2026/04/01/xiao-iclr26-pld/)
  - Bound: a_δ ∈ [−ξ, ξ], with ξ = 0.5 (LIBERO) and 0.1 (SimplerEnv), set by a scheduler. — [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1)
  - Residual and critic: random init (not zero), critic pre-trained with Cal-QL on about 50 successful base rollouts per task, and 100 base-only warm-up episodes. — [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1)
  - "Base-policy probing": the base runs for T_base ~ U[0, αT] steps before the residual takes over (plateau at α = 0.6). The paper notes an initial performance drop during exploration. — [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1)
  - LIBERO results, base → PLD:

    | Base | Spatial | Object | Goal | Avg |
    |---|---|---|---|---|
    | π0 | 95.2 → 97.7 | 97.6 → 98.5 | 87.4 → 95.3 | 93.4 → 97.2 |
    | OpenVLA | 92.9 → 99.5 | 99.1 → 99.1 | 83.25 → 98.9 | 91.8 → 99.2 |

    — [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1)
  - PLD then distills residual-generated data back into the VLA by SFT. Reduced forgetting is hypothesized, not measured. — [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1)
- **DSRL, "Steering Your Diffusion Policy with Latent Space RL" (Wagenmaker et al., CoRL 2025, PMLR v305, arXiv 2506.15799).**
  - RL acts on the input noise w of a frozen diffusion/flow policy, with w in the box [−b_W, b_W]^d (b_W ∈ [1, 3] online). This needs only black-box access to the base. — [arXiv 2506.15799](https://arxiv.org/html/2506.15799v2); [PMLR](https://proceedings.mlr.press/v305/wagenmaker25a.html)
  - π0 on LIBERO: about 20% → about 100% after roughly 10k online samples (approximate values from the text). — [arXiv 2506.15799](https://arxiv.org/html/2506.15799v2)
  - Real Franka with DROID π0: toaster 5/20 → 18/20 (80 episodes); spoon 15/20 → 19/20 (65 episodes). — [arXiv 2506.15799](https://arxiv.org/html/2506.15799v2)
  - ResiP and V-GPS did not significantly improve π0 in their comparison. — [arXiv 2506.15799](https://arxiv.org/html/2506.15799v2)
  - Stated limitation: exploration is bounded by the base distribution, and steerability is not guaranteed. — [arXiv 2506.15799](https://arxiv.org/html/2506.15799v2)
  - RLinf documents a DSRL-π0 example: a SAC agent of about 500K parameters outputs noise while π0 stays frozen. — [RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)

### Inferences
- DSRL-style noise steering can only pick actions inside the base policy's action support. OpenVLA-OFT uses an L1-regression (deterministic) head, not diffusion, so DSRL does not apply directly. More importantly, a faulted plant needs actions outside the nominal support (for example, the negated component under inversion), which latent steering cannot produce. An additive action-space residual is the right family for actuation faults.
- The ResiP lesson carries over directly. With 8-step open-loop chunks, the corrector should run at every control step, conditioned on the base action for that step plus recent history. It should not be one correction per chunk: a sign-flip on one axis drifts within the chunk.
- Nominal-preservation mechanisms in increasing strength:
  1. A bound (α or ξ) limits worst-case damage.
  2. Zero-init (RPL) gives exact equality at step 0.
  3. Probabilistic or probe gating (Policy Decorator ε, PLD T_base) limits damage during training only.
  4. An explicit fault detector gates the correction at deployment. None of the residual-RL papers do this; it comes from the FTC literature (Q3).
- For us, a deployment-time gate (residual = 0 unless the fault statistic exceeds a threshold calibrated on clean logs) is what makes "≥0.96 clean" hold by construction, up to the detector's false-alarm rate.
- The residual bound must cover the fault. For bias b, |a_res| ≥ |b| on the faulted axes. For inversion the needed correction is 2·|a_base| on that axis, so a tight α (0.03–0.1, as in Policy Decorator and ResiP) would cap inversion recovery. A structured correction (described under Q4) avoids this, because the "residual" is a sign multiplier rather than an additive term.

### Gaps
- No paper found evaluates a residual corrector on a frozen VLA under injected actuation faults (bias or inversion). All of them use residuals to raise nominal success.
- None reports clean-task retention as a separate metric with the residual on, after training on a shifted distribution.
- Policy Decorator per-task success numbers are only in figures, not text.
- Johannink et al. quantitative results and venue were not retrieved.
- RLDG and PA-RL were not fetched within the tool budget. RLDG distills RL specialists into a generalist and is not a frozen-base residual. PA-RL optimizes actions from a base policy via a critic. Both are unverified here.

---

## Q2. L1 adaptive control, DOB/ESO, RMA-style latent regression and Nussbaum gains wrapped around learned policies: constant bias and sign reversal

### Takeaway
- **Constant bias.** For a matched fault, one entering through the same channel as the action, the classical answer is an add-on observer loop. It estimates the lumped disturbance from the mismatch between the predicted and measured state change, low-pass filters it, and subtracts it. The learned policy is untouched, and in the nominal case the added term is approximately zero.
  - Cheng et al. (RA-L 2022) show exactly this for RL policies: u = u_RL + u_L1. All policies kept nominal success, and L1 rescued 70% loss-of-effectiveness and disturbance cases.
- **Sign reversal.** This is control-direction uncertainty, which L1 explicitly assumes away (it needs a known sign of Λ). Nussbaum-gain methods handle an unknown but constant sign. They are notorious for huge transients and can go unstable if the sign switches.
- **Practical implication.** For sign inversion, explicit identification (estimate the sign per axis from recent (action, Δstate) pairs, then flip) is the practical route. A hypothesis test with hysteresis over a small discrete set of fault modes is cheaper and safer than Nussbaum adaptation.

### Cited Findings
- **Cheng, Zhao, Wang, Block, Hovakimyan, "Improving the Robustness of RL Policies with L1 Adaptive Control," IEEE RA-L 7(3), 2022 (arXiv 2112.01953; earlier withdrawn version arXiv 2106.02249).** — [arXiv 2112.01953](https://arxiv.org/abs/2112.01953); [arXiv 2106.02249](https://arxiv.org/abs/2106.02249)
  - Model: nominal ẋ = f(x) + g(x)u (known, control-affine, g full column rank). Perturbed: ẋ = f(x) + g(x)Λu + d(t, x).
  - Λ is unknown, strictly row-diagonally dominant, **with known sign on its diagonal**.
  - Only the matched component of σ = g(Λ−I)u + d is compensated; unmatched disturbances are not. — [arXiv 2112.01953](https://arxiv.org/html/2112.01953)
  - Architecture:
    - State predictor: ẋ̂ = F_nom(x, u) + σ̂ − a·x̃.
    - Piecewise-constant adaptive law: σ̂(iT) = −a/(e^{aT}−1)·x̃(iT).
    - Control: u = u_RL + u_L1, with u_L1 = −C(s)σ̂_m and low-pass filter C(s) = K/(s+K).
    - Typical settings: a = 10, T = 0.001–0.005 s, K = 200 (sim); a = 150, K = 100–150 (hardware).
    
    — [arXiv 2112.01953](https://arxiv.org/html/2112.01953)
  - The RL policy is trained on nominal dynamics and left unchanged. The L1 loop makes the perturbed plant behave like the nominal one. — [arXiv 2112.01953](https://arxiv.org/html/2112.01953)
  - Pendubot hardware results:

    | Scenario | PILCO | PILCO+L1 | SAC | SAC+L1 | DR-SAC | DR-SAC+L1 |
    |---|---|---|---|---|---|---|
    | Nominal | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
    | Λ = 0.5 | ✗ | ✔ | ✗ | ✔ | ✔ | ✔ |
    | Λ = 0.5, +450 g | ✗ | ✔ | ✗ | ✗ | ✗ | ✔ |
    | Rubber-band disturbance | ✗ | ✔ | ✗ | ✔ | ✔ | ✔ |

    — [arXiv 2112.01953](https://arxiv.org/html/2112.01953)
  - In sim, cart-pole with L1 tolerated cart mass up to 3 kg (6x nominal). It failed on large pole-mass changes (unmatched). — [arXiv 2112.01953](https://arxiv.org/html/2112.01953)
- **DOB around RL policies.**
  - Kim et al. used a disturbance observer to estimate sim-to-real mismatch as a disturbance and compensate for it, as cited in Cheng et al. — [arXiv 2106.02249](https://arxiv.org/pdf/2106.02249)
  - EIMN-DOB (Park et al., ICCAS 2022) uses an ensemble inverse-model network as the nominal inverse, so no physics model is needed. It was evaluated on SAC policies in MuJoCo. — [CORE](https://core.ac.uk/works/141810016)
  - DiAReL (Malmir et al., 2023) instead feeds disturbance estimates into the policy observation (a disturbance-augmented MDP). This changes the policy and does not preserve it. — [arXiv 2306.09010](https://arxiv.org/html/2306.09010v2)
- **RMA, Rapid Motor Adaptation (Kumar, Fu, Pathak, Malik, RSS 2021, arXiv 2107.04034).**
  - An adaptation module φ is trained by supervised regression to predict the latent extrinsics from the state-action history. It is trainable because the true extrinsics are known in simulation.
  - The key observation: the difference between commanded and actual joint movement depends on the extrinsics.
  - At deployment, φ runs at 10 Hz and the base policy at 100 Hz.
  
  — [arXiv 2107.04034](https://arxiv.org/pdf/2107.04034v1.pdf); [Meta AI blog](https://ai.meta.com/blog/ai-now-enables-robots-to-adapt-rapidly-to-changing-real-world-conditions/)
  - In RMA the base policy is trained to consume the latent, so the base is not frozen-pretrained. A-RMA adds a further PPO fine-tune of the base with φ fixed, to absorb estimation errors. — [arXiv 2205.15299](https://arxiv.org/pdf/2205.15299)
- **Neural-Fly (O'Connell et al., Science Robotics 2022, arXiv 2205.06908).**
  - A learned, wind-invariant basis (DAIML meta-learning, trained on 12 minutes of flight data) is combined with linear coefficients adapted online by a composite adaptation law.
  - Average tracking-error improvement in winds up to 43.6 km/h (12.1 m/s): 66% over nonlinear tracking control, 42% over L1 adaptive control, 35% over INDI.
  
  — [arXiv 2205.06908](https://arxiv.org/abs/2205.06908v2); [Caltech](https://authors.library.caltech.edu/records/q3grb-3vz72/preview/2205.06908.pdf)
- **Unknown control direction and Nussbaum gains.**
  - The Nussbaum function is the standard tool for an unknown sign of the control gain: a self-adapting gain that probes both directions. A backstepping survey notes that disturbances cannot be compensated directly by the Nussbaum gain. — [arXiv 2305.02066](https://arxiv.org/pdf/2305.02066)
  - Nussbaum implementations "often lead to large overshoots when the initial estimates of the control direction are wrong." — [Wang & Guay, extremum-seeking regulator](https://scholars.ln.edu.hk/en/publications/extremum-seeking-regulator-for-a-class-of-nonlinear-systems-with--2/)
  - The Nussbaum design can become unstable when the control direction periodically changes sign. — [as cited in search results, arXiv 2001.11200 context](https://arxiv.org/pdf/2001.11200)
  - Fu & Barmish argue there is "no control over large excursions," report a 300,000% peak overshoot in one example, and note sensitivity to measurement noise. They propose switching-based alternatives. — [Fu & Barmish](https://www.eng.newcastle.edu.au/%7emf140/home/Papers/FuBarmish.pdf)
  - Alternatives include extremum-seeking or Lie-bracket designs (less overshoot, slower convergence) and logic-based switching. — [arXiv 2109.13559](https://arxiv.org/pdf/2109.13559); [arXiv 2001.11200](https://arxiv.org/pdf/2001.11200)

### Inferences
- **Additive bias is matched and the input matrix is known.** In our setup the faulted executed action is a_exec = a_sent + b, and the plant's response to EE delta commands is approximately Δx ≈ G·a_exec with G ≈ diag (calibratable on clean logs). This is the textbook matched-uncertainty case, so a discrete-time DOB/L1 loop applies directly:
  - Estimate: b̂_k = LPF(G⁻¹Δx_k − a_sent,k−1).
  - Correct: a_sent = a_VLA − b̂.
  - It needs no learning, no VLA access and no fault labels. With zero bias, b̂ converges to the clean innovation mean (approximately 0) plus noise, so nominal cost is bounded by the LPF-filtered noise amplitude. A detection gate (Q3) can drive it to exactly 0.
  - The LPF bandwidth trades rejection speed against noise. Cheng et al. used K = 100–200 rad/s at control rates of 200–1000 Hz. At a LIBERO step rate (assumed to be about 20 Hz, unverified), the equivalent is a time constant of a few steps.
- **Contact makes the innovation G⁻¹Δx − a nonzero even in clean runs.** When the gripper pushes an object or the table, the commanded and realized deltas differ. This is "unmatched" from the observer's viewpoint and would be wrongly "compensated." Mitigations:
  - Fit the clean innovation distribution per axis from logged clean rollouts.
  - Adapt only during low-contact phases (gripper open, or low force proxy).
  - Use a slow LPF for a persistent bias, which by definition is constant while contact transients are not.
  
  This is the main nominal-cost risk of the analytical observer.
- **Sign inversion violates L1's known-sign assumption.** The bias-observer innovation under inversion is Δx − a = −2a, which is action-correlated and not constant, so a bias observer alone cannot fix it (and could make it worse). A dedicated sign estimator is needed: per-axis running correlation or RLS of Δx on a_sent. Flip the sign of that axis when the estimate is confidently negative, with hysteresis and a minimum excitation |a| > ε. This is identification-then-accommodation, avoiding Nussbaum's search transients. The fault is persistent within an episode, so a one-shot switch per axis matches its structure.
- **RMA is the template for the "learned" version.** Train φ(history of a_sent, Δx) → fault latent (b, sign vector) by supervised regression on our logged randomized-fault data, where ground truth is known. Then apply an analytical inverse: a_sent = S⁻¹(a_VLA − b̂). Unlike RMA, the frozen VLA does not consume the latent; the correction is applied in action space. This keeps the VLA frozen and makes nominal behavior exact when φ outputs (b = 0, S = I).
- **Why fine-tuning likely struggled with inversion.** This is a hypothesis consistent with Nussbaum and A-RMA reasoning. Inversion needs a discrete, history-dependent change of control direction, and a memoryless policy plus a weak history token must represent it inside a 7B network. A 2-dimensional sign latent with a hard inverse is a far smaller learning problem.

### Gaps
- No paper found applies L1, DOB or ESO to a VLA or to an EE-delta manipulation policy under injected action faults.
- Kim et al.'s DOB-RL paper was not fetched, so its bias-handling details are unverified.
- No quantitative benchmark compares Nussbaum against identification-based switching for sign reversal on robots.
- The discrete-time stability of L1 at low (~10–20 Hz) policy rates with an 8-step chunk delay was not found in the sources.

---

## Q3. Fault-tolerant control of actuator faults (bias, loss of effectiveness, reversal): what is cheap at the action level for an EE-delta controller?

### Takeaway
The FTC literature uses the standard fault model u_exec = λ·u + b, with λ the loss of effectiveness and b the bias, and treats stuck actuators as more severe. Two architectures are standard:
- **Active FTC:** fault detection and diagnosis (FDD), then accommodation or reconfiguration.
- **Passive FTC:** robust control that tolerates faults without detection.

For our action-level setting, active FTC maps cleanly onto a gated corrector: residual generation (predicted vs measured Δx), a detection test (threshold or CUSUM on the innovation), estimation of λ, b and sign, and accommodation (invert the fault model on the VLA's action).

### Cited Findings
- An adaptive-compensation survey classifies failures as total or partial loss of effectiveness and bias faults (zero gain, nonzero offset), and covers combined gain-plus-bias actuator models. — [arXiv 2305.02066](https://arxiv.org/pdf/2305.02066)
- Stuck actuators are generally considered more serious than loss of effectiveness, because they can cause complete actuation loss. — [Zhang et al., IEEE TAC 2020 via Kent repository](https://kar.kent.ac.uk/id/document/3210515)
- Adaptive control allocation for over-actuated marine vehicles handles loss of effectiveness, stuck faults and bias faults under saturation and rate limits. — [Akram et al., ICINCO 2020](https://www.scitepress.org/Papers/2020/98948/)
- Passive FTC can cover the delay before FDD detects a fault, for example Benosman's absolute-stability approach for loss of effectiveness. — [CORE 322394259](https://core.ac.uk/download/322394259.pdf)
- Survey of actuator fault-tolerant vehicle motion control: topologies, tolerated degradations and validation. — [Stolte et al., arXiv 2103.13671](https://arxiv.org/abs/2103.13671)
- A wrong-direction control command can destabilize the system. This is the classic risk of sign or control-reversal faults. — [NN flight control review, arXiv 2206.05596](https://arxiv.org/pdf/2206.05596)

### Inferences
Cheap action-level FTC pipeline for our setting. These are design implications, not sourced results.
1. **Residual generation.** r_k = Δx_k − Ĝ·a_sent,k−1, with Ĝ (per-axis gain) fit by least squares on clean logs.
2. **Detection.**
   - Bias: a CUSUM or GLR test on the mean of r, with the threshold set for a target false-alarm rate on clean episodes.
   - Inversion: a GLR on the sign of the per-axis regression slope of Δx on a.
   - When neither test fires, the correction is exactly zero, so clean behavior is preserved up to the false-alarm rate.
3. **Isolation.** Per axis. The 2-D fault structure means separate per-axis statistics.
4. **Estimation.** b̂ = mean innovation over a window (or LPF). ŝ = sign of the RLS slope.
5. **Accommodation.** a_sent = ŝ ⊙ (a_VLA) − b̂, clipped to action limits. Apply it per step inside the 8-step chunk (as in ResiP), because the VLA re-plans only every 8 steps.
6. **Optional learned layer.** Replace steps 2–4 with a small history network regressing (b, s) as in RMA, trained by BC on logged randomized-fault data with oracle labels. Keep step 5 analytic and keep a gate. An RL residual (Policy Decorator, PLD or ResFiT style) can be layered on top for unmodeled effects, with a tight α and zero-init, if success-based reward data are available.

### Gaps
- No FTC source found treats an end-effector-delta, learned-policy manipulator, so the pipeline above is extrapolated.
- No source found quantifies detection delay against false-alarm rate for such innovations in LIBERO-like contact-rich tasks. This should be measured from our own logs.
- The Wu, Zhang and Zhou (2000) detection-estimation-accommodation paper for loss of effectiveness surfaced only as a citation and is unverified.

---

## Q4. Recipe implications: frozen VLA plus residual or gated corrector, and expected nominal cost

### Takeaway
The evidence favors a structured, gated corrector over a free-form learned residual or VLA fine-tuning:
- Frozen-base residual methods preserve the base weights and start at (or near) the base behavior, but they report early dips and never measure clean retention.
- Observer-based augmentation (L1 on RL policies) kept 100% of nominal successes on hardware while rescuing loss-of-effectiveness and disturbance cases.

Expected nominal cost is dominated by detector false alarms and by contact-induced innovation. It is not dominated by "forgetting," because the VLA is untouched.

### Cited Findings
- L1 augmentation left nominal hardware performance intact for all six policy variants (✔ in every Nominal cell) while recovering Λ = 0.5 faults that the plain policies failed. — [arXiv 2112.01953](https://arxiv.org/html/2112.01953)
- Zero-init makes residual policies start identical to the base. Critic burn-in still left an early dip, which shrank as the burn-in threshold β was lowered. — [arXiv 1812.06298](https://arxiv.org/html/1812.06298)
- Bounded residuals (α ≈ demo action scale; ξ = 0.1–0.5) plus gating schedules are used to avoid destroying the base during RL. Too small an H "can cause complete failure." — [arXiv 2412.13630](https://arxiv.org/html/2412.13630); [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1)
- A per-step residual corrects open-loop chunk execution (Texec = 8), raising, for example, peg-in-hole from 5% to 99%. — [arXiv 2407.16677](https://arxiv.org/html/2407.16677)
- Frozen-OpenVLA plus residual RL reaches 99.2% average on LIBERO (Spatial/Object/Goal), from 91.8%. A residual on a frozen VLA therefore does not inherently degrade it. — [arXiv 2511.00091](https://arxiv.org/html/2511.00091v1)
- An RMA-style history regressor can identify environment latents from commanded-vs-realized motion by supervised learning. — [arXiv 2107.04034](https://arxiv.org/pdf/2107.04034v1.pdf)

### Inferences
Concrete options, in order of cost:
- **A. Analytical observer and identifier, no training.**
  - Bias via DOB/L1-style LPF on the innovation Δx − Ĝa. Inversion via an RLS sign test with hysteresis.
  - Gate with CUSUM thresholds calibrated on clean logs to, for example, ≤1–2% false alarms per episode.
  - Expected clean success ≈ 0.963 × (1 − P(false alarm) × P(failure | false correction)). Any target such as ≥0.95 is reachable by threshold choice.
  - Recovery depends on detection delay relative to episode length and on contact contamination.
  - This is the right first baseline because it isolates whether fault recovery is limited by identification or by the policy.
- **B. Learned identifier, analytic inverse (RMA-style BC on logged data).**
  - A small GRU or MLP over the last N (a_sent, Δx, gripper) tuples regresses b and per-axis sign logits, supervised by oracle fault labels from our randomized-fault logs.
  - Output an explicit "no fault" class, or a calibrated confidence, and apply the correction only above threshold.
  - Train with ~50% clean episodes so the zero output is the default. Initialize the output head at zero, so b̂ = 0 and the sign logits favor +1.
  - Nominal cost equals the classifier's false-positive rate. Unlike LoRA, it cannot change the VLA's clean behavior.
  - If the "identifiability-masked targets" issue (unobservable fault before the arm moves on that axis) applies, mask the loss and the correction until the excitation threshold is met. This is the classic persistent-excitation condition.
- **C. Free residual policy via BC on oracle compensation.**
  - Target a* = S⁻¹(a_VLA − b) − a_VLA. For inversion that is −2a_VLA on the flipped axis, so the bound α must be at least 2·max|a|. This makes the "bounded residual for nominal safety" argument weak for inversion.
  - A structured parameterization (B) is strictly better for these two fault classes.
- **D. Residual RL on top (Policy Decorator, PLD or ResFiT style).**
  - Only if unmodeled effects remain after A or B. Use zero-init, tight α, ε-ramp gating, and success reward on faulted episodes mixed with clean ones to penalize nominal regressions.
  - PLD and ResFiT show this is sample-feasible on LIBERO-class tasks (about 10^4–10^5 steps), but the early dip must be expected and run offline.
- **Interaction with chunking.** OpenVLA-OFT executes 8 actions per inference. The corrector should run per control step and use the most recent Δx, as in ResiP. The VLA itself sees faulted observations only every 8 steps, which is a strength here: the corrector handles intra-chunk drift.
- **Expected nominal cost: about zero by construction for B and A**, apart from false alarms. That compares with −10 points for LoRA fine-tuning. This is a design inference; no paper measures it for VLAs under faults.

### Gaps
- No empirical result found for any VLA plus fault-corrector combination. All nominal-cost estimates above are inferences that need our own A/B measurement.
- The exact relation between OpenVLA-OFT's normalized delta actions and realized EE Δx in LIBERO (gain, latency, contact effects) needs to be measured from our logs. The literature does not provide it.
- RLDG, PA-RL and "residual dynamics adaptation" papers were not retrieved within the tool budget.

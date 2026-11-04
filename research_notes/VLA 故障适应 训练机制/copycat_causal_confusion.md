# Copycat problem and causal confusion in BC with action/observation-history inputs (relevance: efference-copy token for fault-adaptive OpenVLA-OFT)

Scope note: numbers below come from the primary papers' full text (PDFs pulled from arXiv / NeurIPS proceedings and read directly) unless marked otherwise. "Inferences" sections are my reasoning about our setup, not findings from the papers.

## Q1. What do the core papers report about the mechanism and size of the effect?

### Takeaway
Giving a BC learner history, past actions included, often makes held-out imitation loss lower and closed-loop performance worse. The learner exploits the strong temporal correlation of expert actions ("copy the last action"). Changepoints, where the action must change in response to new evidence, are rare in the data, so copying wins under the loss. Reported drops are large: up to 3-5x in reward on some MuJoCo tasks, a CARLA history policy losing about 10 points of success, and diffusion policies with 16-step history collapsing to 0-20% success where history-free policies reach 28-79%.

### Cited Findings
**de Haan, Jayaraman, Levine, "Causal Confusion in Imitation Learning", NeurIPS 2019, arXiv:1905.11979**
- Discriminative BC is non-causal, so "access to more information can yield worse performance" (causal misidentification). — [arXiv:1905.11979](https://arxiv.org/abs/1905.11979)
- Their controlled testbed adds the *previous action* to the observation ("CONFOUNDED") in MountainCar, Hopper and 3 Atari games. CONFOUNDED does significantly worse than ORIGINAL on every task, even though it reaches near-zero or *lower* validation loss on held-out expert data. In Pong, for example, CONFOUNDED has lower validation loss but lower reward. — [paper PDF](https://arxiv.org/pdf/1905.11979)
- They re-analyze Wang et al. (GTA-V driving). The HISTORY policy has better validation perplexity (0.834 vs 0.989) but worse driving: distance 144.9 vs 269.0, interventions 2.94 vs 1.30, collisions 6.49 vs 3.38. They also note that recurrent or history-based imitation "has been consistently observed in prior work to hurt performance". — [paper PDF](https://arxiv.org/pdf/1905.11979)
- Remedy: a causal-graph-parameterized policy, trained with random binary masks over input factors (2^n graphs), plus *targeted interventions* (expert queries or policy-execution reward) to choose the graph. They say it resolves misidentification "using orders of magnitude fewer queries than DAgger". Dropout on the possibly-confounding inputs (Bansal et al., ChauffeurNet) "does indeed help ... but is significantly poorer" than intervention. — [paper PDF](https://arxiv.org/pdf/1905.11979)

**Wen, Lin, Darrell, Jayaraman, Gao, "Fighting Copycat Agents in Behavioral Cloning from Observation Histories", NeurIPS 2020, arXiv:2010.14876**
- Mechanism: copycat arises when (i) expert actions are strongly temporally correlated and (ii) past actions can be recovered from the observation history. The imitator "learns to cheat by predicting the expert's previous action, rather than the next action". Explicit action input is not required, because a_{t-1} can be decoded from frames o_t, o_{t-1}. — [arXiv:2010.14876](https://arxiv.org/abs/2010.14876)
- Smoking-gun metric: train an MLP to predict a_t from a_{t-1..t-9}. The error is *lower* on BC-OH rollouts than on expert rollouts in all 6 MuJoCo tasks (Table 1; for example Ant: expert 6.91e-2 vs BC-OH 1.46e-2). The other per-task values came out with interleaved columns in text extraction and are omitted. "Temporal correlation among expert actions leads to even higher temporal correlation among learned policy actions." The action-predictability ratio is inversely correlated with normalized reward (Fig. 3). — [paper PDF](https://arxiv.org/pdf/2010.14876)
- Qualitative failures: a Walker2d knee that never switches from extended to bent, and HalfCheetah agents that "did not ever start to move from a resting position". — [paper PDF](https://arxiv.org/pdf/2010.14876)
- Remedy, FCA: a target-conditioned adversary D predicts a_{t-1} from (μ_e, a_t), and the encoder maximizes H(a_{t-1}|e_t, a_t). Only previous-action information *not shared with the target* is removed. On top of that sits a VAE-style information bottleneck (KL to N(0,I)). Objective: L(F(e),a_t) + λ·KL − α·L(D(μ_e,a_t),a_{t-1}). — [paper PDF](https://arxiv.org/pdf/2010.14876)
- Results (Table 2, partially observed MuJoCo, positions only, H=2), BC-OH → Ours: Ant 1750→2150, Hopper 293→1086, Humanoid 565→671, Reacher −64→−54, Walker2d 592→1296, HalfCheetah 820→1250. Dropout-BC on past frames was *worse* than BC-OH everywhere (e.g., Ant 830, Walker 283). IB alone ("w/o Adversary") already beats BC-OH, for example Walker 962 and HalfCheetah 1260. TCA is the most important component. An RNN policy is "similar or worse" than stacked BC-OH (e.g., Ant −311). DAgger with 100 queries ≈ FCA; DAgger with 1k queries is much better (e.g., Hopper 4963). — [paper PDF](https://arxiv.org/pdf/2010.14876)
- Leakage check (Hopper): MSE of predicting a_{t-1} from features. BC-OH embedding 0.38e-3; TCA embedding 5.97e-3; a_t alone 5.00e-3. TCA removes essentially all excess past-action information. — [paper PDF](https://arxiv.org/pdf/2010.14876)

**Wen, Lin, Qian, Gao, Jayaraman, "Keyframe-Focused Visual Imitation Learning", ICML 2021, arXiv:2106.06452**
- Formal intuition: let CP be the fraction of "changepoint" frames that an optimal copycat ψ*(a_{t-1}, a_{t-2},...) gets wrong. Copycat wins when MSE_D(θ_R*) > CP, that is, when the changepoint fraction is lower than the training error of the reward-optimal imitator. Copycat therefore becomes more likely with (1) more temporal correlation (lower CP) and (2) a harder imitation problem or noisier targets (higher irreducible error). The same phenomenon is called "copycat" (Wen 2020), "inertia" (Codevilla et al. 2019) and "causal confusion" (de Haan 2019). — [arXiv:2106.06452](https://arxiv.org/abs/2106.06452)
- Remedy: train a small copycat MLP on action history and compute the per-sample action-prediction error APE_t = (ψ*(a_{t-1..}) − a_t)^2. Upweight samples with high APE in the BC loss. — [paper PDF](https://arxiv.org/pdf/2106.06452)
- CARLA100 NoCrash-Dense %success: BC-SO 42.7, BC-OH 33.0, Ours 43.4, FCA 35.7, **History Dropout 34.0**, DAgger (120k queries) 35.2. With speed withheld from observations (CARLA-w/o-speed): BC-SO 9.2, BC-OH 25.7, Ours 36.8, FCA 27.4, History Dropout 25.3, DAgger 28.3. With history available, keyframe weighting only recovers single-frame performance; it beats single-frame only when history is truly needed. — [paper PDF](https://arxiv.org/pdf/2106.06452)
- MuJoCo-image rewards, BC-OH → Ours(softmax): Hopper 740→951, HalfCheetah 615→819, Walker 614→769. Avg APE of BC-OH rollouts is "much lower than the expert in all environments", which is the same predictability signature as in Wen 2020. — [paper PDF](https://arxiv.org/pdf/2106.06452)

**Wen, Qian, Lin, Teng, Jayaraman, Gao, "Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming" (PrimeNet), ICML 2022, arXiv:2206.10816**
- Feed the main network a coarse "priming" estimate of the output, computed from known key inputs only (e.g., the current frame). This gives the optimizer a better, simpler shortcut than copying. It is framed explicitly as a fix for the copycat shortcut in CARLA and MuJoCo. — [arXiv:2206.10816](https://arxiv.org/abs/2206.10816), [PMLR](https://proceedings.mlr.press/v162/wen22d.html)

**Seo, Hwang, Yang, Kim, "Regularized Behavior Cloning for Blocking the Leakage of Past Action Information" (PALR), NeurIPS 2023 spotlight**
- They define "past action leakage" as conditional dependence between the policy's representation and a^E_{t-1} given a^E_t. Leakage (HSCIC) is negatively correlated with normalized score in hopper, walker2d, halfcheetah and ant (Fig. 1). — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/06b71ad997f7e3e4b2e2f2ea12e5a759-Abstract.html)
- Table 1, normalized scores. BC / KF / PrimeNet / RAP / FCA / MINE / **PALR**:
  - hopper w=2: 32.5/32.0/30.0/20.2/31.9/25.0/**42.0**
  - hopper w=4: 47.7/45.7/45.3/32.6/36.9/37.6/**58.4**
  - walker2d w=2: 53.0/50.0/48.5/15.8/63.1/58.6/**79.8**
  - halfcheetah w=2: 74.1/64.3/61.5/63.9/78.2/76.3/**86.4**
  - ant w=4: 64.4/48.6/58.2/48.6/57.7/56.6/**64.6**
  - carla-lane: 52.5/66.6/58.2/25.3/57.1/60.1/**72.9**

  The paper says FCA and MINE "fail to show consistent improvement over BC in most tasks except walker2d", because their adversarial MI lower bounds are inaccurate. PALR uses a closed-form kernel HSCIC penalty with no min-max optimization. — [paper PDF](https://proceedings.nips.cc/paper_files/paper/2023/file/06b71ad997f7e3e4b2e2f2ea12e5a759-Paper-Conference.pdf)
- Independent replications are weaker than the original papers claim: in this re-implementation, KF and PrimeNet are roughly at or below BC on most MuJoCo settings. — same source

**Swamy, Choudhury, Bagnell, Wu, "Causal Imitation Learning under Temporally Correlated Noise", ICML 2022 (oral), PMLR 162:20877–20890, arXiv:2202.01312**
- Setting: an unobserved, *persistent* confounder u_t (their motivating example is literally "persistent wind" on a quadcopter) enters both state and recorded action. BC then learns E[a|s] ≠ E[a|do(s)]. Lemma 6.1: E[u|s] = π_BC(s) − π_IV(s), so the BC bias equals the expected confounder given the state. — [PMLR](https://proceedings.mlr.press/v162/swamy22a.html), [paper PDF](https://arxiv.org/pdf/2202.01312)
- Remedy: use *past states* as instruments (IVR). DoubIL needs a simulator; ResiduIL is offline and game-theoretic. Both match E[a|do(s)] better than BC on confounded LunarLander/Ant/HalfCheetah, especially in the low-data regime. — [paper PDF](https://arxiv.org/pdf/2202.01312)
- Persistence matters: the performance bound J(π_E) − J(π) ≤ c·κ(Π)·T². κ (ill-posedness) grows with the number of steps H the confounder persists. In an LQG example κ rises steeply as H goes from 0 to 15 (Fig. 7), because a valid instrument must come from further back in time. — [paper PDF](https://arxiv.org/pdf/2202.01312)

**Villasevil (Torne), Tang, Liu, Finn, "Learning Long-Context Diffusion Policies via Past-Token Prediction", ICLR 2025 WRL workshop / CoRL 2025 (PMLR v305), arXiv:2505.09561**
- The paper's *opposite* finding: diffusion policies **under**-use history. Their past-action predictability ratio is 10-100x weaker than the expert's. Regression policies **over**-use it, consistent with copycat. — [arXiv HTML v2](https://arxiv.org/html/2505.09561v2)
- Success rates (sim) for PTP / no-PTP (16-step history) / no-history: Square 0.89/0.17/0.79; Tool-Hang 0.75/0.00/0.51; Transport 0.67/0.00/0.60; ALOHA 0.98/0.20/0.28; Long Square 0.93/0.03/0.12; Push-T 0.62/0.59/0.67. On real-robot tasks, PTP averages about 70% vs 15% for no-history; no-PTP is near zero on 3 of 4 tasks. Naive long history is catastrophic, and the auxiliary loss for predicting past action tokens fixes it. — [arXiv HTML v2](https://arxiv.org/html/2505.09561v2)
- Ablation A.3: "adding the previous 16 actions [as input] without PTP does not improve performance". The benefit comes from the auxiliary *prediction* target, not from feeding actions as input. The decoder-only PTP matches full PTP. The text's averages conflict ("50%" vs "30%/60%" vs "3x"); the table numbers above are the reliable ones. — [arXiv HTML v2](https://arxiv.org/html/2505.09561v2)

**Sobol Mark, Liang, Attarian, Fu, Dwibedi, Kumar, Shah, "BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames", arXiv:2602.15010 (Feb 2026, preprint)**
- Naive history policies "latch onto incidental features" because history space coverage shrinks exponentially with horizon. Regularizers gave inconsistent benefits; ground-truth-state regularization *dropped* sim success from 55.5% to 19.0%. — [arXiv:2602.15010](https://arxiv.org/html/2602.15010v1)
- Real-world average success: current-obs 14.4%, naive history 12.8%, PTP 31.8%, BPP (VLM-selected keyframes only) 53.6%. With enough data, naive history eventually catches up. — [arXiv:2602.15010](https://arxiv.org/html/2602.15010v1)

### Inferences
- The effect is robust across observation-history (implicit past actions) and explicit past-action inputs. Its size grows with temporal correlation of targets and with irreducible target noise (Wen 2021 inequality).
- The "lower validation loss, worse rollouts" signature (de Haan, Wang et al., Wen) is the diagnostic to check in our runs: compare BC loss of the efference-token model vs the memoryless model on held-out data against closed-loop success.

### Gaps
- No paper found that reports copycat specifically in a large pretrained VLA with token (discretized, parallel-decoded) actions. All quantitative evidence is from small MLP/CNN, diffusion, or ACT-style policies.
- No paper found quantifying copycat when the history input contains *executed (policy-own) chunks* rather than expert actions.

## Q2. Which remedies work, and how strong is the evidence?

### Takeaway
The most consistent quantitative wins come from two families. The first is representation-level conditional-independence penalties that strip only the past-action information not shared with the target: FCA's TCA, and especially PALR's HSCIC penalty. The second is restructuring: give history as an auxiliary *prediction target* (PTP) rather than raw input, or condition only on a few key frames or a learned latent estimate (BPP, RMA). Naive history dropout is consistently weak. DAgger helps with enough queries but is not a cure on its own.

### Cited Findings
- **History or input dropout (Bansal et al. 2019):**
  - FCA paper: Dropout-BC on past frames performs "uniformly poorly across all tasks", below BC-OH (e.g., Walker2d 283 vs 592). — [Wen 2020](https://arxiv.org/pdf/2010.14876)
  - Keyframe paper: History Dropout 34.0 vs BC-OH 33.0 vs BC-SO 42.7 CARLA %success, so essentially no gain. — [Wen 2021](https://arxiv.org/pdf/2106.06452)
  - de Haan: dropout "does indeed help ... but is significantly poorer" than targeted intervention. — [de Haan 2019](https://arxiv.org/pdf/1905.11979)
- **Information bottleneck:** an IB alone (no adversary) gives large gains over BC-OH (Walker2d 592→962, HalfCheetah 820→1260, Hopper 293→473). — [Wen 2020](https://arxiv.org/pdf/2010.14876)
- **Target-conditioned adversary / conditional-independence regularizers:**
  - TCA is the key FCA component; an *unconditioned* adversary removes too much useful information. — [Wen 2020](https://arxiv.org/pdf/2010.14876)
  - PALR (HSCIC) is best or tied in 9/9 settings, and adversarial versions are inconsistent in the re-evaluation (numbers in Q1). — [Seo 2023](https://proceedings.nips.cc/paper_files/paper/2023/file/06b71ad997f7e3e4b2e2f2ea12e5a759-Paper-Conference.pdf)
- **Keyframe / changepoint reweighting by copycat error (APE):** CARLA 33.0→43.4 and CARLA-w/o-speed 25.7→36.8. Smaller or no gains in the PALR re-implementation (carla-lane 52.5→66.6; MuJoCo ≈ BC). — [Wen 2021](https://arxiv.org/pdf/2106.06452), [Seo 2023](https://proceedings.nips.cc/paper_files/paper/2023/file/06b71ad997f7e3e4b2e2f2ea12e5a759-Paper-Conference.pdf)
- **Priming (PrimeNet):** a coarse estimate from key inputs only, fed as an extra input. Positive in the original paper; ≈ BC in the PALR re-evaluation (e.g., hopper w=2 30.0 vs 32.5). — [Wen 2022](https://arxiv.org/abs/2206.10816), [Seo 2023](https://proceedings.nips.cc/paper_files/paper/2023/file/06b71ad997f7e3e4b2e2f2ea12e5a759-Paper-Conference.pdf)
- **Past-token prediction (auxiliary loss on past actions):** turns 0.00-0.20 into 0.67-0.98 success on long-context diffusion tasks. Also enables test-time verification (pick the sampled chunk whose past-token reconstruction matches the executed actions), worth about +5% at 5 samples. — [Villasevil 2025](https://arxiv.org/html/2505.09561v2)
- **Removing action history:** BC-SO beats BC-OH on CARLA (42.7 vs 33.0), but loses badly when history is needed (9.2 vs 25.7 without speed). — [Wen 2021](https://arxiv.org/pdf/2106.06452)
- **Key-history-frame selection:** BPP beats naive history and PTP on real tasks (53.6% vs 12.8% / 31.8%). — [BPP 2026](https://arxiv.org/html/2602.15010v1)
- **DAgger / on-policy data:**
  - DAgger at 100 queries ≈ FCA; at 1k queries it approaches the expert on several MuJoCo tasks (FCA Table 2). — [Wen 2020](https://arxiv.org/pdf/2010.14876)
  - On CARLA, DAgger with 120k queries (35.2) is worse than keyframe weighting (43.4) and single-frame BC. — [Wen 2021](https://arxiv.org/pdf/2106.06452)
- **Causal-graph parameterization + targeted intervention:** this works in low-dimensional disentangled settings (MountainCar, Hopper, Atari with a VAE). In the FCA comparison it fails on high-dimensional inputs: it cannot run on Ant or Humanoid, and on Hopper it gets 978 vs FCA 1086. — [de Haan 2019](https://arxiv.org/pdf/1905.11979), [Wen 2020](https://arxiv.org/pdf/2010.14876)
- **Instrumental variables (past states as instruments)** against persistent confounders: better than BC in the low-data regime on LunarLander, Ant and HalfCheetah. The bound weakens as persistence grows. — [Swamy 2022](https://arxiv.org/pdf/2202.01312)

### Inferences
- The remedy most directly transferable to a 7B LoRA model is a cheap conditional-independence penalty such as HSCIC on the efference-token's projected embedding: penalize dependence between that embedding and the *previous sent chunk* given the target chunk. It needs no min-max training and stays offline.
- APE-weighting is cheap and especially fitting for us. Fit a trivial copycat predictor ψ(last 2 sent chunks) → target chunk, then upweight samples where it fails. Those are the fault-onset and first-discovery steps, plus steps where the oracle target departs from the previous command.
- PTP as an *auxiliary head* (reconstruct the last sent chunk or the measured Δstate from the efference token) forces the token to be read, without making copying the cheapest route to the *main* target. Caveat: PTP was shown to fix *under*-use in diffusion policies. A regression- or token-CE-style policy may already over-use history, so PTP should be combined with, not substituted for, an anti-leak penalty.

### Gaps
- No head-to-head of these remedies inside a pretrained VLA or under LoRA.
- Independent replications disagree on KF and PrimeNet (strong in their own papers, ≈ BC in Seo 2023).

## Q3. Does conditioning on past actions make adaptation to persistent dynamics changes harder or easier? Evidence on persistent faults?

### Takeaway
Action-plus-state history is *necessary* to identify a persistent dynamics change. The literature that succeeds at it (RMA-style adaptation for motor strength and wear-and-tear) does not feed raw history to an action-cloning policy. It compresses history into a supervised latent estimate of the hidden parameter, trained on on-policy data. RMA explicitly reports that feeding raw state-action history directly to the policy was worse. Persistence also makes confounding harder to undo (Swamy). I found no imitation-learning paper on actuation faults specifically.

### Cited Findings
- RMA (Kumar, Fu, Pathak, Malik, RSS 2021, arXiv:2107.04034), architecture: an adaptation module φ maps the last k=50 states and actions (0.5 s) to an estimate ẑ_t of an extrinsics latent z_t = μ(e_t). The latent encodes motor strength (12 dims), mass, friction and terrain height. φ is trained by **supervised regression on z_t**, not on actions, with **on-policy (DAgger-like) rollouts**, because data from only good trajectories "would not be robust to deviations". — [paper PDF](https://arxiv.org/pdf/2107.04034)
- RMA, raw-history alternative: "we could have trained a base policy which directly takes the state and action history as input ... this leads to unnatural gaits and poor performance in simulation". RMA without adaptation fails on several real terrains (e.g., 0% vs 80% success on one test). — [paper PDF](https://arxiv.org/pdf/2107.04034)
- Swamy 2022: with a persistent hidden disturbance ("persistent wind"), plain BC on the state recovers a biased policy (bias = E[u|s]). The longer the disturbance persists, the weaker past-state instruments become and the looser the performance bound. — [paper PDF](https://arxiv.org/pdf/2202.01312)
- A later legged-robot paper reports zero-shot sim-to-real transfer under actuator faults/degradation with an adaptive control strategy (arXiv:2312.17606). I did not read the full text, so I have no details or numbers. — [arXiv:2312.17606](https://arxiv.org/pdf/2312.17606)

### Inferences
- **Persistent faults change the copycat calculus in both directions.**
  - *After* the policy has correctly compensated once, the right next command is again the compensated command, so copying the last sent chunk is the right answer. Copycat here acts like a useful integrator, which may explain why some of the bias recovery is retained.
  - *Before* the first correct compensation (fault onset, or the first call), the target departs from the history. These are exactly the rare changepoint samples that the CP < MSE(θ*) inequality says get under-fit.
  - Our pattern fits this: modest bias recovery (0.013→0.327), and spread that stays above base after the first call.
- **Inversion vs bias.** Bias can be read off additively: Δstate − f(sent) ≈ constant offset. Inversion needs a sign-flip interaction between sent action and Δstate, and its optimal correction *reverses* the sent action, the opposite of copying. A copycat-prone learner should therefore find inversion hardest, which is consistent with 0.02.
- **RMA's decoupling is the strongest precedent for our design choice.** Estimating the fault (bias vector, sign mask) from the efference copy with a *supervised fault-latent head*, and conditioning the action head on that estimate rather than on raw sent chunks, removes the copy channel by construction.

### Gaps
- No paper found on BC or imitation fine-tuning under injected persistent actuation faults (bias, sign inversion) for manipulation or VLAs.
- RMA's motor-strength-specific ablation numbers were not extracted beyond the quoted statements.

## Q4. Implications for our design (sent-action efference copy; mixed or ambiguous targets producing spread)

### Takeaway
Yes on both counts. Feeding the last 2 sent 8-step chunks into a BC objective whose targets are mostly the policy's own (highly autocorrelated) rollout actions sets up the copycat or inertia shortcut, which plausibly costs clean success. Supervising identical-looking inputs with a mixture of "base" and "compensated" targets makes a likelihood-trained, per-dimension 256-bin token head learn the mixture, so action spread is expected. BC recovers E[a|input] (or the full conditional distribution), not the causal correction.

### Cited Findings
- Copying is favored when targets are strongly autocorrelated and past actions are available. That is directly so for us, since the sent chunks are input and the targets are mostly self-rollout actions. Learned policies end up *more* autocorrelated than their teacher. — [Wen 2020](https://arxiv.org/pdf/2010.14876), [Wen 2021](https://arxiv.org/pdf/2106.06452)
- Higher irreducible target error makes the copycat solution win (MSE_D(θ_R*) > CP). Ambiguous or mixed targets raise MSE_D(θ_R*). — [Wen 2021](https://arxiv.org/pdf/2106.06452)
- Feeding past actions as input without an auxiliary objective gave no gain in PTP's ablation. — [Villasevil 2025](https://arxiv.org/html/2505.09561v2)
- BC fits E[a|s]; under a hidden persistent disturbance the result is biased by E[u|s], and the identifying signal must come from history (instruments). — [Swamy 2022](https://arxiv.org/pdf/2202.01312)
- Supervising the history-encoder against a latent (not actions), trained on-policy, is the working recipe for dynamics adaptation. — [RMA 2021](https://arxiv.org/pdf/2107.04034)

### Inferences
- **Clean drop (0.963→0.863).** Consistent with copycat or inertia on clean episodes: the efference token makes "repeat last chunk" a low-loss shortcut, and closed loop then stops reacting to visual changepoints such as grasp, release and subgoal switches.
  - Diagnostics: (a) the Wen action-predictability metric (MLP predicting the chunk from the previous 2) on clean rollouts of fine-tuned vs base. The copycat signature is lower APE than base. (b) Held-out BC loss lower than base while success is lower.
  - Cheap test: zero or shuffle the efference token at inference on clean episodes. If clean success recovers, the token is being copied.
- **Spread (1.1-1.7x base after the first call).**
  - If the same efference-token content co-occurs with "self action" targets in some samples and "oracle-compensated" targets in others, the training data is unidentifiable from the input. Examples: the token cannot yet reveal the fault, or masked dims mix both sources. Token CE then learns a bimodal per-bin distribution.
  - Parallel decoding samples or argmaxes each dim/step independently, so it cannot coordinate modes across dims or steps. That yields extra spread and incoherent chunks.
  - Remedies: make targets a deterministic function of what the input reveals (the Amendment-4 identifiability masking), drop or relabel samples where the fault is not yet identifiable, or give the head an explicit fault-estimate input so the mode is selected.
- **Design alternatives suggested by the literature.**
  1. Replace raw sent chunks with an *innovation* signal (measured Δstate minus Δstate predicted from the sent action under nominal dynamics). It carries the fault information but cannot be copied as an action. This is my extrapolation from the RMA/Kalman analogy, not tested in a paper.
  2. Add an RMA-style supervised fault-latent head (bias vector / sign per axis) on the efference token, and feed its output to the action head.
  3. Add an HSCIC/PALR penalty between the token embedding and the previous sent chunk, conditioned on the target.
  4. Use APE-based keyframe weighting so fault-onset and correction samples dominate.
  5. Add a PTP-style auxiliary reconstruction of past chunk and Δstate, so the token is used for identification.
  6. Keep collecting on-policy (DAgger-style) data. RMA found this essential for history-based estimators.

### Gaps
- No direct evidence on whether a *2-chunk* (16-step) action history is short enough to avoid copycat in a 7B VLA. History-length effects were measured only for diffusion with PTP (37%→77% mean success going from 2 to 16 obs), which is the opposite regime. — [Villasevil 2025](https://arxiv.org/html/2505.09561v2)
- No literature found on mixture-target spread specifically for parallel-decoded discrete action tokens (OpenVLA-OFT style). The spread mechanism above is inferred.

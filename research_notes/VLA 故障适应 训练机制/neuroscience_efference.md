# Efference copy, internal models and motor adaptation: what neuroscience implies for training a VLA policy to compensate persistent actuation faults

Context: a pretrained, memoryless VLA was given an "efference copy" input (the last 2 sent action chunks plus the measured state change) and fine-tuned under randomized persistent faults. The faults were an additive 2-D bias (analogous to a force field or visuomotor shift) and axis sign inversion (analogous to mirror reversal). Observed result: about 1/3 compensation of bias, no learning of inversion, and some cost on nominal (no-fault) trials.

Note on sourcing: about 17 tool calls. Full text was read for Wilterson & Taylor 2021, Yang, Cowan & Haith 2021, Heald, Lengyel & Wolpert 2021 (PMC preprint) and Smith, Ghazizadeh & Shadmehr 2006. The other papers were checked only at abstract or record level. The Telgen et al. 2014 full text could not be fetched (403), so its findings come from the abstract and repository listings.

## Q1. Efference copy and forward models: state estimation vs adaptation, and sensory prediction error as the training signal

### Takeaway
In the brain, an efference copy (corollary discharge) feeds a forward model that predicts the sensory consequences of a command. That prediction serves two separate roles. (a) Online state estimation: the prediction is fused with delayed, noisy feedback in a Kalman-like way. (b) A teaching signal: the sensory prediction error (SPE), meaning predicted minus observed outcome, drives implicit cerebellar adaptation. The SPE drives this adaptation, not task error or motor corrections. For the VLA, simply feeding past actions and the state change into the controller gives it the raw ingredients of an SPE. It does not give it a forward model, and it gives no explicit training signal that makes the compensation solve for the fault.

### Cited Findings
- **Bibliographic record:** Crapse & Sommer (2008), "Corollary discharge across the animal kingdom," *Nat Rev Neurosci* 9(8):587–600, doi 10.1038/nrn2457. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5153363); [Duke record](https://scholars.duke.edu/publication/742392)
- **Crapse & Sommer content:** The review surveys motor-to-sensory (corollary discharge, CD) circuits across species and sorts CD into lower-order and higher-order classes by function. It credits Sperry with coining "corollary discharge" in work on fish. Sperry's definition was less specific than von Holst's "efference copy" about where in the motor pathway the branch leaves. — [PMC5153363](https://pmc.ncbi.nlm.nih.gov/articles/PMC5153363)
- **Wolpert, Ghahramani & Jordan (1995), *Science* 269:1880–1882, "An internal model for sensorimotor integration":** Participants estimated their hand position after moving in the dark under externally imposed forces. A Kalman filter that combines motor outflow (efference copy) with sensory feedback explained the time course of their estimation errors. The Kalman gain sets the relative weight of the internal simulation and of the sensory correction. The authors present this as direct support for an internal forward model used in state estimation. — [PubMed 7569931](https://pubmed.ncbi.nlm.nih.gov/7569931/); [abstract](https://mlg.eng.cam.ac.uk/zoubin/zoubin/kalman.abstract.html)
- **Shadmehr, Smith & Krakauer (2010), *Annu Rev Neurosci* 33:89–108, "Error correction, sensory prediction, and adaptation in motor control":** Feedback is slow and noisy, so the brain combines forward-model predictions with observations. For those predictions to be useful they must be unbiased. Forward models stay calibrated through motor adaptation, which is learning driven by sensory prediction errors. — [PubMed 20367317](https://pubmed.ncbi.nlm.nih.gov/20367317/); [author PDF](https://groups.seas.harvard.edu/motorlab/Reprints/annualrev10_unformatted.pdf)
- **Tseng, Diedrichsen, Krakauer, Shadmehr & Bastian (2007), *J Neurophysiol* 98(1):54–62, "Sensory prediction errors drive cerebellum-dependent adaptation of reaching":**
  - Controls and patients with cerebellar ataxia adapted to a visuomotor perturbation in two conditions: "shooting" movements with no online correction, and "pointing" movements with correction.
  - In controls, adding online corrections did not change adaptation. Sensory errors alone were enough.
  - Cerebellar patients were equally impaired in both conditions. Adaptation therefore depends on the cerebellum and is driven by the mismatch between predicted and actual sensory outcomes, not by motor corrections.
  - — [Bangor record](https://research.bangor.ac.uk/en/publications/sensory-prediction-errors-drive-cerebellum-dependent-adaptation-o/); [PDF](https://blamlab.squarespace.com/s/Tseng07ErrorsCerebell.pdf)
- **Mazzoni & Krakauer (2006), *J Neurosci* 26(14):3642–3645, "An implicit plan overrides an explicit strategy during visuomotor adaptation":**
  - Participants were told to cancel a 45° rotation by aiming at a neighbouring target. The strategy worked at first, but errors then grew steadily, because implicit adaptation to the rotation ran in parallel with the strategy.
  - The rate of this implicit drift did not differ significantly from adaptation without a strategy.
  - Interpretation: implicit adaptation is driven by the SPE between the aimed direction and the seen outcome, even when the task error is zero. The motor system enforces that planned and executed trajectories stay congruent.
  - — [record](https://profiles.wustl.edu/en/publications/an-implicit-plan-overrides-an-explicit-strategy-during-visuomotor/); [PDF](https://blamlab.squarespace.com/s/D_An-Implicit-Plan-Overrides.pdf)
- **Counterpoint, Hadjiosif, Krakauer & Haith (2021), *J Neurosci* 41(12):2747–2761, "Did we get sensorimotor adaptation wrong? Implicit adaptation as direct policy updating rather than forward-model-based learning":**
  - Under mirror reversal, implicit adaptation still occurs, but it amplifies the errors instead of reducing them.
  - The way it develops over time and across targets fits direct policy learning, not learning that goes through a forward model. A forward-model learner would have no obvious reason to fail under mirror reversal.
  - So implicit adaptation may use the SPE to update the controller directly, through a fixed, hard-wired error-to-correction mapping.
  - — [PMC8018745](https://pmc.ncbi.nlm.nih.gov/articles/PMC8018745); [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.01.22.914473.full.pdf)
- **RL follow-up (Hadjiosif result):** A 2022 arXiv study reports that three major families of RL algorithms did not reproduce the human mirror-reversal pattern, and a model-based variant did. This is reported second-hand through the search summary; the paper was not read. — [arXiv 2208.10892](https://arxiv.org/pdf/2208.10892v2.pdf)

### Inferences
- **Two jobs, two signals.** The brain separates (i) state estimation, where the forward model is fused with feedback, from (ii) calibrating that estimate, where SPE drives cerebellar learning. Our VLA receives the efference copy only as extra controller input and is trained on an imitation or task loss. Nothing trains a forward model, and no SPE is formed explicitly. The network has to discover the SPE (expected minus measured Δstate) implicitly from 2 chunks of history, which is a weak and identifiability-limited signal.
- **Design implication: add an explicit forward-model head.** A head that predicts Δstate from (state, sent action) gives a supervised SPE target. It can be trained on nominal data, so it does not depend on faults. Its residual can then condition an adaptation module (a fault estimate). This mirrors the split between the "cerebellar forward model" and the "controller".
- **Hadjiosif et al. warning: why mapping SPE directly to a correction fails on inversion.** If the learned compensation is effectively "action += −k·(observed error)", that is a sign-fixed policy update. It cancels bias but makes sign inversion worse. A gradient-trained VLA may converge to exactly this, or to a compromise near zero gain on the inverted axis. That could explain zero learning of inversion. A fix needs an explicit estimate of the fault model (a sign or gain per axis), for example through a model-based learner, not a fixed corrective gain.
- **Mazzoni & Krakauer and the nominal cost.** Implicit SPE-driven correction runs even when it conflicts with the goal. A compensator that always runs can hurt nominal performance when its estimate is noisy. A gate (inferred context: "no fault") is useful.

### Gaps
- Primary texts of von Holst & Mittelstaedt (1950) and Sperry (1950) were not fetched. They are cited here only through Crapse & Sommer. Standard bibliography from prior knowledge, not verified this session: von Holst & Mittelstaedt, *Naturwissenschaften* 37:464–476 (1950), "Das Reafferenzprinzip"; Sperry, *J Comp Physiol Psychol* 43:482–489 (1950).
- No quantitative effect sizes were obtained for Tseng 2007 or Mazzoni & Krakauer 2006; only abstract-level data was available.

## Q2. Cerebellar adaptation vs de novo learning: rotation and force fields vs mirror reversal

### Takeaway
Rotations and force fields (our "bias") are handled partly by fast implicit recalibration of an existing controller. Even so, that implicit part saturates at about 15° for a 45° rotation, and the rest comes from an explicit strategy. Mirror reversal (our "sign inversion") cannot be learned by adaptation: implicit adaptation pushes the wrong way and is gradually suppressed. People learn reversal de novo instead. They build a new controller that is slow, costs reaction time, leaves no aftereffects, improves offline like a skill, starts out explicit or deliberative, and only works at low frequencies at first. Both of our observed results (bias about 1/3 compensated, inversion zero) closely match the human pattern of what implicit learning alone can do.

### Cited Findings
- **Telgen, Parvin & Diedrichsen (2014), *J Neurosci* 34(41):13768–13779, "Mirror reversal and visual rotation are learned and consolidated via separate mechanisms: recalibrating or learning de novo?":** Abstract-level findings only.
  - Mirror reversal produced a shift in the speed-accuracy trade-off and offline gains between sessions: session 2 began better than session 1 ended. Rotation memory decayed between sessions.
  - Under mirror reversal, correct initiation and online corrections appeared only at longer latencies. The earliest feedback responses still went in the non-mirrored (wrong) direction even after two training sessions.
  - Under rotation, directional error was not linked to reaction time, and fast feedback responses adapted immediately.
  - Conclusion: mirror reversal requires acquiring a new control policy, while rotation recalibrates an existing one.
  - — [UWO repository](https://ir.lib.uwo.ca/brainpub/164/); [blog summary of abstract](https://getbrainpower.blogspot.com/2014/10/mirror-reversal-and-visual-rotation-are.html)
- **Mirror reversal has no aftereffects:** Citing work characterises mirror-reversal learning as leaving no aftereffects once the perturbation is removed, and attributes the offline gains to Telgen 2014. — [search summary citing core.ac.uk](https://core.ac.uk/download/289079705.pdf) (full text not accessible, 403)
- **Wilterson & Taylor (2021), *eNeuro* 8(4), ENEURO.0312-20.2021, "Implicit visuomotor adaptation remains limited after several days of training"** (n = 12 per experiment, 45° perturbation, 1 h/day for 5 days, 3,808 perturbed trials; aim reported each trial, implicit = hand − aim). — [PMC8362683](https://pmc.ncbi.nlm.nih.gov/articles/PMC8362683)
  - **Rotation:**
    - Hand angle was 44.8° on day 1 and 44.7° on day 5, so performance was near-perfect throughout.
    - Implicit: 16.7 ± 6.4° on day 1 and 13.9 ± 16.3° on day 5. Explicit: 28.1° and 30.7°.
    - The implicit component did not grow over 5 days; the modelled asymptote is about 15°, i.e. about 1/3 of the perturbation.
    - Washout aftereffects were well below 45° for every subject.
    - Individual differences were large: 2 subjects compensated almost entirely implicitly and 5 mostly explicitly.
  - **Mirror reversal:**
    - Day-1 implicit was 8.0 ± 4.6° in the rotation-like direction, which is wrong for a mirror. Explicit was 40.5°.
    - The implicit component declined (p = 0.049) to 2.2 ± 7.8° by day 5 (not different from 0, p = 0.357).
    - The task was solved by re-aiming (43.6° on day 5).
    - Extrapolating, the implicit process would need about 10–14 days just to reverse sign.
    - Re-aiming reaction time halved from 1.4 s to 0.7 s, which may indicate the strategy becoming procedural (cached).
- **Yang, Cowan & Haith (2021), *eLife* 10:e62578, "De novo learning versus adaptation of continuous control in a manual tracking task"** (n = 10 rotation at 90°, n = 10 mirror about a 45° axis; continuous tracking analysed by frequency-domain system identification). — [PMC8266385](https://pmc.ncbi.nlm.nih.gov/articles/PMC8266385); [eLife](https://elifesciences.org/articles/62578v1)
  - **Aftereffects:**
    - Rotation produced clear aftereffects: compensation angle 32.4 ± 1.4°. Aftereffects were about 25°, compared with about 70° late-learning compensation, so most rotation learning was not adaptation either.
    - Mirror produced negligible aftereffects (off-diagonal gains did not differ from baseline, p = 0.21).
  - **Frequency dependence:**
    - Mirror compensation was significant only at the lowest 3 frequencies. High-frequency gains did not flip and stayed at baseline, which the authors link to feedback delays.
    - Rotation was compensated at all frequencies, more completely at low ones.
  - **Delay and mechanism:**
    - Added lag was 83 ± 31 ms under rotation and 191 ± 62 ms under mirror, both under the roughly 300 ms expected from repeated re-aiming. Participants therefore built a new continuous controller.
    - De novo learning and adaptation can run in parallel. The authors propose that de novo control "proceeds initially through explicit processes" before becoming cached or automatized. How the new controller is built remains open.
  - **Sample size:** a follow-up with only 15 reaches (instead of 450) produced weaker compensation (F(1,252) = 37.69, p < 0.0001).
- **Hadjiosif et al. (2021):** Under mirror reversal, implicit adaptation increases error, and participants must recruit additional, likely explicit, mechanisms. — [PMC8018745](https://pmc.ncbi.nlm.nih.gov/articles/PMC8018745)

### Inferences
- **Why reversal cannot be adapted.** Implicit adaptation acts like a fixed-sign error-to-correction map: it moves the command opposite to the observed error, in a fixed coordinate frame. A sign inversion flips the plant's Jacobian, so that map pushes in the wrong direction and amplifies the error. A fix needs a qualitatively different controller (a new mapping), not a small parameter change. Fine-tuning that makes small changes to the pretrained policy is the ANN analogue of "adaptation". By this analogy it should not be expected to learn inversion. That matches our zero result.
- **Bias is about 1/3 compensated in both systems.** Our bias compensation (about 1/3) matches the human implicit ceiling for rotation (about 15/45°). This is suggestive, not mechanistic proof. In humans the rest comes from explicit or de novo processes, which our system lacks.
- **Design implications for inversion:**
  - Treat inversion as a separate module or a discrete latent, for example a per-axis sign s ∈ {+1, −1} that is inferred and then applied as a known transform to the action output. The VLA's nominal controller then never has to relearn it. This is the engineering analogue of "explicit re-aiming": a discrete, model-based re-mapping. Under an inversion fault the transform is exact, so it can be applied with no learning once s is identified.
  - Identifying s needs informative excitation. Humans initially pay reaction-time and latency costs (Telgen; Yang +191 ms). Expect a brief probing or identification phase.
  - Only the low-frequency (slow, quasi-static) parts of reversal are learned first (Yang 2021). A coarse, chunk-level correction is more realistic than high-bandwidth intra-chunk correction.
  - Data volume matters for de novo learning (15 vs 450 reaches: weaker learning). Human de novo learning takes days and benefits from consolidation.

### Gaps
- Telgen 2014 numbers (reaction times, the size of the speed-accuracy shift) were not retrieved because of the 403 on the full text.
- Haith & Krakauer reviews (e.g., Haith & Krakauer 2013, "Model-based and model-free mechanisms of human motor learning") were not fetched.
- No direct cerebellar-patient data on mirror reversal was retrieved.

## Q3. Multiple timescales, savings and context-dependent internal models (MOSAIC, COIN)

### Takeaway
Adaptation is well described by a small state-space model of internal offsets: a fast process that learns quickly and forgets quickly, plus a slow process that learns slowly and retains well. This model explains savings, spontaneous recovery and interference without touching the controller itself. Context-inference models go further: MOSAIC (paired forward/inverse modules weighted by how well each forward model predicts) and COIN (Bayesian inference over an unbounded set of contexts). In these models, much of apparent "learning" is really re-weighting stored memories by inferred context. For our faults, that means keeping the nominal controller intact, keeping a fault-specific module or offset for each fault type, and selecting or weighting them by inferred fault posterior from prediction errors.

### Cited Findings
- **Smith, Ghazizadeh & Shadmehr (2006), *PLoS Biol* 4(6):e179, "Interacting adaptive processes with different timescales underlie short-term motor learning":** — [PMC1463025](https://pmc.ncbi.nlm.nih.gov/articles/PMC1463025)
  - **Model:** Two states: fast (Af = 0.92, Bf = 0.03) and slow (As = 0.996, Bs = 0.004) in the illustrative simulations. Error e(n) = f(n) − x(n), and the output is the sum of the states. The qualitative results need only Bf several times larger than Bs and As closer to 1 than Af.
  - **Force-field fit** (14 participants, viscous curl field, error-clamp channel trials): A = 0.992, B = 0.02 for the slow-like process; A = 0.59, B = 0.21 for the fast-like process (95% CIs in the paper). The slow/fast mapping is inferred from the values.
  - **Spontaneous recovery:** After adaptation to field A and then brief exposure to B, error-clamp trials showed a rebound toward A (p < 0.0001). This matched the two-rate model, not single-state or gain-specific models.
  - **Slow process share:** It accounts for more than half of the adaptation by the end of the first learning block.
  - **Savings:** The latent slow state stays nonzero after washout, so relearning is faster.
  - **Anterograde interference:** The slow state is biased against learning the opposite field.
- **Wolpert & Kawato (1998), *Neural Networks* 11(7–8):1317–1329, "Multiple paired forward and inverse models for motor control":**
  - The architecture uses multiple pairs of inverse models (controllers) and forward models (predictors).
  - Each forward model's prediction accuracy sets how much its paired inverse model contributes to the motor command (its "responsibility"). The system learns both the controllers and how to select among them.
  - — [record](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/60269485)
- **Haruno, Wolpert & Kawato (NIPS 1998), "Multiple paired forward-inverse models for human motor learning and control":** Responsibilities also gate learning: each inverse model learns only when its paired forward model predicts well. Selection combines forward-model prediction errors with sensory feedback. — [NeurIPS](https://papers.nips.cc/paper_files/paper/1998/hash/228499b55310264a8ea0e27b6e7c6ab6-Abstract.html); [PDF](https://www.inf.ed.ac.uk/teaching/courses/mlsc/HW2papers/HarWolKaw.pdf)
- **Heald, Lengyel & Wolpert (2021), *Nature* 600:489–493, "Contextual inference underlies the learning of sensorimotor repertoires" (COIN):** — [PMC8809113](https://pmc.ncbi.nlm.nih.gov/articles/PMC8809113/); [lab page](https://cbl.eng.cam.ac.uk/lengyel/publications/heald-nature-2021)
  - **Model:**
    - Discrete latent contexts switch over time as a Markov chain.
    - Each context has its own state (e.g., force-field strength) with linear-Gaussian dynamics.
    - Hierarchical Dirichlet process priors cover transitions and cues, so the number of contexts is unbounded, and a "novel context" is always available.
    - Inference uses particle learning.
    - Motor output is the context-probability-weighted average of the context states.
  - **"Proper" vs "apparent" learning:** Proper learning is changing a stored state. Apparent learning is changing how strongly existing memories are expressed through predicted context probabilities. Memory creation, updating and expression are all controlled by a single computation: contextual inference.
  - **Re-explained phenomena:**
    - Spontaneous recovery: the earlier context is inferred to be active during channel trials. It is not explained by slow decay.
    - Savings: higher predicted probability of the re-encountered context, not a higher learning rate.
  - **New "evoked recovery" experiment:** Two P+ trials inserted into the channel phase produced strong, lasting recovery. The dual-rate and multi-rate models could not match its time course.
  - **Model comparison:** COIN was favoured by group BIC (ΔBIC 302.6 for spontaneous recovery, 394.1 for evoked recovery), and individually by 6 of 8 participants in each experiment.
  - **Memory updating:** Updating is weighted by responsibility. In a cue experiment with n = 24, single-trial learning was graded by cue and perturbation.

### Inferences
- **Mapping to our faults.**
  - A persistent bias fault corresponds to a "context" whose state is a 2-D offset vector. A COIN- or Kalman-style estimator can track the offset from SPEs (Δstate_measured − Δstate_predicted) and subtract it from the action. This is error-driven adaptation of an internal offset rather than retraining the VLA.
  - The dual-rate structure suggests two estimators: a fast one that tracks a new fault within a few chunks, and a slow one that holds a persistent fault. The fast estimator should decay so that transients do not leave residual offsets, which cause nominal cost.
- **Nominal cost as interference.** In MOSAIC and COIN the nominal controller is one context, so it is protected. Learning updates only the module that is responsible for the current data. Our single fine-tuned network shares all parameters across nominal, bias and inversion data. Gradient interference between contexts is the expected outcome, and it shows up both as nominal degradation and as diluted fault compensation.
  - Modular fix: freeze the base VLA, or keep a nominal adapter. Add fault-specific adapters (e.g., LoRA modules or residual heads), each mixed in by a responsibility score.
  - The responsibility score comes from how well each fault hypothesis's forward model predicts the observed Δstate.
- **Handling ambiguity.** COIN's output is the posterior-weighted average of context states, so a gate falls back toward nominal when evidence is weak. This should reduce the nominal cost seen when the network over-applies compensation. Responsibility-gated learning (Haruno 2001) also keeps one fault's data from corrupting other modules.
- **Inversion fits naturally.** In a modular scheme inversion is just another context, whose inverse model is a sign-flip transform. Selection is by prediction error: under inversion, the "sign-flipped forward model" predicts Δstate far better than the nominal one after one or two chunks, provided the commanded action is non-zero along that axis.

### Gaps
- COIN's full parameter values (transition hyperparameters, process noise) were not retrieved.
- No published robotics or VLA work applying COIN to actuator faults was searched for in this pass. That is a gap for the report writer.
- Exact MOSAIC responsibility equations were not retrieved from primary text. In the NIPS paper they are a softmax over the negative squared forward-model prediction error, per prior knowledge; not verified this session.

## Q4. Design implications for the artificial policy (synthesis)

### Takeaway
The literature points to a split architecture:
- a forward model trained on nominal transitions, producing a sensory prediction error;
- a small, explicit fault-state estimator for bias, a state-space or Kalman model with fast and slow components, that applies an additive correction outside the main policy;
- a discrete, context-inferred module for sign inversion, a de novo or "explicit" transform, selected by forward-model likelihood;
- a nominal controller that stays frozen or is protected.

Asking a single memoryless policy to absorb all of this through fine-tuning is the analogue of relying only on implicit adaptation. That is the regime in which humans also reach only about 1/3 compensation of a rotation and cannot learn mirror reversal at all.

### Cited Findings
- **Implicit cap:** Implicit SPE-driven adaptation alone plateaus at about 15° of a 45° rotation even after 5 days. — [Wilterson & Taylor 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8362683)
- **Mirror reversal needs other mechanisms:** It is not learnable by implicit adaptation, which pushes the wrong way. It needs explicit or de novo mechanisms. — [Hadjiosif et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8018745); [Wilterson & Taylor 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8362683)
- **De novo properties:** A de novo controller can run alongside adaptation, is initially limited to low frequencies, and adds about 190 ms of lag. — [Yang, Cowan & Haith 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8266385)
- **Recalibration vs new policy:** Rotation recalibrates an existing fast policy, while mirror reversal requires a new, slower policy that consolidates offline. — [Telgen et al. 2014 (abstract)](https://ir.lib.uwo.ca/brainpub/164/)
- **Forward models and SPE:** SPE, not motor correction, drives adaptation (Tseng 2007), and forward models must stay unbiased (Shadmehr, Smith & Krakauer 2010). — [Tseng 2007](https://research.bangor.ac.uk/en/publications/sensory-prediction-errors-drive-cerebellum-dependent-adaptation-o/); [Shadmehr 2010](https://pubmed.ncbi.nlm.nih.gov/20367317/)
- **Modular contexts:** Paired forward/inverse modules weighted by predictive accuracy (MOSAIC), and inference over contexts (COIN), explain savings and recovery without overwriting memories. — [Wolpert & Kawato 1998](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/60269485); [Heald et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8809113/)

### Inferences
These are concrete design choices, ranked by how directly the literature supports them.

1. **Explicit forward model and SPE.**
   - Train f(s_t, a_sent) → Δs on nominal data, as a head or a separate small network. Feed the SPE (Δs_meas − f(·)) to the adaptation pathway instead of the raw efference copy.
   - This removes the burden of learning the plant model from the policy, and gives a fault signal that is zero on nominal trials. That is a principled route to reducing the nominal cost.
2. **Bias as a filtered internal offset.**
   - Estimate b̂ with a two-rate state-space update, for example b̂_fast ← 0.6·b̂_fast + 0.2·SPE and b̂_slow ← 0.99·b̂_slow + 0.02·SPE. These magnitudes follow Smith 2006 force-field fits but are per-trial values, so they would need rescaling per chunk.
   - Subtract b̂ from the commanded action, or feed it as conditioning. This is a Kalman-style observer, and it can reach full compensation where implicit-style fine-tuning saturates.
3. **Inversion as a discrete context.**
   - Keep a hypothesis set {nominal, bias, flip-x, flip-y, …}, each with its forward model. Compute responsibilities, i.e. the posterior from the SPE likelihood under each hypothesis, possibly with a COIN-like sticky Markov prior because faults are persistent.
   - Apply the sign transform of the most probable hypothesis as a known output remap. The policy never "learns" inversion; it is identified, as in explicit re-aiming.
   - Expect identification to need non-zero excitation on that axis, and possibly a short probing cost (the human analogue: longer latencies, Telgen 2014).
4. **Protect the nominal controller.** Freeze or regularize the base VLA, and put compensation into gated adapters with responsibility-weighted updates (MOSAIC learning gating). This avoids the cross-context interference that produces nominal cost.
5. **Do not expect a sign-fixed error-to-correction mapping to generalize.** A learned compensator that behaves like "−k·error" will be destabilizing under inversion (Hadjiosif 2021).
   - Check whether our fine-tune learned such a mapping: does the compensation flip sign appropriately under the inversion fault?
   - Also check whether the 2-chunk history is enough to identify the inversion. Identifiability needs the commanded action component on that axis to be non-zero.
6. **Training curriculum.** De novo learning in humans is slow, data-hungry (450 vs 15 reaches) and consolidates. If a learned inversion module is used instead of a hard-coded transform, give it dedicated data and capacity rather than mixing it with bias training.

### Gaps
- No source directly tests these neuroscience-inspired architectures on robot actuator faults. The mapping from neuroscience to the VLA is analogical. Relevant robotics literature (e.g., RMA "Rapid Motor Adaptation", Kumar et al. 2021; adaptive control; fault-tolerant control) is outside this researcher's scope and was not searched.
- The specific suggested filter gains are extrapolations from per-trial human fits and are not validated for chunked robot control.

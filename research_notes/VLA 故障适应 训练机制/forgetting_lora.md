# Nominal-performance loss (forgetting / regression) when fine-tuning pretrained VLAs and LLMs, and remedies that preserve base behavior

Context: OpenVLA-OFT 7B (GRPO on LIBERO), LoRA r=32 on all LLM linears + lm_head, lr 5e-5, 2 epochs, 22.6k self-rollout samples from 10 tasks. Targets are replaced on faulted dims by an oracle compensation, and an extra input token is inserted. Clean success fell 0.963 -> 0.863, mostly on 2/10 tasks and in x/y action spread.

Research budget note: about 20 tool calls. Several numbers live only in paper figures and could not be pulled from the text; these are flagged under Gaps.

## Q1. Evidence of forgetting/regression when fine-tuning VLAs and LLMs (LoRA vs full FT, SFT vs RL)

### Takeaway
How much a model forgets is best predicted by how far the fine-tuned policy's output distribution moves from the base, measured as forward KL on the new-task distribution. Which parameters change, or whether LoRA is used, predicts it less well. On-policy RL with LoRA on a big pretrained VLA forgets almost nothing. SFT, including SFT on targets that differ from the model's own samples, can drift arbitrarily far. LoRA reduces forgetting on average, but this is not guaranteed, and it gets worse at high learning rates.

### Cited Findings
- **LoRA Learns Less and Forgets Less** (Biderman et al., TMLR 2024, Featured Certification). LoRA underperforms full FT on code and math targets but better preserves out-of-domain abilities. It reduces forgetting more than weight decay or dropout and keeps generations more diverse. Full FT learns perturbations of 10-100x higher rank than typical LoRA configs. — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
  - Forgetting metric (mean of HellaSwag/ARC-C/WinoGrande, higher = less forgetting):
    - Code IFT, epoch 16: full FT 0.414 vs LoRA r=64 0.510 / r=256 0.517.
    - Code CPT 20B tokens: full FT 0.545 vs LoRA r=256 0.617.
    - Math: essentially equal (IFT 0.567 LoRA vs 0.559 full; CPT 0.616 vs 0.618).
    - Smaller rank forgets less: code IFT epoch 16, r=16 0.609 vs r=256 0.517. — [HTML v2, Tables S2–S8](https://arxiv.org/html/2405.09673v2)
  - Learning while forgetting less: LoRA r=16 "learns less and forgets less than all other models". r=256 learns about as much as full FT with dropout or weight decay while forgetting less (Fig. 4, qualitative only).
  - Target modules: attention-only underperforms MLP or all modules; most gains come from the MLP modules (Fig. 7).
  - Recommendations: α=2r; LR sweep 1e-5..5e-4; the best LoRA LR is about 10x the full-FT LR (5e-4 code, 2e-4 math vs 5e-5 / 1e-5 for full FT). — [HTML v2](https://arxiv.org/html/2405.09673v2)
- **LoRA vs Full Fine-tuning: An Illusion of Equivalence** (Shuttleworth, Andreas, Torralba, Sharma, NeurIPS 2025).
  - LoRA introduces "intruder dimensions": high-singular-value singular vectors dissimilar to those in the pretrained matrix. Full FT stays spectrally close to the pretrained model.
  - Intruder-dimension count correlates with forgetting (reported Spearman ρ = 0.971). Larger LRs produce more intruder dimensions and more forgetting.
  - LoRA "in general does forget less … it is not a guarantee": with large LRs, LoRA sometimes forgot more than full FT.
  - Scaling down the singular values of intruder dimensions sharply reduced forgetting, with only a minimal drop in downstream test performance.
  - Intruder dimensions accumulate over sequential fine-tunes. — [arXiv 2410.21228](https://arxiv.org/abs/2410.21228); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/115207)
- **RL's Razor: Why Online RL Forgets Less** (Shenfeld, Jyothish Pari, Agrawal, arXiv Sep 2025). At matched new-task accuracy, RL preserves prior abilities far better than SFT. — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259); [HTML](https://arxiv.org/html/2509.04259)
  - Forgetting predictor: forward KL(base‖fine-tuned) on the new-task distribution. Quadratic-fit R² was 0.96 in ParityMNIST vs 0.93 for reverse KL, 0.80 for TV and 0.34–0.58 for weight-change and activation-change metrics. In LLMs, R² = 0.71.
  - Robotics: OpenVLA 7B in SimplerEnv, fine-tuned to pick up a can, with forgetting measured on open/close-drawer tasks. SFT on RT-1 successes is compared with REINFORCE using a binary reward and no KL term. The text says RL forgets less, but the retention values are only in figures.
  - Qwen2.5-3B-Instruct: representation similarity to the base (CKNNA) is 0.94 after RL vs 0.56 after SFT.
  - Key mechanism result: an "oracle SFT" on the KL-minimal distribution that solves the task forgets less than RL. SFT on RL-generated outputs matches RL's trade-off. Hence "the learned output distribution, not the optimizer, governs forgetting".
  - Recommendation: choose methods that explicitly minimize KL from the base.
- **Simple Recipe Works: VLAs are Natural Continual Learners with RL** (Hu, Shim, Tang, Sung, Liu, Stone, Martín-Martín; RLC 2026; best paper at ICRA 2026 RL4IL workshop). Sequential LoRA fine-tuning with on-policy RL (PPO/GRPO) of OpenVLA/OpenVLA-OFT shows "little to no forgetting" and preserves zero-shot generalization. It often beats dedicated continual-RL methods. Removing any one of the three ingredients (large pretrained model, LoRA, on-policy RL) restores classical forgetting. — [arXiv 2603.11653](https://arxiv.org/abs/2603.11653); [GitHub](https://github.com/UT-Austin-RobIn/continual-vla-rl)
  - Numbers on libero-spatial, as reported secondhand by The Batch (DeepLearning.AI):
    - Average success: 81.2% vs DER 73.4%, SLCA 69.9%, EWC 66.1%.
    - Forgetting (mean drop on earlier tasks): 0.3 pp vs EWC 0.7, DER 4.7, SLCA −0.6.
    - Unseen-task success: 57.1% vs 55.2% (DER) and 52.6% (EWC). — [The Batch](https://www.deeplearning.ai/the-batch/sony-and-university-researchers-train-robots-to-learn-without-catastrophic-forgetting)
- **Self-Distillation Bridges Distribution Gap in LM Fine-Tuning** (Yang et al., ACL 2024). The authors attribute fine-tuning's loss of general ability to the gap between the task-data distribution and the model's own distribution. SDFT rewrites the targets in the model's own words, which reduces forgetting on Llama-2-chat while matching or beating vanilla FT downstream, and helps keep helpfulness and safety. — [arXiv 2402.13669](https://arxiv.org/abs/2402.13669)
- **Self-Distillation Enables Continual Learning** (Shenfeld, Damani, Hübotter, Agrawal; ICLR 2026 / ICML 2026 spotlight). A demonstration-conditioned copy of the model acts as teacher, and the student matches it on-policy. This beats SFT on new-task accuracy while "substantially reducing catastrophic forgetting".
  - Offline distillation from the same teacher underperforms, so the benefit comes from being on-policy.
  - Cost: about 2.5x FLOPs and about 4x wall-clock vs SFT. — [arXiv 2601.19897](https://arxiv.org/abs/2601.19897); [ICML 2026](https://icml.cc/virtual/2026/poster/61434)
- **Knowledge Insulation (π0.5-KI)** (Driess et al., Physical Intelligence, NeurIPS 2025). Naively grafting a continuous action expert onto a pretrained VLM "significantly harms both training speed and knowledge transfer". The fix is to stop action-expert gradients from reaching the backbone and to train the backbone on discrete FAST tokens plus VLM co-training data.
  - Naive π0 needs about 7.5x more steps to reach comparable table-bussing performance. The method costs about 20% extra compute.
  - Fully freezing the VLM backbone gave 0% on shirt folding, so freezing everything is not a free remedy.
  - Removing VLM co-training data slightly lowers task completion and hurts language following; the values are only in figures. — [arXiv 2505.23705](https://arxiv.org/html/2505.23705v1); [PI blog](https://pi.website/research/knowledge_insulation)
- VLA continual-learning literature is split. A 2026 real-robot study ("Can VLAs Learn from Real-World Data Continually without Forgetting?", Zhu et al., arXiv 2605.26820) reportedly finds that naive sequential FT forgets severely and that a modest replay budget largely fixes it (search-result summary only). Separately, pretrained VLAs such as GR00T N1.5 on LIBERO have been reported as surprisingly forgetting-resistant (search summary). — [arXiv 2605.26820](https://arxiv.org/pdf/2605.26820.pdf)
- ExpReS-VLA (OpenVLA; experience replay plus retrieval and a mixing schedule) reports in-distribution success rising 84.7% -> 98% and OOD 32% -> 98% (authors' numbers, not verified). — [arXiv 2511.06202](https://arxiv.org/pdf/2511.06202v1)

### Inferences
- The setup is self-rollout SFT, so it is near on-policy on clean dims. But the oracle-compensation targets on faulted dims, plus the new token, make it SFT toward an off-policy distribution. Per RL's Razor, forgetting tracks the KL that this target shift induces. The 0.10 clean drop is consistent with the fine-tune moving the clean-context distribution even though clean targets equal the model's own samples.
  - One likely cause: the shared LoRA delta and lm_head change, learned for faulted contexts, leaks into clean contexts.
  - A second likely cause: the clean contexts in training now carry the inserted token, while evaluation-time clean behavior was never anchored to the base's distribution.
- Damage concentrated in 2 tasks and in x/y spread looks like "learn-less-forget-less" violated locally. Tasks that need precise translational positioning (drawer handle, rack placement) are the most sensitive to small widening of the action-token distribution. This is an inference; no paper ties forgetting to specific action dims.
- r=32 at 5e-5 is not a high LR for LoRA by Biderman's guidance, where the optimum is about 2e-4 to 5e-4. So the intruder-dimension high-LR failure mode is probably not the main cause. Including lm_head in the LoRA targets directly perturbs the output distribution of every context, and Biderman did not study lm_head adaptation.

### Gaps
- No exact OpenVLA retention numbers from RL's Razor's robotics figure (figure-only).
- "SFT Memorizes, RL Generalizes" (Chu et al., 2025) was not fetched in this session. Its numbers are not verified here.
- No paper found that measures nominal-success regression specifically for OpenVLA-OFT LoRA SFT on self-rollouts with partially relabeled targets. The OpenVLA/OpenVLA-OFT GitHub issues were not searched due to budget.
- Ablation magnitudes in Simple Recipe Works (no-LoRA, SFT instead of RL) are only described qualitatively ("collapse").

## Q2. Remedies with documented numbers

### Takeaway
The best-supported levers fall into four groups:
1. Keep the fine-tuned output distribution KL-close to the base on clean inputs: KL or distillation to the frozen base, self-distillation, on-policy data.
2. Rehearse nominal data.
3. Post-hoc shrink the update: WiSE-FT-style interpolation, or scaling down LoRA/intruder singular values.
4. Constrain capacity and LR: smaller rank, fewer steps.

Freezing everything except new modules preserves the base but can kill plasticity (π0 frozen backbone gives 0%).

### Cited Findings
- **KL minimization as the principle.** Forgetting is approximately a quadratic function of forward KL to the base (R² 0.96 toy, 0.71 LLM). Oracle SFT on the KL-minimal task-solving distribution forgets least of all methods. — [RL's Razor](https://arxiv.org/html/2509.04259)
- **Weight-space interpolation (WiSE-FT).** WiSE-FT ensembles zero-shot and fine-tuned weights, θ = (1−α)θ0 + αθ1. It adds 4–6 pp under distribution shift on ImageNet plus 5 shifts, and +1.6 pp on ImageNet itself. Gains are 2–23 pp on 6 further shifts and +0.8 to 3.3 pp on 7 transfer datasets, with zero extra training or inference cost. — [Wortsman et al., CVPR 2022, arXiv 2109.01903](https://arxiv.org/abs/2109.01903)
- **Shrinking the LoRA update spectrum.** Scaling down intruder-dimension singular values greatly reduces forgetting with minimal test-accuracy loss. — [Shuttleworth et al., NeurIPS 2025](https://arxiv.org/abs/2410.21228)
- **Smaller rank / fewer modules.** In code IFT epoch 16, r=16 has forgetting score 0.609 vs 0.517 at r=256 and 0.414 for full FT, but r=16 also learns less (HumanEval 0.358 vs 0.498 at epoch 4). — [Biderman et al.](https://arxiv.org/html/2405.09673v2)
- **Self-distillation / own-distribution targets.**
  - SDFT (ACL 2024) reduces forgetting vs vanilla FT on Llama-2-chat. — [arXiv 2402.13669](https://arxiv.org/abs/2402.13669)
  - On-policy SDFT (2026) beats SFT and offline distillation, at about 2.5x FLOPs. — [arXiv 2601.19897](https://arxiv.org/abs/2601.19897)
  - SFT on RL-generated outputs inherits RL's low forgetting. — [RL's Razor](https://arxiv.org/html/2509.04259)
- **On-policy RL + LoRA for VLAs.** OpenVLA-OFT continual GRPO+LoRA shows 0.3 pp forgetting vs DER 4.7 pp. — [The Batch summary of arXiv 2603.11653](https://www.deeplearning.ai/the-batch/sony-and-university-researchers-train-robots-to-learn-without-catastrophic-forgetting)
- **Gradient insulation and co-training.**
  - Stopping new-module gradients into the backbone preserves VLM knowledge and speeds training, with roughly 7.5x fewer steps than naive π0.
  - Fully freezing the backbone fails (0% shirt folding).
  - VLM co-training data helps language following. — [arXiv 2505.23705](https://arxiv.org/html/2505.23705v1)
- **Replay.** In the real-robot continual VLA study, a modest replay budget largely mitigates severe forgetting (summary only; budget % not extracted). — [arXiv 2605.26820](https://arxiv.org/pdf/2605.26820.pdf)
  - ExpReS-VLA replay+retrieval reports 84.7 -> 98% in-distribution. — [arXiv 2511.06202](https://arxiv.org/pdf/2511.06202v1)
- **Regularizers.** EWC achieves 0.7 pp forgetting in the OpenVLA-OFT CRL comparison, but lower plasticity (66.1% avg success vs 81.2%). — [The Batch](https://www.deeplearning.ai/the-batch/sony-and-university-researchers-train-robots-to-learn-without-catastrophic-forgetting)
  - Weight decay and dropout are weaker than LoRA itself at preventing forgetting. — [Biderman et al.](https://arxiv.org/abs/2405.09673)

### Inferences
Concrete, cheap remedies for the 0.963 -> 0.863 regression, in rough order of cost:
1. **Post-hoc LoRA scaling (WiSE-FT on the delta).** Evaluate W0 + s·BA for s ∈ {0.25, 0.5, 0.75}, and optionally leave the lm_head delta at s=0. This is zero training cost, and both WiSE-FT and intruder-dim scaling show large retention gains with small target loss. It directly tests whether the clean drop is "update magnitude" or "update direction".
2. **Distillation/KL to the frozen base on clean contexts.** For clean samples, use the base model's full action-token distribution on the original (token-free) input as a soft target, at the sampling temperature, rather than one-hot sampled actions. Hard sampled targets from the model's own rollouts are only an unbiased sample of the base distribution; with 2 epochs, the sample noise can still sharpen or widen x/y bins. Alternatively, add λ·KL(base‖student) on clean batches. This is the RL's Razor / SDFT prescription.
3. **Rehearsal ratio.** Ensure a large share of the batches are clean contexts, with the inserted token set to its "no-fault" value, and with base-distilled targets. Also hold out per-task clean probes for the two failing tasks as early-stopping criteria. The ~5–20% replay rule of thumb is generic and was only found on low-quality sites (not cited).
4. **Restrict trainable parameters.** Drop lm_head from the LoRA targets. Consider r=8–16, and possibly only the upper layers or MLPs, per Biderman (smaller r means less forgetting). Lower LR or epochs with early stopping on the clean probe.
5. **On-policy alternative.** A short GRPO phase with clean plus faulted episodes (reward = success) would, per Simple Recipe and RL's Razor, find a KL-minimal solution. It costs rollouts.

### Gaps
- No quantitative result found for KL-to-base regularization strength vs nominal success in VLA SFT specifically.
- Exact replay percentages from 2605.26820 could not be extracted (PDF parsing failed).
- Model merging beyond WiSE-FT (TIES, DARE, task arithmetic) was not researched due to budget.

## Q3. Effect of inserting a new token into a pretrained transformer's input, and remedies

### Takeaway
The documented remedy for adding new inputs to a frozen or pretrained LM is to make the new pathway contribute exactly zero at initialization, so the base function is preserved and the contribution grows only as needed. Examples are Flamingo's zero-initialized tanh gates and LLaMA-Adapter's zero-init gated attention. Position shift caused by a token inserted in the middle of the sequence is a separate effect. I found no direct quantitative study of it in this session.

### Cited Findings
- **LLaMA-Adapter** (Zhang et al., ICLR 2024). It adds learnable adaption prompts only in the higher transformer layers, through zero-initialized attention with a zero gate. It "adaptively injects the new instructional cues … while effectively preserving its pre-trained knowledge". It has only 1.2M trainable params on frozen LLaMA-7B, trains in under 1 hour on 8 A100s, gives responses comparable to fully fine-tuned Alpaca, and transfers to ViT and RoBERTa. — [arXiv 2303.16199](https://arxiv.org/abs/2303.16199)
- **Flamingo** (Alayrac et al., NeurIPS 2022). It inserts GATED XATTN-DENSE blocks between frozen LM layers. Each block has a layer-specific scalar α, initialized to 0, passed through tanh, so at init the model equals the frozen LM. — [Flamingo arXiv 2204.14198](https://arxiv.org/abs/2204.14198) (paper body; zero-init detail confirmed via secondary sources such as [Flamingo notes](https://awesome.papernotes.org/en/era4_foundation_models/2022_flamingo/)). The ablation table includes a tanh-gating row (overall score 66.5 in one extracted copy), but the exact baseline-vs-ablation pairing could not be verified.
- Freezing the whole backbone and training only new parts can preserve the base but lose plasticity: the π0 frozen-VLM variant scored 0% on shirt folding. — [arXiv 2505.23705](https://arxiv.org/html/2505.23705v1)

### Inferences
- If the extra token is inserted in or before the action-token span, every later token's RoPE position shifts by +1. Clean contexts then become out-of-distribution for the base weights even when the token's content is "no fault". The LoRA must partly spend capacity "re-learning" clean behavior under the shift. Remedies (inference, standard practice):
  - (a) Append the token where it does not shift the action/proprio positions the base relied on. Or reuse a position id, or keep the same position ids as the base for all original tokens.
  - (b) Zero-init the new token's projection/embedding with a learnable tanh gate, à la Flamingo or LLaMA-Adapter. At step 0 and on clean inputs, the fine-tuned model is then exactly the base, provided positions also match.
  - (c) Verify this before training: the fine-tuned model at init (LoRA B=0) with the token inserted should reproduce base clean success (~0.96). If it doesn't, part of the 0.10 drop is the insertion itself rather than forgetting.
- Training clean contexts with the "absent/neutral" token value and distilling to the base's token-free outputs (Q2 item 2) directly targets this shift.

### Gaps
- No paper found quantifying success loss from mid-sequence token insertion or position shift in VLAs or LLMs. This needs a direct ablation (LoRA B=0 + token vs no token).
- The Flamingo tanh-gating ablation numbers are not verified from the primary PDF.

## Q4. Recipe implications for preserving nominal success within 0.05

### Takeaway
Aim to minimize the KL from the base on clean contexts, not just to use the same clean targets. Make the new token a no-op at init, and use post-hoc delta scaling as a cheap dial. Together these address every documented forgetting mechanism found above.

### Cited Findings
- Forgetting ≈ f(KL on new-task distribution); output distribution, not the optimizer, governs it. — [RL's Razor](https://arxiv.org/html/2509.04259)
- LoRA + on-policy RL on OpenVLA-OFT gives about 0.3 pp forgetting on LIBERO. — [The Batch / 2603.11653](https://www.deeplearning.ai/the-batch/sony-and-university-researchers-train-robots-to-learn-without-catastrophic-forgetting)
- Weight interpolation recovers robustness at zero cost. — [WiSE-FT](https://arxiv.org/abs/2109.01903)
- Scaling down LoRA's large intruder directions reduces forgetting. — [Shuttleworth et al.](https://arxiv.org/abs/2410.21228)
- Zero-init gated new inputs preserve pretrained knowledge. — [LLaMA-Adapter](https://arxiv.org/abs/2303.16199)

### Inferences (proposed recipe, ordered gates)
1. **Gate 0 (insertion only):** LoRA B=0 with the token inserted gives clean success ≥ 0.95. If it fails, fix positions or zero-init gating first.
2. **Gate 1 (post-hoc):** sweep the LoRA scale s on the existing checkpoint, with lm_head delta on vs off. Pick the largest s with clean success ≥ 0.913 (0.963 − 0.05) and report the fault-recovery trade-off curve.
3. **Retrain:**
   - Clean-context loss = soft KL/CE to the frozen base's distribution on the token-free input, at the sampling temperature.
   - Use a clean:faulted ratio of at least 1:1.
   - Drop lm_head from the LoRA targets; use r=16; use 1 epoch with early stopping on per-task clean probes, especially the drawer and bottle-rack tasks.
   - Optionally add KL(base‖student) on the non-faulted dims inside faulted contexts too, since the x/y spread regression suggests leakage to non-target dims.
4. **If still > 0.05 gap:** a short GRPO/on-policy refinement, or on-policy self-distillation (SDFT-2026 style), with clean episodes in the mix.

### Gaps
- None of the remedies have been validated quantitatively in this exact setting (OpenVLA-OFT, oracle-relabeled fault dims, inserted token). The 0.05 target needs empirical checks via the gates above.

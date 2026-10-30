# X6 reframe, step 1a: model checks

2026-10-30. Read-only checks of the repo (configs, docs, model code) and of public pages, for the cross-architecture fault benchmark (`next_steps_fable_reframe.md` §2.3). Nothing was run. `[inference]` marks a conclusion read from code that the first GPU smoke must confirm.

| model | checkpoint | inputs per call (RLinf LIBERO path) | chunk | memory across calls | use |
|---|---|---|---|---|---|
| OpenVLA-OFT GRPO-130 | `RLinf/RLinf-OpenVLAOFT-LIBERO-130` (on the Volume, registered for X4) | 1 image + instruction | 8 | none | done at n = 16 (pilot); n = 300 in the minimal tier |
| OpenVLA-OFT Base-Lora | `RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora` (on the Volume) | same | 8 | none | second base (solid tier) |
| π0 | `RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT` (`examples/embodiment/config/model/pi0.yaml`: `model_type: openpi`, `num_images_in_input: 2`, `use_proprio: True`, `num_action_chunks: 10`, `num_steps: 4`) | main + wrist image, state, instruction | 10 | none | the extra architecture of the minimal tier |
| π0.5 | `RLinf/RLinf-Pi05-LIBERO-SFT` | as π0 (`pi0_5.yaml`) `[TO VERIFY]` | — | none | optional |
| GR00T N1.5 | `RLinf/RLinf-Gr00t-SFT-Goal` (`gr00t.yaml`: `num_action_chunks: 5`) | images + state | 5 | none | solid tier. **Its documented LIBERO-Goal success is 48.2%** (`docs/.../embodied/gr00t.rst`), so clean success caps what its fault curve can show |
| DreamZero 5B | `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers` (`dreamzero_5b.yaml`; eval `evaluations/libero/libero_spatial_dreamzero_eval{,_sglang}.yaml`, 16 actions per call) | 1 frame + state + instruction | 16 | see below | solid tier, own box |
| WoVR-trained OFT | **not released.** The `RLinf/wovr` collection holds only the Wan world models (`RLinf-Wan-LIBERO-{Spatial,Object,Goal,Long}`, three `-Fullshot` variants); `wovr.rst` reports the policy numbers but links no policy checkpoint | — | — | — | out of every affordable tier; training it is a multi-GPU GRPO job |

**RL siblings.** The docs list RL checkpoints for π0/π0.5 on CALVIN, ManiSkill, MetaWorld and RoboTwin (`RLinf-Pi0-*-RL-FlowNoise/FlowSDE`, `RLinf-Pi0-RoboTwin-PPO-*`). There are none for LIBERO, so an SFT-vs-RL contrast on LIBERO exists only for OFT (E4, X1).

**DreamZero's memory.**
- **Upstream code:** the action head (`dreamzero0/dreamzero` @ `ab790c1`, `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`) keeps a KV cache of past *video frames* across calls (`self.kv_cache1`, `self.current_start_frame`).
  - **No past actions:** the cache update runs with `action=None`, and action noise is regenerated every call.
  - **Reset rules:** it resets the cache on a new or changed instruction, when the frame counter reaches `local_attn_size`, and when the input video has a single frame.
- **RLinf's LIBERO path:** it passes one frame per call. `_rollout_assign_video` takes `main_images` [B, H, W, C] as T = 1, and no frame stacking exists in `rlinf/envs` or `rlinf/workers`. So the cache is reset on every call, and DreamZero is memoryless here `[inference]`.
- **What this would mean:** even a world-model policy that *can* carry visual history (a) never carries its own commands, and (b) runs without history in the standard evaluation. Both strengthen the categorical prediction; (b) must be confirmed by logging `current_start_frame` in the smoke.
- **Cost:** DreamZero inference runs either in the rollout worker or through an SGLang server built from an unmerged SGLang PR (`dreamzero_sglang.rst`). Either way it needs its own environment (`install.sh embodied --model dreamzero`) `[cost TO VERIFY]`.

**Evo-1** (`evo1.yaml`: `state_dim: 24`, 14 actions per call): it takes the state, not past commands. Not pursued (no LIBERO-Goal checkpoint was checked).

**Consequence for the plan.**
- **Minimal tier:** OFT (n = 300) and π0 for the cross-architecture curve.
- **Solid tier:** GR00T (with the 48% caveat) and DreamZero.
- **Dropped:** WoVR-OFT from every tier.
- **Harness work implied:** a capture-off mode, a model-yaml name that differs from `model_type` (`pi0.yaml` holds `model_type: openpi`), per-model chunk length and images, and robust logging when `prev_logprobs` / capture are absent.

# Wave 3 runbook — the merged line-A / line-B plan

What to run, in order, to fill `PREREG_wave3.md`. E1 is CPU-only and is
already run from this repo (`experiments/gru_mismatch/results/e1/`); E2–E5
need the 24 GB GPU box. Earlier results decide later steps, so stop at each
gate.

```
E1 (CPU, done here) ─┐
E2 (GPU, RL control) ┼─> E1 decides which of E4 / E5 is the main line-A run
E3 (GPU, WM gate) ───┴─> pass: E4      fail: stop line B, tell Jiazhuo
E5 step 1 (VRAM)     ──> pass: append the E5 design to the prereg, then train
```

## 0. Machine, installs, downloads

One venv covers LIBERO, OpenVLA-OFT and the Wan world model:

```bash
bash requirements/install.sh embodied --model openvla-oft --env wan
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD:$PYTHONPATH
P=experiments/efference_probe
```

Checkpoints (all public on Hugging Face):

| what | repo | used by |
|---|---|---|
| SFT policy (already used in waves 1–2) | `Haozhan72/Openvla-oft-SFT-libero-goal-traj1` | E2, E4, E5 |
| GRPO sibling of that policy | `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal` | E2 |
| Wan world model (+ VAE, dataset/) | `RLinf/RLinf-Wan-LIBERO-Goal` | E3, E4 |

```bash
SFT=/abs/path/Openvla-oft-SFT-libero-goal-traj1
RL=/abs/path/RLinf-OpenVLAOFT-GRPO-LIBERO-goal
WM=/abs/path/RLinf-Wan-LIBERO-Goal
```

Both policies use `unnorm_key: libero_goal_no_noops` (checked in the GRPO
checkpoint's `dataset_statistics.json`) and temperature 1.6.

## 1. E2 — RL control

```bash
python $P/run_collect.py --config $P/configs/e2_sft.yaml --set model.model_path=$SFT
python $P/run_collect.py --config $P/configs/e2_rl.yaml  --set model.model_path=$RL
python $P/run_collect.py --config $P/configs/e2_rl_mirror.yaml --set model.model_path=$RL
python $P/run_collect.py --config $P/configs/e2_rl_freeze.yaml --set model.model_path=$RL

for r in e2_sft e2_rl e2_rl_mirror e2_rl_freeze; do
  python $P/run_probes.py --run $P/data/$r --stage main --jobs 16 --permutations 200
done
python $P/compare_runs.py \
    --run sft:none=$P/data/e2_sft --run rl:none=$P/data/e2_rl \
    --baseline sft:none --out $P/data/e2_compare
python $P/compare_runs.py \
    --run sft_mirror=$P/data/main02_mirror --run rl_mirror=$P/data/e2_rl_mirror \
    --run sft_freeze=$P/data/main02_freeze --run rl_freeze=$P/data/e2_rl_freeze \
    --out $P/data/e2_compare_arms
```

Gates before probing: `manifest.json` → `verification.passed: true`, and the
probe pass reports `plumbing.passed: true`. Budget guide: main02 took
0.71 GPU-h for 200 episodes; e2_sft (500 episodes) ≈ 1.8 GPU-h, the GRPO runs
less (episodes end earlier). Linear probing ≈ 10–15 min per run at
`--jobs 16`; the optional MLP pass (`--probe-family mlp`) ≈ 2–2.5 h per run.
Use `--jobs` equal to the CPUs you actually have — `-1` resolved to 192 on a
16-CPU pod in wave 2.

Read `e2_compare/compare.md`: the attribution table (each run against its
own floor) and the compensation table are separate on purpose.

## 2. E3/E4 collection (substep frames)

```bash
for t in swap mirror freeze; do
  python $P/run_collect.py --config $P/configs/e4_wm_$t.yaml --set model.model_path=$SFT
done
python $P/run_wm.py check-orientation --run $P/data/e4_wm_swap --wm-ckpt $WM
```

Every substep is rendered and saved (≈ 8 JPEGs per call at q95, roughly
0.5 GB per run). Rendering all substeps is slower than waves 1–2; the
6 GPU-h cap in the configs is a guess, not a measurement — check the first
heartbeat. `check-orientation` must print `orientation matches` (LIBERO frames
are rotated 180° by `get_libero_image`; the WM was trained on RLinf rollouts,
which should match — this verifies it).

## 3. E3 — the line-B gate

```bash
python $P/run_wm.py sensitivity --run $P/data/e4_wm_swap --wm-ckpt $WM \
    --out $P/data/e3_sensitivity --n-queries 300 --max-hours 3
```

Each query generates 11 candidates one at a time at a fixed seed (batching
would change each sample's noise). Rows are appended to `rows.csv`; rerunning
the same command resumes. Read `summary.json → gate`: `eligible` needs
substeps context and ≥ 200 queries; `passed` applies
`median r_freeze ≥ 1 and median r_swap ≥ 1`.

Optional, exploratory only (never the gate): the same on wave-1/2 frames,
`--run $P/data/main02 --context-mode static`.

If the gate fails: stop line B and send Jiazhuo the summary.

## 4. E4 — world model as comparator

```bash
for t in freeze swap mirror; do
  python $P/run_wm.py comparator --run $P/data/e4_wm_$t --wm-ckpt $WM \
      --out $P/data/e4_comparator_$t --max-hours 4
done
```

FREEZE first: it is the decisive condition. `summary.json → per_transform →
<arm> → verdict` applies the pre-registered rule; the AUROCs and the
`adv_distance-novelty_adv` difference carry their bootstrap CIs.

## 5. E5 step 1 — VRAM

```bash
python $P/vram_probe.py --model-path $SFT --out $P/data/e5_vram.json
```

`gate.passed` reads the `prev_chunk` variant (prompt + 56 previous-command
tokens) at batch ≥ 4 against 22 GB. Pass → write the E5 training design as a
dated amendment in `PREREG_wave3.md` before training anything.

## 6. Who gets what

| result | goes to | in the follow-up, one new number |
|---|---|---|
| E1 | Evan Ye | primary cell verdict, and the exp3 6.5% vs ~12.3% question |
| E2 | Hongzhi Zang (and Haozhan Li, B list) | RL margin vs its floor + hijack cost RL − SFT |
| E3/E4 | Jiazhuo Li | median r_freeze, then FREEZE AUROC comparator vs novelty |

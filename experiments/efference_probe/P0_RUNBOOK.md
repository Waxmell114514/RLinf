# P0 re-collection runbook

Fills the three `[Pending]` slots that block the paper's next revision:
**nonlinear (MLP) probes** on a fresh main-run replica, **multi-suite** swap
arms, and **3x control-arm expansion** for mirror/freeze.  Everything here
reuses the existing pipeline; the only new machinery is
`--probe-family mlp` on `run_probes.py`.

Total budget if all six runs are collected: **~10-14 GPU-hours** on one
24 GB workstation card (reference: main01 cost 0.71 GPU-h for 300 episodes)
plus a few CPU-hours of probing.  Disk: keep ~30 GB free per concurrent run;
`hidden/` is the bulk and is deleted only after the analysis dirs are
exported.

## 0. Machine and environment

- One 24 GB GPU (RTX PRO 4000 class or better), >= 10 vCPU, >= 60 GB volume.
- Follow `rental_runbook.sh` / `README.md` for the install; nothing new is
  required -- the MLP probes use scikit-learn, which is already a
  dependency.
- Per-suite checkpoints: the multi-suite arms each need that suite's
  OpenVLA-OFT SFT checkpoint, and `model.unnorm_key` in the config must
  match the suite (already set in the YAMLs).  Do not point a goal-suite
  checkpoint at another suite.

## 1. Collection (GPU)

Run in this order -- the replica first, because everything downstream
(MLP ladder, per-transform, paper tables) reads it:

```bash
cd $REPO_PATH
P=experiments/efference_probe
GOAL_CKPT=/abs/path/to/Openvla-oft-SFT-libero-goal-traj1

# 1. main-run replica, fresh seed, hidden states kept (the MLP data source)
python $P/run_collect.py --config $P/configs/main02.yaml \
    --set model.model_path=$GOAL_CKPT

# 2. control-arm expansion (150 episodes each, pre-registered task set)
python $P/run_collect.py --config $P/configs/main02_mirror.yaml \
    --set model.model_path=$GOAL_CKPT
python $P/run_collect.py --config $P/configs/main02_freeze.yaml \
    --set model.model_path=$GOAL_CKPT

# 3. multi-suite swap arms -- each suite has its OWN checkpoint
python $P/run_collect.py --config $P/configs/suite_spatial.yaml \
    --set model.model_path=/abs/path/to/spatial-oft-sft-ckpt
python $P/run_collect.py --config $P/configs/suite_object.yaml \
    --set model.model_path=/abs/path/to/object-oft-sft-ckpt
python $P/run_collect.py --config $P/configs/suite_long.yaml \
    --set model.model_path=/abs/path/to/libero10-oft-sft-ckpt
```

Gate after every collection, before spending more GPU time: the manifest's
`verification.passed` must be true (the runtime indexing self-check), and
the first probe pass below must report `plumbing.passed: true`.

## 2. Probing (CPU; run per collected run)

```bash
R=$P/data/main02   # repeat for each run directory

# linear ladder + controls + E1/E3 (writes <run>/analysis)
python $P/run_probes.py --run $R --stage main --jobs -1 --permutations 200

# MLP ladder (writes <run>/analysis_mlp; expect 5-15x the linear runtime)
python $P/run_probes.py --run $R --stage main --jobs -1 --probe-family mlp
```

Notes:
- The MLP pass floors itself with a **selection-matched MLP P0** over the
  same 27 cells; never compare MLP rungs against the linear floor.
- The MLP pass deliberately skips cross-task transfer and the E1 ridge
  readout (linear-family diagnostics; the linear pass produces them).
- For the smaller arms (`main02_mirror`, `main02_freeze`, suites), the same
  two commands apply unchanged.
- `--permutations` on the MLP pass is legal but slow; the selection floor is
  the primary null there, so it is fine to omit.

## 3. What the paper consumes

Per run: `analysis/probe_results.csv`, `analysis_mlp/probe_results.csv`
(the `family` column distinguishes them), `analysis*/summary.md`,
`plumbing.json`, `run_card.csv`, and `calls.parquet`.  Export everything
except `hidden/` (as for main01); keep `hidden/` on the volume until the
nonlinear numbers are in the paper, in case a follow-up probe family is
wanted.

## 4. Pre-registered readout (fixed before looking)

- **MLP ladder on main02**: P1/P3/P4 vs. the MLP selection floor.  Above
  floor => the linear-only qualifier falls and the paper's claim narrows to
  readout format; at floor => "no linearly readable" strengthens to "no
  readable (linear or one-hidden-layer MLP)".
- **Linear ladder on main02**: independent replication of main01's null
  (fresh seed); its floor and P2r should reproduce within fold noise.
- **Expanded mirror/freeze**: do the ~0.58 hidden-probe excursions shrink
  toward the floor (as they did with n in the swap arms), and does mirror's
  E3 log-prob trend reach or lose significance at n~130 episodes?
- **Suites**: per-suite ladders vs. per-suite floors; the claim generalises
  only if the pattern (P2r well above floor, hidden probes at floor) holds
  per suite, not on any pooled number.

Deferred (out of P0 scope): token-level attention probes would need
`capture.py` to store the 56 action-window token states, ~1-8 GB per run at
2-3 layers; revisit only if the MLP result makes format questions sharper.

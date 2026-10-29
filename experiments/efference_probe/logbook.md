# Research logbook — efference-copy probe

Append-only. `run_collect.py` writes here automatically (START / heartbeat /
DONE / FAILED lines); add human entries by hand, including honest time
accounting per the MATS 12.0 admissions doc (SPEC §11).

Format: `- timestamp  what happened`. Include the config hash for anything that
produced data, so a number in `results.md` can always be traced back to a run.

## Time log

| Date | Hours | What |
|---|---|---|
| | | |

## Entries

- `2026-08-30 09:36:58` Repository scaffolding written at commit `dd92c62` (agent). No runs yet — no GPU in the authoring environment; offline test suite passes (31 tests).
- `2026-08-30 10:15:15` Two independent Opus code reviews (integration + methodology). Both confirmed the `[-57:-1]` readout-window correction. Fixed: undo-alignment measuring command persistence rather than correction; run rows lost on any crash; budget-truncated episodes counted as failures; block-scaling control was a no-op; E3 Wilcoxon treating within-episode pairs as independent; layer/pool selection inflation unreported; axis-angle delta wrapping; bf16 ctx_mean denominator; verify_indexing tolerance not dtype-aware; per-call GPU temporaries ~125x larger than needed. Corrected an over-general README claim: P2 fails only for mean-preserving transforms (swap), not for mirror/freeze. 40 offline tests pass.
- `2026-09-01 05:30` **T-CODE (agent).** Rescue archive verified against the fork: applying `efference-probe-working-tree.patch` to bundle `main@1580d364` reproduces `origin/main@657a86e7` **byte-identically (sha256, all 6 files)**, so the patch was already pushed and no code was lost. `local-efference_probe/` is content-identical to the repo (mod CRLF), not an older snapshot. Branch restarted from the fork's main (PR #1 already merged).
  Offline suite **47 passed** on Linux/py3.11 (43 pre-existing + 3 new). *Not run on native Windows* -- no Windows host available here; instead the Windows-only failure modes were closed: 11 text reads/writes had no `encoding=`, so a UTF-8 `run_config.yaml` written on a Linux box would fail on a cp936/cp1252 laptop. `sched_getaffinity` guarded. Verified no POSIX-only imports, no `os.rename`/symlink/subprocess/tempfile, npz handles closed via context managers, and the analysis stack imports with torch absent.
  **Perf.** Synthetic fixture generated at the lost run's shape (200 eps, 5,541 calls, 789 `swap` hijacks, hidden `[30,9,3,4096]`, 1.1 GB; cf. real 200/5,887/728/`[40,9,3,4096]`/1.109 GB). Full `--stage main` ladder, 4-core box: **serial ~35 min** (ladder 16.3 + controls 2.3 + E1 15.8) -> **parallel 9 min 53 s** (`ELAPSED_S=593`, ladder 2.78 + controls 1.48 + E1 5.52). Meets the <=10 min gate, and that run was slowed for ~4 min by orphaned workers, so it is an upper bound. **All 113 ladder cells identical serial vs 4-worker** at this scale; unit tests pin ladder and E1 parallel==serial.
  Changes: cells dispatched via joblib (`--jobs`, default -1; library default serial), one duplicate logistic fit per fold removed (the per-C sweep already fits the selected C), one shared `HiddenStore` across ladder/controls/E1/permutations (each previously re-decompressed all 1.1 GB; ~16 s locally, far worse over NFS), E1 parallelised, per-cell progress logging kept.
  **Probe job 539497 (4 h 51 m -> 7 P0 cells) not reproduced.** Locally a P0 cell is 2.87 s (1 BLAS thread) / 6.48 s (4), i.e. the cluster rate is ~290-850x slower. Ruled out: no convergence warnings; cache cap 8 GB > 1.8 GB working set so no thrashing; OpenBLAS self-caps at core count, so `OMP_NUM_THREADS` 16/64 did **not** reproduce a slowdown. Given `remote-permission-and-accounting.txt` documents the `peilab` group loss and the job was user-cancelled (not failed) with MaxRSS only 1.42 GB and no error, the pattern is most consistent with the job stalling on NFS reads when access was lost mid-job -- the same root cause as the jailed data -- rather than a code defect. Recorded as unexplained-but-mitigated, not diagnosed.
  `rental_runbook.sh` added (preflight/bootstrap/checkpoint/smoke/collect/archive on the box; `verify` on the laptop). Laptop half tested end-to-end: passes a good archive (exit 0) and fails a corrupt checksum and a failed indexing check (exit 1). GPU-side phases **untested** -- no box. Gates are the V2.2 operative set; no `P2 >= 0.9` anywhere.
- `2026-09-11 12:40` **T-CODE (agent).** P0 re-collection wave prepared (no runs yet; pod terminated, hidden/ not retained from wave 1). Code: `--probe-family {linear,mlp}` threaded through `run_probe`/`AnalysisConfig`/`run_probes.py`; the MLP rung is a one-hidden-layer (256 ReLU) `MLPClassifier` and the sweep swaps **every** probe including P0, so the MLP ladder is floored by a selection-matched MLP null. Solver finding, measured on an XOR benchmark (n=400, 24 dims, 2 informative): sklearn's default adam+early-stopping stops at chance (val sits at chance through the plateau, fires ~epoch 30, final 0.52; without early stopping 0.72 after 300 epochs), lbfgs solves it in seconds -- lbfgs it is; a real-shape fit (1716x4096) is ~3.4 s. MLP outputs land in `analysis_mlp/`; cross-task and E1 stay linear-only diagnostics. Configs added: `main02` (fresh-seed replica, sampled protocol, seed 1), `main02_{mirror,freeze}` (5 pre-registered tasks x 30 states = 150 eps each), `suite_{spatial,object,long}` (suite-specific checkpoints + unnorm_keys; libero_10 at 64 calls = 512 steps). All six pass `load_config`. Offline suite 56 tests.
  **Pre-registered readout (declared before any P0 data exists):** (1) main02 MLP ladder vs its MLP selection floor -- above floor narrows the paper's claim to readout format, at floor upgrades "no linearly readable" to "no readable (linear or one-hidden-layer MLP)"; (2) main02 linear ladder is an independent replication of main01 (floor and P2r should reproduce within fold noise); (3) expanded mirror/freeze -- do the ~0.58 hidden-probe excursions shrink toward floor with n, and does mirror's E3 log-prob trend (-1.66, p=0.16 at 44 eps) resolve either way at ~130 eps; (4) suites judged per-suite against per-suite floors, never pooled. Runbook: `P0_RUNBOOK.md`.
- `2026-09-12 06:12` **T-CODE (agent). Phase 0 — wave-2 box brought up.** RunPod
  container `0f5fdb34f254`, 1x **NVIDIA RTX PRO 4500 Blackwell, 32 GB VRAM**,
  driver 580.167.08 / CUDA 13.0. Repo cloned to `/workspace/RLinf`, branch
  `claude/rlinf-code-scripts-pi41dg` at `cc5bc289`. Only `/workspace` persists
  (a MooseFS network volume); `HF_HOME=/workspace/hf`,
  `UV_CACHE_DIR=/workspace/uv`, uv-managed CPython at
  `/workspace/uv/python`.
  **Install** per `rental_runbook.sh` bootstrap: `requirements/install.sh
  embodied --model openvla-oft --env libero`, torch pinned by the repo
  (`torch==2.11.0+cu130`, routed through the cu130 index by install.sh's own
  driver probe), transformers 4.40.1 from `moojink/transformers-openvla-oft`,
  flash-attn 2.8.3+cu13torch2.11. Analysis extras: scikit-learn 1.9.1,
  pandas 2.3.3, numpy 1.26.4, joblib 1.6.0, pyarrow, matplotlib, pytest.
  Three install-time obstacles, all environmental, all resolved:
  (1) the pod's uv 0.9.0 has no cpython-3.11.14 in its python-build-standalone
  manifest, so a standalone uv 0.12.13 was installed to `/workspace/bin` and
  used to fetch the pinned interpreter rather than relaxing the pin;
  (2) the pod template exports `HF_HUB_ENABLE_HF_TRANSFER=1` while `uv sync`
  prunes `hf_transfer` out of the venv, so `libero-download-assets` failed five
  times with `ValueError: Fast download using 'hf_transfer' is enabled ... but
  'hf_transfer' package is not available`; installing the package does not
  survive the next sync, so the variable is unset in the session env instead
  (this is exactly the "template version leaking in" failure mode);
  (3) rendering: only `libEGL_nvidia.so.0` shipped, no glvnd `libEGL.so.1`, so
  `libegl1 libgl1 libglvnd0 libgles2 libglx0 libosmesa6 libglew2.2 libglfw3`
  were installed from apt. **Rendering is EGL** (`MUJOCO_GL=egl`,
  `PYOPENGL_PLATFORM=egl`), not the runbook's osmesa default.
  **Offline suite: 56 passed** — after one test fix. `test_resolve_n_jobs_
  maps_negatives_onto_core_count` asserted `_resolve_n_jobs(-99) == 1`, true
  only below 99 cores; this box reports `os.cpu_count() == 112`, so -99 is a
  legitimate 14-worker request and the assertion failed on a healthy install.
  Fixed by scaling the degenerate request off `cores` (commit `cfc8ea5a`,
  `test(experiments): scale the degenerate --jobs case off the host core
  count`, Signed-off-by, touches only `tests/test_offline.py`). **Not pushed**:
  the box has no GitHub credentials and no `gh`; the patch is carried in the
  wave-2 export instead.
  **Two box constraints recorded before any collection.** (a) `os.cpu_count()`
  is 112 but the cgroup CPU quota is `2380000/100000` = **23.8 CPUs**, so
  `--jobs -1` resolves to 112 workers against a 23.8-core budget — the ladder
  will be timed both ways on the first real run before the rest of the wave is
  probed. (b) `/proc/meminfo` reports 251 GB but `memory.max` is
  **61999996928 B = 62 GB**, below the runbook's 96 GB threshold, so the
  documented "halve the env count" branch applies: **`env.num_envs=10` for all
  six runs** instead of the configs' 20. This changes batch partitioning and
  hence the per-episode hijack schedule (the RNG is keyed on
  `(seed, task_id, batch_index, 0xEFFE)`), but not the episode set, the
  protocol, or any probe input; main02 is a fresh-seed replication in any case,
  so the schedule is a fresh draw either way. Held constant across all six
  arms so the arms stay comparable.
  **Checkpoints (Phase 1).** All four LIBERO suites resolved to the same
  release family the configs' placeholder paths name, under `Haozhan72`, not
  `RLinf`: `Openvla-oft-SFT-libero-{goal,spatial,object}-traj1` and
  `Openvla-oft-SFT-libero10-traj1`; the `RLinf/Openvla-oft-...` path in
  `rental_runbook.sh`'s comment 401s. Each was downloaded to
  `/workspace/checkpoints/` (15.1 GB, 4 safetensors shards,
  `OpenVLAForActionPrediction`) and its `dataset_statistics.json` read: keys are
  exactly `libero_goal_no_noops`, `libero_spatial_no_noops`,
  `libero_object_no_noops`, `libero_10_no_noops` — one key each, matching the
  `unnorm_key` its config already declares. **No suite arm is skipped and no
  unnorm_key was guessed.** goal is the same checkpoint wave 1 used.
- `2026-09-12 06:15:35` **START** run_id=`smoke01` config=`/workspace/RLinf/experiments/efference_probe/configs/smoke.yaml` hash=`90155b8ee928` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-goal`
- `2026-09-12 06:18:00` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-goal
- `2026-09-12 06:25:22` building LiberoEnv (2 parallel envs)
- `2026-09-12 06:26:48` indexing check passed: {'max_abs_logprob_diff': 2.3096799850463867e-07, 'tolerance': 0.06954986453056336, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 06:26:48` heartbeat: task 0 batch 0 call 0 | 0.19/0.5 h, 0.00/2 GB, 0 calls
- `2026-09-12 06:26:58` done: 12 calls -> experiments/efference_probe/data/smoke01/calls.parquet (0.00 GB)
- `2026-09-12 06:26:58` **DONE** run_id=`smoke01` -> `experiments/efference_probe/data/smoke01/calls.parquet`
- `2026-09-12 06:28` **T-CODE (agent). Phase 2 — S0 smoke passed on the wave-2
  box.** `configs/smoke.yaml` on the goal checkpoint, config hash `90155b8ee928`,
  730 s wall (of which ~12 min is startup: imports incl. TensorFlow, a 15.1 GB
  checkpoint read off the MooseFS volume, and LIBERO env construction; the 12
  calls themselves take 10 s). Runtime indexing self-check:
  `indexing check passed: {'max_abs_logprob_diff': 2.3096799850463867e-07,
  'tolerance': 0.06954986453056336, 'logits_dtype': 'torch.float32',
  'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33,
  'n_act': 56}`; manifest `verification.passed = true`, `stop_reason =
  completed`. Acceptance: 12 calls / 2 episodes, labels `{'SELF': 8,
  'HIJACK': 4}`, transforms `{'': 8, 'freeze': 4}`, **0 null cells**, 2 hidden
  archives with `h` of shape `(6, 3, 3, 4096)` float16, 26 frames on disk.
  EGL rendering works; no osmesa fallback needed.
- `2026-09-12 06:28:58` **START** run_id=`main02` config=`/workspace/RLinf/experiments/efference_probe/configs/main02.yaml` hash=`e6e761ce152b` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-goal env.num_envs=10`
- `2026-09-12 06:30:14` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-goal
- `2026-09-12 06:34:23` building LiberoEnv (10 parallel envs)
- `2026-09-12 06:35:54` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.23093849182128906, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 06:35:55` heartbeat: task 0 batch 0 call 0 | 0.12/4.0 h, 0.00/30 GB, 0 calls
- `2026-09-12 06:40:58` heartbeat: task 3 batch 0 call 0 | 0.20/4.0 h, 0.35/30 GB, 1864 calls
- `2026-09-12 06:45:59` heartbeat: task 6 batch 0 call 14 | 0.28/4.0 h, 0.69/30 GB, 3636 calls
- `2026-09-12 06:50:59` heartbeat: task 9 batch 1 call 12 | 0.37/4.0 h, 1.06/30 GB, 5637 calls
- `2026-09-12 06:51:39` done: 5886 calls -> experiments/efference_probe/data/main02/calls.parquet (1.11 GB)
- `2026-09-12 06:51:39` **DONE** run_id=`main02` -> `experiments/efference_probe/data/main02/calls.parquet`
- `2026-09-12 07:09:48` **START** run_id=`main02_mirror` config=`/workspace/RLinf/experiments/efference_probe/configs/main02_mirror.yaml` hash=`5f56d5a5e869` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-goal env.num_envs=10`
- `2026-09-12 07:10:53` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-goal
- `2026-09-12 07:16:52` building LiberoEnv (10 parallel envs)
- `2026-09-12 07:18:45` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.2370439910888672, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 07:18:46` heartbeat: task 2 batch 0 call 0 | 0.15/4.0 h, 0.00/30 GB, 0 calls
- `2026-09-12 07:23:47` heartbeat: task 3 batch 2 call 7 | 0.23/4.0 h, 0.33/30 GB, 1719 calls
- `2026-09-12 07:28:47` heartbeat: task 8 batch 1 call 4 | 0.32/4.0 h, 0.67/30 GB, 3553 calls
- `2026-09-12 07:34:04` done: 5314 calls -> experiments/efference_probe/data/main02_mirror/calls.parquet (1.00 GB)
- `2026-09-12 07:34:04` **DONE** run_id=`main02_mirror` -> `experiments/efference_probe/data/main02_mirror/calls.parquet`
- `2026-09-12 07:36:16` **START** run_id=`main02_freeze` config=`/workspace/RLinf/experiments/efference_probe/configs/main02_freeze.yaml` hash=`449009c0ec02` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-goal env.num_envs=10`
- `2026-09-12 07:37:31` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-goal
- `2026-09-12 07:43:48` building LiberoEnv (10 parallel envs)
- `2026-09-12 07:45:38` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.2097229194641113, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 07:45:38` heartbeat: task 2 batch 0 call 0 | 0.16/4.0 h, 0.00/30 GB, 0 calls
- `2026-09-12 07:45` **T-CODE (agent). Phase 3 — driving session lost to an SSH
  drop; wave resumed without losing work.** The agent session that had been
  driving the wave died mid-Phase-3. Nothing on the box died with it: both
  in-flight jobs were running under `tmux` and kept going — `pmlp`
  (`run_probes.py --probe-family mlp` on `main02`, started 07:07) and `freeze`
  (`collect.sh main02_freeze`, started 07:36). State recovered from
  `/workspace/logs/`, the `gpu_ledger.tsv`, and the run directories; no run was
  re-run and nothing was deleted.
  **Recovered state at 07:45.** Phases 0–2 complete. `main02` collected
  (5,886 calls / 200 eps, `verification.passed=true`) and its **linear** round
  complete (`plumbing.passed=true`); `main02_mirror` collected (5,314 calls /
  150 eps, `verification.passed=true`), not yet probed; `main02_freeze`
  collecting; `main02` MLP round in flight. GPU ledger **1.011 h** of the 20 h
  cap; `/workspace` free 485 TB. No hard-stop condition met.
  **Cause of the loss, and the fix.** The wave had been advanced by the agent
  issuing one command at a time, so the queue lived only in the session — an
  SSH drop lost the schedule, not the data. The remaining wave is now driven by
  two on-box queues that survive any further disconnect, each in its own tmux
  session: `qcollect.sh` (GPU lane: `suite_spatial` > `suite_object` >
  `suite_long`, the runbook's priority order, one collection at a time) and
  `qprobe.sh` (CPU lane: for each of `main02_mirror`, `main02_freeze`,
  `suite_spatial01`, `suite_object01`, `suite_long01` — wait for that run's
  collection, then linear `--permutations 200`, then `--probe-family mlp`).
  Shared gate logic in `qlib.sh`: the hard stops (GPU >= 20 h, `/workspace`
  free < 10 GB, a `HALT` file) are re-checked before every job; a collection
  must show `verification.passed=true` before its probes start (gate 1) and the
  linear round must show `plumbing.passed=true` before the MLP round starts
  (gate 2); a run that fails collection twice halts the GPU lane rather than
  retrying a third time. Per-run outcomes land in `/workspace/logs/status/`.
  **Probe workers dropped 23 -> 20.** The cgroup CPU quota is 23.8 CPUs, and
  the lanes now overlap by design (a GPU collection runs while a CPU probe
  round runs), so the probe lane leaves ~4 CPUs for the collection lane's env
  stepping instead of contending for all 23.8. Applies to every probe round
  from `main02_mirror` on; `main02`'s own two rounds ran at 23 and are not
  re-run.
- `2026-09-12 07:50:39` heartbeat: task 3 batch 2 call 12 | 0.24/4.0 h, 0.28/30 GB, 1469 calls
- `2026-09-12 07:53` **T-CODE (agent). Queue gate defect found and fixed before
  it could act.** The first version of `qlib.sh`'s `collect_done()` treated
  `manifest.json` carrying a `stop_reason` as "collection finished". That is
  wrong: `run_collect.py` rewrites the manifest after every batch for
  crash-safety, and the partial manifest already carries
  `stop_reason="completed"` and `verification.passed=true`. Caught on
  `main02_freeze` while it was 60/150 episodes in — the manifest read as a
  finished 150-episode run:
  `keys: ['budget','config_hash','episodes','n_calls','n_episodes','stop_reason','verification']`,
  `stop_reason = 'completed'`, `verification.passed = True`, `n_calls = 1815`,
  `n_episodes = 60`, with `run_collect.py` still running.
  Had the probe lane reached `main02_freeze` before its collection ended, it
  would have probed a truncated run and both gates would have reported pass.
  **No damage:** the probe lane was still blocked on `main02`'s MLP round and
  had touched no run; `queue.log` contained only `qfinalize: armed`. Nothing
  was re-run or deleted.
  Fix: completion is now the collection wrapper's own exit marker — `collect.sh`
  writes `=== COLLECT <run_id> rc=0 elapsed=... ===` only after
  `run_collect.py` returns, so it cannot appear mid-run — plus the
  `logs/status/<run_id>.collect` marker. The trailing space in the grep keeps
  `main02` from matching `main02_mirror`. An earlier draft also guarded on
  "no `run_collect.py` process anywhere", which was wrong in the other
  direction: the lanes overlap by design, so one arm's collection would have
  masked every other arm's completion. The five queue sessions were restarted
  on the corrected library; `pmlp` and `freeze`, the two real jobs, were left
  untouched and stayed alive across the restart.
- `2026-09-12 07:55:40` heartbeat: task 8 batch 1 call 8 | 0.32/4.0 h, 0.60/30 GB, 3178 calls
- `2026-09-12 08:00:43` done: 4907 calls -> experiments/efference_probe/data/main02_freeze/calls.parquet (0.92 GB)
- `2026-09-12 08:00:43` **DONE** run_id=`main02_freeze` -> `experiments/efference_probe/data/main02_freeze/calls.parquet`
- `2026-09-12 08:01:44` **START** run_id=`suite_spatial01` config=`/workspace/RLinf/experiments/efference_probe/configs/suite_spatial.yaml` hash=`79211e3caf98` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-spatial env.num_envs=10`
- `2026-09-12 08:02:42` **START** run_id=`suite_spatial01` config=`/workspace/RLinf/experiments/efference_probe/configs/suite_spatial.yaml` hash=`79211e3caf98` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-spatial env.num_envs=10`
- `2026-09-12 08:02:44` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-spatial
- `2026-09-12 08:04:51` **START** run_id=`suite_spatial01` config=`/workspace/RLinf/experiments/efference_probe/configs/suite_spatial.yaml` hash=`79211e3caf98` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-spatial env.num_envs=10`
- `2026-09-12 08:05` **T-CODE (agent). Phase 3 — goal-suite collections done;
  probe cost measured; one operator incident.**
  Collections (all `num_envs=10`, sampled protocol, goal checkpoint), each
  gated on the wrapper's own `=== COLLECT <rid> rc=0` marker and then on
  `manifest.verification.passed` and `plumbing.passed`:
  `main02` cfg `e6e761ce152b` — 1408 s, 200 episodes, **5886 calls**, 689 swap
  hijacks, 1.11 GB, verification true, plumbing true (all mismatch counts 0).
  For reference main01 was 200 / 5,887 / 728 on the same suite.
  `main02_mirror` cfg `5f56d5a5e869` — 1503 s, 150 episodes, 5314 calls, 654
  mirror hijacks, 1.00 GB, both gates true.
  `main02_freeze` cfg `449009c0ec02` — 1518 s, 150 episodes, 4907 calls, 574
  freeze hijacks, 0.92 GB, both gates true.
  **`--jobs` measured, not assumed.** `run_probes.py`'s compute line reports
  `usable_cpus=112` because `sched_getaffinity` does not see the cgroup CPU
  quota (23.8), so `--jobs -1` (112 workers) oversubscribes ~4.7x. A/B on the
  real main02 P0 sweep (27 cells, `--out` to scratch so no analysis dir was
  touched): **`--jobs -1` 154 s vs `--jobs 23` 99 s (1.56x faster)**, and the
  two `probe_results.csv` are **bit-identical** (max abs diff 0.0 over all
  numeric columns), as the code's contract requires. The wave uses 20-23
  workers, never -1.
  **MLP cost, measured.** One real-shape MLP fit on real hidden features
  (1266 x 4096, layer 0 / ctx_mean) is **106 s** single-threaded. The same fit
  on random Gaussian features of the same shape is **5.5 s**: real hidden
  states are ill-conditioned enough that lbfgs runs its full 300 iterations,
  so the wave-1 estimate of "~3.4 s for a real-shape fit" understates the real
  article by ~30x. At 30 fits per cell (5 outer folds x [3-alpha inner select
  + 3 fold fits]) that is ~53 min of core time per cell; measured end to end,
  main02's 27-cell MLP P0 sweep took 35 min at 23 workers, so a full 108-cell
  MLP ladder is ~2.5 h wall clock per main-size run. This is the wave's long
  pole; the runbook's "a few CPU-hours of probing" covers the linear side only
  (main02 linear, full `--stage main` + 200 permutations + all controls,
  took **553 s**).
  **Incident (operator, no data lost).** Between 07:43 and 07:52 this session
  built an unattended queue (`qcollect`/`qprobe`/`qbrief`/`qfinalize`/
  `qresume` + `qlib.sh`) so the wave survives losing the agent session. At
  08:01 `qcollect` started `suite_spatial01`; at 08:02, still driving by hand,
  the agent started a **second** `run_collect.py` against the same run
  directory. Both were killed within ~100 s, while still loading the
  checkpoint: `data/suite_spatial01/` contained only an empty `hidden/`, no
  `calls.parquet`, no manifest, no hidden archives, so **nothing was
  overwritten and nothing had to be deleted**; the run restarted cleanly under
  the same id at 08:04. Root cause: two drivers for one GPU. Resolution: the
  queue is the only driver from here; the agent supervises and reports.
  `qcollect` already holds the right guard (`while pgrep -f run_collect.py;
  do sleep 60; done`) — it was bypassed by the manual launch, not broken.
  **Gate correctness note (found while writing `qlib.sh`).** `run_collect.py`
  rewrites `manifest.json` after every batch for crash safety, and that
  incremental manifest already carries `stop_reason: completed` and
  `verification.passed: true` **while the run is only part-way through**
  (seen on `main02_freeze` at 60/150 episodes). So `verification.passed` alone
  is not a completion signal; completion is the collect wrapper's exit marker,
  which cannot appear mid-run. Every gate in this wave checks the marker first.
- `2026-09-12 08:05:54` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-spatial
- `2026-09-12 08:08` **T-CODE (agent). A duplicate collection lane appeared;
  cause not established; contained.** At 08:01:04 the GPU lane started
  `suite_spatial01`; at 08:04:10 a *second* `qcollect` process logged the same
  two startup lines and started `suite_spatial01` again. Verbatim from
  `logs/queue.log`:
  `[2026-09-12T08:01:04Z] qcollect: GPU lane starting; gpu_hours=1.433 free=485509GB`
  `[2026-09-12T08:01:04Z] qcollect: suite_spatial01 attempt 1/2 (cfg=suite_spatial ckpt=/workspace/checkpoints/openvla-oft-sft-libero-spatial)`
  `[2026-09-12T08:04:10Z] qcollect: GPU lane starting; gpu_hours=1.433 free=485666GB`
  `[2026-09-12T08:04:10Z] qcollect: suite_spatial01 attempt 1/2 (cfg=suite_spatial ckpt=/workspace/checkpoints/openvla-oft-sft-libero-spatial)`
  **No data was lost or corrupted.** The 08:01 attempt died during model load
  having created only an empty `data/suite_spatial01/hidden/`; no
  `calls.parquet` and no manifest were written. Only one collector was alive on
  inspection (pid 24518, started 08:04:10), and one probe process (the `main02`
  MLP round, untouched throughout).
  **Cause not established, and not guessed.** Not the OOM killer
  (`memory.events: oom_kill 0`; `memory.current` 43.4 GB of the 62 GB
  `memory.max`). Not a tmux restart — server PID 618 has run unbroken since
  05:04:50 and every other session survived. Several `claude` processes live on
  this box (913, 1051, 7050, 18627, 19208, plus one agent on
  `--session-id ecabaa93-...`), so an action by another session cannot be
  excluded, but nothing in the evidence names a culprit. Recorded as
  unexplained-but-contained.
  **Containment, chosen so that no running script is edited** (bash reads a
  script lazily by byte offset; rewriting a file mid-execution corrupts the
  remainder): `qguard.sh` runs in its own tmux session in `MODE=enforce` and
  every 30 s counts real interpreter processes —
  `^([^ ]*/)?python[0-9.]* [^ ]*run_collect\.py` and the `run_probes.py`
  equivalent, patterns verified not to match an agent's own diagnostic command
  line that merely mentions the script name — and kills the *newest* duplicate
  only after it survives two consecutive checks, so a lane handover is never
  mistaken for a duplicate. `qlane.sh` is now the only sanctioned way to start a
  lane: it takes an exclusive `flock` per lane and refuses if that lane already
  holds it. Both are documented in `/workspace/RESUME.md`, which `qresume.sh`
  rewrites every five minutes so the next session inherits the state rather than
  reconstructing it from logs.
- `2026-09-12 08:11:58` building LiberoEnv (10 parallel envs)
- `2026-09-12 08:13:53` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.20661758422851562, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 08:13:54` heartbeat: task 0 batch 0 call 0 | 0.15/4.0 h, 0.00/30 GB, 0 calls
- `2026-09-12 08:18:55` heartbeat: task 2 batch 0 call 15 | 0.23/4.0 h, 0.22/30 GB, 1152 calls
- `2026-09-12 08:23:55` heartbeat: task 4 batch 0 call 27 | 0.32/4.0 h, 0.38/30 GB, 2005 calls
                                                                                                                                                                                                                                                                                                                                                                         - `2026-09-12 08:35:56` **START** run_id=`suite_spatial02` config=`/workspace/RLinf/experiments/efference_probe/configs/suite_spatial.yaml` hash=`79211e3caf98` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-spatial env.num_envs=10 run_id=suite_spatial02`
- `2026-09-12 08:35` **T-CODE (agent). Volume quota exhausted mid-wave;
  `suite_spatial01` lost; re-collected as `suite_spatial02`.** At 08:24 the
  `/workspace` quota filled. The `suite_spatial01` collection died at
  **2,005 calls / 90 episodes of 150**, leaving `ep00088.npz` and
  `ep00089.npz` at **0 bytes**. Verbatim from a write probe:
  `dd: error writing '/workspace/.t2': Disk quota exceeded`
  `dd: closing output file '/workspace/.t2': Disk quota exceeded`
  Writes failed at every size tested, 4 KB included — the HALT marker the
  queue tried to write came out empty, and no logbook entry could be made until
  the volume was expanded.
  **The gate that should have caught this was measuring the wrong thing.**
  `qlib.sh`'s `ws_free_gb()` read `df /workspace`, but `/workspace` is a
  MooseFS network volume: `df` reports the whole cluster pool (2.3 PB, 475 TB
  free) and never this pod's quota. So the red-line check "stop below 10 GB
  free" read *485,509 GB free* at the very moment the quota was exhausted, and
  the GPU lane sailed straight into it. Replaced with `headroom_ok()`, which
  actually writes a 1 GB probe file and deletes it; `hard_stop` now refuses to
  start a job when that write fails. `df` is kept as informational only, with a
  comment saying it must never gate anything. Same failure shape as the
  `collect_done()` defect logged at 07:53: a plausible-looking signal that was
  not the signal.
  **Accounting at the stop.** `/workspace` held 101 GB against what behaved
  like a 100 GB quota: `checkpoints/` 57 GB (4 suites), `uv/` 30 GB (of which
  `archive-v0` 26 GB is hardlink-shared with the venv — `links=2` on the 982 MB
  flash-attn `.so` — and `python/` 371 MB *is* the venv's interpreter, so
  neither is reclaimable), `RLinf/` 24 GB (`.venv` 20 GB + data 3.7 GB).
  Nothing was deleted: the red lines forbid removing any run directory or
  `hidden/`, and the operator chose to expand the volume rather than have the
  agent reclaim space. `suite_spatial01/` is kept intact, 0-byte archives and
  all, as the record of the failed attempt.
  **What survived.** The `main02` MLP round ran straight through the outage
  untouched — it holds its results in memory and writes only at the end — and
  was at P3 with P0 (27/27), P1 (27/27), P2 and P2r already computed when
  writes came back. No completed run lost a byte: `main02`, `main02_mirror` and
  `main02_freeze` all still verify.
  **Resumption.** Volume expanded by the operator; 1 GB write probe passes.
  `run_collect.py` accepts `--set run_id=`, so `collect.sh` now honours a
  `run_id=` override (so its log file and GPU-ledger label follow the real id)
  and the spatial arm re-collects as **`suite_spatial02`** — new id, nothing
  overwritten, nothing deleted, per the runbook's rule for interrupted
  collections. `qprobe`, `qresume` and `wave_report.py` all point at the new id;
  `suite_spatial01` stays in the report as an abandoned arm. GPU ledger at the
  stop: **1.433 h** of the 20 h cap, and the dead spatial attempt's ~20 minutes
  are not billed to any completed run.
- `2026-09-12 08:36:54` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-spatial
- `2026-09-12 08:40` **T-CODE (agent). `suite_spatial01` died mid-collection;
  restarted as `suite_spatial02`.** At `08:23:55` the run was healthy
  (heartbeat `task 4 batch 0 call 27 | 0.32/4.0 h, 0.38/30 GB, 2005 calls`,
  `indexing check passed` at 08:13:53). At 08:24 it wrote `hidden/ep00088.npz`
  and `hidden/ep00089.npz` at **0 bytes** and the process ended. The log
  contains **no traceback, no `STOP:` line and no `=== COLLECT ... rc=` exit
  marker** — the interpreter and its `tee` both went away without unwinding,
  so `run_collect.py`'s `except BaseException` never ran. Ruled out: the
  cgroup OOM killer (`memory.events`: `oom 0`, `oom_kill 0`,
  `oom_group_kill 0`) and `qguard` (it logged no duplicate and no kill; only
  one collector was running). The two 0-byte archives point at a failed write
  on the MooseFS volume. **Not diagnosed further** — recorded as an
  unexplained mid-run death with the evidence above.
  Handling followed the runbook's interruption rule exactly: `suite_spatial01`
  is **kept in place** (461 MB, 383 good npz + the 2 empty ones, no
  `calls.parquet`, no manifest) and the arm was re-collected under a **new
  run id, `suite_spatial02`**, started 08:35:16. Nothing was overwritten and
  nothing deleted.
  **`df` is not a usable disk gate here and no longer gates anything.**
  `/workspace` is a MooseFS mount, so `df` reports the whole cluster pool
  (2.3 PB, ~475 TB free) rather than this pod's share; it read 485 TB free
  throughout the failure. `qlib.sh`'s `hard_stop` now tests headroom by
  actually writing 1 GB and checking for `quota exceeded` / `no space left`.
  Measured after the restart: `/workspace` holds **101 GB** (57 GB
  checkpoints, 24 GB repo+venv, 21 GB uv cache), a 1 GB write runs at
  503 MB/s and a **10 GB write at 515 MB/s**, both clean — so there is real
  headroom now. Remaining data need is ~4 GB (three suite runs) plus ~1 GB of
  export. If headroom ever does get tight, `/workspace/uv` (21 GB) is a
  package cache and can be cleared without touching a run directory, a
  `hidden/`, or the logbook.
  **Follow-up: the same volume flakiness also corrupted a log, so every
  collected artifact was re-opened and checked.** `logs/probe_main02_mlp.log`
  came back as `data` to `file(1)` — MooseFS had written **77 NUL bytes** into
  a plain-text log being appended by `tee` (5062 bytes on disk, 4985 after
  stripping NULs), which silently made `grep` treat it as binary and report
  zero matches where `grep -a` finds 56. Since the same filesystem holds the
  run data, `integrity.py` was written and run over all three completed runs:
  it opens **every** `hidden/*.npz`, checks `h.ndim == 4` and that all values
  are finite, sums the per-archive call counts, re-reads `calls.parquet` and
  re-reads the manifest. Result — **all clean, no bad artifacts**:
  `main02` 200 npz / 5886 hidden calls vs 5886 parquet rows, 0 nulls;
  `main02_mirror` 150 / 5314 vs 5314, 0 nulls;
  `main02_freeze` 150 / 4907 vs 4907, 0 nulls; `verification.passed` true and
  `stop_reason: completed` for all three. The hidden-call totals matching the
  parquet row counts exactly is the strong check: a truncated or NUL-damaged
  archive would break that equality. So the volume's write flakiness touched a
  log file and the dying `suite_spatial01`, and no science data.
- `2026-09-12 08:42:36` building LiberoEnv (10 parallel envs)
- `2026-09-12 08:44:29` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.3562269592285156, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 08:44:30` heartbeat: task 0 batch 0 call 0 | 0.14/4.0 h, 0.00/30 GB, 0 calls
- `2026-09-12 08:49:30` heartbeat: task 2 batch 0 call 13 | 0.23/4.0 h, 0.23/30 GB, 1186 calls
- `2026-09-12 08:54:30` heartbeat: task 4 batch 0 call 27 | 0.31/4.0 h, 0.39/30 GB, 2053 calls
- `2026-09-12 08:59:31` heartbeat: task 6 batch 0 call 38 | 0.39/4.0 h, 0.61/30 GB, 3246 calls
- `2026-09-12 09:04:31` heartbeat: task 8 batch 1 call 4 | 0.48/4.0 h, 0.85/30 GB, 4506 calls
- `2026-09-12 09:08:16` done: 5678 calls -> experiments/efference_probe/data/suite_spatial02/calls.parquet (1.07 GB)
- `2026-09-12 09:08:16` **DONE** run_id=`suite_spatial02` -> `experiments/efference_probe/data/suite_spatial02/calls.parquet`
- `2026-09-12 09:09:05` **START** run_id=`suite_object01` config=`/workspace/RLinf/experiments/efference_probe/configs/suite_object.yaml` hash=`0cff943e3528` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-object env.num_envs=10`
- `2026-09-12 09:10:08` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-object
- `2026-09-12 09:15` **T-CODE (agent). `suite_spatial02` collected; wave resumed
  cleanly after the quota outage.** `suite_spatial02` (the re-collection of the
  arm lost at 08:24) completed rc=0 in 1,988 s: **200 episodes / 5,678 calls /
  682 `swap` hijacks / 0 null cells / 200 hidden archives**, config hash
  `79211e3caf98`, `verification.passed=true`. The GPU lane moved straight on to
  `suite_object01`. GPU ledger **1.985 h** of the 20 h cap; a 2 GB write probe
  passes.
  **The GPU-idle question, answered with measurements and two retracted
  hypotheses.** Collection-phase GPU duty cycle, 60 samples on a healthy disk:
  **mean 43.4%, 21/60 at >=90%, 18/60 at 0%** — the bimodal alternation of VLA
  forward (GPU) with MuJoCo stepping and EGL rendering (CPU), i.e. the shape of
  the workload. Per-run startup — imports incl. TensorFlow, a 15.1 GB
  checkpoint read off MooseFS, LIBERO env construction — costs **5m40s / 7m52s /
  8m07s on main02 / mirror / freeze, 26% / 33% / 34% of each run**, at 0% GPU.
  Whole-run utilisation is therefore ~30%.
  Two wrong attributions, both retracted against data. (1) "The MLP probe starves
  the collector of CPU": `main02_mirror` collected at 1719/1834 calls per 5 min
  *with the MLP round fully overlapping*, against 1864/1772/2001 for `main02`
  with no overlap — a 7% cost, not the 38% seen on spatial. (2) "The spatial
  slowdown is the volume filling up": `suite_spatial02` on a healthy disk
  reproduced `suite_spatial01`'s dying-disk numbers almost exactly —
  **1186, 867** against **1152, 853** — so the lower rate and the within-run
  decline are properties of the `libero_spatial` suite, not of contention and
  not of the disk. A `renice +15` applied to the MLP probe tree on the first
  (wrong) hypothesis is left in place: harmless, but not the remedy for
  anything. **No change made to the collection driver or to `env.num_envs`:**
  the only real waste is the ~24 min of startup left across the three remaining
  runs, against a projected 2.5-3 h wave, and both fixes (loading the model once
  across configs, or raising `num_envs`) would either restructure
  `run_collect.py` mid-wave or break the six-arm comparability the 06:12 entry
  fixed deliberately.
  **Log damage from the outage (data unaffected).** A failed append during the
  quota outage extends a file sparsely, so three logs carry NUL holes —
  `probe_main02_mlp.log` and its tee twin (77 bytes each) and `tmux_qcollect.log`
  (174 bytes). One hole swallowed the `P3 layer=0 pool=ctx_mean bacc=0.514
  auroc=0.511` line timestamped 08:25:27 that had been observed live at 08:33,
  and the NULs make `grep` treat the files as binary and print nothing — which
  briefly looked like a stalled probe. It was not: the round had 25 workers and
  19.6 cores busy at the time. The originals are **not** rewritten in place —
  `tee` still holds them open at an offset, and rewriting would re-create the
  hole — so `export_wave2.sh` now packages a de-NUL'd copy of `logs/` instead.
  `gpu_ledger.tsv` carried only a blank line, now removed (backup at
  `gpu_ledger.tsv.bak`); its total is unchanged at 1.985 h. MLP results are
  written once at the end of the round and are unaffected.
- `2026-09-12 09:14:58` building LiberoEnv (10 parallel envs)
- `2026-09-12 09:16:47` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.23069997787475588, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 09:16:48` heartbeat: task 0 batch 0 call 0 | 0.13/4.0 h, 0.00/30 GB, 0 calls
- `2026-09-12 09:21:48` heartbeat: task 2 batch 1 call 6 | 0.21/4.0 h, 0.34/30 GB, 1805 calls
- `2026-09-12 09:26:49` heartbeat: task 5 batch 0 call 26 | 0.30/4.0 h, 0.71/30 GB, 3779 calls
- `2026-09-12 09:31:50` heartbeat: task 8 batch 1 call 0 | 0.38/4.0 h, 1.17/30 GB, 6186 calls
- `2026-09-12 09:34:11` done: 7069 calls -> experiments/efference_probe/data/suite_object01/calls.parquet (1.33 GB)
- `2026-09-12 09:34:11` **DONE** run_id=`suite_object01` -> `experiments/efference_probe/data/suite_object01/calls.parquet`
- `2026-09-12 09:34:56` **START** run_id=`suite_long01` config=`/workspace/RLinf/experiments/efference_probe/configs/suite_long.yaml` hash=`258165973d78` overrides=`model.model_path=/workspace/checkpoints/openvla-oft-sft-libero-10 env.num_envs=10`
- `2026-09-12 09:35:49` loading checkpoint /workspace/checkpoints/openvla-oft-sft-libero-10
- `2026-09-12 09:39:29` building LiberoEnv (10 parallel envs)
- `2026-09-12 09:41:05` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.1992670059204102, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-09-12 09:41:05` heartbeat: task 0 batch 0 call 0 | 0.10/4.0 h, 0.00/30 GB, 0 calls
- `2026-09-12 09:46:13` heartbeat: task 2 batch 0 call 0 | 0.19/4.0 h, 0.35/30 GB, 1835 calls
- `2026-09-12 09:51:17` heartbeat: task 4 batch 0 call 0 | 0.27/4.0 h, 0.68/30 GB, 3592 calls
- `2026-09-12 09:56:17` heartbeat: task 6 batch 0 call 0 | 0.36/4.0 h, 1.04/30 GB, 5512 calls
- `2026-09-12 10:01:18` heartbeat: task 8 batch 0 call 18 | 0.44/4.0 h, 1.40/30 GB, 7432 calls
- `2026-09-12 10:05:44` done: 9329 calls -> experiments/efference_probe/data/suite_long01/calls.parquet (1.76 GB)
- `2026-09-12 10:05:44` **DONE** run_id=`suite_long01` -> `experiments/efference_probe/data/suite_long01/calls.parquet`
- `2026-09-12 10:32` **T-CODE (agent). All six collections done; two probe
  rounds lost to SIGTERM from our own operator tooling; scope cut rescinded.**
  **Collections complete — the GPU lane finished at `10:05:49` with
  `gpu_hours=2.941`** (cap 20). Six runs on disk: `main02` (200 ep / 5886
  calls), `main02_mirror` (150 / 5314), `main02_freeze` (150 / 4907),
  `suite_spatial02` (5678 calls), `suite_object01`, `suite_long01`; each
  `verification.passed=true` and each confirmed by the collect wrapper's own
  `rc=0` exit marker, not by the incrementally-rewritten manifest alone.
  **Two probe rounds were killed by our own tooling, not by anything in the
  data.** `main02_freeze`'s linear round died at `10:25:49` with, verbatim:
  `joblib.externals.loky.process_executor.TerminatedWorkerError: A worker
  process managed by the executor was unexpectedly terminated. ... The exit
  codes of the workers are {SIGTERM(-15), SIGTERM(-15), SIGTERM(-15),
  SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15),
  SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15)}` — all twelve
  workers signalled at once, which is an external kill, not a fault. The
  restarted `main02` MLP round logged `Terminated` the same way at ~10:28. The
  cgroup OOM killer is excluded (`memory.events: oom_kill 0`). Two mechanisms
  on this box could send that signal and **both were ours**: (a) `qguard` was
  running in `MODE=enforce` with a rule that treated **two concurrent
  `run_probes.py` processes as duplicates** and killed the newer — but running
  the MLP round alongside the linear pass is exactly what `qprobe` is designed
  to do, so the guard and the lane contradicted each other; (b) an operator
  `pkill -f` with a pattern broad enough to match joblib workers. Fixes: the
  duplicate rule now guards **collectors only** (two collectors on one run
  directory is the only real corruption hazard; two probers on different runs
  are harmless), and `pkill -f` is off the table — the same over-broad match
  was reproduced deliberately at 10:29, where `pkill -f "qretry_freeze.sh"`
  matched the agent's **own** shell and returned exit 144.
  **Two further self-inflicted faults, recorded so they are not repeated.**
  `probe.sh` was patched **while a copy of it was executing**; bash reads a
  script incrementally, so the running instance resumed at a stale byte offset
  and produced `/workspace/probe.sh: line 16: _probes.py: command not found`
  and `line 17: syntax error near unexpected token 'fi'`. The file itself was
  fine (`bash -n` clean). And a redundant retry lane (`qretry_freeze.sh`) was
  started for a job `qrepair.sh` already owned; two lanes re-running
  `main02_freeze` linear would have had both writing one `analysis/` directory.
  It was removed, leaving `qrepair` as the sole owner.
  **Logging moved off the volume.** Both lost rounds left **no traceback**,
  because their logs were the hours-long `tee -a` appends that MooseFS had
  already punched NUL holes into. Probe logs now go to `/var/log/wave2` on the
  container disk and are copied onto the volume as one whole-file write when
  the round ends.
  **`SKIP_MLP` rescinded.** A decision earlier in the session cut the five
  non-`main02` MLP rounds on the grounds that only pre-registered reading (1)
  needs an MLP ladder. That is not a sanctioned reason to drop work: the
  declared scope is six runs x two probe rounds, Phase 5 asks for a per-run MLP
  ladder and per-run MLP floor, and the only licence to cut is "from the tail
  when the **budget** is short" — GPU is at 2.941 h of 20, a 10 GB test write
  on `/workspace` succeeds, and no run has failed twice. Wall-clock cost is a
  number to report, not grounds to narrow the deliverable unilaterally. Full
  scope restored; the file is kept as `SKIP_MLP.rescinded` with the reasoning.
- `2026-09-12 11:15` **T-CODE (agent). Phase 0-1 — CPU pod, wave-2 inventory.**
  New RunPod **CPU** container, no GPU (`nvidia-smi` absent), network volume
  re-mounted at `/workspace`. Gates: `nproc`=16, `/sys/fs/cgroup/cpu.max`=
  `max 100000` (no quota) so usable_cpus=16; `/sys/fs/cgroup/memory.max`=
  32 GB (`free` reports the 755 GB host, not the limit); `/workspace` free
  471 TB. All above the 6-CPU / 16 GB / 10 GB stop thresholds.
  No live processes inherited: `tmux ls` empty, no `run_probes.py` or queue
  script running, so the stopped pod left nothing to collide with.
  **venv reused, not rebuilt.** `/workspace/RLinf/.venv/bin/python` (3.11.14)
  survived the pod swap intact — `numpy 1.26.4 / pandas / pyarrow /
  scikit-learn 1.9.1 / scipy / joblib / yaml / matplotlib` all import and
  `run_probes.py --help` runs, so no torch-free probe venv was needed.
  Gate `tests/test_offline.py`: **56 passed** in 27.7 s.
  Git: local `cfc8ea5a` was 1 commit ahead of
  `origin/claude/rlinf-code-scripts-pi41dg` (`cc5bc289`), no divergence, so no
  force-push is in play. 485 uncommitted logbook lines from the GPU pod were
  committed verbatim as `9ce5829a`.

  **Inventory of `data/` (read-only pass).** All 8 run directories report
  `verification.passed=true` and `stop_reason=completed`, and every
  `calls.parquet` row count equals its manifest `n_calls`:

  | run | verification | n_calls = parquet rows | episodes | `hidden/ep*.npz` | linear `analysis/` | `analysis_mlp/` |
  |---|---|---|---|---|---|---|
  | main02 | pass | 5886 | 200 | 200 | **COMPLETE** (perm 200) | PARTIAL |
  | main02_mirror | pass | 5314 | 150 | 150 | **COMPLETE** (perm 200) | MISSING |
  | main02_freeze | pass | 4907 | 150 | 150 | PARTIAL | MISSING |
  | suite_spatial01 | pass | 2005 | 80 | **90 (mismatch)** | MISSING | MISSING |
  | suite_spatial02 | pass | 5678 | 200 | 200 | PARTIAL | MISSING |
  | suite_object01 | pass | 7069 | 200 | 200 | MISSING | MISSING |
  | suite_long01 | pass | 9329 | 150 | 150 | MISSING | MISSING |
  | smoke01 | pass | 12 | 2 | 2 | n/a (smoke) | n/a |

  COMPLETE = `probe_results.csv` + `summary.md` + `plumbing.json` with
  `passed:true`, and for the linear round a `permutation_null` block with
  `n_permutations: 200` present in `summary.md`. Both COMPLETE linear rounds
  have it.

  **Split-session residue found.** (a) `suite_spatial01` is the run the
  duplicate collector touched on 2026-09-12T08:01-08:04Z: its manifest closes
  at 80 episodes but `hidden/` holds **90** `ep*.npz`, i.e. 10 episode files
  with no manifest entry. `suite_spatial02` (200/200, matched) is the intended
  spatial run and is the one carried forward; `suite_spatial01` is **left
  untouched on the volume**, not probed, not deleted. (b) Three half-written
  analysis directories were renamed (never deleted), suffix
  `.partial_20260912T111221Z`: `main02/analysis_mlp` (only
  `global_negatives_results.csv` + `plumbing.json`),
  `main02_freeze/analysis` (only `plumbing.json`),
  `suite_spatial02/analysis` (only `global_negatives_results.csv` +
  `plumbing.json`).

  **Why the two lost rounds died** (both tracebacks recovered, pasted verbatim
  in the Phase 4 report). `main02_freeze` linear, rc=1 after 302 s:
  `joblib.externals.loky.process_executor.TerminatedWorkerError` with worker
  exit codes `{SIGTERM(-15) x12}` raised inside `_permutation_null` ->
  `analysis.run_ladder`. SIGTERM on all twelve workers at 10:25:49Z coincides
  to the second with `qguard`'s
  `*** DUPLICATE qprobe *** count=3 pids=[52287 55360 55368] mode=enforce`
  — the guard killed the process group, so this is the split-session artefact,
  not an OOM or a data fault. `main02` MLP died at the shell, not in Python:
  `/workspace/probe.sh: line 16: _probes.py: command not found` (the `$P`
  run-dir variable collided with the script path inside `probe.sh`).
  Neither failure implicates `calls.parquet` or `hidden/`; both rounds are
  simply re-run here. Counted as one prior failure each.

  Wave-2 collection is **not** re-attempted from this box under any
  circumstance — it has no GPU. Remaining work is the 12 probe rounds
  (6 runs x linear+MLP) minus the 2 already COMPLETE.
- `2026-09-13 01:10` **T-CODE (agent). CPU pod #2 — Phase 0-1 re-inventory.**
  Second CPU container on the same network volume (`/workspace`, MooseFS).
  Gates: `nproc`=16, `cpu.max`=`max 100000` → usable_cpus=16;
  `memory.max`=32 000 000 000 B (32 GB); `/workspace` free 448 TB. No tmux
  server, no `run_probes.py`, no queue script alive on this box.
  Venv `/workspace/RLinf/.venv/bin/python` → 3.11.14 still resolves (symlink
  into `/workspace/uv`), reused. Gate `tests/test_offline.py`: **56 passed**
  in 43.0 s. Git: HEAD `cf3a45a4` is 3 ahead / 0 behind
  `origin/claude/rlinf-code-scripts-pi41dg` (`cc5bc289`); no divergence;
  working tree clean.

  **Residue left by CPU pod #1 (2026-09-12 11:08-~11:30Z).** It committed its
  Phase 0-1 entry (`cf3a45a4`), renamed three partial dirs at
  `20260912T111221Z`, then started `main02` MLP at ~11:14Z; that round left
  only `analysis_mlp/plumbing.json` (root-owned, 11:15Z) and its
  `/var/log/wave2` log died with the container (`logs/cpupod/queue.log` and
  `ledger.tsv` are 0 bytes). Renamed (not deleted):
  `main02/analysis_mlp` → `main02/analysis_mlp.partial_20260913T010847Z`.

  **Re-inventory (read-only, this pod):**

  | run | verification | n_calls = parquet rows | episodes | `hidden/ep*.npz` | linear | MLP |
  |---|---|---|---|---|---|---|
  | main02 | pass | 5886 | 200 | 200 | COMPLETE (perm 200) | PARTIAL → renamed |
  | main02_mirror | pass | 5314 | 150 | 150 | COMPLETE (perm 200) | MISSING |
  | main02_freeze | pass | 4907 | 150 | 150 | MISSING (`.partial_20260912T111221Z`) | MISSING |
  | suite_spatial01 | pass | 2005 | 80 | **90 (mismatch)** | SKIPPED | SKIPPED |
  | suite_spatial02 | pass | 5678 | 200 | 200 | MISSING (`.partial_20260912T111221Z`) | MISSING |
  | suite_object01 | pass | 7069 | 200 | 200 | MISSING | MISSING |
  | suite_long01 | pass | 9329 | 150 | 150 | MISSING | MISSING |
  | smoke01 | pass | 12 | 2 | 2 | n/a | n/a |

  Carried forward: `suite_spatial02` (manifest 200 = hidden 200) for the
  spatial suite; `suite_spatial01` stays on the volume untouched.
  Failure accounting: the earlier lost attempts (freeze linear SIGTERM'd by
  `qguard`; main02 MLP shell error in `probe.sh`; main02 MLP killed by the
  pod stop) are infrastructure interruptions, not `run_probes.py` errors; the
  two-strike stop is counted per round on this pod, by
  `/workspace/cpu2_queue.sh`. Probe wall-clock budget (14 h) is counted from
  this pod's Phase 0, `2026-09-13T01:08:47Z`.
  Queue order: main02 MLP → main02_mirror MLP → main02_freeze linear, MLP →
  suite_spatial02 → suite_object01 → suite_long01 (linear then MLP each),
  subject to the extrapolation check after main02 MLP.
- `2026-09-13 03:25` **T-CODE (agent). CPU pod #2 — main02 MLP done; budget extrapolation and tail cut.**
  `main02` MLP (`--stage main --jobs 16 --probe-family mlp`) finished rc=0 at
  03:19:00Z, **7669 s** wall, COMPLETE (plumbing passed). Probe samples 1266.
  MLP/linear ratio on main02 = 7669 / 553 = 13.9x. Budget origin 01:08:47Z;
  elapsed at that point 7813 s (2.17 h) of 50 400 s.

  Extrapolation (linear in probe samples; MLP 6.06 s/sample from main02,
  linear 0.51 s/sample from main02_mirror at 12 jobs; unprobed runs' samples
  approximated as 1.84 x HIJACK rows in `calls.parquet`, the observed ratio on
  the four probed runs):

  | round | samples | est. s |
  |---|---|---|
  | main02_mirror mlp | 1226 | 7430 |
  | main02_freeze linear | 1056 | 540 |
  | main02_freeze mlp | 1056 | 6400 |
  | suite_spatial02 linear | 1202 | 610 |
  | suite_object01 linear | ~1540 (836 hijacks) | 780 |
  | suite_long01 linear | ~2240 (1220 hijacks) | 1140 |
  | suite_spatial02 mlp | 1202 | 7280 |
  | suite_object01 mlp | ~1540 | 9310 |
  | suite_long01 mlp | ~2240 | 13590 |

  Cumulative through suite_object01 MLP ~41 400 s (11.5 h); adding
  suite_long01 MLP gives ~55 000 s (15.3 h) > 14 h cap. **Cut from the tail:
  `suite_long01:mlp`** (`/workspace/logs/cpupod2/CUT_suite_long01_mlp`,
  03:20Z). Suite linears are all kept. Re-computed after each round with the
  ledger's actual wall clock. Midpoint brief:
  `/workspace/exports/cpupod2_midpoint.md` (numbers only).
- `2026-09-13 08:00` **T-CODE (agent). CPU pod #2 — budget re-check after the freeze rounds.**
  Ledger (all rc=0, COMPLETE): main02 mlp 7669 s (1266 samples, 6.06 s/sample);
  main02_mirror mlp 7566 s (1226, 6.17 s/sample); main02_freeze linear 678 s
  (1056, 0.64 s/sample); main02_freeze mlp 8347 s (1056, **7.90 s/sample**).
  MLP wall does not scale with sample count across these three runs.
  Elapsed at 07:56:24Z: 24 457 s (6.79 h). Worst-rate projection (MLP
  7.90 s/sample, linear 0.64 s/sample): suite linears 770 + 985 + 1434 s →
  7.68 h; + suite_spatial02 mlp ~9500 s → 10.3 h; + suite_object01 mlp
  ~12 170 s → **13.7 h** (< 14 h). Pooled MLP rate 6.65 s/sample gives 12.7 h.
  Decision: keep `suite_object01:mlp` queued; `suite_long01:mlp` stays cut.
  Re-checked from suite_spatial02 MLP progress before it ends (the queue
  starts the next round within ~1 s, so a CUT must be written in advance).
- `2026-09-13 09:05` **T-CODE (agent). CPU pod #2 — operator budget notice; second tail cut.**
  At 09:04Z the operator stated the remaining budget is **2 h 04 min**
  (hard end ~11:06Z), superseding the 14 h probe clock. Rounds done by then:
  7 COMPLETE (ledger 25 875 s). Running: `suite_spatial02` MLP since 08:23:23Z
  (est. 7280-9500 s → ends ~10:25-11:01Z). **Cut `suite_object01:mlp`**
  (`CUT_suite_object01_mlp`); `suite_long01:mlp` was already cut. Deadline for
  `suite_spatial02` MLP: 10:45Z, else aborted and its directory renamed
  `.partial_<UTC>` (never deleted), leaving ~20 min for Phase 4 export.
- `2026-09-13 10:50` **T-CODE (agent). CPU pod #2 — Phase 4 final report (numbers only).**
  Queue ended 10:42:54Z (`REST_DONE`). Full report copied to `reports/p0_wave2_report.md`; reproduced verbatim below.

  # P0 wave 2 — final probe report (CPU pod #2, generated 2026-09-13T10:43:29Z)
  
  Numbers and facts only. Sources: `data/<run>/analysis{,_mlp}/summary.md` + `probe_results.csv`, `manifest.json`, `/workspace/logs/cpupod2/`.
  
  ## 1. Per-run table
  
  | run | episodes | calls | hijacks | success | verification | plumbing | linear | MLP |
  |---|---|---|---|---|---|---|---|---|
  | main02 | 200 | 5886 | 689 | 0.41 | True | lin=True / mlp=True | COMPLETE | COMPLETE |
  | main02_mirror | 150 | 5314 | 654 | 0.24666666666666667 | True | lin=True / mlp=True | COMPLETE | COMPLETE |
  | main02_freeze | 150 | 4907 | 574 | 0.38 | True | lin=True / mlp=True | COMPLETE | COMPLETE |
  | suite_spatial02 | 200 | 5678 | 682 | 0.47 | True | lin=True / mlp=True | COMPLETE | COMPLETE |
  | suite_object01 | 200 | 7069 | 836 | 0.21 | True | lin=True / mlp=- | COMPLETE | MISSING |
  | suite_long01 | 150 | 9329 | 1220 | 0.07333333333333333 | True | lin=True / mlp=- | COMPLETE | MISSING |
  | suite_spatial01 | 80 (hidden 90 npz) | 2005 | - | - | True | - | SKIPPED | SKIPPED |
  
  Success rates by episode type (from linear summary `e3.success_rate`):
  
  | run | overall | probe episodes (n) | clean episodes (n) |
  |---|---|---|---|
  | main02 | 0.410 | 0.387 (163) | 0.514 (37) |
  | main02_mirror | 0.247 | 0.193 (119) | 0.452 (31) |
  | main02_freeze | 0.380 | 0.395 (119) | 0.323 (31) |
  | suite_spatial02 | 0.470 | 0.439 (155) | 0.578 (45) |
  | suite_object01 | 0.210 | 0.181 (155) | 0.311 (45) |
  | suite_long01 | 0.073 | 0.041 (123) | 0.222 (27) |
  
  ## 2. Linear ladders (max over 27 cells; selection floor; permutation null)
  
  **main02 / linear** — selection floor max_over_cells=0.5413 (argmax {'layer': 4.0, 'pool': 'ctx_mean'}, mean_over_cells=0.5020, n_cells=27); permutation null n=200 mean=0.4969 p95=0.5233 max=0.5401
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.541 | 0.026 | 0.548 | L4/ctx_mean | 27 |
  | P1 | 0.543 | 0.035 | 0.538 | L20/act_mean | 27 |
  | P3 | 0.532 | 0.023 | 0.532 | L20/act_mean | 27 |
  | P4 | 0.536 | 0.011 | 0.540 | L28/act_mean | 27 |
  | P2 | 0.463 | 0.022 | 0.454 | - | 1 |
  | P2r | 0.711 | 0.038 | 0.783 | - | 1 |
  | C_cmd | 0.475 | 0.019 | 0.463 | - | 1 |
  | C_dstates | 0.488 | 0.035 | 0.486 | - | 1 |
  | C_phase | 0.513 | 0.028 | 0.514 | - | 1 |
  
  **main02_mirror / linear** — selection floor max_over_cells=0.5210 (argmax {'layer': 8.0, 'pool': 'ctx_mean'}, mean_over_cells=0.5021, n_cells=27); permutation null n=200 mean=0.5013 p95=0.5271 max=0.5397
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.521 | 0.026 | 0.512 | L8/ctx_mean | 27 |
  | P1 | 0.567 | 0.028 | 0.598 | L16/act_mean | 27 |
  | P3 | 0.568 | 0.029 | 0.591 | L16/act_mean | 27 |
  | P4 | 0.597 | 0.011 | 0.616 | L20/act_mean | 27 |
  | P2 | 0.587 | 0.026 | 0.625 | - | 1 |
  | P2r | 0.857 | 0.036 | 0.948 | - | 1 |
  | C_cmd | 0.515 | 0.021 | 0.501 | - | 1 |
  | C_dstates | 0.623 | 0.040 | 0.666 | - | 1 |
  | C_phase | 0.499 | 0.019 | 0.514 | - | 1 |
  
  **main02_freeze / linear** — selection floor max_over_cells=0.5377 (argmax {'layer': 28.0, 'pool': 'ctx_last'}, mean_over_cells=0.5064, n_cells=27); permutation null n=200 mean=0.4980 p95=0.5286 max=0.5729
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.538 | 0.040 | 0.550 | L28/ctx_last | 27 |
  | P1 | 0.543 | 0.030 | 0.557 | L4/act_mean | 27 |
  | P3 | 0.565 | 0.017 | 0.578 | L4/ctx_last | 27 |
  | P4 | 0.566 | 0.045 | 0.560 | L12/ctx_last | 27 |
  | P2 | 0.561 | 0.033 | 0.583 | - | 1 |
  | P2r | 0.886 | 0.035 | 0.954 | - | 1 |
  | C_cmd | 0.503 | 0.018 | 0.509 | - | 1 |
  | C_dstates | 0.594 | 0.052 | 0.558 | - | 1 |
  | C_phase | 0.505 | 0.041 | 0.521 | - | 1 |
  
  **suite_spatial02 / linear** — selection floor max_over_cells=0.5332 (argmax {'layer': 12.0, 'pool': 'ctx_mean'}, mean_over_cells=0.5187, n_cells=27); permutation null n=200 mean=0.5014 p95=0.5321 max=0.5531
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.533 | 0.026 | 0.534 | L12/ctx_mean | 27 |
  | P1 | 0.563 | 0.038 | 0.588 | L28/ctx_mean | 27 |
  | P3 | 0.567 | 0.036 | 0.592 | L24/ctx_mean | 27 |
  | P4 | 0.558 | 0.042 | 0.582 | L24/ctx_mean | 27 |
  | P2 | 0.500 | 0.015 | 0.504 | - | 1 |
  | P2r | 0.757 | 0.021 | 0.837 | - | 1 |
  | C_cmd | 0.483 | 0.022 | 0.498 | - | 1 |
  | C_dstates | 0.499 | 0.028 | 0.511 | - | 1 |
  | C_phase | 0.514 | 0.020 | 0.490 | - | 1 |
  
  **suite_object01 / linear** — selection floor max_over_cells=0.5416 (argmax {'layer': 12.0, 'pool': 'ctx_mean'}, mean_over_cells=0.5142, n_cells=27); permutation null n=200 mean=0.4996 p95=0.5277 max=0.5410
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.542 | 0.018 | 0.556 | L12/ctx_mean | 27 |
  | P1 | 0.566 | 0.026 | 0.585 | L32/act_mean | 27 |
  | P3 | 0.565 | 0.028 | 0.583 | L28/ctx_last | 27 |
  | P4 | 0.576 | 0.022 | 0.601 | L28/act_mean | 27 |
  | P2 | 0.504 | 0.029 | 0.509 | - | 1 |
  | P2r | 0.800 | 0.023 | 0.873 | - | 1 |
  | C_cmd | 0.512 | 0.019 | 0.506 | - | 1 |
  | C_dstates | 0.513 | 0.016 | 0.522 | - | 1 |
  | C_phase | 0.502 | 0.013 | 0.524 | - | 1 |
  
  **suite_long01 / linear** — selection floor max_over_cells=0.5231 (argmax {'layer': 12.0, 'pool': 'ctx_mean'}, mean_over_cells=0.5038, n_cells=27); permutation null n=200 mean=0.5001 p95=0.5206 max=0.5309
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.523 | 0.014 | 0.516 | L12/ctx_mean | 27 |
  | P1 | 0.546 | 0.013 | 0.554 | L8/ctx_mean | 27 |
  | P3 | 0.545 | 0.014 | 0.559 | L20/ctx_last | 27 |
  | P4 | 0.549 | 0.026 | 0.560 | L12/act_mean | 27 |
  | P2 | 0.502 | 0.026 | 0.504 | - | 1 |
  | P2r | 0.774 | 0.018 | 0.849 | - | 1 |
  | C_cmd | 0.488 | 0.017 | 0.496 | - | 1 |
  | C_dstates | 0.512 | 0.016 | 0.511 | - | 1 |
  | C_phase | 0.521 | 0.026 | 0.517 | - | 1 |
  
  ## 3. MLP ladders (max over 27 cells; MLP selection floor)
  
  **main02 / mlp** — selection floor max_over_cells=0.5320 (argmax {'layer': 4.0, 'pool': 'ctx_mean'}, mean_over_cells=0.5015, n_cells=27)
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.532 | 0.029 | 0.546 | L4/ctx_mean | 27 |
  | P1 | 0.538 | 0.026 | 0.545 | L28/ctx_last | 27 |
  | P3 | 0.540 | 0.017 | 0.540 | L32/ctx_last | 27 |
  | P4 | 0.543 | 0.013 | 0.546 | L16/act_mean | 27 |
  | P2 | 0.687 | 0.016 | 0.754 | - | 1 |
  | P2r | 0.776 | 0.018 | 0.856 | - | 1 |
  | C_cmd | 0.499 | 0.030 | 0.508 | - | 1 |
  | C_dstates | 0.522 | 0.015 | 0.545 | - | 1 |
  | C_phase | 0.533 | 0.027 | 0.549 | - | 1 |
  
  **main02_mirror / mlp** — selection floor max_over_cells=0.5322 (argmax {'layer': 12.0, 'pool': 'ctx_last'}, mean_over_cells=0.5002, n_cells=27)
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.532 | 0.025 | 0.520 | L12/ctx_last | 27 |
  | P1 | 0.581 | 0.038 | 0.598 | L12/ctx_mean | 27 |
  | P3 | 0.577 | 0.030 | 0.608 | L20/act_mean | 27 |
  | P4 | 0.612 | 0.022 | 0.628 | L16/act_mean | 27 |
  | P2 | 0.851 | 0.020 | 0.935 | - | 1 |
  | P2r | 0.871 | 0.025 | 0.953 | - | 1 |
  | C_cmd | 0.520 | 0.042 | 0.525 | - | 1 |
  | C_dstates | 0.738 | 0.019 | 0.822 | - | 1 |
  | C_phase | 0.503 | 0.029 | 0.515 | - | 1 |
  
  **main02_freeze / mlp** — selection floor max_over_cells=0.5351 (argmax {'layer': 28.0, 'pool': 'ctx_last'}, mean_over_cells=0.5072, n_cells=27)
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.535 | 0.050 | 0.543 | L28/ctx_last | 27 |
  | P1 | 0.557 | 0.006 | 0.564 | L8/ctx_last | 27 |
  | P3 | 0.588 | 0.031 | 0.603 | L4/ctx_last | 27 |
  | P4 | 0.636 | 0.029 | 0.680 | L8/act_mean | 27 |
  | P2 | 0.835 | 0.040 | 0.922 | - | 1 |
  | P2r | 0.911 | 0.036 | 0.970 | - | 1 |
  | C_cmd | 0.485 | 0.011 | 0.490 | - | 1 |
  | C_dstates | 0.825 | 0.028 | 0.885 | - | 1 |
  | C_phase | 0.498 | 0.038 | 0.476 | - | 1 |
  
  **suite_spatial02 / mlp** — selection floor max_over_cells=0.5419 (argmax {'layer': 12.0, 'pool': 'act_mean'}, mean_over_cells=0.5170, n_cells=27)
  
  | probe | bacc (max over cells) | ±sd | AUROC | cell | n_cells |
  |---|---|---|---|---|---|
  | P0 | 0.542 | 0.044 | 0.543 | L12/act_mean | 27 |
  | P1 | 0.566 | 0.053 | 0.572 | L32/ctx_mean | 27 |
  | P3 | 0.568 | 0.034 | 0.595 | L28/ctx_mean | 27 |
  | P4 | 0.572 | 0.036 | 0.604 | L24/ctx_mean | 27 |
  | P2 | 0.769 | 0.025 | 0.842 | - | 1 |
  | P2r | 0.827 | 0.020 | 0.897 | - | 1 |
  | C_cmd | 0.479 | 0.045 | 0.481 | - | 1 |
  | C_dstates | 0.607 | 0.029 | 0.640 | - | 1 |
  | C_phase | 0.527 | 0.017 | 0.535 | - | 1 |
  
  **suite_object01 / mlp**: MISSING — no numbers
  
  **suite_long01 / mlp**: MISSING — no numbers
  
  ## 4. E3 per arm
  
  | run | Δentropy (hijack−self) | p (Wilcoxon, episode) | Δlogprob_sum | p (Wilcoxon, episode) | n_ep |
  |---|---|---|---|---|---|
  | main02 | +0.00125 | 0.5274 | +0.1448 | 0.4433 | 149 (pairs 633) |
  | main02_mirror | +0.01402 | 0.01593 | -1.9390 | 0.01194 | 119 (pairs 613) |
  | main02_freeze | -0.00361 | 0.9555 | +0.7134 | 0.819 | 115 (pairs 528) |
  | suite_spatial02 | +0.00444 | 0.2641 | -0.1894 | 0.2952 | 138 (pairs 601) |
  | suite_object01 | +0.01280 | 0.01097 | -1.6188 | 0.003553 | 153 (pairs 739) |
  | suite_long01 | +0.00197 | 0.428 | -0.4730 | 0.2603 | 123 (pairs 1177) |
  
  ## 5. Pre-registered readouts (P0_RUNBOOK.md §4) — numbers under each
  
  **(a) MLP ladder on main02: P1/P3/P4 vs. the MLP selection floor.**
  
    - main02 MLP: floor 0.5320; P1 0.538 ± 0.026 (L28/ctx_last); P3 0.540 ± 0.017 (L32/ctx_last); P4 0.543 ± 0.013 (L16/act_mean); P2r 0.776 ± 0.018 (-)
  
  **(b) Linear ladder on main02: replication of main01's null; floor and P2r within fold noise.**
  
    - main02 linear: floor 0.5413; perm p95 0.5233; P1 0.543 ± 0.035 (L20/act_mean); P3 0.532 ± 0.023 (L20/act_mean); P4 0.536 ± 0.011 (L28/act_mean); P2r 0.711 ± 0.038 (-)
    - main01 floor / P2r: not present in this repository or on the volume (no `data/main01`, no main01 numbers in `results.md` or `logbook.md`); comparison not computed here.
  
  **(c) Expanded mirror/freeze: hidden-probe excursions vs. floor; mirror E3 log-prob at n~130 episodes.**
  
    - main02_mirror linear: floor 0.5210; perm p95 0.5271; P1 0.567 ± 0.028 (L16/act_mean); P3 0.568 ± 0.029 (L16/act_mean); P4 0.597 ± 0.011 (L20/act_mean); P2r 0.857 ± 0.036 (-)
    - main02_mirror MLP: floor 0.5322; P1 0.581 ± 0.038 (L12/ctx_mean); P3 0.577 ± 0.030 (L20/act_mean); P4 0.612 ± 0.022 (L16/act_mean); P2r 0.871 ± 0.025 (-)
    - main02_freeze linear: floor 0.5377; perm p95 0.5286; P1 0.543 ± 0.030 (L4/act_mean); P3 0.565 ± 0.017 (L4/ctx_last); P4 0.566 ± 0.045 (L12/ctx_last); P2r 0.886 ± 0.035 (-)
    - main02_freeze MLP: floor 0.5351; P1 0.557 ± 0.006 (L8/ctx_last); P3 0.588 ± 0.031 (L4/ctx_last); P4 0.636 ± 0.029 (L8/act_mean); P2r 0.911 ± 0.036 (-)
    - main02_mirror: Δentropy +0.01402 (p 0.01593); Δlogprob_sum -1.9390 (p 0.01194); n_ep 119 (pairs 613)
    - main02_freeze: Δentropy -0.00361 (p 0.9555); Δlogprob_sum +0.7134 (p 0.819); n_ep 115 (pairs 528)
  
  **(d) Suites: per-suite ladders vs. per-suite floors (P2r vs floor; hidden probes vs floor).**
  
    - suite_spatial02 linear: floor 0.5332; perm p95 0.5321; P1 0.563 ± 0.038 (L28/ctx_mean); P3 0.567 ± 0.036 (L24/ctx_mean); P4 0.558 ± 0.042 (L24/ctx_mean); P2r 0.757 ± 0.021 (-)
    - suite_spatial02 MLP: floor 0.5419; P1 0.566 ± 0.053 (L32/ctx_mean); P3 0.568 ± 0.034 (L28/ctx_mean); P4 0.572 ± 0.036 (L24/ctx_mean); P2r 0.827 ± 0.020 (-)
    - suite_object01 linear: floor 0.5416; perm p95 0.5277; P1 0.566 ± 0.026 (L32/act_mean); P3 0.565 ± 0.028 (L28/ctx_last); P4 0.576 ± 0.022 (L28/act_mean); P2r 0.800 ± 0.023 (-)
    - suite_object01 MLP: not COMPLETE — no numbers
    - suite_long01 linear: floor 0.5231; perm p95 0.5206; P1 0.546 ± 0.013 (L8/ctx_mean); P3 0.545 ± 0.014 (L20/ctx_last); P4 0.549 ± 0.026 (L12/act_mean); P2r 0.774 ± 0.018 (-)
    - suite_long01 MLP: not COMPLETE — no numbers
  
  ## 6. Anomalies
  
  1. `--jobs -1` resolves to workers=192 on this pod (`analysis.py:302` uses `os.cpu_count()`=192; usable CPUs 16, cgroup memory 32 GB). The main02 MLP attempt started 01:09:38Z was aborted by the agent ~1.5 min in and relaunched with `--jobs 16`; no code change; not counted as a failure. Log: `logs/cpupod2/main02_mlp.aborted_jobs192.log`.
  2. The 14 h probe wall-clock budget is counted from this pod's Phase 0, `2026-09-13T01:08:47Z`; probe time spent on the two earlier pods is not included.
  3. Run names actually probed: `suite_spatial02` / `suite_object01` / `suite_long01`. `suite_spatial01`: manifest 80 episodes vs. 90 `hidden/ep*.npz` (split-session duplicate collector, 2026-09-12T08:01–08:04Z) — SKIPPED, left untouched, not probed, not deleted.
  4. Half-written analysis directories renamed (never deleted): `main02/analysis_mlp.partial_20260912T111221Z`, `main02/analysis_mlp.partial_20260913T010847Z`, `main02/analysis_mlp.partial_20260913T011111Z` (jobs=192 attempt), `main02_freeze/analysis.partial_20260912T111221Z`, `suite_spatial02/analysis.partial_20260912T111221Z`.
  5. Pod #1 main02 MLP (~11:14Z 2026-09-12) lost with the pod stop; its `/var/log/wave2` log died with the container.
  6. MLP wall clock does not scale with probe-sample count across runs (see §8: s/sample per MLP round).
  
  Traceback — main02_freeze linear, earlier pod, 2026-09-12 10:25:49Z (rc=1, elapsed 302 s; coincides with qguard `*** DUPLICATE qprobe *** count=3 ... mode=enforce`) (`/workspace/logs/probe_main02_freeze_linear.log`):
  
  ```
  Traceback (most recent call last):
    File "/workspace/RLinf/experiments/efference_probe/run_probes.py", line 377, in <module>
      raise SystemExit(main())
                       ^^^^^^
    File "/workspace/RLinf/experiments/efference_probe/run_probes.py", line 275, in main
      extra["permutation_null"] = _permutation_null(
                                  ^^^^^^^^^^^^^^^^^^
    File "/workspace/RLinf/experiments/efference_probe/run_probes.py", line 104, in _permutation_null
      results = analysis.run_ladder(run, samples, permuted, store=store)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/workspace/RLinf/experiments/efference_probe/analysis.py", line 424, in run_ladder
      for result in stream:
    File "/workspace/RLinf/.venv/lib/python3.11/site-packages/joblib/parallel.py", line 1704, in _get_outputs
      yield from self._retrieve()
    File "/workspace/RLinf/.venv/lib/python3.11/site-packages/joblib/parallel.py", line 1806, in _retrieve
      self._raise_error_fast()
    File "/workspace/RLinf/.venv/lib/python3.11/site-packages/joblib/parallel.py", line 1885, in _raise_error_fast
      error_job.get_result(self.timeout)
    File "/workspace/RLinf/.venv/lib/python3.11/site-packages/joblib/parallel.py", line 780, in get_result
      return self._return_or_raise()
             ^^^^^^^^^^^^^^^^^^^^^^^
    File "/workspace/RLinf/.venv/lib/python3.11/site-packages/joblib/parallel.py", line 795, in _return_or_raise
      raise self._result
  joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.
  
  The exit codes of the workers are {SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15), SIGTERM(-15)}
  Detailed tracebacks of the workers should have been printed to stderr in the executor process if faulthandler was not disabled.
  ```
  
  Shell error — main02 MLP, earlier pod, 2026-09-12 (`/workspace/logs/tmux_probe_main02_mlp2.log`, last lines; no Python traceback, the process was `Terminated` then `probe.sh` failed):
  
  ```
  Terminated
  /workspace/probe.sh: line 16: _probes.py: command not found
  /workspace/probe.sh: line 17: syntax error near unexpected token `fi'
  /workspace/probe.sh: line 17: `fi'
  ```
  
  Queue log lines matching FAIL/HARD STOP/SKIPPED/ABORTED on this pod:
  
  ```
  [2026-09-13T01:11:10Z] ABORTED main02 mlp attempt (started 01:09:38Z) by agent: --jobs -1 resolved to workers=192 (os.cpu_count) on a 16-CPU affinity / 32 GB cgroup; relaunching with --jobs 16. Not counted as a failure.
  ```
  
  ## 7. Skipped / cut
  
  - `suite_spatial01` linear + MLP: SKIPPED (manifest/hidden mismatch, see §6.3).
  - `suite_long01_mlp`: CUT — budget extrapolation at 2026-09-13T03:20Z: projected cumulative probe wall ~15.3 h > 14 h cap if suite_long01 MLP (~2240 samples x 6.06 s/sample ~ 13600 s) runs; cut from the tail
  - `suite_object01_mlp`: CUT — operator notice at 2026-09-13T09:04Z: remaining budget is 2 h 04 min (hard end ~11:06Z); suite_object01 MLP (est. 9000-11700 s) cannot fit
  - queue: `[2026-09-13T10:42:54Z] rest: CUT suite_object01 mlp (operator notice at 2026-09-13T09:04Z: remaining budget is 2 h 04 min (hard end ~11:06Z); suite_object01 MLP (est. 9000-11700 s) cannot fit)`
  - queue: `[2026-09-13T10:42:54Z] rest: CUT suite_long01 mlp (budget extrapolation at 2026-09-13T03:20Z: projected cumulative probe wall ~15.3 h > 14 h cap if suite_long01 MLP (~2240 samples x 6.06 s/sample ~ 13600 s) runs; cut from the tail)`
  - smoke01: not probed (smoke run, metadata only).
  
  ## 8. Wall-clock ledger (`logs/cpupod2/ledger.tsv`, verbatim)
  
  ```
  run	family	rc	elapsed_s	ended_utc
  main02	mlp	0	7669	2026-09-13T03:19:00Z
  main02_mirror	mlp	0	7566	2026-09-13T05:25:57Z
  main02_freeze	linear	0	678	2026-09-13T05:37:16Z
  main02_freeze	mlp	0	8347	2026-09-13T07:56:24Z
  suite_spatial02	linear	0	444	2026-09-13T08:03:49Z
  suite_object01	linear	0	391	2026-09-13T08:10:21Z
  suite_long01	linear	0	780	2026-09-13T08:23:22Z
  suite_spatial02	mlp	0	8370	2026-09-13T10:42:53Z
  ```
  
  Sum of ledger elapsed: **34245 s = 9.51 h** over 8 rounds (excludes the ~92 s aborted jobs=192 attempt).
  Budget clock: wave_start 2026-09-13T01:08:47Z → last round end 2026-09-13T10:42:53Z = **34446 s = 9.57 h** of 14 h.
  
  | round | probe samples | elapsed_s | s/sample |
  |---|---|---|---|
  | main02 mlp | 1266 | 7669 | 6.06 |
  | main02_mirror mlp | 1226 | 7566 | 6.17 |
  | main02_freeze linear | 1056 | 678 | 0.64 |
  | main02_freeze mlp | 1056 | 8347 | 7.90 |
  | suite_spatial02 linear | 1202 | 444 | 0.37 |
  | suite_object01 linear | 1478 | 391 | 0.26 |
  | suite_long01 linear | 2354 | 780 | 0.33 |
  | suite_spatial02 mlp | 1202 | 8370 | 6.96 |
  
  ## 9. Budget changes and git
  
  - 2026-09-13T09:04Z: the operator stated the remaining budget was 2 h 04 min (hard end ~11:06Z), replacing the 14 h probe clock. `suite_object01:mlp` was cut in response (see §7). The deadline for `suite_spatial02` MLP was set at 10:45Z; it finished at 10:42:53Z, rc=0, COMPLETE.
  - Final state: 10 of 12 probe rounds COMPLETE (linear: all 6 runs; MLP: main02, main02_mirror, main02_freeze, suite_spatial02). main02 and main02_mirror linear came from the earlier pod; 8 rounds ran on this pod. Cut: suite_object01 MLP, suite_long01 MLP. Skipped: suite_spatial01.
  - Git: the first `git push` at 09:05Z failed: `fatal: could not read Username for 'https://github.com': No such device or address`. After the operator provided a token, commits through `2600bd15` were pushed at ~10:05Z (fast-forward `cc5bc289..2600bd15`, no force).
  - Tarball (`/workspace/exports/p0_wave2_<date>.tar.gz` + `.sha256`) stays on the volume and is not committed to git.
- `2026-10-26 03:10` **T-CODE (agent). E4 (SFT vs GRPO) prepared; no E4 run exists yet, S0 included.**
  Scope per README S4 / E4: the wave-2 battery, unchanged, on
  A = `RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora` (SFT) and
  B = `RLinf/RLinf-OpenVLAOFT-LIBERO-130` (GRPO from A), both
  `unnorm_key: libero_130_no_noops_trajall`; goal-traj1 wave-2 runs are the
  third point and are not re-collected. Authoring box has no GPU and no torch:
  nothing below was run against a model.
  **How A loads (checked before any smoke run, from the weights).** A's HF repo
  ships full root shards *and* a `lora_adapter/` (r=32, alpha=16, base
  `openvla-7b`), while the harness hard-codes `is_lora=False`. Single tensors
  were read from the published shards by HTTP range request
  (`e4_ckpt_check.py merged-test`, revisions pinned in `e4_plan.yaml`):
  `language_model.model.layers.0.input_layernorm.weight` (no adapter targets
  it) is **bit-identical** to `openvla/openvla-7b`; the targeted
  `layers.0.self_attn.q_proj` and vision `blocks.0.attn.qkv` differ from the
  base by **effective rank 64** above the bf16 rounding floor (singular values
  0.158 -> 0.021 and 0.174 -> 0.012 at the edge; 99.999% / 99.994% of the
  delta's energy), i.e. two merged rank-32 LoRA rounds; the shipped adapter
  explains only part of that delta (cosine 0.268 / 0.580). B sits
  ||B-A||/||A|| = 0.00045 (q_proj) / 0.0033 (qkv) from A's root but 0.203 /
  0.166 from root + adapter, and B's layernorm moved (full-parameter RL). So
  A = the root as published, loaded with `is_lora=false`; applying
  `lora_adapter/` would apply a merged round twice. The model card agrees ("to
  evaluate the model directly, set is_lora to false"). `download` excludes the
  adapter. Correction to a working note: an early look printed the spectrum
  rounded to 3 decimals and read "exact rank 64"; the tail is bf16 noise, not
  zero, and the script now counts rank against a simulated rounding floor.
  **Other checkpoint facts.** `dataset_statistics.json` holds exactly one key,
  `libero_130_no_noops_trajall`, in both. B carries a 5-tensor `value_head`
  that A lacks; the harness sets `add_value_head=false`, so it is dropped at
  load. Pinned: A `a4ea91b3897648f3c1395e5221edb40b8f53d1a4`, B
  `473954241c8148bc9a1bf9ed547a74523e4a7d41`, four shard sha256s each.
  **Protocol decisions (fixed now, not after seeing data).** (1) Decoding:
  every S1/S2 arm samples at T=1.6 (= main02). The card's 49.40% goal number
  for A is *greedy*; A's success here is reported as measured. pilot.yaml's
  `do_sample: false` predates wave 1's switch to sampling, so the E4 pilot
  overrides it to match S2. (2) `env.num_envs=10` for every arm unless
  `s0-envs20` passes on both checkpoints and the operator locks 20; 10 keeps
  batch partitioning identical to all wave-2 runs. (3) Episodes are the
  wave-2 configs' own: goal swap 200, mirror/freeze 150 (tasks {2,3,6,8,9}),
  spatial/object 200, **long 150** (`suite_long.yaml`, 15 states x 10 tasks),
  not the 200 in the request, so long stays comparable with `suite_long01`.
  (4) Seeds: goal arms s1 = seed 1 (= main02), s2 = seed 2; suites seed 0
  (= wave 2). (5) Horizon stays at the configs' 40 calls = 320 steps
  (long: 64 = 512); `env/libero_130.yaml` trains/evaluates at 512, so the
  cards' success numbers need not transfer to the 320-step suites.
  **Tooling.** `e4_plan.yaml` (each run = wave-2 config + listed overrides;
  `tests/test_e4.py` fails if any other field moves), `e4_wave.py` (stages,
  gates, ledger, 2x watchdog, never-reuse run ids, `--jobs` from the cgroup
  quota), `e4_report.py`, `e4_ckpt_check.py`, `modal_e4.py` (untested against
  a live Modal account). Offline: `test_offline.py` 56 passed,
  `test_e4.py` 26 passed. Nothing under `rlinf/`, no probe code, transform or
  gate was modified.
- `2026-10-26 03:10` **E4 PRE-REGISTERED READOUT (fixed before any E4 data exists; S0 not yet run).**
  Notation, per run r and probe family f in {linear, MLP}: F = the selection
  floor (max over the same 27 layer x pool cells of P0, same family); X in
  {P1, P3, P4} at its best of those 27 cells; margin(X) = X - F; C_dstates,
  C_cmd, P2r single-cell, same family; delta = **0.04** balanced accuracy.
  Calibration of delta, done before E4 data: it is the upper end of the
  paper's stated detection limit ("effects below ~0.03-0.04 are undetectable
  at these n"), and applying R1-R2 with delta=0.04 to the committed wave-2
  numbers (`reports/p0_wave2_report.md`) reproduces every published wave-2
  verdict; delta=0.03 would flip `suite_spatial02` and `suite_object01`
  linear to "above floor" by <=0.005, contradicting the paper's rule-(d)
  verdict. Pinned by `test_delta_reproduces_every_published_wave2_verdict`.
  **R1 (rung vs floor).** X is *above floor* iff margin(X) >= 0.04. A run's
  ladder is above floor iff at least one of P1/P3/P4 is; *at floor* iff all
  three margins are < 0.04.
  **R2 (observables-only twin, paper T5).** An above-floor rung counts as
  *beyond observables* only if X >= C_dstates + 0.04 (same run, same family)
  and C_cmd is in [0.45, 0.55]; otherwise it is "explained by outcome
  perception".
  **R3 (informative run, paper T4).** A run is informative only if
  P2r >= F + 0.10; otherwise its verdict is "uninformative (intervention too
  weak)", never "at floor".
  **R4 (the E4 question).** Checkpoint k's ladder is *above its MLP selection
  floor* iff on BOTH libero_goal swap seeds, under the MLP family, at least
  one rung is above floor AND beyond observables, and both runs are
  informative. k is *at floor* iff on both seeds all three MLP rungs are at
  floor and both runs are informative. Anything else (one seed only, a seed
  explained by C_dstates, an uninformative or missing run) is
  *inconclusive / not replicated* and is never reported as a positive. The
  same rule under the linear family is the secondary verdict.
  **R5 (contrast; = paper Sec. 6 pre-registration).** (a) GRPO above, Base-Lora
  at floor -> RL post-training creates an attribution-relevant representation
  SFT lacks. (b) Both at floor -> RL post-training does not create a readable
  attribution trace. (c) Both above -> present in both, not RL-specific
  (then compare with goal-traj1 `main02`, at floor in wave 2). (d) Base-Lora
  above, GRPO at floor -> reported as stated. (e) Either inconclusive -> the
  contrast is inconclusive.
  **R6 (mirror/freeze arms).** Above-floor rungs are expected there (visible
  outcomes). They count only through R2: a rung beyond observables on both
  seeds for GRPO but not for Base-Lora supports (a); otherwise these arms are
  not evidence for attribution.
  **R7 (E3: novelty-gated -> mismatch-gated).** Wave-2 SFT fingerprint: mirror
  significant, freeze null. GRPO's fingerprint has *migrated* iff on both
  GRPO freeze seeds the next call reacts in the surprise direction
  (Delta-entropy > 0 with p < 0.05, or Delta-logprob < 0 with p < 0.05;
  episode-level Wilcoxon) while Base-Lora freeze does so on neither seed.
  Otherwise *not migrated*. Swap E3 is reported but not used (wave 2: 2/5
  swap runs significant).
  **R8 (compensation).** Clean-minus-probe success gap per arm. *Strengthens*
  iff GRPO's gap is smaller than Base-Lora's on both swap and both mirror
  seeds. Used only to split (b): (b) + strengthens -> "RL sharpens the
  visual-servoing loop without new monitoring machinery".
  **R9 (E1 positive control).** Best-cell E1 R^2 (linear) per goal swap run;
  if < 0.30 on an E4 run, that checkpoint's verdict is withheld pending
  inspection. Differences in E1 between checkpoints are descriptive.
  **R10 (suites).** Spatial/object/long are one seed, linear only (S3 is
  goal-only): R1-R3 per suite, labelled "single-seed, linear"; they can
  corroborate, never establish, R5. Never pooled across suites.
  **R11 (n).** GRPO episodes end early, so its runs will carry fewer probe
  samples; floors are per run and absorb this. A run with n < 600 is labelled
  "low n" next to its verdict.
  **R12 (competence).** Per run and task: success rate (non-censored
  episodes), intervention strength (median ||a_exec - a_cmd|| / ||a_cmd|| on
  the 6 motion dims of hijacked calls) and SELF-call end-effector motion. A
  task is flagged if success <= 0.10, or (with >= 3 tasks) strength or motion
  <= 0.5x the run's across-task median. Flagged tasks are listed and kept in
  every analysis; none is dropped.
  `e4_report.py` applies R1-R12 mechanically; `e4_wave.py s2` refuses to start
  unless this entry is in the logbook.
- `2026-10-26 03:33` **T-CODE (agent).** Test count at commit, superseding the 03:10 entry's "test_e4.py 26": two tests were added after it -- an end-to-end readout test (R4-R9 on a complete synthetic E4 row set) and an end-to-end stage-flow test (`init -> s0 -> lock -> s1 -> s2 -> s3 -> report`, `tests/fake_collect.py` standing in for the GPU collector, real `run_probes.py` behind it, then the halt / new-id / two-strike failure paths). Offline suite: `test_offline.py` 56 + `test_e4.py` 28 = **84 passed** (py3.11, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.9.1, joblib 1.6.0 -- the wave-2 analysis versions). `reports/e4_report.md` is committed as the pre-data skeleton: wave-2 rows only (transcribed, marked †), every E4 row `[TO VERIFY: not collected]`. Operator note: this file carries 361 NUL bytes from the wave-2 MooseFS outage, so plain `grep` treats it as binary -- use `grep -a`, and only ever append to it.
- `2026-10-26 05:00` **T-INFRA (agent, E4 Modal session).** First live run of `modal_e4.py`. The image build (install.sh embodied openvla-oft/libero) succeeded; flash-attn 2.8.3 came from the RLinf fallback wheel after the upstream 404. Every container then failed at import: Modal loads the launcher as `/root/modal_e4.py`, so `Path(__file__).resolve().parents[2]` raised `IndexError: 2`. Fix: `REPO_LOCAL` is the client repo root only when `modal.is_local()`, else the mounted `/root/RLinf` (`_repo_local`; test `test_modal_launcher_imports_inside_the_container`). Nothing collected, gated or read out changed. Offline suite: **85 passed** (84 + the new test). After the fix `stage --cmd status` returned rc=0 (empty ledger, 0.00 / 12.0 GPU-h); `download` sft and grpo both rc=0 with `"passed": true`. Local `e4_ckpt_check.py merged-test`: `"passed": true`, same numbers as the 03:10 entry (rank 64; cos 0.268 / 0.580; GRPO-root 0.00045 / 0.0033).
- `2026-10-26 04:20` **T-INFRA (agent, E4 Modal session).** First `s0` (MUJOCO_GL=egl): both smokes reached the model (indexing check passed on both, max |dlogprob| 2.4e-07) and then failed at the first parquet write, rc=1: `e4_sft_smoke` 1.7 min, `e4_grpo_smoke` 1.5 min, both `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`. Cause, checked in the image: install.sh leaves numpy 1.26.4 plus an unpinned pyarrow 26.0.0, which raises `pyarrow requires NumPy 2.0 or newer, found 1.26.4`. Not an EGL failure. Fix: the image's analysis layer pins `pyarrow<20` (`ANALYSIS_PINS`, matching the local analysis venv; test `test_modal_image_pins_a_pyarrow_that_imports_under_numpy_1`). Does not change what is collected, gated or read out. Both failed run dirs are left as they are; the re-run gets new ids. Offline suite: **86 passed**.
- `2026-10-26 04:36` **T-INFRA + S0 (agent, E4 Modal session).** The `pyarrow<20` pin first broke the image build: `run_commands` goes through `/bin/sh`, where a bare `pyarrow<20` is an input redirect (exit 2). Pins are now `shlex.quote`d (test `test_modal_image_pin_survives_the_build_shell`); in the image: numpy 1.26.4, pyarrow 19.0.1, pandas 2.3.3, scikit-learn 1.9.1. Offline suite **87 passed**. `s0` re-run (MUJOCO_GL=egl, L40S): `e4_sft_smoke_r2` rc=0, 60.1 s; `e4_grpo_smoke_r2` rc=0, 50.1 s (ledger `elapsed_s`; expected 683). Gates s0 sft and grpo **PASSED** (indexing max |dlogprob| 2.38e-07, hidden tail (3,3,4096), labels follow the schedule). The first attempts `e4_sft_smoke` / `e4_grpo_smoke` (rc=1, pyarrow) are kept. Speed = max(1.0, round(55.1/683, 2)) = **1.0**. Locked: num_envs=10, MUJOCO_GL=egl, speed_factor=1.0. GPU ledger 0.08 / 12.0 h. Note: the ledger's `peak_vram_mib` reads 9 MiB on both ok smokes (15848 on the failed sft attempt), so it looks undersampled on sub-minute runs; it gates nothing at num_envs=10.
- `2026-10-26 04:45` **S1 STOP (agent, E4 Modal session).** `s1` rc=1: both gates **FAILED**; per the operator settings (GATE_MODE auto, PILOT_CONTROL_BAND strict) the wave stops here. No re-run, no S2. Values from the gate JSONs (`e4_gates`):
  | check | sft `e4_sft_pilot` | grpo `e4_grpo_pilot` |
  |---|---|---|
  | collection ok (wall / expected 600 s) | yes, 2.3 min, 544 calls | yes, 1.7 min, 313 calls |
  | plumbing, zero mismatches | pass (53/53 hijacks, 0 action/state errors) | pass (25/25, 0 errors) |
  | P0 mean over 27 cells in [0.45, 0.55] | pass, 0.493 (max 0.617, L4/ctx_last = pilot floor) | **fail, 0.411** (max 0.537, L0/ctx_mean) |
  | C_cmd in band | pass, 0.546 (sd 0.107) | **fail, 0.367** (sd 0.139) |
  | C_phase in band | **fail, 0.552** (sd 0.119) | **fail, 0.267** (sd 0.190) |
  | step-index overlap | pass | pass |
  | P2r | 0.715 (sd 0.178) | 0.300 (sd 0.139) |
  | probe samples (per class) | 43 / 43 | 14 / 14 |
  | success (task 0, 20 ep): overall / probe / clean | 0.55 / 0.50 / 0.625 | 0.95 / 0.917 / 1.00 |
  | flagged tasks | none | none |
  The sft failure is C_phase only, 0.002 above the band edge with a fold sd of 0.119. The grpo pilot has 28 probe samples (GRPO succeeds in a median ~8-9 calls), so every grpo control sits far from 0.5 with fold sds of 0.14-0.19 and below-chance values, which reads as small n rather than a leak [interpretation, not a gate result]. Whether the pilot needs more episodes for GRPO is a protocol question for the operator. GPU ledger 0.15 / 12.0 h. No efference-e4 app left running.
- `2026-10-26 06:13` **E4 PRE-REGISTRATION AMENDMENT 1 (operator-approved; adopted after S1 failed, before any S2 data exists).**
  Trigger: the S1 gates of 04:45. sft failed only C_phase, 0.552 against the
  band edge 0.55 with a fold sd of 0.119. grpo had 14 probe samples per class:
  GRPO finishes task 0 in a median ~8-9 calls, and under warmup 4 / clean gap 2
  a short episode offers about one hijack, so every grpo control sat far from
  0.5 with fold sds of 0.14-0.19. Stated plainly: these rules are chosen after
  seeing S1. The 2-sd rule was proposed to the operator before S1 ran (the
  strict band had misjudged synthetic pilots at C_cmd 0.570 and 0.430), but it
  was never in this logbook, so it counts as post-S1. The amendment touches
  only the pilot QA gate and the GRPO episode count. **R1-R12 are unchanged.**
  **(A) Pilot gate, S1b only.** Unchanged: P0 mean over its 27 cells in
  [0.45, 0.55], plumbing with zero mismatches, step-index overlap. Changed:
  C_cmd and C_phase pass iff |bacc - 0.5| <= 2 x their fold sd, and the gate
  counts only with >= 40 probe samples per class. The strict band is still
  reported. S1's gate files and verdicts stay as recorded (both FAILED).
  **(B) pilot2 (stage s1b).** pilot.yaml (swap, T=1.6, the locked num_envs=10
  and EGL) over all 10 libero_goal tasks x 8 initial states = 80 episodes, for
  both checkpoints, under new ids `e4_<ckpt>_pilot2`. It serves as the gate
  and as a like-for-like yield measurement on the same tasks.
  **(C) Sizing: GRPO matched to SFT on probe samples, not on episodes.**
  r = (SFT kept positives per finished pilot2 episode) / (the same for GRPO).
  For each GRPO S2 entry, k = min(50, max(k_base, ceil(k_base x r))), where
  k_base is the wave-2 config's own init_states_per_task (goal swap 20,
  mirror/freeze 30, spatial/object 20, long 15). SFT arms are never resized,
  GRPO is never sized below k_base, and 50 is LIBERO's number of initial
  states per task, so no initial state repeats and parity may not be
  reachable. Task-0 yields from S1 (sft 43/20 = 2.15, grpo 14/20 = 0.70,
  r ~ 3.1) would put every GRPO arm at the cap of 50 except long (47); pilot2
  decides. `e4_wave.py size` freezes k before any S2 run and refuses
  afterwards; S2 refuses to start without it. The watchdog expectation of a
  sized arm scales by k / k_base. Assumption, flagged: r is measured on goal
  and applied unchanged to the spatial / object / long suites.
  **(D) Reading consequences.** GRPO arms carry more episodes than SFT arms
  (up to 2.5x on goal swap), so E3 episode counts and success-rate precision
  differ by checkpoint. Floors stay per run, R11 still labels n < 600, and
  every row reports its n.
  **(E) Unchanged.** R1-R12, decoding, num_envs, rendering backend, seeds,
  every SFT arm, and MLP probing on the goal suite only.
- `2026-10-26 06:55` **S1b + sizing (agent, E4 Modal session).** Operator answer verified in the parent transcript (06:07, "修订并按样本量对齐 (Recommended)"); pulled 8f770fd, offline suite 91 passed. `s1b` rc=0, both gates **PASSED** under Amendment 1 (values from the gate JSONs):
  | check | sft `e4_sft_pilot2` | grpo `e4_grpo_pilot2` |
  |---|---|---|
  | collection (wall / expected 900 s) | ok, 6.5 min, 1958 calls, peak tree RSS 62.51 GB | ok, 4.7 min, 1291 calls, 61.22 GB |
  | plumbing, zero mismatches | pass (218 hijacks, 0 action / state errors) | pass (131, 0) |
  | P0 mean over 27 cells in [0.45, 0.55] | pass, 0.510 (max 0.538 L32/ctx_last) | pass, 0.451 (max 0.500 sd 0.000, L0/ctx_last) |
  | C_cmd within 2 sd of 0.5 (strict band) | pass, 0.492 sd 0.049 (in band) | pass, 0.480 sd 0.034 (in band) |
  | C_phase within 2 sd of 0.5 (strict band) | pass, 0.492 sd 0.039 (in band) | pass, 0.479 sd 0.039 (in band) |
  | >= 40 probe samples per class | pass, 191 / 191 | pass, 97 / 97 |
  | step-index overlap (median, class 0 / 1) | pass (18 / 16) | pass (9 / 10) |
  | P2r / C_dstates | 0.772 / 0.479 | 0.613 / 0.494 |
  | success overall / probe / clean (80 ep) | 0.588 / 0.578 / 0.625 | 0.925 / 0.922 / 0.938 |
  | flagged tasks (R12) | [7] (strength 0.581, motion 0.0103) | [5] (strength 0.226) |
  `size` rc=0: positives/episode sft 2.3875, grpo 1.2125, r = 1.97; GRPO S2 init_states_per_task goal_swap_s1/s2 40, goal mirror/freeze s1/s2 50 (cap; uncapped 59), spatial 40, object 40, long 30. GPU ledger 0.37 / 12.0 h [TO VERIFY: 0.26 h before grpo pilot2 + its 4.7 min]. **Infra:** the GPU containers' memory limit is raised from 64 GiB to 96 GiB (`GPU_MEMORY`, request unchanged at 48 GiB; test `test_modal_gpu_limit_clears_the_measured_pilot_rss`), because pilot tree RSS grew with call count (61.0 -> 62.5 GB) toward the old limit and S2 runs are several times longer. Collection is unchanged. Offline suite **92 passed**. Next: `modal run --detach modal_e4.py::wave`.
- `2026-10-26 07:10` **T-INFRA (agent, E4 Modal session).** `modal run --detach modal_e4.py::wave` (06:52, app ap-nwnJdJtbs5ePi9tNRWwAsO): the GPU lane spawned and began `e4_sft_goal_swap_s1`, but `orchestrate` returned on its first poll with no probes. Cause: in modal 1.6 `FunctionCall.get(timeout=0)` on a running call raises the *builtin* `TimeoutError` (`modal/_functions.py` `poll_function`), not `modal.exception.TimeoutError`, so the broad `except Exception` read the lane as finished. The detached app keeps the lane running, so collection is unaffected, but nothing would probe the runs. Fix: `_finished()` treats both TimeoutErrors as running and `OutputExpiredError` as finished, and `wave --no-lane` (`orchestrate(start_lane=False)`) probes behind the lane already running, treating the lane as done when `todo` is halted or has nothing to collect, so a second GPU lane is never started (test `test_modal_orchestrator_does_not_read_a_running_call_as_finished`). Collection, gates and readout are unchanged. Offline suite **93 passed**.
- `2026-10-26 07:50` **T-INFRA (agent, E4 Modal session).** The `wave --no-lane` orchestrator (ap-gu89FYCRzafZL6HGfvmOqh) ran one linear round (`e4_sft_goal_swap_s1`, rc=0, 6.3 min) and spawned its MLP round (07:10:41, `--jobs 24`), then had its input cancelled at 07:12:49. That was exactly the `timeout 600` the agent had put on the local `modal run --detach` client: a detached `.remote()` input is cancelled when its client dies. The MLP round kept running as the last triggered function; nothing else was spawned, so `e4_grpo_goal_swap_s1` and `e4_sft_goal_swap_s2` linear waited. The GPU lane (ap-nwnJ...) is unaffected. Fix: `wave` now `spawn`s the orchestrator and returns, and `--skip run_id:family` names rounds already in flight. A second `probe` on a run would rename its live analysis dir to `.partial_*`, so the running `e4_sft_goal_swap_s1:mlp` is skipped, never duplicated (helper `_offered`, test `test_modal_orchestrator_never_offers_a_round_already_in_flight`). Offline suite **94 passed**. Noted, not changed: `resolve_jobs` gave the 8-core CPU containers 24 workers (the cgroup quota Modal grants), above the wave-2 pod's 16.
- `2026-10-26 09:45` **STOP: Modal workspace spend limit (agent, E4 Modal session).** At the 09:37 check-in both apps (GPU lane ap-nwnJ..., probe orchestrator ap-lfTV...) had 0 tasks, and every `modal run` failed with `Workspace ac-VNEscHwebuqYCe1fZEJ4eY has exceeded its spend limit`. That is the workspace's own limit on Modal, not the operator's $80 SPEND_CAP. The Volume can still be read. State from `e4_ledger` (every completed entry rc=0, outcome ok):
  - Collected (16/18): sft and grpo goal_swap_s1/s2, goal_mirror_s1/s2, goal_freeze_s1/s2, spatial_swap_s0, object_swap_s0. Not complete: `e4_sft_long_swap_s0` (run dir exists, no ledger entry: interrupted; a re-run gets a new id) and grpo long_swap_s0 (never started).
  - Linear rounds: 15/16 collected runs; `e4_grpo_object_swap_s0` linear was cut off.
  - MLP rounds: 2/12 done (`e4_sft_goal_swap_s1` 6720 s, `e4_sft_goal_swap_s2` 6125 s); the other 10 goal-suite rounds were running when the limit hit and left no markers.
  - S2 collection walls, s / expected s x k/k_base: sft goal swap 566 / 571 (1408); grpo goal swap 741 / 776 (2816); sft mirror 456 / 451, grpo mirror 666 / 686 (2505); sft freeze 401 / 400, grpo freeze 556 / 546 (2530); sft spatial 601 (1988), grpo spatial 721 (3976); sft object 621 (1506), grpo object 701 (3012). Every run was well under its watchdog.
  - GPU collection seconds over the whole wave: 10127 s from pilot2 on + ~538 s for S0/S1 = **~2.96 GPU-h** [TO VERIFY against `status` when Modal is reachable]. Ledger probe seconds: 21880 s completed, plus ~10 MLP rounds killed after up to ~2 h each.
  - Spend by the operator formula: ~2.96 x $2.81 = ~$8.3 GPU, plus CPU [TO VERIFY: the killed MLP containers are not in the ledger]. The workspace's actual billed total is on the Modal dashboard [TO VERIFY].
  No resume was attempted. Nothing on the Volume was deleted.
- `2026-10-26 04:12:35` **E4 (driver)** collect `e4_sft_smoke` (s0, sft, smoke.yaml); expected 11 min, stop at 23 min; MUJOCO_GL=egl; GPU used so far 0.00/12.0 h
- `2026-10-26 04:12:38` **START** run_id=`e4_sft_smoke` config=`/root/RLinf/experiments/efference_probe/configs/smoke.yaml` hash=`ec79fdd40555` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_smoke out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.3794 notes=E4 sft smoke (Base-Lora)`
- `2026-10-26 04:12:43` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 04:13:51` building LiberoEnv (2 parallel envs)
- `2026-10-26 04:14:09` indexing check passed: {'max_abs_logprob_diff': 2.384185791015625e-07, 'tolerance': 0.11956295013427735, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 04:14:09` heartbeat: task 0 batch 0 call 0 | 0.03/0.4 h, 0.00/2 GB, 0 calls
- `2026-10-26 04:14:12` **FAILED** run_id=`e4_sft_smoke`: ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
- `2026-10-26 04:14:15` **E4 (driver)** collect `e4_sft_smoke` failed (None): rc=1 1.7 min (expected 11), calls=-1, peak RSS 25.42 GB, peak VRAM 15848 MiB
- `2026-10-26 04:14:15` **E4 (driver)** gate s0 sft (`e4_sft_smoke`): **FAILED** ['collection_ok', 'indexing_check_passed', 'max_abs_logprob_diff_below_1e-3', 'no_null_cells', 'hidden_calls_match_rows', 'labels_follow_schedule']
- `2026-10-26 04:14:16` **E4 (driver)** collect `e4_grpo_smoke` (s0, grpo, smoke.yaml); expected 11 min, stop at 23 min; MUJOCO_GL=egl; GPU used so far 0.03/12.0 h
- `2026-10-26 04:14:18` **START** run_id=`e4_grpo_smoke` config=`/root/RLinf/experiments/efference_probe/configs/smoke.yaml` hash=`f1d70b24b9ab` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_smoke out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.3794 notes=E4 grpo smoke (GRPO)`
- `2026-10-26 04:14:21` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 04:15:30` building LiberoEnv (2 parallel envs)
- `2026-10-26 04:15:39` indexing check passed: {'max_abs_logprob_diff': 2.384185791015625e-07, 'tolerance': 0.09638646125793457, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 04:15:40` heartbeat: task 0 batch 0 call 0 | 0.02/0.4 h, 0.00/2 GB, 0 calls
- `2026-10-26 04:15:42` **FAILED** run_id=`e4_grpo_smoke`: ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
- `2026-10-26 04:15:47` **E4 (driver)** collect `e4_grpo_smoke` failed (None): rc=1 1.5 min (expected 11), calls=-1, peak RSS 21.84 GB, peak VRAM 4 MiB
- `2026-10-26 04:15:47` **E4 (driver)** gate s0 grpo (`e4_grpo_smoke`): **FAILED** ['collection_ok', 'indexing_check_passed', 'max_abs_logprob_diff_below_1e-3', 'no_null_cells', 'hidden_calls_match_rows', 'labels_follow_schedule']
- `2026-10-26 04:28:40` **E4 (driver)** `e4_sft_smoke`: earlier attempt(s) ['e4_sft_smoke'] never completed; collecting as `e4_sft_smoke_r2` (interrupted collection = new id; nothing overwritten or deleted)
- `2026-10-26 04:28:42` **E4 (driver)** collect `e4_sft_smoke_r2` (s0, sft, smoke.yaml); expected 11 min, stop at 23 min; MUJOCO_GL=egl; GPU used so far 0.05/12.0 h
- `2026-10-26 04:28:44` **START** run_id=`e4_sft_smoke_r2` config=`/root/RLinf/experiments/efference_probe/configs/smoke.yaml` hash=`ec79fdd40555` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_smoke_r2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.3794 notes=E4 sft smoke (Base-Lora)`
- `2026-10-26 04:28:48` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 04:29:19` building LiberoEnv (2 parallel envs)
- `2026-10-26 04:29:35` indexing check passed: {'max_abs_logprob_diff': 2.384185791015625e-07, 'tolerance': 0.11956295013427735, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 04:29:38` done: 12 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_smoke_r2/calls.parquet (0.00 GB)
- `2026-10-26 04:29:38` **DONE** run_id=`e4_sft_smoke_r2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_smoke_r2/calls.parquet`
- `2026-10-26 04:29:42` **E4 (driver)** collect `e4_sft_smoke_r2` ok: rc=0 1.0 min (expected 11), calls=12, peak RSS 21.39 GB, peak VRAM 9 MiB
- `2026-10-26 04:29:42` **E4 (driver)** gate s0 sft (`e4_sft_smoke_r2`): **PASSED**
- `2026-10-26 04:29:42` **E4 (driver)** `e4_grpo_smoke`: earlier attempt(s) ['e4_grpo_smoke'] never completed; collecting as `e4_grpo_smoke_r2` (interrupted collection = new id; nothing overwritten or deleted)
- `2026-10-26 04:29:43` **E4 (driver)** collect `e4_grpo_smoke_r2` (s0, grpo, smoke.yaml); expected 11 min, stop at 23 min; MUJOCO_GL=egl; GPU used so far 0.07/12.0 h
- `2026-10-26 04:29:44` **START** run_id=`e4_grpo_smoke_r2` config=`/root/RLinf/experiments/efference_probe/configs/smoke.yaml` hash=`f1d70b24b9ab` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_smoke_r2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.3794 notes=E4 grpo smoke (GRPO)`
- `2026-10-26 04:29:47` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 04:30:17` building LiberoEnv (2 parallel envs)
- `2026-10-26 04:30:25` indexing check passed: {'max_abs_logprob_diff': 2.384185791015625e-07, 'tolerance': 0.09638646125793457, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 04:30:27` done: 12 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_smoke_r2/calls.parquet (0.00 GB)
- `2026-10-26 04:30:27` **DONE** run_id=`e4_grpo_smoke_r2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_smoke_r2/calls.parquet`
- `2026-10-26 04:30:33` **E4 (driver)** collect `e4_grpo_smoke_r2` ok: rc=0 0.8 min (expected 11), calls=12, peak RSS 21.57 GB, peak VRAM 9 MiB
- `2026-10-26 04:30:33` **E4 (driver)** gate s0 grpo (`e4_grpo_smoke_r2`): **PASSED**
- `2026-10-26 04:35:16` **E4 (driver)** wave constants locked for every S1/S2 arm: num_envs=10, MUJOCO_GL=egl, speed_factor=1.0
- `2026-10-26 04:36:04` **E4 (driver)** collect `e4_sft_pilot` (s1, sft, pilot.yaml); expected 10 min, stop at 20 min; MUJOCO_GL=egl; GPU used so far 0.08/12.0 h
- `2026-10-26 04:36:07` **START** run_id=`e4_sft_pilot` config=`/root/RLinf/experiments/efference_probe/configs/pilot.yaml` hash=`60f9b3458f44` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_pilot out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.3333 notes=E4 sft pilot (Base-Lora) env.num_envs=10 model.do_sample=true model.temperature=1.6`
- `2026-10-26 04:36:12` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 04:36:45` building LiberoEnv (10 parallel envs)
- `2026-10-26 04:37:16` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.23237913131713867, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 04:38:16` done: 544 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_pilot/calls.parquet (0.10 GB)
- `2026-10-26 04:38:16` **DONE** run_id=`e4_sft_pilot` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_pilot/calls.parquet`
- `2026-10-26 04:38:21` **E4 (driver)** collect `e4_sft_pilot` ok: rc=0 2.3 min (expected 10), calls=544, peak RSS 60.98 GB, peak VRAM 25464 MiB
- `2026-10-26 04:38:21` **E4 (driver)** probe `e4_sft_pilot` pilot (--jobs 12); stop at 0.2 h
- `2026-10-26 04:38:36` **E4 (driver)** probe `e4_sft_pilot` pilot ok: rc=0 0.2 min, plumbing.passed=True
- `2026-10-26 04:38:36` **E4 (driver)** gate s1 sft (`e4_sft_pilot`): **FAILED** ['C_phase_in_band']
- `2026-10-26 04:38:37` **E4 (driver)** collect `e4_grpo_pilot` (s1, grpo, pilot.yaml); expected 10 min, stop at 20 min; MUJOCO_GL=egl; GPU used so far 0.12/12.0 h
- `2026-10-26 04:38:39` **START** run_id=`e4_grpo_pilot` config=`/root/RLinf/experiments/efference_probe/configs/pilot.yaml` hash=`6d2ee385d18a` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_pilot out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.3333 notes=E4 grpo pilot (GRPO) env.num_envs=10 model.do_sample=true model.temperature=1.6`
- `2026-10-26 04:38:43` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 04:39:14` building LiberoEnv (10 parallel envs)
- `2026-10-26 04:39:31` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.19337639808654788, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 04:40:14` done: 313 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_pilot/calls.parquet (0.06 GB)
- `2026-10-26 04:40:14` **DONE** run_id=`e4_grpo_pilot` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_pilot/calls.parquet`
- `2026-10-26 04:40:17` **E4 (driver)** collect `e4_grpo_pilot` ok: rc=0 1.7 min (expected 10), calls=313, peak RSS 60.13 GB, peak VRAM 25463 MiB
- `2026-10-26 04:40:17` **E4 (driver)** probe `e4_grpo_pilot` pilot (--jobs 12); stop at 0.2 h
- `2026-10-26 04:40:37` **E4 (driver)** probe `e4_grpo_pilot` pilot ok: rc=0 0.3 min, plumbing.passed=True
- `2026-10-26 04:40:38` **E4 (driver)** gate s1 grpo (`e4_grpo_pilot`): **FAILED** ['P0_mean_over_cells_in_band', 'C_cmd_in_band', 'C_phase_in_band']
- `2026-10-26 06:29:53` **E4 (driver)** collect `e4_sft_pilot2` (s1b, sft, pilot.yaml); expected 15 min, stop at 30 min; MUJOCO_GL=egl; GPU used so far 0.15/12.0 h
- `2026-10-26 06:29:56` **START** run_id=`e4_sft_pilot2` config=`/root/RLinf/experiments/efference_probe/configs/pilot.yaml` hash=`54e749573351` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_pilot2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.5000 notes=E4 sft pilot2 (Base-Lora) env.num_envs=10 model.do_sample=true model.temperature=1.6 env.task_ids=null env.init_states_per_task=8`
- `2026-10-26 06:30:01` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 06:30:36` building LiberoEnv (10 parallel envs)
- `2026-10-26 06:31:09` indexing check passed: {'max_abs_logprob_diff': 9.5367431640625e-07, 'tolerance': 0.23847316741943358, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 06:34:57` heartbeat: task 7 batch 0 call 36 | 0.08/0.5 h, 0.28/10 GB, 1452 calls
- `2026-10-26 06:36:17` done: 1958 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_pilot2/calls.parquet (0.37 GB)
- `2026-10-26 06:36:17` **DONE** run_id=`e4_sft_pilot2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_pilot2/calls.parquet`
- `2026-10-26 06:36:27` **E4 (driver)** collect `e4_sft_pilot2` ok: rc=0 6.5 min (expected 15), calls=1958, peak RSS 62.51 GB, peak VRAM 25464 MiB
- `2026-10-26 06:36:27` **E4 (driver)** probe `e4_sft_pilot2` pilot (--jobs 12); stop at 0.2 h
- `2026-10-26 06:38:32` **E4 (driver)** probe `e4_sft_pilot2` pilot ok: rc=0 2.1 min, plumbing.passed=True
- `2026-10-26 06:38:32` **E4 (driver)** `e4_sft_pilot2`: low-competence task flags [7] (kept in every analysis; see e4_derived/<run>/competence.csv)
- `2026-10-26 06:38:32` **E4 (driver)** gate s1b sft (`e4_sft_pilot2`): **PASSED**
- `2026-10-26 06:38:33` **E4 (driver)** collect `e4_grpo_pilot2` (s1b, grpo, pilot.yaml); expected 15 min, stop at 30 min; MUJOCO_GL=egl; GPU used so far 0.26/12.0 h
- `2026-10-26 06:38:35` **START** run_id=`e4_grpo_pilot2` config=`/root/RLinf/experiments/efference_probe/configs/pilot.yaml` hash=`7dc757a87a29` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_pilot2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.5000 notes=E4 grpo pilot2 (GRPO) env.num_envs=10 model.do_sample=true model.temperature=1.6 env.task_ids=null env.init_states_per_task=8`
- `2026-10-26 06:38:39` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 06:39:17` building LiberoEnv (10 parallel envs)
- `2026-10-26 06:39:35` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.2412325668334961, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 06:43:10` done: 1291 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_pilot2/calls.parquet (0.24 GB)
- `2026-10-26 06:43:10` **DONE** run_id=`e4_grpo_pilot2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_pilot2/calls.parquet`
- `2026-10-26 06:43:16` **E4 (driver)** collect `e4_grpo_pilot2` ok: rc=0 4.7 min (expected 15), calls=1291, peak RSS 61.22 GB, peak VRAM 25464 MiB
- `2026-10-26 06:43:16` **E4 (driver)** probe `e4_grpo_pilot2` pilot (--jobs 12); stop at 0.2 h
- `2026-10-26 06:44:31` **E4 (driver)** probe `e4_grpo_pilot2` pilot ok: rc=0 1.3 min, plumbing.passed=True
- `2026-10-26 06:44:31` **E4 (driver)** `e4_grpo_pilot2`: low-competence task flags [5] (kept in every analysis; see e4_derived/<run>/competence.csv)
- `2026-10-26 06:44:31` **E4 (driver)** gate s1b grpo (`e4_grpo_pilot2`): **PASSED**
- `2026-10-26 06:45:04` **E4 (driver)** Amendment 1 sizing frozen: positives/episode sft=2.388, grpo=1.212, r=1.97; grpo S2 init_states_per_task {'goal_swap_s1': 40, 'goal_swap_s2': 40, 'goal_mirror_s1': 50, 'goal_freeze_s1': 50, 'goal_mirror_s2': 50, 'goal_freeze_s2': 50, 'spatial_swap_s0': 40, 'object_swap_s0': 40, 'long_swap_s0': 30} (cap 50)
- `2026-10-26 06:52:48` **E4 (driver)** collect `e4_sft_goal_swap_s1` (s2, sft, main02.yaml); expected 23 min, stop at 47 min; MUJOCO_GL=egl; GPU used so far 0.34/12.0 h
- `2026-10-26 06:52:51` **START** run_id=`e4_sft_goal_swap_s1` config=`/root/RLinf/experiments/efference_probe/configs/main02.yaml` hash=`f061964ffb03` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_goal_swap_s1 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.7822 notes=E4 sft goal_swap_s1 (Base-Lora) env.num_envs=10 env.seed=1 model.do_sample=true model.temperature=1.6`
- `2026-10-26 06:52:54` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 06:53:27` building LiberoEnv (10 parallel envs)
- `2026-10-26 06:53:55` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.2251776885986328, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 06:57:51` heartbeat: task 5 batch 0 call 4 | 0.08/0.8 h, 0.43/30 GB, 2253 calls
- `2026-10-26 07:02:11` done: 4832 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_swap_s1/calls.parquet (0.91 GB)
- `2026-10-26 07:02:11` **DONE** run_id=`e4_sft_goal_swap_s1` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_swap_s1/calls.parquet`
- `2026-10-26 07:02:21` **E4 (driver)** collect `e4_sft_goal_swap_s1` ok: rc=0 9.4 min (expected 23), calls=4832, peak RSS 62.48 GB, peak VRAM 25467 MiB
- `2026-10-26 07:02:25` **E4 (driver)** collect `e4_grpo_goal_swap_s1` (s2, grpo, main02.yaml); expected 47 min, stop at 94 min; MUJOCO_GL=egl; GPU used so far 0.49/12.0 h
- `2026-10-26 07:02:26` **START** run_id=`e4_grpo_goal_swap_s1` config=`/root/RLinf/experiments/efference_probe/configs/main02.yaml` hash=`aaae683ae520` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_goal_swap_s1 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.5644 notes=E4 grpo goal_swap_s1 (GRPO) env.num_envs=10 env.seed=1 env.init_states_per_task=40 model.do_sample=true model.temperature=1.6`
- `2026-10-26 07:02:29` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 07:02:41` **E4 (driver)** probe `e4_sft_goal_swap_s1` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 07:02:59` building LiberoEnv (10 parallel envs)
- `2026-10-26 07:03:14` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.20153170585632324, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 07:07:27` heartbeat: task 3 batch 0 call 38 | 0.08/1.6 h, 0.34/30 GB, 1799 calls
- `2026-10-26 07:08:57` **E4 (driver)** probe `e4_sft_goal_swap_s1` linear ok: rc=0 6.3 min, plumbing.passed=True
- `2026-10-26 07:10:41` **E4 (driver)** probe `e4_sft_goal_swap_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 07:12:33` heartbeat: task 8 batch 0 call 0 | 0.17/1.6 h, 0.97/30 GB, 5133 calls
- `2026-10-26 07:14:42` done: 6346 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_swap_s1/calls.parquet (1.19 GB)
- `2026-10-26 07:14:42` **DONE** run_id=`e4_grpo_goal_swap_s1` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_swap_s1/calls.parquet`
- `2026-10-26 07:14:56` **E4 (driver)** collect `e4_grpo_goal_swap_s1` ok: rc=0 12.3 min (expected 47), calls=6346, peak RSS 61.97 GB, peak VRAM 25467 MiB
- `2026-10-26 07:14:56` **E4 (driver)** `e4_grpo_goal_swap_s1`: low-competence task flags [5] (kept in every analysis; see e4_derived/<run>/competence.csv)
- `2026-10-26 07:15:01` **E4 (driver)** collect `e4_sft_goal_swap_s2` (s2, sft, main02.yaml); expected 23 min, stop at 47 min; MUJOCO_GL=egl; GPU used so far 0.70/12.0 h
- `2026-10-26 07:15:03` **START** run_id=`e4_sft_goal_swap_s2` config=`/root/RLinf/experiments/efference_probe/configs/main02.yaml` hash=`14ed2e58fefb` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_goal_swap_s2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.7822 notes=E4 sft goal_swap_s2 (Base-Lora) env.num_envs=10 env.seed=2 model.do_sample=true model.temperature=1.6`
- `2026-10-26 07:15:06` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 07:15:40` building LiberoEnv (10 parallel envs)
- `2026-10-26 07:15:56` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.25029932022094725, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 07:20:03` heartbeat: task 4 batch 1 call 39 | 0.08/0.8 h, 0.40/30 GB, 2103 calls
- `2026-10-26 07:24:26` done: 4828 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_swap_s2/calls.parquet (0.91 GB)
- `2026-10-26 07:24:26` **DONE** run_id=`e4_sft_goal_swap_s2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_swap_s2/calls.parquet`
- `2026-10-26 07:24:38` **E4 (driver)** collect `e4_sft_goal_swap_s2` ok: rc=0 9.5 min (expected 23), calls=4828, peak RSS 61.17 GB, peak VRAM 25467 MiB
- `2026-10-26 07:24:42` **E4 (driver)** collect `e4_grpo_goal_swap_s2` (s2, grpo, main02.yaml); expected 47 min, stop at 94 min; MUJOCO_GL=egl; GPU used so far 0.86/12.0 h
- `2026-10-26 07:24:44` **START** run_id=`e4_grpo_goal_swap_s2` config=`/root/RLinf/experiments/efference_probe/configs/main02.yaml` hash=`7a690dd3a5fa` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_goal_swap_s2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.5644 notes=E4 grpo goal_swap_s2 (GRPO) env.num_envs=10 env.seed=2 env.init_states_per_task=40 model.do_sample=true model.temperature=1.6`
- `2026-10-26 07:24:47` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 07:25:18` building LiberoEnv (10 parallel envs)
- `2026-10-26 07:25:32` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.27302251815795897, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 07:29:44` heartbeat: task 3 batch 0 call 31 | 0.08/1.6 h, 0.33/30 GB, 1738 calls
- `2026-10-26 07:34:44` heartbeat: task 7 batch 3 call 14 | 0.17/1.6 h, 0.92/30 GB, 4871 calls
- `2026-10-26 07:37:35` done: 6245 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_swap_s2/calls.parquet (1.17 GB)
- `2026-10-26 07:37:35` **DONE** run_id=`e4_grpo_goal_swap_s2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_swap_s2/calls.parquet`
- `2026-10-26 07:37:47` **E4 (driver)** collect `e4_grpo_goal_swap_s2` ok: rc=0 12.9 min (expected 47), calls=6245, peak RSS 62.17 GB, peak VRAM 25467 MiB
- `2026-10-26 07:37:47` **E4 (driver)** `e4_grpo_goal_swap_s2`: low-competence task flags [5] (kept in every analysis; see e4_derived/<run>/competence.csv)
- `2026-10-26 07:37:52` **E4 (driver)** collect `e4_sft_goal_mirror_s1` (s2, sft, main02_mirror.yaml); expected 25 min, stop at 50 min; MUJOCO_GL=egl; GPU used so far 1.07/12.0 h
- `2026-10-26 07:37:53` **START** run_id=`e4_sft_goal_mirror_s1` config=`/root/RLinf/experiments/efference_probe/configs/main02_mirror.yaml` hash=`f380d2b8e140` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_goal_mirror_s1 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.8350 notes=E4 sft goal_mirror_s1 (Base-Lora) env.num_envs=10 env.seed=1 model.do_sample=true model.temperature=1.6`
- `2026-10-26 07:37:56` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 07:38:25` building LiberoEnv (10 parallel envs)
- `2026-10-26 07:38:40` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.235427188873291, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 07:41:21` **E4 (driver)** probe `e4_grpo_goal_swap_s1` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 07:41:23` **E4 (driver)** probe `e4_grpo_goal_swap_s2` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 07:41:24` **E4 (driver)** probe `e4_sft_goal_swap_s2` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 07:42:54` heartbeat: task 8 batch 0 call 18 | 0.08/0.8 h, 0.52/30 GB, 2743 calls
- `2026-10-26 07:45:23` done: 4363 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_mirror_s1/calls.parquet (0.82 GB)
- `2026-10-26 07:45:23` **DONE** run_id=`e4_sft_goal_mirror_s1` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_mirror_s1/calls.parquet`
- `2026-10-26 07:45:33` **E4 (driver)** collect `e4_sft_goal_mirror_s1` ok: rc=0 7.6 min (expected 25), calls=4363, peak RSS 61.12 GB, peak VRAM 25467 MiB
- `2026-10-26 07:45:38` **E4 (driver)** collect `e4_grpo_goal_mirror_s1` (s2, grpo, main02_mirror.yaml); expected 42 min, stop at 84 min; MUJOCO_GL=egl; GPU used so far 1.20/12.0 h
- `2026-10-26 07:45:39` **START** run_id=`e4_grpo_goal_mirror_s1` config=`/root/RLinf/experiments/efference_probe/configs/main02_mirror.yaml` hash=`65cd4b1143e3` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_goal_mirror_s1 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.3917 notes=E4 grpo goal_mirror_s1 (GRPO) env.num_envs=10 env.seed=1 env.init_states_per_task=50 model.do_sample=true model.temperature=1.6`
- `2026-10-26 07:45:42` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 07:46:12` building LiberoEnv (10 parallel envs)
- `2026-10-26 07:46:26` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.20583423614501956, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 07:47:14` **E4 (driver)** probe `e4_sft_goal_swap_s2` linear ok: rc=0 5.8 min, plumbing.passed=True
- `2026-10-26 07:47:20` **E4 (driver)** probe `e4_sft_goal_mirror_s1` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 07:47:25` **E4 (driver)** probe `e4_sft_goal_swap_s2` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 07:50:40` heartbeat: task 6 batch 0 call 11 | 0.08/1.4 h, 0.46/30 GB, 2464 calls
- `2026-10-26 07:54:10` **E4 (driver)** probe `e4_sft_goal_mirror_s1` linear ok: rc=0 6.8 min, plumbing.passed=True
- `2026-10-26 07:55:25` **E4 (driver)** probe `e4_sft_goal_mirror_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 07:55:43` heartbeat: task 9 batch 3 call 0 | 0.17/1.4 h, 0.92/30 GB, 4868 calls
- `2026-10-26 07:56:37` done: 5332 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_mirror_s1/calls.parquet (1.00 GB)
- `2026-10-26 07:56:37` **DONE** run_id=`e4_grpo_goal_mirror_s1` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_mirror_s1/calls.parquet`
- `2026-10-26 07:56:51` **E4 (driver)** collect `e4_grpo_goal_mirror_s1` ok: rc=0 11.1 min (expected 42), calls=5332, peak RSS 61.38 GB, peak VRAM 25467 MiB
- `2026-10-26 07:56:56` **E4 (driver)** collect `e4_sft_goal_freeze_s1` (s2, sft, main02_freeze.yaml); expected 25 min, stop at 51 min; MUJOCO_GL=egl; GPU used so far 1.38/12.0 h
- `2026-10-26 07:56:57` **START** run_id=`e4_sft_goal_freeze_s1` config=`/root/RLinf/experiments/efference_probe/configs/main02_freeze.yaml` hash=`78ef44e8006c` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_goal_freeze_s1 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.8433 notes=E4 sft goal_freeze_s1 (Base-Lora) env.num_envs=10 env.seed=1 model.do_sample=true model.temperature=1.6`
- `2026-10-26 07:57:00` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 07:57:30` **E4 (driver)** probe `e4_grpo_goal_mirror_s1` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 07:57:31` building LiberoEnv (10 parallel envs)
- `2026-10-26 07:57:39` **E4 (driver)** probe `e4_grpo_goal_swap_s2` linear ok: rc=0 16.3 min, plumbing.passed=True
- `2026-10-26 07:57:47` indexing check passed: {'max_abs_logprob_diff': 9.5367431640625e-07, 'tolerance': 0.2522344207763672, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 07:59:28` **E4 (driver)** probe `e4_grpo_goal_swap_s2` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:00:02` **E4 (driver)** probe `e4_grpo_goal_swap_s1` linear ok: rc=0 18.7 min, plumbing.passed=True
- `2026-10-26 08:01:29` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:01:57` heartbeat: task 8 batch 2 call 0 | 0.08/0.8 h, 0.51/30 GB, 2728 calls
- `2026-10-26 08:03:31` done: 3822 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_freeze_s1/calls.parquet (0.72 GB)
- `2026-10-26 08:03:31` **DONE** run_id=`e4_sft_goal_freeze_s1` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_freeze_s1/calls.parquet`
- `2026-10-26 08:03:41` **E4 (driver)** collect `e4_sft_goal_freeze_s1` ok: rc=0 6.7 min (expected 25), calls=3822, peak RSS 60.95 GB, peak VRAM 25467 MiB
- `2026-10-26 08:03:46` **E4 (driver)** collect `e4_grpo_goal_freeze_s1` (s2, grpo, main02_freeze.yaml); expected 42 min, stop at 84 min; MUJOCO_GL=egl; GPU used so far 1.50/12.0 h
- `2026-10-26 08:03:48` **START** run_id=`e4_grpo_goal_freeze_s1` config=`/root/RLinf/experiments/efference_probe/configs/main02_freeze.yaml` hash=`daa037a2499a` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_goal_freeze_s1 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.4056 notes=E4 grpo goal_freeze_s1 (GRPO) env.num_envs=10 env.seed=1 env.init_states_per_task=50 model.do_sample=true model.temperature=1.6`
- `2026-10-26 08:03:50` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 08:04:21` building LiberoEnv (10 parallel envs)
- `2026-10-26 08:04:36` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.21003185272216796, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 08:05:37` **E4 (driver)** probe `e4_sft_goal_freeze_s1` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 08:08:48` heartbeat: task 6 batch 3 call 29 | 0.08/1.4 h, 0.48/30 GB, 2552 calls
- `2026-10-26 08:09:16` **E4 (driver)** probe `e4_grpo_goal_mirror_s1` linear ok: rc=0 11.8 min, plumbing.passed=True
- `2026-10-26 08:09:37` **E4 (driver)** probe `e4_grpo_goal_mirror_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:12:58` done: 4513 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_freeze_s1/calls.parquet (0.85 GB)
- `2026-10-26 08:12:58` **DONE** run_id=`e4_grpo_goal_freeze_s1` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_freeze_s1/calls.parquet`
- `2026-10-26 08:13:08` **E4 (driver)** collect `e4_grpo_goal_freeze_s1` ok: rc=0 9.3 min (expected 42), calls=4513, peak RSS 61.23 GB, peak VRAM 25467 MiB
- `2026-10-26 08:13:12` **E4 (driver)** collect `e4_sft_goal_mirror_s2` (s2, sft, main02_mirror.yaml); expected 25 min, stop at 50 min; MUJOCO_GL=egl; GPU used so far 1.65/12.0 h
- `2026-10-26 08:13:14` **START** run_id=`e4_sft_goal_mirror_s2` config=`/root/RLinf/experiments/efference_probe/configs/main02_mirror.yaml` hash=`2a024cc91c08` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_goal_mirror_s2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.8350 notes=E4 sft goal_mirror_s2 (Base-Lora) env.num_envs=10 env.seed=2 model.do_sample=true model.temperature=1.6`
- `2026-10-26 08:13:17` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 08:13:42` **E4 (driver)** probe `e4_grpo_goal_freeze_s1` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 08:13:48` building LiberoEnv (10 parallel envs)
- `2026-10-26 08:14:03` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.3018893623352051, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 08:16:57` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261026T081657Z` (not deleted)
- `2026-10-26 08:16:57` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:18:14` heartbeat: task 8 batch 0 call 6 | 0.08/0.8 h, 0.50/30 GB, 2626 calls
- `2026-10-26 08:19:54` **E4 (driver)** probe `e4_sft_goal_freeze_s1` linear ok: rc=0 14.3 min, plumbing.passed=True
- `2026-10-26 08:20:37` done: 4198 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_mirror_s2/calls.parquet (0.79 GB)
- `2026-10-26 08:20:37` **DONE** run_id=`e4_sft_goal_mirror_s2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_mirror_s2/calls.parquet`
- `2026-10-26 08:20:48` **E4 (driver)** collect `e4_sft_goal_mirror_s2` ok: rc=0 7.5 min (expected 25), calls=4198, peak RSS 60.85 GB, peak VRAM 25467 MiB
- `2026-10-26 08:20:53` **E4 (driver)** collect `e4_grpo_goal_mirror_s2` (s2, grpo, main02_mirror.yaml); expected 42 min, stop at 84 min; MUJOCO_GL=egl; GPU used so far 1.77/12.0 h
- `2026-10-26 08:20:55` **START** run_id=`e4_grpo_goal_mirror_s2` config=`/root/RLinf/experiments/efference_probe/configs/main02_mirror.yaml` hash=`2de798dac738` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_goal_mirror_s2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.3917 notes=E4 grpo goal_mirror_s2 (GRPO) env.num_envs=10 env.seed=2 env.init_states_per_task=50 model.do_sample=true model.temperature=1.6`
- `2026-10-26 08:20:57` **E4 (driver)** probe `e4_grpo_goal_freeze_s1` linear ok: rc=0 7.3 min, plumbing.passed=True
- `2026-10-26 08:20:57` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 08:21:27` building LiberoEnv (10 parallel envs)
- `2026-10-26 08:21:42` **E4 (driver)** probe `e4_sft_goal_mirror_s2` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 08:21:43` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.21125593185424807, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 08:21:43` **E4 (driver)** probe `e4_sft_goal_freeze_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:21:47` **E4 (driver)** probe `e4_grpo_goal_freeze_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:26:04` heartbeat: task 6 batch 0 call 0 | 0.09/1.4 h, 0.47/30 GB, 2487 calls
- `2026-10-26 08:27:52` **E4 (driver)** probe `e4_sft_goal_mirror_s2` linear ok: rc=0 6.2 min, plumbing.passed=True
- `2026-10-26 08:29:47` **E4 (driver)** probe `e4_sft_goal_mirror_s2` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:31:05` heartbeat: task 9 batch 2 call 19 | 0.17/1.4 h, 0.91/30 GB, 4844 calls
- `2026-10-26 08:32:14` done: 5626 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_mirror_s2/calls.parquet (1.06 GB)
- `2026-10-26 08:32:14` **DONE** run_id=`e4_grpo_goal_mirror_s2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_mirror_s2/calls.parquet`
- `2026-10-26 08:32:26` **E4 (driver)** collect `e4_grpo_goal_mirror_s2` ok: rc=0 11.4 min (expected 42), calls=5626, peak RSS 61.64 GB, peak VRAM 25467 MiB
- `2026-10-26 08:32:31` **E4 (driver)** collect `e4_sft_goal_freeze_s2` (s2, sft, main02_freeze.yaml); expected 25 min, stop at 51 min; MUJOCO_GL=egl; GPU used so far 1.97/12.0 h
- `2026-10-26 08:32:32` **START** run_id=`e4_sft_goal_freeze_s2` config=`/root/RLinf/experiments/efference_probe/configs/main02_freeze.yaml` hash=`555435dd030b` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_goal_freeze_s2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.8433 notes=E4 sft goal_freeze_s2 (Base-Lora) env.num_envs=10 env.seed=2 model.do_sample=true model.temperature=1.6`
- `2026-10-26 08:32:35` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 08:33:06` building LiberoEnv (10 parallel envs)
- `2026-10-26 08:33:21` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.2317694664001465, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 08:33:54` **E4 (driver)** probe `e4_grpo_goal_mirror_s2` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 08:37:33` heartbeat: task 8 batch 2 call 0 | 0.08/0.8 h, 0.48/30 GB, 2532 calls
- `2026-10-26 08:39:08` done: 3612 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_freeze_s2/calls.parquet (0.68 GB)
- `2026-10-26 08:39:08` **DONE** run_id=`e4_sft_goal_freeze_s2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_goal_freeze_s2/calls.parquet`
- `2026-10-26 08:39:17` **E4 (driver)** collect `e4_sft_goal_freeze_s2` ok: rc=0 6.7 min (expected 25), calls=3612, peak RSS 61.01 GB, peak VRAM 25467 MiB
- `2026-10-26 08:39:21` **E4 (driver)** collect `e4_grpo_goal_freeze_s2` (s2, grpo, main02_freeze.yaml); expected 42 min, stop at 84 min; MUJOCO_GL=egl; GPU used so far 2.08/12.0 h
- `2026-10-26 08:39:23` **START** run_id=`e4_grpo_goal_freeze_s2` config=`/root/RLinf/experiments/efference_probe/configs/main02_freeze.yaml` hash=`a54e1c737ded` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_goal_freeze_s2 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.4056 notes=E4 grpo goal_freeze_s2 (GRPO) env.num_envs=10 env.seed=2 env.init_states_per_task=50 model.do_sample=true model.temperature=1.6`
- `2026-10-26 08:39:25` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 08:39:55` building LiberoEnv (10 parallel envs)
- `2026-10-26 08:39:58` **E4 (driver)** probe `e4_sft_goal_freeze_s2` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 08:40:09` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.25908185958862306, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 08:44:10` **E4 (driver)** probe `e4_grpo_goal_mirror_s2` linear ok: rc=0 10.3 min, plumbing.passed=True
- `2026-10-26 08:44:23` heartbeat: task 6 batch 2 call 21 | 0.08/1.4 h, 0.45/30 GB, 2387 calls
- `2026-10-26 08:45:58` **E4 (driver)** probe `e4_sft_goal_freeze_s2` linear ok: rc=0 6.0 min, plumbing.passed=True
- `2026-10-26 08:46:00` **E4 (driver)** probe `e4_grpo_goal_mirror_s2` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:48:06` **E4 (driver)** probe `e4_sft_goal_freeze_s2` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 08:48:22` done: 4482 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_freeze_s2/calls.parquet (0.84 GB)
- `2026-10-26 08:48:22` **DONE** run_id=`e4_grpo_goal_freeze_s2` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_goal_freeze_s2/calls.parquet`
- `2026-10-26 08:48:33` **E4 (driver)** collect `e4_grpo_goal_freeze_s2` ok: rc=0 9.1 min (expected 42), calls=4482, peak RSS 61.51 GB, peak VRAM 25467 MiB
- `2026-10-26 08:48:37` **E4 (driver)** collect `e4_sft_spatial_swap_s0` (s2, sft, suite_spatial.yaml); expected 33 min, stop at 66 min; MUJOCO_GL=egl; GPU used so far 2.23/12.0 h
- `2026-10-26 08:48:39` **START** run_id=`e4_sft_spatial_swap_s0` config=`/root/RLinf/experiments/efference_probe/configs/suite_spatial.yaml` hash=`0950e3ff3dbb` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_spatial_swap_s0 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.1044 notes=E4 sft spatial_swap_s0 (Base-Lora) env.num_envs=10 env.seed=0 model.do_sample=true model.temperature=1.6`
- `2026-10-26 08:48:41` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 08:49:11` building LiberoEnv (10 parallel envs)
- `2026-10-26 08:49:26` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.2686763381958008, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 08:50:06` **E4 (driver)** probe `e4_grpo_goal_freeze_s2` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 08:53:39` heartbeat: task 4 batch 1 call 33 | 0.08/1.1 h, 0.40/30 GB, 2132 calls
- `2026-10-26 08:58:35` done: 4794 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_spatial_swap_s0/calls.parquet (0.90 GB)
- `2026-10-26 08:58:35` **DONE** run_id=`e4_sft_spatial_swap_s0` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_spatial_swap_s0/calls.parquet`
- `2026-10-26 08:58:44` **E4 (driver)** collect `e4_sft_spatial_swap_s0` ok: rc=0 10.0 min (expected 33), calls=4794, peak RSS 65.05 GB, peak VRAM 26288 MiB
- `2026-10-26 08:58:49` **E4 (driver)** collect `e4_grpo_spatial_swap_s0` (s2, grpo, suite_spatial.yaml); expected 66 min, stop at 133 min; MUJOCO_GL=egl; GPU used so far 2.39/12.0 h
- `2026-10-26 08:58:50` **START** run_id=`e4_grpo_spatial_swap_s0` config=`/root/RLinf/experiments/efference_probe/configs/suite_spatial.yaml` hash=`c3d06b40cd04` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_spatial_swap_s0 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=2.2089 notes=E4 grpo spatial_swap_s0 (GRPO) env.num_envs=10 env.seed=0 env.init_states_per_task=40 model.do_sample=true model.temperature=1.6`
- `2026-10-26 08:58:53` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 08:59:23` building LiberoEnv (10 parallel envs)
- `2026-10-26 08:59:40` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.24766237258911133, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 09:00:25` **E4 (driver)** probe `e4_sft_spatial_swap_s0` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 09:02:41` **E4 (driver)** probe `e4_sft_goal_swap_s1` mlp ok: rc=0 112.0 min, plumbing.passed=True
- `2026-10-26 09:03:43` **E4 (driver)** probe `e4_grpo_goal_freeze_s2` linear ok: rc=0 13.6 min, plumbing.passed=True
- `2026-10-26 09:03:53` heartbeat: task 4 batch 1 call 0 | 0.08/2.2 h, 0.43/30 GB, 2262 calls
- `2026-10-26 09:04:11` **E4 (driver)** probe `e4_grpo_goal_freeze_s2` mlp (--jobs 24); stop at 4.7 h
- `2026-10-26 09:06:10` **E4 (driver)** probe `e4_sft_spatial_swap_s0` linear ok: rc=0 5.8 min, plumbing.passed=True
- `2026-10-26 09:08:54` heartbeat: task 8 batch 1 call 16 | 0.17/2.2 h, 0.90/30 GB, 4765 calls
- `2026-10-26 09:10:43` done: 5807 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_spatial_swap_s0/calls.parquet (1.09 GB)
- `2026-10-26 09:10:43` **DONE** run_id=`e4_grpo_spatial_swap_s0` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_spatial_swap_s0/calls.parquet`
- `2026-10-26 09:10:57` **E4 (driver)** collect `e4_grpo_spatial_swap_s0` ok: rc=0 12.0 min (expected 66), calls=5807, peak RSS 65.77 GB, peak VRAM 26288 MiB
- `2026-10-26 09:11:02` **E4 (driver)** collect `e4_sft_object_swap_s0` (s2, sft, suite_object.yaml); expected 25 min, stop at 50 min; MUJOCO_GL=egl; GPU used so far 2.59/12.0 h
- `2026-10-26 09:11:04` **START** run_id=`e4_sft_object_swap_s0` config=`/root/RLinf/experiments/efference_probe/configs/suite_object.yaml` hash=`e9c9ec4aebd5` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_object_swap_s0 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=0.8367 notes=E4 sft object_swap_s0 (Base-Lora) env.num_envs=10 env.seed=0 model.do_sample=true model.temperature=1.6`
- `2026-10-26 09:11:07` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 09:11:38` building LiberoEnv (10 parallel envs)
- `2026-10-26 09:11:57` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.26981407165527344, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 09:12:21` **E4 (driver)** probe `e4_grpo_spatial_swap_s0` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 09:16:04` heartbeat: task 4 batch 0 call 36 | 0.08/0.8 h, 0.45/30 GB, 2367 calls
- `2026-10-26 09:20:52` **E4 (driver)** probe `e4_grpo_spatial_swap_s0` linear ok: rc=0 8.5 min, plumbing.passed=True
- `2026-10-26 09:21:05` heartbeat: task 9 batch 1 call 29 | 0.17/0.8 h, 1.01/30 GB, 5359 calls
- `2026-10-26 09:21:18` done: 5713 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_object_swap_s0/calls.parquet (1.07 GB)
- `2026-10-26 09:21:18` **DONE** run_id=`e4_sft_object_swap_s0` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_sft_object_swap_s0/calls.parquet`
- `2026-10-26 09:21:30` **E4 (driver)** collect `e4_sft_object_swap_s0` ok: rc=0 10.3 min (expected 25), calls=5713, peak RSS 65.41 GB, peak VRAM 25667 MiB
- `2026-10-26 09:21:35` **E4 (driver)** collect `e4_grpo_object_swap_s0` (s2, grpo, suite_object.yaml); expected 50 min, stop at 100 min; MUJOCO_GL=egl; GPU used so far 2.77/12.0 h
- `2026-10-26 09:21:36` **START** run_id=`e4_grpo_object_swap_s0` config=`/root/RLinf/experiments/efference_probe/configs/suite_object.yaml` hash=`265688ad535e` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_object_swap_s0 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.6733 notes=E4 grpo object_swap_s0 (GRPO) env.num_envs=10 env.seed=0 env.init_states_per_task=40 model.do_sample=true model.temperature=1.6`
- `2026-10-26 09:21:39` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo
- `2026-10-26 09:22:08` building LiberoEnv (10 parallel envs)
- `2026-10-26 09:22:27` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.24602540969848635, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 09:22:27` **E4 (driver)** probe `e4_sft_object_swap_s0` linear (--jobs 24); stop at 0.4 h
- `2026-10-26 09:26:37` heartbeat: task 3 batch 2 call 0 | 0.08/1.7 h, 0.51/30 GB, 2690 calls
- `2026-10-26 09:29:30` **E4 (driver)** probe `e4_sft_goal_swap_s2` mlp ok: rc=0 102.1 min, plumbing.passed=True
- `2026-10-26 09:31:37` heartbeat: task 8 batch 1 call 9 | 0.17/1.7 h, 1.12/30 GB, 5934 calls
- `2026-10-26 09:32:12` **E4 (driver)** probe `e4_sft_object_swap_s0` linear ok: rc=0 9.8 min, plumbing.passed=True
- `2026-10-26 09:33:12` done: 7104 calls -> /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_object_swap_s0/calls.parquet (1.33 GB)
- `2026-10-26 09:33:12` **DONE** run_id=`e4_grpo_object_swap_s0` -> `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data/e4_grpo_object_swap_s0/calls.parquet`
- `2026-10-26 09:33:25` **E4 (driver)** collect `e4_grpo_object_swap_s0` ok: rc=0 11.7 min (expected 50), calls=7104, peak RSS 65.53 GB, peak VRAM 25667 MiB
- `2026-10-26 09:33:30` **E4 (driver)** collect `e4_sft_long_swap_s0` (s2, sft, suite_long.yaml); expected 31 min, stop at 62 min; MUJOCO_GL=egl; GPU used so far 2.96/12.0 h
- `2026-10-26 09:33:32` **START** run_id=`e4_sft_long_swap_s0` config=`/root/RLinf/experiments/efference_probe/configs/suite_long.yaml` hash=`8c9f8f269779` overrides=`model.model_path=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_long_swap_s0 out_root=/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/data budget.max_gpu_hours=1.0267 notes=E4 sft long_swap_s0 (Base-Lora) env.num_envs=10 env.seed=0 model.do_sample=true model.temperature=1.6`
- `2026-10-26 09:33:35` loading checkpoint /__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft
- `2026-10-26 09:34:06` building LiberoEnv (10 parallel envs)
- `2026-10-26 04:12:07` **E4 (driver)** checkpoints registered: sft=`RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora@a4ea91b38976` at `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/sft` (sha256 verified), grpo=`RLinf/RLinf-OpenVLAOFT-LIBERO-130@473954241c81` at `/__modal/volumes/vo-LkP78BoclJFMtmrzGoDw4Y/ckpts/grpo` (sha256 verified)
- `2026-10-26 11:30` **INTERIM REPORT (agent, E4 Modal session).** Operator answer verified in the parent transcript ("先出中期报告 (Recommended)": no Modal spend, Volume reads only). Pulled 421dac0. Data fetched with `modal volume get` / `ls` only (no `modal run`, no containers) into `data/` (git-ignored): every S2 and pilot run's manifest.json, run_config.yaml, calls.parquet, and the `analysis/` and `analysis_mlp/` dirs that hold both probe_results.csv and summary.md (S2 linear: 15; MLP: 2; incomplete `analysis_mlp/` dirs on 10 goal runs were skipped), plus `.e4_status/`, `e4_ledger/`, `e4_gates/`, `e4_derived/`, `e4_logbook/`, `e4_logs/`, `e4_state.json`. No hidden/ or frames/; nothing on the Volume modified. `merge-logbook` appended 290 lines (prefix byte-identical, 361 NULs intact). The `init` part had been written under a nested path (`<stamp>_cpu_init_sft_/vol/ckpts/sft.md`) that the `*.md` glob misses; it was merged from a flat local copy (+1 line, so it sits out of time order), and `modal_e4._part_label` now flattens labels (test `test_modal_logbook_part_label_is_a_flat_file_name`; offline suite **96 passed**). `e4_report.py --data-root data` wrote `reports/e4_report.md`. The generated tables were not edited; an INTERIM banner, "Readout in plain words" and "Surprises / could not verify" were added. Generated verdicts: R4/R5 MLP: Base-Lora at floor, GRPO inconclusive (numbers missing), contrast (e). Linear: Base-Lora at floor, GRPO inconclusive (above floor but explained by C_dstates in a seed), contrast (e). R6 [TO VERIFY] (MLP missing). R7 not migrated (0/2, 0/2). R8 does not strengthen on every arm. R9 Base-Lora 0.471 / 0.484, GRPO 0.670 / 0.700. GPU 2.96 h, probe CPU 6.09 h (completed rounds). The 13 MLP driver starts for 12 rounds include `e4_grpo_goal_swap_s1` twice (08:01:29, 08:16:57; the first `analysis_mlp` was renamed `.partial_20261026T081657Z`, kept); cause [TO VERIFY].
- `2026-10-26 11:22` **E4 (report)** Corrected the interim report's R2 reason labels; verdicts unchanged. GRPO goal-swap s2 linear is not beyond observables because C_cmd = 0.552 is outside [0.45, 0.55], while its rungs (0.586/0.591/0.590) clear C_dstates 0.532 + 0.04; the report had blamed C_dstates. `e4_report.py` now names the failing R2 condition (C_dstates, C_cmd, or both). R4 for GRPO linear stays inconclusive because s1 is at floor regardless.
- `2026-10-26 12:10` **E4 (scheduler fix, before any resume)** Why the interim wave lost money: the orchestrator spawned every offered probe round at once (no cap, no budget check), so 10 MLP rounds (8 physical cores, 32 GiB each) were running when the workspace spend limit hit, and all 10 left nothing (MLP rounds cannot checkpoint; probe code is off limits). Reconstructed from the ledger and logbook start times: ~11.9 container-hours, ~$7.5-10 at list prices [TO VERIFY against the Modal dashboard]. GRPO goal-swap s1/s2 MLP (the primary R4/R5 rounds) had run ~78/96 min of a ~100-112 min round. Fix (`modal_e4.py`, `e4_wave.py todo`): rounds start in a ranked queue (goal-swap MLP, linear, mirror/freeze MLP, collections) under caps (1 collection, 2 linear, 2 MLP); before each start the orchestrator reserves the round's worst case (watchdog limit + 300 s, priced at the container's resource limits) and holds that round and everything after it if `--budget-usd` would be passed; every start/end is journalled under `data/e4_spend/<session>/`, and a restarted orchestrator re-attaches to its rounds. Probe containers now state their limits (12 cores, 48 GiB; requests unchanged). Probe code, transforms, gates and the plan are unchanged. Measured inputs for the resume estimate (Volume reads only): SFT MLP rounds 6125/6720 s, peak tree RSS 27.9 GB; GRPO linear rounds were slower only in the best-cell permutation step, which MLP rounds do not run.
- `2026-10-26 12:23` **E4 (infra: new Modal account)** The operator moved to a new Modal workspace ($30 credit). The E4 Volume (`efference-e4`: 17.3 GB of run data, 17.2 GB of it hidden states, plus 30.2 GB of checkpoints) stays in the old workspace. `e4_copy_volume.py` copies it client-side (no container, no compute), resumably, and verifies every file by size; nothing in the old Volume is changed. After the copy, `init` must run again on the new workspace, because `e4_state.json` records the checkpoints under the old Volume's mount path (`/__modal/volumes/vo-LkP78.../ckpts/*`); `init` also re-checks their sha256. Budget plan for the new workspace: `wave --budget-usd 24`; the gate keeps the journalled spend (limit rates, an over-estimate) at or below $24, and the copy, image build and `init` sit outside it. Simulated with measured durations: billed ~$17.5-18, and all 13 rounds start in the central case. No Modal compute was run for this entry.
- `2026-10-26 18:20` **E4 (resume on new Modal workspace, stopped)** Copy: `e4_copy_volume.py` moved 9151 files / 47.50 GB to workspace pascal-lzw ("copy verified"; the first pass hit the agent's 2 h task limit with 33/34 batches done, the resume copied the last). Image build ~276 s; `init` rc=0 re-registered both checkpoints (sha256 verified) under `/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/ckpts/*`; lock and sizing unchanged; `todo` queued the expected 13 items. Wave session `20261026T144109Z`, `--budget-usd 24`, started 14:41 UTC. Completed (rc=0, ok markers): linear `e4_grpo_object_swap_s0` (875.6 s); collect `e4_sft_long_swap_s0` as `e4_sft_long_swap_s0_r2` (old partial dir kept; 996.1 s, 8206 calls); collect `e4_grpo_long_swap_s0` (1361.6 s, 11709 calls); linear `e4_sft_long_swap_s0_r2` (1506.6 s of its 1600 s watchdog). Failed: linear `e4_grpo_long_swap_s0` watchdog-killed at 1606.6 s with 199/200 permutation-null iterations done (6.53 s/iter; SFT long 5.20 s/iter); a full round needs ~1800-2000 s [estimate from the two logs]. GRPO goal-swap MLP s1/s2: first containers (14:41) died ~15:00-15:02 and Modal re-ran both inputs (s1 15:02:31, s2 15:03:02; earlier half-written dirs renamed `analysis_mlp.partial_*`, kept). App logs show "Container terminated due to preemption. Your Function will be restarted with the same input" for the s2 retry after 16:21; no third s2 container appeared, its call graph read FAILURE while `get(timeout=0)` kept raising TimeoutError, so the orchestrator counted it as running. Cause of the ~15:00 deaths [TO VERIFY] (preemption likely; those containers' logs were out of the app-log window). The s1 retry ran ~3x slower than the SFT goal-swap s1 MLP round (sweep ~186 min vs 62 min) and could not finish before its 4.7 h stop. Live-container diagnostics: `compute: workers=12 cpu_count=12 usable_cpus=12` (old workspace MLP rounds logged 24), cfs quota 12 CPUs, memory limit 48 GiB (44.9 GB available), 3 of 12 loky workers busy. Stopped by operator decision: `modal app stop` at 18:12 UTC, zero containers after. Journal (limit rates, open rounds timed to 18:12): $10.24, of which $6.66 is the two swap MLP rounds (s2 includes time with no container); the Modal dashboard bill is [TO VERIFY]. Not completed: linear `e4_grpo_long_swap_s0`; all 10 MLP rounds (grpo goal_swap s1/s2, sft/grpo goal_mirror/freeze s1/s2). No retry; next step is the operator's.
- `2026-10-27 03:01` **E4 OPERATIONAL AMENDMENT 2 (before any local round).** (1) The ten remaining goal-suite MLP rounds and a re-run of `linear:e4_grpo_long_swap_s0` move off Modal to the operator's Windows machine (`e4_local.py`, `LOCAL_RUN.md`). Why: Modal preempted MLP rounds four times across both workspaces, and an MLP round cannot checkpoint, so each preemption restarted it from zero. At 24 CPUs a round costs ~$2-2.5 on Modal, more than the new workspace's remaining credit allows for ten. The new workspace's dashboard read $7.11 when the wave was stopped (operator). (2) The rounds run through the unchanged driver and probe code, with the Modal image's analysis versions pinned (numpy 1.26.4, scikit-learn 1.9.1, pandas 2.3.3, joblib 1.6.0, pyarrow 19.0.1; scipy 1.17.1, since the image's scipy was not recorded [TO VERIFY]). Bit-exact agreement with Modal-run rounds is not expected. (3) Expected times: libero_long linear rounds 1600 s instead of 800 s (`e4_plan.yaml` `expected_s_by_suite`). The SFT long round took 1506.6 s; GRPO's was stopped at the 1600 s watchdog with 199/200 permutations done, and needs ~1800-2000 s. With fewer than 24 workers, probe expected times (and so the watchdog) scale by 24/jobs, as the lock's speed factor does for collections; each status file records the factor. (4) The killed GRPO long linear round's failed marker is superseded by the re-run's; its half-written `analysis/` is kept as `analysis.partial_<UTC>`. (5) Root causes of the second-wave stop, recorded for the report: the 12-core probe limit added in b4963f0 halved the CPUs (one Modal core is one container CPU; now 24); both GRPO swap MLP rounds were preempted at ~15:00 UTC; a preempted call that Modal marks FAILURE still reads as running to the orchestrator (documented gap). Probe code, transforms, gates and the readout are unchanged.
- `2026-10-27 03:07` **E4 (operator decision): the two primary MLP rounds on Modal, non-preemptible.** At the operator's request, `mlp:e4_grpo_goal_swap_s1` then `s2` (R4/R5) run on the new workspace as `cpu_stage_nonpreemptible` (`nonpreemptible=True`, 3x the CPU and memory list price; 24 CPUs, 32 GiB request), one `modal run --detach ...::probe_np` each. The other eight MLP rounds and the long linear re-run stay local (Amendment 2). Credit: $30, $7.11 used per the dashboard. Estimate per round, with 24 cores busy and 32 GiB for the measured 1.7-1.9 h of an SFT goal-swap MLP round: ~$7-7.5, so ~$15 for both. Stop rule, fixed before launch: SFT goal-swap s1 reached its P1 sweep 18.8 min after loading samples. s2 starts once s1 reaches P1 within 28 min (1.5x). s1 is stopped by hand if P1 has not begun by 38 min (2x; projected >3.7 h, >$15), and s2 then does not start. Same driver, gates and watchdog.
- `2026-10-27 03:38` **E4 (non-preemptible s1 stopped, no result).** `mlp:e4_grpo_goal_swap_s1` ran as `cpu_stage_nonpreemptible` (app ap-ZHiRXVa4PTP1ZsBdKd2cdX, container ta-01M5XN96T7QMF1NZEFA26G5ERR) from 03:08:29 UTC; samples (952) loaded at 03:08:33. The first P0 result came at 03:35:16 (+26.7 min, against +11.5 min for the SFT goal-swap s1 round on the old workspace), and P1 projected to ~03:52, past the 03:46:50 stop line. A cgroup read inside the container showed 462 CPU-s used in 20 s (23.1 of 24 cores) and all 24 loky workers at ~96%, so the container had its CPUs and each was ~2.3x slower than the old workspace's. Projected round ~4.2 h at ~$4/h (3x price), ~$17, and the two rounds would not fit the remaining credit. Stopped by hand at 03:38:09, 9 min before the pre-registered 03:47 check, because the check's outcome was already fixed; s2 not launched. No marker was written. Cost ~$2.0 by list price x3 [TO VERIFY on the dashboard]. All ten MLP rounds and the long linear re-run now go to the operator's machine (`e4_local.py`).
- `2026-10-27 03:59` **Repo housekeeping (operator request: keep only paper-related branches).** `main` fast-forwarded 8758879 -> 028f2b9, which takes in wave 3 (`claude/vigilant-dijkstra-q2tpp2`: GRU mismatch E1/E1b, RL arms, world-model comparator, VRAM probe). The E4 branch `claude/lucid-archimedes-6f31ty` deliberately stays off main and keeps its own probe code until the local E4 rounds finish; merging it later conflicts only in README.md and logbook.md, where both sides appended. To delete (the session's git policy refused the deletes, HTTP 403; the operator deletes them on GitHub): `claude/rlinf-code-scripts-pi41dg` @ 5d0fa54b (fully in main), `claude/vigilant-dijkstra-q2tpp2` @ 028f2b90 (fully in main), `claude/rlinf-issue-1422-dot0do` @ e1f35c2e (RLinf issue #1422 dense/hierarchical step rewards plus an upstream merge; no paper files). Naming clash to settle before writing: wave 3 also has an "E4" (world-model comparator, `configs/e4_wm_*.yaml`).
- `2026-10-27 08:50` **E4 (operator decision): the two primary MLP rounds on Modal, preemptible, capped.** The operator's machine runs ~7x slower than Modal, so `mlp:e4_grpo_goal_swap_s1` and `s2` run on the new workspace after all, in parallel. They use `probe_stage` (preemptible, 24 CPUs, 32 GiB request), not the non-preemptible stage, which was ~2.3x slower per core at 3x the price. Each round may start at most twice (`MAX_PROBE_ATTEMPTS`, counted on the Volume, so a Modal restart after preemption counts); a third start refuses. Estimate from the SFT goal-swap s1 MLP timeline (24 CPUs: ~23 cores busy for the 62 min sweep, ~6 for the 50 min tail; 32 GiB): ~$1.8-2.2 per attempt, worst case two attempts each, ~$8 for both. Credit left ~$20.9 ($30 - $7.11 at the last dashboard read - ~$2.0 for the stopped non-preemptible try) [TO VERIFY]. Pace rule, fixed before launch: the SFT reference had its first P0 result 11.5 min after loading samples. A round whose first P0 result has not come by +23 min (2x) is on a slow host and is stopped by hand. The remaining eight MLP rounds and the long linear re-run stay with `e4_local.py`.
- `2026-10-27 09:18` **E4 (preemptible primary rounds, first pace check).** s1 (`ap-eleFyJ4jHmx44IdLPgBsvh`): attempt 1 was preempted ~15 s after start (Modal: "Container terminated due to preemption"); attempt 2 loaded samples at 08:51:53 and had no P0 result by 09:17 (+25 min, over 2.2x the SFT reference with fewer samples, 952 vs 968). Stopped at 09:17:32 under the pace rule, ~26 min of run time, no marker. s2 (`ap-3DgMHc5ZnnIFh6bt5olimc`, attempt 1): first P0 result at 09:13:23 (+21.5 min, 1.87x, inside the 2x line) and continues, projected ~3.5 h. s1 is relaunched once (`probe_round --launch 2`, its own two-attempt count), under the same pace rule. If launch 2 is also slow, it is stopped and the operator decides.
- `2026-10-27 09:45` **E4 (s1 launch 2 stopped; decision to the operator).** `probe_round --launch 2` for `mlp:e4_grpo_goal_swap_s1` (app ap-blaYQVf1TDGIMCdmI8hQLE) loaded samples at 09:19:02 and had no P0 result by 09:44 (+25 min, over the 2x line). Stopped at 09:44:47, ~26 min, no marker; not relaunched (two slow hosts in a row). Pace so far on this workspace: s2 1.87x the SFT reference to its first P0 and 1.93x to P1; s1 >2.2x on both 24-CPU preemptible hosts and 2.3x on the non-preemptible one, but 16 min to its first P0 on a 12-CPU host in the second wave. Host speed varies, and 24 visible CPUs may not mean 24 physical cores [TO VERIFY]. s2 continues, projected to finish ~12:30 UTC. Modal spend since the $7.11 read, at list price: ~$2.0 (non-preemptible try) + ~$0.6 + ~$0.6 (the two s1 stops) + s2 [TO VERIFY on the dashboard].
- `2026-10-27 09:47` **E4 OPERATIONAL AMENDMENT 3 (operator decision, before launch).** On this workspace preemptible 24-CPU hosts ran ~1.9-2.3x the SFT reference pace (s2 1.87x and 1.93x; s1 >2.2x twice), so the old-workspace 2x pace line rejects ordinary hosts. For `mlp:e4_grpo_goal_swap_s1` launch 3 only: driver `--speed-factor 1.5` (MLP watchdog 7.0 h instead of 4.7 h; the factor is recorded in the status file) and Modal function timeout 8 h; pace line 3x (first P0 result by +34.5 min, else stopped by hand). It runs in parallel with s2, at most 2 attempts. Estimate ~4-4.5 h, ~$5-6; worst case with one preemption restart ~$12. No probe, gate or readout change.
- `2026-10-27 13:22` **E4 (preemptible primary rounds: s2 done, s1 no result).** `mlp:e4_grpo_goal_swap_s2` (app ap-3DgMHc5ZnnIFh6bt5olimc, attempt 1) finished ok at 12:21:41 UTC: rc=0, 12590.9 s, plumbing passed, peak tree RSS 27.73 GB, ~2.0x the SFT reference pace. Read through `e4_report.ladder_verdict`: floor 0.527; P1/P3/P4 0.591/0.586/0.578 (above floor by +0.064/+0.059/+0.051); P2r 0.705 (informative); C_dstates 0.568, so no rung clears C_dstates + 0.04 = 0.608 and it is explained by outcome perception (C_dstates); C_cmd 0.513 (in band). Under R4 the GRPO MLP verdict is therefore inconclusive whatever s1 shows. `mlp:e4_grpo_goal_swap_s1` launch 3 (speed factor 1.5): attempt 1 reached the best-hidden P4 at 12:43:45, ~70% of the round, and was preempted before 13:06; attempt 2 started 13:06:15 and was preempted within ~15 min; the restart refused under `MAX_PROBE_ATTEMPTS` and the app was stopped at 13:21. No s1 marker. Preemptions of long CPU rounds seen on this workspace: 6. Modal spend estimate at list price since the $7.11 read: ~$2.0 + $0.6 + $0.6 + ~$3.9 (s2) + ~$3.6 (s1 launch 3) = ~$10.7, so ~$17.8 used and ~$12 left [TO VERIFY on the dashboard].
- `2026-10-26 14:39:41` **E4 (driver)** checkpoints registered: sft=`RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora@a4ea91b38976` at `/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/ckpts/sft` (sha256 verified), grpo=`RLinf/RLinf-OpenVLAOFT-LIBERO-130@473954241c81` at `/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/ckpts/grpo` (sha256 verified)
- `2026-10-26 14:41:18` **E4 (orchestrator)** session `20261026T144109Z` up: budget $24.00, caps {'collect': 1, 'linear': 2, 'mlp': 2}, re-attached []
- `2026-10-26 14:41:33` **E4 (orchestrator)** started `mlp:e4_grpo_goal_swap_s1`: reserves $4.51, committed $4.51 of $24.00
- `2026-10-26 14:41:37` **E4 (orchestrator)** started `mlp:e4_grpo_goal_swap_s2`: reserves $4.51, committed $9.02 of $24.00
- `2026-10-26 14:41:39` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261026T144139Z` (not deleted)
- `2026-10-26 14:41:39` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 12); stop at 4.7 h
- `2026-10-26 14:41:43` **E4 (orchestrator)** started `linear:e4_grpo_object_swap_s0`: reserves $0.50, committed $9.52 of $24.00
- `2026-10-26 14:41:49` **E4 (orchestrator)** started `collect:e4_sft_long_swap_s0`: reserves $4.47, committed $13.99 of $24.00
- `2026-10-26 14:41:55` **E4 (driver)** `e4_sft_long_swap_s0`: earlier attempt(s) ['e4_sft_long_swap_s0'] never completed; collecting as `e4_sft_long_swap_s0_r2` (interrupted collection = new id; nothing overwritten or deleted)
- `2026-10-26 14:41:57` **E4 (driver)** `e4_grpo_goal_swap_s2`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261026T144157Z` (not deleted)
- `2026-10-26 14:41:57` **E4 (driver)** probe `e4_grpo_goal_swap_s2` mlp (--jobs 12); stop at 4.7 h
- `2026-10-26 14:42:13` **E4 (driver)** probe `e4_grpo_object_swap_s0` linear (--jobs 12); stop at 0.4 h
- `2026-10-26 14:42:14` **E4 (driver)** collect `e4_sft_long_swap_s0_r2` (s2, sft, suite_long.yaml); expected 31 min, stop at 62 min; MUJOCO_GL=egl; GPU used so far 2.96/12.0 h
- `2026-10-26 14:42:16` **START** run_id=`e4_sft_long_swap_s0_r2` config=`/root/RLinf/experiments/efference_probe/configs/suite_long.yaml` hash=`ef005b582195` overrides=`model.model_path=/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/ckpts/sft model.unnorm_key=libero_130_no_noops_trajall run_id=e4_sft_long_swap_s0_r2 out_root=/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/data budget.max_gpu_hours=1.0267 notes=E4 sft long_swap_s0 (Base-Lora) env.num_envs=10 env.seed=0 model.do_sample=true model.temperature=1.6`
- `2026-10-26 14:42:20` loading checkpoint /__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/ckpts/sft
- `2026-10-26 14:43:03` building LiberoEnv (10 parallel envs)
- `2026-10-26 14:43:43` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.25886457443237304, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 14:47:17` heartbeat: task 2 batch 0 call 62 | 0.08/1.0 h, 0.31/30 GB, 1638 calls
- `2026-10-26 14:52:17` heartbeat: task 5 batch 1 call 26 | 0.17/1.0 h, 0.86/30 GB, 4560 calls
- `2026-10-26 14:56:49` **E4 (driver)** probe `e4_grpo_object_swap_s0` linear ok: rc=0 14.6 min, plumbing.passed=True
- `2026-10-26 14:57:22` heartbeat: task 9 batch 0 call 0 | 0.25/1.0 h, 1.37/30 GB, 7291 calls
- `2026-10-26 14:58:15` **E4 (orchestrator)** `linear:e4_grpo_object_swap_s0` returned: rc=0
- `2026-10-26 14:58:47` done: 8206 calls -> /__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/data/e4_sft_long_swap_s0_r2/calls.parquet (1.54 GB)
- `2026-10-26 14:58:47` **DONE** run_id=`e4_sft_long_swap_s0_r2` -> `/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/data/e4_sft_long_swap_s0_r2/calls.parquet`
- `2026-10-26 14:59:02` **E4 (driver)** collect `e4_sft_long_swap_s0_r2` ok: rc=0 16.6 min (expected 31), calls=8206, peak RSS 73.27 GB, peak VRAM 29401 MiB
- `2026-10-26 14:59:02` **E4 (driver)** `e4_sft_long_swap_s0_r2`: low-competence task flags [4, 8] (kept in every analysis; see e4_derived/<run>/competence.csv)
- `2026-10-26 15:00:24` **E4 (orchestrator)** `collect:e4_sft_long_swap_s0` returned: rc=0
- `2026-10-26 15:00:34` **E4 (orchestrator)** started `linear:e4_sft_long_swap_s0_r2`: reserves $0.50, committed $10.81 of $24.00
- `2026-10-26 15:00:39` **E4 (orchestrator)** started `collect:e4_grpo_long_swap_s0`: reserves $7.84, committed $18.64 of $24.00
- `2026-10-26 15:00:41` **E4 (driver)** collect `e4_grpo_long_swap_s0` (s2, grpo, suite_long.yaml); expected 62 min, stop at 123 min; MUJOCO_GL=egl; GPU used so far 3.24/12.0 h
- `2026-10-26 15:00:43` **START** run_id=`e4_grpo_long_swap_s0` config=`/root/RLinf/experiments/efference_probe/configs/suite_long.yaml` hash=`45b021aff90d` overrides=`model.model_path=/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/ckpts/grpo model.unnorm_key=libero_130_no_noops_trajall run_id=e4_grpo_long_swap_s0 out_root=/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/data budget.max_gpu_hours=2.0533 notes=E4 grpo long_swap_s0 (GRPO) env.num_envs=10 env.seed=0 env.init_states_per_task=30 model.do_sample=true model.temperature=1.6`
- `2026-10-26 15:00:47` loading checkpoint /__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/ckpts/grpo
- `2026-10-26 15:01:41` **E4 (driver)** probe `e4_sft_long_swap_s0_r2` linear (--jobs 12); stop at 0.4 h
- `2026-10-26 15:01:42` building LiberoEnv (10 parallel envs)
- `2026-10-26 15:02:12` indexing check passed: {'max_abs_logprob_diff': 4.76837158203125e-07, 'tolerance': 0.2100075721740723, 'logits_dtype': 'torch.float32', 'seq_len': 440, 'readout_start': 383, 'passed': True, 'n_hidden_layers': 33, 'n_act': 56}
- `2026-10-26 15:02:31` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261026T150231Z` (not deleted)
- `2026-10-26 15:02:31` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 12); stop at 4.7 h
- `2026-10-26 15:03:02` **E4 (driver)** `e4_grpo_goal_swap_s2`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261026T150302Z` (not deleted)
- `2026-10-26 15:03:02` **E4 (driver)** probe `e4_grpo_goal_swap_s2` mlp (--jobs 12); stop at 4.7 h
- `2026-10-26 15:05:46` heartbeat: task 1 batch 2 call 0 | 0.08/2.1 h, 0.37/30 GB, 1945 calls
- `2026-10-26 15:11:01` heartbeat: task 4 batch 0 call 0 | 0.17/2.1 h, 0.84/30 GB, 4464 calls
- `2026-10-26 15:16:01` heartbeat: task 6 batch 1 call 54 | 0.25/2.1 h, 1.26/30 GB, 6725 calls
- `2026-10-26 15:21:10` heartbeat: task 9 batch 0 call 0 | 0.34/2.1 h, 1.94/30 GB, 10343 calls
- `2026-10-26 15:23:18` done: 11709 calls -> /__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/data/e4_grpo_long_swap_s0/calls.parquet (2.20 GB)
- `2026-10-26 15:23:18` **DONE** run_id=`e4_grpo_long_swap_s0` -> `/__modal/volumes/vo-L53b0bYpCazhX3gz7qhGQO/data/e4_grpo_long_swap_s0/calls.parquet`
- `2026-10-26 15:23:42` **E4 (driver)** collect `e4_grpo_long_swap_s0` ok: rc=0 22.7 min (expected 62), calls=11709, peak RSS 72.41 GB, peak VRAM 29402 MiB
- `2026-10-26 15:25:21` **E4 (orchestrator)** `collect:e4_grpo_long_swap_s0` returned: rc=0
- `2026-10-26 15:25:30` **E4 (orchestrator)** started `linear:e4_grpo_long_swap_s0`: reserves $0.50, committed $12.67 of $24.00
- `2026-10-26 15:25:33` **E4 (driver)** probe `e4_grpo_long_swap_s0` linear (--jobs 12); stop at 0.4 h
- `2026-10-26 15:26:48` **E4 (driver)** probe `e4_sft_long_swap_s0_r2` linear ok: rc=0 25.1 min, plumbing.passed=True
- `2026-10-26 15:27:36` **E4 (orchestrator)** `linear:e4_sft_long_swap_s0_r2` returned: rc=0
- `2026-10-26 15:52:14` **E4 (driver)** watchdog: `run_probes.py` passed 0.44 h (pid 31); stopping it
- `2026-10-26 15:52:14` **E4 (driver)** watchdog: sent SIGINT to process group 31
- `2026-10-26 15:52:19` **E4 (driver)** probe `e4_grpo_long_swap_s0` linear FAILED: rc=-2 26.8 min, plumbing.passed=True
- `2026-10-26 15:54:34` **E4 (orchestrator)** `linear:e4_grpo_long_swap_s0` returned: rc=1
- `2026-10-27 03:08:29` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261027T030829Z` (not deleted)
- `2026-10-27 03:08:29` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-27 08:51:34` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261027T085134Z` (not deleted)
- `2026-10-27 08:51:34` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-27 08:51:48` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-27 08:51:50` **E4 (driver)** `e4_grpo_goal_swap_s2`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261027T085150Z` (not deleted)
- `2026-10-27 08:51:50` **E4 (driver)** probe `e4_grpo_goal_swap_s2` mlp (--jobs 24); stop at 4.7 h
- `2026-10-27 09:18:57` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261027T091857Z` (not deleted)
- `2026-10-27 09:18:57` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24); stop at 4.7 h
- `2026-10-27 09:48:03` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261027T094803Z` (not deleted)
- `2026-10-27 09:48:03` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24, speed factor 1.5); stop at 7.0 h
- `2026-10-27 12:21:41` **E4 (driver)** probe `e4_grpo_goal_swap_s2` mlp ok: rc=0 209.8 min, plumbing.passed=True
- `2026-10-27 13:06:15` **E4 (driver)** `e4_grpo_goal_swap_s1`: half-written `analysis_mlp` renamed to `analysis_mlp.partial_20261027T130615Z` (not deleted)
- `2026-10-27 13:06:15` **E4 (driver)** probe `e4_grpo_goal_swap_s1` mlp (--jobs 24, speed factor 1.5); stop at 7.0 h
- `2026-10-28 03:24` **X0 PRE-REGISTRATION (mechanism analyses on existing E4 data; fixed before any X0 number is computed).** Plan source: `reports/next_steps_fable.md` §B-C. Exploratory. It never changes an R1-R12 verdict. Code is a new `x0_mechanism.py`; probe code, samples and folds are unchanged. Samples are each run's own `analysis*/probe_samples.csv`. Folds are `probes._outer_splits` (seed 0, 5 folds), with the same C / alpha grids and inner selection as `run_probes.py`. Compute is this session's container plus free Volume reads. Runs: the E4 goal swap/mirror/freeze runs (both checkpoints, both seeds), spatial/object swap (both checkpoints) and `e4_sft_long_swap_s0_r2`.
  **A1 efference increment (no fits).** Δeff = P3 - P1 and Δint = P4 - P1 at best cells, same family, for every completed E4 analysis plus the committed wave-2 rows. Also the matched-cell max over cells of P3 - P1. Prediction (M1): |Δeff| < 0.04 in every run.
  **A2 outcome readout.** `probes.ridge_readout` (same alphas, grouped by episode) on up to 4000 non-post-success calls per run (seed 0), at all 27 cells. Targets, R^2 as the uniform average over the 8 state dims, per-dim also reported:
    - (i) s(c+1) = `states_after(c)` from h(c+1);
    - (ii) Δs(c) from h(c+1);
    - (iii) Δs(c) from h(c) ⊕ h(c+1), at the best (i) cell.
  Swap runs only. Predictions:
    - memoryless premise: best (i) >= 0.5 and best (ii) < 0.3;
    - M2 (outcome not metric): best (i) < 0.3;
    - M1-compatible: (iii) >= 0.5.
  **A3 decoded-ingredients oracle.** Per outer fold, ridge decoders are fit on training-episode calls only:
    - â(c) from h(c) at the run's E1 best cell;
    - variant A: s_before(c) from h(c) and s_after(c) from h(c+1) at the A2 (i) best cell, with Δŝ composed as `datasets._delta_states` does (rotation composed);
    - variant B: Δŝ(c) regressed directly from h(c+1).

  Training-sample decodes come from inner 5-fold cross-fitting within the training episodes; test-sample decodes come from decoders fit on all training episodes. P2r's comparison features are computed from (â, Δŝ); the implementation must reproduce the `mismatch` block exactly on true inputs. They are scored by the linear logistic with P2r's C grid and selection. Rule on the 4 goal-swap runs:
    - "ingredients present, comparator missing (M1)" if variant A >= linear floor + 0.04 on >= 3 of 4;
    - "ingredients insufficient" if < floor + 0.04 on >= 3 of 4 while P2r(true) >= floor + 0.10 on them;
    - else "mixed".
  Suites and variant B are reported alongside.
  **A4 outcome visibility (M3).** Per run, the AUROC (hijack vs self) of a non-learned score: the Mahalanobis distance of a sample's Δs(c) from its task's SELF-sample Δs distribution (Ledoit-Wolf covariance). Prediction (M3): GRPO > Base-Lora on both goal-swap seeds. Spearman with the linear P1 margin across all E4 runs is descriptive.
  **A5 locus (no fits).** P1(cell) - floor by layer x pool, GRPO vs Base-Lora goal swap. Prediction (M3): GRPO's top-3 P1 cells are at layer >= 24 or pool `ctx_last`.
  **A6 novelty (M4).** Per matched pair (`pair_id`):
    - Δnov = novelty(hijack) - novelty(self), where novelty is the Mahalanobis distance of h(c+1) at L0/`ctx_mean` in PCA-50 space fit on the run's self samples;
    - Δent = next-call `entropy_mean` difference (E3's quantity);
    - Spearman ρ per run, and pooled over all goal runs with an episode-cluster bootstrap 95% CI (B = 2000, seed 0).

  Prediction (M4): pooled ρ > 0 with the CI excluding 0. A CI covering 0 means the novelty mechanism lacks per-pair support.
  **A7 paired bootstrap.** Linear family: P1 and P3 at their best cells (per `probe_results.csv`) plus C_dstates are refit with out-of-fold predictions kept. Each must reproduce the CSV's fold balanced accuracies within 1e-6, else it is flagged and excluded. An episode-cluster bootstrap (B = 2000, seed 0) is computed on the pooled-OOF balanced-accuracy differences P1 - C_dstates and P3 - P1. MLP is run for GRPO goal-swap s2 only. Fixed reading:
    - the CI is conditional on the selected cell and so optimistic for P1 - C_dstates;
    - a CI covering 0 means "not separable from observables";
    - a CI excluding 0 is not by itself a positive.

  **Order:** A1, A5, A2, A3, A7, A4, A6. Results are logged as they land.
- `2026-10-28 03:31` **X0 A1 result (efference increment, no fits; `x0/a1_efference_increment.csv`).** 30 rows: 20 E4 run×family analyses plus 10 wave-2 rows. Δeff = P3 - P1 at best cells has |Δeff| < 0.04 in all 30. The largest is +0.031 (`main02_freeze` mlp, wave-2); the largest in E4 is +0.025 (`e4_sft_goal_freeze_s2` linear). **Prediction (M1) holds:** the commanded chunk adds nothing to h(t+1) in any run, checkpoint or suite. The matched-cell max of P3 - P1 (a max over 27 cells, so biased upward) peaks at +0.062 (`e4_sft_goal_freeze_s2` linear) and +0.052 (`e4_grpo_goal_swap_s2` linear). Descriptive, not pre-registered: Δint = P4 - P1 is near 0 on swap for every checkpoint (max +0.016), but large on freeze for GRPO (+0.090 s1, +0.084 s2, linear), against Base-Lora (+0.031, +0.008) and goal-traj1 (+0.023 linear, +0.079 mlp). Under freeze the arm does not move, and h(t) ⊕ h(t+1) can register "scene unchanged". That is a perceptual comparison across two observations, not an efference read. C_dstates already carries the same fact (R2 reading unchanged).
- `2026-10-28 03:31` **X0 A5 result (locus, no fits; `x0/a5_locus.csv`).** Goal swap, top-3 P1 cells by margin over the same-family floor. GRPO: 0 of 6 linear cells and 0 of 3 mlp cells (s2 only; the s1 mlp round has no result) are at layer >= 24 or `ctx_last`. All 9 are at L4-L20 with `ctx_mean`/`act_mean`. Base-Lora: 6 of 6 linear and 5 of 6 mlp cells are at layer >= 24 or `ctx_last`. **Prediction (M3: GRPO's excursion sits late) is falsified.** The direction is reversed: GRPO's above-floor P1 signal sits in early-to-mid layers, while Base-Lora's (at floor) sits late. Reading, not a test: an early-layer, context-pooled excursion is what a change in the visual input would produce. That points toward outcome perception or novelty (M4, tested by A6), not toward late re-planning.
- `2026-10-28 03:35` **X0 implementation notes (fixed before any A2-A7 number; A1/A5 are already logged).** Code is `x0_mechanism.py`, tests are `tests/test_x0.py` (7 pass).
  - **Mismatch block:** `mismatch_from_arrays` equals the `mismatch` block exactly (max |diff| 0.0 on `e4_grpo_goal_swap_s2`, `e4_sft_goal_swap_s1` and `e4_sft_long_swap_s0_r2`, and on the synthetic run).
  - **A2 calls:** (episode, c) with call c+1 logged, post-success calls excluded, then 4000 drawn with `DataFrame.sample(random_state=0)`.
  - **A3 decoders:** each is `probes._SharedRidge` (exactly StandardScaler + Ridge), with alpha picked as `ridge_readout` does (first split of a grouped 3-fold, same alpha grid).
    - Decoder rows are every A2-eligible call (no cap) of every episode outside the held-out set, so episodes without probe samples are included.
    - Variant A uses one state decoder, h(k) → state at observation k, fit on the stacked rows (h(c), s_before(c)) and (h(c+1), s_after(c)), so Δŝ is a difference of a single readout.
    - Inner cross-fitting is `GroupKFold(5)` over the training samples' episodes.
  - **A4 / A6 references:** both are cross-fitted on `_outer_splits`. Each held-out fold is scored against a reference fit on that fold's training self samples, so no sample is scored against a distribution that contains it.
    - A4 falls back to all training self samples when a task has fewer than 10 (= dims + 2) in the fold.
    - A6 uses min(50, n_ref - 1) PCA components.
  - **A6 pooling:** pools all 12 E4 goal runs (swap/mirror/freeze × checkpoint × seed).
  - **A7 reproduction:** a refit must also reproduce the CSV's selected C per fold.
- `2026-10-28 07:06` **X0 A2 result, goal swap (outcome readout; `x0/a2_<run>.csv`).** 4000 non-post-success calls per run, 27 cells, ridge as in E1. Best-cell uniform R² over the 8 state dims; no best-cell alpha at the grid edge.

  | run | (i) s(c+1) ← h(c+1) | (ii) Δs(c) ← h(c+1) | (iii) Δs(c) ← h(c) ⊕ h(c+1) |
  |---|---|---|---|
  | GRPO s1 | 0.972 (L12/ctx_mean) | 0.834 (L24/act_mean) | 0.890 (L12/ctx_mean) |
  | GRPO s2 | 0.965 (L8/ctx_mean) | 0.850 (L20/act_mean) | 0.899 (L8/ctx_mean) |
  | Base-Lora s1 | 0.881 (L4/ctx_mean) | 0.584 (L20/act_mean) | 0.706 (L4/ctx_mean) |
  | Base-Lora s2 | 0.875 (L16/ctx_mean) | 0.591 (L20/act_mean) | 0.692 (L16/ctx_mean) |

  **Pre-registered readings:**
  - The memoryless premise as stated fails in all 4 runs. (i) >= 0.5 holds, but (ii) >= 0.58 everywhere, against the predicted < 0.3: a single h(c+1) already predicts the last chunk's motion.
  - M2 (outcome not metric) is rejected in all 4: the state is linearly readable at R² >= 0.875.
  - The runs are M1-compatible: (iii) >= 0.69 in all 4.

  **Descriptive, not pre-registered:** GRPO's h(c+1) carries the state and especially Δs better than Base-Lora's (0.83-0.85 vs 0.58-0.59 on (ii)). The L0 `ctx_last`/`act_mean` cells are at R² ≈ 0, as expected for input embeddings of fixed tokens. Reading: a memoryless policy can get Δs(c) from o(c+1) only through regularities, i.e. where the arm is now, in this scene and task, predicts how it just moved. That regularity is stronger for the more stereotyped GRPO policy. A2 does not separate this from an encoding of motion. An exploratory control, Δs(c) from the true s(c+1) and task, is proposed below and is not part of the pre-registration. Consequence for the ladder: Δs (the C_dstates input) is largely present in h(c+1) itself, so P1 > C_dstates needs no efference read.
- `2026-10-28 07:17` **X0 A2b PRE-REGISTRATION (exploratory follow-up to A2; fixed before any A2b number).** Not part of the original X0 plan. Question: does h(c+1) know the last chunk's motion Δs(c) beyond what the current state predicts?

  **Setup.** Goal-swap runs, the A2 calls (same 4000, seed 0), out-of-fold predictions with `GroupKFold(5)` by episode, ridge alpha picked as in `ridge_readout`. Predictors:
    - **H:** h(c+1) at the run's A2 (ii) best cell;
    - **S1 (per-task linear in the state):** ridge on [s(c+1), task one-hot, s(c+1) ⊗ task one-hot];
    - **S2 (nonparametric in the state):** k-NN regression (k = 10, uniform weights) on standardised s(c+1), fit separately per task on training folds;
    - **H+S1:** H and S1 features concatenated.

  **Readout.** Uniform R² over the 8 Δs dims on all calls, on SELF calls and on HIJACK calls. Each subset's R² uses that subset's own variance.

  **Pre-set readings:**
    - "regularity explains A2 (ii)" if max(S1, S2) on all calls >= H - 0.10;
    - "h(c+1) carries motion beyond the state" if H on HIJACK calls >= max(S1, S2) on HIJACK calls + 0.10;
    - otherwise "unresolved".

  The reading is made per run and summarised by checkpoint.
- `2026-10-28 07:26` **X0 A2b result (`x0/a2b_<run>.json`).** Out-of-fold uniform R² for Δs(c), reported as all / SELF / HIJACK calls. Each run is listed as H, S2 (per-task k-NN on the true s(c+1)), S1:
  - GRPO s1: H 0.834 / 0.851 / 0.666; S2 0.826 / 0.841 / 0.673; S1 0.512.
  - GRPO s2: H 0.849 / 0.863 / 0.718; S2 0.840 / 0.853 / 0.717; S1 0.298.
  - Base-Lora s1: H 0.585 / 0.618 / 0.299; S2 0.577 / 0.603 / 0.339; S1 0.321.
  - Base-Lora s2: H 0.594 / 0.627 / 0.336; S2 0.580 / 0.608 / 0.351; S1 0.301.

  H+S1 is within 0.005 of H everywhere. **Reading:** "regularity explains A2 (ii)" in all 4. "h(c+1) carries motion beyond the state" in none: on HIJACK calls H is at or below S2 in every run. The A2 failure of the memoryless premise is a state effect. h(c+1) knows the last motion only through where the arm now is, as a memoryless policy should.
- `2026-10-28 07:26` **X0 A8 PRE-REGISTRATION (exploratory; fixed before any A8 number).** Follows from A2b. The ladder has no control for the *absolute* state after the hijacked chunk: C_dstates sees only Δs, while h(c+1) carries s(c+1) at R² >= 0.875 (A2).
  **Features,** per sample, with s(c+1) = `states_after(prev_call)`:
    - C_state: [s(c+1), task one-hot, s(c+1) ⊗ task one-hot];
    - C_state+d: C_state ⊕ Δs(prev_call).

  **Fitting.** Both are scored by `probes.run_probe` on the run's own `probe_samples.csv`, with the ladder's folds, grid and selection. The linear family is fit on all 17 E4 runs; the MLP family on GRPO goal-swap s2 (the only goal-swap run with an MLP ladder besides Base-Lora s1/s2, which are also fit).
  **Non-learned score.** Mean k-NN distance (k = 10) of standardised s(c+1) to the training-fold SELF samples of the same task, cross-fitted on `_outer_splits`, reported as AUROC.
  **Readings,** goal swap, same family, P1 at its best cell:
    - "state accounts for P1" if max(C_state, C_state+d) >= P1 - 0.02;
    - "P1 beyond state" if P1 >= max(C_state, C_state+d) + 0.04;
    - otherwise "unresolved".

  P1's best cell is a max over 27 cells, so this comparison favours P1.
- `2026-10-28 07:41` **X0 A3 result (decoded-ingredients oracle, goal swap; `x0/a3_<run>.json`).** Command cell L28/`act_mean` (E1 best) in all 4. Linear balanced accuracy of P2r's comparison features built from decoded inputs:

  | run | variant A | variant B | linear floor | floor + 0.04 | P2r (true inputs) |
  |---|---|---|---|---|---|
  | Base-Lora s1 | **0.602** | 0.567 | 0.525 | 0.565 | 0.798 |
  | GRPO s1 | 0.514 | 0.505 | 0.528 | 0.568 | 0.672 |
  | Base-Lora s2 | **0.606** | 0.515 | 0.526 | 0.566 | 0.827 |
  | GRPO s2 | **0.588** | 0.574 | 0.541 | 0.581 | 0.681 |

  Variant A is >= floor + 0.04 on 3 of 4. **Pre-registered reading: "ingredients present, comparator missing (M1)".**
  - The command and the outcome are both linearly decodable, and a comparison built from the decodes detects the hijack.
  - The model's own representation does not: P1 and P3 sit at or near floor. Base-Lora s1 has P3 0.541 against A 0.602, Base-Lora s2 has P3 0.558 against A 0.606.
  - Descriptive: decoding noise costs a lot relative to true inputs (P2r 0.67-0.83). The one miss is GRPO s1.
  - Suites are pending (they need suite A2).
- `2026-10-28 07:41` **X0 A7 result (paired episode-cluster bootstrap, B = 2000; `x0/a7_linear_selected.csv`, `x0/a7_passes/`).**
  - **Reproduction:** all 17 linear runs and the GRPO s2 MLP reproduce the CSV fold accuracies and the selected C exactly, each in one of three BLAS settings (implementation note in the code, commit dce6ba8). The first pass, at this container's default 4-thread AVX-512, reproduced only 4 of 9 swap runs.
  - **P3 - P1:** the CI covers 0 in 16 of 17 linear runs and in the MLP run. The exception is Base-Lora freeze s2, +0.025 [+0.006, +0.046].
  - **P1 - C_dstates, goal swap linear:** GRPO s1 +0.103 [+0.056, +0.149]; GRPO s2 +0.055 [+0.008, +0.102]; Base-Lora s1 +0.026 [-0.017, +0.069]; Base-Lora s2 +0.018 [-0.026, +0.059]. GRPO s2 MLP (primary family) is +0.023 [-0.019, +0.065].
  - **Elsewhere:** the CI excludes 0 upward on GRPO mirror s1 (+0.074), GRPO spatial (+0.058) and Base-Lora spatial (+0.069). It excludes 0 downward (C_dstates > P1) on Base-Lora freeze s1/s2 and GRPO freeze s2.

  **Fixed reading:** the efference increment is not separable from 0 anywhere it matters. GRPO's linear P1 excess over C_dstates excludes 0 on both goal-swap seeds, but this is conditional on the selected cell, so optimistic, and not by itself a positive. It covers 0 in the MLP family.
- `2026-10-28 07:41` **X0 A4 result (outcome visibility; `x0/a4_visibility.csv`).** Mahalanobis(Δs | task SELF) AUROC on goal swap: GRPO s1 0.520 vs Base-Lora s1 0.528; GRPO s2 0.563 vs Base-Lora s2 0.539. **Prediction (M3: GRPO > Base-Lora on both seeds) fails on s1.**
  - Descriptive: on mirror the AUROC is 0.59-0.67. On freeze it is 0.24-0.34: a frozen chunk is *less* far from the SELF distribution, because a near-zero Δs is unusually small rather than unusual in direction.
  - Spearman of AUROC with the linear P1 margin over the 17 E4 runs is ρ = 0.311 (p = 0.22). Post hoc, |AUROC - 0.5| gives ρ = 0.517 (p = 0.034); that figure is not pre-registered.
- `2026-10-28 07:41` **X0 A6 result (novelty vs reaction; `x0/a6_<run>.csv`, `x0/a6_pooled.json`).**
  - **Per run:** Spearman ρ(Δnovelty at L0/`ctx_mean`, Δentropy) ranges from -0.041 to +0.111 over the 12 goal runs. Only GRPO freeze s1 (+0.111 [+0.003, +0.211]) excludes 0.
  - **Pooled:** 5475 pairs and 1892 episode clusters give ρ = +0.026 [-0.003, +0.055]. **Prediction (M4: pooled ρ > 0 with the CI excluding 0) fails:** the novelty mechanism lacks per-pair support.
  - **Descriptive:** mean Δnovelty is positive on mirror and freeze (+1.1 to +3.1; hijacked observations are more novel at the input layer) but near 0 on swap (-1.39 to +0.46).
- `2026-10-28 07:41` **X0 A8 result (absolute-state control; `x0/a8_<run>_<family>.json`).** Goal swap, max(C_state, C_state+d) against P1 at its best cell:
  - GRPO s1 linear: 0.488 vs P1 0.563, **P1 beyond state**.
  - GRPO s2 linear: 0.565 vs 0.586, unresolved. GRPO s2 MLP: 0.580 vs 0.591, **state accounts for P1**.
  - Base-Lora s1 linear: 0.522 vs 0.543, unresolved. Base-Lora s1 MLP: 0.584 vs 0.536, state accounts.
  - Base-Lora s2 linear: 0.495 vs 0.555, P1 beyond state. Base-Lora s2 MLP: 0.522 vs 0.556, unresolved.

  The k-NN state-distance AUROC is 0.50-0.56 in every run. **Reading: mixed.** The robot's 8-dim end state does not account for GRPO s1's P1 excursion. Whatever h(c+1) uses beyond Δs there is not the proprioceptive state; the scene, such as object positions, is the observable candidate left. That is not tested here.
- `2026-10-28 10:01` **X0 A2 result, suites (reported alongside).** Best-cell R², listed as (i) s(c+1) / (ii) Δs from h(c+1) / (iii) Δs from h(c) ⊕ h(c+1):
  - GRPO spatial: 0.946 / 0.780 / 0.878;
  - Base-Lora spatial: 0.921 / 0.557 / 0.755;
  - GRPO object: 0.960 / 0.761 / 0.864;
  - Base-Lora object: 0.916 / 0.557 / 0.765;
  - Base-Lora long (r2): 0.811 / 0.395 / 0.579.

  (ii)'s best cell is L20/`act_mean` in every suite run. The goal-swap readings carry over unchanged:
  - the premise as stated fails, with (ii) > 0.3 everywhere;
  - M2 is rejected;
  - the runs are M1-compatible;
  - GRPO > Base-Lora on every column.
- `2026-10-28 10:39` **X0 A3 result, suites (reported alongside).** Variant A / variant B / floor + 0.04 / P2r (true inputs):
  - Base-Lora spatial: **0.627** / 0.551 / 0.557 / 0.820;
  - GRPO spatial: **0.583** / 0.548 / 0.555 / 0.647;
  - Base-Lora object: **0.680** / 0.593 / 0.549 / 0.880;
  - GRPO object: **0.577** / 0.559 / 0.545 / 0.625;
  - Base-Lora long (r2): **0.608** / 0.563 / 0.555 / 0.844.

  All 5 are >= floor + 0.04, so 8 of 9 swap runs are counting goal swap. The M1 reading extends to every suite.

  Descriptive, oracle A minus the best linear rung max(P1, P3, P4):
  - Base-Lora: +0.046 (goal s1), +0.048 (goal s2), +0.054 (spatial), +0.129 (object), +0.060 (long);
  - GRPO: -0.050 (goal s1), -0.003 (goal s2), +0.012 (spatial), +0.010 (object).

  Base-Lora's activations hold a comparison its own representation does not express. GRPO's representation already reads out about as much as the decoded comparison does, with less to gain from the comparison (P2r 0.63-0.68 against Base-Lora's 0.80-0.88).
- `2026-10-28 15:03` **X0 FOLLOW-UP PRE-REGISTRATION F1-F7 (fixed before any F number is computed).** Source: `reports/next_steps_fable_x0.md` §5, Tier 0. Exploratory: no R1-R12 or X0 verdict changes.
  **Scope.** Code is a new `x0_followup.py`; it imports the X0 and probe code and changes neither. Compute is this container only, with no Modal; data are the local copies already used for X0.
  **Conventions.** Outer folds are `probes._outer_splits` (seed 0, 5) on each run's `analysis/probe_samples.csv`, with C grid and selection as the ladder (`_select_c`, `_make_classifier`). Ridge alphas are picked as in `ridge_readout`. "Swap runs" are the 9 E4 swap runs (Base-Lora goal s1/s2, spatial, object, long r2; GRPO goal s1/s2, spatial, object). "Goal swap" is the 4 goal-swap runs.
  **BLAS.** Every refit that is compared with a CSV number runs under that run's reproducing BLAS setting from `x0/a7_linear_selected.csv`.

  **F1, BLAS resolution (no fits).** From `x0/a7_passes/*/`:
  - per run and rung, the max |fold balanced accuracy - CSV| over passes;
  - per run, the spread of the pooled P1 - C_dstates and P3 - P1 point estimates across passes.

  Descriptive; one Methods sentence.

  **F6, C_phase and the margin sign test (no fits).** C_phase balanced accuracy for every E4 run and family, from `probe_results.csv`. One-sided binomial sign test that GRPO's linear P1 margin over floor exceeds Base-Lora's across the 4 matched cells (goal s1, goal s2, spatial, object). Descriptive.

  **F3, oracle decomposition.** A3's pipeline unchanged (decoders, folds, inner cross-fitting, classifier), with P2r's comparison features built from three input pairs:
  - (true a, Δŝ_A), with variant A's composed state decode;
  - (â, true Δs);
  - (true a, true Δs), which must reproduce the CSV's P2r fold scores within 1e-6 (else flagged).

  Swap runs. Readings, on the 5 Base-Lora swap runs, with OK meaning ">= P2r(true) - 0.10":
  - "format supports the comparison": both mixed pairs OK on >= 4 of 5;
  - "outcome decode is the bottleneck": (true a, Δŝ_A) not OK on >= 4 of 5 while (â, true Δs) is OK on >= 4 of 5;
  - "command decode is the bottleneck": the mirror image of the previous reading;
  - otherwise "both decodes lossy".

  GRPO is reported alongside.

  **F2, A2b residual test.** Goal swap, the A2 calls (4000, seed 0), the A2 (ii) best cell. Within each outer `GroupKFold(5)` fold:
  - per-task k-NN (k = 10) predictions Ŝ2 for training calls come from inner `GroupKFold(5)` cross-fitting over training episodes; Ŝ2 for test calls comes from a fit on all training calls;
  - a ridge from h(c+1) to the training residual Δs - Ŝ2 is fit on training calls and applied to test calls.

  The gain is R²(Δs, Ŝ2 + r̂) - R²(Δs, Ŝ2), uniform over the 8 dims, on all calls and on HIJACK calls. Readings:
  - "no motion trace beyond the end-effector state": gain < 0.05 on all and on HIJACK calls in 4 of 4;
  - "motion trace beyond state": gain >= 0.05 on HIJACK calls in >= 3 of 4;
  - otherwise "mixed".

  **F4, state readout in physical units, with matched support.** Goal swap, the A2 calls, s(c+1) from h(c+1) at L0/`ctx_mean` and at the A2 (i) best cell, out-of-fold ridge on `GroupKFold(5)`. Report per-dim RMSE, plus the SELF-call SD per dim.
  - **Common scale:** within each seed pair (GRPO sN vs Base-Lora sN), nRMSE = the mean over dims of RMSE_d / SD_d, where SD_d is pooled over both checkpoints' calls.
  - **Matched support:** a call is in support if its 10-NN distance to the other checkpoint's same-task calls, in pooled-standardised state space, is <= the median 10-NN distance among that other checkpoint's own same-task calls.
  - **Gap:** nRMSE(Base-Lora) - nRMSE(GRPO), on all calls and on in-support calls.

  Readings, per cell:
  - "data effect": gap_matched <= 0.5 × gap_all on both pairs;
  - "representation effect": gap_matched >= 0.8 × gap_all on both;
  - otherwise "mixed".

  **F5, residualised P1.** Goal swap, P1's best linear cell. Within each outer fold, h(c+1) of the samples is regressed on A8's per-task state features [s(c+1), task one-hot, s ⊗ task one-hot]; this is least squares with no label use, fit on training samples. The linear probe is then refit on the residuals with the ladder's selection.
  - "state-orthogonal": P1_resid >= P1 - 0.02;
  - "state-explained": P1_resid <= floor + 0.02;
  - otherwise "partial".

  P1 and the floor are the CSV values.

  **F7, where P1's hits are (descriptive).** Swap runs, P1's best linear cell, out-of-fold predictions refit under the reproducing BLAS setting; they must reproduce the CSV folds, else flagged.
  - **Per task:** n, balanced accuracy, and margin share = n_t (bacc_t - 0.5) / Σ_t n_t (bacc_t - 0.5).
  - **Flag:** raised if the run's competence-flagged tasks (`e4_derived/<run>/competence.csv`) carry > 0.5 of the margin.
  - **On HIJACK samples:** Spearman of correct-vs-wrong with intervention strength ‖a_exec - a_cmd_env‖ of the hijacked chunk, and the hit rate with vs without a gripper-sign disagreement between a_exec and a_cmd_env (action dim 6, any step of the chunk).

  **Order:** F1, F6, F3, F2, F4, F5, F7. Results are logged as they land.
- `2026-10-28 15:08` **F1 result (BLAS resolution; `x0/followup/f1_blas.csv`, `f1_summary.json`).** Over the 17 E4 runs and the three BLAS passes:
  - the largest deviation of a single refit fold from its CSV value is 0.054;
  - the largest spread of a pooled point estimate across passes is 0.011 (P1 - C_dstates and P3 - P1 alike).

  Methods sentence: reproduction resolution is about 0.01 on pooled differences, about a quarter of δ. Base-Lora freeze s2's P3 - P1 CI, the one that excludes 0, sits inside it (+0.025 reproducing vs +0.036 in the other two passes).
- `2026-10-28 15:08` **F6 result (C_phase and the margin sign test; `x0/followup/f6_phase_sign.json`).**
  - C_phase is 0.474-0.561 in every E4 run and family. The highest is GRPO goal swap s2 MLP at 0.561, against an MLP floor of 0.527, so +0.034 < δ.
  - GRPO's linear P1 margin over floor exceeds Base-Lora's on 3 of 4 matched cells (goal s1 +0.035 vs +0.018, goal s2 +0.046 vs +0.029, object +0.062 vs +0.037; spatial +0.050 vs +0.056 the other way). The one-sided sign test gives p = 0.31.

  Descriptive: phase does not carry the P1 margins, and the GRPO > Base-Lora margin pattern is not distinguishable from chance on 4 cells.
- `2026-10-28 15:55` **F3 result (oracle decomposition; `x0/followup/f3_<run>.json`).** In every run so far, the (true a, true Δs) pair reproduces the CSV's P2r fold scores exactly.

  | run | true a + decoded Δŝ | decoded â + true Δs | P2r (true) | P2r - 0.10 |
  |---|---|---|---|---|
  | Base-Lora goal s1 | 0.688 | 0.645 | 0.798 | 0.698 |
  | Base-Lora goal s2 | 0.678 | 0.666 | 0.827 | 0.727 |
  | Base-Lora spatial | 0.711 | 0.656 | 0.820 | 0.720 |
  | Base-Lora object | **0.784** | 0.734 | 0.880 | 0.780 |
  | Base-Lora long | 0.666 | 0.690 | 0.844 | 0.744 |
  | GRPO goal s1 | 0.589 | 0.581 | 0.672 | 0.572 |
  | GRPO goal s2 | 0.607 | 0.611 | 0.681 | 0.581 |
  | GRPO object | 0.578 | 0.578 | 0.625 | 0.525 |
  | GRPO spatial | [pending] | | | |

  **Pre-registered reading: "both decodes lossy".** On Base-Lora the outcome side is OK on 1 of 5 and the command side on 0 of 5.
  - Each decoded side alone costs 0.10-0.18 against true inputs on Base-Lora. The combined A3 oracle sits lower still (0.60-0.68).
  - Neither the command decode nor the outcome decode is in command-comparable form at full fidelity. Fable's reading (b), "the format supports the comparison given a supervised state readout", is not supported as stated. Reading (a), "outcome code not in command-comparable form", partly survives on both sides.
  - Descriptive: on GRPO both mixed pairs are within 0.10 of a lower P2r.
- `2026-10-28 15:55` **F2 result (A2b residual test; `x0/followup/f2_<run>.json`).** Gain of S2 + ridge(h(c+1) → S2 residual) over S2, listed as all calls / HIJACK calls:
  - GRPO s1: +0.008 / +0.003;
  - GRPO s2: +0.007 / +0.005;
  - Base-Lora s1: +0.009 / -0.007;
  - Base-Lora s2: +0.005 / +0.002.

  **Pre-registered reading: "no motion trace beyond the end-effector state" (4 of 4).** The "only through the state" sentence is licensed. h(c+1) adds at most 0.009 R² to a per-task k-NN on the true 8-dim end state.
- `2026-10-28 15:55` **F4 result (state readout in physical units, matched support; `x0/followup/f4_state_rmse.json`).** nRMSE is on the pooled-SD scale; gap = Base-Lora - GRPO, listed as all calls → in-support calls:
  - s1 pair: L0/`ctx_mean` 0.151 → 0.049 (ratio 0.32); best cells (L4 vs L12 `ctx_mean`) 0.157 → 0.060 (0.38);
  - s2 pair: L0 0.165 → 0.044 (0.27); best cells (L16 vs L8) 0.160 → 0.033 (0.21).

  **Pre-registered reading: "data effect" at both cells.** The matched gap is <= 0.5 × the all-calls gap in both pairs, so write "RL narrows the state distribution and the state becomes more readable", not "RL sharpens the state estimate".
  - **Caveat:** only 321-327 of Base-Lora's 4000 calls lie in GRPO's support, against 1706-1863 of GRPO's in Base-Lora's.
  - **Physical RMSE at L0:** translation is 1.5-2.1 cm for Base-Lora against 0.9-1.3 cm for GRPO; the rotation axis-angle dims are 0.14-0.43 against 0.07-0.19.
- `2026-10-28 15:55` **F5 result (residualised P1; `x0/followup/f5_<run>.json`).** Linear P1 at its best cell after removing the per-task linear state fit, inside the folds:
  - GRPO s2 (L8/`ctx_mean`): P1 0.586 → 0.549, floor 0.541, **state-explained**;
  - Base-Lora s2 (L32/`ctx_last`): 0.555 → 0.530, floor 0.526, **state-explained**;
  - GRPO s1 (L12/`ctx_mean`): 0.563 → 0.545, floor 0.528;
  - Base-Lora s1 (L28/`act_mean`): 0.543 → 0.530, floor 0.525.

  GRPO s1 and Base-Lora s1 meet *both* pre-registered conditions (P1_resid >= P1 - 0.02 and <= floor + 0.02), because their P1 is itself within 0.04 of floor. They are reported as **indeterminate**. The first code version labelled them "state-orthogonal" by check order; that was fixed (commit 4bbeeaf) before this entry.

  Reading: the one goal-swap P1 excursion beyond δ, GRPO s2, is carried by a linear function of the end-effector state.
- `2026-10-28 15:55` **F7 result (where P1's hits are; `x0/followup/f7_<run>.json`).** All 9 refits reproduce the CSV folds.
  - **Competence flag:** fires only for Base-Lora long, where flagged tasks 4 and 8 carry 0.58 of the margin. GRPO goal's flagged task 5 carries 0.10 (s1) and 0.27 (s2); the margins are spread over tasks elsewhere.
  - **Intervention strength vs HIJACK hits:** Spearman is significant only on object, Base-Lora ρ = 0.198 (p < 0.001) and GRPO ρ = 0.090 (p = 0.036). Elsewhere |ρ| <= 0.07.
  - **Hit rate with vs without a gripper-sign disagreement:**
    - higher with it on object (GRPO 0.675 vs 0.553; Base-Lora 0.594 vs 0.481) and on Base-Lora long (0.605 vs 0.536);
    - reversed on Base-Lora spatial (0.577 vs 0.670);
    - within 0.05 elsewhere.

  Descriptive: P1's hits do not concentrate on low-competence tasks, except Base-Lora long. A magnitude/gripper dependence appears only on object, where a wrong gripper action is visible in the scene.
- `2026-10-28 16:01` **F3, GRPO spatial (fills the [pending] row above).** True a + decoded Δŝ 0.594; decoded â + true Δs 0.586; P2r (true) 0.647, reproduced exactly; P2r - 0.10 = 0.547. Both mixed pairs are within 0.10, as on the other GRPO runs. The F3 reading ("both decodes lossy", made on Base-Lora) is unchanged. F1-F7 are complete.
- `2026-10-29 02:58` **X1 / X4 PRE-REGISTRATION.** `PREREG_X1_X4.md` (sha256 `2054a71c329dba5df565ea422794b3b5254d0228ff5097d9e1e00329fcdfc6bd`) is committed before any X1/X4 code change, collection or number.
  - **X1:** a second SFT→GRPO pair (goal-traj1 → RLinf-OpenVLAOFT-GRPO-LIBERO-goal). It runs swap, mirror and freeze with new joint, object and vision logging, readouts X1-R1 to R7, and a fixed cell L12/`ctx_mean`. Its primary family is linear.
  - **X4:** a P2r monitor on true command and state, with undo recovery, on mirror for both E4 checkpoints. Off/monitor/oracle conditions are randomised per episode over 1000 episodes per checkpoint, with G fixed at 0.167 (Base-Lora) and 0.219 (GRPO).
  - **Status:** it supersedes the never-run wave-3 E2 (`PREREG_wave3.md` on `main`). No Modal spend is approved by it; the user approves X1 and X4 separately, after the §4 code passes offline tests.
- `2026-10-29 04:02` **X4 monitors frozen (PREREG_X1_X4.md, X4 "The monitor"; `x4_monitor.py`).**
  - GRPO: `x4/monitor_grpo.json` sha256 `cce6891642a1fe8858e7f93ed0c7f1182dc5a6d7e65121b5293e6a06bbe7197b`; fit on `e4_grpo_goal_mirror_s1` (1090 probe samples, C = 1.0); τ = 0.9300 from 3952 held-out SELF calls of `e4_grpo_goal_mirror_s2`; replay FPR 0.020, replay TPR 0.746 on 674 HIJACK calls.
  - Base-Lora: `x4/monitor_sft.json` sha256 `a787afe417cc288ad8f984a6cf55415a6c931238a7add928b8b551f67e2a4152`; fit on `e4_sft_goal_mirror_s1` (968 probe samples, C = 1.0); τ = 1.1316 from 3106 held-out SELF calls of `e4_sft_goal_mirror_s2`; replay FPR 0.020, replay TPR 0.748 on 492 HIJACK calls.

  **Replay gate (TPR >= 0.5): passed for both checkpoints**, so X4 keeps all three conditions.
  - The online features equal the probe `mismatch` block exactly on every training sample; the run aborts otherwise.
  - The exported numbers reproduce the scikit-learn pipeline's decision function within 7e-6.
  - The harness loads only these digests.
- `2026-10-29 04:20` **X1 / X4 code ready (offline only; no Modal spend).** Commits 36f1ca7, 5e1c2b5, 10b2ca1 and 9ee353b implement PREREG_X1_X4.md §4: harness flags, the monitor, the plans, the driver/launcher generalisation and both readouts. 171 offline tests pass, the 125 earlier ones unchanged.
  - **Flags-off loop unchanged:** with both flags off, a fake-env run of the new harness writes rows identical to the pre-change harness (36f1ca7).
  - **Config hashes unchanged:** every shipped config keeps its hash.

  **PREREG §6 [TO VERIFY] items resolved:**
  - **GRPO-goal checkpoint:** the HF card's `base_model` is `Haozhan72/Openvla-oft-SFT-libero-goal-traj1`, and both X1 checkpoints are full 4-shard weights with no LoRA adapter.
  - **Pins:** revisions and shard sha256s are pinned in `x1_plan.yaml` (traj1 `d20e1d44…`, GRPO-goal `222f7df4…`).
  - **Driver:** reused with a plan/data-root generalisation plus plan-driven S2 prerequisites, registration markers, checkpoint order, collection overrides, and X1/X4 gate checks. That is slightly more than "plan path and checkpoint keys", but no gate rule changed for E4 (E4's tests pass unchanged).
  - **Still open:** the object key list (at S0) and the Modal balance.

  **Pricing correction (observation; changes no rule).** PREREG §5 priced GPU time at $1.95/h, the L40S alone. Modal also bills the container's CPU and memory at the greater of request and use:
  - at the GPU stage's 8-core / 48 GiB request that is $1.951 + $0.377 + $0.384 ≈ $2.71/h;
  - E4's pilots used ~62 GB, so ~$2.8/h, matching the $2.81/h the E4 report used.

  Re-priced, the §5 plan is ~$3.5 (X1) + ~$4.9 (X4) ≈ $8.4 [estimate]. That is above the pre-registered $8 combined cap; the GPU-hour caps (1.6 + 2.0 h) correspond to up to ~$10. Before any launch, the operator either amends the cap or the X4 size, or runs X1 first and decides X4 on its actual spend.
- `2026-10-29 04:51` **PREREG_X1_X4.md Amendment 1 (before any X1 or X4 data).** An independent review of the §4 code found 7 driver and 4 readout issues; all are fixed, with tests. The amendment records what changes the registered wording:
  - **Object keys:** `_to_robot0` keys are excluded from the object list, so robot state stays out of C_scene.
  - **Driver reuse:** the generalisation went beyond a plan path and checkpoint keys (plan fields, S0 extension checks, `stage --wave`); E4's gates are unchanged.
  - **X4 smoke:** runs at S2's 10 envs, not 20.
  - **Spend journal for `stage`:** one journal (session `x_waves`) shared by X1 and X4, with `--budget-usd` required; it reserves each call's watchdog worst case at container limit rates.

  It also records code brought in line with the wording:
  - X1 smoke sampled;
  - X4 all GRPO seeds before Base-Lora;
  - undo accuracy on hijacked undos only;
  - X1-R3 needing all six runs;
  - the main02 check implemented;
  - how X1-R7 marks "replicates", plus F4 on the X1 pair, hijack counts and labels.

  No rule, threshold or yardstick changed.

  **Budget observation (changes no rule; the cap is the operator's).** A dry simulation of the plan files through the new gate gives:
  - **Billed spend [estimate]:** X1 ≈ $5.3–5.9 (1.8–1.9 GPU-h with start-ups; collection time alone ≈ 1.4 h, under the 1.6 h cap) and X4 ≈ $4.5–5.0. Assumptions: E4-measured speeds (expected_s / 1.5), 1–2 min start-up per call, GRPO arms at up to 50 states.
  - **Gate:** with worst-case reservations at $3.28/h, the gate needs a cap of ≈ $9 to finish X1 alone and ≈ $14–15 for both. The registered $8 would refuse X1's sized GRPO swap arm.
  - **Versus the credit:** the full plan does not fit the ~$12 credit [TO VERIFY] under this gate. Before launch, the operator chooses the cap and whether X4 is cut (e.g. fewer seeds or GRPO only).

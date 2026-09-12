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

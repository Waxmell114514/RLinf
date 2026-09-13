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

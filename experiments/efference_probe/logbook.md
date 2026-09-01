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

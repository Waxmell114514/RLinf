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

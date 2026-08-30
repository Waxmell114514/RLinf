# Results — efference-copy probing on OpenVLA-OFT / LIBERO

> Template. The agent fills in the run metadata and metric tables; the
> **interpretation sections are for the applicant** (SPEC §8.2). Paste the
> generated `analysis/summary.md` blocks under each stage.

## Provenance

| Field | Value |
|---|---|
| Repo | `RLinf`, branch `claude/rlinf-code-scripts-pi41dg` |
| Commit at spec authoring | `dd92c62857da4c67aa5e7c36f731c0d6a121f6d7` |
| Commit for this run | *(fill: `git rev-parse HEAD`)* |
| Checkpoint | *(fill: absolute path, SFT or GRPO)* |
| Suite | *(fill: libero_goal / object / spatial / 10)* |
| Sampling | *(fill: greedy or temperature; see SPEC §3.7 and record why)* |

Every number below comes from `run_probes.py`. The run card (T1) is at
`data/<run_id>/analysis/run_card.csv`.

---

## S0 — infra gate

- Stock eval success rate (unmodified `run_eval.sh`, 20 envs): *(fill)*
- Smoke run id / config hash: *(fill)*
- Indexing check: *(paste the `indexing check passed: {...}` line)*
- Anomalies: *(fill, or "none")*

## S1 — pilot

| Gate | Expected | Observed |
|---|---|---|
| P0 (shuffled labels) | 0.45–0.55 | |
| P2r (mismatch oracle) | ≥ 0.9, every transform | |
| C_cmd (command alone) | ≈ 0.5 | |
| C_phase (task phase alone) | ≈ 0.5 | |
| Step-index overlap | classes overlap | |
| Signed within-pair phase bias | sign test not significant | |
| Positives kept vs available | note the drop; pairing is not random | |

Greedy-vs-sampled decision (SPEC §3.7): *(fill — greedy success rate over the
pilot vs the sampled-eval reference, and the choice made)*

## S2 — main collection

Run card: *(paste `run_card.csv`)*

### The ladder (F2)

| Probe | Input | Balanced acc | AUROC | n |
|---|---|---|---|---|
| P0 | shuffled labels | | | |
| P1 | `h_{m+1}` | | | |
| P3 | `a_cmd[m] ⊕ h_{m+1}` | | | |
| P4 | `h_m ⊕ h_{m+1}` | | | |
| P2 | `a_cmd[m] ⊕ Δstates` (linear oracle) | | | |
| P2r | mismatch features (mechanical ceiling) | | | |
| C_cmd | `a_cmd[m]` alone (control) | | | |
| C_dstates | `Δstates` alone (control) | | | |
| C_phase | `cur_call`, `cur_call²` (control) | | | |

Selection-aware floor (`extra.selection_floor.max_over_cells`): *(fill — this,
not 0.5, is the floor for P1/P3/P4, which are reported at their best cell)*
Permutation null at the selected cell, if run (`--permutations`): *(fill)*

Best `(layer, pool)`: *(fill — and note this is a maximum over 27 cells; F1
shows the sweep)*

### Depth profile (F1)

*(paste or reference `F1_layer_depth.png`; note where P1/P3/P4 peak)*

## S3 — controls

| Control | Result |
|---|---|
| Cross-task (held-out tasks, F3) | |
| Per-transform: swap / mirror / freeze | |
| Global-pool negatives vs phase-matched | |
| `Δstates` alone | |
| Block scaling `sqrt_dim` vs `none` | |

## E1 — efference readout (F4)

R² of the commanded-chunk readout by layer/pool: *(fill; note where the motor
plan becomes linearly readable)*

## E3 — behaviour after hijack (F5)

| Quantity | After SELF | After HIJACK | Paired difference | Wilcoxon p |
|---|---|---|---|---|
| Action-token entropy (nats) | | | | |
| Chunk logprob | | | | |

Success rate — probe episodes vs clean episodes: *(fill; one number, per SPEC
§3.5, and it stays in a caption, not a headline)*

Undo-alignment (stretch): *(fill)*

## S4 — stretch

*(fill: which one was run, and why)*

---

## Anomalies and caveats

*(agent fills: anything unexpected in collection or analysis — budget stops,
skipped hijacks, failed episodes, warnings)*

---

## Interpretation

> **Applicant writes this section.** The hypothesis ledger below is from SPEC
> §1; fill in which row the data actually lands on, and say plainly where the
> evidence is weak.

| Probe result pattern | Interpretation |
|---|---|
| P1 ≈ chance, P3/P4 ≫ P1, near P2 | representations linearly expose achieved state; efference comparison is *available* but unused across calls |
| P1 ≫ chance | an atypicality/OOD signal; dissect before any agency claim |
| P3 ≈ P1 | hidden state does not expose achieved state in a command-comparable form — strongest "no efference substrate" result |
| P4 > P3 | `h_m` carries motor-plan information beyond the raw action vector (check against E1) |

Two things to keep in view when writing this up:

1. **P2 vs P2r is a result about probe class, not about the model.** A linear
   model cannot form a command–outcome comparison from concatenated blocks, so
   P2 is at chance for the mean-preserving transform (`swap`). It is *not* at
   chance for `mirror`/`freeze` on directed motion — there it separates on the
   marginal state delta alone, which `C_dstates` will show. Either way P2 is
   not a ceiling. Comparing P1/P3/P4 against P2r compares against a ceiling a
   linear probe cannot reach; comparing against P2 is the like-for-like
   comparison. Say which is being used, wherever the ladder is shown — and note
   that `P3 > P2` is not by itself evidence of an efference copy.
2. **Q1's null is the expected outcome.** The setup is memoryless and
   vision-only per call, so anything P1 finds above chance is most likely
   visual atypicality, not agency attribution. That should be stated as the
   default reading and argued against, not the reverse.

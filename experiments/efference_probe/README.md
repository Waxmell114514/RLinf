# Efference-copy probing for OpenVLA-OFT on LIBERO

Does a pretrained VLA carry anything like an *efference copy* — a trace of its
own outgoing command that would let it tell self-caused sensory change from
externally-caused change?

This directory holds the code for the experiment described in
`SPEC_efference_probe_rlinf.md`. It is **inference only**: nothing here trains,
fine-tunes, or writes a checkpoint. It reuses RLinf's LIBERO env wrapper, model
loader and action-space conventions, and **no file under `rlinf/` is modified**.

Everything below has been exercised offline against a synthetic run
(`tests/synthetic.py`); the parts that need a GPU, LIBERO and a checkpoint —
`capture.py` and `harness.py` — have not been executed, because this repo
checkout has no torch. Treat the first smoke run as the real integration test,
and see [Before the first run](#before-the-first-run).

---

## Before the first run

Fill these in (SPEC §9):

- [ ] **Checkpoint path** — absolute path to the OpenVLA-OFT **SFT** checkpoint,
      and which LIBERO suite it matches. Set `model.model_path` and
      `model.unnorm_key` in the config (they must agree: a `libero_goal`
      checkpoint needs `unnorm_key: libero_goal_no_noops` and
      `env.task_suite_name: libero_goal`). Note the GRPO checkpoint too if you
      have one — that is E4.
- [ ] **GPU budget** — `budget.max_gpu_hours` and `budget.max_disk_gb` are hard
      stops; the run finalises cleanly when either is hit and records which.
- [ ] **A known-good `run_eval.sh` line** — S0.1 below starts from it.

Environment (same variables `evaluations/run_eval.sh` exports):

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD:$PYTHONPATH
```

`LIBERO_TYPE` must stay `standard`. LIBERO-Pro/Plus perturbation modes are
explicitly out of scope (SPEC §1).

Extra Python packages beyond what RLinf already installs:
`scikit-learn`, `matplotlib`, `pandas` (SPEC §0.5). The analysis half needs
nothing else — the markdown tables are hand-rolled rather than pulling in
`tabulate`.

---

## Staged run

Each stage stops for human review. `logbook.md` is appended to automatically.

### S0 — infra gate

```bash
# 1. Stock sanity: unmodified eval, reduced env count, for a health reference.
bash evaluations/run_eval.sh libero libero_goal_openvlaoft_eval \
    env.eval.total_num_envs=20

# 2. Smoke test the harness: 1 task, 2 envs, 6 calls, aggressive freeze hijacks.
python experiments/efference_probe/run_collect.py \
    --config experiments/efference_probe/configs/smoke.yaml \
    --set model.model_path=/abs/path/to/ckpt
```

**Acceptance.** The run prints `indexing check passed: {...}` (see
[Hidden-state capture](#hidden-state-capture)); `data/smoke01/calls.parquet`
has a full row set with no nulls; `data/smoke01/hidden/ep00000.npz` holds `h`
with shape `[n_calls, 3, 3, 4096]`; frames are on disk; labels follow the
schedule. Check with:

```bash
python - <<'PY'
import numpy as np, pandas as pd
c = pd.read_parquet("experiments/efference_probe/data/smoke01/calls.parquet")
print(c[["episode_id","call_idx","label","transform","terminated"]])
print("nulls:", int(c.isna().sum().sum()))
with np.load("experiments/efference_probe/data/smoke01/hidden/ep00000.npz") as a:
    print("hidden:", a["h"].shape, a["h"].dtype, "calls:", a["call_idx"])
PY
```

`--dry-run` resolves and prints a config without loading the model.

### S1 — pilot (1 task, 20 episodes)

```bash
python experiments/efference_probe/run_collect.py \
    --config experiments/efference_probe/configs/pilot.yaml \
    --set model.model_path=/abs/path/to/ckpt
python experiments/efference_probe/run_probes.py \
    --run experiments/efference_probe/data/pilot01 --stage pilot
```

**Acceptance gates** (read `analysis/summary.md`):

| Gate | Expected | If it fails |
|---|---|---|
| P0 (shuffled labels) | 0.45–0.55 | the pipeline leaks; check grouping and sample construction |
| **Plumbing integrity** (`analysis/plumbing.json`) | `passed=true`; action/state mismatch counts are 0 | labels, transforms, donors, or state alignment are broken — fix before scaling |
| P2r (endpoint mismatch baseline) | report; no universal threshold | inspect transform strength and controller dynamics |
| C_phase (task phase alone) | ≈ 0.5 | residual phase imbalance; check the signed phase-bias block |
| C_cmd (command alone) | ≈ 0.5 | the schedule correlates with something it should not |
| Class step-index distributions | overlapping | phase matching is failing; see `step_index_report.csv` |

`run_probes.py` performs the plumbing check before fitting any probe and exits
early on failure. P2 and P2r remain reported, transform-dependent baselines;
neither is used to infer whether the stored rows are internally consistent.

### S2 — main collection and core probes

```bash
python experiments/efference_probe/run_collect.py \
    --config experiments/efference_probe/configs/main.yaml \
    --set model.model_path=/abs/path/to/ckpt
python experiments/efference_probe/run_probes.py \
    --run experiments/efference_probe/data/main01 --stage main
```

Produces F1 (layer-depth curves), F2 (the ladder), F4, F5, the T1 run card and
`summary.md`.

### S3 — controls

The `--stage main` run already emits the cross-task split, the per-transform
breakdown, the global-pool negatives check, and the `dstates`-only control.
For the T1/T2 transform gradient, collect the two extra runs:

```bash
for t in mirror freeze; do
  python experiments/efference_probe/run_collect.py \
      --config experiments/efference_probe/configs/main_$t.yaml \
      --set model.model_path=/abs/path/to/ckpt
  python experiments/efference_probe/run_probes.py \
      --run experiments/efference_probe/data/main01_$t --stage main
done
```

A `sqrt_dim` block-scaling variant of the concatenated probes (P3/P4) guards
against the 56-dimensional action block being swamped by 4096 hidden dims under
a shared L2 penalty:

```bash
python experiments/efference_probe/run_probes.py --run <run> \
    --block-scaling sqrt_dim --out <run>/analysis_sqrtdim
```

### S4 — stretch

- **P5** (vision/projector features): re-collect with
  `--set capture.capture_vision=true`, then `--probes P5`.
- **E3(iii)** undo-alignment: already reported under `extra.undo_alignment`.
- **E4** SFT vs GRPO: rerun `main.yaml` with the other checkpoint and a new
  `run_id`, then compare ladders.

---

## What the code does

```
config.py     dataclass config + YAML loader + stable config hash
hijack.py     transforms (mirror/freeze/swap) and the schedule      [pure numpy]
capture.py    forward-hook capture of hidden states and entropy     [torch]
harness.py    the collection loop: env + model, no Ray              [torch + rlinf]
storage.py    run-dir layout, parquet/npz/jpeg writers
datasets.py   probe-sample construction and feature assembly        [pure numpy]
probes.py     grouped-CV logistic probes and the E1 ridge readout   [sklearn]
analysis.py   the ladder, controls, E1, E3, markdown summary
figures.py    F1-F5 and the T1 run card
run_collect.py / run_probes.py    CLI entry points
tests/        synthetic run generator + 31 offline tests
```

The split is deliberate: everything from `datasets.py` rightwards runs without
torch or LIBERO, so probes can be re-run on a laptop while the GPU is busy.

### Hidden-state capture

`predict_action_batch` already calls the language model with
`output_hidden_states=True`, so every layer's activations exist whether or not
anyone reads them. `capture.py` attaches a **forward hook** to
`model.language_model`, pools what it needs, and moves it to CPU as float16.
No second forward pass, no monkeypatching, no forked model class — and nothing
to re-sync when `rlinf/` changes.

**Pool definitions (a correction to SPEC §2).** With
`n_act = action_dim × num_action_chunks = 56`, the sequence is

```
[ BOS | 256 vision patches | rest of prompt | 56 action placeholders ]
```

SPEC §2 describes the pools as `[-56:]` / `[:-56]` / `-57`. The placeholder
block really is the last 56 positions, but **its hidden states are not what the
model reads**. `predict_action_batch` takes its action hidden states from
`[-n_act-1 : -1]` and its logits from `[n_patches + n_prompt_tokens : + n_act]`
— the same window, shifted one step left in the usual next-token-prediction
offset. So:

| Pool | Positions | What it is |
|---|---|---|
| `act_mean` | mean over `[-57:-1]` | exactly the window the model decodes all 56 action tokens from |
| `ctx_last` | position `-57` | the last real prompt token; also the value-head position; the scene summary immediately before the policy commits |
| `ctx_mean` | attention-masked mean over `[:-57]` | BOS, vision patches and prompt |

`ctx_mean` is **attention-masked** because prompts are left-padded to
`max_prompt_length`; an unmasked mean would average in pad positions.

This is checked at runtime rather than trusted. On the first call the harness
recomputes the model's own action logprobs from our slice and compares:

```
indexing check passed: {'max_abs_logprob_diff': 0.0007, 'tolerance': 0.05,
                        'seq_len': 440, 'readout_start': 383, 'passed': True}
```

A wrong offset shifts the distribution onto a different token, so the mismatch
would be order 1, not order 0.01. The run aborts if it fails.

### The P2 problem

SPEC §3.3 defines P2 — the mechanical oracle ceiling — as logistic regression
on `a_cmd[m] ⊕ Δstates`, and SPEC §5 uses `P2 ≥ 0.9` on T1/T2 as the S1 gate
for "the labels or Δstates logging are broken".

**A linear probe on that concatenation cannot form the comparison.** "Did the
world move the way I asked?" is an *agreement* between two vectors — bilinear,
not linear — and no linear function of the two blocks side by side can express
it. That much is a theorem, not an empirical claim.

But it does not follow that P2 sits at chance. Agreement is not the only
linearly available signal: whenever a transform shifts the *mean* of `Δstates`,
a linear probe separates the classes on the marginal alone, without ever
forming a comparison. Measured on the identity-dynamics synthetic fixture,
with commands that are zero-mean i.i.d. versus directed (as real reaching
motion is):

| transform | commands | P2 (concat) | P2r (mismatch) | C_dstates alone |
|---|---|---|---|---|
| `swap` | zero-mean | 0.446 | 0.973 | 0.495 |
| `swap` | directed | 0.412 | 0.982 | 0.475 |
| `mirror` | zero-mean | 0.483 | 0.987 | 0.491 |
| `mirror` | directed | **1.000** | 1.000 | 1.000 |
| `freeze` | zero-mean | 0.475 | 1.000 | 0.520 |
| `freeze` | directed | **1.000** | 1.000 | 0.996 |

So, precisely:

- For **`swap`** — mean-preserving, since a donor chunk has the same action
  statistics — P2 really is at chance, in both regimes. `main.yaml`'s primary
  transform is `swap`, so P2 will be uninformative for the headline run.
- For **`mirror` and `freeze`** on directed motion, P2 clears 0.9 easily. These
  are exactly the transforms SPEC §5's gate names, so **the gate will pass on
  real data** — an earlier version of this README claimed it would false-alarm,
  which is wrong.

Either way P2 is **not interpretable as a ceiling**. When it is low it is low
for a parameterisation reason, not because the information is absent; when it
is high, `C_dstates` alone is high too, so it is detecting "the arm did not
move the way arms normally move", not "the outcome disagrees with the command".

Hence both are reported:

- **P2** — the literal concatenation. Kept, because it is the honest *linear*
  oracle and the like-for-like comparison for P3/P4, which are also linear.
- **P2r** — the same information plus explicit endpoint comparison terms: the
  chunk-summed command, state delta, elementwise products, norms and cosines.
  It is a useful baseline, not a transform-independent ceiling: a real LIBERO
  controller scales and clips each action, contacts make the mapping nonlinear,
  and same-task `swap` donors can produce very similar endpoint motion.

**Use `analysis/plumbing.json` for the S1 plumbing gate.** It reconstructs each
logged transform, checks every swap donor, and verifies that one row's
`states_after` equals the next row's `states_before`. These are exact logging
invariants and do not assume a command-to-state dynamics model.

One caution for the write-up: `P3 > P2` is not by itself evidence of an
efference copy. It shows the network has already performed a comparison a
linear probe cannot perform on raw inputs, which is interesting — but the
efference question is P1 versus P3/P4. Do not call either P2 or P2r a universal
mechanical ceiling; report the transform and observed baseline value.

Two further controls, not in the spec, are run by default:

- **C_cmd** (`a_cmd` alone) — must sit at chance. The command is produced
  *before* the hijack, so any signal here means the schedule or the phase
  matching correlates with something it should not.
- **C_dstates** (`Δstates` alone) — how much of the label is trivially
  mechanical. Expect this near 1.0 for `freeze`/`mirror` on directed motion
  (the marginal shift above) and near chance for `swap`. When C_dstates is
  high, P2 is not measuring a comparison.
- **C_phase** (`cur_call`, `cur_call²`) — task phase alone. Bounds how much of
  any hidden-state result could be explained by residual phase imbalance
  between the classes.

### Probe methodology

- 5-fold `StratifiedGroupKFold` **grouped by episode** — consecutive calls
  within an episode are strongly correlated, and a plain k-fold would leak
  neighbouring frames across the split.
- Features standardised; L2 logistic regression with `class_weight="balanced"`.
- `C` chosen **inside each outer fold** on one inner grouped split, so the
  reported score is not inflated by picking `C` against the data it is scored
  on. The per-`C` table is reported alongside so the ladder's ordering can be
  checked for stability. `--fast` skips selection for exploratory sweeps.
- Balanced accuracy and AUROC, mean ± fold standard deviation.
- The `(layer, pool)` cell used for F2/F3 is chosen as the maximum over
  `layers x pools` configurations, so it is a *reported* maximum, not an
  independent estimate. Comparing it against 0.5 overstates it: on a pure null
  the maximum over 27 cells lands around 0.60, roughly +0.10 above chance.
  Three things address that, and the ladder should be read against them rather
  than against 0.5:
  - **`selection_floor`** in `summary.md` runs P0 (shuffled labels) over the
    *identical* grid and reports its maximum. That absorbs exactly the same
    selection advantage, so it is the like-for-like floor. F2 draws it as a
    red line.
  - **F1 facets over every pool and includes P0**, so the whole grid the
    maximum was taken over is visible, along with how high the null climbs
    across it.
  - **`--permutations N`** gives a calibrated null for the selected cell
    specifically (report P1/P3/P4 against its p95, not against 0.5).

  What is still missing: there is no significance test *between* rungs. The
  error bars in F1/F2/F3 are cross-validation fold standard deviations, and
  folds share training data, so they are not calibrated intervals — do not read
  "P4's bar clears P1's" as a test.

### Runtime, and the one performance trap

The ladder is a grid: every hidden-state probe is fitted at each of
`9 layers x 3 pools = 27` cells, and each cell is a 5-fold grouped CV with
per-fold `C` selection. At the real run's scale (~1.3k probe samples, 4096-wide
features) one cell is a few seconds, so the main stage is dominated purely by
cell count.

Cells are independent fits with fixed seeds, so they parallelise exactly —
`--jobs` (default `-1`, all cores) changes wall clock and nothing else. The
numbers are identical to a serial run; there is no sampling, no shared state,
and no ordering dependence.

**Threaded BLAS is slower here than one thread per cell.** Measured on one
real-scale P0 cell (1412 samples x 4096 features, 5-fold with C selection) on a
4-core machine:

| `OMP_NUM_THREADS` | cell | BLAS threads used |
|---|---|---|
| 1 | 2.87 s | 1 |
| 4 | 6.48 s | 4 |
| 16 | 6.45 s | 4 (OpenBLAS caps at core count) |
| 64 | 6.24 s | 4 (same) |

The fits are many small operations rather than a few large ones, so threading
them costs more in synchronisation than it recovers. Four single-threaded
workers therefore beat one four-threaded process by more than 4x, which is why
`--jobs` is superlinear: the full ladder measured 16.3 min serial against
2.78 min at `--jobs -1`.

Note the last two rows: OpenBLAS limits itself to the machine's core count, so
setting a huge thread count does *not* by itself produce runaway
oversubscription. Pinning threads is still worth doing under a scheduler that
grants fewer cores than the node has, because an allocation-unaware BLAS would
size itself to the node:

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
```

`run_probes.py` logs the resolved worker and thread counts at startup, so a run
that ever crawls can be diagnosed from its own log rather than by re-running it.


### Sample construction

Each probe sample is a pair of consecutive calls `(prev, cur)` with
`cur = prev + 1`. Positives had `prev` hijacked; negatives did not. In both
classes `cur` itself is SELF — guaranteed for positives by the scheduler's
clean-gap rule — so the classes differ in exactly one thing.

Negatives are **phase-matched** by solving a minimum-total-distance assignment
between positives and eligible calls *within each episode*. Greedy
nearest-neighbour matching (the obvious implementation) leaves late positives
with whatever early calls are left over, which reintroduces the phase confound
it is meant to remove. Negatives are also floored at `warmup_calls + 1`, the
earliest index a positive can occupy. `--no-controls` aside, the run always
reports both classes' step-index distributions; they must overlap.

`negatives="global"` (reported as a robustness check) samples from the whole
run and deliberately does *not* control for phase.

---

## Design decisions

**Hijacks are applied in the environment action space**, after
`rlinf.envs.action_utils.prepare_actions` has mapped the policy's `[0,1]`
gripper output onto LIBERO's `±1` convention. That makes `freeze` and `mirror`
unambiguous: dimension 6 is already a signed gripper command and 0–5 are
already the deltas the controller integrates. Note that
`prepare_actions_for_libero` rewrites its input **in place and does not copy**,
so the harness always hands it a copy — otherwise the `a_cmd` being logged
would be silently corrupted.

**One task per batch.** Every batch holds `num_envs` different initial states
of a *single* task. Two consequences: swap donors are always same-task
episodes, which is what makes T3 a control for "weird actions" rather than a
new confound; and with `is_eval: True` LIBERO rebuilds its envs only when the
task changes, so that is one rebuild per task instead of one per episode.

**Episode boundaries are managed explicitly** (`auto_reset: False`,
`ignore_terminations: False`), so `chunk_step` reports real per-substep
terminations and the first-success call can be pinned exactly. Recording stops
at first success; post-success calls never enter the data.

**Clean control episodes.** `hijack.probe_episode_fraction` (default 0.8)
leaves a fraction of episodes completely un-hijacked, which is how SPEC §3.2's
"per episode flagged as a probe episode" is realised. It gives E3 its
success-rate comparison within one run instead of paying for a second
collection pass. Clean episodes are still valid swap donors.

**Entropy is computed at temperature 1** and without top-k, so it measures the
policy's own uncertainty rather than the sampling knobs in the run config.
`logprob_sum` comes from the model's own `prev_logprobs` and does reflect them.

**No Ray.** The harness talks to `LiberoEnv` and the model directly. The worker
layer exists to move tensors between processes, which buys nothing for a
single-GPU inference loop and would cost debugging time. What is reused
verbatim is everything where drift would be silent: `rlinf.models.get_model`
for checkpoint loading, `LiberoEnv` for the sim, `prepare_actions` for the
action convention, and the shipped `model/` and `env/` YAMLs as config bases.

**Determinism.** Fixed `seed`, `use_fixed_reset_state_ids: True`, and reset
state ids computed explicitly from LIBERO's `(task, trial)` bin edges. The
hijack RNG is derived from `(seed, task_id, batch_index)`, so a single batch can
be reproduced without replaying the suite. The design is paired within a run, so
bit-exactness across runs is not required — don't spend time chasing it.

---

## Data layout

```
data/<run_id>/
  run_config.yaml     resolved config, git SHA, checkpoint, indexing-check report
  calls.parquet       one row per model call
  hidden/ep00000.npz  h[n_calls, n_layers, n_pools, hidden] float16 + call_idx
  frames/ep00000/     call0000_main.jpg, call0000_wrist.jpg
  manifest.json       counts, budget usage, per-episode summary
  analysis/           written by run_probes.py
```

`calls.parquet` columns: `run_id, episode_id, task_id, reset_state_id,
env_slot, call_idx, is_probe_episode, label, transform, donor_env_slot,
donor_episode_id, skipped_reason, a_cmd_model, a_cmd_env, a_exec,
states_before, states_after, logprob_sum, logprobs, entropy_mean,
entropy_tokens, reward_delta, success_substep, terminated, truncated,
first_success_call, success_flag, post_success`.

`reward_delta` is named for what it is: the LIBERO env configs set
`use_rel_reward`, so per-substep rewards telescope and the per-chunk sum is
*(reward at the end of this chunk) − (reward at the end of the previous one)*,
not a chunk return. Nothing in the analysis reads it; don't average it.

`post_success` is always `False` as things stand — recording stops at the first
success, so no later call is ever written. It is kept so the downstream filter
stays correct if that rule is relaxed.

Rows are written to `calls.parquet` after **every batch**, not only at the end.
A fault three hours into a run therefore costs one batch, not the whole run.
Episodes still running when a budget cap fires are marked `censored: true` in
`manifest.json` and excluded from success-rate reporting — they were cut off,
not failed.

Array columns are stored **flattened**; their shapes are recorded in
`run_config.yaml` under `schema.array_columns`. `a_cmd_model` is the policy's
raw output, `a_cmd_env` the same chunk after the gripper mapping (this is the
canonical 56-dimensional efference vector the probes use), `a_exec` what was
actually stepped. `states` is LIBERO's 8-dim
`[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]`.

Storage: ~221 KB of hidden states per call at 9 layers × 3 pools × 4096, so a
7k-call run is ~1.5 GB — comfortably inside the 30 GB cap.

Videos are not written (it would mean a new dependency). Frames are, so:

```bash
ffmpeg -framerate 5 -pattern_type glob \
    -i 'data/main01/frames/ep00000/*_main.jpg' ep00000.mp4
```

---

## Tests

```bash
python -m pytest experiments/efference_probe/tests/test_offline.py -q   # 43 tests
```

They cover the hijack schedule (warm-up, clean gap, realised rate, donor
selection, no mutation of commanded actions), the sequence arithmetic behind
the pools, probe-sample construction (pairing, phase matching, post-success
exclusion), the direct plumbing gate for all three transforms, probe behaviour
against planted ground truth (P0 at chance, C_cmd at chance, P2r recovering the
label under the fixture's identity dynamics, P2 underperforming P2r), the
storage round-trip, and config validation.

`tests/synthetic.py` writes a full fake run directory, so the analysis half can
be exercised end to end with no GPU:

```bash
python experiments/efference_probe/tests/synthetic.py /tmp/synth
python experiments/efference_probe/run_probes.py --run /tmp/synth --stage main --fast
```

---

## Known limitations

- `capture.py` and `harness.py` have not been run against a real checkpoint —
  this checkout has no torch. The indexing self-check exists precisely because
  that verification has to happen on first contact with the model.
- P5 needs `model.projector` to exist on the OFT model. If it does not, vision
  capture disables itself with a warning rather than sinking the run.
- `skip_intermediate_renders: True` (the default, for speed) means only the
  last substep of each chunk is observed, so `Δstates` is chunk-level. Set it
  false for per-substep states at a substantial rendering cost.
- The undo-alignment metric fits the controller gain by least squares on
  self-caused calls; it is a proxy, and is reported as a stretch number. It
  scores the command *innovation* (the part not predicted by simply continuing
  the previous command) rather than the raw next command — scoring the raw
  command reduces to `+(A_prev · A_cur)` and reports a large fake "correction"
  for any policy whose commands are temporally autocorrelated, which every
  smooth reach is. The uncorrected number is reported alongside so the size of
  that artefact stays visible.
- Episode-level hijack-count/success correlations are deliberately *not*
  reported: hijacks accumulate only while an episode runs and successful
  episodes end early, so that correlation is strongly negative even under a
  null. The probe-vs-clean contrast is randomised per episode and is the
  comparison to use.
- `Δstates` composes the end-effector rotation properly (`R_after · R_beforeᵀ`)
  rather than subtracting absolute axis-angles componentwise, which wraps: two
  orientations a hair apart can otherwise differ by ~2π.

---

## Scope

Out of scope by design (SPEC §1): perturbation-robustness benchmarking. Success
rate under hijack appears only as one supporting number in E3, never as a
headline. `LIBERO_TYPE` stays `standard`.

Per SPEC §8, the code reports numbers and anomalies; interpretation belongs in
the write-up. `results.md` is the template for that.

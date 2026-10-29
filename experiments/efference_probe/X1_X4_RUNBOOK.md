# X1 / X4 runbook

What `PREREG_X1_X4.md` registers, in the order it runs. Nothing here may start before the operator approves the spend; the pricing correction in the logbook (2026-10-29, "X1 / X4 code ready") must be resolved first.

**Conventions:**
- Every `modal run` below is `modal run experiments/efference_probe/modal_e4.py::stage`, shortened to `stage`.
- `--wave x1` / `--wave x4` route the driver to `<wave>_plan.yaml` and `/vol/data/<wave>`, so E4's state is never touched.
- GPU commands (`s0`, `s1b`, `collect`) get an L40S container; everything else gets a CPU container.

## X1

```bash
# Checkpoints (CPU): download, verify against the pinned shard hashes, register.
stage --wave x1 --cmd "download --ckpt sft --dest /vol/ckpts/x1/sft"
stage --wave x1 --cmd "download --ckpt grpo --dest /vol/ckpts/x1/grpo"
stage --wave x1 --cmd "init --sft /vol/ckpts/x1/sft --grpo /vol/ckpts/x1/grpo"

# S0 (GPU): 1 task x 8 states per checkpoint.  The gate includes the new logging.
stage --wave x1 --cmd "s0"
stage --wave x1 --cmd "lock --num-envs 10"

# S1b (GPU) and Amendment 1's sizing (CPU).
stage --wave x1 --cmd "s1b"
stage --wave x1 --cmd "size"

# S2 (GPU), one collection per call, in plan order.
for key in swap mirror freeze; do
  for ckpt in sft grpo; do stage --wave x1 --cmd "collect --ckpt $ckpt --key $key"; done
done
```

**S0 log line.** After S0, the object key list (`manifest.json` → `object_keys`) goes into the logbook as the S0 record of PREREG §6.

Then locally, at $0:

```bash
modal run experiments/efference_probe/modal_e4.py::pull --dest experiments/efference_probe/data
P=experiments/efference_probe
D=$P/data/x1
for r in x1_sft_swap x1_grpo_swap x1_sft_mirror x1_grpo_mirror x1_sft_freeze x1_grpo_freeze; do
  python $P/e4_wave.py --plan $P/x1_plan.yaml --data-root $D probe --run-id $r --family linear
done
python $P/x1_report.py --data-root $D controls <runs>
python $P/x1_report.py --data-root $D resid x1_grpo_swap x1_sft_swap
python $P/x1_report.py --data-root $D increment x1_grpo_swap
python $P/x1_report.py --data-root $D fixed-cell --e4-root $P/data
python $P/x1_report.py --data-root $D battery x1_sft_swap x1_grpo_swap
python $P/x1_report.py --data-root $D report --e4-root $P/data --out $P/reports/x1_report.md
```

- **`pull` and hidden states.** `pull` skips `hidden/`. Fetch the X1 runs' `hidden/` with `modal volume get` before the probe rounds.
- **E4 data for `fixed-cell`.** It needs the four E4 goal-swap runs' hidden states, which X0 already fetched.

## X4

The monitors are frozen and logged ("X4 monitors frozen"); `x4_plan.yaml` loads only those digests.

```bash
# Checkpoints: E4's, already on the Volume; register them under X4's own state.
stage --wave x4 --cmd "init --sft <E4 sft path> --grpo <E4 grpo path>"

# S0 (GPU): 1 task x 8 states, conditions cycled so all three occur.
stage --wave x4 --cmd "s0"
stage --wave x4 --cmd "lock --num-envs 10"

# S2 (GPU): seeds 3-6, GRPO before Base-Lora within each seed.
for key in mirror_s3 mirror_s4 mirror_s5 mirror_s6; do
  for ckpt in grpo sft; do stage --wave x4 --cmd "collect --ckpt $ckpt --key $key"; done
done
```

**E4 checkpoint paths.** Read them from E4's `e4_state.json` → `checkpoints`.

Then locally:

```bash
modal run experiments/efference_probe/modal_e4.py::pull --dest experiments/efference_probe/data
python $P/x4_report.py --data-root $P/data/x4 --out $P/reports/x4_report.md
```

## Stops

- A GPU-hour cap (X1 1.6 h, X4 2.0 h), a watchdog (2× expected), the two-strike rule, or a failed gate stops the stage. The stop is reported, not routed around.
- The spend journal's combined cap is the operator's to set at launch: `$8` as registered, unless amended.

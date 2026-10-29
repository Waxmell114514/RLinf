# X1 / X4 runbook

What `PREREG_X1_X4.md` registers, in the order it runs. Nothing here may start before the operator approves the spend; the pricing correction in the logbook (2026-10-29, "X1 / X4 code ready") must be resolved first.

**Conventions:**
- Every `modal run` below is `modal run experiments/efference_probe/modal_e4.py::stage`, shortened to `stage`.
- `--wave x1` / `--wave x4` route the driver to `<wave>_plan.yaml` and `/vol/data/<wave>`, so E4's state is never touched.
- GPU commands (`s0`, `s1b`, `collect`) get an L40S container; everything else gets a CPU container.
- **Every X call carries `--budget-usd <cap>`**, the combined X1 + X4 cap the operator sets (PREREG §5 registers `8`; Amendment 1, item 4). `stage` refuses an X call without it. One journal on the Volume (session `x_waves`) is shared by X1 and X4, so the same cap is passed to every call of both waves. A call starts only if committed spend plus its worst case fits under the cap; otherwise it is not started, and the refusal is reported.
- Below, `stage` stands for `stage --budget-usd $CAP`.

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

# S2 (GPU), one collection per call, in plan order: swap (SFT, GRPO), mirror, freeze.
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
RUNS="x1_sft_swap x1_grpo_swap x1_sft_mirror x1_grpo_mirror x1_sft_freeze x1_grpo_freeze"
python $P/x1_report.py --data-root $D controls $RUNS                 # R2's new controls
python $P/x1_report.py --data-root $D increment x1_grpo_swap         # R3's paired bootstrap
python $P/x1_report.py --data-root $D fixed-cell --e4-root $P/data   # R4
python $P/x1_report.py --data-root $D resid x1_grpo_swap x1_sft_swap # R5
python $P/x1_report.py --data-root $D battery $RUNS                  # R7: A2, A2b, F2, A3, F3, F5
python $P/x1_report.py --data-root $D battery-pair                   # R7: F4 on the swap pair (after battery)
python $P/x1_report.py --data-root $D report --e4-root $P/data --out $P/reports/x1_report.md
```

- **`pull` and hidden states.** `pull` skips `hidden/`. Fetch the X1 runs' `hidden/` with `modal volume get` before the probe rounds.
- **E4 data for `fixed-cell`.** It needs the four E4 goal-swap runs' hidden states, which X0 already fetched.
- **Readout runs.** Use the run ids as written; an attempt suffix (`_r2`) is mapped back to its planned id. R3 reads "unresolved (k of 6 X1 runs)" until all six runs are probed. `report` also writes the main02 reproduction check (x1_sft_swap against main02: P1, P3, P4 and P2r each within 0.04).

## X4

The monitors are frozen and logged ("X4 monitors frozen"); `x4_plan.yaml` loads only those digests.

```bash
# Checkpoints: E4's, already on the Volume; register them under X4's own state.
stage --wave x4 --cmd "init --sft <E4 sft path> --grpo <E4 grpo path>"

# S0 (GPU): 1 task x 8 states at S2's 10 envs, conditions cycled so all three occur.
stage --wave x4 --cmd "s0"
stage --wave x4 --cmd "lock --num-envs 10"

# S2 (GPU): all four GRPO seeds, then all four Base-Lora seeds (PREREG §3; Amendment 1, item 5).
for ckpt in grpo sft; do
  for key in mirror_s3 mirror_s4 mirror_s5 mirror_s6; do stage --wave x4 --cmd "collect --ckpt $ckpt --key $key"; done
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
- The spend journal's combined cap is the operator's to set at launch: `$8` as registered, unless amended. The journal prices containers at their request limits (GPU $3.28/h, CPU $1.52/h), above the ~$2.8/h actually billed, and reserves each call's watchdog worst case. On the plan files that needs a cap of about $9 to finish X1 alone and about $14–15 for X1 + X4 [estimate; logbook, "Amendment 1"]. At $8 it would refuse X1's sized GRPO swap arm.
- **Order between waves.** X1 runs first; X4 is approved and launched separately (PREREG §5).

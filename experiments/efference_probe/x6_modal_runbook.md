# X6 on Modal: the commands, stage by stage

**Plan:** `PREREG_X6.md` §5, as changed by Amendment 1.

**Approval first.** Nothing here runs without the user's approval of the stage, and every call goes through the x-waves spend journal.

## Setup

**Tokens:** they stay in `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`; never print them.

```bash
export M=experiments/efference_probe/modal_e4.py
export CKPT=/vol/ckpts/grpo          # GRPO-130 on the Volume (X4)
export OUT=/vol/data/x6
export LOG=/vol/data/x6/logbook_x6.md
export CFG=experiments/efference_probe/configs
modal run $M::journal                # committed total; CAP = that + 29.5
export CAP=...                       # set once, before S0
```

## Stages

### S0: wrapper smoke

About 8 min, ≈ $0.5. Gate: `passed` in `ec_smoke.json`.

```bash
modal run $M::script --path experiments/efference_probe/x6_ec_smoke.py \
  --args "--config $CFG/x6_fault_oft.yaml --out $OUT/ec_smoke.json \
          --set model.model_path=$CKPT out_root=$OUT run_id=x6_ec_smoke env.num_envs=4 env.task_ids=[2]" \
  --budget-usd $CAP --max-s 1200
```

### S1: Part A, OFT (= arm F)

1,800 episodes, about 1 h, ≈ $3.3. Gate: clean success ≥ 0.80.

```bash
modal run $M::script --path experiments/efference_probe/run_collect.py \
  --args "--config $CFG/x6_fault_oft.yaml --logbook $LOG --set model.model_path=$CKPT out_root=$OUT" \
  --budget-usd $CAP --max-s 6000
```

### S3: training collection

1,200 episodes with frames, about 50 min, ≈ $2.7. Gate: `x6_ec_report.py --collection`.

```bash
modal run $M::script --path experiments/efference_probe/run_collect.py \
  --args "--config $CFG/x6_ec_collect.yaml --logbook $LOG --set model.model_path=$CKPT out_root=$OUT" \
  --budget-usd $CAP --max-s 5400
```

### S3b: training smoke

H100, 20 steps, ≈ $0.7. Gate: the checks pass, and s/step × 2 epochs ≤ 50 min.

```bash
modal run $M::script --kind train --path experiments/efference_probe/train_ec.py \
  --args "--config $CFG/x6_fault_oft.yaml --runs $OUT/x6_ec_collect --arm A \
          --out $OUT/arms/smoke --max-steps 20 --log-every 1 --set model.model_path=$CKPT" \
  --budget-usd $CAP --max-s 1500
```

### S4: fine-tunes, seed 0

Each about 30–45 min `[S3b measures it]`, at $4.71/h.

```bash
for ARM in A B C D; do
  modal run $M::script --kind train --path experiments/efference_probe/train_ec.py \
    --args "--config $CFG/x6_fault_oft.yaml --runs $OUT/x6_ec_collect --arm $ARM \
            --out $OUT/arms/${ARM}_s0 --seed 0 --set model.model_path=$CKPT" \
    --budget-usd $CAP --max-s 3600
done
```

### S5: evaluations

A runs all three rule cells; B, C and D run the two fault cells. Each run takes 20–30 min.

```bash
modal run $M::script --path experiments/efference_probe/run_collect.py \
  --args "--config $CFG/x6_ec_eval.yaml --logbook $LOG --set model.model_path=$CKPT out_root=$OUT \
          run_id=x6_ec_eval_A efference.arm_dir=$OUT/arms/A_s0" \
  --budget-usd $CAP --max-s 3000
for ARM in B C D; do
  modal run $M::script --path experiments/efference_probe/run_collect.py \
    --args "--config $CFG/x6_ec_eval.yaml --logbook $LOG --set model.model_path=$CKPT out_root=$OUT \
            run_id=x6_ec_eval_$ARM efference.arm_dir=$OUT/arms/${ARM}_s0 fault.sweep=[bias_x:0.3,invert_x]" \
    --budget-usd $CAP --max-s 2400
done
```

## Readout (local, $0)

Copy the runs down with `modal run $M::pull`. The pull skips frames, which the readout does not need.

```bash
python experiments/efference_probe/x6_ec_report.py \
  --arm F=data/x6/x6_fault_oft --arm A=data/x6/x6_ec_eval_A --arm B=data/x6/x6_ec_eval_B \
  --arm C=data/x6/x6_ec_eval_C --arm D=data/x6/x6_ec_eval_D --out data/x6/x6_ec_readout.json
```

## Not yet runnable

- **S2 (π0):** needs its own image (`install.sh embodied --model openpi --env libero`) and the checkpoint on the Volume.

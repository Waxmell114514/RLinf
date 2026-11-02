"""X6-EC: fine-tune one arm on the training collection (PREREG_X6.md section 3.3).

    python experiments/efference_probe/train_ec.py \\
        --config experiments/efference_probe/configs/x6_fault_oft.yaml \\
        --runs /vol/data/x6/x6_ec_collect --arm A --out /vol/data/x6/arms/A_s0

The checkpoint loads through the harness's model config and ``rlinf``'s
``get_model``; :func:`ec_policy.adopt` adds the token (float32 projector,
output layer at zero) and LoRA r = 32 goes on the LLM blocks only.  The loss
is the token NLL of the target chunk through the class's own
``default_forward`` (the path its GRPO actor trains through), plus, for arms D
and E, a forward-model head on the pooled action-position hidden state and
the sent chunk, regressed on the scaled state change.

Before the first step the script checks that :func:`model_inputs` rebuilds
exactly the inputs ``predict_action_batch`` builds from the same observation,
and that decoding the policy's logged tokens gives back its logged commands;
it stops on either mismatch.  The arm, its input statistics and the training
log go to ``--out`` (``ec_policy.save_arm``); ``harness`` evaluates it with
``efference.arm_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import ec_dataset, ec_policy  # noqa: E402
from efference_probe.config import load_config  # noqa: E402

LORA_RANK = 32
FM_WIDTH = 512


def model_inputs(model, images, descriptions, states) -> dict[str, Any]:
    """``predict_action_batch``'s input processing, one image, on the CPU.

    Mirrors its ``env_obs`` branch line for line; ``check_inputs`` compares the
    two on a real batch before training starts.
    """
    import torch

    if model.vision_backbone.get_num_images_in_input() != 1:
        raise ValueError("model_inputs covers the one-image OFT configuration")
    prompts = [
        f"In: What action should the robot take to {t.lower()}?\nOut: "
        for t in descriptions
    ]
    images = torch.as_tensor(images).unsqueeze(1).permute(0, 1, 4, 2, 3)
    inputs = model.input_processor(
        text=prompts,
        images={"images": images},
        proprio_states=torch.as_tensor(states),
        padding="max_length",
        max_length=model.max_prompt_length,
    )
    pixel_values = inputs["pixel_values"]
    b, n, c, h, w = pixel_values.shape
    return {
        "input_ids": inputs["input_ids"].to(torch.long),
        "attention_mask": inputs["attention_mask"].to(torch.bool),
        "pixel_values": pixel_values.reshape(b, n * c, h, w),
    }


def to_device(inputs: dict[str, Any], model) -> dict[str, Any]:
    param = next(model.parameters())
    out = dict(inputs)
    for key in ("input_ids", "attention_mask"):
        out[key] = inputs[key].to(param.device)
    out["pixel_values"] = inputs["pixel_values"].to(param.device, dtype=param.dtype)
    return out


def check_inputs(model, images, descriptions, states, efference) -> None:
    """Stop unless model_inputs matches predict_action_batch's own inputs."""
    import torch

    env_obs = {
        "main_images": torch.as_tensor(images),
        "task_descriptions": list(descriptions),
        "states": torch.as_tensor(states),
        "efference": efference,
    }
    with torch.no_grad():
        _, result = model.predict_action_batch(
            env_obs=env_obs, do_sample=False, temperature=1.0, top_k=-1, top_p=1.0
        )
    ours = to_device(model_inputs(model, images, descriptions, states), model)
    for key in ("input_ids", "attention_mask", "pixel_values"):
        theirs = result["forward_inputs"][key]
        if ours[key].shape != theirs.shape or not torch.equal(ours[key], theirs):
            raise RuntimeError(f"model_inputs differs from predict's {key}")


def cosine_lambda(warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def batches(n: int, batch: int, epochs: int, rng: np.random.Generator):
    """(epoch, indices) for every full batch; the last partial one is dropped."""
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n - batch + 1, batch):
            yield epoch, order[start : start + batch]


def gate_summary(log: list[dict[str, Any]], window: int = 100) -> dict[str, Any]:
    """S4's gate: the loss is finite and its last window is below its first."""
    nll = np.array([entry["nll"] for entry in log], dtype=np.float64)
    if not len(nll):
        return {"passed": False, "reason": "no steps"}
    w = min(window, max(len(nll) // 4, 1))
    first, last = float(nll[:w].mean()), float(nll[-w:].mean())
    finite = bool(np.isfinite(nll).all())
    return {
        "first_window_nll": first,
        "last_window_nll": last,
        "window": w,
        "finite": finite,
        "passed": finite and last < first,
    }


def load_checkpoint(cfg):
    """The checkpoint, through the harness's model config and rlinf's loader."""
    from efference_probe.harness import build_model_config
    from rlinf.models import get_model

    return get_model(build_model_config(cfg)).to("cuda")


def train(args, load_model=load_checkpoint) -> dict[str, Any]:
    import torch
    from peft import get_peft_model

    cfg = load_config(args.config, args.set)
    spec = ec_dataset.ARMS[args.arm]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    calls = ec_dataset.load_calls([Path(r) for r in args.runs])
    model = load_model(cfg)
    device = next(model.parameters()).device
    # The class may extend the configured key (``_no_noops``); decode with its own.
    unnorm_key = getattr(model, "unnorm_key", cfg.model.unnorm_key)
    tokenizer = ec_dataset.ActionTokenizer.from_model(model, unnorm_key)
    error = ec_dataset.round_trip_error(calls, tokenizer)
    if error > ec_dataset.ROUND_TRIP_ATOL:
        raise RuntimeError(f"token round trip off by {error:.4g}: wrong tokenizer")
    proprio = model.norm_stats.get(unnorm_key, {}).get("proprio")
    bounds = (
        (proprio["q01"], proprio["q99"])
        if proprio and len(proprio.get("q01", [])) == ec_dataset.STATE_DIM
        else None
    )
    stats = ec_dataset.fit_input_stats(calls, bounds)
    samples = ec_dataset.build_samples(calls, spec.kind, stats, tokenizer)
    in_dim = samples.efference.shape[1]

    ec_policy.adopt(model, in_dim, seed=args.seed, projector_dtype=torch.float32)
    peft_model = get_peft_model(model, ec_policy.lora_config(LORA_RANK))
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
            param.requires_grad_(True)
            lora_params.append(param)
    head_params = list(model.efference_projector.parameters())
    for param in head_params:
        param.requires_grad_(True)
    llm_dim = model.get_input_embeddings().weight.shape[1]
    n_act = ec_dataset.CHUNK * ec_dataset.ACTION_DIM
    fm_head = None
    pooled: dict[str, Any] = {}
    if spec.forward_model:
        fm_head = torch.nn.Sequential(
            torch.nn.Linear(llm_dim + n_act, FM_WIDTH),
            torch.nn.GELU(),
            torch.nn.Linear(FM_WIDTH, ec_dataset.STATE_DIM),
        ).to(device)
        head_params += list(fm_head.parameters())

        def keep_pooled(module, _args, output):
            # The hidden states the action logits come from (counted from the
            # end, so the efference row does not shift them).
            hidden = output.hidden_states[-1][:, -n_act - 1 : -1]
            pooled["h"] = hidden.float().mean(dim=1)

        model.language_model.register_forward_hook(keep_pooled)
    first = np.arange(min(args.micro_batch, len(samples)))
    check_inputs(
        model,
        ec_dataset.load_frames([samples.frames[i] for i in first]),
        [samples.descriptions[i] for i in first],
        samples.states[first],
        samples.efference[first],
    )
    # No dropout anywhere (LoRA's is 0); the LM alone goes to train mode when
    # checkpointing, which HF applies only in training mode.
    model.eval()
    if args.grad_checkpointing:
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.language_model.train()

    plan = list(batches(len(samples), args.batch, args.epochs, rng))
    if args.max_steps:
        plan = plan[: args.max_steps]
    # Arm C draws a fresh shuffled history each epoch; the others read theirs.
    epoch_efference = {
        epoch: (
            ec_dataset.shuffled_history(samples, rng)
            if spec.shuffle_history
            else samples.efference
        )
        for epoch in sorted({epoch for epoch, _ in plan})
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_proj},
            {"params": lora_params, "lr": args.lr_lora},
        ],
        weight_decay=0.0,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lambda(args.warmup, len(plan))
    )

    def prepare(epoch: int, index: np.ndarray) -> dict[str, Any]:
        return {
            "inputs": model_inputs(
                model,
                ec_dataset.load_frames([samples.frames[i] for i in index]),
                [samples.descriptions[i] for i in index],
                samples.states[index],
            ),
            "efference": torch.as_tensor(epoch_efference[epoch][index]),
            "targets": torch.as_tensor(samples.targets[index]),
            "delta": torch.as_tensor(samples.delta_target[index]),
            "a_sent": torch.as_tensor(samples.a_sent[index]),
        }

    pool = ThreadPoolExecutor(max_workers=2)

    def submit(step: int) -> list:
        """Decode and preprocess one step's micro-batches in the background."""
        if step >= len(plan):
            return []
        epoch, index = plan[step]
        return [
            pool.submit(prepare, epoch, index[s : s + args.micro_batch])
            for s in range(0, len(index), args.micro_batch)
        ]

    log: list[dict[str, Any]] = []
    started = time.time()
    stopped_early = False
    log_path = out / "train_log.jsonl"
    log_path.write_text("")
    pending = submit(0)
    for step in range(len(plan)):
        futures, pending = pending, submit(step + 1)
        nll_sum = mse_sum = 0.0
        for future in futures:
            batch = future.result()
            result = model.default_forward(
                forward_inputs={
                    **to_device(batch["inputs"], model),
                    "action_tokens": batch["targets"].to(device),
                    "efference": batch["efference"],
                },
                compute_logprobs=True,
                output_hidden_states=spec.forward_model,
                temperature=args.loss_temperature,
                top_k=-1,
            )
            nll = -result["logprobs"].float().mean()
            loss = nll
            if fm_head is not None:
                prediction = fm_head(
                    torch.cat([pooled.pop("h"), batch["a_sent"].to(device)], dim=1)
                )
                mse = torch.nn.functional.mse_loss(
                    prediction, batch["delta"].to(device)
                )
                loss = loss + args.lambda_fm * mse
                mse_sum += float(mse.item())
            (loss / len(futures)).backward()
            nll_sum += float(nll.item())
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        entry = {
            "step": step,
            "epoch": int(plan[step][0]),
            "nll": nll_sum / len(futures),
            "mse": mse_sum / len(futures) if fm_head is not None else None,
            "grad_norm": float(grad_norm),
            "lr_lora": schedule.get_last_lr()[1],
            "elapsed_s": time.time() - started,
        }
        log.append(entry)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        if step % args.log_every == 0:
            print(json.dumps(entry), flush=True)
        if not math.isfinite(entry["nll"]) or (
            args.max_minutes and time.time() - started > 60 * args.max_minutes
        ):
            stopped_early = True
            break
    for future in pending:
        future.cancel()
    pool.shutdown()

    summary = {
        "steps": len(log),
        "planned_steps": len(plan),
        "stopped_early": stopped_early,
        "train_s": time.time() - started,
        "peak_gpu_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else None
        ),
        "gate": gate_summary(log),
    }
    meta = {
        "arm": args.arm,
        "kind": spec.kind,
        "shuffle_history": spec.shuffle_history,
        "forward_model": spec.forward_model,
        "in_dim": int(in_dim),
        "history": 2,
        "chunk": ec_dataset.CHUNK,
        "action_dim": ec_dataset.ACTION_DIM,
        **stats,
        "seed": args.seed,
        "checkpoint": cfg.model.model_path,
        "runs": [str(r) for r in args.runs],
        "n_samples": len(samples),
        "families": ec_dataset.family_counts(calls),
        "round_trip_error": error,
        "lora": {"rank": LORA_RANK, "targets": list(ec_policy.LORA_TARGETS)},
        "optim": {
            "batch": args.batch,
            "micro_batch": args.micro_batch,
            "epochs": args.epochs,
            "lr_proj": args.lr_proj,
            "lr_lora": args.lr_lora,
            "warmup": args.warmup,
            "lambda_fm": args.lambda_fm if spec.forward_model else None,
            "loss_temperature": args.loss_temperature,
        },
        "summary": summary,
    }
    ec_policy.save_arm(model, peft_model, out, meta)
    if fm_head is not None:
        torch.save(fm_head.state_dict(), out / "fm_head.pt")
    return meta


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(ec_dataset.ARMS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--lr-proj", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument(
        "--loss-temperature",
        type=float,
        default=1.0,
        help=(
            "temperature of the NLL's softmax. 1.0 is Amendment 1's recipe; "
            "Amendment 3 sets the collection's sampling temperature (1.6), so "
            "that fitting the policy's own samples leaves it unchanged"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0: the full plan")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.batch % args.micro_batch:
        parser.error("--batch must be a multiple of --micro-batch")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    meta = train(parse_args(argv))
    print(json.dumps(meta["summary"], indent=2))
    return 0 if meta["summary"]["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

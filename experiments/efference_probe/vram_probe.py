#!/usr/bin/env python
"""Wave 3, E5 step 1: does command-input LoRA fine-tuning fit on 24 GB?

E5 fine-tunes OpenVLA-OFT with its previous command chunk as an extra input.
The cheapest form of that input needs no architectural change: the previous
chunk is already 56 action tokens in the model's own vocabulary, so it can be
spliced into the prompt.  This script measures what one training step of that
model costs, on the real code path:

1. one ``predict_action_batch`` on a dummy observation yields the exact
   ``forward_inputs`` RLinf's actor trains on (prompt ids, mask, pixels,
   action tokens);
2. the prompt is lengthened by the 56 action tokens just produced -- the
   "previous command" -- inserted before the trailing space token that
   ``default_forward`` asserts on;
3. LoRA (RLinf's own target-module list, rank 32) + gradient checkpointing,
   then ``default_forward(compute_logprobs=True)``, ``-logprob.mean()``
   backward and an AdamW step, timing and recording peak memory.

Each (batch size, prompt variant) runs in isolation; an OOM is recorded, not
fatal.  The pre-registered gate (PREREG_wave3.md, E5) reads the
``prev_chunk`` variant: proceed iff batch >= 4 peaks below 22 GB.

Example::

    python experiments/efference_probe/vram_probe.py \
        --model-path /abs/path/to/Openvla-oft-SFT-libero-goal-traj1 \
        --out experiments/efference_probe/data/e5_vram.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

EXPERIMENTS = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTS.parent
for path in (REPO_ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

GATE_BATCH = 4
GATE_GB = 22.0
PREV_CHUNK_TOKENS = 56


def _dummy_obs(batch: int, task: str):
    import torch

    generator = torch.Generator().manual_seed(0)
    return {
        "main_images": torch.randint(
            0, 256, (batch, 256, 256, 3), dtype=torch.uint8, generator=generator
        ),
        "wrist_images": None,
        "states": torch.zeros(batch, 8),
        "task_descriptions": [task] * batch,
    }


def _with_previous_chunk(forward_inputs: dict, extra: int) -> dict:
    """Insert ``extra`` previous-command tokens before the trailing space."""
    import torch

    if extra == 0:
        return dict(forward_inputs)
    ids = forward_inputs["input_ids"]
    mask = forward_inputs["attention_mask"]
    previous = forward_inputs["action_tokens"].reshape(ids.shape[0], -1)[:, :extra]
    previous = previous.to(device=ids.device, dtype=ids.dtype)
    out = dict(forward_inputs)
    out["input_ids"] = torch.cat([ids[:, :-1], previous, ids[:, -1:]], dim=1)
    out["attention_mask"] = torch.cat(
        [mask[:, :-1], torch.ones_like(previous, dtype=mask.dtype), mask[:, -1:]],
        dim=1,
    )
    return out


def _enable_checkpointing(model) -> str:
    kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    try:
        model.gradient_checkpointing_enable(**kwargs)
        return "model"
    except (AttributeError, ValueError, TypeError):
        language_model = getattr(model, "language_model", None)
        if language_model is None:
            language_model = model.get_base_model().language_model
        language_model.gradient_checkpointing_enable(**kwargs)
        return "language_model"


def measure(model, optimizer, inputs: dict, temperature: float) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = model(
        forward_inputs={k: v.clone() for k, v in inputs.items()},
        compute_logprobs=True,
        temperature=temperature,
        top_k=-1,
        use_cache=False,
    )
    loss = -result["logprobs"].float().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "step_seconds": time.time() - started,
        "seq_len": int(inputs["input_ids"].shape[1]),
        "loss": float(loss.item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--unnorm-key", default="libero_goal_no_noops")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    from omegaconf import open_dict

    from efference_probe.config import RunConfig
    from efference_probe.harness import build_model_config
    from rlinf.models import get_model

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    run_cfg = RunConfig()
    run_cfg.model.model_path = args.model_path
    run_cfg.model.unnorm_key = args.unnorm_key
    model_cfg = build_model_config(run_cfg)
    with open_dict(model_cfg):
        model_cfg.is_lora = True
        model_cfg.lora_rank = args.lora_rank
        model_cfg.lora_path = None
    model = get_model(model_cfg).cuda()
    checkpointing = "off" if args.no_checkpointing else _enable_checkpointing(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)

    report: dict[str, Any] = {
        "model_path": args.model_path,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "lora_rank": args.lora_rank,
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "gradient_checkpointing": checkpointing,
        "weights_gb_after_load": torch.cuda.memory_allocated() / 1024**3,
        "runs": [],
    }
    for batch in args.batch_sizes:
        model.eval()
        with torch.no_grad():
            _, result = model.predict_action_batch(
                env_obs=_dummy_obs(batch, "put the bowl on the plate"),
                do_sample=True,
                temperature=args.temperature,
                top_k=-1,
                top_p=1.0,
            )
        base_inputs = dict(result["forward_inputs"])
        model.train()
        for variant, extra in (("baseline", 0), ("prev_chunk", PREV_CHUNK_TOKENS)):
            entry: dict[str, Any] = {"batch": batch, "variant": variant}
            try:
                entry.update(
                    measure(
                        model,
                        optimizer,
                        _with_previous_chunk(base_inputs, extra),
                        args.temperature,
                    )
                )
            except torch.cuda.OutOfMemoryError as error:
                entry["oom"] = True
                entry["error"] = str(error).splitlines()[0]
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            report["runs"].append(entry)
            print(json.dumps(entry), flush=True)

    gate_runs = [
        r
        for r in report["runs"]
        if r["variant"] == "prev_chunk"
        and r["batch"] >= GATE_BATCH
        and not r.get("oom")
    ]
    report["gate"] = {
        "rule": f"prev_chunk, batch >= {GATE_BATCH}, peak reserved < {GATE_GB} GB",
        "passed": any(r["peak_reserved_gb"] < GATE_GB for r in gate_runs),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

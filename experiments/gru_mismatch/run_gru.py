#!/usr/bin/env python
"""Line A, experiment 1: does keeping the command make mismatch readable?

Trains the exp3/exp4 GRU (faithful port), freezes it, runs the swap / mirror /
freeze hijack arms, probes each arm, and aggregates across seeds.  CPU only;
each (interface, seed) job is single-threaded, so ``--jobs`` scales linearly.

Example::

    # the gate: reproduce exp4's ~56.5% trailing recall at the source seed
    python experiments/gru_mismatch/run_gru.py --interfaces trace_abs --seeds 42 \
        --transforms --out experiments/gru_mismatch/results/repro

    # the pre-registered run (PREREG_wave3.md, E1)
    python experiments/gru_mismatch/run_gru.py --jobs 4 \
        --out experiments/gru_mismatch/results/e1
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gru_mismatch.aggregate import write_summary  # noqa: E402

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_INTERFACES = ("none", "trace_abs", "trace_signed", "command")
DEFAULT_TRANSFORMS = ("swap", "mirror", "freeze")


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one (interface, seed), then collect and probe every arm."""
    # One BLAS thread per job: these are many tiny ops, and threading them
    # costs more than it buys (see efference_probe/README.md, "Runtime").
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)

    from gru_mismatch.analyze import analyze_arm
    from gru_mismatch.mismatch import MismatchConfig, collect
    from gru_mismatch.port import ProtocolConfig, train

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "log.txt", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    protocol = ProtocolConfig(
        interface=job["interface"], seed=job["seed"], scale=job["scale"]
    )
    mismatch = MismatchConfig(steps=max(1, job["phase3_steps"] // job["scale"]))
    started = time.time()
    trained = train(protocol)
    report: dict[str, Any] = {
        "protocol": protocol.to_dict(),
        "mismatch": mismatch.to_dict(),
        "trailing_recall_pct": trained.trailing_recall,
        "trailing_correct": trained.trailing_correct,
        "trailing_total": trained.trailing_total,
        "train_seconds": trained.wall_seconds,
        "arms": {},
    }
    logging.info(
        "trained %s seed %d: trailing recall %.1f%% (%d/%d) in %.0fs",
        protocol.interface,
        protocol.seed,
        trained.trailing_recall,
        trained.trailing_correct,
        trained.trailing_total,
        trained.wall_seconds,
    )

    tables = []
    for transform in job["transforms"]:
        arm = collect(trained, transform, mismatch)
        table, summary = analyze_arm(
            arm,
            warmup=mismatch.warmup,
            min_clean_gap=mismatch.min_clean_gap,
            gamma=protocol.gamma,
            n_permutations=job["permutations"],
            family=job["family"],
        )
        if not table.empty:
            table.insert(0, "transform", transform)
            tables.append(table)
        summary["n_hijacks"] = int((arm.steps["label"] == "HIJACK").sum())
        report["arms"][transform] = summary
        logging.info("%s arm: %s", transform, json.dumps(summary, default=str))
        if job["save_arrays"]:
            arm.steps.to_parquet(out_dir / f"steps_{transform}.parquet", index=False)
            np.savez_compressed(
                out_dir / f"hidden_{transform}.npz",
                h_multi=arm.h_multi.astype(np.float16),
                h_gru=arm.h_gru.astype(np.float16),
            )

    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            out_dir / "probe_results.csv", index=False
        )
    report["total_seconds"] = time.time() - started
    (out_dir / "run.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    root.removeHandler(handler)
    handler.close()
    return {
        "interface": protocol.interface,
        "seed": protocol.seed,
        "trailing_recall_pct": trained.trailing_recall,
        "seconds": report["total_seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="results root")
    parser.add_argument("--interfaces", nargs="*", default=list(DEFAULT_INTERFACES))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=list(DEFAULT_TRANSFORMS),
        help="phase-3 arms; pass none (empty) to train and report recall only",
    )
    parser.add_argument("--phase3-steps", type=int, default=40_000)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--family", default="linear", choices=["linear", "mlp"])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="the source's --quick: every step count divided by 5",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="also write per-step scalars and float16 hidden states per arm",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "interface": interface,
            "seed": seed,
            "scale": 5 if args.quick else 1,
            "transforms": list(args.transforms),
            "phase3_steps": args.phase3_steps,
            "permutations": args.permutations,
            "family": args.family,
            "save_arrays": args.save_arrays,
            "out_dir": str(out_root / interface / f"seed{seed}"),
        }
        for interface in args.interfaces
        for seed in args.seeds
    ]
    logging.info("%d jobs on %d workers -> %s", len(jobs), args.jobs, out_root)
    if args.jobs > 1:
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.jobs) as pool:
            finished = list(pool.imap_unordered(run_job, jobs))
    else:
        finished = [run_job(job) for job in jobs]
    for item in sorted(finished, key=lambda r: (r["interface"], r["seed"])):
        logging.info(
            "%s seed %d: trailing recall %.1f%% (%.0fs)",
            item["interface"],
            item["seed"],
            item["trailing_recall_pct"],
            item["seconds"],
        )
    summary = write_summary(out_root)
    logging.info("wrote %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

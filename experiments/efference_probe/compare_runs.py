#!/usr/bin/env python
"""Compare probe/behaviour results across checkpoints (wave 3, E2 and E5).

Example::

    python experiments/efference_probe/compare_runs.py \
        --run sft:none=experiments/efference_probe/data/e2_sft \
        --run rl:none=experiments/efference_probe/data/e2_rl \
        --baseline sft:none \
        --out experiments/efference_probe/data/e2_compare

Each run must already have ``run_probes.py --stage main`` output.  Labels of
the form ``objective:interface`` are also laid out as the interface x
objective grid; any other label works too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from efference_probe.compare import compare, load_run_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="repeatable; e.g. rl:none=data/e2_rl",
    )
    parser.add_argument("--baseline", default=None, help="label to difference against")
    parser.add_argument(
        "--analysis",
        default="analysis",
        help="analysis subdirectory to read (analysis_mlp for the MLP ladder)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run must look like LABEL=RUN_DIR, got {spec!r}")
        label, run_dir = spec.split("=", 1)
        runs.append(load_run_summary(label, run_dir, args.analysis))
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        parser.error(f"duplicate labels: {labels}")
    if args.baseline is not None and args.baseline not in labels:
        parser.error(f"--baseline {args.baseline!r} is not one of {labels}")
    report = compare(
        runs, baseline=args.baseline, out_dir=Path(args.out), seed=args.seed
    )
    print(report["markdown"])
    print(f"wrote {Path(args.out) / 'compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

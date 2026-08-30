#!/usr/bin/env python
"""Run the probe ladder, controls, E1 and E3 over a collected run.

Example::

    python experiments/efference_probe/run_probes.py \
        --run experiments/efference_probe/data/main01 \
        --stage main
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

import pandas as pd  # noqa: E402

from efference_probe import analysis, figures  # noqa: E402
from efference_probe.datasets import step_index_report  # noqa: E402

# S1 runs only the checks that decide whether scaling up is worth the GPU time.
STAGE_PROBES = {
    "pilot": ["P0", "P1", "P2", "P2r", "C_cmd", "C_dstates"],
    "main": ["P0", "P1", "P2", "P2r", "P3", "P4", "C_cmd", "C_dstates"],
    "all": [spec.name for spec in analysis.PROBE_SPECS],
}


def _best_hidden_config(results: pd.DataFrame) -> tuple[int, str]:
    """Pick the (layer, pool) where P1/P3/P4 are most decodable.

    Selected on the probes themselves, so the chosen cell is a *reported*
    maximum over 27 configurations, not an independent estimate.  F1 shows the
    whole sweep for exactly this reason.
    """
    hidden = results[results["name"].isin(["P1", "P3", "P4"])].dropna(subset=["layer"])
    if hidden.empty:
        return 0, "act_mean"
    grouped = hidden.groupby(["layer", "pool"])["balanced_acc_mean"].mean()
    layer, pool = grouped.idxmax()
    return int(layer), str(pool)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="collection run directory")
    parser.add_argument(
        "--out", default=None, help="output directory (default: <run>/analysis)"
    )
    parser.add_argument("--stage", choices=sorted(STAGE_PROBES), default="main")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--pools", nargs="*", default=None)
    parser.add_argument("--probes", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--block-scaling", choices=["none", "sqrt_dim"], default="none")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip per-fold C selection (uses the largest C); for quick sweeps",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--cache-gb", type=float, default=8.0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run = analysis.load_run(args.run)
    out_dir = Path(args.out) if args.out else Path(args.run) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = analysis.AnalysisConfig(
        layers=args.layers,
        pools=args.pools,
        probes=args.probes or STAGE_PROBES[args.stage],
        n_splits=args.n_splits,
        seed=args.seed,
        select_c=not args.fast,
        block_scaling=args.block_scaling,
        max_cache_gb=args.cache_gb,
    )

    samples = analysis.make_samples(run, config)
    if samples.empty:
        print("no probe samples were constructed; check the hijack schedule")
        return 1
    logging.info(
        "probe samples: %d (%d positive)", len(samples), int((samples["y"] == 1).sum())
    )

    results = analysis.run_ladder(run, samples, config)
    frame = analysis.results_to_frame(results) if results else pd.DataFrame()
    layer, pool = _best_hidden_config(frame) if not frame.empty else (0, "act_mean")
    logging.info("best hidden config: layer=%s pool=%s", layer, pool)

    extra: dict = {"best_hidden_config": {"layer": layer, "pool": pool}}
    extra["e3"] = analysis.run_e3(run, samples)

    if not args.no_controls:
        global_samples = analysis.make_samples(run, config, negatives="global")
        extra["global_negatives_step_index"] = json.loads(
            step_index_report(global_samples).to_json()
        )
        global_results = analysis.run_ladder(
            run,
            global_samples,
            analysis.AnalysisConfig(
                layers=[layer],
                pools=[pool],
                probes=["P1", "P3", "P4"],
                n_splits=args.n_splits,
                seed=args.seed,
                select_c=not args.fast,
                block_scaling=args.block_scaling,
                max_cache_gb=args.cache_gb,
            ),
        )
        analysis.results_to_frame(global_results).to_csv(
            out_dir / "global_negatives_results.csv", index=False
        )

        per_transform = analysis.run_per_transform(run, samples, config, layer, pool)
        analysis.results_to_frame(per_transform).to_csv(
            out_dir / "per_transform_results.csv", index=False
        )

        cross_task = analysis.run_cross_task(run, samples, config, layer, pool)
        cross_task.to_csv(out_dir / "cross_task_results.csv", index=False)

        e1 = analysis.run_e1(run, config)
        e1.to_csv(out_dir / "e1_readout.csv", index=False)

        extra["undo_alignment"] = analysis.undo_alignment(run, samples)
    else:
        cross_task, e1 = pd.DataFrame(), pd.DataFrame()

    card = figures.run_card(run.config, run.manifest, samples)
    card.to_csv(out_dir / "run_card.csv", index=False)

    summary = analysis.summarize(run, samples, results, out_dir, extra=extra)
    print(f"wrote {summary}")

    if not args.no_figures and not frame.empty:
        figures.figure_layer_depth(
            frame, out_dir / "F1_layer_depth.png", pool=pool, n_samples=len(samples)
        )
        figures.figure_ladder(
            frame,
            out_dir / "F2_ladder.png",
            layer=layer,
            pool=pool,
            n_samples=len(samples),
        )
        if not cross_task.empty:
            figures.figure_cross_task(
                frame, cross_task, out_dir / "F3_cross_task.png", layer=layer, pool=pool
            )
        if not e1.empty:
            figures.figure_e1(e1, out_dir / "F4_e1_readout.png")
        success = extra.get("e3", {}).get("success_rate", {})
        note = (
            f"success: probe {success.get('probe_episodes', {}).get('success_rate', float('nan')):.2f} "
            f"vs clean {success.get('clean_episodes', {}).get('success_rate', float('nan')):.2f}"
            if success.get("probe_episodes") and success.get("clean_episodes")
            else ""
        )
        figures.figure_e3(run.calls, samples, out_dir / "F5_e3_behavior.png", note)
        print(f"wrote figures to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

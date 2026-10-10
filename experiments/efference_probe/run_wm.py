#!/usr/bin/env python
"""Wave 3, line B: the Wan world model as an external comparator.

Subcommands::

    # E3 gate: does the WM's prediction move with the action?
    python experiments/efference_probe/run_wm.py sensitivity \
        --run experiments/efference_probe/data/e4_wm_swap \
        --wm-ckpt /path/to/RLinf-Wan-LIBERO-Goal --out <dir>

    # E4: score hijacks by command-vs-outcome distance (one run per transform)
    python experiments/efference_probe/run_wm.py comparator \
        --run experiments/efference_probe/data/e4_wm_freeze \
        --wm-ckpt /path/to/RLinf-Wan-LIBERO-Goal --out <dir>

    # sanity: are our LIBERO frames oriented like the WM's training frames?
    python experiments/efference_probe/run_wm.py check-orientation \
        --run experiments/efference_probe/data/e4_wm_swap \
        --wm-ckpt /path/to/RLinf-Wan-LIBERO-Goal

``--backend toy`` swaps the WM for a numpy stand-in, so the whole pipeline
can be exercised without a GPU.  Rows are appended to ``rows.csv`` as they
are scored and a rerun skips finished queries, so a killed job resumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from efference_probe import analysis  # noqa: E402
from efference_probe.datasets import load_run  # noqa: E402
from efference_probe.wm_data import (  # noqa: E402
    ChunkTable,
    FrameStore,
    build_query,
    calls_with_frames,
    detect_context_mode,
)
from efference_probe.wm_scores import (  # noqa: E402
    comparator_row,
    e3_summary,
    e4_summary,
    prepare,
    sensitivity_row,
)

logger = logging.getLogger("run_wm")


def make_backend(args: argparse.Namespace):
    if args.backend == "toy":
        from efference_probe.wm_model import ToyWorldModel

        return ToyWorldModel()
    if not args.wm_ckpt:
        raise SystemExit("--wm-ckpt is required with --backend wan")
    from efference_probe.wm_model import WanWorldModel

    return WanWorldModel(
        args.wm_ckpt, device=args.device, num_inference_steps=args.inference_steps
    )


class RowLog:
    """Append-only CSV of scored rows, keyed for resuming."""

    def __init__(self, path: Path, key_columns: tuple[str, ...]) -> None:
        self.path = path
        self.key_columns = key_columns
        self.done: set[tuple] = set()
        if path.is_file():
            existing = pd.read_csv(path)
            self.done = {
                tuple(int(v) if isinstance(v, (np.integer, int)) else v for v in row)
                for row in existing[list(key_columns)].itertuples(index=False)
            }

    def key(self, row: dict[str, Any]) -> tuple:
        return tuple(row[c] for c in self.key_columns)

    def append(self, row: dict[str, Any]) -> None:
        frame = pd.DataFrame([row])
        frame.to_csv(self.path, mode="a", header=not self.path.is_file(), index=False)
        self.done.add(self.key(row))

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.is_file() else pd.DataFrame()


def _resolve_mode(run, requested: str) -> str:
    return detect_context_mode(run) if requested == "auto" else requested


def _save_predictions(
    out_dir: Path, tag: str, predictions: dict[str, np.ndarray]
) -> None:
    directory = out_dir / "predictions"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(
        directory / f"{tag}.npz",
        **{
            name.replace("@", "_at_"): (prepare(frames) * 255).astype(np.uint8)
            for name, frames in predictions.items()
        },
    )


# -- E3 ------------------------------------------------------------------------


def run_sensitivity(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = []
    modes = set()
    sources = {}
    for run_dir in args.run:
        run = load_run(run_dir)
        mode = _resolve_mode(run, args.context_mode)
        modes.add(mode)
        frames = FrameStore(run.run_dir)
        calls = calls_with_frames(run, frames, mode)
        calls = calls[(calls["label"] == "SELF") & (calls["call_idx"] >= 1)]
        tag = run.config.get("run_id", Path(run_dir).name)
        sources[tag] = (run, frames, ChunkTable(run), mode)
        pools.append(calls[["episode_id", "call_idx"]].assign(run_id=tag))
        logger.info("%s: %d eligible SELF calls (%s context)", tag, len(calls), mode)
    if len(modes) > 1:
        raise SystemExit(
            f"runs mix context modes {sorted(modes)}; score them separately"
        )
    mode = modes.pop()
    pool = pd.concat(pools, ignore_index=True)
    if pool.empty:
        raise SystemExit("no eligible calls with frames on disk")
    rng = np.random.default_rng(args.seed)
    chosen = pool.iloc[
        rng.choice(len(pool), size=min(args.n_queries, len(pool)), replace=False)
    ]
    chosen = chosen.sort_values(["run_id", "episode_id", "call_idx"])

    backend = make_backend(args)
    log = RowLog(out_dir / "rows.csv", ("run_id", "episode_id", "call_idx"))
    started = time.time()
    for position, (_, item) in enumerate(chosen.iterrows()):
        key = (item["run_id"], int(item["episode_id"]), int(item["call_idx"]))
        if key in log.done:
            continue
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            logger.info("time budget reached after %d queries", position)
            break
        run, frames, table, _ = sources[item["run_id"]]
        query = build_query(
            table,
            frames,
            key[1],
            key[2],
            mode,
            n_alternatives=args.alternatives,
            rng=np.random.default_rng([args.seed, key[1], key[2]]),
            with_transforms=True,
        )
        predictions = backend.predict_candidates(
            query.reference,
            query.context,
            query.condition,
            query.candidates,
            seed=0,
            extra_seed=1,
        )
        prepared = {name: prepare(frames_) for name, frames_ in predictions.items()}
        actual = prepare(query.actual) if query.actual is not None else None
        row = {
            "run_id": key[0],
            "episode_id": key[1],
            "call_idx": key[2],
            "task_id": query.task_id,
        }
        row.update(sensitivity_row(prepared, prepare(query.context[-1]), actual))
        log.append(row)
        if args.save_predictions:
            _save_predictions(
                out_dir, f"{key[0]}_ep{key[1]:05d}_c{key[2]:04d}", predictions
            )
        if (position + 1) % 10 == 0:
            logger.info(
                "E3: %d/%d queries (%.0fs)",
                position + 1,
                len(chosen),
                time.time() - started,
            )

    rows = log.read()
    summary = e3_summary(rows, mode)
    summary["runs"] = list(args.run)
    summary["alternatives"] = args.alternatives
    _write_summary(out_dir, "E3 — world-model action sensitivity", summary)
    return 0


# -- E4 ------------------------------------------------------------------------


def run_comparator(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = load_run(args.run[0])
    if len(args.run) > 1:
        raise SystemExit("comparator scores one run (one transform) at a time")
    mode = _resolve_mode(run, args.context_mode)
    if mode != "substeps" and not args.allow_static:
        raise SystemExit(
            "E4 needs a run collected with frames.substeps (e4_wm_*.yaml); "
            "pass --allow-static for an exploratory pass on per-call frames"
        )
    arm = ((run.config.get("config") or {}).get("hijack") or {}).get("transform", "")
    frames = FrameStore(run.run_dir)
    table = ChunkTable(run)
    available = calls_with_frames(run, frames, mode)
    have = set(
        zip(available["episode_id"].astype(int), available["call_idx"].astype(int))
    )
    samples = analysis.make_samples(run, analysis.AnalysisConfig())
    samples = samples[
        [
            (int(e), int(p)) in have
            for e, p in zip(samples["episode_id"], samples["prev_call"])
        ]
    ]
    if samples.empty:
        raise SystemExit("no probe samples with frames on disk")
    logger.info(
        "%s: %d samples (%d positive), %s context",
        arm,
        len(samples),
        int(samples["y"].sum()),
        mode,
    )
    entropy = {
        (int(e), int(c)): float(h)
        for e, c, h in zip(
            run.calls["episode_id"], run.calls["call_idx"], run.calls["entropy_mean"]
        )
    }

    backend = make_backend(args)
    log = RowLog(out_dir / "rows.csv", ("episode_id", "prev_call"))
    started = time.time()
    for position, (_, sample) in enumerate(samples.iterrows()):
        episode_id, prev_call = int(sample["episode_id"]), int(sample["prev_call"])
        if (episode_id, prev_call) in log.done:
            continue
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            logger.info("time budget reached after %d samples", position)
            break
        query = build_query(
            table,
            frames,
            episode_id,
            prev_call,
            mode,
            n_alternatives=args.alternatives,
            rng=np.random.default_rng([args.seed, episode_id, prev_call]),
        )
        candidates = {k: v for k, v in query.candidates.items() if k != "exec"}
        predictions = backend.predict_candidates(
            query.reference, query.context, query.condition, candidates, seed=0
        )
        prepared = {name: prepare(f) for name, f in predictions.items()}
        row = {
            "episode_id": episode_id,
            "prev_call": prev_call,
            "cur_call": int(sample["cur_call"]),
            "task_id": query.task_id,
            "y": int(sample["y"]),
            "pair_id": int(sample.get("pair_id", -1)),
            "transform": arm,
            "label": query.label,
        }
        row.update(
            comparator_row(prepared, prepare(query.actual), prepare(query.context[-1]))
        )
        row["policy_entropy"] = entropy.get(
            (episode_id, int(sample["cur_call"])), np.nan
        )
        log.append(row)
        if args.save_predictions:
            _save_predictions(
                out_dir, f"ep{episode_id:05d}_c{prev_call:04d}", predictions
            )
        if (position + 1) % 10 == 0:
            logger.info(
                "E4: %d/%d samples (%.0fs)",
                position + 1,
                len(samples),
                time.time() - started,
            )

    rows = log.read()
    summary = {
        "run": args.run[0],
        "context_mode": mode,
        "alternatives": args.alternatives,
        "per_transform": e4_summary(rows, n_bootstrap=args.bootstrap, seed=args.seed),
    }
    _write_summary(out_dir, f"E4 — world model as comparator ({arm})", summary)
    return 0


# -- orientation ---------------------------------------------------------------


def run_check_orientation(args: argparse.Namespace) -> int:
    """Compare our reset frames with the WM dataset's initial frames.

    Initial states differ, so this uses the nearest dataset frame per reset
    frame; a 180-degree mismatch shows up as a clearly lower distance for the
    rotated version.
    """
    if not args.wm_ckpt:
        raise SystemExit("--wm-ckpt is required (its dataset/ holds the WM's frames)")
    dataset_dir = Path(args.wm_ckpt) / "dataset"
    files = sorted(p for p in dataset_dir.glob("*.npy") if not p.stem.endswith("_kir"))
    if not files:
        raise SystemExit(f"no trajectory npy files under {dataset_dir}")
    rng = np.random.default_rng(args.seed)
    files = [
        files[i] for i in rng.choice(len(files), min(200, len(files)), replace=False)
    ]
    dataset_frames = []
    for path in files:
        trajectory = np.load(path, allow_pickle=True)
        first = trajectory[0]
        image = first.get("image") if isinstance(first, dict) else None
        if image is None and isinstance(first, dict):
            image = next(
                (v for v in first.values() if getattr(v, "ndim", 0) == 3), None
            )
        if image is not None:
            dataset_frames.append(prepare(np.asarray(image, dtype=np.uint8)))
    if not dataset_frames:
        raise SystemExit("could not find an image in the dataset frames")
    dataset = np.stack(dataset_frames)

    run = load_run(args.run[0])
    frames = FrameStore(run.run_dir)
    episodes = sorted(run.calls["episode_id"].unique())[:50]
    ours = [
        prepare(frames.call_frame(int(e), 0))
        for e in episodes
        if frames.has_call_frame(int(e), 0)
    ]
    if not ours:
        raise SystemExit("run has no call-0 frames")

    def nearest(frame: np.ndarray) -> float:
        return float(np.min(np.mean((dataset - frame[None]) ** 2, axis=(1, 2, 3))))

    as_is = float(np.mean([nearest(f) for f in ours]))
    rotated = float(np.mean([nearest(f[::-1, ::-1]) for f in ours]))
    report = {
        "n_dataset_frames": int(len(dataset)),
        "n_run_frames": len(ours),
        "mean_nearest_mse_as_is": as_is,
        "mean_nearest_mse_rotated_180": rotated,
        "verdict": "orientation matches"
        if as_is < rotated
        else "ROTATED: frames do not match the WM's",
    }
    print(json.dumps(report, indent=2))
    return 0 if as_is < rotated else 1


# -- shared --------------------------------------------------------------------


def _write_summary(out_dir: Path, title: str, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {title}",
        "",
        "```json",
        json.dumps(summary, indent=2, default=str),
        "```",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sensitivity", "comparator", "check-orientation"):
        p = sub.add_parser(name)
        p.add_argument("--run", nargs="+", required=True)
        p.add_argument("--wm-ckpt", default=None)
        p.add_argument("--seed", type=int, default=0)
        if name == "check-orientation":
            continue
        p.add_argument("--out", required=True)
        p.add_argument("--backend", choices=("wan", "toy"), default="wan")
        p.add_argument("--device", default="cuda")
        p.add_argument("--inference-steps", type=int, default=5)
        p.add_argument("--alternatives", type=int, default=6)
        p.add_argument(
            "--context-mode", choices=("auto", "substeps", "static"), default="auto"
        )
        p.add_argument("--max-hours", type=float, default=0.0)
        p.add_argument("--save-predictions", action="store_true")
        if name == "sensitivity":
            p.add_argument("--n-queries", type=int, default=300)
        else:
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--allow-static", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "sensitivity":
        return run_sensitivity(args)
    if args.command == "comparator":
        return run_comparator(args)
    return run_check_orientation(args)


if __name__ == "__main__":
    raise SystemExit(main())

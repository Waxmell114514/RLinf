"""X6 fault benchmark readout: success, time to success and the reaction trace.

For every run (one model) and every fault of its sweep:

- **success** with an episode bootstrap CI (B = 2000, seed 0);
- **RMST** in sim steps: the restricted mean steps-to-success at the 320-step
  budget (a failure counts 320), so architectures with different chunk
  lengths compare on one scale;
- **drop**: clean success minus this fault's success, with a paired-by-state
  bootstrap CI (the sweep runs the same reset states under every fault);
- **reaction trace**: the mean commanded x (env action space) at each call,
  which shows whether the policy pushes against a bias and whether it ever
  adapts.

Reads only ``manifest.json`` and ``calls.parquet``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

B = 2000
SEED = 0
STEP_BUDGET = 320


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    episodes = pd.DataFrame(manifest["episodes"])
    calls = pd.read_parquet(
        run_dir / "calls.parquet",
        columns=["episode_id", "call_idx", "reset_state_id", "task_id", "a_cmd_env"],
    )
    chunk = len(np.asarray(calls["a_cmd_env"].iloc[0])) // 7
    states = calls.groupby("episode_id")["reset_state_id"].first()
    episodes["reset_state_id"] = episodes["episode_id"].map(states)
    return episodes, calls, chunk


def steps_to_success(episodes: pd.DataFrame, chunk: int) -> np.ndarray:
    first = episodes["first_success_call"].to_numpy()
    success = episodes["success"].astype(bool).to_numpy()
    return np.where(success, np.minimum((first + 1) * chunk, STEP_BUDGET), STEP_BUDGET)


def _ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    idx = rng.integers(0, len(values), size=(B, len(values)))
    low, high = _ci(values[idx].mean(axis=1))
    return {"point": float(values.mean()), "ci_low": low, "ci_high": high}


def paired_drop(
    clean: pd.Series, faulted: pd.Series, rng: np.random.Generator
) -> dict[str, float]:
    """clean - faulted success over the reset states both ran."""
    shared = clean.index.intersection(faulted.index)
    diff = clean.loc[shared].to_numpy(float) - faulted.loc[shared].to_numpy(float)
    out = bootstrap_mean(diff, rng)
    out["n_states"] = int(len(shared))
    return out


def reaction_trace(calls: pd.DataFrame, episodes: pd.DataFrame, fault: str) -> list:
    """Mean commanded x per call index (env action space), over episodes."""
    ids = set(episodes.loc[episodes["fault"] == fault, "episode_id"])
    rows = calls[calls["episode_id"].isin(ids)]
    x = rows["a_cmd_env"].map(
        lambda a: float(np.asarray(a).reshape(-1, 7)[:, 0].mean())
    )
    return [float(v) for v in x.groupby(rows["call_idx"]).mean().to_numpy()]


def readout(run_dir: Path) -> dict[str, Any]:
    episodes, calls, chunk = load_run(run_dir)
    rng = np.random.default_rng(SEED)
    episodes["steps"] = steps_to_success(episodes, chunk)
    key = ["task_id", "reset_state_id"]
    clean = (
        episodes[episodes["fault"] == "clean"].set_index(key)["success"].astype(float)
    )
    out: dict[str, Any] = {"run": run_dir.name, "chunk": chunk, "faults": {}}
    for fault, group in episodes.groupby("fault", sort=False):
        success = group["success"].astype(float).to_numpy()
        entry = {
            "n": int(len(group)),
            "success": bootstrap_mean(success, rng),
            "rmst_steps": bootstrap_mean(group["steps"].to_numpy(float), rng),
            "reaction_x": reaction_trace(calls, episodes, fault),
        }
        if fault != "clean" and len(clean):
            entry["drop"] = paired_drop(
                clean, group.set_index(key)["success"].astype(float), rng
            )
        out["faults"][fault] = entry
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    results = [readout(run) for run in args.runs]
    text = json.dumps(results, indent=2)
    if args.out:
        args.out.write_text(text)
    for r in results:
        for fault, e in r["faults"].items():
            s = e["success"]
            print(
                f"{r['run']:24s} {fault:12s} n={e['n']:4d} "
                f"success {s['point']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] "
                f"rmst {e['rmst_steps']['point']:.0f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

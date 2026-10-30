"""X6 stage 0, step 1: can a command–outcome history identify an actuation fault?

Pre-registered in ``logbook.md`` ("X6 STAGE-0 PRE-REGISTRATION").  Descriptive:
it sets the history length K of the X6-EC efference-copy input.  Faults are
injected offline into logged SELF transitions: each transition keeps its real
out-of-fold forward-model residual (so the noise is the policy's own, heavy
tails included) and gains the fault's predicted effect on the state change.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

KS = (1, 2, 4, 8)
STEPS = 8  # sim steps per chunk
Z_IDENTIFIED = 3.0
K_CAP = 4


def load_transitions(pattern: str) -> pd.DataFrame:
    """SELF, non-post-success transitions of every run matching ``pattern``."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        calls = pd.read_parquet(
            path,
            columns=[
                "episode_id",
                "call_idx",
                "task_id",
                "label",
                "a_cmd_env",
                "states_before",
                "states_after",
                "post_success",
            ],
        )
        calls["episode_id"] = (
            Path(path).parent.name + ":" + calls["episode_id"].astype(str)
        )
        frames.append(calls)
    calls = pd.concat(frames, ignore_index=True)
    keep = (calls["label"] == "SELF") & ~calls["post_success"].astype(bool)
    return calls[keep].sort_values(["episode_id", "call_idx"]).reset_index(drop=True)


def design(calls: pd.DataFrame):
    before = np.stack(calls["states_before"]).astype(np.float64)
    after = np.stack(calls["states_after"]).astype(np.float64)
    commands = (
        np.stack(calls["a_cmd_env"]).astype(np.float64).reshape(len(calls), STEPS, -1)
    )
    tasks = pd.get_dummies(calls["task_id"]).to_numpy(np.float64)
    inputs = np.concatenate([before, commands.reshape(len(calls), -1), tasks], axis=1)
    return before, after - before, commands, tasks, inputs


def oof_residuals(
    inputs: np.ndarray, delta: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    predicted = np.zeros_like(delta)
    for train, test in GroupKFold(5).split(inputs, delta, groups):
        model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(inputs[train], delta[train])
        predicted[test] = model.predict(inputs[test])
    return delta - predicted


def axis_gains(
    before: np.ndarray, delta: np.ndarray, summed: np.ndarray, tasks: np.ndarray
) -> np.ndarray:
    """Per-axis OLS slope of delta pos on the chunk's summed command (m/unit)."""
    gains = []
    for axis in range(3):
        x = np.concatenate([summed[:, [axis]], before, tasks], axis=1)
        coef, *_ = np.linalg.lstsq(x, delta[:, axis], rcond=None)
        gains.append(coef[0])
    return np.asarray(gains)


def _episodes(groups: np.ndarray, k: int) -> list[np.ndarray]:
    """Row indices of each episode's first k transitions (episodes with >= k)."""
    out = []
    for _, rows in pd.Series(np.arange(len(groups))).groupby(groups):
        if len(rows) >= k:
            out.append(rows.to_numpy()[:k])
    return out


def bias_z(residual: np.ndarray, rows: np.ndarray, shift: float, sd: float) -> float:
    r = residual[rows] + shift
    return float(r.mean() / (sd / np.sqrt(len(rows))))


def slope_z(
    residual: np.ndarray, rows: np.ndarray, effect: np.ndarray, sd: float
) -> float:
    """z of the least-squares slope of (residual + effect) on ``effect``."""
    u = effect[rows]
    r = residual[rows] + u
    ss = float((u**2).sum())
    if ss <= 0:
        return 0.0
    return float((r * u).sum() / ss / (sd / np.sqrt(ss)))


def tables(calls: pd.DataFrame) -> dict[str, Any]:
    groups = calls["episode_id"].to_numpy()
    before, delta, commands, tasks, inputs = design(calls)
    residual = oof_residuals(inputs, delta[:, :3], groups)
    sd = residual.std(axis=0)
    summed = commands.sum(axis=1)
    gain = axis_gains(before, delta[:, :3], summed, tasks)
    out: dict[str, Any] = {
        "n_transitions": int(len(calls)),
        "n_episodes": int(len(set(groups))),
        "residual_sd_cm": (100 * sd).round(3).tolist(),
        "gain_cm_per_unit": (100 * gain).round(3).tolist(),
        "faults": {},
    }

    def summarise(zs: list[float]) -> dict[str, float]:
        zs = np.asarray(zs)
        return {
            "median_z": float(np.median(zs)),
            "share_z_ge_3": float(np.mean(zs >= Z_IDENTIFIED)),
            "n_episodes": int(len(zs)),
        }

    for k in KS:
        episodes = _episodes(groups, k)
        entry: dict[str, Any] = {}
        for b in (0.15, 0.30):
            shift = STEPS * b * gain[0]
            entry[f"bias_x_{b}"] = summarise(
                [bias_z(residual[:, 0], rows, shift, sd[0]) for rows in episodes]
            )
        inversion = -2 * gain[0] * summed[:, 0]
        entry["invert_x"] = summarise(
            [slope_z(residual[:, 0], rows, inversion, sd[0]) for rows in episodes]
        )
        for g in (0.6, 0.4):
            zs = []
            for rows in episodes:
                # Gain loss on x and y: pool the two axes' evidence.
                zx = slope_z(
                    residual[:, 0], rows, -(1 - g) * gain[0] * summed[:, 0], sd[0]
                )
                zy = slope_z(
                    residual[:, 1], rows, -(1 - g) * gain[1] * summed[:, 1], sd[1]
                )
                zs.append((zx + zy) / np.sqrt(2))
            entry[f"gain_{g}"] = summarise(zs)
        out["faults"][f"K={k}"] = entry

    previous = calls.groupby("episode_id")["call_idx"].shift(1)
    consecutive = (calls["call_idx"] - previous) == 1
    lagged = pd.DataFrame(summed).groupby(groups).shift(1).to_numpy()
    mask = consecutive.to_numpy()
    out["copycat_corr_xyz"] = [
        float(np.corrcoef(summed[mask, axis], lagged[mask, axis])[0, 1])
        for axis in range(3)
    ]
    out["history_K"] = history_length(out["faults"])
    return out


def history_length(faults: dict[str, Any]) -> Optional[int]:
    """Smallest K at which bias 0.30 and inversion are identified (cap 4)."""
    for k in KS:
        if k > K_CAP:
            break
        entry = faults[f"K={k}"]
        if (
            entry["bias_x_0.3"]["median_z"] >= Z_IDENTIFIED
            and entry["invert_x"]["median_z"] >= Z_IDENTIFIED
        ):
            return k
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--x4-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = {
        ckpt: tables(
            load_transitions(str(args.x4_root / f"x4_{ckpt}_mirror_s*/calls.parquet"))
        )
        for ckpt in ("grpo", "sft")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

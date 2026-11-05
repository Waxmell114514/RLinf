"""Readout of Amendment 8 (PREREG_X6.md): the frame estimator, closed loop.

R0' (estimator - F on clean >= -0.05, CI_low >= -0.10) and RE' (estimator - F
on the fault >= 0.5 * G_f, CI_low > 0) per cell.  G_f = F clean - F fault.  The
rotation cells pair with ``x6_fault_rot`` (F under rotation), the others with
``x6_fault_oft``.  The estimator run may be split over several run dirs (a
part restricted to a task subset); ``--part DIR:TASKS`` keeps only those tasks
of that part, so a part that died mid-task contributes only finished tasks.

Diagnostics from the logged ``est_A``: the first call whose estimate departs
from the identity, and the Frobenius error of the last estimate per episode
against the injected map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import x6_ec_report as rep
import x6_fault_report

CELLS = ["clean", "bias_x:0.3", "invert_x", "rotate:45", "rotate:90"]


def true_map(cell: str) -> np.ndarray:
    if cell.startswith("rotate:"):
        t = np.deg2rad(float(cell.split(":")[1]))
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    if cell == "invert_x":
        return np.diag([-1.0, 1.0])
    return np.eye(2)


def load_parts(parts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, calls = [], []
    for n, part in enumerate(parts):
        path, _, tasks = part.partition(":")
        frame, _call, _chunk = x6_fault_report.load_run(Path(path))
        call = pd.read_parquet(
            Path(path) / "calls.parquet",
            columns=["episode_id", "task_id", "call_idx", "fault", "est_A"],
        )
        # Episode ids restart in every part; keep them apart.
        frame["episode_id"] = f"{n}:" + frame["episode_id"].astype(str)
        call["episode_id"] = f"{n}:" + call["episode_id"].astype(str)
        if tasks:
            keep = [int(t) for t in tasks.split(",")]
            frame = frame[frame["task_id"].isin(keep)]
            call = call[call["task_id"].isin(keep)]
        frames.append(frame)
        calls.append(call)
    return pd.concat(frames, ignore_index=True), pd.concat(calls, ignore_index=True)


def diagnostics(calls: pd.DataFrame, cell: str) -> dict:
    rows = calls[calls["fault"] == cell].sort_values(["episode_id", "call_idx"])
    truth = true_map(cell)
    first, err = [], []
    for _, ep in rows.groupby("episode_id"):
        maps = np.stack([np.asarray(a, float).reshape(2, 2) for a in ep["est_A"]])
        moved = np.where(np.abs(maps - np.eye(2)).sum(axis=(1, 2)) > 1e-9)[0]
        first.append(int(ep["call_idx"].iloc[moved[0]]) if len(moved) else -1)
        err.append(float(np.linalg.norm(maps[-1] - truth)))
    first = np.asarray(first)
    hit = first >= 0
    return {
        "episodes": int(len(first)),
        "frac_moved": float(hit.mean()),
        "first_call_mean": float(first[hit].mean()) if hit.any() else None,
        "first_call_median": float(np.median(first[hit])) if hit.any() else None,
        "last_A_err_mean": float(np.mean(err)),
        "last_A_err_median": float(np.median(err)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True)
    ap.add_argument("--f-run", default="data/x6/x6_fault_oft")
    ap.add_argument("--f-rot-run", default="data/x6/x6_fault_rot")
    ap.add_argument("--oracle-run", default="data/x6/x6_oracle_bound")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(rep.SEED)
    est, calls = load_parts(args.part)
    f = rep.episodes(Path(args.f_run))
    f_rot = rep.episodes(Path(args.f_rot_run))
    oracle = rep.episodes(Path(args.oracle_run))
    f_clean = rep.success_by_pair(f, "clean")

    out: dict = {"n_episodes": {}, "success": {}, "cells": {}}
    for cell in CELLS:
        s = rep.success_by_pair(est, cell)
        out["n_episodes"][cell] = int(len(s))
        out["success"][cell] = float(s.mean())
    clean = rep.paired_difference(rep.success_by_pair(est, "clean"), f_clean, rng)
    out["R0p"] = {
        "est_minus_F_clean": clean,
        "passed": bool(clean["point"] >= -0.05 and clean["ci_low"] >= -0.10),
    }
    for cell in CELLS[1:]:
        ref = f_rot if cell.startswith("rotate:") else f
        f_cell = rep.success_by_pair(ref, cell)
        g_f = float(f_clean.mean() - f_cell.mean())
        d = rep.paired_difference(rep.success_by_pair(est, cell), f_cell, rng)
        entry = {
            "F_success": float(f_cell.mean()),
            "G_f": g_f,
            "est_minus_F": d,
            "RE_p": bool(d["point"] >= 0.5 * g_f and d["ci_low"] > 0),
            "diagnostics": diagnostics(calls, cell),
        }
        if cell in set(oracle["fault"]):
            o = rep.success_by_pair(oracle, cell)
            entry["oracle_success"] = float(o.mean())
            entry["est_minus_oracle"] = rep.paired_difference(
                rep.success_by_pair(est, cell), o, rng
            )
        out["cells"][cell] = entry
    out["diagnostics_clean"] = diagnostics(calls, "clean")
    rot = [out["cells"][c]["RE_p"] for c in ("rotate:45", "rotate:90")]
    out["framing_stands"] = bool(out["R0p"]["passed"] and all(rot))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""X6 readout: Rule M (Part A) and Rules E, N, L, 0 (Part B), as in PREREG_X6.md.

    python x6_ec_report.py --arm F=data/x6/x6_fault_oft --arm A=data/x6/x6_ec_eval_A \\
        --arm B=... --arm C=... --arm D=... [--part-a data/x6/x6_fault_pi0 ...]

Every arm's run must hold the cells it is scored on (F: clean, bias_x:0.3 and
invert_x, from Part A's OFT run; A: all three; B, C, D: the two fault cells).
Differences between arms are bootstrapped over the (task, reset state) pairs
both ran (B = 2000, seed 0, 95% percentile CI).  Reads only ``manifest.json``
and ``calls.parquet``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import x6_fault_report  # noqa: E402

B = 2000
SEED = 0
CLEAN = "clean"
FAMILIES = {"bias": "bias_x:0.3", "inversion": "invert_x"}
RULE_M_MIN_CLEAN = 0.40
KEY = ["task_id", "reset_state_id"]


def episodes(run_dir: Path) -> pd.DataFrame:
    frame, _calls, _chunk = x6_fault_report.load_run(run_dir)
    return frame


def success_by_pair(frame: pd.DataFrame, cell: str) -> pd.Series:
    rows = frame[frame["fault"] == cell]
    if rows.empty:
        raise ValueError(f"no {cell!r} episodes")
    if rows.duplicated(KEY).any():
        raise ValueError(f"{cell!r} repeats a (task, reset state) pair")
    return rows.set_index(KEY)["success"].astype(float)


def paired_difference(
    first: pd.Series, second: pd.Series, rng: np.random.Generator
) -> dict[str, float]:
    """mean(first - second) over shared pairs, with its bootstrap CI."""
    shared = first.index.intersection(second.index)
    if not len(shared):
        raise ValueError("the two runs share no (task, reset state) pair")
    diff = first.loc[shared].to_numpy() - second.loc[shared].to_numpy()
    draws = diff[rng.integers(0, len(diff), size=(B, len(diff)))].mean(axis=1)
    return {
        "point": float(diff.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_pairs": int(len(shared)),
    }


def rule_m(run_dir: Path) -> dict[str, Any]:
    """Part A: does this memoryless model fail bias_x 0.30 and invert_x?"""
    readout = x6_fault_report.readout(run_dir)
    faults = readout["faults"]
    clean = faults[CLEAN]["success"]["point"]
    out: dict[str, Any] = {"run": readout["run"], "clean": clean, "cells": {}}
    if clean < RULE_M_MIN_CLEAN:
        out["verdict"] = "descriptive (clean success below 0.40)"
        return out
    passes = []
    for family, cell in FAMILIES.items():
        entry = faults[cell]
        ok = entry["success"]["point"] <= 0.5 * clean and entry["drop"]["ci_low"] > 0
        out["cells"][family] = {
            "success": entry["success"],
            "drop": entry["drop"],
            "fails_as_predicted": bool(ok),
        }
        passes.append(ok)
    out["verdict"] = "fails as predicted" if all(passes) else "counterexample to Rule M"
    return out


def collection(
    run_dir: Path, expected_episodes: Optional[int] = None
) -> dict[str, Any]:
    """S3's readout: episodes logged, and success by fault family x compensation.

    Compensated faulted episodes should succeed about as often as clean ones
    (the oracle is exact inside the action bounds); uncompensated ones should
    fail as the frozen policy does.
    """
    from efference_probe.faults import parse as parse_fault

    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    frame = pd.DataFrame(manifest["episodes"])
    family = frame["fault"].map(lambda spec: parse_fault(spec).family)
    family = family.replace({"invert_x": "inversion", "invert_y": "inversion"})
    key = family + np.where(frame["compensated"], " compensated", "")
    out: dict[str, Any] = {
        "episodes": int(len(frame)),
        "calls": int(manifest.get("n_calls", 0)),
        "stop_reason": manifest.get("stop_reason"),
        "by_group": {
            str(group): {
                "n": int(len(rows)),
                "success": float(rows["success"].astype(float).mean()),
            }
            for group, rows in frame.groupby(key)
        },
    }
    if expected_episodes:
        out["logged_fraction"] = len(frame) / expected_episodes
        out["logged_gate"] = bool(len(frame) >= 0.95 * expected_episodes)
    return out


def part_b(arms: dict[str, Path]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    frames = {arm: episodes(path) for arm, path in arms.items()}

    def s(arm: str, cell: str) -> pd.Series:
        return success_by_pair(frames[arm], cell)

    out: dict[str, Any] = {"success": {}, "families": {}}
    for arm, frame in frames.items():
        out["success"][arm] = {
            cell: float(group["success"].astype(float).mean())
            for cell, group in frame.groupby("fault")
        }
    clean = paired_difference(s("A", CLEAN), s("F", CLEAN), rng)
    rule_0 = clean["point"] >= -0.05 and clean["ci_low"] >= -0.10
    out["rule_0"] = {"A_minus_F_clean": clean, "passed": bool(rule_0)}
    for family, cell in FAMILIES.items():
        g_f = float(s("F", CLEAN).mean() - s("F", cell).mean())
        a_f = paired_difference(s("A", cell), s("F", cell), rng)
        a_b = paired_difference(s("A", cell), s("B", cell), rng)
        a_c = paired_difference(s("A", cell), s("C", cell), rng)
        d_b = paired_difference(s("D", cell), s("B", cell), rng)
        rule_e = a_f["point"] >= 0.5 * g_f and a_f["ci_low"] > 0
        rule_n = all(d["point"] >= 0.15 and d["ci_low"] > 0 for d in (a_b, a_c))
        rule_l = (
            abs(d_b["point"]) <= 0.05
            and d_b["ci_low"] >= -0.10
            and d_b["ci_high"] <= 0.10
        )
        entry = {
            "cell": cell,
            "G_f": g_f,
            "A_minus_F": a_f,
            "A_minus_B": a_b,
            "A_minus_C": a_c,
            "D_minus_B": d_b,
            "rule_E": bool(rule_e),
            "rule_N": bool(rule_n),
            "rule_L": bool(rule_l),
        }
        if "E" in frames:
            entry["E_minus_A"] = paired_difference(s("E", cell), s("A", cell), rng)
        entry["reading"] = reading(rule_e, rule_n, rule_0, rule_l, d_b["point"])
        out["families"][family] = entry
    both = all(f["rule_E"] and f["rule_N"] for f in out["families"].values()) and bool(
        rule_0
    )
    some = [
        family
        for family, f in out["families"].items()
        if f["rule_E"] and f["rule_N"] and rule_0
    ]
    out["headline"] = (
        "E, N and 0 on both families: the headline claim"
        if both
        else f"claim scoped to {some}"
        if some
        else "no family passes E, N and 0"
    )
    return out


def reading(rule_e: bool, rule_n: bool, rule_0: bool, rule_l: bool, d_b: float) -> str:
    """PREREG_X6.md section 3.5's readings, for one family."""
    if not rule_0:
        main = "the fine-tune damaged the policy (Rule 0 fails)"
    elif rule_e and rule_n:
        main = "the copy restores success (E and N)"
    elif rule_e:
        main = "fine-tuning under faults, not the copy, restores success"
    elif rule_n:
        main = "the copy is read, but not enough"
    else:
        main = "no effect"
    if rule_l:
        loss = "the loss without the channel is inert (L)"
    elif d_b > 0:
        loss = "L fails with D > B: the warning about forward-model losses is wrong"
    else:
        loss = "L fails with D < B: the loss hurts"
    return f"{main}; {loss}"


def _arm_pairs(values: list[str]) -> dict[str, Path]:
    arms = {}
    for value in values:
        name, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--arm takes NAME=RUN_DIR, got {value!r}")
        arms[name] = Path(path)
    return arms


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--part-a", nargs="*", type=Path, default=[])
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=1200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    arms = _arm_pairs(args.arm)
    result: dict[str, Any] = {}
    if args.collection:
        result["collection"] = collection(args.collection, args.expected_episodes)
    part_a = list(args.part_a) + ([arms["F"]] if "F" in arms else [])
    if part_a:
        result["rule_M"] = [rule_m(run) for run in part_a]
    missing = {"F", "A", "B", "C", "D"} - set(arms)
    if not missing:
        result["part_b"] = part_b(arms)
    elif args.arm:
        result["part_b"] = f"not scored: missing arms {sorted(missing)}"
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

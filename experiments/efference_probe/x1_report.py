"""X1 readout (PREREG_X1_X4.md, X1 "Pre-registered readouts").

    P=experiments/efference_probe/x1_report.py
    python $P --data-root D/x1 controls RUN...   # R2: C_joint, C_scene, C_obs probes
    python $P --data-root D/x1 resid RUN...      # R5: P1 residualised on C_obs
    python $P --data-root D/x1 increment RUN...  # R3: paired bootstrap of P3 - P1
    python $P --data-root D/x1 fixed-cell --e4-root D   # R4 on X1 and E4 pairs
    python $P --data-root D/x1 battery RUN...    # R7: the X0 analyses, unchanged
    python $P --data-root D/x1 report --e4-root D --out reports/x1_report.md

Primary family is linear (PREREG §1).  Every refit runs in this process; X1's
probe rounds also run here, so CSV and refit share one BLAS setting.
Outputs go to ``<data-root>/x1_readout/``.
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

from efference_probe import e4_common as common  # noqa: E402
from efference_probe import e4_report as report  # noqa: E402
from efference_probe import x0_followup as fu  # noqa: E402
from efference_probe import x0_mechanism as x0  # noqa: E402
from efference_probe.datasets import _delta_states, call_lookup  # noqa: E402
from efference_probe.probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    _outer_splits,
    run_probe,
)

PLAN_PATH = common.PROBE_DIR / "x1_plan.yaml"
FIXED_CELL = (12, "ctx_mean")  # named in PREREG_X1_X4.md before X1 data
DELTA = 0.04
CMD_BAND = (0.45, 0.55)
RUNGS = ("P1", "P3", "P4")
CONTROLS = ("C_state+d", "C_scene", "C_joint", "C_obs")
B, SEED = 2000, 0


def out_dir(data_root: Path) -> Path:
    path = data_root / "x1_readout"
    path.mkdir(parents=True, exist_ok=True)
    return path


def completed(data_root: Path, plan: dict[str, Any], stage: str = "s2") -> dict:
    """{planned run_id: completed attempt id} for the stage's runs."""
    out = {}
    for run in common.planned_runs(plan, stage):
        attempt = common.completed_attempt(data_root, run.run_id)
        if attempt:
            out[run.run_id] = attempt
    return out


# -- R2: the new observables controls -----------------------------------------


def _stack(rows: pd.DataFrame, column: str) -> np.ndarray:
    widths = rows[column].map(len)
    if widths.nunique() != 1:
        raise ValueError(
            f"{column} has varying widths {sorted(widths.unique())}; "
            "the X1 controls assume one scene per run (LIBERO-Goal)"
        )
    return np.stack(rows[column].map(np.asarray).to_list()).astype(np.float64)


def observables_features(run, samples: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each control's features for the hijacked (prev) call of every sample."""
    lookup = call_lookup(run.calls)
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    rows = run.calls.iloc[[lookup[(int(e), int(c))] for e, c in prev]]
    tasks = samples["task_id"].to_numpy()
    onehot = (tasks[:, None] == np.unique(tasks)[None, :]).astype(np.float64)
    joint_after = _stack(rows, "joint_pos_after")
    joint = [joint_after, joint_after - _stack(rows, "joint_pos_before"), onehot]
    objects_after = _stack(rows, "objects_after")
    scene = [objects_after, objects_after - _stack(rows, "objects_before"), onehot]
    state = np.concatenate(
        [
            x0._per_task_state(x0.states(run, prev, "states_after"), tasks),
            _delta_states(run.calls, lookup, prev),
        ],
        axis=1,
    )
    features = {
        "C_state+d": state,
        "C_scene": np.concatenate(scene, axis=1),
        "C_joint": np.concatenate(joint, axis=1),
    }
    features["C_obs"] = np.concatenate(
        [features["C_state+d"], *scene[:2], *joint[:2]], axis=1
    )
    return {k: v.astype(np.float32) for k, v in features.items()}


def controls(data_root: Path, run_id: str) -> dict[str, Any]:
    run, samples, _ = x0.load(data_root, run_id)
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    out = {"run": run_id}
    for name, x in observables_features(run, samples).items():
        result = run_probe(
            x, y, groups, name=name, blocks=[name], c_values=DEFAULT_C_VALUES
        )
        out[name] = {
            "bacc": result.balanced_acc_mean,
            "sd": result.balanced_acc_std,
            "folds": result.fold_balanced_acc,
            "n_features": int(x.shape[1]),
        }
    return out


def r2_beyond(numbers: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """X1-R2: a rung is beyond the full observables iff >= every twin + delta."""
    twins = {"C_dstates": numbers["C_dstates"]["bacc"]}
    twins.update({name: control[name]["bacc"] for name in CONTROLS})
    bar = max(twins.values()) + DELTA
    cmd = numbers["C_cmd"]["bacc"]
    cmd_ok = CMD_BAND[0] <= cmd <= CMD_BAND[1]
    beyond = [r for r in RUNGS if numbers[r]["bacc"] >= bar and cmd_ok]
    return {
        "twins": twins,
        "bar": bar,
        "c_cmd": cmd,
        "cmd_in_band": cmd_ok,
        "beyond": beyond,
    }


# -- R3: the efference increment ----------------------------------------------


def r3_reading(
    increments: dict[str, float],
    x1_grpo_swap: Optional[dict[str, Any]],
    e4_grpo: list[dict[str, Any]],
    n_planned: Optional[int] = None,
) -> str:
    """X1-R3.  Bootstraps are ``x0.a7_bootstrap`` results (P3-P1 entry).

    "In every X1 run" needs every planned run: a budget-cut wave stays
    unresolved rather than reading from the runs it happened to finish.
    """
    if n_planned is not None and len(increments) < n_planned:
        return f"unresolved ({len(increments)} of {n_planned} X1 runs)"
    if x1_grpo_swap is None or not x1_grpo_swap.get("reproduced"):
        return "unresolved (no reproduced bootstrap on x1_grpo_swap)"
    ci = x1_grpo_swap["P3-P1"]

    def positive(d):
        return d["point"] >= DELTA and d["ci_low"] > 0

    if (
        all(abs(v) < DELTA for v in increments.values())
        and ci["ci_low"] <= 0 <= ci["ci_high"]
    ):
        return "no efference increment on two pairs"
    if positive(ci) and any(
        positive(b["P3-P1"]) for b in e4_grpo if b.get("reproduced")
    ):
        return "RL produces an efference increment"
    return "unresolved"


# -- R4: the fixed cell ---------------------------------------------------------


def fixed_cell(data_root: Path, run_id: str, cell=FIXED_CELL) -> dict[str, Any]:
    """Out-of-fold linear P1 at a pre-named cell, kept per sample."""
    run, samples, store = x0.load(data_root, run_id)
    x, _ = x0.build_features(
        run, samples, ["h_cur"], layer=int(cell[0]), pool=str(cell[1]), store=store
    )
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    predicted, folds, chosen = x0.oof_predictions(x, y, groups, "linear")
    return {
        "run": run_id,
        "cell": f"L{cell[0]}/{cell[1]}",
        "fold_bacc": folds,
        "selected_c": chosen,
        "y": y,
        "predicted": predicted,
        "episodes": groups,
    }


def _pooled_bacc(y, predicted) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, predicted))


def _episode_draw(result: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    clusters = (
        pd.Series(np.arange(len(result["y"]))).groupby(result["episodes"]).indices
    )
    keys = list(clusters)
    picks = rng.integers(0, len(keys), len(keys))
    return np.concatenate([clusters[keys[i]] for i in picks])


def fixed_cell_difference(grpo: dict[str, Any], sft: dict[str, Any]) -> dict[str, Any]:
    """GRPO - SFT pooled OOF balanced accuracy; episodes resampled within each run."""
    rng = np.random.default_rng(SEED)
    point = _pooled_bacc(grpo["y"], grpo["predicted"]) - _pooled_bacc(
        sft["y"], sft["predicted"]
    )
    draws = []
    for _ in range(B):
        g, s = _episode_draw(grpo, rng), _episode_draw(sft, rng)
        draws.append(
            _pooled_bacc(grpo["y"][g], grpo["predicted"][g])
            - _pooled_bacc(sft["y"][s], sft["predicted"][s])
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(low), "ci_high": float(high)}


def r4_reading(x1: dict[str, float], e4: list[dict[str, float]]) -> str:
    def raises(d):
        return d["point"] >= DELTA and d["ci_low"] > 0

    if raises(x1) and all(raises(d) for d in e4):
        return "RL raises P1 at the fixed cell"
    if x1["ci_low"] <= 0 <= x1["ci_high"]:
        return "no RL effect at the fixed cell"
    return "mixed"


# -- R5: P1 residualised on the full observables --------------------------------


def residualised_p1(data_root: Path, run_id: str) -> dict[str, Any]:
    """F5 with C_obs's features as the residualiser (no label use)."""
    run, samples, store = x0.load(data_root, run_id)
    numbers = report._family_numbers(common.analysis_dir(run.run_dir, "linear"))
    row = x0.best_rows(x0.probe_frame(run.run_dir, "linear"))["P1"]
    layer, pool = int(row["layer"]), str(row["pool"])
    cur = list(zip(samples["episode_id"], samples["cur_call"]))
    hidden = np.asarray(
        store.vectors(cur, run.layer_index(layer), run.pool_index(pool)), np.float64
    )
    design = observables_features(run, samples)["C_obs"].astype(np.float64)
    design = np.concatenate([design, np.ones((len(design), 1))], axis=1)
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    folds = []
    for train, test in _outer_splits(hidden, y, groups, x0.N_SPLITS, SEED):
        coef = np.linalg.lstsq(design[train], hidden[train], rcond=None)[0]
        residual = (hidden - design @ coef).astype(np.float32)
        folds.append(
            fu._linear_score(
                residual[train], y[train], groups[train], residual[test], y[test]
            )
        )
    result = {
        "run": run_id,
        "cell": f"L{layer}/{pool}",
        "P1": float(numbers["P1"]["bacc"]),
        "floor": float(numbers["floor"]),
        "P1_resid": float(np.mean(folds)),
        "P1_resid_folds": folds,
    }
    return {**result, **fu.f5_reading(result)}


# -- R6: hijack cost ----------------------------------------------------------


def _episodes(data_root: Path, run_id: str) -> pd.DataFrame:
    manifest = common.read_json(data_root / run_id / "manifest.json") or {}
    frame = pd.DataFrame(manifest.get("episodes", []))
    return frame[
        ~frame.get("censored", pd.Series(False, index=frame.index)).astype(bool)
    ]


def _cost_draw(frame: pd.DataFrame, rng: np.random.Generator) -> float:
    clean = (
        frame.loc[~frame["is_probe_episode"].astype(bool), "success"]
        .astype(float)
        .to_numpy()
    )
    probe = (
        frame.loc[frame["is_probe_episode"].astype(bool), "success"]
        .astype(float)
        .to_numpy()
    )
    if rng is None:
        return float(clean.mean() - probe.mean())
    return float(
        rng.choice(clean, len(clean)).mean() - rng.choice(probe, len(probe)).mean()
    )


def hijack_cost(data_root: Path, sft_run: str, grpo_run: str) -> dict[str, Any]:
    """clean - probe success per run, and RL - SFT, episodes resampled in each."""
    rng = np.random.default_rng(SEED)
    sft, grpo = _episodes(data_root, sft_run), _episodes(data_root, grpo_run)
    draws = [_cost_draw(grpo, rng) - _cost_draw(sft, rng) for _ in range(B)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "sft_cost": _cost_draw(sft, None),
        "grpo_cost": _cost_draw(grpo, None),
        "rl_minus_sft": _cost_draw(grpo, None) - _cost_draw(sft, None),
        "ci_low": float(low),
        "ci_high": float(high),
    }


# -- R7: the X0 battery, unchanged code ---------------------------------------


def battery(data_root: Path, run_id: str, out: Path) -> dict[str, str]:
    """A2, A2b, F2, A3, F3 and F5 on one X1 run; A1 is read off its ladder."""
    done = {}
    a2 = x0.a2_outcome_readout(data_root, run_id, n_jobs=1)
    # x0_followup's F4 reads A2 from <data-root>/x0/, as in X0.
    a2.to_csv(x0.out_dir(data_root) / f"a2_{run_id}.csv", index=False)
    state_cell = x0.best_cell(a2[a2["target"] == "s_next"])
    delta_cell = x0.best_cell(a2[a2["target"] == "delta_from_next"])
    steps = {
        "a2b": lambda: x0.a2b_motion_beyond_state(data_root, run_id, delta_cell),
        "f2": lambda: fu.f2_residual(data_root, run_id, delta_cell),
        "a3": lambda: x0.a3_oracle(data_root, run_id, state_cell, delta_cell),
        "f3": lambda: fu.f3_oracle_decomposition(data_root, run_id, state_cell),
        "f5": lambda: fu.f5_residualised_p1(data_root, run_id),
    }
    for name, step in steps.items():
        path = out / f"{name}_{run_id}.json"
        path.write_text(json.dumps(step(), indent=2, default=_json))
        done[name] = str(path)
    return done


def battery_pair(data_root: Path, plan: dict[str, Any], out: Path) -> dict[str, Any]:
    """F4 on the X1 swap pair (needs both runs' battery A2)."""
    runs = completed(data_root, plan)
    result = fu.f4_state_rmse(data_root, ((runs["x1_sft_swap"], runs["x1_grpo_swap"]),))
    (out / "f4_x1_swap.json").write_text(json.dumps(result, indent=2, default=_json))
    return result


# E4's readings under each X0 rule (logbook, X0 and F1-F7 results), per
# checkpoint role, for the "replicates / does not" marks of X1-R7.
E4_REFERENCE = {
    "a1_within_delta": {"sft": True, "grpo": True},
    "a2_state_readable": {"sft": True, "grpo": True},  # (i) >= 0.5
    "a2_motion_premise_fails": {"sft": True, "grpo": True},  # (ii) >= 0.3
    "a2_m1_compatible": {"sft": True, "grpo": True},  # (iii) >= 0.5
    "f2_no_motion_trace": {"sft": True, "grpo": True},
    "a3_above_floor": {"sft": True, "grpo": True},  # Base-Lora 5/5, GRPO 3/4
    "f3_outcome_ok": {"sft": False, "grpo": True},
    "f3_command_ok": {"sft": False, "grpo": True},
    "f4_data_effect": {"pair": True},
}


def _read(path: Path) -> Optional[dict[str, Any]]:
    return json.loads(path.read_text()) if path.is_file() else None


def battery_readings(data_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Each X0 rule applied to each X1 run, marked against E4's reading."""
    out_path = out_dir(data_root)
    marks: dict[str, Any] = {}
    for planned, run_id in completed(data_root, plan).items():
        role = "grpo" if planned.startswith("x1_grpo") else "sft"
        flags: dict[str, Optional[bool]] = {}
        numbers = run_numbers(data_root, run_id).families.get("linear")
        if numbers:
            flags["a1_within_delta"] = (
                abs(numbers["P3"]["bacc"] - numbers["P1"]["bacc"]) < DELTA
            )
        a2_path = x0.out_dir(data_root) / f"a2_{run_id}.csv"
        if a2_path.is_file():
            best = pd.read_csv(a2_path).groupby("target")["r2_mean"].max()
            flags["a2_state_readable"] = bool(best["s_next"] >= 0.5)
            flags["a2_motion_premise_fails"] = bool(best["delta_from_next"] >= 0.3)
            flags["a2_m1_compatible"] = bool(best["delta_from_both"] >= 0.5)
        f2 = _read(out_path / f"f2_{run_id}.json")
        if f2:
            flags["f2_no_motion_trace"] = (
                f2["all"]["gain"] < 0.05 and f2["hijack"]["gain"] < 0.05
            )
        a3 = _read(out_path / f"a3_{run_id}.json")
        if a3:
            flags["a3_above_floor"] = a3["oracle_A"] >= a3["floor"] + DELTA
        f3 = _read(out_path / f"f3_{run_id}.json")
        if f3:
            flags["f3_outcome_ok"] = f3["true_a+decoded_d"] >= f3["p2r_csv"] - 0.10
            flags["f3_command_ok"] = f3["decoded_a+true_d"] >= f3["p2r_csv"] - 0.10
        marks[planned] = {
            name: {
                "x1": value,
                "e4": E4_REFERENCE[name][role],
                "replicates": value == E4_REFERENCE[name][role],
            }
            for name, value in flags.items()
        }
        f5 = _read(out_path / f"f5_{run_id}.json")
        if f5:
            marks[planned]["f5_reading"] = f5["reading"]
    f4 = _read(out_path / "f4_x1_swap.json")
    if f4:
        data_effect = all(
            r["reading"] == "data effect" for r in f4["readings"].values()
        )
        marks["pair"] = {
            "f4_data_effect": {
                "x1": data_effect,
                "e4": True,
                "replicates": data_effect,
            }
        }
    return marks


def main02_reproduction(numbers: report.RunNumbers) -> dict[str, Any]:
    """X1's SFT swap against main02's wave-2 ladder, each within 0.04."""
    wave2 = report.read_wave2_report("main02", "goal-traj1", "goal", "swap", 1)
    ours, theirs = numbers.families.get("linear"), wave2.families.get("linear")
    if not ours or not theirs:
        return {"status": "missing"}
    diffs = {r: ours[r]["bacc"] - theirs[r]["bacc"] for r in ("P1", "P3", "P4", "P2r")}
    return {
        "differences": diffs,
        "within_0.04": all(abs(d) <= 0.04 for d in diffs.values()),
    }


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return str(value)


# -- assembling the report ----------------------------------------------------


def run_numbers(data_root: Path, run_id: str) -> report.RunNumbers:
    return report.read_run(data_root / run_id, x0.checkpoint(run_id), "goal", "", None)


def readout(
    data_root: Path, plan: dict[str, Any], e4_root: Optional[Path] = None
) -> dict[str, Any]:
    """R1, R2, R3 and R5 from what is on disk; R4 and R6 are added by the CLI."""
    out: dict[str, Any] = {"runs": {}}
    readout_dir = out_dir(data_root)
    for planned, run_id in completed(data_root, plan).items():
        numbers = run_numbers(data_root, run_id)
        entry: dict[str, Any] = {"run_id": run_id}
        if "linear" in numbers.families:
            fam = numbers.families["linear"]
            entry["R1"] = report.ladder_verdict(numbers, "linear")
            entry["increment"] = fam["P3"]["bacc"] - fam["P1"]["bacc"]
            entry["d_int"] = fam["P4"]["bacc"] - fam["P1"]["bacc"]  # A1's other half
            control_path = readout_dir / f"controls_{planned}.json"
            if control_path.is_file():
                entry["R2"] = r2_beyond(fam, json.loads(control_path.read_text()))
        resid_path = readout_dir / f"resid_{planned}.json"
        if resid_path.is_file():
            entry["R5"] = json.loads(resid_path.read_text())
        out["runs"][planned] = entry
    increments = {k: e["increment"] for k, e in out["runs"].items() if "increment" in e}
    boot_path = readout_dir / "increment_x1_grpo_swap.json"
    x1_boot = json.loads(boot_path.read_text()) if boot_path.is_file() else None
    e4_boots = []
    if e4_root is not None:
        selected = x0.out_dir(e4_root) / "a7_linear_selected.csv"
        if selected.is_file():
            frame = pd.read_csv(selected)
            for _, row in frame[
                frame["run"].str.startswith("e4_grpo_goal_swap")
            ].iterrows():
                e4_boots.append(
                    {
                        "run": row["run"],
                        "reproduced": bool(row["reproduced"]),
                        "P3-P1": {
                            "point": row["P3-P1_point"],
                            "ci_low": row["P3-P1_ci_low"],
                            "ci_high": row["P3-P1_ci_high"],
                        },
                    }
                )
    out["R3"] = {
        "increments": increments,
        "x1_grpo_swap_bootstrap": x1_boot,
        "e4_grpo_bootstraps": e4_boots,
        "reading": r3_reading(
            increments, x1_boot, e4_boots, len(common.planned_runs(plan, "s2"))
        ),
    }
    swap = out["runs"].get("x1_sft_swap", {}).get("run_id")
    if swap:
        out["main02_reproduction"] = main02_reproduction(run_numbers(data_root, swap))
    battery_dir = readout_dir
    if any(battery_dir.glob("a3_*.json")):
        out["R7"] = battery_readings(data_root, plan)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("controls", "resid", "increment", "battery"):
        sub.add_parser(name).add_argument("runs", nargs="+")
    sub.add_parser("battery-pair")
    p = sub.add_parser("fixed-cell")
    p.add_argument("--e4-root", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--e4-root", type=Path)
    p.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    plan = common.load_plan(args.plan)
    root, out = args.data_root, out_dir(args.data_root)
    if args.command in ("controls", "resid", "increment", "battery"):
        runs = completed(root, plan)
        for run_id in args.runs:
            if args.command == "controls":
                result = controls(root, run_id)
            elif args.command == "resid":
                result = residualised_p1(root, run_id)
            elif args.command == "increment":
                result = x0.a7_bootstrap(root, run_id, "linear")
            else:
                result = battery(root, run_id, out)
            # Readouts are keyed by planned id; attempts (_r2) map back to it.
            planned = next((k for k, v in runs.items() if v == run_id), run_id)
            (out / f"{args.command}_{planned}.json").write_text(
                json.dumps(result, indent=2, default=_json)
            )
            print(json.dumps(result, indent=2, default=_json))
        return 0
    if args.command == "battery-pair":
        print(json.dumps(battery_pair(root, plan, out), indent=2, default=_json))
        return 0
    if args.command == "fixed-cell":
        runs = completed(root, plan)
        x1 = fixed_cell_difference(
            fixed_cell(root, runs["x1_grpo_swap"]),
            fixed_cell(root, runs["x1_sft_swap"]),
        )
        e4 = [
            fixed_cell_difference(
                fixed_cell(args.e4_root, f"e4_grpo_goal_swap_s{s}"),
                fixed_cell(args.e4_root, f"e4_sft_goal_swap_s{s}"),
            )
            for s in (1, 2)
        ]
        result = {
            "x1": x1,
            "e4_s1": e4[0],
            "e4_s2": e4[1],
            "reading": r4_reading(x1, e4),
        }
        (out / "fixed_cell.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0
    result = readout(root, plan, args.e4_root)
    runs = completed(root, plan)
    for arm in ("swap", "mirror", "freeze"):
        sft, grpo = runs.get(f"x1_sft_{arm}"), runs.get(f"x1_grpo_{arm}")
        if sft and grpo:
            result.setdefault("R6", {})[arm] = hijack_cost(root, sft, grpo)
    if (out / "fixed_cell.json").is_file():
        result["R4"] = json.loads((out / "fixed_cell.json").read_text())
    text = (
        "# X1 report\n\n```json\n"
        + json.dumps(result, indent=2, default=_json)
        + "\n```\n"
    )
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

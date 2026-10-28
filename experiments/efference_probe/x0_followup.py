"""X0 follow-up analyses F1-F7.

Pre-registered in ``logbook.md`` ("X0 FOLLOW-UP PRE-REGISTRATION F1-F7")
before any number was computed; the plan is ``reports/next_steps_fable_x0.md``
§5, Tier 0.  Exploratory: nothing here changes an R1-R12 or X0 verdict.  It
reuses ``x0_mechanism`` and the probe code and changes neither.

    P=experiments/efference_probe/x0_followup.py
    python $P --data-root D f1            # BLAS resolution (no fits)
    python $P --data-root D f6            # C_phase and the margin sign test
    python $P --data-root D f3 RUN...     # oracle decomposition
    python $P --data-root D f2 RUN...     # A2b residual test
    python $P --data-root D f4            # state readout RMSE, matched support
    python $P --data-root D f5 RUN...     # residualised P1
    python $P --data-root D f7 RUN...     # where P1's hits are

F3, F5 and F7 are compared with CSV numbers, so each run is computed in a
child process under its reproducing BLAS setting (``x0/a7_linear_selected.csv``).
Outputs go to ``<data-root>/x0/followup/``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import e4_common as common  # noqa: E402
from efference_probe import e4_report as report  # noqa: E402
from efference_probe import x0_mechanism as x0  # noqa: E402
from efference_probe.datasets import _delta_states, call_lookup  # noqa: E402
from efference_probe.probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    _make_classifier,
    _outer_splits,
    _select_c,
    _SharedRidge,
)

SEED, N_SPLITS = x0.SEED, x0.N_SPLITS
SWAP_RUNS = (
    "e4_sft_goal_swap_s1",
    "e4_sft_goal_swap_s2",
    "e4_sft_spatial_swap_s0",
    "e4_sft_object_swap_s0",
    "e4_sft_long_swap_s0_r2",
    "e4_grpo_goal_swap_s1",
    "e4_grpo_goal_swap_s2",
    "e4_grpo_spatial_swap_s0",
    "e4_grpo_object_swap_s0",
)
MATCHED = (  # (Base-Lora, GRPO) cells with the same suite and seed
    ("e4_sft_goal_swap_s1", "e4_grpo_goal_swap_s1"),
    ("e4_sft_goal_swap_s2", "e4_grpo_goal_swap_s2"),
    ("e4_sft_spatial_swap_s0", "e4_grpo_spatial_swap_s0"),
    ("e4_sft_object_swap_s0", "e4_grpo_object_swap_s0"),
)
# x0_mechanism.BLAS_PASSES as environment settings.
BLAS_ENV = {
    "avx2_1t": {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1"},
    "avx512_4t": {"OPENBLAS_CORETYPE": "SkylakeX", "OPENBLAS_NUM_THREADS": "4"},
    "avx512_1t": {"OPENBLAS_CORETYPE": "SkylakeX", "OPENBLAS_NUM_THREADS": "1"},
}
REFITS = ("f3", "f5", "f7")


def out_dir(data_root: Path) -> Path:
    path = x0.out_dir(data_root) / "followup"
    path.mkdir(parents=True, exist_ok=True)
    return path


def blas_setting(data_root: Path, run_id: str) -> Optional[str]:
    """The BLAS pass that reproduced this run's linear CSV in A7."""
    path = x0.out_dir(data_root) / "a7_linear_selected.csv"
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["run"] == run_id and row["reproduced"] == "True":
                return row["blas"]
    return None


def _linear_score(x_train, y_train, groups_train, x_test, y_test) -> float:
    """The ladder's linear probe on one outer fold: inner C selection, fit, score."""
    from sklearn.metrics import balanced_accuracy_score

    c = _select_c(x_train, y_train, groups_train, DEFAULT_C_VALUES, SEED, 2000)
    model = _make_classifier(c, 2000, None, "linear", SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return float(balanced_accuracy_score(y_test, model.predict(x_test)))


def _uniform_r2(y: np.ndarray, predicted: np.ndarray) -> float:
    from sklearn.metrics import r2_score

    return float(r2_score(y, predicted, multioutput="uniform_average"))


# -- F1 / F6: no fits ----------------------------------------------------------


def f1_blas_resolution(data_root: Path) -> pd.DataFrame:
    rows = []
    passes = x0.out_dir(data_root) / "a7_passes"
    for setting in x0.BLAS_PASSES:
        for path in sorted((passes / setting).glob("a7_*_linear.json")):
            result = json.loads(path.read_text())
            row = {"run": result["run"], "blas": setting}
            row["reproduced"] = result["reproduced"]
            for rung, check in result["checks"].items():
                row[f"{rung}_max_fold_diff"] = check["max_fold_diff"]
            row["P1-C_dstates"] = result["P1-C_dstates"]["point"]
            row["P3-P1"] = result["P3-P1"]["point"]
            rows.append(row)
    return pd.DataFrame(rows)


def f1_summary(frame: pd.DataFrame) -> dict[str, Any]:
    folds = frame[[c for c in frame.columns if c.endswith("_max_fold_diff")]]
    spread = frame.groupby("run")[["P1-C_dstates", "P3-P1"]].agg(np.ptp)
    return {
        "max_fold_diff": float(folds.to_numpy().max()),
        "max_spread_P1-C_dstates": float(spread["P1-C_dstates"].max()),
        "max_spread_P3-P1": float(spread["P3-P1"].max()),
        "runs_with_more_than_one_pass": int((frame.groupby("run").size() > 1).sum()),
    }


def f6_phase_and_sign(data_root: Path) -> dict[str, Any]:
    from scipy.stats import binomtest

    phase = []
    for run_id in x0.completed_runs(data_root):
        for family in ("linear", "mlp"):
            path = common.analysis_dir(data_root / run_id, family)
            if not (path / "probe_results.csv").is_file():
                continue
            numbers = report._family_numbers(path)
            if numbers.get("C_phase") is None:
                continue
            phase.append(
                {
                    "run": run_id,
                    "checkpoint": x0.checkpoint(run_id),
                    "family": family,
                    "C_phase": numbers["C_phase"]["bacc"],
                    "floor": numbers["floor"],
                }
            )
    margins = []
    for sft, grpo in MATCHED:
        pair = {}
        for run_id in (sft, grpo):
            numbers = report._family_numbers(
                common.analysis_dir(data_root / run_id, "linear")
            )
            pair[run_id] = numbers["P1"]["bacc"] - numbers["floor"]
        margins.append({"sft": sft, "grpo": grpo, **pair, "grpo_higher": None})
        margins[-1]["grpo_higher"] = bool(pair[grpo] > pair[sft])
    k = sum(m["grpo_higher"] for m in margins)
    test = binomtest(k, len(margins), 0.5, alternative="greater")
    return {
        "c_phase": phase,
        "margins": margins,
        "sign_test": {"k": k, "n": len(margins), "p_one_sided": float(test.pvalue)},
    }


# -- F3: oracle decomposition ---------------------------------------------------


def f3_oracle_decomposition(
    data_root: Path, run_id: str, state_cell: tuple[int, str]
) -> dict[str, Any]:
    """A3's pipeline with one side of the comparison decoded and the other true."""
    from sklearn.model_selection import GroupKFold

    run, samples, store = x0.load(data_root, run_id)
    e1 = pd.read_csv(common.analysis_dir(run.run_dir, "linear") / "e1_readout.csv")
    cmd_cell = x0.best_cell(e1)
    n_chunks = int(run.schema["num_action_chunks"])
    action_dim = int(run.schema["action_dim"])

    def h(pairs, cell):
        return store.vectors(pairs, run.layer_index(cell[0]), run.pool_index(cell[1]))

    pairs = x0.readout_pairs(run, max_calls=10**9)
    cur = list(zip(pairs["episode_id"], pairs["call"]))
    nxt = [(e, c + 1) for e, c in cur]
    d_groups = pairs["episode_id"].to_numpy()
    d_hcmd, d_hstate_cur, d_hstate_nxt = (
        h(cur, cmd_cell),
        h(cur, state_cell),
        h(nxt, state_cell),
    )
    d_cmd = x0.states(run, cur, "a_cmd_env")
    d_before = x0.states(run, cur, "states_before")
    d_after = x0.states(run, cur, "states_after")
    s_prev = list(zip(samples["episode_id"], samples["prev_call"]))
    s_cur = list(zip(samples["episode_id"], samples["cur_call"]))
    s_hcmd, s_hstate_prev, s_hstate_cur = (
        h(s_prev, cmd_cell),
        h(s_prev, state_cell),
        h(s_cur, state_cell),
    )
    true_cmd = x0.states(run, s_prev, "a_cmd_env").reshape(-1, n_chunks, action_dim)
    true_delta = _delta_states(run.calls, call_lookup(run.calls), s_prev)
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()

    def decode(held_out: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
        """A3's decoders; one side decoded, the other true."""
        fit = ~np.isin(d_groups, held_out)
        cmd = x0._Decoder(d_hcmd[fit], d_cmd[fit], d_groups[fit])
        state = x0._Decoder(
            np.concatenate([d_hstate_cur[fit], d_hstate_nxt[fit]]),
            np.concatenate([d_before[fit], d_after[fit]]),
            np.concatenate([d_groups[fit], d_groups[fit]]),
        )
        commanded = cmd.predict(s_hcmd[rows]).reshape(-1, n_chunks, action_dim)
        delta = x0.compose_delta(
            state.predict(s_hstate_prev[rows]), state.predict(s_hstate_cur[rows])
        )
        return {
            "true_a+decoded_d": x0.mismatch_from_arrays(true_cmd[rows], delta),
            "decoded_a+true_d": x0.mismatch_from_arrays(commanded, true_delta[rows]),
        }

    true_features = x0.mismatch_from_arrays(true_cmd, true_delta)
    keys = ("true_a+decoded_d", "decoded_a+true_d", "true_a+true_d")
    scores: dict[str, list[float]] = {k: [] for k in keys}
    folds = _outer_splits(np.zeros((len(y), 1)), y, groups, N_SPLITS, SEED)
    for fold, (train, test) in enumerate(folds):
        test_episodes = np.unique(groups[test])
        train_features: dict[str, np.ndarray] = {}
        inner = GroupKFold(n_splits=N_SPLITS).split(
            np.zeros((len(train), 1)), y[train], groups[train]
        )
        for _, inner_held in inner:
            held = np.concatenate([test_episodes, np.unique(groups[train[inner_held]])])
            for key, value in decode(held, train[inner_held]).items():
                if key not in train_features:
                    train_features[key] = np.zeros(
                        (len(train), value.shape[1]), np.float32
                    )
                train_features[key][inner_held] = value
        test_features = decode(test_episodes, test)
        train_features["true_a+true_d"] = true_features[train]
        test_features["true_a+true_d"] = true_features[test]
        for key in keys:
            scores[key].append(
                _linear_score(
                    train_features[key],
                    y[train],
                    groups[train],
                    test_features[key],
                    y[test],
                )
            )
        print(
            f"{run_id} F3 fold {fold + 1}/{N_SPLITS}: "
            + "  ".join(f"{k} {scores[k][-1]:.3f}" for k in keys),
            flush=True,
        )
    frame = x0.probe_frame(run.run_dir, "linear")
    p2r_folds = ast.literal_eval(
        frame.loc[frame["name"] == "P2r", "fold_balanced_acc"].iloc[0]
    )
    numbers = report._family_numbers(common.analysis_dir(run.run_dir, "linear"))
    return {
        "run": run_id,
        "checkpoint": x0.checkpoint(run_id),
        "command_cell": list(cmd_cell),
        "state_cell": list(state_cell),
        **{k: float(np.mean(v)) for k, v in scores.items()},
        "folds": scores,
        "p2r_csv": float(numbers["P2r"]["bacc"]),
        "p2r_reproduced": bool(
            np.max(np.abs(np.asarray(scores["true_a+true_d"]) - np.asarray(p2r_folds)))
            < 1e-6
        ),
        "floor": float(numbers["floor"]),
        "n": int(len(y)),
    }


def f3_reading(results: list[dict[str, Any]]) -> dict[str, Any]:
    sft = [r for r in results if r["checkpoint"] == "Base-Lora"]

    def ok(result, key):
        return result[key] >= result["p2r_csv"] - 0.10

    outcome = sum(ok(r, "true_a+decoded_d") for r in sft)
    command = sum(ok(r, "decoded_a+true_d") for r in sft)
    n, need = len(sft), 4  # ">= 4 of 5"
    if outcome >= need and command >= need:
        reading = "format supports the comparison"
    elif n - outcome >= need and command >= need:
        reading = "outcome decode is the bottleneck"
    elif n - command >= need and outcome >= need:
        reading = "command decode is the bottleneck"
    else:
        reading = "both decodes lossy"
    return {
        "n_base_lora": n,
        "outcome_ok": outcome,
        "command_ok": command,
        "reading": reading,
    }


# -- F2: the A2b residual test ---------------------------------------------------


def _knn_per_task(state, y, tasks, fit, rows, k: int = 10) -> np.ndarray:
    """A2b's S2: per-task k-NN on the state standardised on the fit rows."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    predicted = np.zeros((len(rows), y.shape[1]), dtype=np.float32)
    for task in np.unique(tasks[rows]):
        source = fit[tasks[fit] == task]
        target = np.flatnonzero(tasks[rows] == task)
        if len(source) == 0:
            source = fit
        scaler = StandardScaler().fit(state[source])
        model = KNeighborsRegressor(n_neighbors=min(k, len(source)))
        model.fit(scaler.transform(state[source]), y[source])
        predicted[target] = model.predict(scaler.transform(state[rows[target]]))
    return predicted


def f2_residual(data_root: Path, run_id: str, cell: tuple[int, str]) -> dict[str, Any]:
    """Does h(c+1) predict what the end-effector state leaves unexplained?"""
    from sklearn.model_selection import GroupKFold

    run, _, store = x0.load(data_root, run_id)
    pairs = x0.readout_pairs(run)
    cur = list(zip(pairs["episode_id"], pairs["call"]))
    nxt = [(e, c + 1) for e, c in cur]
    groups = pairs["episode_id"].to_numpy()
    lookup = call_lookup(run.calls)
    positions = np.array([lookup[(int(e), int(c))] for e, c in cur])
    hijack = run.calls["label"].to_numpy()[positions] == "HIJACK"
    tasks = run.calls["task_id"].to_numpy()[positions]
    delta = _delta_states(run.calls, lookup, cur).astype(np.float32)
    state = x0.states(run, cur, "states_after")
    hidden = store.vectors(nxt, run.layer_index(cell[0]), run.pool_index(cell[1]))
    hidden = np.asarray(hidden, np.float32)
    knn = np.zeros_like(delta)
    combined = np.zeros_like(delta)
    for train, test in GroupKFold(n_splits=N_SPLITS).split(hidden, delta, groups):
        knn_train = np.zeros((len(train), delta.shape[1]), np.float32)
        for inner_fit, inner_held in GroupKFold(n_splits=N_SPLITS).split(
            np.zeros((len(train), 1)), delta[train], groups[train]
        ):
            knn_train[inner_held] = _knn_per_task(
                state, delta, tasks, train[inner_fit], train[inner_held]
            )
        knn[test] = _knn_per_task(state, delta, tasks, train, test)
        residual = delta[train] - knn_train
        alpha = x0._ridge_alpha(hidden[train], residual, groups[train])
        correction = _SharedRidge(hidden[train], residual).predict(hidden[test], alpha)
        combined[test] = knn[test] + correction
    subsets = {"all": np.ones(len(delta), bool), "hijack": hijack}
    out = {"run": run_id, "checkpoint": x0.checkpoint(run_id), "cell": list(cell)}
    for name, rows in subsets.items():
        base, both = (
            _uniform_r2(delta[rows], knn[rows]),
            _uniform_r2(delta[rows], combined[rows]),
        )
        out[name] = {"S2": base, "S2+h": both, "gain": both - base}
    out["n"], out["n_hijack"] = int(len(delta)), int(hijack.sum())
    return out


def f2_reading(results: list[dict[str, Any]]) -> str:
    small = sum(r["all"]["gain"] < 0.05 and r["hijack"]["gain"] < 0.05 for r in results)
    large = sum(r["hijack"]["gain"] >= 0.05 for r in results)
    if small == len(results):
        return "no motion trace beyond the end-effector state"
    if large >= 3:
        return "motion trace beyond state"
    return "mixed"


# -- F4: state readout RMSE with matched support --------------------------------


def _oof_state(data_root: Path, run_id: str, cell: tuple[int, str]):
    run, _, store = x0.load(data_root, run_id)
    pairs = x0.readout_pairs(run)
    cur = list(zip(pairs["episode_id"], pairs["call"]))
    nxt = [(e, c + 1) for e, c in cur]
    lookup = call_lookup(run.calls)
    positions = np.array([lookup[(int(e), int(c))] for e, c in cur])
    hidden = store.vectors(nxt, run.layer_index(cell[0]), run.pool_index(cell[1]))
    target = x0.states(run, cur, "states_after")
    predicted = x0._oof_ridge(hidden, target, pairs["episode_id"].to_numpy())
    return {
        "state": target,
        "predicted": predicted,
        "tasks": run.calls["task_id"].to_numpy()[positions],
        "self": run.calls["label"].to_numpy()[positions] == "SELF",
    }


def matched_support(
    mine: np.ndarray,
    mine_tasks: np.ndarray,
    other: np.ndarray,
    other_tasks,
    k: int = 10,
) -> np.ndarray:
    """In support: my k-NN distance to the other's same-task calls is at most the
    median k-NN distance among the other's own same-task calls."""
    from sklearn.neighbors import NearestNeighbors

    inside = np.zeros(len(mine), dtype=bool)
    for task in np.unique(mine_tasks):
        ref = other[other_tasks == task]
        rows = np.flatnonzero(mine_tasks == task)
        if len(ref) <= k:
            continue
        index = NearestNeighbors(n_neighbors=k + 1).fit(ref)
        own = index.kneighbors(ref)[0][:, 1:].mean(axis=1)
        theirs = index.kneighbors(mine[rows], n_neighbors=k)[0].mean(axis=1)
        inside[rows] = theirs <= np.median(own)
    return inside


def _nrmse(truth, predicted, scale) -> float:
    rmse = np.sqrt(np.mean((truth - predicted) ** 2, axis=0))
    return float(np.mean(rmse / scale))


def f4_state_rmse(data_root: Path) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    pairs_out = []
    for sft, grpo in MATCHED[:2]:
        readouts = {}
        for run_id in (sft, grpo):
            a2 = pd.read_csv(x0.out_dir(data_root) / f"a2_{run_id}.csv")
            best = x0.best_cell(a2[a2["target"] == "s_next"])
            for name, cell in (("L0/ctx_mean", (0, "ctx_mean")), ("best", best)):
                readouts[(run_id, name)] = {**_oof_state(data_root, run_id, cell)}
                readouts[(run_id, name)]["cell"] = list(cell)
        for name in ("L0/ctx_mean", "best"):
            a, b = readouts[(sft, name)], readouts[(grpo, name)]
            pooled = np.concatenate([a["state"], b["state"]])
            mean, scale = pooled.mean(axis=0), pooled.std(axis=0)
            scale = np.where(scale > 0, scale, 1.0)
            za, zb = (a["state"] - mean) / scale, (b["state"] - mean) / scale
            in_a = matched_support(za, a["tasks"], zb, b["tasks"])
            in_b = matched_support(zb, b["tasks"], za, a["tasks"])
            row = {"pair": [sft, grpo], "cell": name}
            for label, r, inside in (("sft", a, in_a), ("grpo", b, in_b)):
                row[label] = {
                    "readout_cell": r["cell"],
                    "rmse_per_dim": np.sqrt(
                        np.mean((r["state"] - r["predicted"]) ** 2, axis=0)
                    ).tolist(),
                    "self_sd_per_dim": r["state"][r["self"]].std(axis=0).tolist(),
                    "nrmse_all": _nrmse(r["state"], r["predicted"], scale),
                    "nrmse_matched": _nrmse(
                        r["state"][inside], r["predicted"][inside], scale
                    ),
                    "n_matched": int(inside.sum()),
                    "n": int(len(inside)),
                }
            row["gap_all"] = row["sft"]["nrmse_all"] - row["grpo"]["nrmse_all"]
            row["gap_matched"] = (
                row["sft"]["nrmse_matched"] - row["grpo"]["nrmse_matched"]
            )
            pairs_out.append(row)
    for name in ("L0/ctx_mean", "best"):
        rows = [r for r in pairs_out if r["cell"] == name]
        if any(r["gap_all"] <= 0 for r in rows):
            reading = "no GRPO advantage on all calls in some pair"
        elif all(r["gap_matched"] <= 0.5 * r["gap_all"] for r in rows):
            reading = "data effect"
        elif all(r["gap_matched"] >= 0.8 * r["gap_all"] for r in rows):
            reading = "representation effect"
        else:
            reading = "mixed"
        cells[name] = {"reading": reading}
    return {"pairs": pairs_out, "readings": cells}


# -- F5: residualised P1 --------------------------------------------------------


def f5_residualised_p1(data_root: Path, run_id: str) -> dict[str, Any]:
    run, samples, store = x0.load(data_root, run_id)
    numbers = report._family_numbers(common.analysis_dir(run.run_dir, "linear"))
    row = x0.best_rows(x0.probe_frame(run.run_dir, "linear"))["P1"]
    layer, pool = int(row["layer"]), str(row["pool"])
    cur = list(zip(samples["episode_id"], samples["cur_call"]))
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    hidden = store.vectors(cur, run.layer_index(layer), run.pool_index(pool))
    hidden = np.asarray(hidden, np.float64)
    tasks = samples["task_id"].to_numpy()
    design = x0._per_task_state(x0.states(run, prev, "states_after"), tasks)
    design = np.concatenate([design, np.ones((len(design), 1))], axis=1).astype(
        np.float64
    )
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    folds = []
    for train, test in _outer_splits(hidden, y, groups, N_SPLITS, SEED):
        coef = np.linalg.lstsq(design[train], hidden[train], rcond=None)[0]
        residual = (hidden - design @ coef).astype(np.float32)
        folds.append(
            _linear_score(
                residual[train], y[train], groups[train], residual[test], y[test]
            )
        )
    p1, floor = float(numbers["P1"]["bacc"]), float(numbers["floor"])
    resid = float(np.mean(folds))
    if resid >= p1 - 0.02:
        reading = "state-orthogonal"
    elif resid <= floor + 0.02:
        reading = "state-explained"
    else:
        reading = "partial"
    return {
        "run": run_id,
        "checkpoint": x0.checkpoint(run_id),
        "cell": f"L{layer}/{pool}",
        "P1": p1,
        "floor": floor,
        "P1_resid": resid,
        "P1_resid_folds": folds,
        "reading": reading,
        "n": int(len(y)),
    }


# -- F7: where P1's hits are ----------------------------------------------------


def margin_shares(y: np.ndarray, predicted: np.ndarray, tasks: np.ndarray):
    from sklearn.metrics import balanced_accuracy_score

    rows = []
    for task in np.unique(tasks):
        mask = tasks == task
        if len(np.unique(y[mask])) < 2:
            bacc = float("nan")
        else:
            bacc = float(balanced_accuracy_score(y[mask], predicted[mask]))
        rows.append({"task_id": int(task), "n": int(mask.sum()), "bacc": bacc})
    frame = pd.DataFrame(rows)
    weight = frame["n"] * (frame["bacc"] - 0.5)
    frame["margin_share"] = weight / weight.sum()
    return frame


def f7_hits(data_root: Path, run_id: str) -> dict[str, Any]:
    from scipy.stats import spearmanr

    run, samples, store = x0.load(data_root, run_id)
    row = x0.best_rows(x0.probe_frame(run.run_dir, "linear"))["P1"]
    layer, pool = int(row["layer"]), str(row["pool"])
    x, _ = x0.build_features(
        run, samples, ["h_cur"], layer=layer, pool=pool, store=store
    )
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    tasks = samples["task_id"].to_numpy()
    predicted, folds, chosen = x0.oof_predictions(x, y, groups, "linear")
    expected = ast.literal_eval(row["fold_balanced_acc"])
    reproduced = bool(
        np.max(np.abs(np.asarray(folds) - np.asarray(expected))) < 1e-6
        and chosen == [float(c) for c in ast.literal_eval(row["selected_c"])]
    )
    shares = margin_shares(y, predicted, tasks)
    competence = common.derived_dir(data_root, run_id) / "competence.csv"
    flagged = []
    if competence.is_file():
        table = pd.read_csv(competence)
        flagged = (
            table.loc[table["flagged"].astype(bool), "task_id"].astype(int).tolist()
        )
    flagged_share = float(
        shares.loc[shares["task_id"].isin(flagged), "margin_share"].sum()
    )
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    executed = x0.states(run, prev, "a_exec")
    commanded = x0.states(run, prev, "a_cmd_env")
    n_chunks = int(run.schema["num_action_chunks"])
    action_dim = int(run.schema["action_dim"])
    strength = np.linalg.norm(executed - commanded, axis=1)
    gripper = (
        np.sign(executed.reshape(-1, n_chunks, action_dim)[:, :, action_dim - 1])
        != np.sign(commanded.reshape(-1, n_chunks, action_dim)[:, :, action_dim - 1])
    ).any(axis=1)
    hijack = y == 1
    correct = (predicted == y).astype(int)
    rho = spearmanr(correct[hijack], strength[hijack])
    return {
        "run": run_id,
        "checkpoint": x0.checkpoint(run_id),
        "cell": f"L{layer}/{pool}",
        "reproduced": reproduced,
        "per_task": shares.to_dict(orient="records"),
        "flagged_tasks": flagged,
        "flagged_margin_share": flagged_share,
        "flag": flagged_share > 0.5,
        "hijack_hit_vs_strength_spearman": float(rho.statistic),
        "hijack_hit_vs_strength_p": float(rho.pvalue),
        "hit_rate_gripper_disagree": float(correct[hijack & gripper].mean())
        if (hijack & gripper).any()
        else None,
        "n_gripper_disagree": int((hijack & gripper).sum()),
        "hit_rate_gripper_agree": float(correct[hijack & ~gripper].mean()),
        "n_gripper_agree": int((hijack & ~gripper).sum()),
    }


# -- CLI -----------------------------------------------------------------------


def _child(data_root: Path, analysis: str, run_id: str) -> int:
    """Re-run this script for one run under that run's reproducing BLAS setting."""
    setting = blas_setting(data_root, run_id)
    env = dict(os.environ, X0_FOLLOWUP_CHILD="1")
    if setting is None:
        print(f"{run_id}: no reproducing BLAS setting; skipped", flush=True)
        return 1
    env.update(BLAS_ENV[setting])
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"]
    command = [sys.executable, "-X", "utf8", "-W", "ignore", __file__]
    command += ["--data-root", str(data_root), analysis, run_id]
    print(f"{run_id}: {analysis} under {setting}", flush=True)
    return subprocess.call(command, env=env)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "analysis", choices=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "read"]
    )
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args(argv)
    root, out = args.data_root, out_dir(args.data_root)
    if args.analysis in REFITS and not os.environ.get("X0_FOLLOWUP_CHILD"):
        return max([_child(root, args.analysis, r) for r in args.runs] or [0])
    if args.analysis == "f1":
        frame = f1_blas_resolution(root)
        frame.to_csv(out / "f1_blas.csv", index=False)
        summary = f1_summary(frame)
        (out / "f1_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    elif args.analysis == "f6":
        result = f6_phase_and_sign(root)
        (out / "f6_phase_sign.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    elif args.analysis == "f4":
        result = f4_state_rmse(root)
        (out / "f4_state_rmse.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result["readings"], indent=2))
    elif args.analysis == "read":
        readings = {}
        f3 = [json.loads(p.read_text()) for p in sorted(out.glob("f3_*.json"))]
        if f3:
            readings["F3"] = f3_reading(f3)
        f2 = [json.loads(p.read_text()) for p in sorted(out.glob("f2_*.json"))]
        if f2:
            readings["F2"] = f2_reading(f2)
        print(json.dumps(readings, indent=2))
    else:
        for run_id in args.runs:
            path = out / f"{args.analysis}_{run_id}.json"
            if args.analysis == "f3":
                a2 = pd.read_csv(x0.out_dir(root) / f"a2_{run_id}.csv")
                result = f3_oracle_decomposition(
                    root, run_id, x0.best_cell(a2[a2["target"] == "s_next"])
                )
            elif args.analysis == "f2":
                a2 = pd.read_csv(x0.out_dir(root) / f"a2_{run_id}.csv")
                result = f2_residual(
                    root, run_id, x0.best_cell(a2[a2["target"] == "delta_from_next"])
                )
            elif args.analysis == "f5":
                result = f5_residualised_p1(root, run_id)
            else:
                result = f7_hits(root, run_id)
            path.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

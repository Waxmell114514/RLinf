"""X0: mechanism analyses on existing E4 runs.

Pre-registered in ``logbook.md`` ("X0 PRE-REGISTRATION") before any number
was computed.  Exploratory: nothing here changes an R1-R12 verdict.  Every
analysis reads each run's own ``probe_samples.csv`` and reuses the probe
code's folds, C / alpha grids and inner selection; the probe code is called,
never changed.

    P=experiments/efference_probe/x0_mechanism.py
    python $P --data-root D a1          # efference increment (no fits)
    python $P --data-root D a5          # locus of the P1 excursion (no fits)
    python $P --data-root D a2 RUN...   # outcome readout (ridge)
    python $P --data-root D a3 RUN...   # decoded-ingredients oracle
    python $P --data-root D a4          # outcome visibility (no hidden states)
    python $P --data-root D a6 RUN...   # novelty vs next-call reaction
    python $P --data-root D a7 RUN...   # paired bootstrap of rung differences

Outputs go to ``<data-root>/x0/`` as one CSV / JSON per analysis and run.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
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
from efference_probe.datasets import (  # noqa: E402
    HiddenStore,
    _axis_angle_to_matrix,
    _delta_states,
    _matrix_to_axis_angle,
    build_features,
    call_lookup,
    load_run,
)
from efference_probe.probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    DEFAULT_MLP_ALPHAS,
    _make_classifier,
    _outer_splits,
    _select_c,
    _SharedRidge,
    ridge_readout,
)

SEED = 0
N_SPLITS = 5
DELTA = 0.04
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)  # ridge_readout's default
MAX_CALLS = 4000
RUNGS = ("P0", "P1", "P3", "P4")
GOAL_SWAP = (
    "e4_sft_goal_swap_s1",
    "e4_sft_goal_swap_s2",
    "e4_grpo_goal_swap_s1",
    "e4_grpo_goal_swap_s2",
)


def checkpoint(run_id: str) -> str:
    return "GRPO" if run_id.startswith("e4_grpo") else "Base-Lora"


def out_dir(data_root: Path) -> Path:
    path = data_root / "x0"
    path.mkdir(parents=True, exist_ok=True)
    return path


def completed_runs(data_root: Path) -> list[str]:
    """E4 S2 runs with a passing linear marker, in plan order (attempt ids)."""
    plan = common.load_plan()
    runs = []
    for run in common.planned_runs(plan, "s2"):
        run_id = common.completed_attempt(data_root, run.run_id)
        if run_id and (common.job_status(data_root, run_id, "probe-linear") or {}).get(
            "ok"
        ):
            runs.append(run_id)
    return runs


def probe_frame(run_dir: Path, family: str) -> Optional[pd.DataFrame]:
    path = common.analysis_dir(run_dir, family) / "probe_results.csv"
    return pd.read_csv(path) if path.is_file() else None


def best_rows(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Each rung at its best (layer, pool) cell -- the report's convention."""
    out = {}
    for name in frame["name"].unique():
        rows = frame[frame["name"] == name]
        out[name] = rows.loc[rows["balanced_acc_mean"].idxmax()]
    return out


# -- A1 / A5: no fits ---------------------------------------------------------


def _a1_row(row: report.RunNumbers, family: str, matched: Optional[pd.Series]):
    numbers = row.families[family]
    if not all(numbers.get(r) for r in ("P1", "P3", "P4")):
        return None
    p1, p3, p4 = (numbers[r]["bacc"] for r in ("P1", "P3", "P4"))
    return {
        "run": row.run_id,
        "checkpoint": row.label,
        "suite": row.suite,
        "transform": row.transform,
        "family": family,
        "floor": numbers["floor"],
        "P1": p1,
        "P3": p3,
        "P4": p4,
        "P1_cell": numbers["P1"]["cell"],
        "P3_cell": numbers["P3"]["cell"],
        "d_eff": p3 - p1,
        "d_int": p4 - p1,
        "matched_max_d_eff": np.nan if matched is None else float(matched.max()),
        "matched_mean_d_eff": np.nan if matched is None else float(matched.mean()),
        "source": row.source,
    }


def a1_efference_increment(data_root: Path) -> pd.DataFrame:
    """P3 - P1 and P4 - P1 at best cells, as the report reads them."""
    plan = common.load_plan()
    planned = {r.run_id: r for r in common.planned_runs(plan, "s2")}
    rows = []
    for run_id in completed_runs(data_root):
        run = planned[re.sub(r"_r\d+$", "", run_id)]
        row = report.read_run(
            data_root / run_id, checkpoint(run_id), run.suite, run.transform, run.seed
        )
        for family in row.families:
            frame = probe_frame(data_root / run_id, family)
            p1 = frame[frame["name"] == "P1"].set_index(["layer", "pool"])
            p3 = frame[frame["name"] == "P3"].set_index(["layer", "pool"])
            matched = (p3["balanced_acc_mean"] - p1["balanced_acc_mean"]).dropna()
            rows.append(_a1_row(row, family, matched))
    for entry in plan["existing"]["runs"]:
        row = report.read_wave2_report(
            entry["run_id"],
            entry["label"],
            entry["suite"],
            entry["transform"],
            entry["seed"],
        )
        for family in row.families:
            rows.append(_a1_row(row, family, None))
    return pd.DataFrame([r for r in rows if r])


def a5_locus(data_root: Path, top: int = 3) -> pd.DataFrame:
    rows = []
    for run_id in GOAL_SWAP:
        for family in ("linear", "mlp"):
            frame = probe_frame(data_root / run_id, family)
            if frame is None:
                continue
            floor = frame[frame["name"] == "P0"]["balanced_acc_mean"].max()
            p1 = frame[frame["name"] == "P1"].nlargest(top, "balanced_acc_mean")
            for rank, (_, cell) in enumerate(p1.iterrows(), 1):
                layer, pool = int(cell["layer"]), cell["pool"]
                rows.append(
                    {
                        "run": run_id,
                        "checkpoint": checkpoint(run_id),
                        "family": family,
                        "rank": rank,
                        "layer": layer,
                        "pool": pool,
                        "margin": cell["balanced_acc_mean"] - floor,
                        "late_or_ctx_last": layer >= 24 or pool == "ctx_last",
                    }
                )
    return pd.DataFrame(rows)


# -- shared helpers for the fitted analyses ----------------------------------


def load(data_root: Path, run_id: str):
    run = load_run(data_root / run_id)
    samples = pd.read_csv(
        common.analysis_dir(run.run_dir, "linear") / "probe_samples.csv"
    )
    store = HiddenStore(run.run_dir / "hidden", max_cache_gb=4.0)
    return run, samples, store


def readout_pairs(run, max_calls: int = MAX_CALLS) -> pd.DataFrame:
    """(episode, c) with call c+1 present, post-success calls excluded, seed 0."""
    calls = run.calls
    if "post_success" in calls.columns:
        calls = calls[~calls["post_success"].astype(bool)]
    present = set(zip(run.calls["episode_id"], run.calls["call_idx"]))
    keep = [
        (int(e), int(c))
        for e, c in zip(calls["episode_id"], calls["call_idx"])
        if (int(e), int(c) + 1) in present
    ]
    frame = pd.DataFrame(keep, columns=["episode_id", "call"])
    if len(frame) > max_calls:
        frame = frame.sample(max_calls, random_state=SEED)
    return frame.sort_values(["episode_id", "call"]).reset_index(drop=True)


def states(run, pairs: list[tuple[int, int]], column: str) -> np.ndarray:
    lookup = call_lookup(run.calls)
    series = run.calls[column].to_numpy()
    return np.stack(
        [
            np.asarray(series[lookup[(int(e), int(c))]], dtype=np.float32)
            for e, c in pairs
        ]
    )


def compose_delta(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """``datasets._delta_states`` on arrays: translation/gripper differences,
    rotation composed as R_after @ R_before^T."""
    delta = (after - before).astype(np.float32)
    if after.shape[1] >= 6:
        relative = _axis_angle_to_matrix(
            after[:, 3:6].astype(np.float64)
        ) @ np.transpose(
            _axis_angle_to_matrix(before[:, 3:6].astype(np.float64)), (0, 2, 1)
        )
        delta[:, 3:6] = _matrix_to_axis_angle(relative).astype(np.float32)
    return delta


def mismatch_from_arrays(commanded: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """``datasets._mismatch_features`` from arrays (commanded [n, chunks, dim])."""
    summed = commanded.sum(axis=1)
    n_shared = min(summed.shape[1] - 1, delta.shape[1])
    parts = [summed, delta, summed[:, :n_shared] * delta[:, :n_shared]]
    for lo, hi in ((0, 3), (3, 6)):
        if summed.shape[1] < hi or delta.shape[1] < hi:
            continue
        command_block, delta_block = summed[:, lo:hi], delta[:, lo:hi]
        command_norm = np.linalg.norm(command_block, axis=1)
        delta_norm = np.linalg.norm(delta_block, axis=1)
        cosine = (command_block * delta_block).sum(axis=1) / np.maximum(
            command_norm * delta_norm, 1e-8
        )
        parts.append(
            np.stack(
                [
                    command_norm,
                    delta_norm,
                    cosine,
                    delta_norm - command_norm,
                    np.linalg.norm(delta_block - command_block, axis=1),
                ],
                axis=1,
            )
        )
    return np.concatenate(parts, axis=1).astype(np.float32)


def _ridge_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """ridge_readout's inner selection: first split of a grouped 3-fold."""
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    train, validation = next(iter(GroupKFold(n_splits=3).split(x, y, groups)))
    solver = _SharedRidge(x[train], y[train])
    scores = [
        r2_score(
            y[validation],
            solver.predict(x[validation], a),
            multioutput="uniform_average",
        )
        for a in RIDGE_ALPHAS
    ]
    return RIDGE_ALPHAS[int(np.argmax(scores))]


def best_cell(frame: pd.DataFrame, column: str = "r2_mean") -> tuple[int, str]:
    row = frame.loc[frame[column].idxmax()]
    return int(row["layer"]), str(row["pool"])


# -- A2: outcome readout -------------------------------------------------------


def _a2_cell(layer, pool, h_next, s_next, delta, groups) -> list[dict[str, Any]]:
    rows = []
    for target, y in (("s_next", s_next), ("delta_from_next", delta)):
        result = ridge_readout(h_next, y, groups, n_splits=N_SPLITS, seed=SEED)
        rows.append(
            {
                "target": target,
                "layer": layer,
                "pool": pool,
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "per_dim_r2": json.dumps(result["per_dim_r2"]),
                "alpha_at_grid_edge": result["alpha_at_grid_edge"],
                "n": result["n_samples"],
            }
        )
    return rows


def a2_outcome_readout(data_root: Path, run_id: str, n_jobs: int = 2) -> pd.DataFrame:
    from joblib import Parallel, delayed

    run, _, store = load(data_root, run_id)
    pairs = readout_pairs(run)
    cur = list(zip(pairs["episode_id"], pairs["call"]))
    nxt = [(e, c + 1) for e, c in cur]
    groups = pairs["episode_id"].to_numpy()
    s_next = states(run, cur, "states_after")
    delta = _delta_states(run.calls, call_lookup(run.calls), cur)

    def cells():
        for layer in run.layers:
            for pool in run.pools:
                h_next = store.vectors(
                    nxt, run.layer_index(layer), run.pool_index(pool)
                )
                yield layer, pool, h_next

    rows = []
    for chunk in Parallel(n_jobs=n_jobs)(
        delayed(_a2_cell)(layer, pool, h, s_next, delta, groups)
        for layer, pool, h in cells()
    ):
        rows.extend(chunk)
    frame = pd.DataFrame(rows)
    layer, pool = best_cell(frame[frame["target"] == "s_next"])
    both = np.concatenate(
        [
            store.vectors(cur, run.layer_index(layer), run.pool_index(pool)),
            store.vectors(nxt, run.layer_index(layer), run.pool_index(pool)),
        ],
        axis=1,
    )
    result = ridge_readout(both, delta, groups, n_splits=N_SPLITS, seed=SEED)
    rows.append(
        {
            "target": "delta_from_both",
            "layer": layer,
            "pool": pool,
            "r2_mean": result["r2_mean"],
            "r2_std": result["r2_std"],
            "per_dim_r2": json.dumps(result["per_dim_r2"]),
            "alpha_at_grid_edge": result["alpha_at_grid_edge"],
            "n": result["n_samples"],
        }
    )
    frame = pd.DataFrame(rows)
    frame.insert(0, "run", run_id)
    return frame


# -- A2b: does h(c+1) know the motion beyond the state? ---------------------


def _oof_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold ridge predictions on ``ridge_readout``'s folds and alphas."""
    from sklearn.model_selection import GroupKFold

    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    predicted = np.zeros_like(y)
    for train, test in GroupKFold(n_splits=N_SPLITS).split(x, y, groups):
        alpha = _ridge_alpha(x[train], y[train], groups[train])
        predicted[test] = _SharedRidge(x[train], y[train]).predict(x[test], alpha)
    return predicted


def _oof_knn(
    state: np.ndarray, y: np.ndarray, groups: np.ndarray, tasks: np.ndarray, k: int = 10
) -> np.ndarray:
    """Per-task k-NN on the standardised state, on the same grouped folds."""
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    predicted = np.zeros_like(y, dtype=np.float32)
    for train, test in GroupKFold(n_splits=N_SPLITS).split(state, y, groups):
        for task in np.unique(tasks[test]):
            fit, rows = train[tasks[train] == task], test[tasks[test] == task]
            if len(fit) == 0:
                fit = train
            scaler = StandardScaler().fit(state[fit])
            model = KNeighborsRegressor(n_neighbors=min(k, len(fit)))
            model.fit(scaler.transform(state[fit]), y[fit])
            predicted[rows] = model.predict(scaler.transform(state[rows]))
    return predicted


def a2b_motion_beyond_state(data_root: Path, run_id: str, cell: tuple[int, str]):
    """R² of Δs(c) from h(c+1) against state-only controls, by call label."""
    from sklearn.metrics import r2_score

    run, _, store = load(data_root, run_id)
    pairs = readout_pairs(run)
    cur = list(zip(pairs["episode_id"], pairs["call"]))
    nxt = [(e, c + 1) for e, c in cur]
    groups = pairs["episode_id"].to_numpy()
    lookup = call_lookup(run.calls)
    positions = np.array([lookup[(int(e), int(c))] for e, c in cur])
    hijack = run.calls["label"].to_numpy()[positions] == "HIJACK"
    tasks = run.calls["task_id"].to_numpy()[positions]
    delta = _delta_states(run.calls, lookup, cur)
    state = states(run, cur, "states_after")
    onehot = (tasks[:, None] == np.unique(tasks)[None, :]).astype(np.float32)
    per_task = np.concatenate(
        [
            state,
            onehot,
            (state[:, :, None] * onehot[:, None, :]).reshape(len(state), -1),
        ],
        axis=1,
    )
    hidden = store.vectors(nxt, run.layer_index(cell[0]), run.pool_index(cell[1]))
    predictions = {
        "H": _oof_ridge(hidden, delta, groups),
        "S1": _oof_ridge(per_task, delta, groups),
        "S2": _oof_knn(state, delta, groups, tasks),
        "H+S1": _oof_ridge(np.concatenate([hidden, per_task], axis=1), delta, groups),
    }
    subsets = {"all": np.ones(len(delta), bool), "self": ~hijack, "hijack": hijack}
    r2 = {
        name: {
            subset: float(
                r2_score(delta[rows], pred[rows], multioutput="uniform_average")
            )
            for subset, rows in subsets.items()
        }
        for name, pred in predictions.items()
    }
    best_state_all = max(r2["S1"]["all"], r2["S2"]["all"])
    best_state_hijack = max(r2["S1"]["hijack"], r2["S2"]["hijack"])
    flags = {
        "regularity explains A2 (ii)": best_state_all >= r2["H"]["all"] - 0.10,
        "h(c+1) carries motion beyond the state": (
            r2["H"]["hijack"] >= best_state_hijack + 0.10
        ),
    }
    reading = " and ".join(k for k, v in flags.items() if v) or "unresolved"
    return {
        "run": run_id,
        "checkpoint": checkpoint(run_id),
        "cell": list(cell),
        "n": int(len(delta)),
        "n_hijack": int(hijack.sum()),
        "r2": r2,
        "flags": flags,
        "reading": reading,
    }


# -- A3: decoded-ingredients oracle -----------------------------------------


class _Decoder:
    """Ridge from one hidden cell to one target, alpha picked like ridge_readout."""

    def __init__(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
        self.alpha = _ridge_alpha(x, y, groups)
        self.solver = _SharedRidge(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.solver.predict(x, self.alpha)


def a3_oracle(
    data_root: Path,
    run_id: str,
    state_cell: tuple[int, str],
    delta_cell: tuple[int, str],
) -> dict[str, Any]:
    """Score P2r's comparison features built from *decoded* command and outcome."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    run, samples, store = load(data_root, run_id)
    e1 = pd.read_csv(common.analysis_dir(run.run_dir, "linear") / "e1_readout.csv")
    cmd_cell = best_cell(e1)
    n_chunks, action_dim = (
        int(run.schema["num_action_chunks"]),
        int(run.schema["action_dim"]),
    )

    def h(pairs, cell):
        return store.vectors(pairs, run.layer_index(cell[0]), run.pool_index(cell[1]))

    # Decoder training rows: every call c (with c+1) of an episode, seed-0 capped.
    pairs = readout_pairs(run, max_calls=10**9)
    cur = list(zip(pairs["episode_id"], pairs["call"]))
    nxt = [(e, c + 1) for e, c in cur]
    d_groups = pairs["episode_id"].to_numpy()
    d_hcmd, d_hstate_cur, d_hstate_nxt, d_hdelta = (
        h(cur, cmd_cell),
        h(cur, state_cell),
        h(nxt, state_cell),
        h(nxt, delta_cell),
    )
    d_cmd = states(run, cur, "a_cmd_env")
    d_before = states(run, cur, "states_before")
    d_after = states(run, cur, "states_after")
    d_delta = _delta_states(run.calls, call_lookup(run.calls), cur)
    # Each sample's decoding inputs, from the same cells.
    s_prev = list(zip(samples["episode_id"], samples["prev_call"]))
    s_cur = list(zip(samples["episode_id"], samples["cur_call"]))
    s_hcmd, s_hstate_prev, s_hstate_cur, s_hdelta = (
        h(s_prev, cmd_cell),
        h(s_prev, state_cell),
        h(s_cur, state_cell),
        h(s_cur, delta_cell),
    )
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()

    def decode(held_out: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Fit on every decoder row outside ``held_out`` episodes; decode ``rows``.

        One state decoder serves both endpoints: h(k) is read out as the state
        at observation k, so s_before(c) comes from h(c) and s_after(c) from
        h(c+1) through the same map, and Δŝ is a difference of one readout.
        """
        fit = ~np.isin(d_groups, held_out)
        cmd = _Decoder(d_hcmd[fit], d_cmd[fit], d_groups[fit])
        state = _Decoder(
            np.concatenate([d_hstate_cur[fit], d_hstate_nxt[fit]]),
            np.concatenate([d_before[fit], d_after[fit]]),
            np.concatenate([d_groups[fit], d_groups[fit]]),
        )
        delta = _Decoder(d_hdelta[fit], d_delta[fit], d_groups[fit])
        commanded = cmd.predict(s_hcmd[rows]).reshape(-1, n_chunks, action_dim)
        variant_a = compose_delta(
            state.predict(s_hstate_prev[rows]), state.predict(s_hstate_cur[rows])
        )
        variant_b = delta.predict(s_hdelta[rows])
        return {
            "A": mismatch_from_arrays(commanded, variant_a),
            "B": mismatch_from_arrays(commanded, variant_b),
        }

    folds = _outer_splits(np.zeros((len(y), 1)), y, groups, N_SPLITS, SEED)
    scores: dict[str, list[float]] = {"A": [], "B": []}
    for fold, (train, test) in enumerate(folds):
        test_episodes = np.unique(groups[test])
        # Training samples get out-of-sample decodes too (inner cross-fitting
        # over training episodes), so train and test features are alike.
        train_features = {k: np.zeros((len(train), 0), np.float32) for k in scores}
        inner = GroupKFold(n_splits=N_SPLITS).split(
            np.zeros((len(train), 1)), y[train], groups[train]
        )
        for _, inner_held in inner:
            held = np.concatenate([test_episodes, np.unique(groups[train[inner_held]])])
            decoded = decode(held, train[inner_held])
            for key in scores:
                if train_features[key].shape[1] == 0:
                    train_features[key] = np.zeros(
                        (len(train), decoded[key].shape[1]), np.float32
                    )
                train_features[key][inner_held] = decoded[key]
        test_features = decode(test_episodes, test)
        for key in scores:
            c = _select_c(
                train_features[key],
                y[train],
                groups[train],
                DEFAULT_C_VALUES,
                SEED,
                2000,
            )
            model = _make_classifier(c, 2000, None, "linear", SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train_features[key], y[train])
            scores[key].append(
                float(
                    balanced_accuracy_score(y[test], model.predict(test_features[key]))
                )
            )
        print(
            f"{run_id} A3 fold {fold + 1}/{N_SPLITS}: "
            f"A {scores['A'][-1]:.3f}  B {scores['B'][-1]:.3f}",
            flush=True,
        )
    numbers = report._family_numbers(common.analysis_dir(run.run_dir, "linear"))
    return {
        "run": run_id,
        "checkpoint": checkpoint(run_id),
        "command_cell": list(cmd_cell),
        "state_cell": list(state_cell),
        "delta_cell": list(delta_cell),
        "oracle_A": float(np.mean(scores["A"])),
        "oracle_A_folds": scores["A"],
        "oracle_B": float(np.mean(scores["B"])),
        "oracle_B_folds": scores["B"],
        "floor": float(numbers["floor"]),
        "p2r_true": float(numbers["P2r"]["bacc"]),
        "n": int(len(y)),
    }


def check_mismatch_reimplementation(data_root: Path, run_id: str) -> float:
    """Max |difference| between ``mismatch_from_arrays`` and the probe block."""
    run, samples, _ = load(data_root, run_id)
    reference, _ = build_features(run, samples, ["mismatch"])
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    commanded = states(run, prev, "a_cmd_env").reshape(
        -1, int(run.schema["num_action_chunks"]), int(run.schema["action_dim"])
    )
    delta = _delta_states(run.calls, call_lookup(run.calls), prev)
    return float(np.max(np.abs(mismatch_from_arrays(commanded, delta) - reference)))


# -- A4: outcome visibility -----------------------------------------------------


def a4_visibility(data_root: Path, run_id: str) -> dict[str, Any]:
    """AUROC of a non-learned Mahalanobis score on Δs, cross-fitted by fold."""
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score

    run, samples, _ = load(data_root, run_id)
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    delta = _delta_states(run.calls, call_lookup(run.calls), prev).astype(np.float64)
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    tasks = samples["task_id"].to_numpy()
    score = np.full(len(y), np.nan)
    for train, test in _outer_splits(np.zeros((len(y), 1)), y, groups, N_SPLITS, SEED):
        for task in np.unique(tasks[test]):
            ref = train[(tasks[train] == task) & (y[train] == 0)]
            rows = test[tasks[test] == task]
            if len(ref) < delta.shape[1] + 2:
                ref = train[y[train] == 0]  # too few self samples in this task
            model = LedoitWolf().fit(delta[ref])
            score[rows] = model.mahalanobis(delta[rows])
    linear = best_rows(probe_frame(run.run_dir, "linear"))
    return {
        "run": run_id,
        "checkpoint": checkpoint(run_id),
        "auroc": float(roc_auc_score(y, score)),
        "p1_margin_linear": float(
            linear["P1"]["balanced_acc_mean"] - linear["P0"]["balanced_acc_mean"]
        ),
        "c_dstates_linear": float(linear["C_dstates"]["balanced_acc_mean"]),
        "n": int(len(y)),
    }


# -- A6: novelty vs next-call reaction --------------------------------------


def a6_novelty(data_root: Path, run_id: str, cell=(0, "ctx_mean")) -> pd.DataFrame:
    """Per matched pair: Δnovelty of h(c+1) and Δentropy of the next call."""
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    run, samples, store = load(data_root, run_id)
    cur = list(zip(samples["episode_id"], samples["cur_call"]))
    hidden = store.vectors(
        cur, run.layer_index(cell[0]), run.pool_index(cell[1])
    ).astype(np.float64)
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    novelty = np.full(len(y), np.nan)
    for train, test in _outer_splits(np.zeros((len(y), 1)), y, groups, N_SPLITS, SEED):
        ref = train[y[train] == 0]
        pca = PCA(n_components=min(50, len(ref) - 1), random_state=SEED).fit(
            hidden[ref]
        )
        model = LedoitWolf().fit(pca.transform(hidden[ref]))
        novelty[test] = model.mahalanobis(pca.transform(hidden[test]))
    lookup = call_lookup(run.calls)
    entropy = run.calls["entropy_mean"].to_numpy()
    logprob = run.calls["logprob_sum"].to_numpy()
    positions = np.array([lookup[(int(e), int(c))] for e, c in cur])
    frame = pd.DataFrame(
        {
            "episode_id": groups,
            "pair_id": samples["pair_id"].to_numpy(),
            "y": y,
            "novelty": novelty,
            "entropy": entropy[positions],
            "logprob": logprob[positions],
        }
    )
    wide = frame.pivot_table(index=["episode_id", "pair_id"], columns="y")
    wide = wide.dropna()
    out = pd.DataFrame(
        {
            "episode_id": wide.index.get_level_values("episode_id"),
            "d_novelty": (wide[("novelty", 1)] - wide[("novelty", 0)]).to_numpy(),
            "d_entropy": (wide[("entropy", 1)] - wide[("entropy", 0)]).to_numpy(),
            "d_logprob": (wide[("logprob", 1)] - wide[("logprob", 0)]).to_numpy(),
        }
    )
    out.insert(0, "run", run_id)
    return out


def cluster_bootstrap_spearman(frame: pd.DataFrame, b: int = 2000) -> dict[str, float]:
    """Spearman ρ(Δnovelty, Δentropy) with an episode-cluster bootstrap CI."""
    from scipy.stats import spearmanr

    rho = float(spearmanr(frame["d_novelty"], frame["d_entropy"]).statistic)
    clusters = frame.groupby(["run", "episode_id"]).indices
    keys = list(clusters)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([clusters[keys[i]] for i in pick])
        draws.append(
            spearmanr(
                frame["d_novelty"].to_numpy()[rows], frame["d_entropy"].to_numpy()[rows]
            ).statistic
        )
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "rho": rho,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": int(len(frame)),
        "n_clusters": len(keys),
    }


# -- A7: paired bootstrap -------------------------------------------------------


def oof_predictions(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, family: str
) -> tuple[np.ndarray, list[float], list[float]]:
    """``run_probe``'s outer loop, keeping the out-of-fold predictions."""
    from sklearn.metrics import balanced_accuracy_score

    grid = DEFAULT_C_VALUES if family == "linear" else DEFAULT_MLP_ALPHAS
    predicted = np.full(len(y), -1)
    folds, chosen = [], []
    for train, test in _outer_splits(x, y, groups, N_SPLITS, SEED):
        c = _select_c(x[train], y[train], groups[train], grid, SEED, 2000, None, family)
        model = _make_classifier(c, 2000, None, family, SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x[train], y[train])
        predicted[test] = model.predict(x[test])
        folds.append(float(balanced_accuracy_score(y[test], predicted[test])))
        chosen.append(float(c))
    return predicted, folds, chosen


def a7_bootstrap(
    data_root: Path, run_id: str, family: str, b: int = 2000
) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score

    run, samples, store = load(data_root, run_id)
    frame = probe_frame(run.run_dir, family)
    best = best_rows(frame)
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    predictions, checks = {}, {}
    for name in ("P1", "P3", "C_dstates"):
        row = best[name]
        blocks = str(row["blocks"]).split("+")
        layer = None if pd.isna(row["layer"]) else int(row["layer"])
        pool = None if pd.isna(row["pool"]) else str(row["pool"])
        x, _ = build_features(run, samples, blocks, layer=layer, pool=pool, store=store)
        predictions[name], folds, chosen = oof_predictions(x, y, groups, family)
        expected = ast.literal_eval(row["fold_balanced_acc"])
        checks[name] = {
            "cell": f"L{layer}/{pool}" if layer is not None else "-",
            "max_fold_diff": float(
                np.max(np.abs(np.asarray(folds) - np.asarray(expected)))
            ),
            "same_c": chosen == [float(c) for c in ast.literal_eval(row["selected_c"])],
        }
    clusters = pd.Series(np.arange(len(y))).groupby(groups).indices
    keys = list(clusters)
    rng = np.random.default_rng(SEED)

    def bacc(name, rows):
        return balanced_accuracy_score(y[rows], predictions[name][rows])

    everything = np.arange(len(y))
    point = {
        "P1-C_dstates": bacc("P1", everything) - bacc("C_dstates", everything),
        "P3-P1": bacc("P3", everything) - bacc("P1", everything),
    }
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(b):
        rows = np.concatenate(
            [clusters[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        )
        draws["P1-C_dstates"].append(bacc("P1", rows) - bacc("C_dstates", rows))
        draws["P3-P1"].append(bacc("P3", rows) - bacc("P1", rows))
    out = {
        "run": run_id,
        "checkpoint": checkpoint(run_id),
        "family": family,
        "checks": checks,
        "reproduced": all(
            c["max_fold_diff"] < 1e-6 and c["same_c"] for c in checks.values()
        ),
        "n": int(len(y)),
        "n_episodes": len(keys),
    }
    for key, value in point.items():
        low, high = np.percentile(draws[key], [2.5, 97.5])
        out[key] = {"point": float(value), "ci_low": float(low), "ci_high": float(high)}
    return out


# Host types the Modal probe rounds ran on differ in BLAS kernels and thread
# counts, and lbfgs predictions near the boundary follow them, so a refit can
# miss its CSV by a sample or two.  A7 is therefore run once per setting below
# (``OPENBLAS_CORETYPE`` / ``OPENBLAS_NUM_THREADS``), each into
# ``x0/a7_passes/<setting>/``; a run counts as reproduced when one setting
# reproduces all of its rungs, and that setting's numbers are used.
BLAS_PASSES = ("avx2_1t", "avx512_4t", "avx512_1t")


def a7_select(out: Path, family: str) -> pd.DataFrame:
    rows = {}
    for setting in BLAS_PASSES:
        for path in sorted((out / "a7_passes" / setting).glob(f"a7_*_{family}.json")):
            result = json.loads(path.read_text())
            if result["run"] in rows and rows[result["run"]]["reproduced"]:
                continue
            row = {"run": result["run"], "checkpoint": result["checkpoint"]}
            row.update(family=family, blas=setting, reproduced=result["reproduced"])
            for key in ("P1-C_dstates", "P3-P1"):
                for stat in ("point", "ci_low", "ci_high"):
                    row[f"{key}_{stat}"] = result[key][stat]
            row["n_episodes"] = result["n_episodes"]
            rows[result["run"]] = row
    return pd.DataFrame(list(rows.values()))


# -- CLI -----------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "analysis",
        choices=[
            "a1",
            "a2",
            "a2b",
            "a3",
            "a4",
            "a5",
            "a6",
            "a6-pool",
            "a7",
            "a7-select",
            "check",
        ],
    )
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--family", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    root, out = args.data_root, out_dir(args.data_root)
    if args.analysis == "a1":
        frame = a1_efference_increment(root)
        frame.to_csv(out / "a1_efference_increment.csv", index=False)
        print(frame.to_string(index=False))
    elif args.analysis == "a5":
        frame = a5_locus(root)
        frame.to_csv(out / "a5_locus.csv", index=False)
        print(frame.to_string(index=False))
    elif args.analysis == "a4":
        rows = [a4_visibility(root, r) for r in (args.runs or completed_runs(root))]
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "a4_visibility.csv", index=False)
        print(frame.to_string(index=False))
    elif args.analysis == "a6-pool":
        frame = pd.concat(
            [pd.read_csv(out / f"a6_{run_id}.csv") for run_id in args.runs],
            ignore_index=True,
        )
        result = {"runs": args.runs, **cluster_bootstrap_spearman(frame)}
        (out / "a6_pooled.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    elif args.analysis == "a7-select":
        frame = a7_select(out, args.family)
        frame.to_csv(out / f"a7_{args.family}_selected.csv", index=False)
        print(frame.to_string(index=False))
    elif args.analysis == "check":
        for run_id in args.runs:
            print(
                run_id,
                "mismatch max |diff|:",
                check_mismatch_reimplementation(root, run_id),
            )
    else:
        for run_id in args.runs:
            target = (
                out
                / f"{args.analysis}_{run_id}{'_' + args.family if args.analysis == 'a7' else ''}"
            )
            if args.analysis == "a2":
                path = target.with_suffix(".csv")
                if path.is_file():
                    print(f"{run_id}: {path.name} exists; skipping")
                    continue
                frame = a2_outcome_readout(root, run_id, n_jobs=args.jobs)
                frame.to_csv(path, index=False)
                print(
                    frame.sort_values("r2_mean", ascending=False)
                    .groupby("target")
                    .head(1)
                    .to_string(index=False)
                )
            elif args.analysis == "a2b":
                a2 = pd.read_csv(out / f"a2_{run_id}.csv")
                cell = best_cell(a2[a2["target"] == "delta_from_next"])
                result = a2b_motion_beyond_state(root, run_id, cell)
                target.with_suffix(".json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result, indent=2))
            elif args.analysis == "a3":
                a2 = pd.read_csv(out / f"a2_{run_id}.csv")
                result = a3_oracle(
                    root,
                    run_id,
                    best_cell(a2[a2["target"] == "s_next"]),
                    best_cell(a2[a2["target"] == "delta_from_next"]),
                )
                target.with_suffix(".json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result, indent=2))
            elif args.analysis == "a6":
                frame = a6_novelty(root, run_id)
                frame.to_csv(target.with_suffix(".csv"), index=False)
                print(run_id, cluster_bootstrap_spearman(frame))
            elif args.analysis == "a7":
                result = a7_bootstrap(root, run_id, args.family)
                target.with_suffix(".json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

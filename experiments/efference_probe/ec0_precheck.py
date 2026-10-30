"""EC0: a label-free efference comparator on frozen VLA features.

Pre-registered in ``logbook.md`` ("EC0 PRE-REGISTRATION").  Given the
commanded chunk a_cmd(c) (the efference copy) and a forward model learned only
from the policy's own SELF transitions, can an override of chunk c be detected
at call c + 1?  No HIJACK transition enters any fit: state decoders, forward
model and residual scales are all fit on SELF pairs of the training episodes.

- **V_feat** (primary): s(c) and s(c+1) are decoded from h(c) and h(c+1) by one
  ridge state decoder at the run's A2 (i) best ``s_next`` cell.  Training-row
  decodes are cross-fitted over the training episodes, so the forward model
  sees decodes as noisy as the held-out ones (as A3 does).
- **V_true** (upper bound): the simulator's 8-dim states.

Forward model: ridge [s(c), a_cmd(c) flattened, task one-hot] -> delta s.  The
score is the residual's squared norm in units of the training SELF residual
SD per dimension.  TPR is read at the 98th percentile of the run's
out-of-fold SELF scores (FPR 0.02).
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

from efference_probe import x0_mechanism as x0  # noqa: E402
from efference_probe.datasets import HiddenStore, call_lookup, load_run  # noqa: E402

FPR = 0.02
N_SPLITS = 5
TPR_GATE = 0.5
X1_RUNS = tuple(
    f"x1_{ckpt}_{key}"
    for key in ("swap", "mirror", "freeze")
    for ckpt in ("sft", "grpo")
)
GATED = ("x1_sft_mirror", "x1_grpo_mirror", "x1_sft_freeze", "x1_grpo_freeze")


def forward_inputs(
    state: np.ndarray, commands: np.ndarray, tasks: np.ndarray, task_ids: np.ndarray
) -> np.ndarray:
    onehot = (tasks[:, None] == task_ids[None, :]).astype(np.float32)
    flat = commands.reshape(len(commands), -1).astype(np.float32)
    return np.concatenate([state.astype(np.float32), flat, onehot], axis=1)


def _residual_scores(
    before: np.ndarray,
    after: np.ndarray,
    commands: np.ndarray,
    tasks: np.ndarray,
    fit: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    task_ids: np.ndarray,
    forward: str = "ridge",
    dims: Optional[slice] = None,
) -> np.ndarray:
    """Fit the forward model on ``fit`` rows; score ``score`` rows.

    ``forward`` and ``dims`` are EC0b's exploratory variants; the registered
    EC0 is ridge over all 8 dims.
    """
    delta = x0.compose_delta(before, after)
    inputs = forward_inputs(before, commands, tasks, task_ids)
    model = (
        x0._Decoder(inputs[fit], delta[fit], groups[fit])
        if forward == "ridge"
        else _MlpForward(inputs[fit], delta[fit])
    )
    train_residual = delta[fit] - model.predict(inputs[fit])
    scale = np.maximum(train_residual.std(axis=0), 1e-8)
    residual = (delta[score] - model.predict(inputs[score])) / scale
    if dims is not None:
        residual = residual[:, dims]
    return (residual**2).sum(axis=1)


class _MlpForward:
    """EC0b (a): an MLP forward model on standardised inputs and targets."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor

        self.x_mean, self.x_sd = x.mean(0), np.maximum(x.std(0), 1e-8)
        self.y_mean, self.y_sd = y.mean(0), np.maximum(y.std(0), 1e-8)
        self.model = MLPRegressor(
            hidden_layer_sizes=(256, 256),
            alpha=1e-3,
            early_stopping=True,
            random_state=0,
        ).fit((x - self.x_mean) / self.x_sd, (y - self.y_mean) / self.y_sd)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.model.predict((x - self.x_mean) / self.x_sd)
        return z * self.y_sd + self.y_mean


def oof_scores(
    s_before: np.ndarray,
    s_after: np.ndarray,
    commands: np.ndarray,
    tasks: np.ndarray,
    is_self: np.ndarray,
    groups: np.ndarray,
    h_before: Optional[np.ndarray] = None,
    h_after: Optional[np.ndarray] = None,
    forward: str = "ridge",
    dims: Optional[slice] = None,
) -> np.ndarray:
    """Out-of-fold comparator scores; decoded states iff hidden states given.

    Every fit (state decoder, forward model, residual scale) uses SELF rows of
    the training episodes only.
    """
    from sklearn.model_selection import GroupKFold

    decoded = h_before is not None
    task_ids = np.unique(tasks)
    scores = np.full(len(groups), np.nan)
    for train, test in GroupKFold(n_splits=N_SPLITS).split(s_before, groups=groups):
        if not decoded:
            before, after = s_before, s_after
        else:
            before = np.zeros_like(s_before, dtype=np.float32)
            after = np.zeros_like(s_after, dtype=np.float32)

            def decode_into(fit_rows: np.ndarray, rows: np.ndarray) -> None:
                decoder = x0._Decoder(
                    np.concatenate([h_before[fit_rows], h_after[fit_rows]]),
                    np.concatenate([s_before[fit_rows], s_after[fit_rows]]),
                    np.concatenate([groups[fit_rows], groups[fit_rows]]),
                )
                before[rows] = decoder.predict(h_before[rows])
                after[rows] = decoder.predict(h_after[rows])

            # Held-out rows: a decoder on every training SELF row.
            decode_into(train[is_self[train]], test)
            # Training rows: cross-fitted over the training episodes.
            inner = GroupKFold(n_splits=N_SPLITS).split(train, groups=groups[train])
            for inner_fit, inner_out in inner:
                fit_rows = train[inner_fit]
                decode_into(fit_rows[is_self[fit_rows]], train[inner_out])
        fit = train[is_self[train]]
        scores[test] = _residual_scores(
            before, after, commands, tasks, fit, test, groups, task_ids, forward, dims
        )
    return scores


def operating_point(scores: np.ndarray, is_self: np.ndarray, is_hijack: np.ndarray):
    from sklearn.metrics import roc_auc_score

    tau = float(np.quantile(scores[is_self], 1 - FPR))
    labels = np.r_[np.zeros(is_self.sum()), np.ones(is_hijack.sum())]
    values = np.r_[scores[is_self], scores[is_hijack]]
    return {
        "tau": tau,
        "fpr": float(np.mean(scores[is_self] > tau)),
        "tpr": float(np.mean(scores[is_hijack] > tau)),
        "auroc": float(roc_auc_score(labels, values)),
    }


def state_cell(data_root: Path, run_id: str) -> tuple[int, str]:
    a2 = pd.read_csv(x0.out_dir(data_root) / f"a2_{run_id}.csv")
    return x0.best_cell(a2[a2["target"] == "s_next"])


def run_ec0(data_root: Path, run_id: str, ec0b: bool = False) -> dict[str, Any]:
    run = load_run(data_root / run_id)
    store = HiddenStore(run.run_dir / "hidden", max_cache_gb=4.0)
    pairs = x0.readout_pairs(run, max_calls=10**9)
    lookup = call_lookup(run.calls)
    rows = [
        lookup[(int(e), int(c))] for e, c in zip(pairs["episode_id"], pairs["call"])
    ]
    labels = run.calls["label"].to_numpy()[rows]
    keep = np.isin(labels, ("SELF", "HIJACK"))
    pairs, labels = pairs[keep].reset_index(drop=True), labels[keep]
    cur = list(zip(pairs["episode_id"], pairs["call"]))
    nxt = [(e, c + 1) for e, c in cur]
    is_self, is_hijack = labels == "SELF", labels == "HIJACK"
    groups = pairs["episode_id"].to_numpy()
    tasks = run.calls["task_id"].to_numpy()[np.asarray(rows)[keep]]
    commands = x0.states(run, cur, "a_cmd_env")
    s_before = x0.states(run, cur, "states_before")
    s_after = x0.states(run, cur, "states_after")
    cell = state_cell(data_root, run_id)
    layer, pool = run.layer_index(cell[0]), run.pool_index(cell[1])
    h_before, h_after = store.vectors(cur, layer, pool), store.vectors(nxt, layer, pool)
    out: dict[str, Any] = {
        "run": run_id,
        "n_self": int(is_self.sum()),
        "n_hijack": int(is_hijack.sum()),
        "state_cell": f"L{cell[0]}/{cell[1]}",
    }
    true_scores = oof_scores(s_before, s_after, commands, tasks, is_self, groups)
    out["V_true"] = operating_point(true_scores, is_self, is_hijack)
    if ec0b:
        mlp = oof_scores(
            s_before, s_after, commands, tasks, is_self, groups, forward="mlp"
        )
        out["EC0b_mlp"] = operating_point(mlp, is_self, is_hijack)
        translation = oof_scores(
            s_before, s_after, commands, tasks, is_self, groups, dims=slice(0, 3)
        )
        out["EC0b_translation"] = operating_point(translation, is_self, is_hijack)
        return out
    feat_scores = oof_scores(
        s_before, s_after, commands, tasks, is_self, groups, h_before, h_after
    )
    out["V_feat"] = operating_point(feat_scores, is_self, is_hijack)
    return out


def reading(results: dict[str, dict[str, Any]]) -> str:
    gated = [r for r in GATED if r in results]
    if len(gated) < len(GATED):
        return f"unresolved ({len(gated)} of {len(GATED)} gated runs)"
    passed = sum(results[r]["V_feat"]["tpr"] >= TPR_GATE for r in GATED)
    if passed == len(GATED):
        return "go"
    return "mixed" if passed >= 2 else "no-go"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ec0b", action="store_true", help="exploratory EC0b variants (V_true only)"
    )
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    prefix = "ec0b" if args.ec0b else "ec0"
    for run_id in args.runs:
        results[run_id] = run_ec0(args.data_root, run_id, args.ec0b)
        (args.out / f"{prefix}_{run_id}.json").write_text(
            json.dumps(results[run_id], indent=2)
        )
        print(json.dumps(results[run_id]), flush=True)
    if args.ec0b:
        return 0
    summary = {"runs": results, "reading": reading(results)}
    (args.out / "ec0_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"reading": summary["reading"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

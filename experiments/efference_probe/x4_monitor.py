"""X4: fit and freeze the external comparator (PREREG_X1_X4.md, X4).

    P=experiments/efference_probe/x4_monitor.py
    python $P --data-root D --ckpt grpo     # writes x4/monitor_grpo.json
    python $P --data-root D --ckpt sft

- Model: P2r's comparison block on the probe samples of
  ``e4_<ckpt>_goal_mirror_s1``, with the ladder's linear pipeline and C
  selection.
- Threshold tau: the 98th percentile of the model's scores on every SELF call
  with call_idx >= 4 of the held-out run ``e4_<ckpt>_goal_mirror_s2``, so the
  nominal online false-positive rate is 2%.
- Replay gate: TPR on that run's HIJACK calls (call_idx >= 4) must be >= 0.5,
  else X4 drops the monitor condition for this checkpoint.

The artifact is plain numbers plus a sha256 (``monitor.LinearMonitor``); the
harness loads it without scikit-learn and refuses any other digest.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import e4_common as common  # noqa: E402
from efference_probe.datasets import build_features, load_run  # noqa: E402
from efference_probe.monitor import LinearMonitor, mismatch_features  # noqa: E402
from efference_probe.probes import (  # noqa: E402
    DEFAULT_C_VALUES,
    _make_classifier,
    _select_c,
)

SEED = 0
FPR = 0.02
SCORE_FROM_CALL = 4
MIN_REPLAY_TPR = 0.5
OUT_DIR = Path(__file__).resolve().parent / "x4"


def call_features(run, rows: pd.DataFrame) -> np.ndarray:
    """Online features of logged calls, from their own rows."""
    chunks = int(run.schema["num_action_chunks"])
    dim = int(run.schema["action_dim"])
    commanded = np.stack(rows["a_cmd_env"].map(np.asarray).to_list()).reshape(
        -1, chunks, dim
    )
    before = np.stack(rows["states_before"].map(np.asarray).to_list())
    after = np.stack(rows["states_after"].map(np.asarray).to_list())
    return mismatch_features(commanded, before, after)


def fit_monitor(run_dir: Path) -> tuple[LinearMonitor, dict[str, Any]]:
    """The ladder's linear P2r probe, fit on all of one run's probe samples."""
    run = load_run(run_dir)
    samples = pd.read_csv(
        common.analysis_dir(run.run_dir, "linear") / "probe_samples.csv"
    )
    x, _ = build_features(run, samples, ["mismatch"])
    # The harness computes the same features from a call's own row.
    lookup = {
        (int(e), int(c)): i
        for i, (e, c) in enumerate(zip(run.calls["episode_id"], run.calls["call_idx"]))
    }
    rows = run.calls.iloc[
        [
            lookup[(int(e), int(c))]
            for e, c in zip(samples["episode_id"], samples["prev_call"])
        ]
    ]
    online = call_features(run, rows)
    if not np.array_equal(online, x):
        raise RuntimeError(
            "online features differ from the probe mismatch block: "
            f"max |diff| {np.max(np.abs(online - x)):.3g}"
        )
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    c = _select_c(x, y, groups, DEFAULT_C_VALUES, SEED, 2000)
    pipeline = _make_classifier(c, 2000, None, "linear", SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(x, y)
    scaler, clf = pipeline.named_steps["scale"], pipeline.named_steps["clf"]
    monitor = LinearMonitor(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coef=np.asarray(clf.coef_, dtype=np.float64).ravel(),
        intercept=float(clf.intercept_[0]),
        threshold=float("inf"),
    )
    gap = float(np.max(np.abs(monitor.score(x) - pipeline.decision_function(x))))
    if gap > 1e-5:
        raise RuntimeError(f"exported monitor differs from the pipeline by {gap:.3g}")
    return monitor, {
        "train_run": run_dir.name,
        "C": float(c),
        "n_train": int(len(y)),
        "n_train_hijack": int(y.sum()),
        "export_max_abs_diff": gap,
    }


def scored_calls(run_dir: Path, monitor: LinearMonitor) -> pd.DataFrame:
    """Every call with call_idx >= SCORE_FROM_CALL, with its monitor score."""
    run = load_run(run_dir)
    rows = run.calls[run.calls["call_idx"] >= SCORE_FROM_CALL]
    return pd.DataFrame(
        {
            "label": rows["label"].to_numpy(),
            "score": monitor.score(call_features(run, rows)),
        }
    )


def set_threshold(
    monitor: LinearMonitor, holdout_dir: Path, fpr: float = FPR
) -> dict[str, Any]:
    """tau from held-out SELF calls; the replay TPR on held-out HIJACK calls."""
    scored = scored_calls(holdout_dir, monitor)
    self_scores = scored.loc[scored["label"] == "SELF", "score"].to_numpy()
    hijack_scores = scored.loc[scored["label"] == "HIJACK", "score"].to_numpy()
    tau = float(np.quantile(self_scores, 1.0 - fpr))
    monitor.threshold = tau
    replay_tpr = float(np.mean(hijack_scores > tau))
    return {
        "holdout_run": holdout_dir.name,
        "quantile": 1.0 - fpr,
        "threshold": tau,
        "replay_fpr": float(np.mean(self_scores > tau)),
        "replay_tpr": replay_tpr,
        "n_holdout_self": int(len(self_scores)),
        "n_holdout_hijack": int(len(hijack_scores)),
        "gate_min_tpr": MIN_REPLAY_TPR,
        "gate_passed": replay_tpr >= MIN_REPLAY_TPR,
    }


def build(data_root: Path, ckpt: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    import sklearn

    monitor, fit_info = fit_monitor(data_root / f"e4_{ckpt}_goal_mirror_s1")
    gate = set_threshold(monitor, data_root / f"e4_{ckpt}_goal_mirror_s2")
    monitor.meta = {
        **fit_info,
        **gate,
        "ckpt": ckpt,
        "features": "P2r mismatch block (monitor.mismatch_features)",
        "score_from_call": SCORE_FROM_CALL,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"monitor_{ckpt}.json"
    digest = monitor.save(path)
    return {"artifact": str(path), "sha256": digest, **monitor.meta}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ckpt", choices=["sft", "grpo"], required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    result = build(args.data_root, args.ckpt, args.out_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

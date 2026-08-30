"""Drives the probe ladder, the controls, E1 and E3 over a collection run.

Produces CSV tables plus a markdown block ready to paste into ``results.md``.
Reports numbers and anomalies only -- interpretation is the applicant's job
(SPEC 8.2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .datasets import (
    HiddenStore,
    RunData,
    build_features,
    build_probe_samples,
    load_run,
    step_index_report,
)
from .probes import (
    DEFAULT_C_VALUES,
    ProbeResult,
    results_to_frame,
    ridge_readout,
    run_probe,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeSpec:
    """One rung of the ladder."""

    name: str
    blocks: tuple[str, ...]
    needs_hidden: bool
    shuffle_labels: bool = False
    description: str = ""


# SPEC 3.3, plus two additions.
#
# P2r exists because P2 as literally specified cannot work: logistic regression
# on `a_cmd` concatenated with `dstates` has no way to form the *agreement*
# between a command and an outcome, so it lands near chance even when the
# labels are fully determined by the mechanics.  P2 is kept as the honest
# "linear oracle" number; P2r supplies the interaction terms and is the real
# mechanical ceiling -- and the one to check when validating the plumbing.
#
# C_cmd is a leakage check: the commanded chunk is produced *before* the hijack,
# so on its own it must sit at chance.  If it does not, the schedule or the
# phase matching is correlated with something it should not be.
PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec("P0", ("h_cur",), True, shuffle_labels=True, description="chance floor"),
    ProbeSpec("P1", ("h_cur",), True, description="anomaly trace without efference"),
    ProbeSpec(
        "P2", ("a_cmd", "dstates"), False, description="linear oracle (as specified)"
    ),
    ProbeSpec(
        "P2r",
        ("mismatch",),
        False,
        description="mechanical oracle with comparison features",
    ),
    ProbeSpec(
        "P3", ("a_cmd", "h_cur"), True, description="efference-augmented decodability"
    ),
    ProbeSpec(
        "P4",
        ("h_prev", "h_cur"),
        True,
        description="internal-only efference availability",
    ),
    ProbeSpec(
        "P5",
        ("a_cmd", "vis_cur"),
        False,
        description="perception-only baseline (stretch)",
    ),
    ProbeSpec(
        "C_cmd", ("a_cmd",), False, description="control: command alone must be chance"
    ),
    ProbeSpec(
        "C_dstates",
        ("dstates",),
        False,
        description="control: trivially mechanical part",
    ),
)

PROBE_BY_NAME = {spec.name: spec for spec in PROBE_SPECS}


@dataclass
class AnalysisConfig:
    """Knobs for a probe sweep."""

    layers: Optional[list[int]] = None
    pools: Optional[list[str]] = None
    probes: Optional[list[str]] = None
    c_values: tuple[float, ...] = DEFAULT_C_VALUES
    n_splits: int = 5
    seed: int = 0
    select_c: bool = True
    block_scaling: str = "none"
    a_cmd_column: str = "a_cmd_env"
    negatives: str = "phase_matched"
    max_cache_gb: float = 8.0
    cross_task_splits: int = 4


def _scheduler_settings(run: RunData) -> tuple[int, int]:
    """Recover the collection-time gap and warm-up from the run config."""
    hijack = (run.config.get("config") or {}).get("hijack") or {}
    return int(hijack.get("min_clean_gap", 2)), int(hijack.get("warmup_calls", 4))


def make_samples(
    run: RunData, config: AnalysisConfig, negatives: Optional[str] = None
) -> pd.DataFrame:
    min_clean_gap, warmup_calls = _scheduler_settings(run)
    return build_probe_samples(
        run.calls,
        min_clean_gap=min_clean_gap,
        warmup_calls=warmup_calls,
        negatives=negatives or config.negatives,
        rng=np.random.default_rng(config.seed),
    )


def run_ladder(
    run: RunData,
    samples: pd.DataFrame,
    config: AnalysisConfig,
    store: Optional[HiddenStore] = None,
) -> list[ProbeResult]:
    """Run every requested probe, sweeping layers/pools for hidden-state ones."""
    if samples.empty:
        logger.warning("no probe samples; nothing to run")
        return []
    store = store or HiddenStore(run.run_dir / "hidden", config.max_cache_gb)
    layers = config.layers if config.layers is not None else run.layers
    pools = config.pools if config.pools is not None else run.pools
    names = (
        config.probes if config.probes is not None else [s.name for s in PROBE_SPECS]
    )

    groups = samples["episode_id"].to_numpy()
    labels = samples["y"].to_numpy()
    results: list[ProbeResult] = []

    for name in names:
        spec = PROBE_BY_NAME.get(name)
        if spec is None:
            raise ValueError(f"unknown probe {name!r}; have {sorted(PROBE_BY_NAME)}")
        combos = (
            [(layer, pool) for layer in layers for pool in pools]
            if spec.needs_hidden
            else [(None, None)]
        )
        for layer, pool in combos:
            try:
                features, _spans = build_features(
                    run,
                    samples,
                    list(spec.blocks),
                    layer=layer,
                    pool=pool,
                    store=store,
                    a_cmd_column=config.a_cmd_column,
                    block_scaling=config.block_scaling,
                )
            except (KeyError, FileNotFoundError) as error:
                logger.warning(
                    "skipping %s (layer=%s pool=%s): %s", name, layer, pool, error
                )
                continue
            result = run_probe(
                features,
                labels,
                groups,
                name=name,
                blocks=list(spec.blocks),
                layer=layer,
                pool=pool,
                c_values=config.c_values,
                n_splits=config.n_splits,
                seed=config.seed,
                shuffle_labels=spec.shuffle_labels,
                select_c=config.select_c,
            )
            result.notes = (result.notes + " " + spec.description).strip()
            results.append(result)
            logger.info(
                "%s layer=%s pool=%s bacc=%.3f auroc=%.3f",
                name,
                layer,
                pool,
                result.balanced_acc_mean,
                result.auroc_mean,
            )
    return results


def run_per_transform(
    run: RunData, samples: pd.DataFrame, config: AnalysisConfig, layer: int, pool: str
) -> list[ProbeResult]:
    """Split the ladder by hijack transform (SPEC 3.3a).

    Each positive keeps its matched negative, so a per-transform subset stays
    phase-balanced.
    """
    store = HiddenStore(run.run_dir / "hidden", config.max_cache_gb)
    results: list[ProbeResult] = []
    transforms = sorted(
        t for t in samples.loc[samples["y"] == 1, "transform"].unique() if t
    )
    for transform in transforms:
        pair_ids = set(
            samples.loc[
                (samples["y"] == 1) & (samples["transform"] == transform), "pair_id"
            ]
        )
        subset = samples[samples["pair_id"].isin(pair_ids)]
        if subset["y"].nunique() < 2:
            continue
        for name in ("P1", "P2r", "P3", "P4"):
            spec = PROBE_BY_NAME[name]
            features, _ = build_features(
                run,
                subset,
                list(spec.blocks),
                layer=layer if spec.needs_hidden else None,
                pool=pool if spec.needs_hidden else None,
                store=store,
                a_cmd_column=config.a_cmd_column,
                block_scaling=config.block_scaling,
            )
            result = run_probe(
                features,
                subset["y"].to_numpy(),
                subset["episode_id"].to_numpy(),
                name=f"{name}[{transform}]",
                blocks=list(spec.blocks),
                layer=layer if spec.needs_hidden else None,
                pool=pool if spec.needs_hidden else None,
                c_values=config.c_values,
                n_splits=min(config.n_splits, subset["episode_id"].nunique()),
                seed=config.seed,
                select_c=config.select_c,
            )
            result.notes = f"transform={transform}"
            results.append(result)
    return results


def run_cross_task(
    run: RunData, samples: pd.DataFrame, config: AnalysisConfig, layer: int, pool: str
) -> pd.DataFrame:
    """Train on some tasks, test on held-out tasks (SPEC 3.3b, figure F3)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    store = HiddenStore(run.run_dir / "hidden", config.max_cache_gb)
    task_ids = samples["task_id"].to_numpy()
    n_tasks = len(np.unique(task_ids))
    n_splits = min(config.cross_task_splits, n_tasks)
    if n_splits < 2:
        logger.warning("cross-task split needs >= 2 tasks; have %d", n_tasks)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name in ("P1", "P3", "P4"):
        spec = PROBE_BY_NAME[name]
        features, _ = build_features(
            run,
            samples,
            list(spec.blocks),
            layer=layer,
            pool=pool,
            store=store,
            a_cmd_column=config.a_cmd_column,
            block_scaling=config.block_scaling,
        )
        labels = samples["y"].to_numpy()
        splitter = GroupKFold(n_splits=n_splits)
        fold_balanced, fold_auroc, held_out = [], [], []
        for train_index, test_index in splitter.split(features, labels, task_ids):
            if len(np.unique(labels[test_index])) < 2:
                continue
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=max(config.c_values),
                            solver="lbfgs",
                            max_iter=2000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            model.fit(features[train_index], labels[train_index])
            predicted = model.predict(features[test_index])
            fold_balanced.append(balanced_accuracy_score(labels[test_index], predicted))
            fold_auroc.append(
                roc_auc_score(
                    labels[test_index], model.decision_function(features[test_index])
                )
            )
            held_out.append(sorted(np.unique(task_ids[test_index]).tolist()))
        rows.append(
            {
                "probe": name,
                "layer": layer,
                "pool": pool,
                "split": "held_out_task",
                "balanced_acc_mean": float(np.mean(fold_balanced))
                if fold_balanced
                else np.nan,
                "balanced_acc_std": float(np.std(fold_balanced, ddof=1))
                if len(fold_balanced) > 1
                else np.nan,
                "auroc_mean": float(np.mean(fold_auroc)) if fold_auroc else np.nan,
                "n_folds": len(fold_balanced),
                "held_out_tasks": json.dumps(held_out),
            }
        )
    return pd.DataFrame(rows)


def run_e1(
    run: RunData, config: AnalysisConfig, max_samples: int = 6000
) -> pd.DataFrame:
    """E1: ridge-regress the commanded chunk from the hidden state (figure F4).

    Uses every recorded call, not just probe samples: the question is where the
    motor plan lives, which has nothing to do with the hijack labels.
    """
    store = HiddenStore(run.run_dir / "hidden", config.max_cache_gb)
    calls = run.calls
    if "post_success" in calls.columns:
        calls = calls[~calls["post_success"].astype(bool)]
    if len(calls) > max_samples:
        calls = calls.sample(max_samples, random_state=config.seed)
    calls = calls.sort_values(["episode_id", "call_idx"])

    pairs = list(zip(calls["episode_id"], calls["call_idx"]))
    targets = np.stack(
        [np.asarray(v, dtype=np.float32) for v in calls[config.a_cmd_column]]
    )
    groups = calls["episode_id"].to_numpy()

    layers = config.layers if config.layers is not None else run.layers
    pools = config.pools if config.pools is not None else run.pools
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for pool in pools:
            features = store.vectors(
                pairs, run.layer_index(layer), run.pool_index(pool)
            )
            readout = ridge_readout(
                features, targets, groups, n_splits=config.n_splits, seed=config.seed
            )
            rows.append(
                {
                    "layer": layer,
                    "pool": pool,
                    "r2_mean": readout["r2_mean"],
                    "r2_std": readout["r2_std"],
                    "n_samples": readout["n_samples"],
                    "n_features": readout["n_features"],
                }
            )
            logger.info("E1 layer=%s pool=%s R2=%.3f", layer, pool, readout["r2_mean"])
    return pd.DataFrame(rows)


def run_e3(run: RunData, samples: pd.DataFrame) -> dict[str, Any]:
    """E3: behaviour after a hijack, from already-logged data only."""
    calls = run.calls
    lookup = {
        (int(e), int(c)): position
        for position, (e, c) in enumerate(zip(calls["episode_id"], calls["call_idx"]))
    }
    entropy = calls["entropy_mean"].to_numpy()
    logprob = calls["logprob_sum"].to_numpy()

    positions = np.asarray(
        [
            lookup[(int(e), int(c))]
            for e, c in zip(samples["episode_id"], samples["cur_call"])
        ]
    )
    labels = samples["y"].to_numpy()
    report: dict[str, Any] = {
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
    }
    for metric_name, values in (("entropy_mean", entropy), ("logprob_sum", logprob)):
        after_hijack = values[positions[labels == 1]]
        after_self = values[positions[labels == 0]]
        report[metric_name] = {
            "hijack_mean": float(np.mean(after_hijack)),
            "hijack_std": float(np.std(after_hijack, ddof=1)),
            "self_mean": float(np.mean(after_self)),
            "self_std": float(np.std(after_self, ddof=1)),
            "difference": float(np.mean(after_hijack) - np.mean(after_self)),
        }
        report[metric_name].update(
            _paired_test(samples, values, positions, labels, metric_name)
        )

    report["success_rate"] = _success_rates(run)
    return report


def _paired_test(
    samples: pd.DataFrame,
    values: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    metric_name: str,
) -> dict[str, Any]:
    """Paired comparison across matched (positive, negative) pairs."""
    if "pair_id" not in samples.columns:
        return {}
    frame = pd.DataFrame(
        {
            "pair_id": samples["pair_id"].to_numpy(),
            "y": labels,
            "value": values[positions],
        }
    )
    wide = frame.pivot_table(index="pair_id", columns="y", values="value")
    if 0 not in wide.columns or 1 not in wide.columns:
        return {}
    wide = wide.dropna()
    if wide.empty:
        return {}
    differences = (wide[1] - wide[0]).to_numpy()
    result = {
        "paired_n": int(len(differences)),
        "paired_mean_difference": float(np.mean(differences)),
        "paired_std_difference": float(np.std(differences, ddof=1)),
    }
    try:
        from scipy import stats

        statistic, p_value = stats.wilcoxon(differences)
        result["wilcoxon_statistic"] = float(statistic)
        result["wilcoxon_p"] = float(p_value)
    except (ImportError, ValueError):
        pass
    return result


def _success_rates(run: RunData) -> dict[str, Any]:
    """Episode success rate, probe episodes vs clean episodes."""
    episodes = pd.DataFrame(run.manifest.get("episodes", []))
    if episodes.empty:
        return {}
    report: dict[str, Any] = {
        "overall": float(episodes["success"].mean()),
        "n_episodes": int(len(episodes)),
    }
    if "is_probe_episode" in episodes.columns:
        for flag, label in ((True, "probe_episodes"), (False, "clean_episodes")):
            subset = episodes[episodes["is_probe_episode"] == flag]
            if not subset.empty:
                report[label] = {
                    "success_rate": float(subset["success"].mean()),
                    "n": int(len(subset)),
                }
    if "n_hijacks" in episodes.columns and episodes["n_hijacks"].nunique() > 1:
        report["corr_hijacks_success"] = float(
            episodes["n_hijacks"].corr(episodes["success"].astype(float))
        )
    return report


def undo_alignment(run: RunData, samples: pd.DataFrame) -> dict[str, Any]:
    """E3(iii) stretch: does the next command point back along the error?

    The controller's command-to-displacement gain is unknown, so it is fit on
    self-caused calls first: ``delta_eef ~ k * sum(a_cmd[:3])``.  The residual
    on the manipulated call is the displacement the policy did not ask for; we
    then measure how much of the *next* command lies along its negative.
    """
    schema = run.schema
    n_chunks, action_dim = int(schema["num_action_chunks"]), int(schema["action_dim"])
    calls = run.calls
    lookup = {
        (int(e), int(c)): position
        for position, (e, c) in enumerate(zip(calls["episode_id"], calls["call_idx"]))
    }

    def summed_command(position: int) -> np.ndarray:
        chunk = np.asarray(calls["a_cmd_env"].iloc[position], dtype=np.float32)
        return chunk.reshape(n_chunks, action_dim).sum(axis=0)[:3]

    def displacement(position: int) -> np.ndarray:
        after = np.asarray(calls["states_after"].iloc[position], dtype=np.float32)
        before = np.asarray(calls["states_before"].iloc[position], dtype=np.float32)
        return (after - before)[:3]

    self_positions = [
        position for position, label in enumerate(calls["label"]) if label == "SELF"
    ]
    if not self_positions:
        return {}
    command_matrix = np.stack([summed_command(p) for p in self_positions]).reshape(-1)
    displacement_matrix = np.stack([displacement(p) for p in self_positions]).reshape(
        -1
    )
    denominator = float(command_matrix @ command_matrix)
    gain = (
        float(command_matrix @ displacement_matrix) / denominator
        if denominator
        else 0.0
    )

    alignments: dict[int, list[float]] = {0: [], 1: []}
    for _, sample in samples.iterrows():
        previous = lookup.get((int(sample["episode_id"]), int(sample["prev_call"])))
        current = lookup.get((int(sample["episode_id"]), int(sample["cur_call"])))
        if previous is None or current is None:
            continue
        error = displacement(previous) - gain * summed_command(previous)
        next_command = summed_command(current)
        norms = np.linalg.norm(error) * np.linalg.norm(next_command)
        if norms < 1e-8:
            continue
        alignments[int(sample["y"])].append(float(-(error @ next_command) / norms))

    return {
        "controller_gain": gain,
        "hijack_mean_alignment": float(np.mean(alignments[1]))
        if alignments[1]
        else float("nan"),
        "self_mean_alignment": float(np.mean(alignments[0]))
        if alignments[0]
        else float("nan"),
        "n_hijack": len(alignments[1]),
        "n_self": len(alignments[0]),
    }


def to_markdown(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a markdown table.

    Hand-rolled because ``DataFrame.to_markdown`` needs ``tabulate``, and the
    project is capped at scikit-learn / matplotlib / pandas (SPEC 0.5).
    """
    frame = frame.reset_index() if index else frame
    headers = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    rows = [
        [render(value) for value in record] for record in frame.itertuples(index=False)
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows
        else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    lines += [
        "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def summarize(
    run: RunData,
    samples: pd.DataFrame,
    results: list[ProbeResult],
    out_dir: Path,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write CSVs plus a markdown block for ``results.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = results_to_frame(results)
    frame.to_csv(out_dir / "probe_results.csv", index=False)
    samples.to_csv(out_dir / "probe_samples.csv", index=False)
    phase = step_index_report(samples)
    phase.to_csv(out_dir / "step_index_report.csv")

    lines = [
        f"### Run `{run.config.get('run_id')}` (git `{run.config.get('git_sha', '')[:12]}`)",
        "",
        f"- config hash: `{run.config.get('config_hash')}`",
        f"- calls: {len(run.calls)}, episodes: {run.calls['episode_id'].nunique()}",
        f"- probe samples: {len(samples)} "
        f"({int((samples['y'] == 1).sum())} positive / "
        f"{int((samples['y'] == 0).sum())} negative)",
        "",
        "#### Class step-index distributions (must overlap)",
        "",
        to_markdown(phase.round(2), index=True),
        "",
        "#### Probe results",
        "",
    ]
    if not frame.empty:
        columns = [
            "name",
            "blocks",
            "layer",
            "pool",
            "n_features",
            "balanced_acc_mean",
            "balanced_acc_std",
            "auroc_mean",
        ]
        table = frame[[c for c in columns if c in frame.columns]].copy()
        for column in ("balanced_acc_mean", "balanced_acc_std", "auroc_mean"):
            if column in table:
                table[column] = table[column].round(3)
        lines.append(to_markdown(table))
    if extra:
        lines += [
            "",
            "#### Extra",
            "",
            "```json",
            json.dumps(extra, indent=2, default=str),
            "```",
        ]

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


__all__ = [
    "AnalysisConfig",
    "PROBE_SPECS",
    "load_run",
    "make_samples",
    "run_cross_task",
    "run_e1",
    "run_e3",
    "run_ladder",
    "run_per_transform",
    "summarize",
    "to_markdown",
    "undo_alignment",
]

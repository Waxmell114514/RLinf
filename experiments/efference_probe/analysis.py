"""Drives the probe ladder, the controls, E1 and E3 over a collection run.

Produces CSV tables plus a markdown block ready to paste into ``results.md``.
Reports numbers and anomalies only -- interpretation is the applicant's job
(SPEC 8.2).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import GroupKFold

from .datasets import (
    HiddenStore,
    RunData,
    build_features,
    build_probe_samples,
    count_raw_positives,
    load_run,
    phase_bias_report,
    step_index_report,
)
from .hijack import HIJACK, SELF, apply_freeze, apply_mirror
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
# "linear oracle" number; P2r supplies explicit comparison terms, but remains
# an empirical endpoint-state baseline.  Plumbing is validated directly by
# plumbing_report(), without assuming any controller dynamics.
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
        description="endpoint mismatch baseline with comparison features",
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
    ProbeSpec(
        "C_phase",
        ("phase",),
        False,
        description="control: task phase alone; bounds residual phase confound",
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
    # Cells (probe x layer x pool) are independent fits, so they parallelise
    # exactly: same seeds, same splits, same numbers, N times less wall clock.
    # Defaults to serial so library callers and the test suite stay
    # single-process; the CLI turns it up.
    n_jobs: int = 1


def _scheduler_settings(run: RunData) -> tuple[int, int]:
    """Recover the collection-time gap and warm-up from the run config.

    Deliberately raises rather than defaulting: both numbers define which calls
    may belong to which class, so guessing them wrong mis-specifies the whole
    sample set without any visible symptom.
    """
    hijack = (run.config.get("config") or {}).get("hijack") or {}
    missing = [key for key in ("min_clean_gap", "warmup_calls") if key not in hijack]
    if missing:
        raise KeyError(
            f"run_config.yaml is missing config.hijack.{missing} -- these set the "
            "class boundaries for probe samples and must not be guessed"
        )
    return int(hijack["min_clean_gap"]), int(hijack["warmup_calls"])


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


def plumbing_report(run: RunData, atol: float = 1e-6) -> dict[str, Any]:
    """Check logged labels, executed transforms, donors, and state continuity."""
    required = {
        "episode_id",
        "call_idx",
        "label",
        "transform",
        "donor_episode_id",
        "a_cmd_env",
        "a_exec",
        "states_before",
        "states_after",
    }
    missing = sorted(required - set(run.calls.columns))
    if missing:
        raise KeyError(f"calls.parquet is missing plumbing columns: {missing}")

    calls = run.calls.reset_index(drop=True)
    shape = (int(run.schema["num_action_chunks"]), int(run.schema["action_dim"]))
    lookup = {
        (int(row.episode_id), int(row.call_idx)): index
        for index, row in enumerate(calls.itertuples(index=False))
    }
    action_mismatches = 0
    missing_donors = unknown_labels = effective_hijacks = 0
    n_self = n_hijack = 0
    max_action_error = 0.0

    for row in calls.itertuples(index=False):
        commanded = np.asarray(row.a_cmd_env, dtype=np.float32).reshape(shape)
        executed = np.asarray(row.a_exec, dtype=np.float32).reshape(shape)
        if row.label == SELF:
            n_self += 1
            expected = commanded
        elif row.label == HIJACK:
            n_hijack += 1
            effective_hijacks += int(
                not np.allclose(commanded, executed, rtol=0.0, atol=atol)
            )
            if row.transform == "mirror":
                expected = apply_mirror(commanded)
            elif row.transform == "freeze":
                expected = apply_freeze(commanded)
            elif row.transform == "swap":
                donor = lookup.get((int(row.donor_episode_id), int(row.call_idx)))
                if donor is None:
                    missing_donors += 1
                    action_mismatches += 1
                    continue
                expected = np.asarray(
                    calls.iloc[donor]["a_cmd_env"], dtype=np.float32
                ).reshape(shape)
            else:
                action_mismatches += 1
                continue
        else:
            unknown_labels += 1
            action_mismatches += 1
            continue

        error = float(np.max(np.abs(expected - executed)))
        max_action_error = max(max_action_error, error)
        action_mismatches += int(error > atol)

    state_links = state_discontinuities = 0
    max_state_error = 0.0
    for _episode_id, episode in calls.groupby("episode_id", sort=False):
        rows = list(episode.sort_values("call_idx").itertuples(index=False))
        for previous, current in zip(rows, rows[1:]):
            if int(current.call_idx) != int(previous.call_idx) + 1:
                continue
            state_links += 1
            error = float(
                np.max(
                    np.abs(
                        np.asarray(previous.states_after, dtype=np.float32)
                        - np.asarray(current.states_before, dtype=np.float32)
                    )
                )
            )
            max_state_error = max(max_state_error, error)
            state_discontinuities += int(error > atol)

    passed = bool(
        len(calls)
        and n_hijack
        and not action_mismatches
        and not state_discontinuities
        and not missing_donors
        and not unknown_labels
    )
    return {
        "passed": passed,
        "n_rows": len(calls),
        "n_self": n_self,
        "n_hijack": n_hijack,
        "n_effective_hijacks": effective_hijacks,
        "n_action_mismatches": action_mismatches,
        "n_missing_donors": missing_donors,
        "n_unknown_labels": unknown_labels,
        "max_action_abs_error": max_action_error,
        "n_state_links": state_links,
        "n_state_discontinuities": state_discontinuities,
        "max_state_abs_error": max_state_error,
        "atol": atol,
    }


def _resolve_n_jobs(n_jobs: int) -> int:
    """Turn a joblib-style ``n_jobs`` into a concrete worker count.

    ``-1`` means every core, ``-2`` all but one, and anything that resolves
    below 2 runs serially in-process -- which keeps the default path free of
    process spawning, and so identical on Windows, in pytest, and in CI.
    """
    if n_jobs is None:
        return 1
    if n_jobs < 0:
        available = os.cpu_count() or 1
        return max(1, available + 1 + n_jobs)
    return max(1, n_jobs)


def _run_cell(
    cell: tuple[str, ProbeSpec, Optional[int], Optional[str], np.ndarray, list],
    labels: np.ndarray,
    groups: np.ndarray,
    config: AnalysisConfig,
) -> ProbeResult:
    """Fit one (probe, layer, pool) cell.

    Module level and free of the run/store, so joblib can ship it to a worker
    process.  Everything that decides the numbers -- splits, seeds, C grid --
    comes from ``config``, so a cell scores identically wherever it runs.
    """
    name, spec, layer, pool, features, spans = cell
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
        spans=spans if config.block_scaling == "sqrt_dim" else None,
    )
    result.notes = (result.notes + " " + spec.description).strip()
    return result


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

    def _cells():
        """Yield one ready-to-fit cell at a time.

        A generator on purpose: the hidden blocks are tens of megabytes each,
        and joblib pulls lazily, so only the cells actually in flight are
        held in memory rather than all of them at once.
        """
        for name in names:
            spec = PROBE_BY_NAME.get(name)
            if spec is None:
                raise ValueError(
                    f"unknown probe {name!r}; have {sorted(PROBE_BY_NAME)}"
                )
            combos = (
                [(layer, pool) for layer in layers for pool in pools]
                if spec.needs_hidden
                else [(None, None)]
            )
            for layer, pool in combos:
                try:
                    features, spans = build_features(
                        run,
                        samples,
                        list(spec.blocks),
                        layer=layer,
                        pool=pool,
                        store=store,
                        a_cmd_column=config.a_cmd_column,
                    )
                except (KeyError, FileNotFoundError) as error:
                    logger.warning(
                        "skipping %s (layer=%s pool=%s): %s", name, layer, pool, error
                    )
                    continue
                yield (name, spec, layer, pool, features, spans)

    n_jobs = _resolve_n_jobs(config.n_jobs)
    if n_jobs == 1:
        results = [_run_cell(cell, labels, groups, config) for cell in _cells()]
    else:
        # inner_max_num_threads=1: each worker already owns a core, so letting
        # its BLAS fan out too would oversubscribe and run slower than serial.
        with parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
            results = Parallel()(
                delayed(_run_cell)(cell, labels, groups, config)
                for cell in _cells()
            )

    for result in results:
        logger.info(
            "%s layer=%s pool=%s bacc=%.3f auroc=%.3f",
            result.name,
            result.layer,
            result.pool,
            result.balanced_acc_mean,
            result.auroc_mean,
        )
    return list(results)


def run_per_transform(
    run: RunData,
    samples: pd.DataFrame,
    config: AnalysisConfig,
    layer: int,
    pool: str,
    store: Optional[HiddenStore] = None,
) -> list[ProbeResult]:
    """Split the ladder by hijack transform (SPEC 3.3a).

    Each positive keeps its matched negative, so a per-transform subset stays
    phase-balanced.
    """
    store = store or HiddenStore(run.run_dir / "hidden", config.max_cache_gb)
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
            features, spans = build_features(
                run,
                subset,
                list(spec.blocks),
                layer=layer if spec.needs_hidden else None,
                pool=pool if spec.needs_hidden else None,
                store=store,
                a_cmd_column=config.a_cmd_column,
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
                n_splits=max(2, min(config.n_splits, subset["episode_id"].nunique())),
                seed=config.seed,
                select_c=config.select_c,
                spans=spans if config.block_scaling == "sqrt_dim" else None,
            )
            result.notes = f"transform={transform}"
            results.append(result)
    return results


def run_cross_task(
    run: RunData,
    samples: pd.DataFrame,
    config: AnalysisConfig,
    layer: int,
    pool: str,
    store: Optional[HiddenStore] = None,
) -> pd.DataFrame:
    """Train on some tasks, test on held-out tasks (SPEC 3.3b, figure F3)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    store = store or HiddenStore(run.run_dir / "hidden", config.max_cache_gb)
    task_ids = samples["task_id"].to_numpy()
    n_tasks = len(np.unique(task_ids))
    n_splits = min(config.cross_task_splits, n_tasks)
    if n_splits < 2:
        logger.warning("cross-task split needs >= 2 tasks; have %d", n_tasks)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name in ("P1", "P3", "P4"):
        spec = PROBE_BY_NAME[name]
        features, _spans = build_features(
            run,
            samples,
            list(spec.blocks),
            layer=layer,
            pool=pool,
            store=store,
            a_cmd_column=config.a_cmd_column,
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
    run: RunData,
    config: AnalysisConfig,
    max_samples: int = 6000,
    store: Optional[HiddenStore] = None,
) -> pd.DataFrame:
    """E1: ridge-regress the commanded chunk from the hidden state (figure F4).

    Uses every recorded call, not just probe samples: the question is where the
    motor plan lives, which has nothing to do with the hijack labels.
    """
    store = store or HiddenStore(run.run_dir / "hidden", config.max_cache_gb)
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
            per_dim = np.asarray(readout["per_dim_r2"])
            rows.append(
                {
                    "layer": layer,
                    "pool": pool,
                    "r2_mean": readout["r2_mean"],
                    "r2_std": readout["r2_std"],
                    # A single near-constant output dimension (the gripper
                    # token often is) drags the uniform average down, and a
                    # boundary-pinned alpha silently under-fits.  Both are
                    # invisible unless reported.
                    "r2_median_per_dim": float(np.median(per_dim)),
                    "r2_min_per_dim": float(per_dim.min()),
                    "selected_alpha": json.dumps(readout["selected_alpha"]),
                    "alpha_at_grid_edge": bool(readout["alpha_at_grid_edge"]),
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
    if (samples["pair_id"] < 0).any():
        # Global-pool negatives are not genuinely paired; a "paired" test on
        # them would be meaningless.
        return {"paired_n": 0, "note": "samples are not paired; no paired test"}
    frame = pd.DataFrame(
        {
            "pair_id": samples["pair_id"].to_numpy(),
            "episode_id": samples["episode_id"].to_numpy(),
            "y": labels,
            "value": values[positions],
        }
    )
    wide = frame.pivot_table(
        index=["episode_id", "pair_id"], columns="y", values="value"
    )
    if 0 not in wide.columns or 1 not in wide.columns:
        return {}
    wide = wide.dropna()
    if wide.empty:
        return {}
    differences = (wide[1] - wide[0]).to_numpy()
    # Pairs are not independent replicates: an episode contributes several, and
    # the effect plausibly varies by episode and task.  Testing at the pair
    # level inflates type-I error once that heterogeneity exists, so the test
    # runs on per-episode mean differences and the pair-level count is reported
    # alongside for transparency.
    per_episode = (wide[1] - wide[0]).groupby(level="episode_id").mean().to_numpy()
    result = {
        "paired_n": int(len(differences)),
        "paired_n_episodes": int(len(per_episode)),
        "paired_mean_difference": float(np.mean(differences)),
        "paired_std_difference": float(np.std(differences, ddof=1)),
    }
    try:
        from scipy import stats

        statistic, p_value = stats.wilcoxon(per_episode)
        result["wilcoxon_statistic"] = float(statistic)
        result["wilcoxon_p"] = float(p_value)
        result["wilcoxon_unit"] = "episode"
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
    # A correlation between per-episode hijack count and success is *not*
    # reported: hijacks can only accumulate while an episode runs and
    # successful episodes end early, so the correlation is strongly negative
    # even when hijacking has no causal effect at all.  The probe-vs-clean
    # contrast above is randomised per episode, so it is the unconfounded
    # comparison; use that one.
    report["note"] = (
        "compare probe_episodes vs clean_episodes (randomised); episode-level "
        "hijack-count correlations are confounded by episode length"
    )
    return report


def undo_alignment(
    run: RunData, samples: pd.DataFrame, a_cmd_column: str = "a_cmd_env"
) -> dict[str, Any]:
    """E3(iii) stretch: does the next command point back along the error?

    Two corrections make this measure what it claims to.

    First, the *error*.  The controller gain is unknown, so it is fit on
    self-caused calls: ``delta_eef ~ gain * sum(a_cmd[:3]) + offset``, per axis
    and with an intercept, on pre-success rows only.  The residual on the
    manipulated call is the displacement the policy did not ask for.

    Second -- and this is the part that makes or breaks the metric -- the *next
    command*.  Scoring the raw next command against the negated error measures
    nothing about correction: for ``freeze`` the residual is exactly
    ``-gain * A_prev`` and for ``mirror`` exactly ``-2 * gain * [A_x, A_y, 0]``,
    so the score reduces to ``+(A_prev . A_cur)``.  Any temporal
    autocorrelation in the commanded chunk then yields a large positive
    "undo alignment" from a policy that never looked at the state at all --
    and the SELF control does not catch it, because for SELF samples the
    residual is pure noise and the control sits at zero by construction.

    So the next command is replaced by its *innovation*: the part not predicted
    by simply continuing the previous command, ``A_cur - rho * A_prev``, with
    ``rho`` estimated from consecutive self-caused calls.
    """
    schema = run.schema
    n_chunks, action_dim = int(schema["num_action_chunks"]), int(schema["action_dim"])
    calls = run.calls
    if "post_success" in calls.columns:
        clean = calls[~calls["post_success"].astype(bool)]
    else:
        clean = calls
    lookup = {
        (int(e), int(c)): position
        for position, (e, c) in enumerate(zip(calls["episode_id"], calls["call_idx"]))
    }
    commands = calls[a_cmd_column].to_numpy()
    states_after = calls["states_after"].to_numpy()
    states_before = calls["states_before"].to_numpy()

    def summed_command(position: int) -> np.ndarray:
        chunk = np.asarray(commands[position], dtype=np.float64)
        return chunk.reshape(n_chunks, action_dim).sum(axis=0)[:3]

    def displacement(position: int) -> np.ndarray:
        after = np.asarray(states_after[position], dtype=np.float64)
        before = np.asarray(states_before[position], dtype=np.float64)
        return (after - before)[:3]

    self_positions = [
        lookup[(int(e), int(c))]
        for e, c, label in zip(clean["episode_id"], clean["call_idx"], clean["label"])
        if label == "SELF"
    ]
    if len(self_positions) < 8:
        return {"note": "not enough self-caused calls to fit the controller gain"}

    command_rows = np.stack([summed_command(p) for p in self_positions])
    displacement_rows = np.stack([displacement(p) for p in self_positions])

    # Per-axis gain with an intercept: the three translation axes have
    # different controller scaling, and a shared gainless fit absorbs that into
    # the residual we are about to call "the displacement nobody asked for".
    gains = np.zeros(3)
    offsets = np.zeros(3)
    for axis in range(3):
        design = np.stack([command_rows[:, axis], np.ones(len(command_rows))], axis=1)
        solution, *_ = np.linalg.lstsq(design, displacement_rows[:, axis], rcond=None)
        gains[axis], offsets[axis] = solution

    rho = _command_autocorrelation(clean, lookup, summed_command)

    alignments: dict[int, list[float]] = {0: [], 1: []}
    raw_alignments: dict[int, list[float]] = {0: [], 1: []}
    for _, sample in samples.iterrows():
        previous = lookup.get((int(sample["episode_id"]), int(sample["prev_call"])))
        current = lookup.get((int(sample["episode_id"]), int(sample["cur_call"])))
        if previous is None or current is None:
            continue
        previous_command = summed_command(previous)
        error = displacement(previous) - (gains * previous_command + offsets)
        next_command = summed_command(current)
        innovation = next_command - rho * previous_command
        label = int(sample["y"])
        for target, vector in (
            (alignments, innovation),
            (raw_alignments, next_command),
        ):
            norms = np.linalg.norm(error) * np.linalg.norm(vector)
            if norms >= 1e-8:
                target[label].append(float(-(error @ vector) / norms))

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "controller_gain_per_axis": gains.tolist(),
        "controller_offset_per_axis": offsets.tolist(),
        "command_autocorrelation_rho": rho,
        "hijack_mean_alignment": mean(alignments[1]),
        "self_mean_alignment": mean(alignments[0]),
        "n_hijack": len(alignments[1]),
        "n_self": len(alignments[0]),
        # Kept only so the artefact stays visible: this is the uncorrected
        # number, which tracks command autocorrelation rather than correction.
        "uncorrected_hijack_mean_alignment": mean(raw_alignments[1]),
        "uncorrected_self_mean_alignment": mean(raw_alignments[0]),
    }


def _command_autocorrelation(calls: pd.DataFrame, lookup, summed_command) -> float:
    """Least-squares rho in ``A_{m+1} ~ rho * A_m`` over self-caused pairs."""
    previous_rows, next_rows = [], []
    labels = {
        (int(e), int(c)): label
        for e, c, label in zip(calls["episode_id"], calls["call_idx"], calls["label"])
    }
    for (episode_id, call_idx), label in labels.items():
        if label != "SELF" or labels.get((episode_id, call_idx + 1)) != "SELF":
            continue
        previous = lookup.get((episode_id, call_idx))
        current = lookup.get((episode_id, call_idx + 1))
        if previous is None or current is None:
            continue
        previous_rows.append(summed_command(previous))
        next_rows.append(summed_command(current))
    if len(previous_rows) < 8:
        return 0.0
    previous_flat = np.concatenate(previous_rows)
    next_flat = np.concatenate(next_rows)
    denominator = float(previous_flat @ previous_flat)
    return float(previous_flat @ next_flat) / denominator if denominator else 0.0


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


def sample_accounting(run: RunData, samples: pd.DataFrame) -> dict[str, Any]:
    """Kept vs available positives, and the signed phase check.

    Paired matching drops positives it cannot pair, non-randomly (episodes with
    dense hijacks lose the most).  Reporting only the kept count would let a
    write-up quote "n post-hijack calls" without noting the exclusion.
    """
    min_clean_gap, warmup_calls = _scheduler_settings(run)
    raw = count_raw_positives(
        run.calls, min_clean_gap=min_clean_gap, warmup_calls=warmup_calls
    )
    kept = int((samples["y"] == 1).sum())
    return {
        "n_positive_available": raw,
        "n_positive_kept": kept,
        "n_positive_dropped": raw - kept,
        "fraction_kept": (kept / raw) if raw else float("nan"),
        "phase_bias": phase_bias_report(samples),
    }


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

    accounting = sample_accounting(run, samples)
    lines = [
        f"### Run `{run.config.get('run_id')}` (git `{run.config.get('git_sha', '')[:12]}`)",
        "",
        f"- config hash: `{run.config.get('config_hash')}`",
        f"- calls: {len(run.calls)}, episodes: {run.calls['episode_id'].nunique()}",
        f"- probe samples: {len(samples)} "
        f"({int((samples['y'] == 1).sum())} positive / "
        f"{int((samples['y'] == 0).sum())} negative)",
        f"- positives available: {accounting['n_positive_available']}, "
        f"kept after pairing: {accounting['n_positive_kept']} "
        f"(dropped {accounting['n_positive_dropped']}; pairing is not random, "
        f"so report this alongside n)",
        "",
        "#### Class step-index distributions (must overlap)",
        "",
        to_markdown(phase.round(2), index=True),
        "",
        "#### Signed within-pair phase bias",
        "",
        "```json",
        json.dumps(accounting["phase_bias"], indent=2, default=str),
        "```",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "AnalysisConfig",
    "PROBE_SPECS",
    "load_run",
    "make_samples",
    "plumbing_report",
    "run_cross_task",
    "run_e1",
    "run_e3",
    "run_ladder",
    "run_per_transform",
    "sample_accounting",
    "summarize",
    "to_markdown",
    "undo_alignment",
]

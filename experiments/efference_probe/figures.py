"""Write-up figures F1-F5 and the T1 run card (SPEC 6).

Matplotlib defaults throughout: axis labels, units, n, and fold standard
deviations as error bars, and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CHANCE = 0.5
LADDER_ORDER = ("P0", "P1", "P3", "P4", "P2", "P2r")


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_layer_depth(
    results: pd.DataFrame,
    out_path: Path,
    pools: Sequence[str],
    probes: tuple[str, ...] = ("P0", "P1", "P3", "P4"),
    n_samples: Optional[int] = None,
) -> Path:
    """F1: balanced accuracy vs layer, one panel per pool.

    Every pool is shown, and P0 is drawn alongside: since F2 reports the
    hidden-state probes at their best cell, the reader needs to see the whole
    grid the maximum was taken over, and how high the shuffled-label null
    climbs across the same grid.
    """
    pools = list(pools)
    figure, axes = plt.subplots(
        1, len(pools), figsize=(4.2 * len(pools), 4.2), sharey=True, squeeze=False
    )
    for axis, pool in zip(axes[0], pools):
        subset = results[results["pool"] == pool]
        for probe in probes:
            rows = subset[subset["name"] == probe].sort_values("layer")
            if rows.empty:
                continue
            axis.errorbar(
                rows["layer"],
                rows["balanced_acc_mean"],
                yerr=rows["balanced_acc_std"],
                marker="o" if probe != "P0" else "x",
                linestyle="-" if probe != "P0" else ":",
                capsize=3,
                label=probe,
            )
        axis.axhline(CHANCE, linestyle="--", color="grey")
        axis.set_xlabel("LM layer (0 = embeddings)")
        axis.set_title(f"pool = {pool}")
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("Balanced accuracy")
    axes[0][-1].legend(fontsize="small")
    suffix = f" (n={n_samples})" if n_samples else ""
    figure.suptitle(f"Hijack decodability by depth{suffix}")
    return _save(figure, Path(out_path))


def figure_ladder(
    results: pd.DataFrame,
    out_path: Path,
    layer: int,
    pool: str,
    n_samples: Optional[int] = None,
    selection_floor: Optional[float] = None,
) -> Path:
    """F2: the ladder, P0 through the mechanical oracle."""
    heights, errors, labels = [], [], []
    for probe in LADDER_ORDER:
        rows = results[results["name"] == probe]
        hidden_rows = rows[rows["layer"].notna()]
        if not hidden_rows.empty:
            rows = hidden_rows[
                (hidden_rows["layer"] == layer) & (hidden_rows["pool"] == pool)
            ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        heights.append(row["balanced_acc_mean"])
        errors.append(row["balanced_acc_std"])
        labels.append(probe)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    positions = np.arange(len(labels))
    axis.bar(positions, heights, yerr=errors, capsize=4, color="steelblue")
    axis.axhline(CHANCE, linestyle="--", color="grey", label="chance (0.5)")
    if selection_floor is not None:
        # P1/P3/P4 are reported at the cell where they peak, so 0.5 is not the
        # right floor for them: this is how high the shuffled-label null gets
        # under the same maximisation.
        axis.axhline(
            selection_floor,
            linestyle="-.",
            color="firebrick",
            label=f"selection-aware floor ({selection_floor:.3f})",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Balanced accuracy")
    # Full range: a truncated axis exaggerates every difference, and any bar
    # below the cut would render as nothing at all.
    axis.set_ylim(0.0, 1.02)
    axis.legend(fontsize="small")
    suffix = f", n={n_samples}" if n_samples else ""
    axis.set_title(f"Probe ladder (layer={layer}, pool={pool}{suffix})")
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, Path(out_path))


def figure_cross_task(
    in_task: pd.DataFrame,
    held_out: pd.DataFrame,
    out_path: Path,
    layer: int,
    pool: str,
) -> Path:
    """F3: in-task versus held-out-task generalisation."""
    probes = [p for p in ("P1", "P3", "P4") if p in set(held_out.get("probe", []))]
    in_values, in_errors, out_values, out_errors = [], [], [], []
    for probe in probes:
        rows = in_task[
            (in_task["name"] == probe)
            & (in_task["layer"] == layer)
            & (in_task["pool"] == pool)
        ]
        in_values.append(
            rows["balanced_acc_mean"].iloc[0] if not rows.empty else np.nan
        )
        in_errors.append(rows["balanced_acc_std"].iloc[0] if not rows.empty else np.nan)
        held = held_out[held_out["probe"] == probe]
        out_values.append(
            held["balanced_acc_mean"].iloc[0] if not held.empty else np.nan
        )
        out_errors.append(
            held["balanced_acc_std"].iloc[0] if not held.empty else np.nan
        )

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    positions = np.arange(len(probes))
    width = 0.36
    axis.bar(
        positions - width / 2,
        in_values,
        width,
        yerr=in_errors,
        capsize=4,
        label="in-task (episode-grouped CV)",
    )
    axis.bar(
        positions + width / 2,
        out_values,
        width,
        yerr=out_errors,
        capsize=4,
        label="held-out task",
    )
    axis.axhline(CHANCE, linestyle="--", color="grey")
    axis.set_xticks(positions)
    axis.set_xticklabels(probes)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title(f"Cross-task generalisation (layer={layer}, pool={pool})")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, Path(out_path))


def figure_e1(e1: pd.DataFrame, out_path: Path) -> Path:
    """F4: R^2 of the commanded-action readout by layer."""
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for pool, rows in e1.groupby("pool"):
        rows = rows.sort_values("layer")
        axis.errorbar(
            rows["layer"],
            rows["r2_mean"],
            yerr=rows["r2_std"],
            marker="o",
            capsize=3,
            label=str(pool),
        )
    axis.axhline(0.0, linestyle="--", color="grey")
    axis.set_xlabel("LM layer (0 = embeddings)")
    axis.set_ylabel(r"$R^2$ of commanded chunk readout")
    axis.set_title("E1: where the motor plan is linearly readable")
    axis.legend(title="pool")
    axis.grid(alpha=0.3)
    return _save(figure, Path(out_path))


def figure_e3(
    calls: pd.DataFrame,
    samples: pd.DataFrame,
    out_path: Path,
    success_note: str = "",
) -> Path:
    """F5: action-token entropy and logprob at m+1, SELF vs HIJACK."""
    lookup = {
        (int(e), int(c)): position
        for position, (e, c) in enumerate(zip(calls["episode_id"], calls["call_idx"]))
    }
    positions = np.asarray(
        [
            lookup[(int(e), int(c))]
            for e, c in zip(samples["episode_id"], samples["cur_call"])
        ]
    )
    labels = samples["y"].to_numpy()

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for axis, column, title in (
        (axes[0], "entropy_mean", "Action-token entropy (nats)"),
        (axes[1], "logprob_sum", "Chunk logprob (sum over 56 tokens)"),
    ):
        values = calls[column].to_numpy()
        groups = [values[positions[labels == 0]], values[positions[labels == 1]]]
        axis.violinplot(groups, showmeans=True)
        axis.set_xticks([1, 2])
        axis.set_xticklabels(
            [f"after SELF\n(n={len(groups[0])})", f"after HIJACK\n(n={len(groups[1])})"]
        )
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle(
        f"E3: policy state at call m+1{('  |  ' + success_note) if success_note else ''}"
    )
    return _save(figure, Path(out_path))


def run_card(
    run_config: dict[str, Any],
    manifest: dict[str, Any],
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """T1: the run card -- checkpoint, suite, sampling, counts, budget."""
    config = run_config.get("config", {})
    model = config.get("model", {})
    env = config.get("env", {})
    hijack = config.get("hijack", {})
    budget = manifest.get("budget", {})
    episodes = pd.DataFrame(manifest.get("episodes", []))
    rows = [
        ("run_id", run_config.get("run_id")),
        ("git_sha", run_config.get("git_sha")),
        ("config_hash", run_config.get("config_hash")),
        ("checkpoint", model.get("model_path")),
        ("model_type", model.get("model_type")),
        ("suite", env.get("task_suite_name")),
        ("unnorm_key", model.get("unnorm_key")),
        (
            "sampling",
            "greedy"
            if not model.get("do_sample")
            else f"temperature={model.get('temperature')}",
        ),
        ("parallel envs", env.get("num_envs")),
        ("init states / task", env.get("init_states_per_task")),
        ("hijack transform", hijack.get("transform")),
        ("p(hijack)", hijack.get("p_hijack")),
        ("probe-episode fraction", hijack.get("probe_episode_fraction")),
        ("episodes", manifest.get("n_episodes")),
        ("model calls", manifest.get("n_calls")),
        ("probe samples", len(samples)),
        (
            "success rate",
            round(float(episodes["success"].mean()), 3) if not episodes.empty else None,
        ),
        ("GPU hours", round(float(budget.get("elapsed_hours", float("nan"))), 2)),
        ("disk GB", round(float(budget.get("disk_gb", float("nan"))), 2)),
        ("stop reason", manifest.get("stop_reason")),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])

"""E3 (action sensitivity) and E4 (comparator) scores, torch-free.

Definitions and thresholds are those of ``PREREG_wave3.md``; the constants
below must not change after results exist (append an amendment instead).

Frames are compared at 64 x 64 (4 x 4 block mean of the 256 x 256 frames),
as floats in [0, 1]; ``d`` is the MSE over every predicted frame.  In
``static`` context mode only the last predicted frame has a real counterpart
(the next call's frame), so comparisons against reality use it alone.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

# -- pre-registered (PREREG_wave3.md, E3/E4) --------------------------------
DOWNSAMPLE = 4
E3_GATE_RATIO = 1.0
E3_MIN_QUERIES = 200
E4_MIN_AUROC = 0.65
E4_MIN_ADVANTAGE = 0.10
E4_NOVELTY_CEILING = 0.60
N_BOOTSTRAP = 2000

# Score columns, oriented so that larger = "more likely hijacked".
COMPARATOR_SCORES = ("adv_distance", "raw_error", "rank_score", "neg_adv_cosine")
NOVELTY_SCORES = ("novelty_adv", "novelty", "motion", "policy_entropy")
PRIMARY_COMPARATOR = "adv_distance"
PRIMARY_NOVELTY = "novelty_adv"


def prepare(frames: np.ndarray, factor: int = DOWNSAMPLE) -> np.ndarray:
    """uint8 ``[..., H, W, 3]`` -> float32 ``[..., H/f, W/f, 3]`` in [0, 1]."""
    x = np.asarray(frames, dtype=np.float32) / 255.0
    if factor > 1:
        *lead, height, width, channels = x.shape
        if height % factor or width % factor:
            raise ValueError(f"frame size {height}x{width} not divisible by {factor}")
        x = x.reshape(
            *lead, height // factor, factor, width // factor, factor, channels
        )
        x = x.mean(axis=(-4, -2))
    return x


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def align(prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """The predicted frames that have a real counterpart in ``actual``."""
    if len(actual) == len(prediction):
        return prediction
    if len(actual) == 1:
        return prediction[-1:]
    raise ValueError(
        f"cannot align {len(prediction)} predicted with {len(actual)} real frames"
    )


def _alternatives(predictions: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    return [predictions[name] for name in predictions if name.startswith("alt")]


def orthonormal_basis(directions: list[np.ndarray], rtol: float = 1e-6) -> np.ndarray:
    """Columns spanning ``directions`` (flattened); rank-deficient ones dropped."""
    if not directions:
        return np.zeros((0, 0))
    matrix = np.stack([np.ravel(d).astype(np.float64) for d in directions], axis=1)
    q, r = np.linalg.qr(matrix)
    diagonal = np.abs(np.diag(r))
    scale = diagonal.max() if diagonal.size else 0.0
    if scale == 0.0:
        return np.zeros((matrix.shape[0], 0))
    return q[:, diagonal > rtol * scale]


# -- E3 ---------------------------------------------------------------------


def sensitivity_row(
    predictions: Mapping[str, np.ndarray],
    context_last: np.ndarray,
    actual: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Per-query E3 numbers from prepared (float) predictions.

    ``predictions`` must hold ``cmd`` and ``cmd@noise`` (same chunk, another
    seed); ``mirror``/``freeze``/``swap`` and ``alt*`` are used when present.
    """
    p_cmd = predictions["cmd"]
    noise = distance(p_cmd, predictions["cmd@noise"])
    row: dict[str, float] = {"noise": noise}
    for transform in ("mirror", "freeze", "swap"):
        if transform in predictions:
            effect = distance(p_cmd, predictions[transform])
            row[f"effect_{transform}"] = effect
            row[f"r_{transform}"] = effect / noise if noise > 0 else float("inf")
    alternatives = _alternatives(predictions)
    if alternatives:
        mean_alt = np.mean(alternatives, axis=0)
        spread = float(np.mean([distance(a, mean_alt) for a in alternatives]))
        motion = distance(mean_alt, np.broadcast_to(context_last, mean_alt.shape))
        row["action_spread"] = spread
        row["state_motion"] = motion
        row["jiazhuo_ratio"] = spread / motion if motion > 0 else float("inf")
        if actual is not None:
            d_cmd = distance(align(p_cmd, actual), actual)
            d_alt = [distance(align(a, actual), actual) for a in alternatives]
            row["fidelity_cmd"] = d_cmd
            row["fidelity_alt_median"] = float(np.median(d_alt))
            row["cmd_beats_alt_median"] = float(d_cmd < np.median(d_alt))
    return row


def e3_summary(rows: pd.DataFrame, context_mode: str) -> dict[str, Any]:
    """Medians, fractions above 1, and the pre-registered gate."""
    summary: dict[str, Any] = {
        "n_queries": int(len(rows)),
        "context_mode": context_mode,
    }
    for column in rows.columns:
        if column.startswith("r_") or column in (
            "noise",
            "action_spread",
            "state_motion",
            "jiazhuo_ratio",
        ):
            values = rows[column].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            summary[column] = {
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
            if column.startswith("r_"):
                summary[column]["fraction_above_1"] = float((values > 1).mean())
    if "cmd_beats_alt_median" in rows:
        summary["fidelity_cmd_beats_alt_median"] = float(
            rows["cmd_beats_alt_median"].mean()
        )
    medians = {
        t: (summary.get(f"r_{t}") or {}).get("median", float("nan"))
        for t in ("freeze", "swap")
    }
    eligible = context_mode == "substeps" and len(rows) >= E3_MIN_QUERIES
    passed = all(np.isfinite(v) and v >= E3_GATE_RATIO for v in medians.values())
    summary["gate"] = {
        "rule": f"median r_freeze >= {E3_GATE_RATIO} and median r_swap >= {E3_GATE_RATIO}",
        "eligible": bool(eligible),
        "eligibility_note": (
            "gate decisions need substeps context and >= "
            f"{E3_MIN_QUERIES} queries; static-context results are exploratory"
        ),
        "median_r_freeze": medians["freeze"],
        "median_r_swap": medians["swap"],
        "passed": bool(passed) if eligible else None,
    }
    return summary


# -- E4 ---------------------------------------------------------------------


def comparator_row(
    predictions: Mapping[str, np.ndarray],
    actual: np.ndarray,
    context_last: np.ndarray,
) -> dict[str, float]:
    """Per-sample E4 scores from prepared predictions and the real outcome."""
    p_cmd = align(predictions["cmd"], actual)
    alternatives = [align(a, actual) for a in _alternatives(predictions)]
    if not alternatives:
        raise ValueError("E4 needs alternative candidates (alt1..altK)")
    mean_alt = np.mean(alternatives, axis=0)
    raw = distance(actual, p_cmd)
    d_alt = [distance(actual, a) for a in alternatives]

    residual = np.ravel(actual - mean_alt).astype(np.float64)
    predicted = np.ravel(p_cmd - mean_alt).astype(np.float64)
    size = residual.size
    basis = orthonormal_basis([p_cmd - mean_alt] + [a - mean_alt for a in alternatives])
    proj_residual = basis.T @ residual
    proj_predicted = basis.T @ predicted
    norms = np.linalg.norm(proj_residual) * np.linalg.norm(proj_predicted)
    cosine = float(proj_residual @ proj_predicted / norms) if norms > 0 else 0.0
    alt_basis = orthonormal_basis([a - mean_alt for a in alternatives])

    return {
        "adv_distance": float(np.sum((proj_residual - proj_predicted) ** 2) / size),
        "raw_error": raw,
        "rank_score": raw - float(np.mean(d_alt)),
        "neg_adv_cosine": -cosine,
        "novelty_adv": float(np.sum((alt_basis.T @ residual) ** 2) / size),
        "novelty": distance(actual, mean_alt),
        "motion": distance(actual[-1], context_last),
        "subspace_rank": float(basis.shape[1]),
    }


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    drawn = rng.choice(unique, size=len(unique), replace=True)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    return np.concatenate([members[g] for g in drawn])


def auroc_with_ci(
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    groups: np.ndarray,
    pairs: tuple[tuple[str, str], ...] = (),
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """AUROC per score and for score differences, episode-level bootstrap."""
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    point = {
        name: _auroc(labels, np.asarray(s, dtype=float)) for name, s in scores.items()
    }
    draws: dict[str, list[float]] = {name: [] for name in scores}
    pair_draws: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    for _ in range(n_bootstrap):
        index = _bootstrap_indices(groups, rng)
        sample = {
            name: _auroc(labels[index], np.asarray(s, dtype=float)[index])
            for name, s in scores.items()
        }
        for name, value in sample.items():
            draws[name].append(value)
        for a, b in pairs:
            pair_draws[(a, b)].append(sample[a] - sample[b])
    out: dict[str, Any] = {}
    for name in scores:
        values = np.asarray(draws[name], dtype=float)
        out[name] = {
            "auroc": point[name],
            "ci95": [
                float(np.nanquantile(values, 0.025)),
                float(np.nanquantile(values, 0.975)),
            ]
            if np.isfinite(values).any()
            else [float("nan"), float("nan")],
        }
    for a, b in pairs:
        values = np.asarray(pair_draws[(a, b)], dtype=float)
        out[f"{a}-{b}"] = {
            "difference": point[a] - point[b],
            "ci95": [
                float(np.nanquantile(values, 0.025)),
                float(np.nanquantile(values, 0.975)),
            ]
            if np.isfinite(values).any()
            else [float("nan"), float("nan")],
        }
    return out


def e4_verdict(stats: Mapping[str, Any]) -> str:
    """The pre-registered FREEZE readout, applied to one transform's stats."""
    primary = stats.get(PRIMARY_COMPARATOR) or {}
    novelty = stats.get(PRIMARY_NOVELTY) or {}
    difference = stats.get(f"{PRIMARY_COMPARATOR}-{PRIMARY_NOVELTY}") or {}
    auroc = primary.get("auroc", float("nan"))
    low = (primary.get("ci95") or [float("nan")])[0]
    diff = difference.get("difference", float("nan"))
    diff_low, diff_high = difference.get("ci95") or [float("nan"), float("nan")]
    if not np.isfinite(auroc):
        return "insufficient data"
    if (
        auroc >= E4_MIN_AUROC
        and low > 0.5
        and diff >= E4_MIN_ADVANTAGE
        and diff_low > 0
    ):
        return "comparator supplies what the policy lacks"
    neither = (
        auroc < E4_NOVELTY_CEILING and novelty.get("auroc", 1.0) < E4_NOVELTY_CEILING
    )
    if neither or (diff_low <= 0 <= diff_high):
        return "comparator only reads novelty"
    return "intermediate (see AUROCs)"


def e4_summary(
    rows: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0
) -> dict[str, Any]:
    """Per-transform AUROCs, the primary difference, and the verdict."""
    out: dict[str, Any] = {}
    columns = [c for c in COMPARATOR_SCORES + NOVELTY_SCORES if c in rows.columns]
    for transform, group in rows.groupby("transform"):
        stats = auroc_with_ci(
            group["y"].to_numpy(),
            {c: group[c].to_numpy(dtype=float) for c in columns},
            group["episode_id"].to_numpy(),
            pairs=((PRIMARY_COMPARATOR, PRIMARY_NOVELTY),),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        stats["n_samples"] = int(len(group))
        stats["n_positive"] = int(group["y"].sum())
        stats["n_episodes"] = int(group["episode_id"].nunique())
        stats["verdict"] = e4_verdict(stats)
        out[str(transform)] = stats
    return out


__all__ = [
    "COMPARATOR_SCORES",
    "NOVELTY_SCORES",
    "PRIMARY_COMPARATOR",
    "PRIMARY_NOVELTY",
    "auroc_with_ci",
    "comparator_row",
    "distance",
    "e3_summary",
    "e4_summary",
    "e4_verdict",
    "orthonormal_basis",
    "prepare",
    "sensitivity_row",
]

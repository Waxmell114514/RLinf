"""Linear probes with episode-grouped cross-validation.

Every probe is L2-regularised logistic regression on standardised features,
scored with balanced accuracy and AUROC over 5 folds grouped by episode --
grouping matters because consecutive calls within an episode are strongly
correlated, and a plain k-fold would leak neighbouring frames across the split.

The regularisation strength is chosen *inside* each outer fold (one inner
grouped split), so the reported score is not inflated by picking ``C`` on the
data it is scored against.  The per-``C`` table is reported alongside, so the
ladder's ordering can be checked for stability rather than taken on trust.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_C_VALUES = (0.01, 0.1, 1.0)


@dataclass
class ProbeResult:
    """Cross-validated scores for one probe configuration."""

    name: str
    blocks: list[str]
    layer: Optional[int] = None
    pool: Optional[str] = None
    n_samples: int = 0
    n_positive: int = 0
    n_features: int = 0
    n_groups: int = 0
    balanced_acc_mean: float = float("nan")
    balanced_acc_std: float = float("nan")
    auroc_mean: float = float("nan")
    auroc_std: float = float("nan")
    fold_balanced_acc: list[float] = field(default_factory=list)
    fold_auroc: list[float] = field(default_factory=list)
    selected_c: list[float] = field(default_factory=list)
    per_c: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["blocks"] = "+".join(self.blocks)
        row["fold_balanced_acc"] = list(self.fold_balanced_acc)
        row["fold_auroc"] = list(self.fold_auroc)
        return row


def _make_classifier(c_value: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                # L2 is the solver default; naming it explicitly is
                # deprecated in scikit-learn >= 1.8.
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=max_iter,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _fit_score(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    balanced = balanced_accuracy_score(y_test, predicted)
    # AUROC is undefined when a fold happens to hold a single class.
    if len(np.unique(y_test)) < 2:
        auroc = float("nan")
    else:
        scores = model.decision_function(x_test)
        auroc = roc_auc_score(y_test, scores)
    return float(balanced), float(auroc)


def _outer_splits(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
):
    """Grouped, class-stratified folds, falling back to plain GroupKFold."""
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        return list(splitter.split(x, y, groups))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(x, y, groups))


def run_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    name: str,
    blocks: list[str],
    layer: Optional[int] = None,
    pool: Optional[str] = None,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    n_splits: int = 5,
    seed: int = 0,
    shuffle_labels: bool = False,
    max_iter: int = 2000,
    select_c: bool = True,
) -> ProbeResult:
    """Fit and score one probe.

    Args:
        shuffle_labels: permute ``y`` before splitting, giving the P0 chance
            floor.  Anything much above 0.5 here means the pipeline leaks.
        select_c: choose ``C`` per outer fold on an inner grouped split.  With
            ``False`` the largest value in ``c_values`` is used throughout,
            which is much faster for exploratory sweeps.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    result = ProbeResult(
        name=name,
        blocks=list(blocks),
        layer=layer,
        pool=pool,
        n_samples=int(x.shape[0]),
        n_positive=int((y == 1).sum()),
        n_features=int(x.shape[1]) if x.ndim == 2 else 0,
        n_groups=int(len(np.unique(groups))),
    )
    if x.shape[0] < n_splits * 2 or len(np.unique(y)) < 2:
        result.notes = "insufficient samples or single-class labels"
        return result

    if shuffle_labels:
        y = np.random.default_rng(seed).permutation(y)

    per_c_scores: dict[float, list[float]] = {c: [] for c in c_values}
    for train_index, test_index in _outer_splits(x, y, groups, n_splits, seed):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train = groups[train_index]

        if select_c and len(c_values) > 1:
            chosen = _select_c(x_train, y_train, groups_train, c_values, seed, max_iter)
        else:
            chosen = max(c_values)
        model = _make_classifier(chosen, max_iter)
        balanced, auroc = _fit_score(model, x_train, y_train, x_test, y_test)
        result.fold_balanced_acc.append(balanced)
        result.fold_auroc.append(auroc)
        result.selected_c.append(float(chosen))

        for c_value in c_values:
            fold_model = _make_classifier(c_value, max_iter)
            fold_balanced, _ = _fit_score(fold_model, x_train, y_train, x_test, y_test)
            per_c_scores[c_value].append(fold_balanced)

    result.balanced_acc_mean = float(np.mean(result.fold_balanced_acc))
    result.balanced_acc_std = float(np.std(result.fold_balanced_acc, ddof=1))
    result.auroc_mean = float(np.nanmean(result.fold_auroc))
    result.auroc_std = float(np.nanstd(result.fold_auroc, ddof=1))
    result.per_c = {
        f"C={c}": {
            "balanced_acc_mean": float(np.mean(scores)),
            "balanced_acc_std": float(np.std(scores, ddof=1)),
        }
        for c, scores in per_c_scores.items()
    }
    return result


def _select_c(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c_values: tuple[float, ...],
    seed: int,
    max_iter: int,
) -> float:
    """Pick C on a single inner grouped split of the outer-training data."""
    try:
        inner = list(
            StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(
                x, y, groups
            )
        )
    except ValueError:
        return max(c_values)
    if not inner:
        return max(c_values)
    train_index, validation_index = inner[0]
    best_c, best_score = max(c_values), -np.inf
    for c_value in c_values:
        model = _make_classifier(c_value, max_iter)
        score, _ = _fit_score(
            model,
            x[train_index],
            y[train_index],
            x[validation_index],
            y[validation_index],
        )
        if score > best_score:
            best_c, best_score = c_value, score
    return best_c


def ridge_readout(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """E1: how well does a hidden state linearly predict the commanded chunk?

    Ridge regression from ``h_m`` to the 56-dimensional commanded action, with
    the same episode-grouped folds as the classifiers.  Reports the uniform
    average R^2 across output dimensions plus the per-dimension breakdown.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    groups = np.asarray(groups)

    fold_r2: list[float] = []
    per_dim: list[np.ndarray] = []
    selected: list[float] = []
    splitter = GroupKFold(n_splits=n_splits)
    for train_index, test_index in splitter.split(x, y, groups):
        best_alpha, best_score = alphas[0], -np.inf
        inner = list(
            GroupKFold(n_splits=3).split(
                x[train_index], y[train_index], groups[train_index]
            )
        )
        inner_train, inner_validation = inner[0]
        for alpha in alphas:
            model = Pipeline(
                [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
            )
            model.fit(x[train_index][inner_train], y[train_index][inner_train])
            score = r2_score(
                y[train_index][inner_validation],
                model.predict(x[train_index][inner_validation]),
                multioutput="uniform_average",
            )
            if score > best_score:
                best_alpha, best_score = alpha, score

        model = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))]
        )
        model.fit(x[train_index], y[train_index])
        predicted = model.predict(x[test_index])
        fold_r2.append(
            float(r2_score(y[test_index], predicted, multioutput="uniform_average"))
        )
        per_dim.append(
            np.asarray(
                r2_score(y[test_index], predicted, multioutput="raw_values"),
                dtype=np.float64,
            )
        )
        selected.append(float(best_alpha))

    return {
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2, ddof=1)),
        "fold_r2": fold_r2,
        "per_dim_r2": np.mean(np.stack(per_dim), axis=0).tolist(),
        "selected_alpha": selected,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }


def results_to_frame(results: list[ProbeResult]) -> pd.DataFrame:
    """Flatten probe results into a table ready for CSV or markdown."""
    if not results:
        return pd.DataFrame()
    rows = []
    for result in results:
        row = result.to_row()
        per_c = row.pop("per_c", {})
        for key, values in per_c.items():
            row[f"{key}_bacc"] = values["balanced_acc_mean"]
        rows.append(row)
    return pd.DataFrame(rows)

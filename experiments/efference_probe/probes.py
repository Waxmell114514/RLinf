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
from scipy import linalg as scipy_linalg
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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


class BlockScaler(BaseEstimator, TransformerMixin):
    """Divide each feature block by the square root of its width.

    Applied *after* standardisation, which is the only place it can do
    anything: ``StandardScaler`` divides every column by its own standard
    deviation, so any per-block constant applied beforehand is exactly
    cancelled.  The point is to stop a 56-dimensional action block from being
    swamped by a 4096-dimensional hidden block under a shared L2 penalty, so
    the reweighting has to survive to the classifier.
    """

    def __init__(self, spans: Optional[list[tuple[str, int]]] = None) -> None:
        self.spans = spans

    def fit(self, x, y=None):  # noqa: D102 - sklearn API
        return self

    def transform(self, x):  # noqa: D102 - sklearn API
        if not self.spans:
            return x
        scaled = np.array(x, dtype=np.float64, copy=True)
        start = 0
        for _name, width in self.spans:
            scaled[:, start : start + width] /= np.sqrt(width)
            start += width
        if start != scaled.shape[1]:
            raise ValueError(
                f"block spans cover {start} columns but the matrix has "
                f"{scaled.shape[1]}"
            )
        return scaled


def _make_classifier(
    c_value: float, max_iter: int, spans: Optional[list[tuple[str, int]]] = None
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("blocks", BlockScaler(spans)),
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
    spans: Optional[list[tuple[str, int]]] = None,
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
            chosen = _select_c(
                x_train, y_train, groups_train, c_values, seed, max_iter, spans
            )
        else:
            chosen = max(c_values)
        # The per-C sweep refits this fold at every candidate C, and `chosen`
        # is always one of them, so scoring the selected model separately
        # would repeat a fit that is already being done.  Sweep first, then
        # read the fold's score back out: identical numbers, one fewer fit
        # per fold.
        fold_scores: dict[float, tuple[float, float]] = {}
        for c_value in c_values:
            fold_model = _make_classifier(c_value, max_iter, spans)
            fold_scores[c_value] = _fit_score(
                fold_model, x_train, y_train, x_test, y_test
            )
            per_c_scores[c_value].append(fold_scores[c_value][0])

        if chosen not in fold_scores:  # defensive: _select_c returns a listed C
            model = _make_classifier(chosen, max_iter, spans)
            fold_scores[chosen] = _fit_score(model, x_train, y_train, x_test, y_test)
        balanced, auroc = fold_scores[chosen]
        result.fold_balanced_acc.append(balanced)
        result.fold_auroc.append(auroc)
        result.selected_c.append(float(chosen))

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
    spans: Optional[list[tuple[str, int]]] = None,
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
        model = _make_classifier(c_value, max_iter, spans)
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


class _SharedRidge:
    """Ridge at many alphas from one factorisation of the training data.

    ``ridge_readout`` scores every candidate alpha on the *same* inner-training
    split, and scikit-learn rebuilds the entire normal-equation system for each
    one.  Alpha only shifts the diagonal, so the expensive part -- the Gram
    matrix, or the kernel matrix when features outnumber samples -- is built
    once here and reused.  That is where E1's time goes: E1 is 40 ridge fits
    per cell over roughly (n_calls x 4096), and 35 of those 40 differ from a
    neighbour by nothing but a scalar.

    This mirrors ``Pipeline([StandardScaler(), Ridge(alpha)])`` exactly rather
    than approximating it: standardise by the population standard deviation
    with constant columns left at scale 1, centre X and y, solve the centred
    system without penalising the intercept, and fold the intercept back in as
    ``y_mean - x_mean @ w``.  It also switches to the dual form below
    ``n_samples < n_features`` for the same reason scikit-learn does -- the
    system is then n x n instead of p x p.

    Not an SVD: for this shape a thin SVD of the 4096-column design costs more
    than all seven of the fits it would replace (measured 13.2s against 3.9s).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        # scikit-learn's _handle_zeros_in_scale: a constant column would
        # otherwise divide by ~0, so its scale is pinned to 1.
        scale = np.where(scale < 10 * np.finfo(scale.dtype).eps, 1.0, scale)
        self.scale_ = scale
        scaled = (x - self.mean_) / self.scale_

        # Ridge centres its inputs and recovers the intercept afterwards, so
        # the penalty never touches it.  `scaled` is already centred to within
        # rounding; centring again reproduces sklearn's arithmetic exactly.
        self.x_offset_ = scaled.mean(axis=0)
        self.y_offset_ = y.mean(axis=0)
        self._x = scaled - self.x_offset_
        self._y = y - self.y_offset_

        n_samples, n_features = self._x.shape
        self._dual = n_samples < n_features
        if self._dual:
            self._gram = self._x @ self._x.T
        else:
            self._gram = self._x.T @ self._x
            self._xty = self._x.T @ self._y

    def coef(self, alpha: float) -> np.ndarray:
        """Coefficients for one alpha, reusing the stored factorisable system."""
        system = self._gram.copy()
        system.flat[:: system.shape[0] + 1] += alpha
        target = self._y if self._dual else self._xty
        try:
            solution = scipy_linalg.solve(system, target, assume_a="pos")
        except (scipy_linalg.LinAlgError, ValueError):
            # A Cholesky-hostile system is possible at tiny alpha; the general
            # solver is slower but does not change the answer.
            solution = np.linalg.solve(system, target)
        return self._x.T @ solution if self._dual else solution

    def predict(self, x_new: np.ndarray, alpha: float) -> np.ndarray:
        scaled = (np.asarray(x_new) - self.mean_) / self.scale_
        return (scaled - self.x_offset_) @ self.coef(alpha) + self.y_offset_


def ridge_readout(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5),
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
        # One system for the whole alpha sweep; see _SharedRidge.
        inner_solver = _SharedRidge(
            x[train_index][inner_train], y[train_index][inner_train]
        )
        for alpha in alphas:
            score = r2_score(
                y[train_index][inner_validation],
                inner_solver.predict(x[train_index][inner_validation], alpha),
                multioutput="uniform_average",
            )
            if score > best_score:
                best_alpha, best_score = alpha, score

        predicted = _SharedRidge(x[train_index], y[train_index]).predict(
            x[test_index], best_alpha
        )
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
        # A grid-edge alpha means the search never bracketed the optimum, so
        # the R^2 is a lower bound on what this representation supports.
        "alpha_at_grid_edge": any(
            alpha in (min(alphas), max(alphas)) for alpha in selected
        ),
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

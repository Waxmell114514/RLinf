"""Turn a collection run into probe-ready matrices.

No torch, no LIBERO: this module reads ``calls.parquet`` and the per-episode
``hidden/*.npz`` files, so probes can be re-run on a laptop.

The unit of analysis is a *pair* of consecutive model calls ``(prev, cur)``
where ``cur == prev + 1``:

* **positive** -- the chunk executed at ``prev`` was hijacked, so the world
  changed in a way the policy did not command.
* **negative** -- the chunk executed at ``prev`` was the policy's own.

In both cases the call at ``cur`` is itself SELF, guaranteed for positives by
the scheduler's clean-gap rule.  Negatives additionally require a fully clean
recent history (``min_clean_gap + 1`` preceding calls), so the two classes
differ in exactly one thing: whether the last transition was self-caused.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

HIJACK = "HIJACK"
SELF = "SELF"

# Feature blocks a probe can be built from.
BLOCK_A_CMD = "a_cmd"
BLOCK_DSTATES = "dstates"
BLOCK_MISMATCH = "mismatch"
BLOCK_PHASE = "phase"
BLOCK_H_CUR = "h_cur"
BLOCK_H_PREV = "h_prev"
BLOCK_VIS_CUR = "vis_cur"


@dataclass
class RunData:
    """A loaded collection run."""

    run_dir: Path
    calls: pd.DataFrame
    schema: dict[str, Any]
    config: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def layers(self) -> list[int]:
        return list(self.schema["layers"])

    @property
    def pools(self) -> list[str]:
        return list(self.schema["pools"])

    def layer_index(self, layer: int) -> int:
        try:
            return self.layers.index(layer)
        except ValueError as error:
            raise ValueError(
                f"layer {layer} not captured; run has {self.layers}"
            ) from error

    def pool_index(self, pool: str) -> int:
        try:
            return self.pools.index(pool)
        except ValueError as error:
            raise ValueError(
                f"pool {pool!r} not captured; run has {self.pools}"
            ) from error


def load_run(run_dir: str | Path) -> RunData:
    """Read ``calls.parquet``, ``run_config.yaml`` and ``manifest.json``."""
    import json

    run_dir = Path(run_dir)
    with open(run_dir / "run_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    calls = pd.read_parquet(run_dir / "calls.parquet")
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    return RunData(
        run_dir=run_dir,
        calls=calls,
        schema=config["schema"],
        config=config,
        manifest=manifest,
    )


class HiddenStore:
    """Lazy reader for per-episode hidden-state archives, with a size cap."""

    def __init__(self, hidden_dir: str | Path, max_cache_gb: float = 8.0) -> None:
        self.hidden_dir = Path(hidden_dir)
        self.max_cache_bytes = max_cache_gb * (1024**3)
        self._cache: "OrderedDict[int, dict[str, np.ndarray]]" = OrderedDict()
        self._cache_bytes = 0

    def _load(self, episode_id: int) -> dict[str, np.ndarray]:
        cached = self._cache.get(episode_id)
        if cached is not None:
            self._cache.move_to_end(episode_id)
            return cached
        path = self.hidden_dir / f"ep{episode_id:05d}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"no hidden states for episode {episode_id}: {path}"
            )
        with np.load(path) as archive:
            payload = {key: archive[key] for key in archive.files}
        # call_idx -> row, so lookups do not scan.
        payload["_index"] = {
            int(call): row for row, call in enumerate(payload["call_idx"])
        }
        size = sum(
            value.nbytes for value in payload.values() if isinstance(value, np.ndarray)
        )
        self._cache[episode_id] = payload
        self._cache_bytes += size
        while self._cache_bytes > self.max_cache_bytes and len(self._cache) > 1:
            _key, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= sum(
                value.nbytes
                for value in evicted.values()
                if isinstance(value, np.ndarray)
            )
        return payload

    def vectors(
        self,
        pairs: Iterable[tuple[int, int]],
        layer_index: int,
        pool_index: int,
    ) -> np.ndarray:
        """Stack ``h[layer, pool]`` for each ``(episode_id, call_idx)``."""
        rows = []
        for episode_id, call_idx in pairs:
            payload = self._load(int(episode_id))
            row = payload["_index"].get(int(call_idx))
            if row is None:
                raise KeyError(
                    f"episode {episode_id} has no hidden state for call {call_idx}"
                )
            rows.append(payload["h"][row, layer_index, pool_index].astype(np.float32))
        return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)

    def vision(self, pairs: Iterable[tuple[int, int]]) -> np.ndarray:
        rows = []
        for episode_id, call_idx in pairs:
            payload = self._load(int(episode_id))
            if "vis" not in payload:
                raise KeyError(
                    "run has no vision features; re-collect with "
                    "capture.capture_vision=true"
                )
            row = payload["_index"].get(int(call_idx))
            if row is None:
                # `payload["vis"][None]` is np.newaxis, not an error: without
                # this guard a missing call silently returns the whole array.
                raise KeyError(
                    f"episode {episode_id} has no vision features for call {call_idx}"
                )
            rows.append(payload["vis"][row].astype(np.float32))
        return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def build_probe_samples(
    calls: pd.DataFrame,
    min_clean_gap: int = 2,
    warmup_calls: int = 4,
    negatives: str = "phase_matched",
    rng: Optional[np.random.Generator] = None,
    drop_post_success: bool = True,
    max_phase_distance: Optional[int] = None,
    paired: bool = True,
) -> pd.DataFrame:
    """Pair every post-hijack call with a matched self-caused call.

    A hijack can never land before ``warmup_calls``, so a positive can never
    sit earlier than ``warmup_calls + 1``.  Negative candidates are held to the
    same floor: without it the negative class fills up with early-episode calls
    and the probe can separate the classes on task phase alone.

    Args:
        calls: rows from ``calls.parquet``.
        min_clean_gap: the scheduler's gap, used to size the clean window a
            negative must sit in.
        warmup_calls: the scheduler's warm-up, used to set the earliest call
            index either class may occupy.
        negatives: ``"phase_matched"`` solves a minimum-total-distance
            assignment between positives and eligible calls *within each
            episode*; ``"global"`` samples from the pool of eligible calls
            across all episodes, which deliberately does not control for task
            phase and serves as the robustness check.
        rng: source of randomness for tie-breaks and global sampling.
        drop_post_success: exclude calls at or after the episode's first
            success, whose observations reflect a solved task.
        max_phase_distance: drop a matched pair whose call indices differ by
            more than this.  ``None`` accepts any distance.
        paired: with phase-matched negatives, drop positives that could not be
            matched, keeping a strictly paired design.

    Returns:
        One row per probe sample: ``episode_id, task_id, cur_call, prev_call,
        y, transform, source``, plus a ``pair_id`` linking matched rows.
    """
    rng = rng or np.random.default_rng(0)
    frame = calls
    if drop_post_success and "post_success" in frame.columns:
        frame = frame[~frame["post_success"].astype(bool)]
    earliest_call = warmup_calls + 1

    positives: list[dict[str, Any]] = []
    negative_candidates: list[dict[str, Any]] = []

    for episode_id, episode in frame.groupby("episode_id", sort=True):
        episode = episode.sort_values("call_idx")
        labels = dict(zip(episode["call_idx"], episode["label"]))
        task_id = int(episode["task_id"].iloc[0])
        transforms = dict(zip(episode["call_idx"], episode["transform"]))
        present = set(labels)

        for call_idx in sorted(present):
            prev_call = call_idx - 1
            if prev_call not in present:
                continue
            if labels[call_idx] != SELF:
                # A hijacked `cur` would confound the target hidden state.
                continue
            record = {
                "episode_id": int(episode_id),
                "task_id": task_id,
                "cur_call": int(call_idx),
                "prev_call": int(prev_call),
            }
            if labels[prev_call] == HIJACK:
                record["y"] = 1
                record["transform"] = transforms[prev_call]
                positives.append(record)
            else:
                # Negatives need a clean recent history matching the window the
                # scheduler guarantees before a positive, and must sit in the
                # call range a positive could have occupied.
                if call_idx < earliest_call:
                    continue
                window = range(call_idx - min_clean_gap - 1, call_idx + 1)
                if any(labels.get(w, SELF) == HIJACK for w in window):
                    continue
                record["y"] = 0
                record["transform"] = ""
                negative_candidates.append(record)

    if not positives:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "task_id",
                "cur_call",
                "prev_call",
                "y",
                "transform",
                "source",
            ]
        )

    if negatives == "phase_matched":
        positives, chosen = _match_by_phase(
            positives, negative_candidates, max_phase_distance, paired
        )
    elif negatives == "global":
        chosen = _sample_global(positives, negative_candidates, rng)
        # -1 marks "not paired": a global negative has no counterpart, and a
        # paired test keyed on a positional pair_id would silently compare
        # unrelated calls.
        for record in positives + chosen:
            record["pair_id"] = -1
            record["phase_distance"] = -1
    else:
        raise ValueError(f"unknown negatives mode: {negatives!r}")

    for record in positives:
        record["source"] = "positive"
    for record in chosen:
        record["source"] = negatives

    if not positives:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "task_id",
                "cur_call",
                "prev_call",
                "y",
                "transform",
                "source",
                "pair_id",
            ]
        )

    samples = pd.DataFrame(positives + chosen)
    return samples.sort_values(["episode_id", "cur_call"]).reset_index(drop=True)


def _match_by_phase(
    positives: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    max_phase_distance: Optional[int],
    paired: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign each positive a distinct clean call from the same episode.

    Greedy nearest-neighbour matching leaves late positives with whatever early
    calls are left over, which is exactly the phase skew we are trying to
    avoid.  Minimising the *total* call-index distance per episode instead
    keeps the two classes' step distributions on top of each other.
    """
    by_episode_positive: dict[int, list[dict[str, Any]]] = {}
    by_episode_candidate: dict[int, list[dict[str, Any]]] = {}
    for positive in positives:
        by_episode_positive.setdefault(positive["episode_id"], []).append(positive)
    for candidate in candidates:
        by_episode_candidate.setdefault(candidate["episode_id"], []).append(candidate)

    kept_positives: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []
    pair_id = 0
    for episode_id, episode_positives in sorted(by_episode_positive.items()):
        pool = by_episode_candidate.get(episode_id, [])
        if not pool:
            if not paired:
                for positive in episode_positives:
                    unmatched = dict(positive)
                    unmatched["pair_id"] = -1
                    unmatched["phase_distance"] = -1
                    kept_positives.append(unmatched)
            continue
        cost = np.abs(
            np.asarray([p["cur_call"] for p in episode_positives])[:, None]
            - np.asarray([c["cur_call"] for c in pool])[None, :]
        )
        rows, columns = _assign(cost)
        matched_rows = set()
        for row, column in zip(rows, columns):
            distance = int(cost[row, column])
            if max_phase_distance is not None and distance > max_phase_distance:
                continue
            positive = dict(episode_positives[row])
            negative = dict(pool[column])
            positive["pair_id"] = pair_id
            negative["pair_id"] = pair_id
            positive["phase_distance"] = distance
            negative["phase_distance"] = distance
            kept_positives.append(positive)
            chosen.append(negative)
            matched_rows.add(row)
            pair_id += 1
        if not paired:
            for row, positive in enumerate(episode_positives):
                if row not in matched_rows:
                    unmatched = dict(positive)
                    unmatched["pair_id"] = -1
                    unmatched["phase_distance"] = -1
                    kept_positives.append(unmatched)
    return kept_positives, chosen


def _assign(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost one-to-one assignment, with a greedy fallback."""
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:  # pragma: no cover - scipy ships with scikit-learn
        rows, columns = [], []
        used = set()
        for row in np.argsort(cost.min(axis=1)):
            order = np.argsort(cost[row])
            for column in order:
                if column not in used:
                    used.add(int(column))
                    rows.append(int(row))
                    columns.append(int(column))
                    break
        return np.asarray(rows), np.asarray(columns)


def _sample_global(
    positives: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Draw negatives uniformly from the whole run, ignoring task phase."""
    if not candidates:
        return []
    count = min(len(positives), len(candidates))
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [dict(candidates[i]) for i in indices]


def call_lookup(calls: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Map ``(episode_id, call_idx)`` to a row position in ``calls``."""
    return {
        (int(episode), int(call)): position
        for position, (episode, call) in enumerate(
            zip(calls["episode_id"], calls["call_idx"])
        )
    }


def build_features(
    run: RunData,
    samples: pd.DataFrame,
    blocks: list[str],
    layer: Optional[int] = None,
    pool: Optional[str] = None,
    store: Optional[HiddenStore] = None,
    a_cmd_column: str = "a_cmd_env",
    block_scaling: str = "none",
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Assemble the feature matrix for one probe.

    Args:
        blocks: any of ``a_cmd``, ``dstates``, ``h_cur``, ``h_prev``,
            ``vis_cur``.
        layer, pool: required whenever a hidden-state block is requested.
        block_scaling: accepted only as ``"none"``.  Down-weighting a block
            has to happen *after* standardisation -- ``StandardScaler`` divides
            every column by its own standard deviation and exactly cancels any
            constant applied here -- so it lives in
            :class:`efference_probe.probes.BlockScaler`, which consumes the
            ``block_spans`` returned below.

    Returns:
        ``(X, block_spans)`` where ``block_spans`` records each block's width.
    """
    needs_hidden = {BLOCK_H_CUR, BLOCK_H_PREV} & set(blocks)
    if needs_hidden and (layer is None or pool is None):
        raise ValueError(f"blocks {sorted(needs_hidden)} require layer and pool")
    if store is None and (needs_hidden or BLOCK_VIS_CUR in blocks):
        store = HiddenStore(run.run_dir / "hidden")

    lookup = call_lookup(run.calls)
    cur_pairs = list(zip(samples["episode_id"], samples["cur_call"]))
    prev_pairs = list(zip(samples["episode_id"], samples["prev_call"]))
    layer_index = run.layer_index(layer) if layer is not None else None
    pool_index = run.pool_index(pool) if pool is not None else None

    parts: list[np.ndarray] = []
    spans: list[tuple[str, int]] = []
    for block in blocks:
        if block == BLOCK_A_CMD:
            values = _rows_from_calls(run.calls, lookup, prev_pairs, a_cmd_column)
        elif block == BLOCK_DSTATES:
            values = _delta_states(run.calls, lookup, prev_pairs)
        elif block == BLOCK_MISMATCH:
            values = _mismatch_features(run, lookup, prev_pairs, a_cmd_column)
        elif block == BLOCK_H_CUR:
            values = store.vectors(cur_pairs, layer_index, pool_index)
        elif block == BLOCK_H_PREV:
            values = store.vectors(prev_pairs, layer_index, pool_index)
        elif block == BLOCK_VIS_CUR:
            values = store.vision(cur_pairs)
        elif block == BLOCK_PHASE:
            # Task phase alone.  Bounds how much of any hidden-state result
            # could be explained by residual phase imbalance between classes.
            calls_index = samples["cur_call"].to_numpy(dtype=np.float32)
            values = np.stack([calls_index, calls_index**2], axis=1)
        else:
            raise ValueError(f"unknown feature block: {block!r}")
        if block_scaling != "none":
            raise ValueError(
                "block scaling is applied after standardisation by "
                "probes.BlockScaler; pass the returned block_spans to "
                "run_probe(spans=...) instead of scaling here"
            )
        parts.append(values.astype(np.float32))
        spans.append((block, values.shape[1]))

    return np.concatenate(parts, axis=1), spans


def _axis_angle_to_matrix(vectors: np.ndarray) -> np.ndarray:
    """Rodrigues: ``[n, 3]`` axis-angle to ``[n, 3, 3]`` rotation matrices."""
    angles = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(angles < 1e-8, 1.0, angles)
    axes = vectors / safe
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    zeros = np.zeros_like(x)
    skew = np.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=1).reshape(
        -1, 3, 3
    )
    sin = np.sin(angles)[..., None]
    cos = np.cos(angles)[..., None]
    identity = np.broadcast_to(np.eye(3), (len(vectors), 3, 3))
    return identity + sin * skew + (1.0 - cos) * (skew @ skew)


def _matrix_to_axis_angle(matrices: np.ndarray) -> np.ndarray:
    """``[n, 3, 3]`` rotation matrices to ``[n, 3]`` axis-angle."""
    trace = np.trace(matrices, axis1=1, axis2=2)
    angles = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    axes = np.stack(
        [
            matrices[:, 2, 1] - matrices[:, 1, 2],
            matrices[:, 0, 2] - matrices[:, 2, 0],
            matrices[:, 1, 0] - matrices[:, 0, 1],
        ],
        axis=1,
    )
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    unit = np.divide(axes, norms, out=np.zeros_like(axes), where=norms > 1e-8)
    return unit * angles[:, None]


def _delta_states(
    calls: pd.DataFrame,
    lookup: dict[tuple[int, int], int],
    pairs: list[tuple[int, int]],
    geometric_rotation: bool = True,
) -> np.ndarray:
    """State delta over one chunk.

    Dimensions 3:6 of a LIBERO state are an *absolute* end-effector axis-angle.
    Subtracting them componentwise is not a rotation delta and wraps: two
    orientations a hair apart can differ by ~2*pi in the raw difference, which
    both blows up the magnitude and produces outliers that standardise badly.
    The rotation block is therefore composed properly, as
    ``R_after @ R_before^T`` mapped back to axis-angle; translation and gripper
    are plain differences.
    """
    after = _rows_from_calls(calls, lookup, pairs, "states_after")
    before = _rows_from_calls(calls, lookup, pairs, "states_before")
    delta = after - before
    if geometric_rotation and after.shape[1] >= 6:
        relative = _axis_angle_to_matrix(
            after[:, 3:6].astype(np.float64)
        ) @ np.transpose(
            _axis_angle_to_matrix(before[:, 3:6].astype(np.float64)), (0, 2, 1)
        )
        delta[:, 3:6] = _matrix_to_axis_angle(relative).astype(delta.dtype)
    return delta


def _mismatch_features(
    run: RunData,
    lookup: dict[tuple[int, int], int],
    pairs: list[tuple[int, int]],
    a_cmd_column: str,
) -> np.ndarray:
    """Explicit command-vs-outcome comparison features.

    A logistic regression on ``a_cmd`` concatenated with ``dstates`` cannot
    answer "did the world move the way I asked?": agreement between two vectors
    is a distance, and a linear model has no way to form one.  The literal
    concatenation probe therefore understates what is *mechanically* knowable,
    which matters because SPEC 3.3 uses P2 as the ceiling the hidden-state
    probes are compared against.

    This block hands the linear model the interaction terms it would otherwise
    have to invent: the chunk-summed command, the achieved state delta, their
    elementwise products, and per-group norms and cosines.  Translation and
    rotation are treated separately because they live in different units.
    """
    schema = run.schema
    action_dim = int(schema["action_dim"])
    n_chunks = int(schema["num_action_chunks"])
    commanded = _rows_from_calls(run.calls, lookup, pairs, a_cmd_column)
    commanded = commanded.reshape(len(pairs), n_chunks, action_dim)
    summed = commanded.sum(axis=1)
    delta = _delta_states(run.calls, lookup, pairs)

    n_shared = min(summed.shape[1] - 1, delta.shape[1])
    products = summed[:, :n_shared] * delta[:, :n_shared]

    parts = [summed, delta, products]
    for lo, hi in ((0, 3), (3, 6)):
        if summed.shape[1] < hi or delta.shape[1] < hi:
            continue
        command_block = summed[:, lo:hi]
        delta_block = delta[:, lo:hi]
        command_norm = np.linalg.norm(command_block, axis=1)
        delta_norm = np.linalg.norm(delta_block, axis=1)
        denominator = np.maximum(command_norm * delta_norm, 1e-8)
        cosine = (command_block * delta_block).sum(axis=1) / denominator
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


def _rows_from_calls(
    calls: pd.DataFrame,
    lookup: dict[tuple[int, int], int],
    pairs: list[tuple[int, int]],
    column: str,
) -> np.ndarray:
    positions = []
    for episode_id, call_idx in pairs:
        key = (int(episode_id), int(call_idx))
        if key not in lookup:
            raise KeyError(f"no call row for episode {episode_id} call {call_idx}")
        positions.append(lookup[key])
    series = calls[column].to_numpy()
    return np.stack([np.asarray(series[p], dtype=np.float32) for p in positions])


def step_index_report(samples: pd.DataFrame) -> pd.DataFrame:
    """Call-index distribution per class -- the check that phase is matched."""
    grouped = samples.groupby("y")["cur_call"]
    return pd.DataFrame(
        {
            "n": grouped.count(),
            "mean": grouped.mean(),
            "std": grouped.std(),
            "p05": grouped.quantile(0.05),
            "median": grouped.median(),
            "p95": grouped.quantile(0.95),
        }
    )


def phase_bias_report(samples: pd.DataFrame) -> dict[str, Any]:
    """Signed within-pair phase bias, which the marginal report cannot see.

    Comparing the two classes' marginal call-index distributions is a weak
    check: it passes even for globally-sampled negatives, and it hides a
    consistent *signed* offset inside pairs.  The offset is structural --
    a negative needs several clean calls before it, and that availability
    decays as hijacks accumulate, skewing the negative pool earlier -- and
    hidden states drift with task progress, so it is exactly the confound the
    matching exists to remove.
    """
    if "pair_id" not in samples.columns or (samples["pair_id"] < 0).all():
        return {"note": "samples are not paired; signed phase bias undefined"}
    paired = samples[samples["pair_id"] >= 0]
    wide = paired.pivot_table(index="pair_id", columns="y", values="cur_call").dropna()
    if wide.empty or 0 not in wide.columns or 1 not in wide.columns:
        return {"note": "no complete pairs"}
    differences = (wide[1] - wide[0]).to_numpy()
    report: dict[str, Any] = {
        "n_pairs": int(len(differences)),
        "signed_mean_difference": float(np.mean(differences)),
        "signed_median_difference": float(np.median(differences)),
        "abs_mean_difference": float(np.mean(np.abs(differences))),
        "fraction_negative_earlier": float(np.mean(differences > 0)),
    }
    nonzero = differences[differences != 0]
    if len(nonzero):
        try:
            from scipy import stats

            report["sign_test_p"] = float(
                stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue
            )
        except ImportError:
            pass
    return report


def count_raw_positives(
    calls: pd.DataFrame,
    min_clean_gap: int = 2,
    warmup_calls: int = 4,
    drop_post_success: bool = True,
) -> int:
    """How many post-hijack calls exist before pairing discards any.

    Paired matching keeps ``min(n_positives, n_negatives)`` per episode, and
    negatives are structurally scarcer (a negative needs a clean window, a
    positive only needs a hijack).  Episodes with dense hijacks therefore lose
    positives, non-randomly -- so the raw count has to be reported next to the
    kept count rather than inferred from it.
    """
    frame = calls
    if drop_post_success and "post_success" in frame.columns:
        frame = frame[~frame["post_success"].astype(bool)]
    total = 0
    for _episode_id, episode in frame.groupby("episode_id", sort=False):
        labels = dict(zip(episode["call_idx"], episode["label"]))
        for call_idx, label in labels.items():
            if label != SELF or call_idx < warmup_calls + 1:
                continue
            if labels.get(call_idx - 1) == HIJACK:
                total += 1
    return total

"""Attribution probes on a phase-3 arm, scored exactly like the VLA ladder.

Sample construction and probes are the VLA pipeline's own code
(:func:`efference_probe.datasets.build_probe_samples` and
:func:`efference_probe.probes.run_probe`), so the GRU and the 7B policy are
judged by one yardstick: phase-matched negatives, 5-fold CV grouped by
segment, nested ``C`` selection, balanced accuracy against a selection-aware
shuffled-label floor.

Mapping onto the VLA vocabulary.  A *segment* (one burst of actions) plays the
role of an episode and is the CV group; steps since burst onset play the role
of the call index, so negatives are matched by burst phase.  The sample's
hidden state is the one right *after* the manipulated step -- the GRU update
that has just integrated the hijacked outcome, which is what the VLA's
``h(t+1)`` is.

Rungs (``h`` = ``h_multi`` after the step, ``hg`` = the raw GRU state):

=========  ====================  =============================================
P0 / P0g   h / hg, shuffled       selection-aware floor (max over both cells)
P1 / P1g   h / hg                 attribution read out of the state (Q1)
P3         command + h            the command handed to the probe (Q2)
P4         h before + h after     before/after states (Q2)
P2         command + residual     linear concatenation oracle
P2r        comparison features    mechanical ceiling / positive control
C_cmd      command                leakage: must sit at chance
C_out      residual               outcome novelty alone (no command)
C_eff      efference input        the model's own efference channel alone
C_phase    burst phase            residual phase imbalance
=========  ====================  =============================================

``residual`` is the observation minus the model's own prediction made before
the step: the outcome-side surprise, i.e. the GRU's "visual novelty".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from efference_probe.datasets import build_probe_samples
from efference_probe.probes import ProbeResult, run_probe

from .mismatch import ArmData

logger = logging.getLogger(__name__)

HIDDEN_PROBES = ("P1", "P1g", "P3", "P4")


@dataclass(frozen=True)
class GRUProbeSpec:
    name: str
    blocks: tuple[str, ...]
    shuffle_labels: bool = False


PROBE_SPECS: tuple[GRUProbeSpec, ...] = (
    GRUProbeSpec("P0", ("h",), shuffle_labels=True),
    GRUProbeSpec("P0g", ("hg",), shuffle_labels=True),
    GRUProbeSpec("P1", ("h",)),
    GRUProbeSpec("P1g", ("hg",)),
    GRUProbeSpec("P3", ("cmd", "h")),
    GRUProbeSpec("P4", ("h_before", "h")),
    GRUProbeSpec("P2", ("cmd", "out")),
    GRUProbeSpec("P2r", ("cmp",)),
    GRUProbeSpec("C_cmd", ("cmd",)),
    GRUProbeSpec("C_out", ("out",)),
    GRUProbeSpec("C_eff", ("eff",)),
    GRUProbeSpec("C_phase", ("phase",)),
)


def build_samples(arm: ArmData, warmup: int, min_clean_gap: int) -> pd.DataFrame:
    """Phase-matched positive/negative pairs, via the VLA sample builder.

    Returns the builder's frame plus ``index``: the global step whose
    post-update state is the sample.
    """
    steps = arm.steps
    usable = steps[
        steps["active"] & (steps["segment"] >= 1) & (steps["since_onset"] >= 0)
    ]
    calls = pd.DataFrame(
        {
            "episode_id": usable["segment"].to_numpy(),
            "task_id": 0,
            "call_idx": usable["since_onset"].to_numpy(),
            "label": usable["label"].to_numpy(),
            "transform": usable["transform"].to_numpy(),
        }
    )
    samples = build_probe_samples(
        calls,
        min_clean_gap=min_clean_gap,
        warmup_calls=warmup,
        negatives="phase_matched",
        rng=np.random.default_rng(0),
        drop_post_success=False,
    )
    if samples.empty:
        return samples
    # The builder's sample is (prev, cur=prev+1) with the manipulation on
    # `prev`; in the GRU the state right after `prev` already holds the
    # outcome, so `prev` is the step to read.
    lookup = {
        (int(s), int(c)): int(i)
        for s, c, i in zip(usable["segment"], usable["since_onset"], usable["index"])
    }
    samples = samples.copy()
    samples["index"] = [
        lookup[(int(e), int(p))]
        for e, p in zip(samples["episode_id"], samples["prev_call"])
    ]
    return samples


def feature_block(
    arm: ArmData, index: np.ndarray, block: str, gamma: float
) -> np.ndarray:
    steps = arm.steps
    command = steps["command"].to_numpy(dtype=np.float64)[index]
    residual = np.stack(
        [steps[f"res{k}"].to_numpy(dtype=np.float64)[index] for k in range(4)], axis=1
    )
    if block == "h":
        return arm.h_multi[index]
    if block == "hg":
        return arm.h_gru[index]
    if block == "h_before":
        if (index < 1).any():
            raise ValueError("a sample sits at step 0; it has no preceding state")
        return arm.h_multi[index - 1]
    if block == "cmd":
        return np.stack([command, np.abs(command)], axis=1)
    if block == "out":
        return np.concatenate([residual, np.abs(residual[:, :1])], axis=1)
    if block == "eff":
        return np.stack(
            [
                steps["efference"].to_numpy(dtype=np.float64)[index],
                steps["trace"].to_numpy(dtype=np.float64)[index],
            ],
            axis=1,
        )
    if block == "phase":
        phase = steps["since_onset"].to_numpy(dtype=np.float64)[index]
        return np.stack([phase, phase**2], axis=1)
    if block == "cmp":
        # Explicit command-vs-outcome terms: the action enters channel 0 only.
        outcome = residual[:, 0]
        expected = gamma * command
        return np.stack(
            [
                command,
                np.abs(command),
                outcome,
                np.abs(outcome),
                command * outcome,
                np.abs(outcome) - np.abs(expected),
                np.abs(outcome - expected),
            ],
            axis=1,
        )
    raise ValueError(f"unknown block {block!r}")


def features(
    arm: ArmData, samples: pd.DataFrame, blocks: tuple[str, ...], gamma: float
) -> np.ndarray:
    index = samples["index"].to_numpy(dtype=np.int64)
    parts = [feature_block(arm, index, block, gamma) for block in blocks]
    return np.concatenate(parts, axis=1).astype(np.float32)


def permutation_null(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_permutations: int, seed: int = 0
) -> dict[str, Any]:
    """Shuffled-label scores at one fixed cell (no C selection, for speed)."""
    scores = []
    for k in range(n_permutations):
        result = run_probe(
            x,
            y,
            groups,
            name="perm",
            blocks=["h"],
            shuffle_labels=True,
            seed=seed + 1000 + k,
            select_c=False,
        )
        scores.append(result.balanced_acc_mean)
    if not scores:
        return {}
    array = np.asarray(scores)
    return {
        "n_permutations": int(len(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def analyze_arm(
    arm: ArmData,
    warmup: int,
    min_clean_gap: int,
    gamma: float,
    n_permutations: int = 0,
    family: str = "linear",
    seed: int = 0,
    specs: Optional[tuple[GRUProbeSpec, ...]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run every rung on one arm; return the per-probe table and a summary."""
    samples = build_samples(arm, warmup=warmup, min_clean_gap=min_clean_gap)
    if samples.empty:
        return pd.DataFrame(), {"note": "no probe samples"}
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    results: list[ProbeResult] = []
    for spec in specs or PROBE_SPECS:
        x = features(arm, samples, spec.blocks, gamma)
        results.append(
            run_probe(
                x,
                y,
                groups,
                name=spec.name,
                blocks=list(spec.blocks),
                shuffle_labels=spec.shuffle_labels,
                seed=seed,
                family=family,
            )
        )
    frame = pd.DataFrame([r.to_row() for r in results])
    score = dict(zip(frame["name"], frame["balanced_acc_mean"]))

    floor_cells = [score[n] for n in ("P0", "P0g") if n in score]
    hidden_cells = [score[n] for n in ("P1", "P1g") if n in score]
    summary: dict[str, Any] = {
        "transform": arm.transform,
        "n_samples": int(len(samples)),
        "n_positive": int(y.sum()),
        "n_groups": int(len(np.unique(groups))),
        "selection_floor": float(max(floor_cells)) if floor_cells else float("nan"),
        "best_hidden": float(max(hidden_cells)) if hidden_cells else float("nan"),
        "best_hidden_any": float(max(score[n] for n in HIDDEN_PROBES if n in score))
        if any(n in score for n in HIDDEN_PROBES)
        else float("nan"),
    }
    summary["margin"] = summary["best_hidden"] - summary["selection_floor"]
    if n_permutations:
        x_h = features(arm, samples, ("h",), gamma)
        summary["permutation_null_h"] = permutation_null(
            x_h, y, groups, n_permutations, seed=seed
        )
    return frame, summary


__all__ = [
    "PROBE_SPECS",
    "analyze_arm",
    "build_samples",
    "features",
    "permutation_null",
]
